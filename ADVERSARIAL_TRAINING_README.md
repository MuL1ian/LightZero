# Generative Adversarial Training for MassGym Environment

## Overview

This implementation adds generative adversarial training with preference learning to the GAG MuZero policy for the MassGym environment. The system treats generated instances from MCTS search as negative examples and training data as positive examples, using the reward network as a critic to distinguish between them.

## Key Components

### 1. Global Reward Network Integration

The reward network (`MoleculeSpectrumMatcher`) from `reward_nn.py` is integrated globally across all modules:

- **Location**: `lzero/model/global_reward_network.py`
- **Purpose**: Provides a singleton reward network that computes cosine similarity between SELFIES and spectrum embeddings
- **Input Format**: 
  - SELFIES tokens: `[batch_size, max_selfies_len]`
  - Spectrum embeddings: `[batch_size, 4096]` (pre-computed)
  - Formula tokens: `[batch_size, formula_max_len]`

### 2. Adversarial Training in GAG MuZero Policy

**Location**: `lzero/policy/gag_muzero.py`

#### Configuration Parameters

```python
# Adversarial Training Configuration
reward_learning_rate=1e-4,              # Learning rate for reward network
reward_weight_decay=1e-5,               # Weight decay for reward network optimizer
reward_regularization_weight=0.01,      # L2 regularization weight for rewards
enable_adversarial_training=True,       # Enable/disable adversarial training
```

#### Key Methods

1. **`_extract_adversarial_training_data()`**
   - Extracts positive (training) and negative (generated) examples
   - Parses observation batch: spectrum (4096) + selfies (100) + formula (50)
   - Returns data in reward network format

2. **`_generate_negative_examples()`**
   - Generates negative SELFIES sequences from current policy
   - Uses same spectrum embeddings with different molecules
   - Creates realistic negative examples for training

3. **`_compute_preference_loss()`**
   - Implements preference learning with margin-based loss
   - Encourages positive examples to have higher similarity scores
   - Combines preference loss with regularization terms

### 3. Data Flow

```
Training Batch (MassGym Environment)
    ↓
Observation: [spectrum_embeds(4096) + selfies_tokens(100) + formula_tokens(50)]
    ↓
Extract Positive Data (training examples)
    ↓
Generate Negative Data (policy-generated examples)
    ↓
Reward Network Similarity Computation
    ↓
Preference Learning Loss
    ↓
Update Reward Network Parameters
```

### 4. RLHF-Style Pairwise Preference Loss

The adversarial training uses the **Bradley-Terry model** from RLHF (Reinforcement Learning from Human Feedback):

1. **Pairwise Preference Model**: `P(positive > negative) = sigmoid(reward_positive - reward_negative)`
2. **RLHF Loss**: `-log(P(positive > negative)) = softplus(-(reward_positive - reward_negative))`
3. **L2 Regularization**: Optional regularization to prevent reward collapse

```python
# RLHF-style pairwise preference loss
reward_diff = positive_rewards - negative_rewards
preference_loss = F.softplus(-reward_diff).mean()

# Optional L2 regularization on rewards
regularization_loss = reward_regularization_weight * (positive_rewards**2 + negative_rewards**2).mean()

total_loss = preference_loss + regularization_loss
```

This approach is identical to the preference learning used in ChatGPT and other RLHF systems, where the model learns to prefer positive examples over negative examples through pairwise comparisons.

## Integration Points

### Environment (MassGym)
- **File**: `zoo/masspecgym/envs/massgymenv.py`
- **Integration**: Uses global reward network for reward computation
- **Data Format**: Provides pre-computed spectrum embeddings in observations

### Model (MuZero Transformer)
- **File**: `lzero/model/muzero_transformer.py`
- **Integration**: Uses global reward network during MCTS search
- **Purpose**: Consistent reward computation across training and inference

### Policy (GAG MuZero)
- **File**: `lzero/policy/gag_muzero.py`
- **Integration**: Main adversarial training implementation
- **Training**: Updates reward network parameters during policy learning

## Usage

### 1. Enable Adversarial Training

```python
cfg = EasyDict(GAGMuZeroPolicy.config)
cfg.enable_adversarial_training = True
cfg.reward_learning_rate = 1e-4
cfg.preference_margin = 0.5
```

### 2. Initialize Policy

```python
policy = GAGMuZeroPolicy(cfg, model=model)
# Reward network and optimizer are automatically initialized
```

### 3. Training Loop

The adversarial training runs automatically during `_forward_learn()`:

1. Standard MuZero policy and value losses are computed
2. Positive and negative examples are extracted from the batch
3. Preference learning loss is computed using the reward network
4. Reward network parameters are updated via separate optimizer

## Benefits

1. **RLHF-Style Learning**: Uses the same pairwise preference learning approach as ChatGPT and other successful RLHF systems
2. **Improved Reward Signal**: The reward network learns to distinguish between correct and incorrect molecule-spectrum pairs through pairwise comparisons
3. **Adversarial Robustness**: Generated examples help the reward network generalize better and avoid overfitting
4. **Relative Preferences**: The Bradley-Terry model learns relative preferences rather than absolute scores, which is more robust
5. **End-to-End Training**: Reward network and policy are trained jointly, allowing for co-evolution and better alignment

## Testing

Run the test script to verify the implementation:

```bash
cd /home/zirui/MassEnv/main/LightZero
python test_adversarial_training.py
```

The test verifies:
- Module imports and initialization
- Data extraction and formatting
- Preference loss computation
- Integration with GAG MuZero policy

## Technical Details

### Observation Format
- **Total Size**: 4246 dimensions
- **Spectrum**: 4096 pre-computed embeddings from SpectraEncoderGrowing
- **SELFIES**: 100 token IDs from SelfiesTokenizer
- **Formula**: 50 token IDs from BERT tokenizer

### Reward Network Architecture
- **SELFIES Encoder**: Transformer-based with positional encoding
- **Spectrum Encoder**: Pre-trained frozen SpectraEncoderGrowing (4096 → 4096)
- **Fusion**: Projects both to common space and computes cosine similarity

### Memory Efficiency
- Uses pre-computed spectrum embeddings to avoid repeated encoding
- Freezes spectrum encoder parameters to reduce memory usage
- Efficient batch processing for positive/negative pairs

## Future Improvements

1. **Advanced Negative Sampling**: Use MCTS rollouts for more realistic negatives
2. **Curriculum Learning**: Gradually increase difficulty of negative examples
3. **Multi-Modal Training**: Include additional molecular properties
4. **Contrastive Learning**: Extend to full contrastive learning framework 