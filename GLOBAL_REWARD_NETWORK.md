# Global Reward Network Integration

This document describes the implementation of a global reward network that can be shared across different modules in the MassGym environment and MuZero transformer.

## Overview

The global reward network replaces the simple exact-match reward system with a neural network that computes cosine similarity between mass spectrum embeddings and SELFIES representations. The network is implemented as a singleton pattern to ensure:

1. **Single Instance**: Only one reward network exists across all modules
2. **Shared Parameters**: All modules use the same network parameters
3. **Synchronized Updates**: Parameter updates in one module are immediately available to all others

## Architecture

### Global Reward Network Manager (`lzero/model/global_reward_network.py`)

The `GlobalRewardNetworkManager` class implements the singleton pattern and manages:

- **Network Initialization**: Creates and configures the reward network
- **Parameter Management**: Handles parameter updates and synchronization
- **Forward Function**: Provides a callable interface for reward computation
- **Device Management**: Handles GPU/CPU placement

### Key Components

1. **MoleculeSpectrumMatcher**: The actual neural network that computes similarity
2. **SelfiesTokenizer**: Tokenizes SELFIES strings for the network
3. **Forward Function**: Thread-safe function for computing rewards

## Integration Points

### 1. MassGym Environment (`zoo/masspecgym/envs/massgymenv.py`)

**Initialization**:
```python
# Initialize global reward network if available
if GLOBAL_REWARD_AVAILABLE:
    initialize_global_reward_network(
        max_selfies_len=cfg.get('max_len', 100),
        device='cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_path=cfg.get('reward_network_checkpoint', None)
    )
    self.reward_function = get_reward_function()
```

**Reward Computation**:
```python
# Use reward network for final reward computation
if self.use_reward_network and self.reward_function:
    similarity_score = self.reward_function(
        self.current_selfies, 
        spectrum_embed, 
        formula
    )
    raw_reward = similarity_score
```

### 2. MuZero Transformer (`lzero/model/muzero_transformer.py`)

**Recurrent Inference**:
```python
# Compute reward using global reward network if available
if GLOBAL_REWARD_AVAILABLE:
    reward_function = get_reward_function()
    for batch_idx in range(batch_size):
        similarity_score = reward_function(
            current_selfies, spectrum_embed, formula
        )
        reward_scores.append(similarity_score)
```

### 3. GumbelMuZero Agent (`lzero/agent/gumbel_muzero.py`)

**Training Loop**:
```python
# Train global reward network if available
if self.global_reward_manager is not None:
    reward_network = self.global_reward_manager.get_network()
    # Update network parameters during training
```

## Usage

### Basic Setup

```python
from lzero.model.global_reward_network import initialize_global_reward_network, get_reward_function

# Initialize the global reward network
initialize_global_reward_network(
    vocab_size=1000,
    max_selfies_len=100,
    device='cuda'
)

# Get the reward function
reward_function = get_reward_function()

# Compute reward
score = reward_function(selfies_string, spectrum_embed, formula_string)
```

### Configuration Options

- `vocab_size`: Size of SELFIES vocabulary (auto-detected if not provided)
- `selfies_embed_dim`: Embedding dimension for SELFIES (default: 1024)
- `spectrum_fingerprint_dim`: Spectrum embedding dimension (default: 4096)
- `fusion_dim`: Fusion layer dimension (default: 1024)
- `dropout`: Dropout rate (default: 0.1)
- `max_selfies_len`: Maximum SELFIES sequence length (default: 100)
- `device`: Device for computation ('cuda' or 'cpu')
- `checkpoint_path`: Path to pretrained weights (optional)

## Benefits

1. **Improved Reward Signal**: Neural network provides more nuanced similarity scores than exact matching
2. **Shared Learning**: All modules benefit from the same learned representations
3. **Efficient Memory Usage**: Single network instance reduces memory footprint
4. **Synchronized Updates**: Parameter updates are immediately available across modules
5. **Flexible Integration**: Easy to enable/disable via import availability

## Testing

Run the integration test to verify everything works correctly:

```bash
cd main/LightZero
python test_global_reward_network.py
```

The test script verifies:
- Global reward network initialization
- Environment integration
- Transformer integration
- Singleton pattern behavior
- Parameter sharing and updates

## Error Handling

The implementation includes comprehensive error handling:

- **Import Failures**: Graceful fallback to dummy implementations
- **Initialization Errors**: Detailed error messages and fallback options
- **Runtime Errors**: Safe error handling during reward computation
- **Device Mismatches**: Automatic device placement and error recovery

## Future Extensions

The global reward network can be extended to support:

1. **Multi-Modal Inputs**: Additional molecular representations
2. **Hierarchical Rewards**: Different reward components for different aspects
3. **Dynamic Networks**: Networks that adapt during training
4. **Distributed Training**: Support for multi-GPU and multi-node training

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the reward_model directory is properly set up
2. **CUDA Errors**: Check GPU availability and memory
3. **Tokenizer Issues**: Verify SELFIES tokenizer is properly initialized
4. **Memory Issues**: Reduce batch size or use CPU for testing

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

This will provide detailed information about network initialization, parameter updates, and reward computations. 