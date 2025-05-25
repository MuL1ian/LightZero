# Enhanced MuZero Transformer with BERT Formula Tokenization

## Summary

The MuZero transformer has been successfully updated to handle the new observation structure that includes BERT-tokenized chemical formulas alongside spectrum embeddings and SELFIES tokens.

## Key Changes Made

### 1. Updated Observation Structure

The observation dimension has been updated from **4196** to **4246**:
- **Spectrum embeddings**: 4096 dimensions (unchanged)
- **SELFIES tokens**: 100 dimensions (unchanged) 
- **Formula tokens**: 50 dimensions (NEW - BERT tokenized)

### 2. Base Class Updates (`MuZeroSelfiesTransformer`)

#### Updated Default Observation Shape
- Changed `observation_shape` default from 4096 to 4246

#### Fixed `initial_inference` Method
- Now extracts only spectrum and SELFIES parts for transformer processing
- Properly handles the new observation structure with formula tokens

#### Fixed `recurrent_inference` Method  
- Updated to process only spectrum and SELFIES parts
- Maintains formula tokens unchanged in latent state

#### Fixed `_dynamics` Method
- Now only modifies SELFIES portion of latent state
- Preserves formula tokens during action updates
- Uses proper indexing to find padding tokens in SELFIES section only

#### Fixed `step_prediction` Function
- Removed duplicate SOS token addition (environment tokenizer already includes it)
- Simplified to use pre-tokenized sequences directly

### 3. Enhanced Class (`MuZeroSelfiesTransformerEnhanced`)

#### New Dimension Handling
- Added `formula_max_len` parameter (default: 50)
- Added `selfies_start_idx` and `formula_start_idx` for proper indexing
- Updated observation shape default to 4246

#### Enhanced `_dynamics` Method
- Overrides base class to handle new latent state structure
- Only modifies SELFIES tokens (indices 4096-4196)
- Preserves formula tokens (indices 4196-4246) unchanged

#### New Formula Extraction Method
- Added `_extract_formula_from_latent_state()` method
- Uses BERT tokenizer to decode formula tokens back to text
- Handles batch processing and error cases

#### Enhanced SELFIES Extraction
- Updated `_extract_selfies_from_latent_state()` to use correct indices
- Now extracts only from SELFIES portion (4096-4196)

#### Formula-Based Action Masking
- Automatically extracts target formula from observation
- Sets `target_formula` during `initial_inference`
- Applies formula constraints to policy logits

#### Enhanced Logging
- Added formula information to completion status logging
- Better debugging output for molecule generation

### 4. Test Infrastructure

#### Created `test_enhanced_transformer.py`
- Tests real environment integration
- Tests dimension consistency
- Validates SELFIES and formula extraction
- Tests multi-step inference

#### Fixed Synthetic Data Generation
- Uses valid token IDs for testing
- Respects vocabulary constraints
- Proper structure for all observation components

## Usage

### Basic Usage
```python
from lzero.model.muzero_transformer import MuZeroSelfiesTransformerEnhanced

model = MuZeroSelfiesTransformerEnhanced(
    observation_shape=4246,
    max_len=100,
    formula_max_len=50,
    target_formula="C6H12O6"  # Optional
)
```

### With Environment
```python
from zoo.masspecgym.envs.massgym_wrapper import MassGymLightZeroEnv

env = MassGymLightZeroEnv({
    'max_len': 100,
    'formula_max_len': 50,
    'formula_masking': True
})

obs = env.reset()
output = model.initial_inference(obs['observation'])
```

## Key Benefits

1. **Backward Compatibility**: Base class still works with existing code
2. **Formula Integration**: Seamless integration of BERT-tokenized formulas
3. **Action Masking**: Intelligent formula-based action constraints
4. **Proper State Management**: Formula tokens preserved during inference
5. **Robust Testing**: Comprehensive test suite validates functionality

## Technical Details

### Observation Layout
```
[0:4096]     - Spectrum embeddings (float)
[4096:4196]  - SELFIES tokens (int, tokenized)
[4196:4246]  - Formula tokens (int, BERT tokenized)
```

### Action Updates
- Only SELFIES tokens (indices 4096-4196) are modified during actions
- Formula tokens remain constant throughout episode
- Proper padding token detection in SELFIES section only

### Formula Processing
- BERT tokenizer with max_length=50, padding='max_length'
- Includes [CLS] and [SEP] special tokens
- Automatic truncation for long formulas

## Testing

Run the test suite:
```bash
cd main/LightZero
python zoo/masspecgym/test_enhanced_transformer.py
```

Both environment integration and dimension consistency tests should pass.

## Files Modified

1. `main/LightZero/lzero/model/muzero_transformer.py` - Main transformer implementation
2. `main/LightZero/zoo/masspecgym/test_enhanced_transformer.py` - Test suite
3. `main/LightZero/zoo/masspecgym/envs/massgymenv.py` - Environment (already updated)
4. `main/LightZero/zoo/masspecgym/envs/massgym_wrapper.py` - Wrapper (already updated)

The enhanced transformer is now fully compatible with the BERT formula tokenization system! 