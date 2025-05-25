# BERT Formula Tokenization in MassGymEnv

This document explains how to use BERT tokenizer to tokenize chemical formulas and include them as part of the observation in the MassGymEnv.

## Overview

The MassGymEnv now supports BERT-based tokenization of chemical formulas, which are included as part of the observation alongside the spectrum embeddings and SELFIES tokens. This provides the model with structured information about the target molecular formula.

## Configuration

To enable formula tokenization, set the following parameters in your environment configuration:

```python
cfg = {
    'env_id': 'mass_spec_env',
    'max_len': 100,              # Maximum length for SELFIES tokens
    'formula_max_len': 50,       # Maximum length for formula tokens
    'debug': True,
    'formula_masking': True,     # Enable formula-based action masking
}
```

## Observation Structure

The observation now consists of three concatenated parts:

1. **Spectrum embeddings**: 4096 dimensions
2. **SELFIES tokens**: `max_len` dimensions (default: 100)
3. **Formula tokens**: `formula_max_len` dimensions (default: 50)

**Total observation dimension**: 4096 + 100 + 50 = **4246**

## How It Works

### 1. Formula Encoding

The `_encode_formula()` method uses BERT tokenizer to encode chemical formulas:

```python
def _encode_formula(self, formula):
    """
    Encode a chemical formula using BERT tokenizer with fixed length.
    
    Args:
        formula (str): Chemical formula string (e.g., "C6H12O6")
        
    Returns:
        torch.Tensor: Fixed-length tensor of formula token IDs
    """
    formula_token_ids = self.formula_tokenizer.encode(
        formula,
        add_special_tokens=True,  # Add [CLS] and [SEP] tokens
        max_length=self.formula_max_len,
        padding='max_length',     # Pad to max_length
        truncation=True,          # Truncate if longer than max_length
        return_tensors='pt'       # Return PyTorch tensors
    )
    return formula_token_ids.squeeze(0)
```

### 2. Observation Construction

In both `reset()` and `step()` methods, the observation is constructed as:

```python
spectrum = self.target_spectrum['embeds']  # [4096]
formula = self.target_spectrum['formulas']
formula_token_ids = self._encode_formula(formula).float()  # [50]
token = self.token_ids.float()  # [100]

# Combine all observations
combined_obs = torch.cat([spectrum, token, formula_token_ids], dim=-1)  # [4246]
```

## Example Usage

### Basic Environment Usage

```python
from zoo.masspecgym.envs.massgymenv import MassGymEnv

cfg = {
    'max_len': 100,
    'formula_max_len': 50,
    'debug': True,
    'formula_masking': True,
}

env = MassGymEnv(cfg)
obs_timestep = env.reset()
obs = obs_timestep.obs

print(f"Observation shape: {obs['observation'].shape}")  # torch.Size([4246])
print(f"Current formula: {env.target_spectrum['formulas']}")
```

### Using the LightZero Wrapper

```python
from zoo.masspecgym.envs.massgym_wrapper import MassGymLightZeroEnv

cfg = {
    'max_len': 100,
    'formula_max_len': 50,
    'debug': True,
    'formula_masking': True,
}

env = MassGymLightZeroEnv(cfg)
obs = env.reset()

print(f"Expected dimension: {env.expected_obs_dim}")  # 4246
print(f"Observation shape: {obs['observation'].shape}")  # torch.Size([4246])
```

## Formula Examples

The BERT tokenizer handles various chemical formulas:

- `"C6H12O6"` → `[CLS] c6h12o6 [SEP] [PAD] [PAD] ...`
- `"H2O"` → `[CLS] h2o [SEP] [PAD] [PAD] ...`
- `"C8H10N4O2"` → `[CLS] c8h10n4o2 [SEP] [PAD] [PAD] ...`
- `""` (empty) → `[CLS] [SEP] [PAD] [PAD] ...`

## Key Features

1. **Fixed Length**: All formula tokens are padded/truncated to `formula_max_len`
2. **Special Tokens**: Includes `[CLS]` and `[SEP]` tokens for proper BERT formatting
3. **Consistent Dimensions**: Observation dimension is always the same regardless of formula length
4. **Integration**: Seamlessly integrated with existing SELFIES and spectrum observations

## Testing

Run the provided test scripts to verify functionality:

```bash
# Test basic formula tokenization
python zoo/masspecgym/test_formula_tokenization.py

# Test wrapper functionality
python zoo/masspecgym/test_wrapper.py
```

## Notes

- The BERT tokenizer is loaded from `'bert-base-uncased'`
- Formula tokens are converted to float for consistency with other observation components
- The implementation maintains backward compatibility with existing code
- Action masking can still be based on formula constraints when `formula_masking=True` 