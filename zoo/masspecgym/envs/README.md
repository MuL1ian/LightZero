# MassSpec Environment Utility Functions

This directory contains a modular implementation of the MassSpec environment with decoupled utility functions for molecular formula parsing, atom tracking, and action masking.

## Overview

The environment has been refactored to separate core functionality into standalone utility functions that can be used independently of the environment class. This makes the code more modular, testable, and reusable.

## Files

- `massgymenv.py` - Main environment class (now uses utility functions)
- `utils.py` - Standalone utility functions for molecular operations
- `example_usage.py` - Examples of using utility functions independently
- `mass_tokenizers.py` - SELFIES tokenizer implementation

## Key Features

### 1. Formula Parsing and Atom Tracking

The environment now properly tracks the number of atoms used and ensures they match the target molecular formula:

- **Formula parsing**: Extracts element counts from molecular formulas (e.g., "C6H12O6" → {'C': 6, 'H': 12, 'O': 6})
- **Atom tracking**: Maintains count of used elements during molecule building
- **Validation**: Ensures atom usage doesn't exceed formula requirements

### 2. Action Masking

The action masking system now considers both element types and atom counts:

- **Element filtering**: Only allows elements present in the target formula
- **Count constraints**: Masks out actions when element quota is reached
- **Dynamic masking**: Updates valid actions based on current molecule state

### 3. Modular Design

All core functionality is available as standalone functions:

```python
from utils import (
    parse_formula_counts,
    get_action_mask,
    update_atom_counts,
    validate_selfies_addition,
    # ... and more
)
```

## Utility Functions

### Core Functions

#### `parse_formula_counts(formula: str) -> Dict[str, int]`
Parse a molecular formula and return element counts.

```python
>>> parse_formula_counts("C6H12O6")
{'C': 6, 'H': 12, 'O': 6}
```

#### `get_action_mask(...) -> np.ndarray`
Generate a boolean mask indicating which actions are valid given the current state.

```python
mask = get_action_mask(
    formula="CH4",
    used_element_counts={'C': 1, 'H': 2},
    actions_list=['[C]', '[H]', '[O]', '<END>'],
    atom_tokens=['[C]', '[H]', '[O]'],
    bonded_atom_tokens=[],
    formula_masking=True
)
# Returns: [False, True, False, True]  # C maxed out, H available, O not in formula, END always valid
```

#### `update_atom_counts(token: str, counts: Dict[str, int], increment: bool) -> Dict[str, int]`
Update element counts when atoms are added or removed.

```python
>>> update_atom_counts('[C]', {'C': 1, 'H': 2}, increment=True)
{'C': 2, 'H': 2}
```

#### `validate_selfies_addition(current_selfies: str, new_token: str) -> bool`
Check if adding a token results in valid SELFIES.

```python
>>> validate_selfies_addition("[C]", "[H]")
True
```

### Helper Functions

- `extract_element_from_token(token)` - Extract element symbol from SELFIES token
- `get_allowed_elements_from_formula(formula)` - Get all allowed element variants
- `remove_last_token_from_selfies(selfies)` - Remove last token from SELFIES string
- `calculate_formula_completion_reward(formula, counts)` - Calculate completion reward
- `check_formula_match(formula, counts)` - Check if counts match formula exactly
- `get_state_info(formula, selfies, counts)` - Get comprehensive state information

## Usage Examples

### Standalone Usage

```python
from utils import parse_formula_counts, get_action_mask, update_atom_counts

# Define state
target_formula = "CH4"
current_selfies = "[C]"
used_counts = {"C": 1}
actions = ['[C]', '[H]', '[O]', '<END>']

# Get valid actions
mask = get_action_mask(
    formula=target_formula,
    used_element_counts=used_counts,
    actions_list=actions,
    atom_tokens=['[C]', '[H]', '[O]'],
    bonded_atom_tokens=[]
)

# Add a hydrogen atom
if mask[1]:  # [H] is valid
    used_counts = update_atom_counts('[H]', used_counts, increment=True)
    current_selfies += '[H]'
```

### Environment Integration

The environment class now uses these utility functions internally:

```python
env = MassGymEnv(config)
obs = env.reset()

# The environment automatically:
# 1. Parses the target formula
# 2. Tracks atom usage
# 3. Generates valid action masks
# 4. Validates SELFIES operations

action = env.action_space.sample()
obs, reward, done, info = env.step(action)
```

## Testing

Run the built-in tests to verify functionality:

```bash
cd main/LightZero/zoo/masspecgym/envs
python -c "from utils import test_utils; test_utils()"
```

Run the example usage script:

```bash
python example_usage.py
```

## Benefits of the Refactored Design

1. **Modularity**: Functions can be used independently of the environment
2. **Testability**: Each function can be tested in isolation
3. **Reusability**: Functions can be used in other projects or contexts
4. **Maintainability**: Cleaner separation of concerns
5. **Flexibility**: Easy to modify or extend individual components

## State Variables for External Use

Given a state consisting of:
- `formula`: Target molecular formula (e.g., "C6H12O6")
- `current_selfies`: Current SELFIES string (e.g., "[C][C][O]")
- `used_element_counts`: Dictionary of used elements (e.g., {"C": 2, "O": 1})
- `next_token`: Token to potentially add (e.g., "[H]")

You can use the utility functions to:
- Get valid actions: `get_action_mask(...)`
- Calculate rewards: `calculate_formula_completion_reward(...)`
- Validate operations: `validate_selfies_addition(...)`
- Update state: `update_atom_counts(...)`

This provides a clean interface for external systems to interact with the molecular building logic without needing the full environment class. 