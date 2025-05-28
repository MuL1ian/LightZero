# Global Reward Network - Circular Import Fix

## Problem Solved

The original implementation had a circular import issue where modules were trying to import functions from each other during initialization, causing the error:
```
cannot import name 'get_reward_function' from partially initialized module 'lzero.model.global_reward_network'
```

## Solution

**Replaced singleton pattern with global variables approach:**

### Before (Problematic):
- Used singleton class with complex initialization
- Modules imported specific functions: `from lzero.model.global_reward_network import get_reward_function`
- Circular dependencies during module initialization

### After (Fixed):
- Used simple global variables with thread-safe access
- Modules import the entire module: `from lzero.model import global_reward_network`
- Access functions through module: `global_reward_network.get_reward_function()`

## Key Changes

### 1. Global Reward Network Module (`lzero/model/global_reward_network.py`)

**Global Variables:**
```python
_global_reward_network = None
_global_reward_function = None
_global_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_lock = threading.Lock()
```

**Functions:**
- `initialize_global_reward_network()` - Initialize the network
- `get_reward_function()` - Get the reward function
- `get_reward_network()` - Get the network for training
- `update_reward_network_parameters()` - Update network parameters
- Thread-safe operations using `_lock`

### 2. Module Imports

**All modules now use:**
```python
from lzero.model import global_reward_network
```

**Instead of:**
```python
from lzero.model.global_reward_network import get_reward_function, initialize_global_reward_network
```

### 3. Usage Pattern

**Initialization:**
```python
global_reward_network.initialize_global_reward_network(
    max_selfies_len=100,
    device='cpu',
    checkpoint_path=None
)
```

**Getting reward function:**
```python
reward_function = global_reward_network.get_reward_function()
score = reward_function(selfies_string, spectrum_embed, formula_string)
```

**Getting network for training:**
```python
network = global_reward_network.get_reward_network()
if network is not None:
    # Train the network
    optimizer.step()
```

## Benefits

1. **No Circular Imports**: Modules import the entire module, not specific functions
2. **Thread Safety**: Global variables protected with locks
3. **Single Instance**: Only one reward network exists globally
4. **Shared Parameters**: All modules access the same network instance
5. **Easy Updates**: Parameter updates in one module are immediately available to all others

## Integration Points

### 1. MassGymEnv (`zoo/masspecgym/envs/massgymenv.py`)
- Initializes global reward network in `__init__`
- Uses reward function in `step()` method for computing rewards

### 2. MuZero Transformer (`lzero/model/muzero_transformer.py`)
- Uses reward function in `recurrent_inference()` for search-time rewards
- Computes similarity scores during MCTS rollouts

### 3. GumbelMuZero Agent (`lzero/agent/gumbel_muzero.py`)
- Initializes global reward network at agent startup
- Can update network parameters during training
- Placeholder for reward network training integration

## Testing

The `test_global_reward_network.py` script verifies:
- ✅ Module imports work without circular dependencies
- ✅ Network initialization succeeds
- ✅ Reward function calls work correctly
- ✅ Environment integration works
- ✅ Transformer integration works
- ✅ Parameter sharing and updates work
- ✅ Thread safety is maintained

## Usage Example

```python
# Initialize once (typically in main training script)
from lzero.model import global_reward_network
global_reward_network.initialize_global_reward_network(
    vocab_size=1000,
    max_selfies_len=100,
    device='cuda'
)

# Use in any module
reward_fn = global_reward_network.get_reward_function()
similarity_score = reward_fn("[C][C][O]", spectrum_embedding, "C2H6O")

# Update during training
network = global_reward_network.get_reward_network()
if network is not None:
    # Training code here
    loss.backward()
    optimizer.step()
    # Updates are automatically available to all modules
```

This approach ensures that the reward network is truly global, shared across all modules, and can be updated from any module while maintaining thread safety and avoiding circular import issues. 