"""
Global Reward Network Manager for MassGym Environment

This module provides a singleton pattern for managing a global reward network
that can be shared across different modules (environment, transformer, etc.)
and updated during training.

NEW: Added reward server architecture for subprocess-based environments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Callable, List, Tuple
import threading
import multiprocessing as mp
import queue
import time
import uuid
import os
import signal
from collections import defaultdict

# Global variables for the reward network
_global_reward_network = None
_global_reward_function = None
_global_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_lock = threading.Lock()

# Reward server globals
_reward_server = None
_reward_server_process = None
_request_queue = None
_response_queue = None
_server_enabled = False

# Try to import dependencies, but don't fail if they're not available
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))
    from reward_model.src.reward_nn import (
        MoleculeSpectrumMatcher,
        EMBED_DIM,
        FUSION_DIM,
        MAX_LEN,
        SPECTRUM_DIM,
        DROPOUT,
        SELFIES_MAX_LEN
    )
    REWARD_NN_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Could not import reward network: {e}")
    MoleculeSpectrumMatcher = None
    REWARD_NN_AVAILABLE = False
    # Fallback values if import fails
    EMBED_DIM = 1024
    FUSION_DIM = 1024
    MAX_LEN = 250
    SPECTRUM_DIM = 4096
    DROPOUT = 0.1
    SELFIES_MAX_LEN = 250

try:
    import selfies as sf
    from transformers import BertTokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Could not import tokenizers: {e}")
    TOKENIZERS_AVAILABLE = False


class DummyTokenizer:
    """Dummy tokenizer for testing when real tokenizer is not available"""
    def __init__(self, max_len=None):
        self.max_length = max_len or MAX_LEN
        self.pad_token_id = 0
        
    def encode_selfies(self, selfies_string, add_special_tokens=True):
        # Return dummy tokens
        return [1] * min(len(selfies_string), self.max_length)
        
    def get_vocab(self):
        return {f"token_{i}": i for i in range(1000)}


def _create_dummy_reward_function():
    """Create a dummy reward function that returns 0.0"""
    def dummy_reward_fn(selfies_string: str, spectrum_embed: torch.Tensor, formula_string: str = None) -> float:
        return 0.0
    return dummy_reward_fn


def _create_spectrum_batch_from_raw_data(peaks_data, device):
    """Create spectrum batch from raw peak data (fallback for non-embedded data)"""
    # This is a placeholder - implement based on your spectrum data format
    return {
        'spectrum_embeds': torch.zeros(1, 4096).to(device)
    }


def _create_real_reward_function(network, tokenizer, device):
    """Create the real reward function using the network"""
    def reward_fn(
        selfies_string: str,
        spectrum_embed: torch.Tensor,
        formula_string: str = None
    ) -> float:
        """
        Compute similarity between SELFIES and spectrum
        
        Args:
            selfies_string: SELFIES representation of molecule
            spectrum_embed: Spectrum embedding tensor [4096] (pre-computed)
            formula_string: Optional formula string (not used in current implementation)
            
        Returns:
            Similarity score as reward
        """
        if network is None:
            return 0.0
            
        try:
            network.eval()
            with torch.no_grad():
                # Encode SELFIES
                if tokenizer is not None:
                    if hasattr(tokenizer, 'encode_selfies'):
                        # Real SelfiesTokenizer
                        selfies_tokens = tokenizer.encode_selfies(selfies_string, add_special_tokens=True)
                        # The tokenizer should already handle padding to max_length
                        selfies_tensor = torch.tensor(selfies_tokens, dtype=torch.long).unsqueeze(0).to(device)
                        selfies_mask = (selfies_tensor != tokenizer.pad_token_id).to(device)
                    elif hasattr(tokenizer, 'max_length'):
                        # Dummy tokenizer
                        selfies_tokens = tokenizer.encode_selfies(selfies_string, add_special_tokens=True)
                        # Pad to max length
                        max_len = tokenizer.max_length
                        if len(selfies_tokens) > max_len:
                            selfies_tokens = selfies_tokens[:max_len]
                        else:
                            selfies_tokens.extend([tokenizer.pad_token_id] * (max_len - len(selfies_tokens)))
                        
                        selfies_tensor = torch.tensor(selfies_tokens, dtype=torch.long).unsqueeze(0).to(device)
                        selfies_mask = (selfies_tensor != tokenizer.pad_token_id).to(device)
                    else:
                        # Fallback
                        selfies_tensor = torch.zeros(1, MAX_LEN, dtype=torch.long).to(device)
                        selfies_mask = torch.ones(1, MAX_LEN, dtype=torch.bool).to(device)
                else:
                    # Dummy implementation if tokenizer not available
                    selfies_tensor = torch.zeros(1, MAX_LEN, dtype=torch.long).to(device)
                    selfies_mask = torch.ones(1, MAX_LEN, dtype=torch.bool).to(device)
                
                # Check if spectrum_embed is already a pre-computed embedding
                if spectrum_embed.dim() == 1 and spectrum_embed.shape[0] == 4096:
                    # Use pre-computed spectrum embedding directly
                    spectrum_embeds = spectrum_embed.unsqueeze(0).to(device)  # Add batch dimension
                    
                    # Encode SELFIES using the network's SELFIES encoder
                    molecule_embeds = network.selfies_encoder(selfies_tensor, selfies_mask)
                    molecule_embeds = network.molecule_projection(molecule_embeds)
                    
                    # Project spectrum embeddings to fusion space
                    spectrum_embeds = network.fingerprint_projection(spectrum_embeds)
                    
                    # Normalize embeddings
                    import torch.nn.functional as F
                    molecule_embeds = F.normalize(molecule_embeds, p=2, dim=-1)
                    spectrum_embeds = F.normalize(spectrum_embeds, p=2, dim=-1)
                    
                    # Compute similarity
                    similarity = torch.sum(molecule_embeds * spectrum_embeds, dim=1)
                    
                    return float(similarity.item()) / network.temperature
                else:
                    # Handle raw spectrum data (fallback to original implementation)
                    # This would be used if we had raw spectrum data instead of pre-computed embeddings
                    print("[INFO] Processing raw spectrum data through SpectraEncoderGrowing")
                    
                    # Create spectrum batch from raw data
                    spectrum_batch = _create_spectrum_batch_from_raw_data(spectrum_embed, device)
                    
                    # Get similarity score using the full network pipeline
                    similarity = network.predict_similarity(
                        selfies_tensor, selfies_mask, spectrum_batch
                    )
                    
                    return float(similarity.item())
                
        except Exception as e:
            print(f"[WARN] Error in reward network forward: {e}")
            return 0.0
    
    return reward_fn


def initialize_global_reward_network(
    vocab_size: int = None,
    selfies_embed_dim: int = EMBED_DIM,
    spectrum_fingerprint_dim: int = SPECTRUM_DIM,
    fusion_dim: int = FUSION_DIM,
    dropout: float = DROPOUT,
    max_selfies_len: int = SELFIES_MAX_LEN,
    device: str = None,
    checkpoint_path: str = None,
    enable_batching: bool = False,
    batch_size: int = 32,
    batch_timeout: float = 0.1,
    use_reward_server: bool = None  # New parameter
):
    """
    Initialize the global reward network.
    
    Args:
        vocab_size: Size of SELFIES vocabulary
        selfies_embed_dim: Embedding dimension for SELFIES (default: EMBED_DIM from reward_nn.py = 1024)
        spectrum_fingerprint_dim: Dimension of spectrum fingerprints (default: SPECTRUM_DIM from reward_nn.py = 4096)
        fusion_dim: Fusion layer dimension (default: FUSION_DIM from reward_nn.py = 1024)
        dropout: Dropout rate (default: DROPOUT from reward_nn.py = 0.1)
        max_selfies_len: Maximum SELFIES sequence length (default: MAX_LEN from reward_nn.py = 100)
        device: Device to run the network on
        checkpoint_path: Path to load pretrained weights. If None, uses default path 
                        'reward_model/diffms/models/reward_model/best_model.pt'.
        enable_batching: Whether to enable batched reward computation (deprecated - use reward server)
        batch_size: Maximum batch size for batched computation
        batch_timeout: Timeout for batched computation (seconds)
        use_reward_server: Whether to use reward server for subprocess environments.
                          If None, auto-detects based on multiprocessing context.
    """
    global _global_reward_network, _global_reward_function, _global_device
    
    print("[INFO] Initializing global reward network...")
    
    with _lock:
        if _global_reward_network is not None:
            print("[INFO] Global reward network already initialized")
            return
            
        if device is not None:
            _global_device = torch.device(device)
            
        # Set default checkpoint path if none provided
        if checkpoint_path is None:
            # Try to find the default checkpoint path relative to the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Navigate up to find the reward_model directory
            default_checkpoint_path = os.path.join(
                current_dir, '..', '..', '..', '..', 
                'reward_model', 'diffms', 'models', 'reward_model', 'best_model.pt'
            )
            default_checkpoint_path = os.path.abspath(default_checkpoint_path)
            
            if os.path.exists(default_checkpoint_path):
                checkpoint_path = default_checkpoint_path
                print(f"[INFO] Using default checkpoint path: {checkpoint_path}")
            else:
                print(f"[WARN] Default checkpoint not found at: {default_checkpoint_path}")
                print("[INFO] Continuing without checkpoint")
        
        # Auto-detect whether to use reward server
        if use_reward_server is None:
            # Check if we're in a subprocess environment
            current_process = mp.current_process()
            use_reward_server = current_process.name != 'MainProcess'
            if use_reward_server:
                print(f"[INFO] Auto-detected subprocess environment '{current_process.name}', using reward server")
            else:
                print("[INFO] Auto-detected main process, checking configuration for reward server...")
        
        # If use_reward_server is explicitly set to True, respect that setting
        if use_reward_server:
            current_process = mp.current_process()
            if current_process.name == 'MainProcess':
                print("[INFO] Main process configured to use reward server, starting server...")
            else:
                print(f"[INFO] Subprocess '{current_process.name}' configured to use reward server, connecting to server...")
        else:
            print("[INFO] Using direct reward computation")
        
        # Initialize tokenizer
        tokenizer = None
        if TOKENIZERS_AVAILABLE:
            try:
                # Try to import the real SELFIES tokenizer
                try:
                    from lzero.model.selfies_tokenizer import SelfiesTokenizer
                    tokenizer = SelfiesTokenizer(max_len=max_selfies_len)
                    if vocab_size is None:
                        vocab_size = len(tokenizer.get_vocab())
                    print("[INFO] Using real SelfiesTokenizer")
                except ImportError:
                    # Fallback to dummy tokenizer
                    tokenizer = DummyTokenizer(max_len=max_selfies_len)
                    if vocab_size is None:
                        vocab_size = len(tokenizer.get_vocab())
                    print("[WARN] Using dummy tokenizer")
            except Exception as e:
                print(f"[WARN] Error creating tokenizer: {e}")
                vocab_size = vocab_size or 1000
        else:
            vocab_size = vocab_size or 1000
        
        # Choose initialization strategy based on environment
        if use_reward_server:
            # For subprocess environments, use reward server
            print("[INFO] Initializing reward server for subprocess environment...")
            try:
                start_reward_server(
                    batch_size=batch_size,
                    timeout=batch_timeout,
                    checkpoint_path=checkpoint_path,
                    device=device
                )
                _global_reward_function = _create_server_reward_function()
                print("[INFO] Reward server initialization completed")
            except Exception as e:
                print(f"[WARN] Failed to initialize reward server: {e}, falling back to dummy function")
                _global_reward_function = _create_dummy_reward_function()
        else:
            # For main process, use direct network initialization
            print("[INFO] Initializing direct reward network for main process...")
            
            # Initialize reward network
            if REWARD_NN_AVAILABLE and MoleculeSpectrumMatcher is not None:
                try:
                    print(f"[INFO] Creating reward network with max_selfies_len: {max_selfies_len}")
                    _global_reward_network = MoleculeSpectrumMatcher(
                        vocab_size=vocab_size,
                        selfies_embed_dim=selfies_embed_dim,
                        spectrum_fingerprint_dim=spectrum_fingerprint_dim,
                        fusion_dim=fusion_dim,
                        dropout=dropout,
                        max_selfies_len=max_selfies_len
                    ).to(_global_device)
                    
                    # Load checkpoint if provided
                    if checkpoint_path and os.path.exists(checkpoint_path):
                        checkpoint = torch.load(checkpoint_path, map_location=_global_device)
                        if 'model_state_dict' in checkpoint:
                            _global_reward_network.load_state_dict(checkpoint['model_state_dict'])
                        else:
                            _global_reward_network.load_state_dict(checkpoint)
                        print(f"[INFO] Loaded reward network from {checkpoint_path}")
                    
                    # Create reward function
                    _global_reward_function = _create_real_reward_function(_global_reward_network, tokenizer, _global_device)
                    
                    print(f"[INFO] Global reward network initialized on {_global_device}")
                    
                except Exception as e:
                    print(f"[WARN] Error initializing reward network: {e}")
                    import traceback
                    traceback.print_exc()
                    _global_reward_network = None
                    _global_reward_function = _create_dummy_reward_function()
            else:
                print("[WARN] MoleculeSpectrumMatcher not available, using dummy network")
                _global_reward_network = None
                _global_reward_function = _create_dummy_reward_function()


def get_reward_function() -> Callable:
    """Get the global reward function"""
    global _global_reward_function
    
    if _global_reward_function is None:
        # Initialize with dummy function if not initialized
        _global_reward_function = _create_dummy_reward_function()
    
    return _global_reward_function


def get_reward_network() -> Optional[nn.Module]:
    """Get the global reward network for training"""
    global _global_reward_network
    return _global_reward_network


def is_batching_enabled() -> bool:
    """Check if batched reward computation is enabled (always False - deprecated)"""
    return False


def configure_batched_rewards_for_training(
    num_envs: int,
    batch_size: Optional[int] = None,
    batch_timeout: Optional[float] = None,
    auto_batch_size: bool = True
):
    """
    Configure batched rewards optimally for a training setup with multiple environments.
    
    Args:
        num_envs: Number of environment processes
        batch_size: Manual batch size (if None, will be auto-calculated)
        batch_timeout: Timeout for batching (if None, will be auto-calculated)
        auto_batch_size: Whether to automatically calculate optimal batch size
    
    Returns:
        dict: Configuration dictionary for environments
    """
    if auto_batch_size and batch_size is None:
        # Auto-calculate batch size based on number of environments
        # Use a reasonable fraction of environments for batching
        batch_size = min(max(num_envs // 4, 8), 64)  # Between 8 and 64
    elif batch_size is None:
        batch_size = 32
    
    if batch_timeout is None:
        # Auto-calculate timeout based on batch size
        # Smaller batches need shorter timeouts to maintain responsiveness
        batch_timeout = max(0.05, min(0.2, batch_size / 200.0))
    
    config = {
        'enable_batched_rewards': False,  # Deprecated - use reward server instead
        'batch_size': batch_size,
        'batch_timeout': batch_timeout,
    }
    
    print(f"[INFO] Configured batched rewards for {num_envs} environments:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Batch timeout: {batch_timeout:.3f}s")
    print(f"  - Expected throughput improvement: {min(batch_size / 4, 8):.1f}x")
    
    return config


def get_batching_stats() -> Dict[str, Any]:
    """
    Get statistics about the current batching performance.
    
    Returns:
        dict: Statistics about batching performance (deprecated - returns empty stats)
    """
    return {
        'batching_enabled': False,
        'batch_size': 0,
        'timeout': 0.0,
        'queue_size': 0,
        'pending_responses': 0,
        'note': 'Batching deprecated - use reward server instead'
    }


def cleanup_batched_rewards():
    """
    Clean up batched reward resources. Call this when shutting down training.
    """
    # Stop reward server if running
    stop_reward_server()
    print("[INFO] Batched reward resources cleaned up")


# For backward compatibility
class GlobalRewardNetworkManager:
    """Backward compatibility wrapper"""
    
    def get_forward_function(self):
        return get_reward_function()
    
    def get_network(self):
        return get_reward_network()
    
    def update_network_parameters(self, state_dict):
        # Deprecated - no longer supported
        print("[WARN] update_network_parameters is deprecated")
        pass
    
    def get_network_parameters(self):
        # Deprecated - no longer supported
        print("[WARN] get_network_parameters is deprecated")
        return None
    
    def enable_batching(self, batch_size=32, timeout=0.1):
        """Enable batched reward computation (deprecated)"""
        print("[WARN] enable_batching is deprecated - use reward server instead")
        return False
    
    def disable_batching(self):
        """Disable batched reward computation (deprecated)"""
        print("[WARN] disable_batching is deprecated")
        return False
    
    def get_batching_stats(self):
        """Get batching statistics"""
        return get_batching_stats()


def get_global_reward_network() -> GlobalRewardNetworkManager:
    """Get a manager wrapper for backward compatibility"""
    return GlobalRewardNetworkManager()


class RewardServer:
    """
    A dedicated reward server that runs in a separate process and handles
    batched reward computation for multiple environment subprocesses.
    """
    
    def __init__(self, network, tokenizer, device, batch_size=32, timeout=0.1):
        self.network = network
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.timeout = timeout
        self.running = False
        
    def start_server(self, request_queue, response_queue):
        """Start the reward server main loop"""
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.running = True
        
        print(f"[INFO] Reward server started (PID: {os.getpid()}, batch_size: {self.batch_size})")
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            self._server_loop()
        except KeyboardInterrupt:
            print("[INFO] Reward server interrupted")
        except Exception as e:
            print(f"[ERROR] Reward server error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[INFO] Reward server shutting down")
            self.running = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"[INFO] Reward server received signal {signum}, shutting down...")
        self.running = False
    
    def _server_loop(self):
        """Main server loop that processes reward requests in batches"""
        while self.running:
            try:
                # Collect requests for batching
                requests = []
                start_time = time.time()
                
                # Collect requests until batch is full or timeout
                while (len(requests) < self.batch_size and 
                       time.time() - start_time < self.timeout and 
                       self.running):
                    try:
                        request = self.request_queue.get(timeout=0.01)
                        requests.append(request)
                    except:  # queue.Empty or other queue exceptions
                        continue
                
                # Process batch if we have any requests
                if requests and self.running:
                    self._process_batch(requests)
                    
            except Exception as e:
                print(f"[ERROR] Error in reward server loop: {e}")
                time.sleep(0.01)
    
    def _process_batch(self, requests: List[Tuple[str, str, torch.Tensor, str]]):
        """Process a batch of reward computation requests"""
        try:
            if self.network is None:
                # Return dummy rewards
                for request_id, _, _, _ in requests:
                    self.response_queue.put((request_id, 0.0))
                return
            
            # Extract data from requests
            request_ids = [req[0] for req in requests]
            selfies_strings = [req[1] for req in requests]
            spectrum_embeds = [req[2] for req in requests]
            formula_strings = [req[3] for req in requests]
            
            self.network.eval()
            with torch.no_grad():
                # Batch encode SELFIES
                selfies_tensors = []
                selfies_masks = []
                
                for selfies_string in selfies_strings:
                    if self.tokenizer is not None:
                        if hasattr(self.tokenizer, 'encode_selfies'):
                            # Real SelfiesTokenizer
                            selfies_tokens = self.tokenizer.encode_selfies(selfies_string, add_special_tokens=True)
                            selfies_tensor = torch.tensor(selfies_tokens, dtype=torch.long).to(self.device)
                            selfies_mask = (selfies_tensor != self.tokenizer.pad_token_id).to(self.device)
                        else:
                            # Fallback
                            selfies_tensor = torch.zeros(MAX_LEN, dtype=torch.long).to(self.device)
                            selfies_mask = torch.ones(MAX_LEN, dtype=torch.bool).to(self.device)
                    else:
                        # Dummy implementation
                        selfies_tensor = torch.zeros(MAX_LEN, dtype=torch.long).to(self.device)
                        selfies_mask = torch.ones(MAX_LEN, dtype=torch.bool).to(self.device)
                    
                    selfies_tensors.append(selfies_tensor)
                    selfies_masks.append(selfies_mask)
                
                # Stack into batch tensors
                batch_selfies = torch.stack(selfies_tensors)  # [batch_size, max_len]
                batch_masks = torch.stack(selfies_masks)      # [batch_size, max_len]
                
                # Stack spectrum embeddings
                batch_spectrums = torch.stack([embed.to(self.device) for embed in spectrum_embeds])  # [batch_size, 4096]
                
                # Encode SELFIES using the network's SELFIES encoder
                molecule_embeds = self.network.selfies_encoder(batch_selfies, batch_masks)
                molecule_embeds = self.network.molecule_projection(molecule_embeds)
                
                # Project spectrum embeddings to fusion space
                spectrum_embeds_proj = self.network.fingerprint_projection(batch_spectrums)
                
                # Normalize embeddings
                molecule_embeds = F.normalize(molecule_embeds, p=2, dim=-1)
                spectrum_embeds_proj = F.normalize(spectrum_embeds_proj, p=2, dim=-1)
                
                # Compute similarities
                similarities = torch.sum(molecule_embeds * spectrum_embeds_proj, dim=1)
                similarities = similarities / self.network.temperature
                
                # Send results back
                for request_id, similarity in zip(request_ids, similarities):
                    self.response_queue.put((request_id, float(similarity.item())))
                    
        except Exception as e:
            print(f"[ERROR] Error processing batch in reward server: {e}")
            # Return dummy rewards for failed batch
            for request_id, _, _, _ in requests:
                self.response_queue.put((request_id, 0.0)) 


def start_reward_server(
    batch_size: int = 32,
    timeout: float = 0.1,
    checkpoint_path: str = None,
    device: str = None
):
    """
    Start a dedicated reward server process for subprocess-based environments.
    
    Args:
        batch_size: Maximum batch size for reward computation
        timeout: Timeout for batching (seconds)
        checkpoint_path: Path to reward network checkpoint
        device: Device to run the network on
    
    Returns:
        tuple: (request_queue, response_queue) for communicating with the server
    """
    global _reward_server_process, _request_queue, _response_queue, _server_enabled
    
    if _server_enabled and _reward_server_process is not None:
        print("[INFO] Reward server already running")
        return _request_queue, _response_queue
    
    print(f"[INFO] Starting reward server (batch_size={batch_size}, timeout={timeout}s)")
    
    # Create multiprocessing queues for communication
    ctx = mp.get_context('spawn')  # Use spawn for CUDA compatibility
    _request_queue = ctx.Queue()
    _response_queue = ctx.Queue()
    
    # Start the server process
    _reward_server_process = ctx.Process(
        target=_reward_server_worker,
        args=(_request_queue, _response_queue, batch_size, timeout, checkpoint_path, device),
        daemon=True,
        name='reward_server'
    )
    _reward_server_process.start()
    _server_enabled = True
    
    print(f"[INFO] Reward server started (PID: {_reward_server_process.pid})")
    return _request_queue, _response_queue


def _reward_server_worker(request_queue, response_queue, batch_size, timeout, checkpoint_path, device):
    """Worker function that runs the reward server in a separate process"""
    try:
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        
        # Initialize tokenizer
        tokenizer = None
        if TOKENIZERS_AVAILABLE:
            try:
                from lzero.model.selfies_tokenizer import SelfiesTokenizer
                tokenizer = SelfiesTokenizer(max_len=SELFIES_MAX_LEN)
                print("[INFO] Reward server using real SelfiesTokenizer")
            except ImportError:
                tokenizer = DummyTokenizer(max_len=SELFIES_MAX_LEN)
                print("[INFO] Reward server using dummy tokenizer")
        else:
            tokenizer = DummyTokenizer(max_len=SELFIES_MAX_LEN)
            print("[INFO] Reward server using dummy tokenizer (tokenizers not available)")
        
        # Initialize reward network
        network = None
        if REWARD_NN_AVAILABLE and MoleculeSpectrumMatcher is not None:
            try:
                vocab_size = len(tokenizer.get_vocab()) if tokenizer else 1000
                network = MoleculeSpectrumMatcher(
                    vocab_size=vocab_size,
                    selfies_embed_dim=EMBED_DIM,
                    spectrum_fingerprint_dim=SPECTRUM_DIM,
                    fusion_dim=FUSION_DIM,
                    dropout=DROPOUT,
                    max_selfies_len=SELFIES_MAX_LEN
                ).to(device)
                
                # Load checkpoint if provided
                if checkpoint_path and os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    if 'model_state_dict' in checkpoint:
                        network.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        network.load_state_dict(checkpoint)
                    print(f"[INFO] Reward server loaded checkpoint from {checkpoint_path}")
                
                print(f"[INFO] Reward server initialized network on {device}")
                
            except Exception as e:
                print(f"[WARN] Reward server failed to initialize network: {e}")
                network = None
        else:
            print("[WARN] Reward server using dummy network (MoleculeSpectrumMatcher not available)")
        
        # Create and start the server
        server = RewardServer(network, tokenizer, device, batch_size, timeout)
        server.start_server(request_queue, response_queue)
        
    except Exception as e:
        print(f"[ERROR] Reward server worker error: {e}")
        import traceback
        traceback.print_exc()


def stop_reward_server():
    """Stop the reward server process"""
    global _reward_server_process, _server_enabled, _request_queue, _response_queue
    
    if _reward_server_process is not None:
        print("[INFO] Stopping reward server...")
        _reward_server_process.terminate()
        _reward_server_process.join(timeout=5)
        if _reward_server_process.is_alive():
            print("[WARN] Reward server did not stop gracefully, killing...")
            _reward_server_process.kill()
        _reward_server_process = None
        _server_enabled = False
        _request_queue = None
        _response_queue = None
        print("[INFO] Reward server stopped")


def _create_server_reward_function():
    """Create a reward function that communicates with the reward server"""
    def server_reward_fn(
        selfies_string: str,
        spectrum_embed: torch.Tensor,
        formula_string: str = None
    ) -> float:
        """
        Compute similarity between SELFIES and spectrum using the reward server
        
        Args:
            selfies_string: SELFIES representation of molecule
            spectrum_embed: Spectrum embedding tensor [4096] (pre-computed)
            formula_string: Optional formula string (not used in current implementation)
            
        Returns:
            Similarity score as reward
        """
        global _request_queue, _response_queue, _server_enabled
        
        if not _server_enabled or _request_queue is None or _response_queue is None:
            print("[WARN] Reward server not available, returning dummy reward")
            return 0.0
        
        try:
            # Generate unique request ID
            request_id = str(uuid.uuid4())
            
            # Send request to server
            request = (request_id, selfies_string, spectrum_embed.cpu(), formula_string)
            _request_queue.put(request, timeout=1.0)
            
            # Wait for response
            start_time = time.time()
            while time.time() - start_time < 5.0:  # 5 second timeout
                try:
                    response_id, reward = _response_queue.get(timeout=0.1)
                    if response_id == request_id:
                        return reward
                    else:
                        # Put back response for another request
                        _response_queue.put((response_id, reward))
                except:  # queue.Empty
                    continue
            
            print(f"[WARN] Timeout waiting for reward server response for request {request_id}")
            return 0.0
            
        except Exception as e:
            print(f"[WARN] Error communicating with reward server: {e}")
            return 0.0
    
    return server_reward_fn


def is_reward_server_enabled() -> bool:
    """Check if the reward server is enabled and running (subprocess-safe)"""
    global _server_enabled, _reward_server_process
    
    # In subprocess environments, we can't safely call is_alive() on a process
    # created in the parent process. Instead, we rely on the _server_enabled flag
    # and check if we have valid communication queues.
    import multiprocessing as mp
    current_process = mp.current_process()
    
    if current_process.name != 'MainProcess':
        # We're in a subprocess - only check the flag and queue availability
        return _server_enabled and _request_queue is not None and _response_queue is not None
    else:
        # We're in the main process - safe to check process status
        return _server_enabled and _reward_server_process is not None and _reward_server_process.is_alive() 