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
import numpy as np
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
_reward_processor_process = None  # Renamed from _reward_server_process
_queue_manager_process = None     # New: separate process for queue management
_request_queue = None
_response_dict = None             # Changed from response_queue to response_dict
_processor_request_queue = None   # Queue Manager -> Reward Processor (requests)
_processor_response_queue = None  # Reward Processor -> Queue Manager (responses)
_server_enabled = False
_client_timeout = 5.0  # Default client timeout for server communication
_multiprocessing_manager = None   # Manager for shared response dictionary

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
    use_reward_server: bool = None,  # New parameter
    client_timeout: float = None  # New parameter for client-side timeout
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
        client_timeout: Timeout for client requests to reward server (seconds).
                       If None, auto-calculated as min(60.0, max(5.0, batch_timeout * 10 + 5)).
    """
    global _global_reward_network, _global_reward_function, _global_device, _client_timeout
    
    print("[INFO] Initializing global reward network...")
    
    with _lock:
        if _global_reward_network is not None:
            print("[INFO] Global reward network already initialized")
            return
            
        if device is not None:
            _global_device = torch.device(device)
        
        # Set client timeout based on batch timeout if not provided
        if client_timeout is None:
            # Client timeout should be larger than batch timeout to allow for:
            # 1. Batching delay (batch_timeout)
            # 2. Network processing time
            # 3. Inter-process communication overhead
            # 4. Multiple batches if the request gets queued
            # Auto-calculate: minimum 5s, maximum 60s, or batch_timeout * 10 + 5
            _client_timeout = min(60.0, max(5.0, batch_timeout * 10 + 5))
        else:
            _client_timeout = client_timeout
        
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
        
        # Auto-detect whether to use reward server
        if use_reward_server is None:
            # Check if we're in a subprocess environment
            current_process = mp.current_process()
            use_reward_server = current_process.name != 'MainProcess'
        
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
                except ImportError:
                    # Fallback to dummy tokenizer
                    tokenizer = DummyTokenizer(max_len=max_selfies_len)
                    if vocab_size is None:
                        vocab_size = len(tokenizer.get_vocab())
            except Exception as e:
                vocab_size = vocab_size or 1000
        else:
            vocab_size = vocab_size or 1000
        
        # Choose initialization strategy based on environment
        if use_reward_server:
            # For subprocess environments, use reward server
            try:
                start_reward_server(
                    batch_size=batch_size,
                    timeout=batch_timeout,
                    checkpoint_path=checkpoint_path,
                    device=device
                )
                _global_reward_function = _create_server_reward_function()
            except Exception as e:
                print(f"[WARN] Failed to initialize reward server: {e}, falling back to dummy function")
                _global_reward_function = _create_dummy_reward_function()
        else:
            # For main process, use direct network initialization
            # Initialize reward network
            if REWARD_NN_AVAILABLE and MoleculeSpectrumMatcher is not None:
                try:
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
                    
                    # Create reward function
                    _global_reward_function = _create_real_reward_function(_global_reward_network, tokenizer, _global_device)
                    
                    print(f"[INFO] Global reward network initialized on {_global_device}")
                    
                except Exception as e:
                    print(f"[WARN] Error initializing reward network: {e}")
                    _global_reward_network = None
                    _global_reward_function = _create_dummy_reward_function()
            else:
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
    
    # Reduced logging for batched rewards configuration
    
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
                
                # First, try to get at least one request (blocking)
                try:
                    request = self.request_queue.get(timeout=1.0)  # Wait up to 1s for first request
                    requests.append(request)
                except:  # queue.Empty
                    continue  # No requests, continue loop
                
                # Now collect additional requests until batch is full or timeout
                while len(requests) < self.batch_size and self.running:
                    remaining_time = self.timeout - (time.time() - start_time)
                    if remaining_time <= 0:
                        break  # Timeout reached
                    
                    try:
                        # Use the remaining time as timeout to avoid waiting too long
                        timeout = min(remaining_time, 0.01)  # At most 10ms per attempt
                        request = self.request_queue.get(timeout=timeout)
                        requests.append(request)
                    except:  # queue.Empty
                        # No more requests available immediately
                        # Check if we should wait a bit more or process what we have
                        if len(requests) >= self.batch_size // 2:  # Half batch threshold
                            break  # Process what we have
                        elif remaining_time > 0.01:
                            time.sleep(0.001)  # Small sleep to avoid busy waiting
                        else:
                            break  # Timeout reached
                
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
    Start a dedicated reward server with two-process architecture:
    1. Queue Manager: Handles client requests and response routing
    2. Reward Processor: Performs batched neural network inference
    
    Args:
        batch_size: Maximum batch size for reward computation
        timeout: Timeout for batching (seconds)
        checkpoint_path: Path to reward network checkpoint
        device: Device to run the network on
    
    Returns:
        tuple: (request_queue, response_dict) for communicating with the server
    """
    global _reward_processor_process, _queue_manager_process, _request_queue, _response_dict, _processor_request_queue, _processor_response_queue, _server_enabled
    global _multiprocessing_manager
    
    if _server_enabled and _reward_processor_process is not None:
        print("[INFO] Reward server already running")
        return _request_queue, _response_dict
    
    print(f"[INFO] Starting reward server (batch_size={batch_size}, timeout={timeout}s)")
    
    # Create multiprocessing queues and manager for communication
    ctx = mp.get_context('spawn')  # Use spawn for CUDA compatibility
    _multiprocessing_manager = ctx.Manager()
    
    _request_queue = ctx.Queue()                      # Client -> Queue Manager
    _response_dict = _multiprocessing_manager.dict()  # Queue Manager -> Client (shared dict)
    _processor_request_queue = ctx.Queue()             # Queue Manager -> Reward Processor (requests)
    _processor_response_queue = ctx.Queue()            # Reward Processor -> Queue Manager (responses)
    
    # Start the reward processor process (does neural network computation)
    _reward_processor_process = ctx.Process(
        target=_reward_processor_worker,
        args=(_processor_request_queue, _processor_response_queue, batch_size, timeout, checkpoint_path, device),
        daemon=True,
        name='reward_processor'
    )
    _reward_processor_process.start()
    
    # Start the queue manager process (handles request/response routing)
    _queue_manager_process = ctx.Process(
        target=_queue_manager_worker,
        args=(_request_queue, _response_dict, _processor_request_queue, _processor_response_queue),
        daemon=True,
        name='queue_manager'
    )
    _queue_manager_process.start()
    
    _server_enabled = True
    
    print(f"[INFO] Reward processor started (PID: {_reward_processor_process.pid})")
    print(f"[INFO] Queue manager started (PID: {_queue_manager_process.pid})")
    return _request_queue, _response_dict


def _queue_manager_worker(request_queue, response_dict, processor_request_queue, processor_response_queue):
    """Queue manager worker process that routes requests and responses"""
    try:
        print(f"[INFO] Queue manager started (PID: {os.getpid()})")
        
        def signal_handler(signum, frame):
            print(f"[INFO] Queue manager received signal {signum}, shutting down...")
            return
        
        # Register signal handlers for graceful shutdown
        import signal
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Processing settings
        processing_interval = 0.01  # Process every 10ms
        
        # Monitoring counters (reduced reporting frequency)
        total_requests = 0
        total_responses = 0
        last_report_time = time.time()
        report_interval = 300.0  # Report stats every 5 minutes instead of 30 seconds
        
        while True:
            try:
                start_time = time.time()
                
                # Process multiple client requests in one cycle
                requests_processed = 0
                for _ in range(10):  # Process up to 10 requests per cycle
                    try:
                        # Read request from client
                        client_request = request_queue.get(timeout=0.001)  # Very short timeout
                        
                        # Handle different request types
                        if len(client_request) == 2 and client_request[0].startswith('train_'):
                            # Training request: (request_id, training_data)
                            request_id = client_request[0]
                            processor_request_queue.put(client_request)  # Forward as-is
                        elif len(client_request) == 4:
                            # Inference request: (request_id, selfies_string, spectrum_embed, formula_string)
                            request_id = client_request[0]
                            processor_request_queue.put(client_request)  # Forward as-is
                        else:
                            # Only log warnings, not every request
                            continue
                        
                        requests_processed += 1
                        
                    except:  # queue.Empty
                        break
                
                # Process multiple processor responses in one cycle
                responses_processed = 0
                for _ in range(10):  # Process up to 10 responses per cycle
                    try:
                        # Read response from processor
                        response = processor_response_queue.get(timeout=0.001)  # Very short timeout
                        
                        # Handle both types of responses
                        if len(response) == 2:
                            # Both inference and training responses have 2 elements
                            request_id, result = response
                            # Store response in shared dictionary by request ID
                            response_dict[request_id] = result
                            responses_processed += 1
                        
                    except:  # queue.Empty
                        break
                
                # Update counters
                total_requests += requests_processed
                total_responses += responses_processed
                
                # Periodic monitoring report (much less frequent)
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    if total_requests > 0 or total_responses > 0:
                        print(f"[INFO] Queue manager stats: {total_requests} requests, {total_responses} responses in last {report_interval/60:.1f} minutes")
                        total_requests = 0
                        total_responses = 0
                    last_report_time = current_time
                
                # Sleep for the remainder of the processing interval
                elapsed = time.time() - start_time
                sleep_time = max(0, processing_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"[ERROR] Queue manager error: {e}")
                time.sleep(processing_interval)
                
    except KeyboardInterrupt:
        print("[INFO] Queue manager interrupted")
    except Exception as e:
        print(f"[ERROR] Queue manager worker error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[INFO] Queue manager shutting down")


def _reward_processor_worker(processor_request_queue, processor_response_queue, batch_size, timeout, checkpoint_path, device):
    """
    Extended reward processor worker that handles both serving and training requests.
    This process can switch between inference mode and training mode based on requests.
    """
    try:
        print("[INFO] Reward processor starting with training support...")
        
        # Set device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)
        
        # Initialize network and tokenizer
        if REWARD_NN_AVAILABLE and MoleculeSpectrumMatcher is not None:
            try:
                # Initialize network
                network = MoleculeSpectrumMatcher(
                    vocab_size=50000,  # Will be updated when tokenizer loads
                    selfies_embed_dim=EMBED_DIM,
                    spectrum_fingerprint_dim=SPECTRUM_DIM,
                    fusion_dim=FUSION_DIM,
                    dropout=DROPOUT,
                    max_selfies_len=SELFIES_MAX_LEN
                ).to(device)
                
                # Load checkpoint if provided
                if checkpoint_path and os.path.exists(checkpoint_path):
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    network.load_state_dict(checkpoint, strict=False)
                    print(f"[INFO] Reward network checkpoint loaded from {checkpoint_path}")
                
                # Initialize tokenizer
                if TOKENIZERS_AVAILABLE:
                    from main.LightZero.lzero.model.selfies_tokenizer import SelfiesTokenizer
                    tokenizer = SelfiesTokenizer(max_len=MAX_LEN)
                    # Update network vocab size if needed
                    if hasattr(network, 'update_vocab_size'):
                        network.update_vocab_size(len(tokenizer.get_vocab()))
                else:
                    tokenizer = DummyTokenizer(max_len=MAX_LEN)
                
            except Exception as e:
                print(f"[ERROR] Failed to initialize reward network: {e}")
                network = None
                tokenizer = DummyTokenizer(max_len=MAX_LEN)
        else:
            network = None
            tokenizer = DummyTokenizer(max_len=MAX_LEN)
        
        # Create the enhanced processor with training capabilities
        processor = EnhancedRewardProcessor(network, tokenizer, device, batch_size, timeout)
        processor.start_processing(processor_request_queue, processor_response_queue)
        
    except Exception as e:
        print(f"[ERROR] Reward processor worker error: {e}")
        import traceback
        traceback.print_exc()


class EnhancedRewardProcessor:
    """
    Enhanced reward processor that handles both inference and training requests.
    Supports switching between serving mode and training mode based on request types.
    """
    
    def __init__(self, network, tokenizer, device, batch_size=32, timeout=0.1):
        self.network = network
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.timeout = timeout
        self.running = False
        
        # Training-related attributes
        self.optimizer = None
        self.training_enabled = False
        self.current_mode = "serving"  # "serving" or "training"
        
        # Initialize optimizer if network is available
        if self.network is not None:
            self._initialize_optimizer()
        
    def _initialize_optimizer(self):
        """Initialize optimizer for reward network training"""
        try:
            # Default training configuration - can be updated via training requests
            learning_rate = 1e-4
            weight_decay = 1e-4
            
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            self.training_enabled = True
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize optimizer: {e}")
            self.optimizer = None
            self.training_enabled = False
    
    def start_processing(self, processor_request_queue, processor_response_queue):
        """Start the enhanced processor main loop with training support"""
        self.processor_request_queue = processor_request_queue
        self.processor_response_queue = processor_response_queue
        self.running = True
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        try:
            self._processor_loop()
        except KeyboardInterrupt:
            print("[INFO] Enhanced reward processor interrupted")
        except Exception as e:
            print(f"[ERROR] Enhanced reward processor error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("[INFO] Enhanced reward processor shutting down")
            self.running = False
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print(f"[INFO] Enhanced reward processor received signal {signum}, shutting down...")
        self.running = False
    
    def _processor_loop(self):
        """Enhanced main processing loop with both inference and training support"""
        processing_interval = 0.01  # Process every 10ms
        
        # Monitoring counters (reduced reporting frequency)
        total_inference_batches = 0
        total_training_batches = 0
        total_inference_requests = 0
        total_training_requests = 0
        last_report_time = time.time()
        report_interval = 300.0  # Report stats every 5 minutes instead of 30 seconds
        
        print(f"[INFO] Enhanced reward processor loop started with training support")
        
        while self.running:
            try:
                start_time = time.time()
                
                # Collect all available requests (up to batch_size) without blocking
                inference_requests = []
                training_requests = []
                
                for _ in range(self.batch_size * 2):  # Allow more requests since we have two types
                    try:
                        request_data = self.processor_request_queue.get(timeout=0.001)  # Very short timeout
                        
                        # Classify request type based on the structure
                        if len(request_data) == 2 and request_data[0].startswith('train_'):
                            # Training request: ('train_REQUEST_ID', training_data_dict)
                            training_requests.append(request_data)
                        elif len(request_data) == 4:
                            # Inference request: (request_id, selfies, spectrum_embed, formula)
                            inference_requests.append(request_data)
                            
                    except:  # queue.Empty
                        break  # No more requests available immediately
                
                # Process inference requests (serving mode)
                if inference_requests:
                    self.current_mode = "serving"
                    try:
                        self._process_inference_batch(inference_requests)
                        total_inference_batches += 1
                        total_inference_requests += len(inference_requests)
                    except Exception as e:
                        print(f"[ERROR] Enhanced reward processor failed to process inference batch: {e}")
                        # Send dummy responses for failed batch
                        for request_id, _, _, _ in inference_requests:
                            self.processor_response_queue.put((request_id, 0.0))
                
                # Process training requests (training mode)
                if training_requests and self.training_enabled:
                    self.current_mode = "training"
                    try:
                        self._process_training_batch(training_requests)
                        total_training_batches += 1
                        total_training_requests += len(training_requests)
                    except Exception as e:
                        print(f"[ERROR] Enhanced reward processor failed to process training batch: {e}")
                        # Send error responses for failed training batch
                        for request_id, _ in training_requests:
                            self.processor_response_queue.put((request_id, {
                                'status': 'error',
                                'message': str(e)
                            }))
                
                # Periodic monitoring report (much less frequent)
                current_time = time.time()
                if current_time - last_report_time >= report_interval:
                    total_batches = total_inference_batches + total_training_batches
                    total_requests = total_inference_requests + total_training_requests
                    
                    if total_batches > 0:
                        avg_batch_size = total_requests / total_batches if total_batches > 0 else 0
                        print(f"[INFO] Enhanced reward processor stats: "
                              f"Inference: {total_inference_batches} batches ({total_inference_requests} requests), "
                              f"Training: {total_training_batches} batches ({total_training_requests} requests), "
                              f"avg batch size: {avg_batch_size:.1f} in last {report_interval/60:.1f} minutes")
                    
                    # Reset counters
                    total_inference_batches = 0
                    total_training_batches = 0
                    total_inference_requests = 0
                    total_training_requests = 0
                    last_report_time = current_time
                
                # Sleep for the remainder of the processing interval
                elapsed = time.time() - start_time
                sleep_time = max(0, processing_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except Exception as e:
                print(f"[ERROR] Error in enhanced reward processor loop: {e}")
                time.sleep(processing_interval)  # Still maintain the interval on error
    
    def _process_inference_batch(self, requests):
        """Process a batch of reward computation requests (inference mode)"""
        try:
            if self.network is None:
                # Return dummy rewards
                for request_id, _, _, _ in requests:
                    self.processor_response_queue.put((request_id, 0.0))
                return
            
            # Extract data from requests
            request_ids = [req[0] for req in requests]
            selfies_strings = [req[1] for req in requests]
            spectrum_embeds = [req[2] for req in requests]
            formula_strings = [req[3] for req in requests]
            
            # Perform batched neural network inference
            rewards = self._compute_batch_rewards(selfies_strings, spectrum_embeds)
            
            # Send results back via response queue
            for request_id, reward in zip(request_ids, rewards):
                self.processor_response_queue.put((request_id, reward))
                
        except Exception as e:
            print(f"[ERROR] Error processing inference batch: {e}")
            # Return dummy rewards for failed batch
            for request_id, _, _, _ in requests:
                self.processor_response_queue.put((request_id, 0.0))
    
    def _process_training_batch(self, requests):
        """Process a batch of training requests (training mode)"""
        if self.network is None or self.optimizer is None:
            # Send error responses
            for request_id, _ in requests:
                self.processor_response_queue.put((request_id, {
                    'status': 'error', 
                    'message': 'Network or optimizer not available'
                }))
            return
        
        try:
            # Only log training batches occasionally to reduce noise
            if len(requests) > 5:  # Only log larger training batches
                print(f"[INFO] Processing training batch with {len(requests)} requests")
            
            # Process each training request
            for request_id, training_data in requests:
                try:
                    result = self._train_on_gag_data(training_data)
                    self.processor_response_queue.put((request_id, result))
                except Exception as e:
                    print(f"[ERROR] Error training on request {request_id}: {e}")
                    self.processor_response_queue.put((request_id, {
                        'status': 'error',
                        'message': str(e)
                    }))
                    
        except Exception as e:
            print(f"[ERROR] Error processing training batch: {e}")
            # Send error responses for failed batch
            for request_id, _ in requests:
                self.processor_response_queue.put((request_id, {
                    'status': 'error',
                    'message': str(e)
                }))
    
    def _train_on_gag_data(self, training_data):
        """
        Train the reward network on GAG (Generated vs Ground-truth Adversarial) data
        
        Args:
            training_data: Dictionary containing:
                - generated_selfies: List of generated SELFIES strings
                - ground_truth_selfies: List of ground-truth SELFIES strings  
                - spectrum_embeds: List of spectrum embeddings
                - learning_rate: Optional learning rate override
                - weight_decay: Optional weight decay override
        
        Returns:
            Training results dictionary
        """
        try:
            self.network.train()
            
            # Extract training data
            generated_selfies = training_data.get('generated_selfies', [])
            ground_truth_selfies = training_data.get('ground_truth_selfies', [])
            spectrum_embeds = training_data.get('spectrum_embeds', [])
            
            # Optional hyperparameter overrides
            lr_override = training_data.get('learning_rate', None)
            wd_override = training_data.get('weight_decay', None)
            
            if not generated_selfies or not ground_truth_selfies or not spectrum_embeds:
                return {'status': 'error', 'message': 'Empty training data'}
            
            if len(generated_selfies) != len(ground_truth_selfies) or len(generated_selfies) != len(spectrum_embeds):
                return {'status': 'error', 'message': 'Mismatched training data lengths'}
            
            # Update optimizer if hyperparameters changed
            if lr_override is not None or wd_override is not None:
                self._update_optimizer_params(lr_override, wd_override)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Compute rewards for generated and ground-truth SELFIES
            # These should already be tensors with gradients from _compute_batch_rewards
            generated_rewards = self._compute_batch_rewards_with_grad(generated_selfies, spectrum_embeds)
            ground_truth_rewards = self._compute_batch_rewards_with_grad(ground_truth_selfies, spectrum_embeds)
            
            # Ensure tensors are on correct device
            generated_rewards_tensor = generated_rewards.to(self.device)
            ground_truth_rewards_tensor = ground_truth_rewards.to(self.device)
            
            # Compute GAG losses - focus only on preference loss to avoid conflicting objectives
            preference_loss = self._compute_preference_loss(generated_rewards_tensor, ground_truth_rewards_tensor)
            
            # Total loss (only preference loss to ensure GT > Generated)
            total_loss = preference_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Return training results
            return {
                'status': 'success',
                'preference_loss': float(preference_loss.item()),
                'total_loss': float(total_loss.item()),
                'mean_generated_reward': float(torch.mean(generated_rewards_tensor).item()),
                'mean_ground_truth_reward': float(torch.mean(ground_truth_rewards_tensor).item()),
                'num_pairs': len(generated_selfies),
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'weight_decay': self.optimizer.param_groups[0]['weight_decay']
            }
            
        except Exception as e:
            print(f"[ERROR] Error in _train_on_gag_data: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'error', 'message': str(e)}
    
    def _update_optimizer_params(self, learning_rate=None, weight_decay=None):
        """Update optimizer parameters"""
        if learning_rate is not None:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        if weight_decay is not None:
            for param_group in self.optimizer.param_groups:
                param_group['weight_decay'] = weight_decay
    
    def _compute_adversarial_loss(self, generated_rewards, ground_truth_rewards):
        """Compute adversarial loss (encourage ground-truth rewards to be high)"""
        return -torch.mean(ground_truth_rewards)
    
    def _compute_preference_loss(self, generated_rewards, ground_truth_rewards):
        """Compute preference loss (encourage ground-truth to be preferred over generated)"""
        # Simple ranking loss: minimize (generated - ground_truth) to make GT > Generated
        ranking_loss = torch.mean(torch.relu(generated_rewards - ground_truth_rewards + 0.1))  # 0.1 margin
        return ranking_loss
    
    def _compute_batch_rewards(self, selfies_strings, spectrum_embeds):
        """Compute rewards for a batch of SELFIES and spectrum embeddings (same as original)"""
        # Enable gradients only during training mode
        ctx = torch.enable_grad() if self.current_mode == "training" else torch.no_grad()
        with ctx:
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
            
            return [float(sim.item()) for sim in similarities]
    
    def _compute_batch_rewards_with_grad(self, selfies_strings, spectrum_embeds):
        """Compute rewards for training (returns tensor with gradients)"""
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
        
        return similarities  # Return tensor with gradients


def stop_reward_server():
    """Stop the reward server processes"""
    global _reward_processor_process, _queue_manager_process, _server_enabled, _request_queue, _response_dict, _processor_request_queue, _processor_response_queue
    global _multiprocessing_manager
    
    if _reward_processor_process is not None:
        print("[INFO] Stopping reward processor...")
        _reward_processor_process.terminate()
        _reward_processor_process.join(timeout=5)
        if _reward_processor_process.is_alive():
            print("[WARN] Reward processor did not stop gracefully, killing...")
            _reward_processor_process.kill()
        _reward_processor_process = None
    
    if _queue_manager_process is not None:
        print("[INFO] Stopping queue manager...")
        _queue_manager_process.terminate()
        _queue_manager_process.join(timeout=5)
        if _queue_manager_process.is_alive():
            print("[WARN] Queue manager did not stop gracefully, killing...")
            _queue_manager_process.kill()
        _queue_manager_process = None
    
    # Clean up manager resources
    if _multiprocessing_manager is not None:
        try:
            _multiprocessing_manager.shutdown()
        except:
            pass
        _multiprocessing_manager = None
    
    _server_enabled = False
    _request_queue = None
    _response_dict = None
    _processor_request_queue = None
    _processor_response_queue = None
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
        global _request_queue, _response_dict, _server_enabled, _client_timeout
        
        if not _server_enabled or _request_queue is None or _response_dict is None:
            print("[WARN] Reward server not available, returning dummy reward")
            return 0.0
        
        try:
            # Generate unique request ID
            request_id = str(uuid.uuid4())
            
            # Ensure spectrum_embed is a tensor (server expects tensors!)
            if isinstance(spectrum_embed, torch.Tensor):
                spectrum_embed_tensor = spectrum_embed.cpu()  # Move to CPU for serialization
            elif isinstance(spectrum_embed, np.ndarray):
                spectrum_embed_tensor = torch.from_numpy(spectrum_embed).float()
            else:
                # Convert other types to tensor
                spectrum_embed_tensor = torch.tensor(spectrum_embed, dtype=torch.float32)
            
            # Send request to server
            request = (request_id, selfies_string, spectrum_embed_tensor, formula_string)
            _request_queue.put(request, timeout=_client_timeout)
            
            # Wait for response by polling the shared dictionary with adaptive intervals
            start_time = time.time()
            check_interval = 0.001  # Start with 1ms checks
            max_interval = 0.01     # Maximum 10ms between checks
            
            while time.time() - start_time < _client_timeout:
                if request_id in _response_dict:
                    reward = _response_dict[request_id]
                    # Clean up the response from the dict
                    del _response_dict[request_id]
                    return reward
                
                # Adaptive polling - start fast, then slow down slightly
                time.sleep(check_interval)
                check_interval = min(max_interval, check_interval * 1.1)
            
            # Timeout
            print(f"[WARN] Timeout waiting for reward server response for request {request_id} (timeout: {_client_timeout:.3f}s)")
            # Try to clean up the request if it somehow got processed after timeout
            _response_dict.pop(request_id, None)
            return 0.0
            
        except Exception as e:
            print(f"[WARN] Error communicating with reward server: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    return server_reward_fn


def get_client_timeout() -> float:
    """Get the current client timeout for reward server communication"""
    global _client_timeout
    return _client_timeout


def is_reward_server_enabled() -> bool:
    """Check if the reward server is enabled and running (subprocess-safe)"""
    global _server_enabled, _reward_processor_process, _queue_manager_process
    
    # In subprocess environments, we can't safely call is_alive() on a process
    # created in the parent process. Instead, we rely on the _server_enabled flag
    # and check if we have valid communication queues.
    import multiprocessing as mp
    current_process = mp.current_process()
    
    if current_process.name != 'MainProcess':
        # We're in a subprocess - only check the flag and queue availability
        return _server_enabled and _request_queue is not None and _response_dict is not None
    else:
        # We're in the main process - safe to check process status
        return (_server_enabled and 
                _reward_processor_process is not None and _reward_processor_process.is_alive() and
                _queue_manager_process is not None and _queue_manager_process.is_alive())


# Helper function to send training data to reward server
def send_training_data_to_server(
    generated_selfies: List[str],
    ground_truth_selfies: List[str], 
    spectrum_embeds: List[torch.Tensor],
    learning_rate: float = None,
    weight_decay: float = None,
    timeout: float = 10.0
) -> Dict[str, Any]:
    """
    Send training data to the reward server for network training
    
    Args:
        generated_selfies: List of generated SELFIES strings
        ground_truth_selfies: List of ground-truth SELFIES strings
        spectrum_embeds: List of spectrum embeddings
        learning_rate: Optional learning rate override
        weight_decay: Optional weight decay override  
        timeout: Timeout for server communication
        
    Returns:
        Training results dictionary
    """
    global _request_queue, _response_dict, _server_enabled
    
    if not _server_enabled or _request_queue is None or _response_dict is None:
        return {'status': 'error', 'message': 'Reward server not available'}
    
    try:
        # Generate unique training request ID
        request_id = f"train_{str(uuid.uuid4())}"
        
        # Prepare training data - ensure all spectrum embeds are tensors (server expects tensors!)
        processed_spectrum_embeds = []
        for embed in spectrum_embeds:
            if isinstance(embed, torch.Tensor):
                processed_spectrum_embeds.append(embed.cpu())  # Move to CPU for serialization
            elif isinstance(embed, np.ndarray):
                processed_spectrum_embeds.append(torch.from_numpy(embed).float())
            else:
                # Convert other types to tensor
                processed_spectrum_embeds.append(torch.tensor(embed, dtype=torch.float32))
        
        training_data = {
            'generated_selfies': generated_selfies,
            'ground_truth_selfies': ground_truth_selfies,
            'spectrum_embeds': processed_spectrum_embeds,
        }
        
        if learning_rate is not None:
            training_data['learning_rate'] = learning_rate
        if weight_decay is not None:
            training_data['weight_decay'] = weight_decay
        
        # Send training request to server
        request = (request_id, training_data)
        _request_queue.put(request, timeout=timeout)
        
        # Wait for training response
        start_time = time.time()
        check_interval = 0.01  # Check every 10ms for training responses
        
        while time.time() - start_time < timeout:
            if request_id in _response_dict:
                result = _response_dict[request_id]
                # Clean up the response from the dict
                del _response_dict[request_id]
                return result
            
            time.sleep(check_interval)
        
        # Timeout
        print(f"[WARN] Timeout waiting for training response from reward server (timeout: {timeout:.1f}s)")
        _response_dict.pop(request_id, None)
        return {'status': 'error', 'message': 'Server timeout'}
        
    except Exception as e:
        print(f"[ERROR] Error sending training data to reward server: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'message': str(e)} 