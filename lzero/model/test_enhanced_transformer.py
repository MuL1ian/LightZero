#!/usr/bin/env python3
"""
Test script for the enhanced MuZero SELFIES transformer with formula masking.
"""

import torch
import numpy as np
from muzero_transformer import MuZeroSelfiesTransformerEnhanced

def test_enhanced_transformer():
    """Test the enhanced transformer with formula masking"""
    print("=== Testing Enhanced MuZero SELFIES Transformer ===")
    
    # Initialize the enhanced model with a target formula
    target_formula = "CH4"  # Methane
    model = MuZeroSelfiesTransformerEnhanced(
        observation_shape=4246,
        max_len=100,
        target_formula=target_formula,
        device='cpu'  # Use CPU for testing
    )
    
    print(f"Target formula: {target_formula}")
    print(f"Model initialized with spectrum_dim: {model.spectrum_dim}")
    
    # Create a dummy observation (spectrum + empty SELFIES)
    spectrum_part = torch.randn(4096)
    # The SELFIES part should be much smaller initially - just padding tokens
    selfies_part = torch.full((10,), model.tok.pad_token_id, dtype=torch.float)
    obs = torch.cat([spectrum_part, selfies_part])
    
    print(f"Observation shape: {obs.shape}")
    
    # Test initial inference
    print("\n--- Initial Inference ---")
    initial_output = model.initial_inference(obs)
    print(f"Initial value shape: {initial_output.value.shape}")
    print(f"Initial policy logits shape: {initial_output.policy_logits.shape}")
    print(f"Initial latent state shape: {initial_output.latent_state.shape}")
    
    # Check which actions are allowed initially
    initial_probs = torch.softmax(initial_output.policy_logits, dim=-1)
    top_actions = torch.topk(initial_probs, k=5)
    print(f"Top 5 initial actions (indices): {top_actions.indices.tolist()}")
    print(f"Top 5 initial action probs: {top_actions.values.tolist()}")
    
    # Test recurrent inference with a carbon action
    print("\n--- Recurrent Inference (Adding Carbon) ---")
    carbon_action_idx = None
    for i, action in enumerate(model.transformer.action_token_ids):
        token = model.tok.decode([action])
        if token == "[C]":
            carbon_action_idx = i
            break
    
    if carbon_action_idx is not None:
        action_tensor = torch.tensor([carbon_action_idx], dtype=torch.float).unsqueeze(0)
        print(f"Taking action: [C] (index {carbon_action_idx})")
        
        recurrent_output = model.recurrent_inference(initial_output.latent_state, action_tensor)
        print(f"After carbon - value shape: {recurrent_output.value.shape}")
        print(f"After carbon - policy logits shape: {recurrent_output.policy_logits.shape}")
        
        # Check which actions are allowed after adding carbon
        carbon_probs = torch.softmax(recurrent_output.policy_logits, dim=-1)
        top_actions_carbon = torch.topk(carbon_probs, k=5)
        print(f"Top 5 actions after carbon (indices): {top_actions_carbon.indices.tolist()}")
        print(f"Top 5 action probs after carbon: {top_actions_carbon.values.tolist()}")
        
        # Test adding hydrogen atoms
        print("\n--- Adding Hydrogen Atoms ---")
        hydrogen_action_idx = None
        for i, action in enumerate(model.transformer.action_token_ids):
            token = model.tok.decode([action])
            if token == "[H]":
                hydrogen_action_idx = i
                break
        
        if hydrogen_action_idx is not None:
            current_state = recurrent_output.latent_state
            
            # Add 4 hydrogen atoms to complete CH4
            for h_count in range(1, 5):
                h_action_tensor = torch.tensor([hydrogen_action_idx], dtype=torch.float).unsqueeze(0)
                print(f"Adding hydrogen #{h_count}")
                
                h_output = model.recurrent_inference(current_state, h_action_tensor)
                current_state = h_output.latent_state
                
                # Check if molecule is complete
                current_selfies = model._extract_selfies_from_latent_state(current_state)
                completion_status = model._check_completion_status(current_selfies)
                
                print(f"  Current SELFIES: {current_selfies[0]}")
                print(f"  Is complete: {completion_status[0]}")
                
                if completion_status[0]:
                    print(f"  ✅ Molecule completed after {h_count} hydrogens!")
                    
                    # Check final action probabilities
                    final_probs = torch.softmax(h_output.policy_logits, dim=-1)
                    end_token_idx = None
                    try:
                        from muzero_transformer import actions_list
                        end_token_idx = actions_list.index("<END>")
                        print(f"  END token probability: {final_probs[0, end_token_idx].item():.4f}")
                    except:
                        print("  Could not find END token index")
                    
                    break
    
    print("\n--- Testing Formula Change ---")
    # Test changing the target formula
    model.set_target_formula("C2H6")  # Ethane
    print(f"Changed target formula to: C2H6")
    
    # Test initial inference with new formula
    new_initial_output = model.initial_inference(obs)
    new_initial_probs = torch.softmax(new_initial_output.policy_logits, dim=-1)
    new_top_actions = torch.topk(new_initial_probs, k=5)
    print(f"Top 5 actions with C2H6 target: {new_top_actions.indices.tolist()}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_enhanced_transformer() 