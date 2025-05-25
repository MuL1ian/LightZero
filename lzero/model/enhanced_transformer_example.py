#!/usr/bin/env python3
"""
Usage example for the enhanced MuZero SELFIES transformer.
Shows how to integrate it into a training or inference pipeline.
"""

import torch
from muzero_transformer import MuZeroSelfiesTransformerEnhanced

class EnhancedMoleculeGenerator:
    """Example wrapper class showing how to use the enhanced transformer"""
    
    def __init__(self, target_formula: str, device: str = 'cuda'):
        self.target_formula = target_formula
        self.device = device
        
        # Initialize the enhanced model
        self.model = MuZeroSelfiesTransformerEnhanced(
            observation_shape=4096,
            max_len=128,
            target_formula=target_formula,
            device=device
        )
        
    def generate_molecule(self, spectrum: torch.Tensor, max_steps: int = 50):
        """
        Generate a molecule given a mass spectrum.
        
        Args:
            spectrum: Mass spectrum tensor of shape (4096,)
            max_steps: Maximum number of generation steps
            
        Returns:
            dict: Generation results including SELFIES string and completion status
        """
        # Prepare initial observation
        # Start with a small SELFIES part that will grow during generation
        selfies_part = torch.full((10,), self.model.tok.pad_token_id, dtype=torch.float)
        obs = torch.cat([spectrum, selfies_part]).to(self.device)
        
        # Initial inference
        output = self.model.initial_inference(obs)
        current_state = output.latent_state
        
        generated_actions = []
        selfies_history = []
        
        for step in range(max_steps):
            # Get current SELFIES string
            current_selfies = self.model._extract_selfies_from_latent_state(current_state)
            selfies_history.append(current_selfies[0])
            
            # Check if molecule is complete
            completion_status = self.model._check_completion_status(current_selfies)
            
            if completion_status[0]:
                print(f"Molecule completed at step {step}: {current_selfies[0]}")
                return {
                    'success': True,
                    'final_selfies': current_selfies[0],
                    'steps': step,
                    'actions': generated_actions,
                    'selfies_history': selfies_history,
                    'completion_reason': 'formula_complete'
                }
            
            # Get action probabilities
            logits = output.policy_logits if step == 0 else recurrent_output.policy_logits
            probs = torch.softmax(logits, dim=-1)
            
            # Sample action (or use greedy)
            action_idx = torch.multinomial(probs, 1).item()
            # action_idx = torch.argmax(probs).item()  # Greedy alternative
            
            generated_actions.append(action_idx)
            
            # Take action
            action_tensor = torch.tensor([action_idx], dtype=torch.float).unsqueeze(0).to(self.device)
            recurrent_output = self.model.recurrent_inference(current_state, action_tensor)
            current_state = recurrent_output.latent_state
            
            # Check if END token was selected
            try:
                from muzero_transformer import actions_list
                if actions_list[action_idx] == "<END>":
                    final_selfies = self.model._extract_selfies_from_latent_state(current_state)
                    return {
                        'success': True,
                        'final_selfies': final_selfies[0],
                        'steps': step + 1,
                        'actions': generated_actions,
                        'selfies_history': selfies_history,
                        'completion_reason': 'end_token'
                    }
            except:
                pass
        
        # Max steps reached
        final_selfies = self.model._extract_selfies_from_latent_state(current_state)
        return {
            'success': False,
            'final_selfies': final_selfies[0],
            'steps': max_steps,
            'actions': generated_actions,
            'selfies_history': selfies_history,
            'completion_reason': 'max_steps'
        }
    
    def update_target_formula(self, new_formula: str):
        """Update the target formula for generation"""
        self.target_formula = new_formula
        self.model.set_target_formula(new_formula)
        print(f"Updated target formula to: {new_formula}")

def example_usage():
    """Example of how to use the enhanced molecule generator"""
    print("=== Enhanced Molecule Generator Example ===")
    
    # Create generator for methane
    generator = EnhancedMoleculeGenerator(target_formula="CH4", device='cpu')
    
    # Create dummy spectrum
    spectrum = torch.randn(4096)
    
    print(f"Generating molecule for formula: {generator.target_formula}")
    result = generator.generate_molecule(spectrum, max_steps=20)
    
    print(f"Generation result:")
    print(f"  Success: {result['success']}")
    print(f"  Final SELFIES: {result['final_selfies']}")
    print(f"  Steps taken: {result['steps']}")
    print(f"  Completion reason: {result['completion_reason']}")
    print(f"  SELFIES history: {result['selfies_history']}")
    
    # Try with a different formula
    print(f"\n--- Switching to C2H6 ---")
    generator.update_target_formula("C2H6")
    
    result2 = generator.generate_molecule(spectrum, max_steps=30)
    print(f"Generation result for C2H6:")
    print(f"  Success: {result2['success']}")
    print(f"  Final SELFIES: {result2['final_selfies']}")
    print(f"  Steps taken: {result2['steps']}")
    print(f"  Completion reason: {result2['completion_reason']}")

if __name__ == "__main__":
    example_usage() 