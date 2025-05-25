#!/usr/bin/env python3
"""
Example usage of the decoupled utility functions for molecular formula parsing,
atom tracking, and action masking.

This demonstrates how to use the functions independently of the environment class.
"""

from utils import (
    parse_formula_counts,
    extract_element_from_token,
    get_allowed_elements_from_formula,
    get_action_mask,
    update_atom_counts,
    validate_selfies_addition,
    remove_last_token_from_selfies,
    calculate_formula_completion_reward,
    check_formula_match,
    get_state_info
)
import numpy as np


def example_standalone_usage():
    """
    Example of using the utility functions independently to get action masks,
    rewards, etc. given a state (formula, current selfies, next token).
    """
    print("=== Standalone Utility Functions Example ===\n")
    
    # Define the state
    target_formula = "C6H12O6"  # Glucose
    current_selfies = "[C][C][O]"  # Partial molecule
    used_element_counts = {"C": 2, "O": 1}  # Current atom usage
    
    # Define action space
    actions_list = [
        '[C]', '[H]', '[O]', '[N]', '[=C]', '[=O]', 
        '[Ring1]', '[Branch1]', '<END>', '<REMOVE>'
    ]
    atom_tokens = ['[C]', '[H]', '[O]', '[N]']
    bonded_atom_tokens = ['[=C]', '[=O]']
    
    print(f"Target Formula: {target_formula}")
    print(f"Current SELFIES: {current_selfies}")
    print(f"Used Elements: {used_element_counts}")
    print(f"Actions: {actions_list}\n")
    
    # 1. Parse formula to get target counts
    target_counts = parse_formula_counts(target_formula)
    print(f"1. Target element counts: {target_counts}")
    
    # 2. Get allowed elements from formula
    allowed_elements = get_allowed_elements_from_formula(target_formula)
    print(f"2. Allowed element tokens: {sorted(list(allowed_elements))[:10]}...")  # Show first 10
    
    # 3. Get action mask
    action_mask = get_action_mask(
        formula=target_formula,
        used_element_counts=used_element_counts,
        actions_list=actions_list,
        atom_tokens=atom_tokens,
        bonded_atom_tokens=bonded_atom_tokens,
        current_selfies=current_selfies,
        formula_masking=True,
        end_token='<END>',
        remove_token='<REMOVE>'
    )
    
    print(f"3. Action mask: {action_mask}")
    valid_actions = [actions_list[i] for i, valid in enumerate(action_mask) if valid]
    print(f"   Valid actions: {valid_actions}")
    
    # 4. Test adding a new token
    next_token = '[H]'
    print(f"\n4. Testing addition of token: {next_token}")
    
    # Check if SELFIES addition is valid
    is_valid = validate_selfies_addition(current_selfies, next_token)
    print(f"   SELFIES validation: {is_valid}")
    
    # Update atom counts
    if is_valid:
        new_used_counts = update_atom_counts(next_token, used_element_counts, increment=True)
        print(f"   Updated atom counts: {new_used_counts}")
        
        # Calculate completion reward
        completion_reward = calculate_formula_completion_reward(target_formula, new_used_counts)
        print(f"   Completion reward: {completion_reward:.3f}")
        
        # Check if formula matches
        formula_match = check_formula_match(target_formula, new_used_counts)
        print(f"   Formula match: {formula_match}")
    
    # 5. Test removing a token
    print(f"\n5. Testing token removal from: {current_selfies}")
    updated_selfies, removed_token = remove_last_token_from_selfies(current_selfies)
    print(f"   Updated SELFIES: {updated_selfies}")
    print(f"   Removed token: {removed_token}")
    
    if removed_token:
        new_used_counts = update_atom_counts(removed_token, used_element_counts, increment=False)
        print(f"   Updated atom counts after removal: {new_used_counts}")
    
    # 6. Get comprehensive state info
    print(f"\n6. Comprehensive state information:")
    state_info = get_state_info(target_formula, current_selfies, used_element_counts)
    for key, value in state_info.items():
        print(f"   {key}: {value}")


def example_step_by_step_molecule_building():
    """
    Example of building a molecule step by step using the utility functions.
    """
    print("\n\n=== Step-by-Step Molecule Building Example ===\n")
    
    target_formula = "CH4"  # Methane
    current_selfies = ""
    used_element_counts = {}
    
    # Define action space
    actions_list = ['[C]', '[H]', '[=C]', '[=H]', '<END>', '<REMOVE>']
    atom_tokens = ['[C]', '[H]']
    bonded_atom_tokens = ['[=C]', '[=H]']
    
    print(f"Building molecule with formula: {target_formula}")
    print(f"Target counts: {parse_formula_counts(target_formula)}\n")
    
    # Sequence of actions to build CH4
    action_sequence = ['[C]', '[H]', '[H]', '[H]', '[H]', '<END>']
    
    for step, action in enumerate(action_sequence):
        print(f"Step {step + 1}: Adding {action}")
        print(f"  Current SELFIES: '{current_selfies}'")
        print(f"  Used counts: {used_element_counts}")
        
        # Get action mask
        mask = get_action_mask(
            formula=target_formula,
            used_element_counts=used_element_counts,
            actions_list=actions_list,
            atom_tokens=atom_tokens,
            bonded_atom_tokens=bonded_atom_tokens,
            current_selfies=current_selfies,
            formula_masking=True,
            end_token='<END>',
            remove_token='<REMOVE>'
        )
        
        valid_actions = [actions_list[i] for i, valid in enumerate(mask) if valid]
        print(f"  Valid actions: {valid_actions}")
        
        if action in valid_actions:
            print(f"  ✅ Action {action} is valid")
            
            if action == '<END>':
                # Check if molecule is complete
                formula_match = check_formula_match(target_formula, used_element_counts)
                completion_reward = calculate_formula_completion_reward(target_formula, used_element_counts)
                print(f"  Formula complete: {formula_match}")
                print(f"  Final reward: {completion_reward:.3f}")
                break
            else:
                # Validate and add token
                if validate_selfies_addition(current_selfies, action):
                    current_selfies += action
                    used_element_counts = update_atom_counts(action, used_element_counts, increment=True)
                    print(f"  ✅ Token added successfully")
                else:
                    print(f"  ❌ Invalid SELFIES addition")
        else:
            print(f"  ❌ Action {action} is not valid")
        
        print()


if __name__ == "__main__":
    # Run the examples
    example_standalone_usage()
    example_step_by_step_molecule_building()
    
    print("\n=== Summary ===")
    print("The utility functions provide a clean interface for:")
    print("1. Parsing molecular formulas")
    print("2. Extracting elements from SELFIES tokens")
    print("3. Generating action masks based on formula constraints")
    print("4. Tracking atom usage")
    print("5. Validating SELFIES operations")
    print("6. Calculating rewards and completion status")
    print("\nThese functions can be used independently of the environment class!") 