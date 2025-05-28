#!/usr/bin/env python3
"""
Test script to verify Chemical Formula tokenization in MassGymEnv.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from zoo.masspecgym.envs.massgymenv import MassGymEnv, ChemicalFormulaTokenizer

def test_chemical_formula_tokenizer():
    """Test the ChemicalFormulaTokenizer directly."""
    
    print("Testing ChemicalFormulaTokenizer directly:")
    print("=" * 50)
    
    tokenizer = ChemicalFormulaTokenizer(max_length=50)
    
    # Test different formulas
    test_formulas = [
        "C6H12O6",    # glucose - should preserve C, H, O
        "H2O",        # water - should preserve H, O
        "C2H6O",      # ethanol - should preserve C, H, O
        "CaCl2",      # calcium chloride - should preserve Ca, Cl (case sensitive!)
        "NaCl",       # sodium chloride - should preserve Na, Cl
        "C8H10N4O2",  # caffeine - should preserve all elements
        "C21H30O2",   # very long formula
        "",           # empty formula
        "Ca",         # single two-letter element
        "C",          # single one-letter element
        "C12",        # element with large number
    ]
    
    for formula in test_formulas:
        print(f"\nFormula: '{formula}'")
        
        # Test tokenization
        tokens = tokenizer.tokenize_formula(formula)
        print(f"  Tokens: {tokens}")
        
        # Test encoding
        encoded = tokenizer.encode(formula, return_tensors='pt')
        print(f"  Encoded shape: {encoded.shape}")
        print(f"  Encoded (first 10): {encoded[0][:10].tolist()}")
        
        # Test decoding
        decoded = tokenizer.decode(encoded[0], skip_special_tokens=True)
        print(f"  Decoded: '{decoded}'")
        
        # Check if original formula is preserved
        if decoded == formula:
            print(f"  ✓ Perfect reconstruction!")
        else:
            print(f"  ✗ Reconstruction failed: '{formula}' -> '{decoded}'")

def test_formula_tokenization():
    """Test the formula tokenization functionality in MassGymEnv."""
    
    # Create environment configuration
    cfg = {
        'env_id': 'mass_spec_env',
        'max_len': 100,
        'formula_max_len': 50,
        'debug': True,
        'formula_masking': True,
    }
    
    # Create environment
    env = MassGymEnv(cfg)
    
    # Test different formulas
    test_formulas = [
        "C6H12O6",  # glucose
        "H2O",      # water
        "C2H6O",    # ethanol
        "CaCl2",    # calcium chloride - test case sensitivity
        "C8H10N4O2", # caffeine
        "C21H30O2",  # very long formula
        "",          # empty formula
    ]
    
    print("\n\nTesting Chemical Formula tokenization in MassGymEnv:")
    print("=" * 60)
    
    for formula in test_formulas:
        print(f"\nFormula: '{formula}'")
        
        # Test the _encode_formula method
        try:
            encoded = env._encode_formula(formula)
            print(f"  Encoded shape: {encoded.shape}")
            print(f"  Encoded type: {type(encoded)}")
            print(f"  First 10 tokens: {encoded[:10].tolist()}")
            
            # Decode back to check
            decoded = env.formula_tokenizer.decode(encoded, skip_special_tokens=True)
            print(f"  Decoded: '{decoded}'")
            
            # Check preservation of case-sensitive elements
            if 'Ca' in formula and 'Ca' in decoded:
                print(f"  ✓ Calcium (Ca) case preserved!")
            elif 'Cl' in formula and 'Cl' in decoded:
                print(f"  ✓ Chlorine (Cl) case preserved!")
            
            if decoded == formula:
                print(f"  ✓ Perfect formula reconstruction!")
            else:
                print(f"  ⚠ Formula changed: '{formula}' -> '{decoded}'")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Testing environment reset with formula tokenization:")
    
    # Test environment reset
    try:
        timestep = env.reset()
        obs = timestep.obs['observation']
        print(f"Observation shape: {obs.shape}")
        print(f"Expected shape: 4246 (4096 spectrum + 100 selfies + 50 formula)")
        
        # Extract formula part from observation
        formula_part = obs[4196:]  # Last 50 elements
        decoded_formula = env.formula_tokenizer.decode(formula_part.long(), skip_special_tokens=True)
        print(f"Formula from observation: '{decoded_formula}'")
        print(f"Target formula: '{env.target_spectrum['formulas']}'")
        
        if decoded_formula == env.target_spectrum['formulas']:
            print("✓ Formula correctly embedded in observation!")
        else:
            print("⚠ Formula embedding may have issues")
            
    except Exception as e:
        print(f"Error during environment test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chemical_formula_tokenizer()
    test_formula_tokenization() 