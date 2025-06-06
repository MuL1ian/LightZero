"""
Utility functions for molecular formula parsing, atom tracking, and action masking.
These functions are decoupled from the environment class for better modularity.
"""

import re
import numpy as np
import selfies as sf
from typing import Dict, Set, List, Optional, Tuple, Any


def parse_formula_counts(formula: str) -> Dict[str, int]:
    """
    Parse a molecular formula and return a dictionary with element counts.
    
    Args:
        formula (str): The formula string, e.g. "C6H12O6"
        
    Returns:
        Dict[str, int]: Dictionary mapping element symbols to their counts
        
    Examples:
        >>> parse_formula_counts("C6H12O6")
        {'C': 6, 'H': 12, 'O': 6}
        >>> parse_formula_counts("CH4")
        {'C': 1, 'H': 4}
        >>> parse_formula_counts("NaCl")
        {'Na': 1, 'Cl': 1}
    """
    if not formula:
        return {}
    
    element_counts = {}
    i = 0
    
    while i < len(formula):
        # Handle two-letter elements (e.g., Cl, Br)
        if i + 1 < len(formula) and formula[i].isupper() and formula[i+1].islower():
            element = formula[i:i+2]
            i += 2
        # Handle single-letter elements (e.g., C, N, O)
        elif formula[i].isupper():
            element = formula[i]
            i += 1
        # Skip non-element characters
        elif formula[i].isdigit():
            i += 1
            continue
        else:
            i += 1
            continue
        
        # Parse the count following the element
        count_str = ""
        while i < len(formula) and formula[i].isdigit():
            count_str += formula[i]
            i += 1
        
        count = int(count_str) if count_str else 1
        element_counts[element] = element_counts.get(element, 0) + count
    
    return element_counts


def extract_element_from_token(token: str) -> Optional[str]:
    """
    Extract the base element symbol from a SELFIES token.
    
    Args:
        token (str): SELFIES token like '[C]', '[=N+1]', etc.
        
    Returns:
        Optional[str]: Base element symbol or None if not found
        
    Examples:
        >>> extract_element_from_token('[C]')
        'C'
        >>> extract_element_from_token('[=N+1]')
        'N'
        >>> extract_element_from_token('[Ring1]')
        None
    """
    # Handle special tokens
    special_tokens = {
        '[Ring1]', '[Ring2]', '[Ring3]', 
        '[Branch1]', '[Branch2]', '[Branch3]',
        '[=Ring1]', '[=Ring2]', '[=Ring3]', 
        '[=Branch1]', '[=Branch2]', '[=Branch3]',
        '[#Branch1]', '[#Branch2]', '[#Branch3]'
    }
    
    if token in special_tokens:
        return None
    
    # Extract element from token like [C], [=N+1], [#O], etc.
    match = re.match(r'^\[([-=#+]*)([A-Za-z]+)', token)
    if match:
        return match.group(2)
    return None


def get_allowed_elements_from_formula(formula: str) -> Set[str]:
    """
    Extract allowed elements from a formula and return a set of all variants.
    
    Args:
        formula (str): The formula string, e.g. "C6H12O6"
        
    Returns:
        Set[str]: Set of allowed element tokens including all variants
    """
    if not formula:
        return set()
    
    base_elements = set()
    i = 0
    
    # Define element variants (excluding hydrogen since it's implicit in SELFIES)
    element_variants = {
        'C': ['[C]', '[C+1]', '[C-1]', '[=C]', '[=C+1]', '[=C-1]', '[#C]', '[#C+1]', '[#C-1]'],
        'N': ['[N]', '[N+1]', '[N-1]', '[=N]', '[=N+1]', '[=N-1]', '[#N]', '[#N+1]'],
        'O': ['[O]', '[O+1]', '[O-1]', '[=O]', '[=O+1]'],
        'S': ['[S]', '[S+1]', '[S-1]', '[=S]', '[=S+1]', '[=S-1]', '[#S]', '[#S+1]', '[#S-1]'],
        'P': ['[P]', '[P+1]', '[P-1]', '[=P]', '[=P+1]', '[=P-1]', '[#P]', '[#P+1]', '[#P-1]'],
        'B': ['[B]', '[B+1]', '[B-1]', '[=B]', '[=B+1]', '[=B-1]', '[#B]', '[#B-1]'],
        'F': ['[F]'],
        'I': ['[I]'],
        'Cl': ['[Cl]'],
        'Br': ['[Br]']
        # Note: Hydrogen [H] removed since hydrogens are implicit in SELFIES
    }
    
    ring_variants = ['[Ring1]', '[Ring2]', '[Ring3]']
    branch_variants = ['[Branch1]', '[Branch2]', '[Branch3]']
    
    # Parse formula to get base elements
    while i < len(formula):
        if i + 1 < len(formula) and formula[i].isupper() and formula[i+1].islower():
            symbol = formula[i:i+2]
            i += 2
        elif formula[i].isupper():
            symbol = formula[i]
            i += 1
        elif formula[i].isdigit():
            i += 1
            continue
        else:
            i += 1
            continue
        
        base_elements.add(symbol)
        
        while i < len(formula) and formula[i].isdigit():
            i += 1
    
    # Build allowed tokens set
    allowed_tokens = set()
    for element in base_elements:
        if element in element_variants:
            allowed_tokens.update(element_variants[element])
    
    # Always allow ring and branch tokens
    allowed_tokens.update(ring_variants)
    allowed_tokens.update(branch_variants)
    
    return allowed_tokens


def get_action_mask(
    formula: str,
    used_element_counts: Dict[str, int],
    actions_list: List[str],
    atom_tokens: List[str],
    bonded_atom_tokens: List[str],
    current_selfies: str = "",
    formula_masking: bool = True,
    end_token: str = "<END>",
    remove_token: str = "<REMOVE>",
    special_tokens: List[str] = None,
    min_formula_completion: float = 0.8,  # Minimum completion ratio before END is allowed
    allow_early_end_after_steps: int = 20  # Allow END after this many steps even if incomplete
) -> np.ndarray:
    """
    Generate a boolean mask over the action space indicating which actions are valid.
    
    Args:
        formula (str): Target molecular formula
        used_element_counts (Dict[str, int]): Current count of used elements
        actions_list (List[str]): List of all possible actions
        atom_tokens (List[str]): List of pure atom tokens
        bonded_atom_tokens (List[str]): List of bonded atom tokens
        current_selfies (str): Current SELFIES string
        formula_masking (bool): Whether to apply formula-based masking
        end_token (str): End token identifier
        remove_token (str): Remove token identifier
        special_tokens (List[str]): List of special tokens to mask out
        min_formula_completion (float): Minimum completion ratio before END token is allowed
        allow_early_end_after_steps (int): Allow END token after this many steps even if incomplete
        
    Returns:
        np.ndarray: Boolean mask array
    """
    mask = np.ones(len(actions_list), dtype=np.bool_)
    
    if special_tokens is None:
        special_tokens = []
    
    try:
        if formula_masking and formula:
            target_element_counts = parse_formula_counts(formula)
            allowed_elements = get_allowed_elements_from_formula(formula)
            
            for i, action in enumerate(actions_list):
                if action in atom_tokens or action in bonded_atom_tokens:
                    # Check if element type is allowed
                    if action not in allowed_elements:
                        mask[i] = False
                        continue
                    
                    # Check if we haven't exceeded the atom count for this element
                    element = extract_element_from_token(action)
                    if element:
                        target_count = target_element_counts.get(element, 0)
                        used_count = used_element_counts.get(element, 0)
                        
                        # If we've already used all atoms of this type, mask it out
                        if used_count >= target_count:
                            mask[i] = False
    
    except Exception as e:
        print(f"[WARN] get_action_mask formula filtering failed: {e}")
        # fallback: allow all
        mask[:] = True
    
    # Intelligent END token masking to prevent early termination
    if end_token in actions_list:
        idx = actions_list.index(end_token)
        
        # Calculate formula completion ratio
        formula_completion = 0.0
        if formula_masking and formula:
            try:
                formula_completion = calculate_formula_completion_reward(formula, used_element_counts)
            except:
                formula_completion = 0.0
        
        # Calculate number of steps taken (approximated by used atom count)
        total_used_atoms = sum(used_element_counts.values()) if used_element_counts else 0
        
        # Allow END token only if:
        # 1. Formula completion is high enough, OR
        # 2. We've taken many steps (prevents infinite episodes), OR
        # 3. Formula masking is disabled
        if (formula_completion >= min_formula_completion or 
            total_used_atoms >= allow_early_end_after_steps or 
            not formula_masking or 
            not formula):
            mask[idx] = True
        else:
            mask[idx] = False  # Mask out END token to prevent early termination
    
    # Remove token not allowed if no current molecule
    if not current_selfies and remove_token in actions_list:
        idx = actions_list.index(remove_token)
        mask[idx] = False
    
    # Mask out special tokens
    for special in special_tokens:
        if special in actions_list:
            mask[actions_list.index(special)] = False
    
    return mask


def update_atom_counts(
    action_token: str,
    used_element_counts: Dict[str, int],
    increment: bool = True
) -> Dict[str, int]:
    """
    Update the used element counts when an atom is added or removed.
    
    Args:
        action_token (str): The SELFIES token being added/removed
        used_element_counts (Dict[str, int]): Current count of used elements
        increment (bool): Whether to increment (True) or decrement (False)
        
    Returns:
        Dict[str, int]: Updated element counts
    """
    updated_counts = used_element_counts.copy()
    element = extract_element_from_token(action_token)
    
    if element:
        current_count = updated_counts.get(element, 0)
        if increment:
            updated_counts[element] = current_count + 1
        else:
            updated_counts[element] = max(0, current_count - 1)
    
    return updated_counts


def validate_selfies_addition(current_selfies: str, new_token: str) -> bool:
    """
    Check if adding a new token to the current SELFIES string results in valid SELFIES.
    
    Args:
        current_selfies (str): Current SELFIES string
        new_token (str): Token to be added
        
    Returns:
        bool: True if the resulting SELFIES is valid, False otherwise
    """
    try:
        new_selfies_candidate = current_selfies + new_token
        sf.split_selfies(new_selfies_candidate)
        return True
    except:
        return False


def remove_last_token_from_selfies(current_selfies: str) -> Tuple[str, Optional[str]]:
    """
    Remove the last token from a SELFIES string and return the updated string and removed token.
    
    Args:
        current_selfies (str): Current SELFIES string
        
    Returns:
        Tuple[str, Optional[str]]: (updated_selfies, removed_token)
    """
    try:
        if not current_selfies:
            return current_selfies, None
        
        tokens = sf.split_selfies(current_selfies)
        if not tokens:
            return "", None
        
        removed_token = tokens[-1]
        new_tokens = tokens[:-1]
        updated_selfies = ''.join(new_tokens)
        
        return updated_selfies, removed_token
    except:
        return current_selfies, None


def calculate_formula_completion_reward(
    target_formula: str,
    used_element_counts: Dict[str, int]
) -> float:
    """
    Calculate a reward based on how close the current molecule is to the target formula.
    
    Args:
        target_formula (str): Target molecular formula
        used_element_counts (Dict[str, int]): Current count of used elements
        
    Returns:
        float: Completion reward between 0 and 1
    """
    if not target_formula:
        return 0.0
    
    target_counts = parse_formula_counts(target_formula)
    
    if not target_counts:
        return 0.0
    
    total_target_atoms = sum(target_counts.values())
    total_used_atoms = sum(used_element_counts.values())
    
    # Calculate how many atoms are correctly placed
    correct_atoms = 0
    for element, target_count in target_counts.items():
        used_count = used_element_counts.get(element, 0)
        correct_atoms += min(used_count, target_count)
    
    # Penalize for using wrong elements or too many atoms
    penalty = max(0, total_used_atoms - total_target_atoms) * 0.1
    
    # Calculate completion ratio
    completion_ratio = correct_atoms / total_target_atoms if total_target_atoms > 0 else 0.0
    
    return max(0.0, completion_ratio - penalty)


def check_formula_match(
    target_formula: str,
    used_element_counts: Dict[str, int]
) -> bool:
    """
    Check if the used element counts exactly match the target formula.
    
    Args:
        target_formula (str): Target molecular formula
        used_element_counts (Dict[str, int]): Current count of used elements
        
    Returns:
        bool: True if counts match exactly, False otherwise
    """
    if not target_formula:
        return len(used_element_counts) == 0
    
    target_counts = parse_formula_counts(target_formula)
    
    # Check if all target elements are present with correct counts
    for element, target_count in target_counts.items():
        if used_element_counts.get(element, 0) != target_count:
            return False
    
    # Check if no extra elements are used
    for element, used_count in used_element_counts.items():
        if used_count > 0 and target_counts.get(element, 0) == 0:
            return False
    
    return True


def get_state_info(
    formula: str,
    current_selfies: str,
    used_element_counts: Dict[str, int]
) -> Dict[str, Any]:
    """
    Get comprehensive state information for the current molecular building state.
    
    Args:
        formula (str): Target molecular formula
        current_selfies (str): Current SELFIES string
        used_element_counts (Dict[str, int]): Current count of used elements
        
    Returns:
        Dict[str, Any]: State information dictionary
    """
    target_counts = parse_formula_counts(formula)
    completion_reward = calculate_formula_completion_reward(formula, used_element_counts)
    formula_match = check_formula_match(formula, used_element_counts)
    
    total_target_atoms = sum(target_counts.values()) if target_counts else 0
    total_used_atoms = sum(used_element_counts.values())
    
    return {
        'target_formula': formula,
        'target_element_counts': target_counts,
        'used_element_counts': used_element_counts,
        'current_selfies': current_selfies,
        'total_target_atoms': total_target_atoms,
        'total_used_atoms': total_used_atoms,
        'completion_reward': completion_reward,
        'formula_match': formula_match,
        'progress_ratio': total_used_atoms / total_target_atoms if total_target_atoms > 0 else 0.0
    }


def get_element_counts_from_selfies(selfies_string: str) -> Dict[str, int]:
    """
    Extract element counts from a SELFIES string by parsing its tokens.
    
    Args:
        selfies_string (str): SELFIES string to analyze
        
    Returns:
        Dict[str, int]: Dictionary mapping element symbols to their counts
        
    Examples:
        >>> get_element_counts_from_selfies("[C][C][H][H][H][H]")
        {'C': 2, 'H': 4}
        >>> get_element_counts_from_selfies("[C][=O][H][H]")
        {'C': 1, 'O': 1, 'H': 2}
        >>> get_element_counts_from_selfies("")
        {}
    """
    if not selfies_string:
        return {}
    
    element_counts = {}
    
    try:
        # Split the SELFIES string into individual tokens
        tokens = sf.split_selfies(selfies_string)
        
        for token in tokens:
            # Extract the base element from each token
            element = extract_element_from_token(token)
            if element:
                element_counts[element] = element_counts.get(element, 0) + 1
    
    except Exception as e:
        print(f"[WARN] Failed to parse SELFIES string '{selfies_string}': {e}")
        return {}
    
    return element_counts


def get_action_mask_from_selfies_string(
    formula: str,
    current_selfies: str,
    actions_list: List[str],
    atom_tokens: List[str],
    bonded_atom_tokens: List[str],
    formula_masking: bool = True,
    end_token: str = "<END>",
    remove_token: str = "<REMOVE>",
    special_tokens: List[str] = None,
    min_formula_completion: float = 0.8,  # Minimum completion ratio before END is allowed
    allow_early_end_after_steps: int = 20  # Allow END after this many steps even if incomplete
) -> np.ndarray:
    """
    Generate a boolean mask over the action space by extracting element counts from SELFIES string.
    
    This is a convenience function that combines get_element_counts_from_selfies with get_action_mask.
    
    Args:
        formula (str): Target molecular formula
        current_selfies (str): Current SELFIES string to extract element counts from
        actions_list (List[str]): List of all possible actions
        atom_tokens (List[str]): List of pure atom tokens
        bonded_atom_tokens (List[str]): List of bonded atom tokens
        formula_masking (bool): Whether to apply formula-based masking
        end_token (str): End token identifier
        remove_token (str): Remove token identifier
        special_tokens (List[str]): List of special tokens to mask out
        min_formula_completion (float): Minimum completion ratio before END token is allowed
        allow_early_end_after_steps (int): Allow END token after this many steps even if incomplete
        
    Returns:
        np.ndarray: Boolean mask array
        
    Examples:
        >>> actions = ['[C]', '[H]', '[O]', '<END>']
        >>> atom_tokens = ['[C]', '[H]', '[O]']
        >>> mask = get_action_mask_from_selfies_string(
        ...     formula="CH4",
        ...     current_selfies="[C][H][H]",
        ...     actions_list=actions,
        ...     atom_tokens=atom_tokens,
        ...     bonded_atom_tokens=[]
        ... )
        >>> # Should allow H (2 more needed) but not C (1 already used) or O (not in formula)
    """
    # Extract element counts from the current SELFIES string
    used_element_counts = get_element_counts_from_selfies(current_selfies)
    
    # Use the existing get_action_mask function with the extracted counts
    return get_action_mask(
        formula=formula,
        used_element_counts=used_element_counts,
        actions_list=actions_list,
        atom_tokens=atom_tokens,
        bonded_atom_tokens=bonded_atom_tokens,
        current_selfies=current_selfies,
        formula_masking=formula_masking,
        end_token=end_token,
        remove_token=remove_token,
        special_tokens=special_tokens,
        min_formula_completion=min_formula_completion,
        allow_early_end_after_steps=allow_early_end_after_steps
    )


def test_intelligent_end_masking():
    """Test the intelligent END token masking functionality."""
    print("Testing intelligent END token masking...")
    
    # Test setup
    actions = ['[C]', '[H]', '[O]', '[N]', '<END>']
    atom_tokens = ['[C]', '[H]', '[O]', '[N]']
    bonded_tokens = []
    
    # Test case 1: Low completion should mask END token
    print("\n1. Testing low completion ratio (should mask END token)")
    used_counts = {'C': 1, 'H': 1}  # Only used 2 out of 8 total atoms in CH4O2
    mask = get_action_mask(
        formula="C2H4O2",  # Acetic acid - 8 total atoms
        used_element_counts=used_counts,
        actions_list=actions,
        atom_tokens=atom_tokens,
        bonded_atom_tokens=bonded_tokens,
        formula_masking=True,
        min_formula_completion=0.8,
        allow_early_end_after_steps=20
    )
    end_idx = actions.index('<END>')
    print(f"Formula: C2H4O2, Used: {used_counts}, END masked: {not mask[end_idx]}")
    assert not mask[end_idx], "END token should be masked with low completion"
    
    # Test case 2: High completion should allow END token
    print("\n2. Testing high completion ratio (should allow END token)")
    used_counts = {'C': 2, 'H': 4, 'O': 1}  # Used 7 out of 8 atoms
    mask = get_action_mask(
        formula="C2H4O2",
        used_element_counts=used_counts,
        actions_list=actions,
        atom_tokens=atom_tokens,
        bonded_atom_tokens=bonded_tokens,
        formula_masking=True,
        min_formula_completion=0.8,
        allow_early_end_after_steps=20
    )
    print(f"Formula: C2H4O2, Used: {used_counts}, END allowed: {mask[end_idx]}")
    assert mask[end_idx], "END token should be allowed with high completion"
    
    # Test case 3: Many steps should allow END token even with low completion
    print("\n3. Testing many steps override (should allow END token)")
    used_counts = {'C': 25}  # More than 20 atoms used
    mask = get_action_mask(
        formula="CH4",
        used_element_counts=used_counts,
        actions_list=actions,
        atom_tokens=atom_tokens,
        bonded_atom_tokens=bonded_tokens,
        formula_masking=True,
        min_formula_completion=0.8,
        allow_early_end_after_steps=20
    )
    print(f"Formula: CH4, Used: {used_counts}, END allowed (many steps): {mask[end_idx]}")
    assert mask[end_idx], "END token should be allowed after many steps"
    
    # Test case 4: Disabled formula masking should always allow END
    print("\n4. Testing disabled formula masking (should allow END token)")
    used_counts = {'C': 1}  # Low completion
    mask = get_action_mask(
        formula="C6H12O6",
        used_element_counts=used_counts,
        actions_list=actions,
        atom_tokens=atom_tokens,
        bonded_atom_tokens=bonded_tokens,
        formula_masking=False,  # Disabled
        min_formula_completion=0.8,
        allow_early_end_after_steps=20
    )
    print(f"Formula masking disabled, Used: {used_counts}, END allowed: {mask[end_idx]}")
    assert mask[end_idx], "END token should be allowed when formula masking is disabled"
    
    print("\n✅ All intelligent END token masking tests passed!")


# Update the main test function to include the new test
def test_utils():
    """Test the utility functions."""
    print("Testing utility functions...")
    
    # Test formula parsing
    assert parse_formula_counts("C6H12O6") == {'C': 6, 'H': 12, 'O': 6}
    assert parse_formula_counts("CH4") == {'C': 1, 'H': 4}
    print("✅ Formula parsing tests passed")
    
    # Test element extraction
    assert extract_element_from_token('[C]') == 'C'
    assert extract_element_from_token('[=N+1]') == 'N'
    assert extract_element_from_token('[Ring1]') is None
    print("✅ Element extraction tests passed")
    
    # Test action masking - use a different formula that doesn't include hydrogen
    # since hydrogen is filtered out by the environment
    actions = ['[C]', '[O]', '[N]', '[S]', '<END>']
    atom_tokens = ['[C]', '[O]', '[N]', '[S]']  # No hydrogen since it's filtered
    bonded_tokens = []
    used_counts = {'C': 1, 'O': 1}  # Used 2 out of 3 atoms in CO2
    
    mask = get_action_mask(
        formula="CO2",  # Carbon dioxide - 3 total atoms
        used_element_counts=used_counts,
        actions_list=actions,
        atom_tokens=atom_tokens,
        bonded_atom_tokens=bonded_tokens,
        min_formula_completion=0.5  # Lower completion threshold for this test
    )
    
    # Carbon should be masked (used 1 out of 1), Oxygen should be available (used 1 out of 2)
    # Nitrogen and Sulfur should be masked (not in formula)
    # END should be available since completion ratio is 2/3 = 0.67 > 0.5
    expected_mask = [False, True, False, False, True]  # [C, O, N, S, END]
    assert np.array_equal(mask, expected_mask), f"Expected {expected_mask}, got {mask.tolist()}"
    print("✅ Action masking tests passed")
    
    # Test SELFIES element counting
    assert get_element_counts_from_selfies("[C][C][H][H][H][H]") == {'C': 2, 'H': 4}
    assert get_element_counts_from_selfies("[C][=O][H][H]") == {'C': 1, 'O': 1, 'H': 2}
    assert get_element_counts_from_selfies("") == {}
    assert get_element_counts_from_selfies("[C][Ring1][H][H][H]") == {'C': 1, 'H': 3}  # Ring1 should be ignored
    print("✅ SELFIES element counting tests passed")
    
    # Test action masking from SELFIES string
    mask_from_selfies = get_action_mask_from_selfies_string(
        formula="CO2",
        current_selfies="[C][=O]",  # Already has 1 C and 1 O
        actions_list=actions,
        atom_tokens=atom_tokens,
        bonded_atom_tokens=bonded_tokens,
        min_formula_completion=0.5  # Same threshold as previous test
    )
    
    # Should be same as previous test since we're using equivalent element counts
    assert np.array_equal(mask_from_selfies, expected_mask), f"Expected {expected_mask}, got {mask_from_selfies.tolist()}"
    print("✅ Action masking from SELFIES tests passed")
    
    # Test intelligent END token masking
    test_intelligent_end_masking()
    
    print("🎉 All utility function tests passed!")


if __name__ == "__main__":
    test_utils() 