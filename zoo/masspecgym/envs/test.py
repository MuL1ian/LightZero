import selfies as sf

def split_atoms(atom_tokens):
    """
    Split a list of SELFIES tokens into different categories.
    
    Args:
        atom_tokens (list): List of SELFIES tokens.
        
    Returns:
        tuple: Four lists containing pure atoms, bonded atoms, branch tokens, and ring tokens.
    """
    pure_atom_tokens = []
    bonded_atom_tokens = []
    branch_tokens = []
    ring_tokens = []

    default_branch = {"[Branch1]", "[Branch2]", "[Branch3]"}
    default_ring = {"[Ring1]", "[Ring2]", "[Ring3]"}

    for token in atom_tokens:
        if token in default_branch:
            branch_tokens.append(token)
        elif token in default_ring:
            ring_tokens.append(token)
        elif token.startswith("[=") or token.startswith("[#"):
            if "Branch" in token or "Ring" in token:
                continue
            else:
                bonded_atom_tokens.append(token)
        else:
            pure_atom_tokens.append(token)
    return pure_atom_tokens, bonded_atom_tokens, branch_tokens, ring_tokens

all_atom_tokens = sf.get_semantic_robust_alphabet()
#{'[Branch2]', '[=P]', '[=S]', '[#P-1]', '[=B-1]', '[#N+1]', '[N+1]', '[=N-1]', '[=Ring3]', '[#O+1]', '[#C+1]', '[=N+1]', '[O+1]', '[P-1]', '[=S+1]', '[=P+1]', '[O-1]', '[#C-1]', '[#B-1]', '[=S-1]', '[H]}

atom_tokens, bonded_atom_tokens, branch_tokens, ring_tokens = split_atoms(all_atom_tokens)

valid_atoms = atom_tokens
print(valid_atoms)