import difflib
import torch.nn as nn

def get_norm_layer(name):
    if not isinstance(name, str):
        return name

    # 1. Get all candidates in torch.nn (filtering for things likely to be Norm layers)
    # We look for any class that contains 'Norm' (LayerNorm, BatchNorm, etc.)
    candidates = [attr for attr in dir(nn) if "Norm" in attr]
    
    # 2. Try an exact case-insensitive match first (faster/more accurate)
    target = name.lower()
    for attr in candidates:
        if attr.lower() == target:
            return getattr(nn, attr)
    
    # 3. Use string distance to find the closest match
    # cutoff=0.1 allows for very loose matches (like "ln" to "LayerNorm")
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.1)
    
    if matches:
        match = matches[0]
        return getattr(nn, match)
            
    raise ValueError(f"Could not find a norm layer in torch.nn similar to '{name}'")