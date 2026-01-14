import random
import os
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Sets all seeds and ensures deterministic behavior for ICML reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Newer torch versions requirement
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError:
        # Fallback for operations that don't support determinism
        print("Warning: Some operations may not support deterministic mode.")
        
    print(f"Global seed set to: {seed}")