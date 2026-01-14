import os
import torch
from typing import Dict, Any

def save_checkpoint(state: Dict[str, Any], path: str, filename: str = "checkpoint.pth") -> None:
    """
    Save model checkpoint.

    Args:
        state (Dict[str, Any]): State dict containing model, optimizer, etc.
        path (str): Directory to save in.
        filename (str): Checkpoint filename.
    """
    os.makedirs(path, exist_ok=True)
    torch.save(state, os.path.join(path, filename))
    print(f"Checkpoint saved to {os.path.join(path, filename)}")

def load_checkpoint(path: str, filename: str = "checkpoint.pth") -> Dict[str, Any]:
    """
    Load model checkpoint.

    Args:
        path (str): Directory of checkpoint.
        filename (str): Checkpoint filename.

    Returns:
        Dict[str, Any]: Loaded state dict.
    """
    checkpoint_path = os.path.join(path, filename)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path)
        print(f"Checkpoint loaded from {checkpoint_path}")
        return state
    else:
        print(f"No checkpoint found at {checkpoint_path}")
        return {}