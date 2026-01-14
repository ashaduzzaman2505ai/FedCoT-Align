import os
import torch
from typing import Dict, Any, Optional

def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    round_idx: int, 
    path: str, 
    filename: str = "latest.pth"
) -> None:
    """Saves full state for training resumption."""
    os.makedirs(path, exist_ok=True)
    state = {
        'round': round_idx,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'np_rng_state': np.random.get_state(),
        'random_rng_state': random.getstate(),
    }
    save_path = os.path.join(path, filename)
    torch.save(state, save_path)
    # Metadata for logs
    print(f"Saved checkpoint: {save_path}")

def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None) -> int:
    """Loads state and restores RNG for deterministic resumption."""
    if not os.path.exists(path):
        return 0
        
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore RNG
    torch.set_rng_state(checkpoint['rng_state'])
    if checkpoint['cuda_rng_state'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    np.random.set_state(checkpoint['np_rng_state'])
    random.setstate(checkpoint['random_rng_state'])
    
    return checkpoint['round']