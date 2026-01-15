import torch
import torch.nn as nn

class CoTProjector(nn.Module):
    """
    Projector for CoT embeddings to a shared latent space for FL.
    Uses LayerNorm to stabilize the alignment during federated averaging.
    Args:
        input_dim (int): CoT state dim.
        embed_dim (int): Projected embed dim.
    """
    
    def __init__(self, input_dim: int, embed_dim: int):
    """
    Forward pass.

    Args:
        x (torch.Tensor): CoT states.

    Returns:
        torch.Tensor: Projected embeddings.
    """
        super().__init__()
        self.projection = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(x))
