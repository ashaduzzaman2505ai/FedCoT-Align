import torch
import torch.nn as nn

class VerifierHead(nn.Module):
    """
    Verifier Head for hallucination detection.
    Performs binary classification on the fused hidden and reasoning states.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)