import torch.nn as nn

class VerifierHead(nn.Module):
    """
    Verifier Head for hallucination detection.

    Args:
        input_dim (int): Input dimension (hidden + cot).
        hidden_dim (int): Hidden layer dim.
        num_classes (int): 2 for binary.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Logits for hallucination classification.
        """
        x = self.relu(self.fc1(x))
        return self.fc2(x)