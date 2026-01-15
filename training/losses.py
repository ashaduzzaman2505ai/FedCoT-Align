import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class FedCoTLosses:
    """
    Tripartite loss for FedCoT-Align:
    L_total = L_answer + λ1 * L_verifier + λ2 * L_cot_alignment
    """
    def __init__(self, lambda_verifier: float = 0.5, lambda_align: float = 0.1):
        self.lambda_v = lambda_verifier
        self.lambda_a = lambda_align
        # Binary Cross Entropy with Logits for Verifier (more stable than Sigmoid + BCE)
        self.verifier_loss_fn = nn.BCEWithLogitsLoss()

    def compute(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], 
                global_prototype: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        # 1. L_answer: Use the T5 native loss if available, otherwise return 0
        l_answer = outputs.get('loss', torch.tensor(0.0, device=global_prototype.device))

        # 2. L_verifier: Binary classification of hallucination
        # Ensure target is float and same shape as logits [Batch, 1]
        target_v = batch['is_hallucination'].float().unsqueeze(-1)
        l_verifier = self.verifier_loss_fn(outputs['verifier_logits'], target_v)

        # 3. L_cot_alignment: MSE to global prototype
        cot_embedding = outputs['cot_embedding'] # [Batch, Embed_Dim]
        # Expand prototype to match batch size
        target_proto = global_prototype.unsqueeze(0).expand(cot_embedding.size(0), -1)
        l_alignment = F.mse_loss(cot_embedding, target_proto)

        # Total Loss
        l_total = l_answer + (self.lambda_v * l_verifier) + (self.lambda_a * l_alignment)

        return {
            'total': l_total,
            'answer': l_answer,
            'verifier': l_verifier,
            'alignment': l_alignment
        }