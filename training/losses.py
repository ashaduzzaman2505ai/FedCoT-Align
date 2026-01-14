import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, torch.Tensor

class FedCoTLosses:
    """
    Tripartite loss for FedCoT-Align.

    L_total = L_answer + λ1 * L_verifier + λ2 * L_cot_alignment

    Args:
        lambda1 (float): Weight for verifier loss.
        lambda2 (float): Weight for alignment loss.
    """
    def __init__(self, lambda1: float = 1.0, lambda2: float = 0.5):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.answer_loss = nn.CrossEntropyLoss(ignore_index=-100)  # For seq2seq
        self.verifier_loss = nn.BCEWithLogitsLoss()  # Binary for hallucination

    def compute(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor, hallucination_labels: torch.Tensor,
                global_prototype: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.

        Args:
            outputs (Dict[str, Tensor]): Model forward outputs.
            labels (Tensor): Answer labels.
            hallucination_labels (Tensor): Hallucination labels (binary).
            global_prototype (Tensor): Global CoT prototype for alignment.

        Returns:
            Dict[str, Tensor]: Losses dict with 'total', 'answer', 'verifier', 'alignment'.
        """
        # L_answer: Shift labels for seq2seq
        shift_logits = outputs['answer_logits'][:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        l_answer = self.answer_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # L_verifier
        verifier_logits = outputs['verifier_logits'].squeeze(-1) if hallucination_labels.dim() == 1 else outputs['verifier_logits']
        l_verifier = self.verifier_loss(verifier_logits, hallucination_labels.float())

        # L_cot_alignment: MSE to global prototype
        cot_embedding = outputs['cot_embedding']
        l_alignment = F.mse_loss(cot_embedding, global_prototype.expand_as(cot_embedding))

        # Total
        l_total = l_answer + self.lambda1 * l_verifier + self.lambda2 * l_alignment

        return {
            'total': l_total,
            'answer': l_answer,
            'verifier': l_verifier,
            'alignment': l_alignment
        }