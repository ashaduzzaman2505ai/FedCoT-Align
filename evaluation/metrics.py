import torch
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import roc_auc_score
from evaluate import load  # Hugging Face evaluate

class EvaluationMetrics:
    """
    Collection of core metrics: EM/Accuracy, ROUGE, Verifier AUROC.
    """
    def __init__(self):
        self.rouge = load("rouge")
        self.exact_match = load("exact_match")  # or implement manually

    def compute_em_accuracy(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """
        Exact Match and Accuracy.

        Args:
            predictions (List[str]): Model-generated answers.
            references (List[str]): Ground truth.

        Returns:
            Dict[str, float]: {'exact_match': float, 'accuracy': float}
        """
        em_results = self.exact_match.compute(predictions=predictions, references=references)
        accuracy = np.mean([p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)])
        return {"exact_match": em_results["exact_match"], "accuracy": accuracy}

    def compute_rouge(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, float]:
        """
        ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).

        Returns:
            Dict[str, float]: Rouge scores.
        """
        results = self.rouge.compute(predictions=predictions, references=references)
        return {k: v.mid.fmeasure for k, v in results.items()}

    def compute_auroc(
        self, verifier_logits: List[float], true_labels: List[int]
    ) -> float:
        """
        AUROC for hallucination verifier.

        Args:
            verifier_logits (List[float]): Raw logits or probabilities.
            true_labels (List[int]): Binary ground truth (1 = hallucination).

        Returns:
            float: AUROC score.
        """
        if len(set(true_labels)) < 2:
            return 0.5  # Degenerate case
        probs = torch.sigmoid(torch.tensor(verifier_logits)).numpy()
        return roc_auc_score(true_labels, probs)