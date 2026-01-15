import torch
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import roc_auc_score
from evaluate import load

class EvaluationMetrics:
    def __init__(self):
        self.rouge = load("rouge")
        self.exact_match = load("exact_match")

    def compute_em_accuracy(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        # Filter out empty strings to avoid crashes
        if not predictions or not references:
            return {"exact_match": 0.0, "accuracy": 0.0}
            
        em_results = self.exact_match.compute(predictions=predictions, references=references)
        accuracy = np.mean([
            p.strip().lower() == r.strip().lower() for p, r in zip(predictions, references)
        ])
        return {"exact_match": em_results["exact_match"], "accuracy": float(accuracy)}

    def compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        # The 'evaluate' library returns a dict of floats directly in newer versions
        results = self.rouge.compute(predictions=predictions, references=references)
        return {k: float(v) for k, v in results.items()}

    def compute_auroc(self, verifier_logits: List[float], true_labels: List[int]) -> float:
        if len(set(true_labels)) < 2:
            return 0.5  # Neutral score for single-class batches
        
        # Ensure we are dealing with probabilities
        logits_tensor = torch.tensor(verifier_logits)
        probs = torch.sigmoid(logits_tensor).numpy()
        return float(roc_auc_score(true_labels, probs))