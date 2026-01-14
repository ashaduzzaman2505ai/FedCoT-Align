import torch
from typing import List, Dict
import numpy as np

class SelfConsistencyEvaluator:
    """
    Self-consistency variance: Generate multiple reasoning paths → variance in final answers.
    Lower variance → better consistency (less hallucination-prone).
    """
    def compute_consistency_variance(
        self,
        multiple_predictions: List[List[str]],  # [num_samples, num_paths]
        reference_answers: List[str]
    ) -> Dict[str, float]:
        """
        Compute variance of final answers across multiple CoT samples.

        Args:
            multiple_predictions (List[List[str]]): Multiple generations per instance.
            reference_answers (List[str]): Ground truth (for optional accuracy).

        Returns:
            Dict[str, float]: Variance metrics.
        """
        variances = []
        accuracies = []

        for preds, ref in zip(multiple_predictions, reference_answers):
            # Simple string-based variance (can use edit distance or semantic later)
            cleaned_preds = [p.strip().lower() for p in preds]
            unique_answers = set(cleaned_preds)
            if len(unique_answers) == 0:
                continue
            # Variance proxy: 1 - (max frequency / total)
            freqs = [cleaned_preds.count(ans) for ans in unique_answers]
            variance_proxy = 1.0 - (max(freqs) / len(cleaned_preds))
            variances.append(variance_proxy)

            # Optional: majority vote accuracy
            majority = max(unique_answers, key=cleaned_preds.count)
            accuracies.append(majority == ref.strip().lower())

        return {
            "consistency_variance": float(np.mean(variances)) if variances else 0.0,
            "self_consistency_accuracy": float(np.mean(accuracies)) if accuracies else 0.0
        }