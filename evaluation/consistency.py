import numpy as np
from typing import List, Dict

class SelfConsistencyEvaluator:
    """
    Measures 'Reasoning Convergence' across multiple CoT paths.
    """
    def compute_consistency_variance(
        self,
        multiple_predictions: List[List[str]], # Shape: [num_test_samples, num_cot_paths]
        reference_answers: List[str]
    ) -> Dict[str, float]:
        variances = []
        accuracies = []

        for paths, ref in zip(multiple_predictions, reference_answers):
            if not paths: continue
            
            cleaned_paths = [p.strip().lower() for p in paths]
            total_paths = len(cleaned_paths)
            
            # Count occurrences of each unique answer
            counts = {}
            for p in cleaned_paths:
                counts[p] = counts.get(p, 0) + 1
            
            # Variance Proxy: 1 - (frequency of most common answer / total paths)
            max_freq = max(counts.values())
            variance = 1.0 - (max_freq / total_paths)
            variances.append(variance)
            
            # Majority Vote Accuracy
            majority_answer = max(counts, key=counts.get)
            accuracies.append(1.0 if majority_answer == ref.strip().lower() else 0.0)

        return {
            "consistency_variance": float(np.mean(variances)),
            "self_consistency_accuracy": float(np.mean(accuracies))
        }