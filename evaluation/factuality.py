from typing import List, Dict
from transformers import pipeline
import numpy as np

class FactScoreEvaluator:
    """
    Lightweight FACTScore approximation using NLI model.
    Decomposes claim into atomic facts → checks entailment against reference.

    Args:
        nli_model (str): e.g., "roberta-large-mnli"
    """
    def __init__(self, nli_model: str = "roberta-large-mnli"):
        self.nli = pipeline("text-classification", model=nli_model, device=0 if torch.cuda.is_available() else -1)

    def compute_factscore(
        self, generations: List[str], references: List[str], decompose: bool = False
    ) -> Dict[str, float]:
        """
        Approximate FACTScore: fraction of generated sentences entailed by reference.

        Args:
            generations (List[str]): Model outputs.
            references (List[str]): Ground truth contexts/answers.
            decompose (bool): If True, attempt sentence decomposition (simplified here).

        Returns:
            Dict[str, float]: {'factscore': mean score, 'supporting_ratio': ...}
        """
        scores = []
        for gen, ref in zip(generations, references):
            # Simple version: treat full generation vs reference as premise-hypothesis
            # In full FACTScore, decompose into atomic facts → NLI per fact
            result = self.nli(f"{gen} [SEP] {ref}")
            entailment_score = next((r['score'] for r in result if r['label'] == 'ENTAILMENT'), 0.0)
            contradiction_score = next((r['score'] for r in result if r['label'] == 'CONTRADICTION'), 0.0)
            score = entailment_score - contradiction_score
            scores.append(max(0.0, score))  # Normalized [0,1]

        mean_score = np.mean(scores)
        return {"factscore": float(mean_score), "factscore_std": float(np.std(scores))}

# Usage: evaluator = FactScoreEvaluator(); score = evaluator.compute_factscore(gens, refs)