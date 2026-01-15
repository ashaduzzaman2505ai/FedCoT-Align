import torch
from typing import List, Dict
from transformers import pipeline
import numpy as np
import re

class FactScoreEvaluator:
    """
    Refined FACTScore proxy using sentence-level NLI decomposition.
    """
    def __init__(self, nli_model: str = "roberta-large-mnli"):
        device = 0 if torch.cuda.is_available() else -1
        self.nli = pipeline("text-classification", model=nli_model, device=device)

    def _split_sentences(self, text: str) -> List[str]:
        # Basic regex split for sentences
        sentences = re.split(r'(?<=[.!?]) +', text)
        return [s.strip() for s in sentences if len(s.strip()) > 5]

    def compute_factscore(self, generations: List[str], references: List[str]) -> Dict[str, float]:
        all_instance_scores = []
        
        for gen, ref in zip(generations, references):
            sentences = self._split_sentences(gen)
            if not sentences:
                all_instance_scores.append(0.0)
                continue
            
            # Check entailment for each sentence (atomic fact) against the reference
            sent_scores = []
            for sent in sentences:
                # NLI expects format: premise [SEP] hypothesis (context [SEP] claim)
                # Pipeline handle: {"text": premise, "text_pair": hypothesis}
                result = self.nli({"text": ref, "text_pair": sent})
                
                # Roberta MNLI labels: 0: contradiction, 1: neutral, 2: entailment
                # We normalize: Entailment = 1, others = 0
                label = result['label'].upper()
                score = 1.0 if "ENTAIL" in label else 0.0
                sent_scores.append(score)
            
            all_instance_scores.append(np.mean(sent_scores))
            
        return {
            "factscore": float(np.mean(all_instance_scores)),
            "factscore_std": float(np.std(all_instance_scores))
        }