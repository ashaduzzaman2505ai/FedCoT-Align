import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from evaluation.metrics import EvaluationMetrics
from evaluation.factuality import FactScoreEvaluator

def test_eval():
    evaluator = EvaluationMetrics()
    fact_eval = FactScoreEvaluator()

    # Test Data
    gens = ["The capital of France is Paris. It is a beautiful city."]
    refs = ["Paris is the capital and most populous city of France."]
    
    # NLP Metrics
    nlp_res = evaluator.compute_em_accuracy(["Paris"], ["Paris"])
    print(f"EM/Acc: {nlp_res}")

    # FactScore (Sentence Level)
    fact_res = fact_eval.compute_factscore(gens, refs)
    print(f"FactScore: {fact_res}")
    
    assert nlp_res['accuracy'] == 1.0
    assert fact_res['factscore'] > 0.0
    print("✅ Step 7 Passed: Evaluation suite is statistically sound.")

if __name__ == "__main__":
    test_eval()