"""
Runs the impact-reasoning pipeline against a hand-labeled ground-truth set
and reports precision/recall/F1 - this is what lets you say "my matching
engine achieves X% precision" in an interview instead of "it seems to work".

Usage:
    python -m eval.evaluate
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.reasoning.impact_reasoner import assess_impact


def load_eval_set(path="eval/eval_set.json"):
    with open(path) as f:
        return json.load(f)


def run_evaluation():
    from app.cache import clear_cache
    clear_cache()  # ensure every case is freshly assessed, not served from a prior run's cache

    cases = load_eval_set()
    tp = fp = tn = fn = 0
    results = []

    for i, case in enumerate(cases):
        result = assess_impact(
            clause_text=case["clause_text"],
            policy_id=case["policy_name"],
            prompt_version="eval",
            policy_name=case["policy_name"],
            policy_description=case["policy_name"],  # using name as description for eval simplicity
        )
        predicted = result["impacts_policy"]
        expected = case["expected_impact"]

        if predicted and expected:
            tp += 1
        elif predicted and not expected:
            fp += 1
        elif not predicted and not expected:
            tn += 1
        else:
            fn += 1

        results.append({
            "clause": case["clause_text"][:80] + "...",
            "policy": case["policy_name"],
            "expected": expected,
            "predicted": predicted,
            "correct": predicted == expected,
            "note": case["note"],
        })

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / len(cases) if cases else 0

    report = {
        "total_cases": len(cases),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
        "results": results,
    }

    with open("eval/eval_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== RegWatch Evaluation Report ===")
    print(f"Cases: {report['total_cases']}  |  Precision: {report['precision']}  Recall: {report['recall']}  F1: {report['f1']}  Accuracy: {report['accuracy']}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}\n")
    for r in results:
        status = "PASS" if r["correct"] else "FAIL"
        print(f"[{status}] expected={r['expected']} predicted={r['predicted']} | {r['policy']} | {r['clause']}")
    print(f"\nFull report saved to eval/eval_report.json")
    return report


if __name__ == "__main__":
    run_evaluation()
