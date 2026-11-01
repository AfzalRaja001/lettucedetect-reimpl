"""Build one comparison table from every saved prediction dump in results/.

Needs no GPU and no checkpoints: evaluate.py stores the raw per-token
probabilities, so every number here is recomputed offline. Run it after any
evaluation to refresh results/summary.json and the printed table.

Columns:
  F1@0.5       example-level F1 at the default threshold
  flag-all F1  the F1 you would get by calling every example hallucinated.
               A score near this number means the model adds nothing.
  ex AUROC     threshold-free example-level ranking quality (0.5 = chance)
  tok F1@0.5   token-level F1 at the default threshold
  tok AUROC    threshold-free token-level ranking quality
"""

import json
from pathlib import Path

import numpy as np

from scoring import example_auroc, token_auroc

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
THRESHOLD = 0.5

# The first RAGTruth run predates the tag convention and kept the bare names.
LEGACY_NAMES = {"test_predictions": "ragtruth_baseline_on_ragtruth"}


def f1_percent(pred: np.ndarray, gold: np.ndarray) -> float:
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    return 100.0 * 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def load_dump(path: Path) -> dict:
    """Split the flat arrays in an evaluate.py dump back into per-example lists."""
    data = np.load(path, allow_pickle=True)
    offsets = np.concatenate([[0], np.cumsum(data["lengths"])])
    n = len(data["lengths"])
    return {
        "probs": [data["probs"][offsets[i] : offsets[i + 1]].astype(np.float32) for i in range(n)],
        "golds": [data["golds"][offsets[i] : offsets[i + 1]] for i in range(n)],
        "task_types": [str(t) for t in data["task_types"]],
    }


def summarize(path: Path) -> dict:
    dump = load_dump(path)
    probs, golds, tasks = dump["probs"], dump["golds"], dump["task_types"]

    max_prob = np.array([float(p.max()) if len(p) else 0.0 for p in probs])
    gold_example = np.array([bool((g == 1).any()) for g in golds])
    pred_example = max_prob >= THRESHOLD

    flat_probs = np.concatenate(probs)
    flat_gold = np.concatenate(golds) == 1
    flat_pred = flat_probs >= THRESHOLD

    auroc = example_auroc(probs, golds, group_by=tasks)
    return {
        "dataset": "psiloqa" if "PsiloQA" in set(tasks) else "ragtruth",
        "n": len(probs),
        "example_positive_rate": float(gold_example.mean()),
        "example_f1": f1_percent(pred_example, gold_example),
        "flag_all_f1": f1_percent(np.ones_like(gold_example), gold_example),
        "share_flagged": float(pred_example.mean()),
        "example_auroc": auroc["example_auroc"],
        "auroc_by_task_type": auroc["auroc_by_task_type"],
        "token_f1": f1_percent(flat_pred, flat_gold),
        "token_auroc": token_auroc(probs, golds),
    }


def run_name(path: Path) -> str:
    stem = path.stem
    if stem in LEGACY_NAMES:
        return LEGACY_NAMES[stem]
    return stem.removesuffix("_predictions")


def fmt(value, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def main() -> dict:
    summary = {}
    for path in sorted(RESULTS_DIR.glob("*_predictions.npz")):
        summary[run_name(path)] = summarize(path)

    with open(RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    header = f"{'run':<34}{'test set':<9}{'F1@0.5':>8}{'flag-all':>9}{'flagged':>9}{'ex AUROC':>10}{'tok F1':>8}{'tok AUROC':>10}"
    print(header)
    print("-" * len(header))
    for name, s in summary.items():
        print(
            f"{name:<34}{s['dataset']:<9}{s['example_f1']:>8.1f}{s['flag_all_f1']:>9.1f}"
            f"{100 * s['share_flagged']:>8.0f}%{fmt(s['example_auroc'], '>10.3f')}"
            f"{s['token_f1']:>8.1f}{fmt(s['token_auroc'], '>10.3f')}"
        )

    print("\nExample AUROC by RAGTruth task type")
    for name, s in summary.items():
        if s["dataset"] == "ragtruth":
            per_task = "  ".join(f"{t}={fmt(v, '.3f')}" for t, v in s["auroc_by_task_type"].items())
            print(f"  {name:<34}{per_task}")
    return summary


if __name__ == "__main__":
    main()
