"""Sweep the decision threshold over saved test predictions - no GPU needed.

The trained model outputs a probability per answer token; turning that into
a 0/1 decision needs a cutoff, and the paper simply fixes it at 0.5. There
is no reason 0.5 is optimal: it is optimal only if the costs of a false
positive and a false negative are equal AND the model's probabilities are
well calibrated, neither of which is established here.

Honesty problem this script has to solve: if we pick the threshold that
maximizes F1 on the test set and then report that same F1 as our result,
the number is no longer a held-out estimate - we fit one parameter to the
test set and reported the fit. Retraining with a proper validation split
would solve it, but costs GPU hours.

Since a threshold is applied *after* inference, there is a cheaper fix that
is just as sound: split the *test set itself* into two halves, grouped by
source_id so no source document appears in both halves. Tune the threshold
on half A, report the resulting F1 on half B. Half B was never used to
choose anything, so its number is honest. We report both that and the full
sweep curve (which is legitimate as sensitivity analysis, as long as its
peak is not quoted as "our result").
"""

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import example_level_f1, span_level_f1

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DUMP = REPO_ROOT / "results" / "test_predictions.npz"


def load_predictions(dump_path: Path = DEFAULT_DUMP) -> dict:
    """Load the per-token probability dump written by evaluate.py.

    :param dump_path: path to test_predictions.npz
    :return: dict with per-example lists: probs, golds, plus task_types/source_ids
    """
    data = np.load(dump_path, allow_pickle=False)
    lengths = data["lengths"]
    offsets = np.concatenate([[0], np.cumsum(lengths)])

    probs, golds = [], []
    for i in range(len(lengths)):
        start, end = offsets[i], offsets[i + 1]
        probs.append(data["probs"][start:end].astype(np.float32))
        golds.append(data["golds"][start:end].astype(int).tolist())

    return {
        "probs": probs,
        "golds": golds,
        "task_types": [str(t) for t in data["task_types"]],
        "source_ids": [str(s) for s in data["source_ids"]],
    }


def score_at_threshold(probs, golds, threshold, task_types=None) -> dict:
    """Apply one threshold and compute example-level and span-level metrics.

    :param probs: per-example arrays of P(hallucinated) for each answer token
    :param golds: per-example lists of gold 0/1 labels
    :param threshold: cutoff above which a token is called hallucinated
    :param task_types: optional per-example task labels for a breakdown
    :return: metrics dict
    """
    preds = [(p >= threshold).astype(int).tolist() for p in probs]
    metrics = example_level_f1(golds, preds, group_by=task_types)
    metrics.update(span_level_f1(golds, preds))
    metrics["threshold"] = round(float(threshold), 3)
    return metrics


def split_by_source(source_ids: list[str], fraction: float = 0.5, seed: int = 42):
    """Split example indices into two halves with no shared source document.

    Grouping matters: RAGTruth has ~6 model responses per source item, all
    sharing one context. A random per-row split would put the same context
    in both halves, so a threshold tuned on one half would be partly tuned
    on the other - exactly the leak this split exists to avoid.

    :param source_ids: per-example source document id
    :param fraction: portion of source items assigned to the tuning half
    :param seed: RNG seed, so the split is reproducible across runs
    :return: (tune_indices, report_indices)
    """
    unique = sorted(set(source_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    tune_sources = set(unique[: int(len(unique) * fraction)])

    tune_idx = [i for i, s in enumerate(source_ids) if s in tune_sources]
    report_idx = [i for i, s in enumerate(source_ids) if s not in tune_sources]
    return tune_idx, report_idx


def subset(data: dict, indices: list[int]) -> tuple:
    """Select a subset of examples by index."""
    return (
        [data["probs"][i] for i in indices],
        [data["golds"][i] for i in indices],
        [data["task_types"][i] for i in indices],
    )


def main(dump_path: Path = DEFAULT_DUMP, step: float = 0.05) -> dict:
    data = load_predictions(dump_path)
    thresholds = np.arange(0.05, 0.96, step)

    # 1. Full-test sweep - reportable as a sensitivity curve, NOT as a result.
    sweep = [
        score_at_threshold(data["probs"], data["golds"], t, data["task_types"])
        for t in thresholds
    ]
    print(f"{'thresh':>7} {'ex_P':>7} {'ex_R':>7} {'ex_F1':>7} {'span_F1':>8}")
    for row in sweep:
        print(
            f"{row['threshold']:>7.2f} {row['example_precision']:>7.3f} "
            f"{row['example_recall']:>7.3f} {row['example_f1']:>7.3f} "
            f"{row['span_f1']:>8.3f}"
        )

    baseline = score_at_threshold(data["probs"], data["golds"], 0.5, data["task_types"])

    # 2. Honest protocol: tune on half A (grouped by source), report on half B.
    tune_idx, report_idx = split_by_source(data["source_ids"])
    tune_probs, tune_golds, tune_tasks = subset(data, tune_idx)
    rep_probs, rep_golds, rep_tasks = subset(data, report_idx)

    tune_scores = [score_at_threshold(tune_probs, tune_golds, t) for t in thresholds]
    best_on_tune = max(tune_scores, key=lambda r: r["example_f1"])
    chosen = best_on_tune["threshold"]

    held_out_tuned = score_at_threshold(rep_probs, rep_golds, chosen, rep_tasks)
    held_out_default = score_at_threshold(rep_probs, rep_golds, 0.5, rep_tasks)

    result = {
        "protocol": (
            "Threshold chosen by maximizing example-level F1 on a source-grouped "
            "half of the RAGTruth test set; reported score is on the untouched "
            "other half, which was never used for any selection."
        ),
        "chosen_threshold": chosen,
        "n_tune_examples": len(tune_idx),
        "n_report_examples": len(report_idx),
        "held_out_at_default_0.5": held_out_default,
        "held_out_at_tuned_threshold": held_out_tuned,
        "full_test_at_0.5_for_reference": baseline,
        "sweep_full_test": sweep,
    }

    out_path = REPO_ROOT / "results" / "threshold_tuning.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nChosen threshold (tuned on held-in half): {chosen}")
    print(f"Held-out half @ 0.5    : example F1 = {held_out_default['example_f1']:.4f}")
    print(f"Held-out half @ {chosen:.2f}   : example F1 = {held_out_tuned['example_f1']:.4f}")
    print(f"\nwrote {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    main(args.dump, args.step)
