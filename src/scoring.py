"""Threshold-free scores computed from per-token hallucination probabilities.

F1 at a fixed 0.5 threshold depends on how well calibrated the model is for
the dataset it is scored on. A model trained on PsiloQA, where about half of
all answer tokens are hallucinated, flags far too much on RAGTruth, where 4%
are. AUROC (the chance that a hallucinated item outscores a clean one) ignores
the threshold, so it separates "ranks badly" from "is miscalibrated".

Only numpy and scikit-learn are needed, so this runs on a machine without a GPU.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """roc_auc_score, or None when only one class is present."""
    if labels.min() == labels.max():
        return None
    return float(roc_auc_score(labels, scores))


def example_scores(prob_seqs: list, gold_seqs: list) -> tuple[np.ndarray, np.ndarray]:
    """Collapse each example to (max token probability, any token hallucinated)."""
    scores = np.array([float(np.max(p)) if len(p) else 0.0 for p in prob_seqs])
    labels = np.array([int(any(g == 1 for g in seq)) for seq in gold_seqs])
    return scores, labels


def example_auroc(prob_seqs: list, gold_seqs: list, group_by: list[str] | None = None) -> dict:
    """AUROC of the example-level score (max token probability).

    :param prob_seqs: per-example arrays of P(hallucinated) over answer tokens
    :param gold_seqs: per-example gold 0/1 labels (same lengths)
    :param group_by: optional per-example task type
    :return: {"example_auroc": float, "auroc_by_task_type": {task: float}}
    """
    scores, labels = example_scores(prob_seqs, gold_seqs)
    result = {"example_auroc": _auroc(scores, labels)}
    if group_by is not None:
        groups = np.asarray(group_by)
        result["auroc_by_task_type"] = {
            str(task): _auroc(scores[groups == task], labels[groups == task])
            for task in sorted(set(group_by))
        }
    return result


def token_auroc(prob_seqs: list, gold_seqs: list) -> float | None:
    """AUROC over every answer token pooled together."""
    scores = np.concatenate([np.asarray(p, dtype=np.float32) for p in prob_seqs])
    labels = np.concatenate([np.asarray(g) for g in gold_seqs])
    return _auroc(scores, (labels == 1).astype(int))
