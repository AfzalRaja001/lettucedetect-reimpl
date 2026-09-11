"""Score a trained checkpoint with the paper's two headline metrics.

Both metrics below work entirely in *token-index* space, not on raw text.
That's a deliberate simplification, not a shortcut: tokens already cut the
answer into contiguous, ordered chunks, so merging consecutive
predicted/gold tokens into "spans" and checking whether two spans overlap
by token index is equivalent to checking whether they overlap by character
index - it just doesn't require re-loading the original answer text at
evaluation time, since the cached labels already are the ground truth in
that space.

Span-level F1 has no official reference implementation - RAGTruth doesn't
ship one, and the LettuceDetect paper says it wrote its own. The matching
rule here (a predicted span counts as correct if it shares at least one
token with some gold span, and symmetrically for recall) is our own
documented choice. Don't expect this number to match the paper's exactly;
do expect it to move in the same direction as real improvements to the
model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from dataset import load_cached_split

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "run" / "best"


def tokens_to_spans(label_seq: list[int], target_label: int = 1) -> list[tuple[int, int]]:
    """Merge consecutive tokens carrying `target_label` into (start, end) index ranges.

    :param label_seq: per-token labels for one example, already stripped of -100
    :param target_label: the label value that marks a "hallucinated" token
    :return: list of (start_idx, end_idx_exclusive) spans, in token-index space
    """
    spans = []
    start = None
    for i, label in enumerate(label_seq):
        if label == target_label:
            if start is None:
                start = i
        elif start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(label_seq)))
    return spans


def spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if the two token-index spans share at least one token."""
    return a[0] < b[1] and b[0] < a[1]


def span_level_f1(gold_seqs: list[list[int]], pred_seqs: list[list[int]]) -> dict:
    """Any-overlap span matching, aggregated over the whole test set.

    :param gold_seqs: per-example, per-token gold 0/1 labels
    :param pred_seqs: per-example, per-token predicted 0/1 labels (same shape)
    :return: dict with span_precision, span_recall, span_f1
    """
    tp = fp = fn = 0
    for gold_seq, pred_seq in zip(gold_seqs, pred_seqs):
        gold_spans = tokens_to_spans(gold_seq)
        pred_spans = tokens_to_spans(pred_seq)

        for p in pred_spans:
            if any(spans_overlap(p, g) for g in gold_spans):
                tp += 1
            else:
                fp += 1
        for g in gold_spans:
            if not any(spans_overlap(g, p) for p in pred_spans):
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"span_precision": precision, "span_recall": recall, "span_f1": f1}


def example_level_f1(
    gold_seqs: list[list[int]],
    pred_seqs: list[list[int]],
    group_by: list[str] | None = None,
) -> dict:
    """Collapse each example to one label (any hallucinated token -> 1) and score it.

    This is what the paper reports as its headline number, and what we're
    comparing against the 76.07 / 79.22 targets.

    :param gold_seqs: per-example, per-token gold 0/1 labels
    :param pred_seqs: per-example, per-token predicted 0/1 labels (same shape)
    :param group_by: optional per-example group key (task_type), to also
        report per-task F1 the way the paper's Table 2 does
    :return: dict with example_precision/recall/f1, plus by_task_type if given
    """
    gold_example = [int(any(label == 1 for label in seq)) for seq in gold_seqs]
    pred_example = [int(any(label == 1 for label in seq)) for seq in pred_seqs]

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_example, pred_example, average="binary", zero_division=0
    )
    result = {"example_precision": precision, "example_recall": recall, "example_f1": f1}

    if group_by is not None:
        result["by_task_type"] = {}
        for task in sorted(set(group_by)):
            idx = [i for i, t in enumerate(group_by) if t == task]
            p, r, f, _ = precision_recall_fscore_support(
                [gold_example[i] for i in idx],
                [pred_example[i] for i in idx],
                average="binary",
                zero_division=0,
            )
            result["by_task_type"][task] = {"precision": p, "recall": r, "f1": f, "n": len(idx)}

    return result


def run(checkpoint_dir: Path = DEFAULT_CHECKPOINT, threshold: float = 0.5) -> dict:
    """Run inference over the cached RAGTruth test split and score it.

    :param checkpoint_dir: a directory saved by train.py's model.save_pretrained()
    :param threshold: probability above which a token counts as hallucinated
    :return: the full metrics dict (also written to results/metrics.json)
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_dir)

    test_dataset = load_cached_split("test")
    task_types = test_dataset["task_type"]
    source_ids = test_dataset["source_id"]

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(REPO_ROOT / "checkpoints" / "_eval_scratch"),
            per_device_eval_batch_size=8,
            report_to=[],
        ),
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )

    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions  # shape: [N, seq_len, 2]
    gold = predictions.label_ids  # shape: [N, seq_len], -100 for ignored positions

    probs_hallucinated = torch.softmax(torch.tensor(logits), dim=-1)[..., 1].numpy()
    preds = (probs_hallucinated >= threshold).astype(int)

    gold_seqs, pred_seqs, prob_seqs = [], [], []
    for gold_row, pred_row, prob_row in zip(gold, preds, probs_hallucinated):
        mask = gold_row != -100
        gold_seqs.append(gold_row[mask].tolist())
        pred_seqs.append(pred_row[mask].tolist())
        prob_seqs.append(prob_row[mask])

    # Dump raw per-token probabilities so every downstream question about
    # this checkpoint (threshold sweeps, calibration/ECE, per-task analysis)
    # can be answered offline on any machine, without re-running the model
    # on a GPU. Only answer-token positions are kept, so this stays small.
    dump_path = REPO_ROOT / "results" / "test_predictions.npz"
    dump_path.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        dump_path,
        probs=np.concatenate(prob_seqs).astype(np.float16),
        golds=np.concatenate([np.array(g, dtype=np.int8) for g in gold_seqs]),
        lengths=np.array([len(g) for g in gold_seqs], dtype=np.int32),
        task_types=np.array(task_types),
        source_ids=np.array([str(s) for s in source_ids]),
    )

    metrics = {}
    metrics.update(example_level_f1(gold_seqs, pred_seqs, group_by=task_types))
    metrics.update(span_level_f1(gold_seqs, pred_seqs))
    metrics["threshold"] = threshold
    metrics["checkpoint"] = str(checkpoint_dir)
    metrics["n_test_examples"] = len(gold_seqs)

    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    final_metrics = run(args.checkpoint, args.threshold)
    print(json.dumps(final_metrics, indent=2))
