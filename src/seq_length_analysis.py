"""How much could the 2048-token cap have cost us versus the paper's 4096?

Retraining at 4096 tokens does not fit on an 8GB GPU, so this answers the
question from data we already have: re-tokenize RAGTruth with no cap, count
how many examples 2048 actually truncates, and compare how the trained model
scores on truncated versus untruncated test examples.
"""

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from data_prep import TOKENIZER_NAME, load_raw_ragtruth
from tune_threshold import load_predictions

REPO_ROOT = Path(__file__).resolve().parent.parent
CAPS = (2048, 4096)
THRESHOLD = 0.5


def untruncated_lengths(samples: list[dict], tokenizer) -> np.ndarray:
    """Token length of [CLS] context [SEP] answer [SEP] with no truncation."""
    lengths = []
    batch = 512
    for start in range(0, len(samples), batch):
        chunk = samples[start : start + batch]
        enc = tokenizer(
            [s["context"] for s in chunk],
            [s["answer"] for s in chunk],
            truncation=False,
        )
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return np.array(lengths)


def f1_from_flags(pred: np.ndarray, gold: np.ndarray) -> float:
    tp = int((pred & gold).sum())
    fp = int((pred & ~gold).sum())
    fn = int((~pred & gold).sum())
    denom = 2 * tp + fp + fn
    return 100.0 * 2 * tp / denom if denom else 0.0


def main() -> dict:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    samples = load_raw_ragtruth()
    lengths = untruncated_lengths(samples, tokenizer)
    tasks = np.array([s["task_type"] for s in samples])
    splits = np.array([s["split"] for s in samples])

    result = {"length_stats": {}, "test_scores": {}}

    print("Untruncated sequence length (tokens)")
    print(f"{'split/task':<18}{'n':>6}{'mean':>8}{'p95':>8}{'max':>8}" + "".join(f"{'>' + str(c):>9}" for c in CAPS))
    for split in ("train", "test"):
        for task in ["ALL"] + sorted(set(tasks)):
            mask = (splits == split) & ((tasks == task) | (task == "ALL"))
            L = lengths[mask]
            row = {"n": int(mask.sum()), "mean": float(L.mean()), "p95": float(np.percentile(L, 95)), "max": int(L.max())}
            for c in CAPS:
                row[f"pct_over_{c}"] = float(100 * (L > c).mean())
            result["length_stats"][f"{split}/{task}"] = row
            print(
                f"{split + '/' + task:<18}{row['n']:>6}{row['mean']:>8.0f}{row['p95']:>8.0f}{row['max']:>8}"
                + "".join(f"{row[f'pct_over_{c}']:>8.1f}%" for c in CAPS)
            )

    # Align the saved test predictions with the raw test examples (same order).
    test_idx = np.where(splits == "test")[0]
    preds = load_predictions()
    assert len(test_idx) == len(preds["probs"]), "prediction dump does not match test split size"
    assert [samples[i]["source_id"] for i in test_idx] == preds["source_ids"], "order mismatch"

    max_prob = np.array([p.max() if len(p) else 0.0 for p in preds["probs"]])
    gold = np.array([any(g == 1 for g in gl) for gl in preds["golds"]])
    pred = max_prob >= THRESHOLD
    test_len = lengths[test_idx]
    test_tasks = tasks[test_idx]
    truncated = test_len > CAPS[0]

    print(f"\nTest examples truncated at {CAPS[0]}: {int(truncated.sum())} of {len(truncated)} ({100 * truncated.mean():.1f}%)")
    print(f"\n{'subset':<28}{'n':>6}{'gold+':>7}{'F1@0.5':>9}")
    rows = [("all", np.ones_like(truncated)), ("not truncated", ~truncated), ("truncated", truncated)]
    for task in sorted(set(test_tasks)):
        rows.append((f"{task} (all)", test_tasks == task))
        rows.append((f"{task} truncated", (test_tasks == task) & truncated))
    for name, mask in rows:
        n = int(mask.sum())
        f1 = f1_from_flags(pred[mask], gold[mask]) if n else float("nan")
        result["test_scores"][name] = {"n": n, "gold_positive": int(gold[mask].sum()), "f1": f1}
        print(f"{name:<28}{n:>6}{int(gold[mask].sum()):>7}{f1:>9.1f}")

    # Upper bound: score every truncated example perfectly and see how far overall F1 moves.
    oracle = pred.copy()
    oracle[truncated] = gold[truncated]
    result["overall_f1"] = f1_from_flags(pred, gold)
    result["upper_bound_f1_if_truncated_perfect"] = f1_from_flags(oracle, gold)
    print(f"\nOverall F1 as trained        : {result['overall_f1']:.2f}")
    print(f"Upper bound if truncated were perfect: {result['upper_bound_f1_if_truncated_perfect']:.2f}")

    out_path = REPO_ROOT / "results" / "seq_length_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")
    return result


if __name__ == "__main__":
    main()
