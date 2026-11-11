"""Paired grouped bootstrap: is model B really better than model A?

Both models are scored on the same test examples, so the comparison is paired:
every bootstrap replicate resamples the *same* documents for both models and
records the difference B - A. Resampling is by source_id (the document an
answer was written about), not by example, because several answers share a
source and are not independent of each other.

Reads the raw prediction dumps written by evaluate.py, so it needs no GPU.

Usage:
    python src/bootstrap_compare.py --a ragtruth_val --b modernbert-large
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from evaluate import spans_overlap, tokens_to_spans

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
THRESHOLD = 0.5
METRICS = ["token_f1", "span_f1", "example_f1", "example_auroc", "token_auroc"]


def load(tag: str, dataset: str) -> dict:
    """One prediction dump, split into per-example pieces."""
    data = np.load(RESULTS / f"{tag}_on_{dataset}_predictions.npz", allow_pickle=True)
    lengths = data["lengths"]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    probs = data["probs"].astype(np.float32)
    golds = data["golds"].astype(np.int8)
    return {
        "probs": probs,
        "golds": golds,
        "offsets": offsets,
        "lengths": lengths,
        "source_ids": data["source_ids"],
    }


def per_example_counts(d: dict) -> dict:
    """Sufficient statistics per example, so each replicate is just a sum."""
    n = len(d["lengths"])
    tok = np.zeros((n, 3))  # tp, fp, fn at the threshold
    spn = np.zeros((n, 3))
    ex_gold = np.zeros(n, dtype=bool)
    ex_pred = np.zeros(n, dtype=bool)
    ex_score = np.zeros(n)
    for i in range(n):
        lo, hi = d["offsets"][i], d["offsets"][i + 1]
        p, g = d["probs"][lo:hi], d["golds"][lo:hi] == 1
        pred = p >= THRESHOLD
        tok[i] = [(g & pred).sum(), (~g & pred).sum(), (g & ~pred).sum()]
        gs, ps = tokens_to_spans(g.astype(int).tolist()), tokens_to_spans(pred.astype(int).tolist())
        tp = sum(any(spans_overlap(x, y) for y in gs) for x in ps)
        fn = sum(not any(spans_overlap(y, x) for x in ps) for y in gs)
        spn[i] = [tp, len(ps) - tp, fn]
        ex_gold[i], ex_pred[i] = g.any(), pred.any()
        ex_score[i] = p.max() if len(p) else 0.0
    return {"tok": tok, "spn": spn, "ex_gold": ex_gold, "ex_pred": ex_pred, "ex_score": ex_score}


def f1(tp: float, fp: float, fn: float) -> float:
    return 100 * 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if labels.min() != labels.max() else float("nan")


def metrics_for(d: dict, c: dict, idx: np.ndarray) -> dict:
    """All headline metrics for the examples listed in idx (repeats allowed)."""
    tok, spn = c["tok"][idx].sum(axis=0), c["spn"][idx].sum(axis=0)
    gold, pred = c["ex_gold"][idx], c["ex_pred"][idx]
    token_slices = [slice(d["offsets"][i], d["offsets"][i + 1]) for i in idx]
    token_labels = np.concatenate([d["golds"][s] for s in token_slices]) == 1
    token_scores = np.concatenate([d["probs"][s] for s in token_slices])
    return {
        "token_f1": f1(tok[0], tok[1], tok[2]),
        "span_f1": f1(spn[0], spn[1], spn[2]),
        "example_f1": f1((gold & pred).sum(), (~gold & pred).sum(), (gold & ~pred).sum()),
        "example_auroc": safe_auc(gold.astype(int), c["ex_score"][idx]),
        "token_auroc": safe_auc(token_labels.astype(int), token_scores),
    }


def compare(tag_a: str, tag_b: str, dataset: str, reps: int, seed: int) -> dict:
    a, b = load(tag_a, dataset), load(tag_b, dataset)
    assert np.array_equal(a["lengths"], b["lengths"]) and np.array_equal(a["golds"], b["golds"]), (
        "the two dumps are not over the same test examples"
    )
    ca, cb = per_example_counts(a), per_example_counts(b)

    sources = a["source_ids"]
    unique = np.unique(sources)
    members = {s: np.flatnonzero(sources == s) for s in unique}

    everyone = np.arange(len(sources))
    point = {m: (metrics_for(a, ca, everyone)[m], metrics_for(b, cb, everyone)[m]) for m in METRICS}

    rng = np.random.default_rng(seed)
    deltas = {m: [] for m in METRICS}
    for _ in range(reps):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[s] for s in picked])
        ma, mb = metrics_for(a, ca, idx), metrics_for(b, cb, idx)
        for m in METRICS:
            deltas[m].append(mb[m] - ma[m])

    out = {"dataset": dataset, "a": tag_a, "b": tag_b, "n_examples": len(sources),
           "n_sources": len(unique), "reps": reps, "metrics": {}}
    for m in METRICS:
        d = np.array(deltas[m])
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["metrics"][m] = {
            "a": point[m][0], "b": point[m][1], "delta": point[m][1] - point[m][0],
            "ci95": [float(lo), float(hi)],
            "share_of_replicates_b_not_better": float((d <= 0).mean()),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="ragtruth_val", help="baseline tag, e.g. ragtruth_val")
    parser.add_argument("--b", default="modernbert-large", help="challenger tag")
    parser.add_argument("--datasets", nargs="+", default=["ragtruth", "psiloqa"])
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = []
    for dataset in args.datasets:
        r = compare(args.a, args.b, dataset, args.reps, args.seed)
        results.append(r)
        print(f"\n{args.b} minus {args.a} on {dataset} ({r['n_examples']} examples, "
              f"{r['n_sources']} sources, {r['reps']} resamples)")
        print(f"{'metric':<15}{'A':>9}{'B':>9}{'delta':>9}   95% CI           not-better")
        for m, v in r["metrics"].items():
            scale = 1 if "auroc" not in m else 100
            print(f"{m:<15}{v['a']:>9.2f}{v['b']:>9.2f}{v['delta']:>+9.2f}   "
                  f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]   {v['share_of_replicates_b_not_better']:.3f}"
                  if scale == 1 else
                  f"{m:<15}{100*v['a']:>9.2f}{100*v['b']:>9.2f}{100*v['delta']:>+9.2f}   "
                  f"[{100*v['ci95'][0]:+.2f}, {100*v['ci95'][1]:+.2f}]   {v['share_of_replicates_b_not_better']:.3f}")

    out_path = RESULTS / f"bootstrap_{args.a}_vs_{args.b}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwritten to {out_path}")


if __name__ == "__main__":
    main()
