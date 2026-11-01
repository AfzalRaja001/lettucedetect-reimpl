"""Run the whole end-sem experiment queue with one command, resumably.

    python src/run_experiments.py

Every step checks whether its output already exists and is skipped if so, so
after a crash, a reboot, or a Ctrl+C you just run the same command again and
it carries on from the first unfinished step.

Queue, in order:
  1. RAGTruth baseline checkpoint scored on the PsiloQA test set
  2. Rebuild the RAGTruth cache with a validation split
  3. Train on RAGTruth + PsiloQA together ("both"), score on both test sets
  4. Train on RAGTruth alone with a validation split ("ragtruth_val"), score on both
  5. Print and save the comparison table

Step 4 is the fair control for step 3. The original RAGTruth run chose its
checkpoint without a validation split, so it is not directly comparable.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
PROCESSED = REPO_ROOT / "data" / "processed"
CHECKPOINTS = REPO_ROOT / "checkpoints"
RESULTS = REPO_ROOT / "results"

BASELINE_CHECKPOINT = CHECKPOINTS / "run" / "best"
VAL_FRACTION = "0.1"


def run_script(script: str, *args: str) -> None:
    """Run one of the project scripts with the same interpreter as this one."""
    command = [sys.executable, str(SRC / script), *args]
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def evaluate_step(checkpoint: Path, dataset: str, tag: str) -> None:
    if (RESULTS / f"{tag}_metrics.json").exists():
        print(f"[skip] {tag} already evaluated")
        return
    run_script("evaluate.py", "--checkpoint", str(checkpoint), "--dataset", dataset, "--tag", tag)


def train_step(train_on: str, run_name: str) -> Path:
    best = CHECKPOINTS / run_name / "best"
    if (best / "config.json").exists():
        print(f"[skip] {run_name} already trained")
    else:
        run_script("train.py", "--train-on", train_on, "--run-name", run_name)
    return best


def preflight() -> None:
    """Fail early, before hours of work, if the machine cannot do the job."""
    import torch

    if not torch.cuda.is_available():
        sys.exit("CUDA is not available - training would run on the CPU. Fix the torch install first.")
    missing = [p for p in ("psiloqa_train", "psiloqa_val", "psiloqa_test", "ragtruth_train", "ragtruth_test")
               if not (PROCESSED / f"{p}.jsonl").exists()]
    if missing:
        sys.exit(f"Missing cached data {missing}. Run data_prep.py for both datasets first.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="only print the plan")
    args = parser.parse_args()

    plan = [
        ("baseline on PsiloQA test", BASELINE_CHECKPOINT / "config.json"),
        ("ragtruth validation split", PROCESSED / "ragtruth_val.jsonl"),
        ("train both + eval", CHECKPOINTS / "both" / "best" / "config.json"),
        ("train ragtruth_val + eval", CHECKPOINTS / "ragtruth_val" / "best" / "config.json"),
    ]
    if args.dry_run:
        for name, marker in plan:
            print(f"{'done' if marker.exists() else 'todo'}  {name}   ({marker})")
        return

    started = time.time()
    preflight()

    # 1. The other transfer direction. Skipped (not fatal) if this machine
    #    does not hold the original checkpoint.
    if BASELINE_CHECKPOINT.exists():
        evaluate_step(BASELINE_CHECKPOINT, "psiloqa", "ragtruth_on_psiloqa")
    else:
        print(f"[warn] {BASELINE_CHECKPOINT} not found, skipping ragtruth_on_psiloqa")

    # 2. Validation split for RAGTruth (also needed to train "both").
    if (PROCESSED / "ragtruth_val.jsonl").exists():
        print("[skip] ragtruth_val.jsonl already built")
    else:
        run_script("data_prep.py", "--dataset", "ragtruth", "--val-fraction", VAL_FRACTION)

    # 3 and 4. The two training runs, each scored on both test sets.
    for train_on, run_name in (("both", "both"), ("ragtruth", "ragtruth_val")):
        best = train_step(train_on, run_name)
        evaluate_step(best, "ragtruth", f"{run_name}_on_ragtruth")
        evaluate_step(best, "psiloqa", f"{run_name}_on_psiloqa")

    run_script("summarize_results.py")
    print(f"\nAll done in {(time.time() - started) / 3600:.1f} h. "
          f"Commit results/ and push.")


if __name__ == "__main__":
    main()
