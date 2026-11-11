"""Kaggle entry point: train and score ModernBERT-large on RAGTruth.

Runs unattended with "Save & Run All" (kaggle kernels push). Stages:
  1. copy the project from the attached Kaggle Dataset into /kaggle/working/repo
  2. memory/stability check on the 48 longest examples (aborts early if it OOMs
     or the fp16 loss goes NaN, instead of failing hours in)
  3. train, with a time budget that stops cleanly at an epoch boundary
  4. score the best checkpoint on both test sets and rebuild the summary table
  5. keep only results/ and the best checkpoint in /kaggle/working

Uses one GPU (CUDA_VISIBLE_DEVICES=0) so the effective batch size stays 8.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "repo"
CONFIG = "configs/kaggle_large.yaml"
BACKBONE = "modernbert-large"
RUN_NAME = "modernbert-large"
# Kaggle stops a session at 12 h. Leave room for evaluation and copying.
TIME_BUDGET_HOURS = 10.5


def sh(command: str) -> None:
    print(f"\n>>> {command}", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=REPO if REPO.exists() else WORK)


def find_project() -> Path:
    hits = glob.glob("/kaggle/input/**/src/train.py", recursive=True)
    if not hits:
        sys.exit("project dataset not attached: no src/train.py under /kaggle/input")
    return Path(hits[0]).parent.parent


def main() -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Kaggle pipes child output through a legacy codec ("charmap"), and tqdm /
    # transformers print non-ASCII characters, which crashed the first run at
    # the memory check. Force UTF-8 for every child process.
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sh("nvidia-smi")
    sh("pip install -q -U 'transformers>=4.48' datasets accelerate scikit-learn pyyaml")

    if REPO.exists():
        shutil.rmtree(REPO)
    shutil.copytree(find_project(), REPO)

    train = f"python src/train.py --backbone {BACKBONE} --config {CONFIG}"
    sh(f"{train} --max-steps 6 --run-name memcheck")
    shutil.rmtree(REPO / "checkpoints" / "memcheck", ignore_errors=True)

    sh(f"{train} --train-on ragtruth --run-name {RUN_NAME} --resume --time-budget-hours {TIME_BUDGET_HOURS}")

    best = REPO / "checkpoints" / RUN_NAME / "best"
    for dataset in ("ragtruth", "psiloqa"):
        sh(f"python src/evaluate.py --checkpoint {best} --dataset {dataset} "
           f"--tag {RUN_NAME}_on_{dataset} --eval-batch-size 2")
    sh("python src/summarize_results.py")

    # Keep the small, valuable files; drop the 5GB intermediate checkpoints.
    shutil.copytree(REPO / "results", WORK / "results", dirs_exist_ok=True)
    shutil.move(str(best), str(WORK / f"{RUN_NAME}-best"))
    shutil.rmtree(REPO)
    with open(WORK / "done.json", "w", encoding="utf-8") as f:
        json.dump({"status": "finished", "run_name": RUN_NAME}, f)
    print("\nFinished. Download with: kaggle kernels output <owner>/<kernel> -p <dir>")


if __name__ == "__main__":
    main()
