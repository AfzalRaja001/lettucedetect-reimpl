"""Kaggle entry point: train Ettin-68M and Ettin-150M on RAGTruth, one per GPU.

Runs unattended ("Save & Run All" via kaggle kernels push). The Kaggle T4 x2
session has two GPUs, so the two models train at the same time, each pinned to
its own GPU (CUDA_VISIBLE_DEVICES) and working in its own copy of the project
so checkpoints and scratch folders cannot collide.

Each pipeline: memory/stability check, train (with a time budget), score the
best checkpoint on both test sets. A failure in one pipeline does not stop the
other. Only results files and the best checkpoints are kept in /kaggle/working.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

WORK = Path("/kaggle/working")
CONFIG = "configs/kaggle_small.yaml"
# (backbone key in src/model.py, which GPU it trains on)
JOBS = [("ettin-150m", 0), ("ettin-68m", 1)]
TIME_BUDGET_HOURS = 10.5  # Kaggle stops a session at 12 h

status: dict[str, str] = {}
print_lock = threading.Lock()


def find_project() -> Path:
    hits = glob.glob("/kaggle/input/**/src/train.py", recursive=True)
    if not hits:
        sys.exit("project dataset not attached: no src/train.py under /kaggle/input")
    return Path(hits[0]).parent.parent


def sh(label: str, repo: Path, gpu: int, command: str) -> None:
    """Run one command pinned to a GPU, prefixing every output line with the job label."""
    with print_lock:
        print(f"\n[{label}] >>> {command}", flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false",
               PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(command, shell=True, cwd=repo, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        with print_lock:
            print(f"[{label}] {line}", end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"{command!r} exited with {proc.returncode}")


def pipeline(source: Path, backbone: str, gpu: int) -> None:
    repo = WORK / f"repo_{backbone}"
    try:
        if repo.exists():
            shutil.rmtree(repo)
        shutil.copytree(source, repo)

        train = f"python src/train.py --backbone {backbone} --config {CONFIG}"
        sh(backbone, repo, gpu, f"{train} --max-steps 6 --run-name memcheck")
        shutil.rmtree(repo / "checkpoints" / "memcheck", ignore_errors=True)

        sh(backbone, repo, gpu, f"{train} --train-on ragtruth --run-name {backbone} --resume "
                                f"--time-budget-hours {TIME_BUDGET_HOURS}")
        best = repo / "checkpoints" / backbone / "best"
        for dataset in ("ragtruth", "psiloqa"):
            sh(backbone, repo, gpu, f"python src/evaluate.py --checkpoint {best} --dataset {dataset} "
                                    f"--tag {backbone}_on_{dataset} --eval-batch-size 4")

        (WORK / "results").mkdir(exist_ok=True)
        for f in (repo / "results").glob(f"{backbone}_on_*"):
            shutil.copy2(f, WORK / "results" / f.name)
        shutil.move(str(best), str(WORK / f"{backbone}-best"))
        shutil.rmtree(repo, ignore_errors=True)
        status[backbone] = "finished"
    except Exception as error:  # keep the other pipeline running
        status[backbone] = f"failed: {error}"
        with print_lock:
            print(f"[{backbone}] FAILED: {error}", flush=True)


def main() -> None:
    subprocess.run("nvidia-smi", shell=True, check=False)
    subprocess.run("pip install -q -U 'transformers>=4.48' datasets accelerate scikit-learn pyyaml",
                   shell=True, check=True)
    source = find_project()
    threads = [threading.Thread(target=pipeline, args=(source, name, gpu)) for name, gpu in JOBS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    with open(WORK / "done.json", "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print("\nFinished:", status)
    if not all(v == "finished" for v in status.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
