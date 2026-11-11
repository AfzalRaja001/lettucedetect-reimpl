"""Train and score other backbones on RAGTruth, resumably, with one command.

    python src/run_backbones.py                      # ettin-68m then ettin-150m
    python src/run_backbones.py --backbones ettin-68m --seeds 42 1337

Same recipe as the ModernBERT-base run (configs/base_config.yaml): RAGTruth
train with a validation split for checkpoint selection, then both test sets.
Each step is skipped if its output already exists, so a crash or Ctrl+C is
fixed by running the same command again.

Run names: "<backbone>" for seed 42, "<backbone>_s<seed>" for other seeds.
The seed-42 ModernBERT-base run already exists as "ragtruth_val", so it is
not retrained here.
"""

import argparse
import time

from model import BACKBONES
from run_experiments import CHECKPOINTS, evaluate_step, preflight, run_script

DEFAULT_SEED = 42


def run_name(backbone: str, seed: int) -> str:
    return backbone if seed == DEFAULT_SEED else f"{backbone}_s{seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbones", nargs="+", default=["ettin-68m", "ettin-150m"], choices=sorted(BACKBONES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[DEFAULT_SEED])
    parser.add_argument("--config", default=None, help="config file; defaults to configs/base_config.yaml")
    args = parser.parse_args()

    started = time.time()
    preflight()

    for backbone in args.backbones:
        for seed in args.seeds:
            if backbone == "modernbert-base" and seed == DEFAULT_SEED:
                print("[skip] modernbert-base seed 42 already exists as 'ragtruth_val'")
                continue
            name = run_name(backbone, seed)
            best = CHECKPOINTS / name / "best"
            if (best / "config.json").exists():
                print(f"[skip] {name} already trained")
            else:
                train_args = ["--train-on", "ragtruth", "--run-name", name, "--backbone", backbone, "--seed", str(seed)]
                if args.config:
                    train_args += ["--config", args.config]
                run_script("train.py", *train_args)
            evaluate_step(best, "ragtruth", f"{name}_on_ragtruth")
            evaluate_step(best, "psiloqa", f"{name}_on_psiloqa")

    run_script("summarize_results.py")
    print(f"\nAll done in {(time.time() - started) / 3600:.1f} h. Commit results/ and push.")


if __name__ == "__main__":
    main()
