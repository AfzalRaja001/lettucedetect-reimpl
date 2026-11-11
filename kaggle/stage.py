"""Stage the project for Kaggle: a dataset folder and a kernel folder.

    python kaggle/stage.py --username YOURNAME --out <absolute staging folder>

Creates <out>/dataset (src, configs, results, data/processed, metadata) and
<out>/kernel (the chosen training script and metadata). Nothing is uploaded; the commands
to do that are printed. The output folder is outside the repo on purpose.
"""

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_SLUG = "lettucedetect-repo"
# kernel name -> (metadata template, entry-point script)
KERNELS = {
    "large": ("kernel-metadata.template.json", "train_large.py"),
    "ettin": ("kernel-metadata-ettin.template.json", "train_ettin.py"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True)
    parser.add_argument("--out", type=Path, required=True, help="absolute path of the staging folder")
    parser.add_argument("--kernel", choices=sorted(KERNELS), default="large",
                        help="which training kernel to stage next to the dataset")
    args = parser.parse_args()

    out = args.out.resolve()
    dataset_dir, kernel_dir = out / "dataset", out / "kernel"
    for folder in (dataset_dir, kernel_dir):
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)

    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(REPO_ROOT / "src", dataset_dir / "src", ignore=ignore)
    shutil.copytree(REPO_ROOT / "configs", dataset_dir / "configs")
    shutil.copytree(REPO_ROOT / "results", dataset_dir / "results")
    (dataset_dir / "data" / "processed").mkdir(parents=True)
    for jsonl in sorted((REPO_ROOT / "data" / "processed").glob("*.jsonl")):
        shutil.copy2(jsonl, dataset_dir / "data" / "processed" / jsonl.name)

    with open(dataset_dir / "dataset-metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "title": "lettucedetect repo",
            "id": f"{args.username}/{DATASET_SLUG}",
            "licenses": [{"name": "other"}],
        }, f, indent=2)

    template_name, script_name = KERNELS[args.kernel]
    template = (REPO_ROOT / "kaggle" / template_name).read_text(encoding="utf-8")
    (kernel_dir / "kernel-metadata.json").write_text(template.replace("USERNAME", args.username), encoding="utf-8")
    shutil.copy2(REPO_ROOT / "kaggle" / script_name, kernel_dir / script_name)

    size_mb = sum(p.stat().st_size for p in dataset_dir.rglob("*") if p.is_file()) / 1e6
    print(f"staged {size_mb:.0f} MB under {out}")
    print(f"  kaggle datasets version -p {dataset_dir} -r zip -m 'update'   (dataset already exists)")
    print(f"  kaggle kernels push -p {kernel_dir} --accelerator NvidiaTeslaT4")


if __name__ == "__main__":
    main()
