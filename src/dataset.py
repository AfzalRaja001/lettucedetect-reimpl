
"""Load cached RAGTruth examples into a Hugging Face Dataset.

data_prep.py performs tokenization and token-level label alignment.
This module loads those cached examples and keeps them unpadded.

Padding is intentionally deferred until batching. The training pipeline
uses DataCollatorForTokenClassification to dynamically pad each batch,
which avoids unnecessary padding for shorter examples."""

import json
from pathlib import Path

from datasets import Dataset


REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

VALID_SPLITS = {"train", "val", "test"}
VALID_DATASETS = {"ragtruth", "psiloqa"}


def load_cached_split(
    split: str,
    processed_dir: Path = PROCESSED_DIR,
    dataset: str = "ragtruth",
) -> Dataset:
    """Load one cached split of one dataset.

    Parameters
    ----------
    split:
        Dataset split to load: "train", "val", or "test".
    processed_dir:
        Directory containing the processed JSONL files.
    dataset:
        "ragtruth" or "psiloqa".

    Returns
    -------
    Dataset
        Hugging Face Dataset containing input_ids, attention_mask,
        labels, task_type, source_id, and response_id.
    """
    if split not in VALID_SPLITS:
        raise ValueError(
            f"Unsupported split '{split}'. "
            f"Expected one of: {sorted(VALID_SPLITS)}"
        )

    if dataset not in VALID_DATASETS:
        raise ValueError(
            f"Unsupported dataset '{dataset}'. "
            f"Expected one of: {sorted(VALID_DATASETS)}"
        )

    cache_path = processed_dir / f"{dataset}_{split}.jsonl"

    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path} does not exist. Build it with data_prep.py "
            f"(RAGTruth validation needs --val-fraction 0.1)."
        )

    records = []

    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            required_fields = {
                "input_ids",
                "attention_mask",
                "labels",
                "task_type",
            }

            missing_fields = required_fields - row.keys()

            if missing_fields:
                raise ValueError(
                    f"Missing fields in {cache_path}: "
                    f"{sorted(missing_fields)}"
                )

            records.append(
                {
                    "input_ids": row["input_ids"],
                    "attention_mask": row["attention_mask"],
                    "labels": row["labels"],
                    "task_type": row["task_type"],
                    "source_id": row["source_id"],
                    "response_id": row["response_id"],
                }
            )

    return Dataset.from_list(records)
