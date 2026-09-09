"""Load the cached, tokenized RAGTruth splits into a Hugging Face Dataset.

data_prep.py already did the expensive part - tokenizing and label-aligning
every example - and wrote the result to data/processed/ragtruth_{split}.jsonl.
Each cached record's input_ids/attention_mask/labels are plain Python lists,
deliberately left unpadded: padding every example up front to one fixed
length would waste compute on short examples, so instead we pad per-batch,
at collate time, only as much as that specific batch needs. That's what
DataCollatorForTokenClassification (used directly in train.py) does - this
module's only job is handing it un-padded examples to work with.
"""

import json
from pathlib import Path

from datasets import Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def load_cached_split(split: str, processed_dir: Path = PROCESSED_DIR) -> Dataset:
    """Load one cached split written by data_prep.py.

    :param split: "train" or "test"
    :param processed_dir: directory containing ragtruth_{split}.jsonl
    :return: a Hugging Face Dataset exposing input_ids, attention_mask, labels,
        and task_type (kept for per-task metric breakdowns; Trainer ignores
        it automatically since it isn't one of the model's forward arguments)
    """
    path = processed_dir / f"ragtruth_{split}.jsonl"
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            records.append(
                {
                    "input_ids": row["input_ids"],
                    "attention_mask": row["attention_mask"],
                    "labels": row["labels"],
                    "task_type": row["task_type"],
                }
            )
    return Dataset.from_list(records)
