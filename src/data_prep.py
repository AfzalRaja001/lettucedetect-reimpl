"""Preparing RAGTruth data for token-level hallucination detection.

RAGTruth stores source information and model responses in separate JSONL
files. This module joins them using source_id and converts the annotated
character spans into token-level labels.

The original prompt is kept as the context followed by the model response.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TOKENIZER_NAME = "answerdotai/ModernBERT-base"
DEFAULT_MAX_LENGTH = 2048


def load_raw_ragtruth(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Loading the RAGTruth files and combine each response with its source."""

    sources_by_id: dict[str, dict] = {}

    with open(raw_dir / "source_info.jsonl", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            sources_by_id[record["source_id"]] = record

    samples = []

    with open(raw_dir / "response.jsonl", encoding="utf-8") as f:
        for line in f:
            response = json.loads(line)
            source = sources_by_id[response["source_id"]]

            samples.append(
                {
                    "context": source["prompt"],
                    "answer": response["response"],
                    "spans": [
                        {
                            "start": label["start"],
                            "end": label["end"],
                            "category": label["label_type"],
                        }
                        for label in response["labels"]
                    ],
                    "task_type": source["task_type"],
                    "model": response["model"],
                    "split": response["split"],
                    "source_id": response["source_id"],
                    "response_id": response["id"],
                }
            )

    return samples


def tokenize_and_align_labels(sample: dict,tokenizer: PreTrainedTokenizerBase,max_length: int = DEFAULT_MAX_LENGTH,) -> dict:
    """Tokenize a sample and create labels for the answer tokens."""

    encoding = tokenizer(
        sample["context"],
        sample["answer"],
        truncation="only_first",
        max_length=max_length,
        return_offsets_mapping=True,
    )

    offsets = encoding.pop("offset_mapping")

    # The context may be truncated, so find the answer tokens from the end.
    answer_only_ids = tokenizer(
        sample["answer"],
        add_special_tokens=False
    )["input_ids"]

    answer_token_count = len(answer_only_ids)
    total_len = len(encoding["input_ids"])

    answer_start_token = total_len - answer_token_count - 1

    labels = [-100] * total_len
    answer_char_offset = offsets[answer_start_token][0]

    for i in range(answer_start_token, total_len):
        token_start, token_end = offsets[i]

        # Skip special tokens with empty offsets.
        if token_start == token_end:
            continue

        rel_start = token_start - answer_char_offset
        rel_end = token_end - answer_char_offset

        label = 0

        for span in sample["spans"]:
            if rel_end > span["start"] and rel_start < span["end"]:
                label = 1
                break

        labels[i] = label

    encoding["labels"] = labels

    return encoding


def build_and_cache(raw_dir: Path = RAW_DIR,processed_dir: Path = PROCESSED_DIR,max_length: int = DEFAULT_MAX_LENGTH,) -> None:
    """Tokenize the dataset and save the processed splits."""

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    samples = load_raw_ragtruth(raw_dir)

    by_split: dict[str, list[dict]] = {"train": [], "test": []}

    for sample in samples:
        encoded = tokenize_and_align_labels(
            sample,
            tokenizer,
            max_length,
        )

        record = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encoded["labels"],
            "task_type": sample["task_type"],
            "source_id": sample["source_id"],
            "response_id": sample["response_id"],
        }

        by_split[sample["split"]].append(record)

    processed_dir.mkdir(parents=True, exist_ok=True)

    for split, records in by_split.items():
        out_path = processed_dir / f"ragtruth_{split}.jsonl"

        with open(out_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        print(f"wrote {len(records)} examples to {out_path}")


if __name__ == "__main__":
    build_and_cache()