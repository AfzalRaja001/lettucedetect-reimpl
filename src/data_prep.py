
"""Prepare RAGTruth examples for token-level hallucination detection.

RAGTruth stores source information and model responses in separate JSONL
files. This module joins those files, keeps the original prompt as context,
and converts character-level hallucination annotations into token-level
labels suitable for ModernBERT.

For each response we create one example containing:
    - context: the original RAGTruth prompt
    - answer: the generated response
    - spans: annotated hallucinated character ranges
    - metadata: task type, model, split, and IDs

Only answer tokens receive 0/1 labels. Context and special tokens are
assigned -100 so that they do not contribute to the training loss."""

import json
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

TOKENIZER_NAME = "answerdotai/ModernBERT-base"
DEFAULT_MAX_LENGTH = 2048

# Labels used by the token-classification model.
IGNORE_LABEL = -100
SUPPORTED_LABEL = 0
HALLUCINATED_LABEL = 1


def load_raw_ragtruth(raw_dir: Path = RAW_DIR) -> list[dict]:
    """Join RAGTruth source and response files into flat training samples.

    Parameters
    ----------
    raw_dir:
        Directory containing source_info.jsonl and response.jsonl.

    Returns
    -------
    list[dict]
        One dictionary per response containing context, answer,
        hallucination spans, and relevant metadata.
    """
    sources_by_id: dict[str, dict] = {}

    with open(raw_dir / "source_info.jsonl", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            sources_by_id[record["source_id"]] = record

    samples = []

    with open(raw_dir / "response.jsonl", encoding="utf-8") as f:
        for line in f:
            response = json.loads(line)
            source_id = response["source_id"]

            # Every response should have a corresponding source record.
            # failing here gives a clearer error than a later KeyError.
            if source_id not in sources_by_id:
                raise ValueError(
                    f"Missing source information for source_id={source_id}"
                )

            source = sources_by_id[source_id]

            hallucination_spans = [
                {
                    "start": label["start"],
                    "end": label["end"],
                    "category": label["label_type"],
                }
                for label in response["labels"]
            ]

            samples.append(
                {
                    "context": source["prompt"],
                    "answer": response["response"],
                    "spans": hallucination_spans,
                    "task_type": source["task_type"],
                    "model": response["model"],
                    "split": response["split"],
                    "source_id": source_id,
                    "response_id": response["id"],
                }
            )

    return samples


def validate_token_labels(
    input_ids: list[int],
    labels: list[int],
) -> None:
    """Ensure that every token has exactly one corresponding label."""
    if len(input_ids) != len(labels):
        raise ValueError(
            f"Token/label length mismatch: "
            f"{len(input_ids)} tokens vs {len(labels)} labels"
        )


def tokenize_and_align_labels(
    sample: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict:
    """Tokenize a sample and create token-level hallucination labels.

    The tokenizer creates the following layout:

        [CLS] context [SEP] answer [SEP]

    ``truncation="only_first"`` ensures that when the sequence is too long,
    tokens are removed only from the context. The answer therefore remains
    intact and can be aligned with the original hallucination spans.

    Parameters
    ----------
    sample:
        One item returned by load_raw_ragtruth().
    tokenizer:
        Fast tokenizer compatible with ModernBERT.
    max_length:
        Maximum total sequence length.

    Returns
    -------
    dict
        Tokenized inputs together with one label per token.
    """
    encoding = tokenizer(
        sample["context"],
        sample["answer"],
        truncation="only_first",
        max_length=max_length,
        return_offsets_mapping=True,
    )

    offsets = encoding.pop("offset_mapping")

    # Because only the first sequence can be truncated, the answer is
    # always preserved at the end of the tokenized sequence. Counting
    # backward lets us find its starting token without assuming a fixed
    # context length.
    answer_only_ids = tokenizer(
        sample["answer"],
        add_special_tokens=False,
    )["input_ids"]

    answer_token_count = len(answer_only_ids)
    total_len = len(encoding["input_ids"])

    # The final token is the trailing [SEP].
    answer_start_token = total_len - answer_token_count - 1

    # Context and special tokens are ignored during loss computation.
    labels = [IGNORE_LABEL] * total_len

    # The first answer token gives us the character offset at which the
    # answer begins inside the pair-encoded sequence.
    answer_char_offset = offsets[answer_start_token][0]

    for i in range(answer_start_token, total_len):
        token_start, token_end = offsets[i]

        # Special tokens have zero-width offsets and should not receive
        # hallucination labels.
        if token_start == token_end:
            continue

        # Convert the token's absolute offset back to an offset relative
        # to the beginning of the answer.
        rel_start = token_start - answer_char_offset
        rel_end = token_end - answer_char_offset

        label = SUPPORTED_LABEL

        # A token is considered hallucinated when its character range
        # overlaps any human-annotated hallucination span.
        for span in sample["spans"]:
            if rel_end > span["start"] and rel_start < span["end"]:
                label = HALLUCINATED_LABEL
                break

        labels[i] = label

    validate_token_labels(
        encoding["input_ids"],
        labels,
    )

    encoding["labels"] = labels
    return encoding


def build_and_cache(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> None:
    """Tokenize RAGTruth and save one processed file for each split.

    Parameters
    ----------
    raw_dir:
        Directory containing the raw RAGTruth JSONL files.
    processed_dir:
        Directory where processed JSONL files will be written.
    max_length:
        Maximum sequence length passed to the tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    samples = load_raw_ragtruth(raw_dir)

    by_split: dict[str, list[dict]] = {
        "train": [],
        "test": [],
    }

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

        split = sample["split"]

        if split not in by_split:
            raise ValueError(
                f"Unexpected dataset split '{split}'. "
                f"Expected one of: {sorted(by_split)}"
            )

        by_split[split].append(record)

    processed_dir.mkdir(parents=True, exist_ok=True)

    for split, records in by_split.items():
        output_path = processed_dir / f"ragtruth_{split}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        print(f"Wrote {len(records)} examples to {output_path}")


if __name__ == "__main__":
    build_and_cache()