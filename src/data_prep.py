
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

import argparse
import hashlib
import json
import random
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase


REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PSILOQA_DIR = RAW_DIR / "psiloqa"

PSILOQA_HF_ID = "s-nlp/PsiloQA"
PSILOQA_LANGUAGE = "en"
# PsiloQA calls its dev split "validation"; the caches use the short name "val".
PSILOQA_SPLITS = {"train": "train", "validation": "val", "test": "test"}

# Mirrors the RAGTruth QA prompt, so PsiloQA examples look like the QA examples
# the model already sees, instead of introducing a second input format.
PSILOQA_PROMPT = (
    "Briefly answer the following question:\n{question}\n"
    "Bear in mind that your response should be strictly based on the following passage:\n"
    "passage 1:{passage}\n\n"
    "In case the passage does not contain the necessary information to answer "
    'the question, please reply with: "Unable to answer based on given passages."\n'
    "output:"
)

VAL_SEED = 42

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


def encode_record(
    sample: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict:
    """Tokenize one sample into the record format stored in the processed caches."""
    encoded = tokenize_and_align_labels(sample, tokenizer, max_length)

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": encoded["labels"],
        "task_type": sample["task_type"],
        "source_id": sample["source_id"],
        "response_id": sample["response_id"],
    }


def write_split(records: list[dict], output_path: Path) -> None:
    """Write one cached split as JSONL and report its size."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} examples to {output_path}")


def split_train_val_by_source(
    train_samples: list[dict],
    val_fraction: float,
    seed: int = VAL_SEED,
) -> tuple[list[dict], list[dict]]:
    """Hold out whole source documents for validation.

    RAGTruth has about six responses per source item, all sharing one
    context. Splitting per response would put the same context on both sides
    and make validation scores optimistic, so the split is per source_id.
    """
    source_ids = sorted({s["source_id"] for s in train_samples})
    random.Random(seed).shuffle(source_ids)
    val_sources = set(source_ids[: int(len(source_ids) * val_fraction)])

    train = [s for s in train_samples if s["source_id"] not in val_sources]
    val = [s for s in train_samples if s["source_id"] in val_sources]
    return train, val


def build_and_cache(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    max_length: int = DEFAULT_MAX_LENGTH,
    val_fraction: float = 0.0,
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
    val_fraction:
        Share of the official train split's source documents to hold out as
        a validation split. 0 keeps the original train/test-only layout.
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    samples = load_raw_ragtruth(raw_dir)

    by_split: dict[str, list[dict]] = {
        "train": [],
        "test": [],
    }

    for sample in samples:
        split = sample["split"]

        if split not in by_split:
            raise ValueError(
                f"Unexpected dataset split '{split}'. "
                f"Expected one of: {sorted(by_split)}"
            )

        by_split[split].append(sample)

    if val_fraction > 0:
        by_split["train"], by_split["val"] = split_train_val_by_source(
            by_split["train"], val_fraction
        )

    processed_dir.mkdir(parents=True, exist_ok=True)

    for split, split_samples in by_split.items():
        records = [encode_record(s, tokenizer, max_length) for s in split_samples]
        write_split(records, processed_dir / f"ragtruth_{split}.jsonl")


def download_psiloqa(raw_dir: Path = PSILOQA_DIR, language: str = PSILOQA_LANGUAGE) -> None:
    """Download PsiloQA from the Hugging Face Hub and keep one language as JSONL.

    ModernBERT-base is an English model, so only English rows are kept.
    Fields the pipeline does not use (golden answer, URLs, complexity) are dropped.
    """
    from datasets import load_dataset

    raw_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(PSILOQA_HF_ID)

    keep = ["id", "wiki_title", "llm_checkpoint", "wiki_passage", "question", "llm_answer", "labels"]

    for hf_split in PSILOQA_SPLITS:
        rows = [r for r in dataset[hf_split] if r["lang"] == language]
        output_path = raw_dir / f"{hf_split}.jsonl"

        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps({k: row[k] for k in keep}) + "\n")

        print(f"Wrote {len(rows)} '{language}' rows to {output_path}")


def load_raw_psiloqa(split: str, raw_dir: Path = PSILOQA_DIR) -> list[dict]:
    """Load one PsiloQA split as samples in the same shape load_raw_ragtruth returns.

    PsiloQA stores the passage, question, and the model's answer separately.
    The passage and question are rebuilt into a RAGTruth-style QA prompt, and
    the character spans are already relative to the answer text.

    ``source_id`` is a hash of the Wikipedia passage, so that threshold
    tuning and validation splits can group examples by passage, the same way
    RAGTruth is grouped by source document.
    """
    if split not in PSILOQA_SPLITS:
        raise ValueError(f"Unknown PsiloQA split '{split}'. Expected one of: {sorted(PSILOQA_SPLITS)}")

    samples = []
    skipped_empty = 0

    with open(raw_dir / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if not row["llm_answer"].strip():
                skipped_empty += 1
                continue

            samples.append(
                {
                    "context": PSILOQA_PROMPT.format(
                        question=row["question"], passage=row["wiki_passage"]
                    ),
                    "answer": row["llm_answer"],
                    "spans": [
                        {"start": start, "end": end, "category": "HAL"}
                        for start, end in row["labels"]
                    ],
                    "task_type": "PsiloQA",
                    "model": row["llm_checkpoint"],
                    "split": PSILOQA_SPLITS[split],
                    "source_id": hashlib.sha1(row["wiki_passage"].encode("utf-8")).hexdigest()[:12],
                    "response_id": row["id"],
                }
            )

    if skipped_empty:
        print(f"Skipped {skipped_empty} PsiloQA rows with an empty answer in '{split}'")

    return samples


def build_and_cache_psiloqa(
    raw_dir: Path = PSILOQA_DIR,
    processed_dir: Path = PROCESSED_DIR,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> None:
    """Tokenize PsiloQA and save psiloqa_{train,val,test}.jsonl."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for hf_split, cache_split in PSILOQA_SPLITS.items():
        samples = load_raw_psiloqa(hf_split, raw_dir)
        records = [encode_record(s, tokenizer, max_length) for s in samples]
        write_split(records, processed_dir / f"psiloqa_{cache_split}.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ragtruth", "psiloqa", "both"], default="ragtruth")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.0,
        help="RAGTruth only: share of train source documents held out as a validation split",
    )
    parser.add_argument("--download", action="store_true", help="PsiloQA only: fetch the raw data first")
    args = parser.parse_args()

    if args.dataset in ("ragtruth", "both"):
        build_and_cache(max_length=args.max_length, val_fraction=args.val_fraction)

    if args.dataset in ("psiloqa", "both"):
        if args.download or not (PSILOQA_DIR / "train.jsonl").exists():
            download_psiloqa()
        build_and_cache_psiloqa(max_length=args.max_length)