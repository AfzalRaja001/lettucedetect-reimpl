"""Load RAGTruth and turn it into token-level hallucination labels for ModernBERT.

RAGTruth ships as two JSONL files that have to be joined by source_id:
  - source_info.jsonl: one row per (task, source document, question), i.e. the
    thing the LLM was asked to do.
  - response.jsonl: one row per LLM *answer* to a source item (up to 6 answers
    per source item, one from each model RAGTruth used), plus the human-
    annotated hallucination spans for that answer.

For each response, we build one training example:
    context = the exact prompt text the generating LLM was originally shown
               (instructions + question + retrieved passages, all together)
    answer  = the LLM's response text
    spans   = character-offset ranges within `answer` that are hallucinated

We deliberately use the *whole original prompt* as the context, not a
hand-split (context, question) pair. See the module docstring in
verify_alignment.py for why - it matches how the paper's own reference
implementation constructs its input, and RAGTruth's three task types (QA,
summarization, data-to-text) don't share a common way to split "question"
out of the prompt anyway.
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
    """Join response.jsonl and source_info.jsonl into flat training samples.

    :param raw_dir: directory containing response.jsonl and source_info.jsonl
    :return: list of dicts with keys context, answer, spans, task_type,
        model, split, source_id, response_id
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
            source = sources_by_id[response["source_id"]]
            samples.append(
                {
                    "context": source["prompt"],
                    "answer": response["response"],
                    "spans": [
                        {"start": lab["start"], "end": lab["end"], "category": lab["label_type"]}
                        for lab in response["labels"]
                    ],
                    "task_type": source["task_type"],
                    "model": response["model"],
                    "split": response["split"],
                    "source_id": response["source_id"],
                    "response_id": response["id"],
                }
            )
    return samples


def tokenize_and_align_labels(
    sample: dict,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict:
    """Tokenize (context, answer) as a pair and produce a -100/0/1 label per token.

    Layout produced by the tokenizer's pair-encoding: [CLS] context [SEP] answer [SEP].
    truncation="only_first" guarantees the context gets cut before the answer
    ever does, so every answer token survives even on long RAGTruth examples.

    :param sample: one item from load_raw_ragtruth()
    :param tokenizer: a ModernBERT-compatible fast tokenizer
    :param max_length: total sequence length cap (context + answer + specials)
    :return: dict with input_ids, attention_mask, labels (all same length)
    """
    encoding = tokenizer(
        sample["context"],
        sample["answer"],
        truncation="only_first",
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = encoding.pop("offset_mapping")

    # The answer is never truncated, so it always sits at the very end of the
    # sequence. Counting backward from the end locates it correctly no matter
    # how much of the context got cut off - counting forward from the start
    # would break the moment truncation actually happens.
    answer_only_ids = tokenizer(sample["answer"], add_special_tokens=False)["input_ids"]
    answer_token_count = len(answer_only_ids)
    total_len = len(encoding["input_ids"])
    answer_start_token = total_len - answer_token_count - 1  # -1 skips the trailing [SEP]

    labels = [-100] * total_len
    answer_char_offset = offsets[answer_start_token][0]

    for i in range(answer_start_token, total_len):
        token_start, token_end = offsets[i]
        if token_start == token_end:
            # Special tokens (the trailing [SEP]) report a zero-width (0, 0)
            # offset - never treat that as real answer text.
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


def build_and_cache(
    raw_dir: Path = RAW_DIR,
    processed_dir: Path = PROCESSED_DIR,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> None:
    """Tokenize every RAGTruth sample and write one cache file per split.

    :param raw_dir: directory containing the raw RAGTruth JSONL files
    :param processed_dir: directory to write ragtruth_{split}.jsonl into
    :param max_length: sequence length cap passed to the tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    samples = load_raw_ragtruth(raw_dir)

    by_split: dict[str, list[dict]] = {"train": [], "test": []}
    for sample in samples:
        encoded = tokenize_and_align_labels(sample, tokenizer, max_length)
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
