"""Sanity-check label alignment by hand before trusting it on the full dataset.
"""

from transformers import AutoTokenizer

from data_prep import RAW_DIR, TOKENIZER_NAME, load_raw_ragtruth, tokenize_and_align_labels


def decode_predicted_spans(sample: dict, tokenizer, max_length: int = 2048) -> list[str]:
    """Re-derive the answer substrings that ended up labeled 1, from the tokenized encoding.
    """
    encoded = tokenize_and_align_labels(sample, tokenizer, max_length)

    # Re-tokenize to get offsets back (tokenize_and_align_labels pops them internally).
    full_encoding = tokenizer(
        sample["context"],
        sample["answer"],
        truncation="only_first",
        max_length=max_length,
        return_offsets_mapping=True,
    )
    offsets = full_encoding["offset_mapping"]
    labels = encoded["labels"]

    answer_only_ids = tokenizer(sample["answer"], add_special_tokens=False)["input_ids"]
    answer_start_token = len(encoded["input_ids"]) - len(answer_only_ids) - 1
    answer_char_offset = offsets[answer_start_token][0]

    spans = []
    current_start = None
    for i in range(answer_start_token, len(labels)):
        if labels[i] == 1:
            token_start, token_end = offsets[i]
            if current_start is None:
                current_start = token_start - answer_char_offset
            current_end = token_end - answer_char_offset
        else:
            if current_start is not None:
                spans.append(sample["answer"][current_start:current_end])
                current_start = None
    if current_start is not None:
        spans.append(sample["answer"][current_start:current_end])

    return spans


def main(num_examples: int = 10) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    samples = load_raw_ragtruth(RAW_DIR)

    # Only look at samples that actually have an annotated hallucination -
    # checking clean examples tells you nothing about whether the offset math works.
    hallucinated_samples = [s for s in samples if s["spans"]]

    print(f"{len(hallucinated_samples)} of {len(samples)} RAGTruth responses "
          f"have at least one annotated hallucination.\n")

    mismatches = 0
    for sample in hallucinated_samples[:num_examples]:
        predicted_spans = decode_predicted_spans(sample, tokenizer)
        gold_spans = [sample["answer"][s["start"]:s["end"]] for s in sample["spans"]]

        print(f"--- source_id={sample['source_id']} response_id={sample['response_id']} "
              f"task={sample['task_type']} ---")
        print(f"  gold spans:      {gold_spans}")
        print(f"  recovered spans: {predicted_spans}")

        match = set(predicted_spans) == set(gold_spans) or all(
            any(g in p or p in g for p in predicted_spans) for g in gold_spans
        )
        print(f"  MATCH: {match}\n")
        if not match:
            mismatches += 1

    print(f"Checked {min(num_examples, len(hallucinated_samples))} examples, "
          f"{mismatches} mismatches.")


if __name__ == "__main__":
    main()
