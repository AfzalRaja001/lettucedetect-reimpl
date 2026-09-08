"""Fine-tune ModernBERT for token-level hallucination classification.

Uses Hugging Face's Trainer rather than a hand-written training loop, per the
project brief - Trainer already implements the batching loop, gradient
accumulation, mixed precision, checkpoint saving, and evaluation scheduling,
all of which are the same regardless of what model or task you're training.
Writing that loop by hand would just be re-deriving library code.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from dataset import load_cached_split
from model import MODEL_NAME, build_model

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "base_config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Read the YAML config and resolve every path field against the repo root.

    Storing "data/processed" in the YAML file (rather than a full absolute
    path) is what lets the same config work unmodified on both this machine
    and the RTX 4060 machine - only the code doing the resolving needs to
    know where the repo root actually is, via `Path(__file__).resolve()`.

    :param config_path: path to a YAML config file
    :return: the parsed config dict, with paths/* replaced by absolute Paths
    """
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["paths"] = {key: REPO_ROOT / value for key, value in config["paths"].items()}
    return config


def compute_token_metrics(eval_pred) -> dict:
    """Token-level precision/recall/F1, ignoring -100 (context) positions.

    This is deliberately just the checkpoint-selection metric mentioned in
    the project brief's hyperparameter table (token-level F1) - not the
    example-level or span-level F1 that the paper actually reports results
    with. Those need aggregation logic beyond a single batch of predictions
    and belong in evaluate.py (Phase 4), run once against a finished model.

    :param eval_pred: a transformers EvalPrediction with .predictions
        (raw logits, shape [batch, seq_len, 2]) and .label_ids
        (shape [batch, seq_len], containing -100 for ignored positions)
    :return: dict with token_precision, token_recall, token_f1
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels, true_predictions = [], []
    for pred_row, label_row in zip(predictions, labels):
        for pred_token, label_token in zip(pred_row, label_row):
            if label_token != -100:
                true_labels.append(label_token)
                true_predictions.append(pred_token)

    return {
        "token_precision": precision_score(true_labels, true_predictions, zero_division=0),
        "token_recall": recall_score(true_labels, true_predictions, zero_division=0),
        "token_f1": f1_score(true_labels, true_predictions, zero_division=0),
    }


def build_training_arguments(config: dict, output_dir: Path, smoke_test: bool) -> TrainingArguments:
    """Translate the YAML config into a transformers TrainingArguments object.

    :param config: parsed config from load_config()
    :param output_dir: where checkpoints and logs get written
    :param smoke_test: if True, use a tiny, fast configuration instead of the
        real hyperparameters, so this can be run on a CPU dev machine in
        seconds rather than hours
    :return: a TrainingArguments instance
    """
    training = config["training"]
    use_cuda = torch.cuda.is_available()

    if smoke_test:
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=1,
            report_to=[],
        )

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=training["epochs"],
        per_device_train_batch_size=training["batch_size"],
        per_device_eval_batch_size=training["batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        # bf16 only makes sense with a CUDA GPU that supports it (the RTX
        # 4060 does); on a CPU dev machine this would either error or be a
        # silent no-op, so gate it on what hardware is actually running.
        bf16=use_cuda and training["precision"] == "bf16",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=config["evaluation"]["metric_for_best_model"],
        seed=training["seed"],
        report_to=[],
    )


def run(config_path: Path = DEFAULT_CONFIG_PATH, smoke_test: bool = False) -> None:
    """Load data and model per the config, then fine-tune.

    :param config_path: path to a YAML config file
    :param smoke_test: if True, train on a small slice of real data for one
        fast epoch, to catch runtime bugs (shape mismatches, bad field
        names) before committing to a real multi-hour run on the GPU machine
    """
    config = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = build_model(MODEL_NAME)

    train_dataset = load_cached_split("train", config["paths"]["processed_data_dir"])
    eval_dataset = load_cached_split("test", config["paths"]["processed_data_dir"])

    if smoke_test:
        # Sort by length and take the shortest examples, so the smoke test
        # stays fast on a CPU-only dev machine. This is still real RAGTruth
        # data, not synthetic - only the *selection* of examples is biased
        # toward short ones, purely to keep runtime down.
        train_dataset = train_dataset.map(lambda ex: {"length": len(ex["input_ids"])})
        train_dataset = train_dataset.sort("length").select(range(20)).remove_columns("length")
        eval_dataset = eval_dataset.map(lambda ex: {"length": len(ex["input_ids"])})
        eval_dataset = eval_dataset.sort("length").select(range(10)).remove_columns("length")

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    output_dir = config["paths"]["checkpoints_dir"] / ("smoke_test" if smoke_test else "run")
    training_args = build_training_arguments(config, output_dir, smoke_test)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_token_metrics,
    )
    trainer.train()

    if not smoke_test:
        model.save_pretrained(output_dir / "best")
        tokenizer.save_pretrained(output_dir / "best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    run(config_path=args.config, smoke_test=args.smoke_test)
