"""Fine-tune ModernBERT for token-level hallucination classification.

Uses Hugging Face's Trainer rather than a hand-written training loop, per the
project brief - Trainer already implements the batching loop, gradient
accumulation, mixed precision, checkpoint saving, and evaluation scheduling,
all of which are the same regardless of what model or task you're training.
Writing that loop by hand would just be re-deriving library code.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import concatenate_datasets
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from dataset import load_cached_split
from model import BACKBONES, build_model

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "base_config.yaml"

TRAIN_SOURCES = {
    "ragtruth": ["ragtruth"],
    "psiloqa": ["psiloqa"],
    "both": ["ragtruth", "psiloqa"],
}


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


class StopOnNonFinite(TrainerCallback):
    """Abort as soon as the logged loss is NaN or infinite.

    fp16 training on older GPUs (T4) can overflow. Without this, a run would
    keep going for hours producing garbage.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None and not math.isfinite(loss):
            raise RuntimeError(f"training loss is {loss} at step {state.global_step}; aborting")


class TimeBudget(TrainerCallback):
    """Stop cleanly at an epoch boundary if the next epoch would exceed the budget.

    Kaggle kills a session after 12 hours. Stopping between epochs still
    evaluates and saves the epoch just finished, so the best checkpoint so far
    is kept instead of losing the whole run.
    """

    def __init__(self, hours: float):
        self.budget_seconds = hours * 3600
        self.start = 0.0
        self.epoch_start = 0.0
        self.longest_epoch = 0.0

    def on_train_begin(self, args, state, control, **kwargs):
        self.start = self.epoch_start = time.time()

    def on_epoch_end(self, args, state, control, **kwargs):
        now = time.time()
        self.longest_epoch = max(self.longest_epoch, now - self.epoch_start)
        self.epoch_start = now
        if now - self.start + self.longest_epoch > self.budget_seconds:
            print(f"[time budget] stopping after epoch {state.epoch:.0f}: next epoch would exceed "
                  f"{self.budget_seconds / 3600:.1f} h", flush=True)
            control.should_training_stop = True


def build_training_arguments(
    config: dict,
    output_dir: Path,
    smoke_test: bool,
    seed: int | None = None,
    max_steps: int | None = None,
) -> TrainingArguments:
    """Translate the YAML config into a transformers TrainingArguments object.

    :param config: parsed config from load_config()
    :param output_dir: where checkpoints and logs get written
    :param smoke_test: if True, use a tiny, fast configuration instead of the
        real hyperparameters, so this can be run on a CPU dev machine in
        seconds rather than hours
    :param seed: overrides training.seed from the config when given
    :param max_steps: if given, run only this many optimizer steps with no
        evaluation or saving (a memory and stability check, not a real run)
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

    args = dict(
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
        # fp16 is for GPUs without bf16 (Kaggle's T4).
        bf16=use_cuda and training["precision"] == "bf16",
        fp16=use_cuda and training["precision"] == "fp16",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,
        metric_for_best_model=config["evaluation"]["metric_for_best_model"],
        seed=seed if seed is not None else training["seed"],
        logging_steps=50,
        report_to=[],
    )
    # Memory savers for the 395M model; absent from the base config, so the
    # base recipe is unchanged.
    if training.get("gradient_checkpointing"):
        args.update(gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    if training.get("optim"):
        args["optim"] = training["optim"]
    if max_steps:
        args.update(
            max_steps=max_steps,
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            logging_steps=5,
        )
    return TrainingArguments(**args)


def load_combined_split(split: str, datasets: list[str], processed_dir: Path):
    """Load one split from every named dataset and concatenate them."""
    parts = [load_cached_split(split, processed_dir, dataset=name) for name in datasets]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def run(
    config_path: Path = DEFAULT_CONFIG_PATH,
    smoke_test: bool = False,
    train_on: str = "ragtruth",
    run_name: str | None = None,
    backbone: str = "modernbert-base",
    seed: int | None = None,
    max_steps: int | None = None,
    resume: bool = False,
    time_budget_hours: float | None = None,
) -> None:
    """Load data and model per the config, then fine-tune.

    Checkpoints are selected on a validation split, never on the test split.

    :param config_path: path to a YAML config file
    :param smoke_test: if True, train on a small slice of real data for one
        fast epoch, to catch runtime bugs (shape mismatches, bad field
        names) before committing to a real multi-hour run on the GPU machine
    :param train_on: "ragtruth", "psiloqa", or "both"
    :param run_name: checkpoint folder name under checkpoints/; defaults to train_on
    :param backbone: key of model.BACKBONES
    :param seed: overrides the config seed
    :param max_steps: run only this many optimizer steps on the longest
        examples, with no evaluation or saving. Tests the worst case for GPU
        memory and fp16 stability before a long run.
    :param resume: continue from the newest checkpoint in the run folder, if any
    :param time_budget_hours: stop at an epoch boundary rather than exceed this
    """
    config = load_config(config_path)
    model_id = BACKBONES[backbone]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = build_model(model_id)

    datasets = TRAIN_SOURCES[train_on]
    processed_dir = config["paths"]["processed_data_dir"]
    train_dataset = load_combined_split("train", datasets, processed_dir)
    eval_dataset = load_combined_split("val", datasets, processed_dir)
    print(f"Training {backbone} on {datasets}: {len(train_dataset)} train / {len(eval_dataset)} val examples")

    if smoke_test:
        # Sort by length and take the shortest examples, so the smoke test
        # stays fast on a CPU-only dev machine. This is still real data, not
        # synthetic - only the *selection* of examples is biased toward short
        # ones, purely to keep runtime down.
        train_dataset = train_dataset.map(lambda ex: {"length": len(ex["input_ids"])})
        train_dataset = train_dataset.sort("length").select(range(20)).remove_columns("length")
        eval_dataset = eval_dataset.map(lambda ex: {"length": len(ex["input_ids"])})
        eval_dataset = eval_dataset.sort("length").select(range(10)).remove_columns("length")

    if max_steps:
        # Longest examples first: if the worst batch fits in memory, so does
        # every other batch in the real run.
        needed = max_steps * config["training"]["batch_size"] * config["training"]["gradient_accumulation_steps"]
        train_dataset = train_dataset.map(lambda ex: {"length": len(ex["input_ids"])})
        train_dataset = train_dataset.sort("length", reverse=True).select(range(needed)).remove_columns("length")
        print(f"Memory check: {needed} longest examples, {max_steps} optimizer steps, no saving")

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    name = run_name or train_on
    output_dir = config["paths"]["checkpoints_dir"] / (f"smoke_test_{name}" if smoke_test else name)
    training_args = build_training_arguments(config, output_dir, smoke_test, seed=seed, max_steps=max_steps)

    callbacks = [StopOnNonFinite()]
    if time_budget_hours:
        callbacks.append(TimeBudget(time_budget_hours))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_token_metrics,
        callbacks=callbacks,
    )
    print("========== DEVICE INFO ==========")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
    print("Trainer device:", trainer.args.device)
    print("Trainer n_gpu:", trainer.args.n_gpu)
    print("Precision: bf16=%s fp16=%s" % (training_args.bf16, training_args.fp16))
    print("=================================")

    last_checkpoint = get_last_checkpoint(str(output_dir)) if resume and output_dir.exists() else None
    if last_checkpoint:
        print(f"Resuming from {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    if max_steps:
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        print(f"Memory check passed: peak GPU memory {peak:.1f} GB, loss stayed finite")
        return

    if not smoke_test:
        model.save_pretrained(output_dir / "best")
        tokenizer.save_pretrained(output_dir / "best")
        run_info = {
            "backbone": backbone,
            "model_id": model_id,
            "train_on": train_on,
            "seed": training_args.seed,
            "epochs_completed": trainer.state.epoch,
            "epochs_planned": config["training"]["epochs"],
            "best_metric": trainer.state.best_metric,
            "precision": "bf16" if training_args.bf16 else "fp16" if training_args.fp16 else "fp32",
            "config": str(config_path),
        }
        with open(output_dir / "best" / "run_info.json", "w", encoding="utf-8") as f:
            json.dump(run_info, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--train-on", choices=sorted(TRAIN_SOURCES), default="ragtruth")
    parser.add_argument("--run-name", default=None, help="checkpoint folder name; defaults to --train-on")
    parser.add_argument("--backbone", choices=sorted(BACKBONES), default="modernbert-base")
    parser.add_argument("--seed", type=int, default=None, help="overrides the config seed")
    parser.add_argument("--max-steps", type=int, default=None, help="memory/stability check on the longest examples")
    parser.add_argument("--resume", action="store_true", help="continue from the newest checkpoint of this run")
    parser.add_argument("--time-budget-hours", type=float, default=None, help="stop at an epoch boundary past this")
    args = parser.parse_args()

    run(
        config_path=args.config,
        smoke_test=args.smoke_test,
        train_on=args.train_on,
        run_name=args.run_name,
        backbone=args.backbone,
        seed=args.seed,
        max_steps=args.max_steps,
        resume=args.resume,
        time_budget_hours=args.time_budget_hours,
    )
