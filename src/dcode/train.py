"""Training pipeline for text-to-gcode models."""

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
)

from .dataset import load_dataset
from .models import ModelConfig, ModelType, get_model_config

# Optional imports
try:
    import ray
    from ray import tune

    HAS_RAY = True
except ImportError:
    HAS_RAY = False


@dataclass
class TrainConfig:
    model: str = "flan-t5-small"
    output_dir: str = "checkpoints"
    epochs: int = 10
    batch_size: int = 32  # H100 can handle large batches
    lr: float = 5e-5
    max_input_len: int = 128
    max_output_len: int = 512
    seed: int = 42
    gradient_accumulation: int = 2
    warmup_steps: int = 100
    fp16: bool = True
    logging_steps: int = 50
    save_steps: int = 1000
    eval_steps: int = 1000
    dataloader_workers: int = 8


class GcodeTrainer:
    """Trains models for text-to-gcode generation."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.model_config = get_model_config(config.model)
        self.tokenizer = None
        self.model = None

    def setup(self):
        """Load tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.hf_id)

        if self.model_config.type == ModelType.SEQ2SEQ:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_config.hf_id)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_config.hf_id)
            # Set pad token for causal models
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

    def preprocess_seq2seq(self, examples):
        """Preprocess for encoder-decoder models."""
        inputs = self.tokenizer(
            examples["prompt"],
            max_length=self.config.max_input_len,
            padding="max_length",
            truncation=True,
        )
        targets = self.tokenizer(
            text_target=examples["gcode"],
            max_length=self.config.max_output_len,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"],
        }

    def preprocess_causal(self, examples):
        """Preprocess for decoder-only models."""
        # Format: <prompt> [SEP] <gcode>
        texts = [
            f"{p}\n###\n{g}" for p, g in zip(examples["prompt"], examples["gcode"])
        ]
        tokenized = self.tokenizer(
            texts,
            max_length=self.config.max_input_len + self.config.max_output_len,
            padding="max_length",
            truncation=True,
        )
        # For causal LM, labels = input_ids (shifted internally)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def train(self, manifest_path: Path) -> Path:
        """Run training."""
        self.setup()
        torch.manual_seed(self.config.seed)

        # Load dataset
        dataset = load_dataset(manifest_path, max_gcode_lines=100)
        splits = dataset.train_test_split(test_size=0.1, seed=self.config.seed)

        # Preprocess with multiprocessing
        preprocess_fn = (
            self.preprocess_seq2seq
            if self.model_config.type == ModelType.SEQ2SEQ
            else self.preprocess_causal
        )
        tokenized = splits.map(
            preprocess_fn,
            batched=True,
            remove_columns=splits["train"].column_names,
            num_proc=8,
            batch_size=1000,
        )

        # Output directory
        output_dir = Path(self.config.output_dir) / f"{self.model_config.name}_seed{self.config.seed}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Training setup based on model type
        if self.model_config.type == ModelType.SEQ2SEQ:
            args = Seq2SeqTrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.config.epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.lr,
                warmup_steps=self.config.warmup_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_strategy="steps",
                eval_steps=self.config.eval_steps,
                save_total_limit=2,
                load_best_model_at_end=True,
                predict_with_generate=True,
                seed=self.config.seed,
                report_to="none",
                remove_unused_columns=False,
                dataloader_num_workers=self.config.dataloader_workers,
                dataloader_pin_memory=True,
                bf16=torch.cuda.is_available(),
            )
            data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=args,
                train_dataset=tokenized["train"],
                eval_dataset=tokenized["test"],
                data_collator=data_collator,
            )
        else:
            args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.config.epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.lr,
                warmup_steps=self.config.warmup_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation,
                logging_steps=self.config.logging_steps,
                save_steps=self.config.save_steps,
                eval_strategy="steps",
                eval_steps=self.config.eval_steps,
                save_total_limit=2,
                load_best_model_at_end=True,
                seed=self.config.seed,
                report_to="none",
                remove_unused_columns=False,
                dataloader_num_workers=self.config.dataloader_workers,
                dataloader_pin_memory=True,
                bf16=torch.cuda.is_available(),
            )
            data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
            trainer = Trainer(
                model=self.model,
                args=args,
                train_dataset=tokenized["train"],
                eval_dataset=tokenized["test"],
                data_collator=data_collator,
            )

        # Train
        trainer.train()

        # Save final
        final_path = output_dir / "final"
        trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))

        return final_path


def train_single(
    manifest_path: str,
    model: str = "flan-t5-small",
    output_dir: str = "checkpoints",
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 5e-5,
    seed: int = 42,
) -> Path:
    """Single training run."""
    config = TrainConfig(
        model=model,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )
    trainer = GcodeTrainer(config)
    return trainer.train(Path(manifest_path))


def run_sweep(manifest_path: str, num_trials: int = 8):
    """Hyperparameter sweep across models and configs."""
    if not HAS_RAY:
        raise RuntimeError("Ray required. Install: uv pip install 'dcode[ray]'")

    ray.init(ignore_reinit_error=True)
    abs_manifest = str(Path(manifest_path).resolve())
    abs_output = str(Path("checkpoints").resolve())

    def train_fn(config):
        result = train_single(
            manifest_path=abs_manifest,
            model=config["model"],
            output_dir=config["output_dir"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            seed=config["seed"],
        )
        return {"checkpoint": str(result)}

    search_space = {
        "model": tune.choice([
            "flan-t5-small",
            "flan-t5-base",
            "gpt2",
            "codegen-350m",
        ]),
        "lr": tune.loguniform(1e-5, 1e-4),
        "batch_size": tune.choice([16, 32, 64]),  # H100 batch sizes
        "seed": tune.randint(0, 10000),
        "epochs": tune.choice([5, 10]),
        "output_dir": abs_output,
    }

    analysis = tune.run(
        train_fn,
        config=search_space,
        num_samples=num_trials,
        resources_per_trial={"gpu": 1, "cpu": 4},
    )

    return analysis.best_config
