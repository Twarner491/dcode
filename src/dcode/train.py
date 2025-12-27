"""Training pipeline for gcode diffusion models."""

import json
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from diffusers import StableDiffusionPipeline

# Ray is optional (Linux only)
try:
    import ray
    from ray import tune
    from ray.tune.integration.wandb import WandbLoggerCallback
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


class GcodeTrainer:
    """Finetunes diffusion models for gcode generation."""

    def __init__(
        self,
        model_name: str = "stabilityai/stable-diffusion-2-1-base",
        output_dir: Path = Path("checkpoints"),
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_dataset(self, manifest_path: Path) -> Dataset:
        """Load dataset from manifest."""
        with open(manifest_path) as f:
            manifest = json.load(f)

        data = {"image_path": [], "gcode": [], "algorithm": []}
        for pair in manifest["pairs"]:
            gcode_path = Path(pair["gcode"])
            if gcode_path.exists():
                data["image_path"].append(pair["image"])
                data["algorithm"].append(pair.get("algorithm", "unknown"))
                with open(gcode_path) as f:
                    data["gcode"].append(f.read())

        return Dataset.from_dict(data)

    def train(
        self,
        dataset: Dataset,
        epochs: int = 10,
        batch_size: int = 4,
        lr: float = 1e-5,
        seed: int = 42,
    ):
        """Train model on dataset."""
        wandb.init(project="dcode", config={"epochs": epochs, "batch_size": batch_size, "lr": lr})

        torch.manual_seed(seed)

        # Load base model
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
        )

        if torch.cuda.is_available():
            pipe = pipe.to("cuda")

        # NOTE: Full training loop would involve:
        # 1. Tokenizing gcode as target sequences
        # 2. Encoding images as conditioning
        # 3. Training diffusion model to generate gcode tokens
        # This is a skeleton - actual implementation depends on chosen architecture

        checkpoint_path = self.output_dir / f"checkpoint_seed{seed}"
        checkpoint_path.mkdir(exist_ok=True)

        wandb.finish()
        return checkpoint_path


def run_sweep(manifest_path: str, num_trials: int = 4):
    """Run hyperparameter sweep with Ray Tune."""
    if not HAS_RAY:
        raise RuntimeError("Ray not available. Install with: uv pip install 'dcode[ray]'")

    ray.init(ignore_reinit_error=True)

    config = {
        "lr": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([2, 4, 8]),
        "seed": tune.randint(0, 10000),
        "epochs": tune.choice([5, 10, 20]),
    }

    def train_fn(config):
        trainer = GcodeTrainer()
        dataset = trainer.load_dataset(Path(manifest_path))
        trainer.train(
            dataset,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            lr=config["lr"],
            seed=config["seed"],
        )
        return {"loss": 0.0}  # Placeholder

    analysis = tune.run(
        train_fn,
        config=config,
        num_samples=num_trials,
        callbacks=[WandbLoggerCallback(project="dcode")],
        resources_per_trial={"gpu": 1},
    )

    return analysis.best_config
