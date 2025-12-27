"""Command-line interface."""

from pathlib import Path

import click

from .config import Config
from .data import DataGenerator
from .inference import GcodeGenerator
from .train import GcodeTrainer, run_sweep
from .validator import GcodeValidator


@click.group()
def main():
    """dcode - Diffusion models for gcode generation."""
    pass


@main.command()
@click.option("--input", "-i", "input_dir", required=True, type=Path, help="Input images directory")
@click.option("--output", "-o", "output_dir", required=True, type=Path, help="Output dataset directory")
@click.option("--max-samples", "-n", type=int, help="Max samples to generate")
def generate(input_dir: Path, output_dir: Path, max_samples: int | None):
    """Generate training dataset from images."""
    gen = DataGenerator()
    manifest = gen.generate_dataset(input_dir, output_dir, max_samples)
    click.echo(f"Generated {len(manifest['pairs'])} pairs, {len(manifest['failed'])} failed")


@main.command()
@click.option("--manifest", "-m", required=True, type=Path, help="Dataset manifest path")
@click.option("--epochs", "-e", default=10, type=int)
@click.option("--batch-size", "-b", default=4, type=int)
@click.option("--lr", default=1e-5, type=float)
@click.option("--seed", "-s", default=42, type=int)
def train(manifest: Path, epochs: int, batch_size: int, lr: float, seed: int):
    """Train model on dataset."""
    trainer = GcodeTrainer()
    dataset = trainer.load_dataset(manifest)
    checkpoint = trainer.train(dataset, epochs, batch_size, lr, seed)
    click.echo(f"Saved checkpoint to {checkpoint}")


@main.command()
@click.option("--manifest", "-m", required=True, type=Path, help="Dataset manifest path")
@click.option("--trials", "-t", default=4, type=int, help="Number of trials")
def sweep(manifest: Path, trials: int):
    """Run hyperparameter sweep."""
    best = run_sweep(str(manifest), trials)
    click.echo(f"Best config: {best}")


@main.command()
@click.argument("prompt")
@click.option("--model", "-m", type=Path, help="Model checkpoint path")
@click.option("--output", "-o", type=Path, help="Output gcode file")
@click.option("--steps", "-s", default=50, type=int, help="Inference steps")
def infer(prompt: str, model: Path | None, output: Path | None, steps: int):
    """Generate gcode from prompt."""
    gen = GcodeGenerator(model_path=model)
    gcode = gen.generate(prompt, num_inference_steps=steps)

    if output:
        output.write_text(gcode)
        click.echo(f"Saved to {output}")
    else:
        click.echo(gcode)


@main.command()
@click.argument("gcode_file", type=Path)
@click.option("--fix", is_flag=True, help="Auto-fix issues")
@click.option("--output", "-o", type=Path, help="Output fixed gcode")
def validate(gcode_file: Path, fix: bool, output: Path | None):
    """Validate gcode for machine limits."""
    config = Config.load()
    validator = GcodeValidator(config)

    gcode = gcode_file.read_text()
    result = validator.validate(gcode, auto_correct=fix)

    if result.errors:
        click.echo("Errors:")
        for e in result.errors:
            click.echo(f"  {e}")

    if result.warnings:
        click.echo("Warnings:")
        for w in result.warnings:
            click.echo(f"  {w}")

    if result.valid:
        click.echo("Valid")
    elif fix and result.corrected_gcode:
        if output:
            output.write_text(result.corrected_gcode)
            click.echo(f"Fixed and saved to {output}")
        else:
            click.echo(result.corrected_gcode)


if __name__ == "__main__":
    main()

