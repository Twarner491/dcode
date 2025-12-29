"""CLI for dcode."""

from pathlib import Path

import click


@click.group()
def main():
    """dcode - Text to gcode via diffusion."""
    pass


@main.command()
@click.option("--input", "-i", "input_dir", required=True, type=Path)
@click.option("--output", "-o", "output_dir", required=True, type=Path)
@click.option("--max-samples", "-n", type=int)
@click.option("--algorithms", "-a", multiple=True)
def generate(input_dir: Path, output_dir: Path, max_samples: int | None, algorithms):
    """Generate image→gcode dataset."""
    from .data import DataGenerator

    gen = DataGenerator()
    manifest = gen.generate_dataset(
        input_dir, output_dir, max_samples, list(algorithms) or None
    )
    click.echo(f"Generated {manifest['stats']['total_pairs']} pairs")


@main.command()
@click.option("--manifest", "-m", required=True, type=Path)
@click.option("--output", "-o", required=True, type=Path)
def caption(manifest: Path, output: Path):
    """Add BLIP captions to dataset."""
    from .dataset import caption_images

    n = caption_images(manifest, output)
    click.echo(f"Captioned {n} images → {output}")


@main.command()
@click.option("--manifest", "-m", required=True, type=Path)
@click.option("--model", default="flan-t5-small", help="Model name or path")
@click.option("--epochs", "-e", default=10, type=int)
@click.option("--batch-size", "-b", default=4, type=int)
@click.option("--lr", default=5e-5, type=float)
@click.option("--seed", "-s", default=42, type=int)
@click.option("--output-dir", "-o", default="checkpoints", type=Path)
def train(manifest: Path, model: str, epochs: int, batch_size: int, lr: float, seed: int, output_dir: Path):
    """Train text→gcode model."""
    from .train import train_single

    result = train_single(
        manifest_path=str(manifest),
        model=model,
        output_dir=str(output_dir),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
    )
    click.echo(f"Saved: {result}")


@main.command()
@click.option("--manifest", "-m", required=True, type=Path)
@click.option("--trials", "-t", default=8, type=int)
def sweep(manifest: Path, trials: int):
    """Hyperparameter sweep with Ray."""
    from .train import run_sweep

    best = run_sweep(str(manifest), trials)
    click.echo(f"Best: {best}")


@main.command()
@click.argument("prompt")
@click.option("--model", "-m", required=True, type=Path, help="Trained model path")
@click.option("--output", "-o", type=Path)
@click.option("--temperature", "-t", default=0.8, type=float)
def infer(prompt: str, model: Path, output: Path | None, temperature: float):
    """Generate gcode from prompt."""
    from .inference import GcodeGenerator

    gen = GcodeGenerator(model)
    gcode = gen.generate(prompt, temperature=temperature)

    if output:
        output.write_text(gcode)
        click.echo(f"Saved: {output}")
    else:
        click.echo(gcode)


@main.command()
@click.argument("gcode_file", type=Path)
@click.option("--fix", is_flag=True)
@click.option("--output", "-o", type=Path)
@click.option("--stats", is_flag=True)
def validate(gcode_file: Path, fix: bool, output: Path | None, stats: bool):
    """Validate gcode for machine limits."""
    from .config import Config
    from .validator import GcodeValidator

    validator = GcodeValidator(Config.load())
    result = validator.validate(gcode_file.read_text(), auto_correct=fix)

    if result.errors:
        click.echo(click.style(f"Errors: {len(result.errors)}", fg="red"))
        for e in result.errors[:10]:
            click.echo(f"  {e}")

    if result.warnings:
        click.echo(click.style(f"Warnings: {len(result.warnings)}", fg="yellow"))

    if stats:
        click.echo(f"Stats: {result.stats}")

    if result.valid:
        click.echo(click.style("OK", fg="green"))
    elif fix and output:
        output.write_text(result.corrected_gcode)
        click.echo(f"Fixed → {output}")


@main.command()
def models():
    """List available models."""
    from .models import MODELS

    for name, cfg in MODELS.items():
        click.echo(f"{name}: {cfg.hf_id} ({cfg.type.value})")


if __name__ == "__main__":
    main()
