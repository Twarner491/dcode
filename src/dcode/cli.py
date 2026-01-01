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


@main.command("train-diffusion")
@click.option("--manifest", "-m", required=True, type=Path)
@click.option("--output", "-o", default="checkpoints/latent_gcode", type=Path)
@click.option("--epochs", "-e", default=10, type=int)
@click.option("--batch-size", "-b", default=4, type=int)
@click.option("--lr", default=1e-4, type=float)
@click.option("--max-len", default=1024, type=int)  # Reduced for memory
@click.option("--grad-accum", default=4, type=int)
def train_diffusion(manifest: Path, output: Path, epochs: int, batch_size: int, lr: float, max_len: int, grad_accum: int):
    """Train latent-to-gcode diffusion model."""
    from .train_diffusion import train

    result = train(
        manifest_path=str(manifest),
        output_dir=str(output),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        max_gcode_len=max_len,
        gradient_accumulation=grad_accum,
    )
    click.echo(f"Saved: {result}")


@main.command("infer-diffusion")
@click.argument("prompt")
@click.option("--model", "-m", required=True, type=Path, help="Trained latent-gcode model")
@click.option("--sd-model", default="stabilityai/stable-diffusion-2-1-base")
@click.option("--output", "-o", type=Path)
@click.option("--image-output", type=Path)
@click.option("--temperature", "-t", default=0.8, type=float)
@click.option("--steps", default=30, type=int)
@click.option("--seed", "-s", type=int)
def infer_diffusion(prompt: str, model: Path, sd_model: str, output: Path | None, image_output: Path | None, temperature: float, steps: int, seed: int | None):
    """Generate gcode from text using diffusion model."""
    from .inference_diffusion import DcodeInference

    inferencer = DcodeInference(str(model), sd_model)
    gcode, image = inferencer.generate(
        prompt,
        num_inference_steps=steps,
        temperature=temperature,
        seed=seed,
    )

    if output:
        output.write_text(gcode)
        click.echo(f"Gcode saved: {output}")
    else:
        click.echo(gcode[:1000])

    if image_output:
        image.save(image_output)
        click.echo(f"Image saved: {image_output}")


@main.command("train-sd-gcode")
@click.option("--manifest", "-m", required=True, type=Path)
@click.option("--output", "-o", default="checkpoints/sd_gcode", type=Path)
@click.option("--sd-model", default="runwayml/stable-diffusion-v1-5")
@click.option("--epochs", "-e", default=10, type=int)
@click.option("--batch-size", "-b", default=2, type=int)
@click.option("--grad-accum", default=16, type=int)
@click.option("--lr", default=1e-5, type=float)
@click.option("--max-len", default=512, type=int)
@click.option("--diffusion-steps", default=10, type=int)
@click.option("--seed", "-s", default=42, type=int)
def train_sd_gcode(
    manifest: Path, output: Path, sd_model: str, epochs: int, 
    batch_size: int, grad_accum: int, lr: float, max_len: int, 
    diffusion_steps: int, seed: int
):
    """Post-train Stable Diffusion for text→gcode (single model, end-to-end)."""
    from .train_sd_gcode import train

    result = train(
        manifest_path=str(manifest),
        output_dir=str(output),
        sd_model=sd_model,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation=grad_accum,
        learning_rate=lr,
        max_gcode_len=max_len,
        num_diffusion_steps=diffusion_steps,
        seed=seed,
    )
    click.echo(f"Saved: {result}")


@main.command("infer-sd-gcode")
@click.argument("prompt")
@click.option("--model", "-m", required=True, type=Path, help="Trained SD-Gcode model")
@click.option("--output", "-o", type=Path)
@click.option("--temperature", "-t", default=0.8, type=float)
@click.option("--steps", default=20, type=int)
@click.option("--max-len", default=512, type=int)
def infer_sd_gcode(prompt: str, model: Path, output: Path | None, temperature: float, steps: int, max_len: int):
    """Generate gcode from text using post-trained SD model."""
    from .sd_gcode import SDGcodeModel

    model_obj = SDGcodeModel.from_pretrained(str(model))
    model_obj.eval()
    
    gcode = model_obj.generate(
        prompt,
        num_diffusion_steps=steps,
        max_gcode_length=max_len,
        temperature=temperature,
    )

    if output:
        output.write_text(gcode)
        click.echo(f"Saved: {output}")
    else:
        click.echo(gcode)


@main.command("train-sd-gcode-v2")
@click.option("--manifest", "-m", required=True, type=Path)
@click.option("--output", "-o", default="checkpoints/sd_gcode_v2", type=Path)
@click.option("--sd-model", default="runwayml/stable-diffusion-v1-5")
@click.option("--epochs", "-e", default=10, type=int)
@click.option("--batch-size", "-b", default=8, type=int)
@click.option("--grad-accum", default=4, type=int)
@click.option("--lr", default=1e-4, type=float)
@click.option("--max-len", default=1024, type=int)
def train_sd_gcode_v2(
    manifest: Path, output: Path, sd_model: str, epochs: int,
    batch_size: int, grad_accum: int, lr: float, max_len: int
):
    """Train decoder on VAE-encoded image latents (CORRECT approach).
    
    Uses deterministic image→latent encoding instead of random diffusion.
    """
    from .train_sd_gcode_v2 import train

    result = train(
        manifest_path=str(manifest),
        output_dir=str(output),
        sd_model_id=sd_model,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation=grad_accum,
        learning_rate=lr,
        max_gcode_len=max_len,
    )
    click.echo(f"Saved: {result}")


if __name__ == "__main__":
    main()
