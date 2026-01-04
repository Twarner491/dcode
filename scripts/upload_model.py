#!/usr/bin/env python3
"""Upload dcode model to HuggingFace with comprehensive model card."""

import os
import shutil
from pathlib import Path


MODEL_CARD = '''---
license: mit
library_name: diffusers
pipeline_tag: text-to-image
tags:
  - gcode
  - cnc
  - plotter
  - polargraph
  - stable-diffusion
  - text-to-gcode
  - diffusion
base_model: runwayml/stable-diffusion-v1-5
datasets:
  - twarner/dcode-imagenet-sketch
---

# dcode: Text-to-Gcode Diffusion Model

An end-to-end diffusion model that converts **text prompts directly into G-code** for CNC machines, plotters, and polargraph drawing robots.

## Overview

dcode is a fine-tuned Stable Diffusion model with a custom G-code decoder head. It takes a text description (e.g., "a sketch of a horse") and outputs machine-executable G-code.

| Component | Description |
|-----------|-------------|
| Base Model | [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
| Decoder | 200M param transformer (12 layers, 1024 hidden, 16 heads) |
| Tokenizer | Custom BPE tokenizer for G-code |
| Training Data | [dcode-imagenet-sketch](https://huggingface.co/datasets/twarner/dcode-imagenet-sketch) |

## Architecture

```
Text Prompt
    ↓
[CLIP Text Encoder] ← frozen
    ↓
[UNet Diffusion] ← frozen
    ↓
Latent (4×64×64)
    ↓
[CNN Projector] ← trained
    ↓
[Transformer Decoder] ← trained
    ↓
G-code Tokens
    ↓
G-code Text
```

## Usage

### With Diffusers

```python
import torch
from diffusers import StableDiffusionPipeline
from huggingface_hub import hf_hub_download
from transformers import PreTrainedTokenizerFast

# Load components
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Download decoder weights
weights = hf_hub_download("twarner/dcode-sd-gcode-v3", "pytorch_model.bin")
tokenizer_path = hf_hub_download("twarner/dcode-sd-gcode-v3", "gcode_tokenizer/tokenizer.json")

# Load custom gcode tokenizer
gcode_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)

# Generate latent from text
with torch.no_grad():
    latent = pipe("a sketch of a horse", output_type="latent").images

# ... decode with GcodeDecoderV3 (see repo for full inference code)
```

### Interactive Demo

Try the model live: **[huggingface.co/spaces/twarner/dcode](https://huggingface.co/spaces/twarner/dcode)**

## Training

- **Dataset**: 50,000 ImageNet-Sketch images → 200,000 G-code files
- **Hardware**: 8× NVIDIA H100 80GB
- **Epochs**: 50
- **Batch Size**: 256 effective (32 × 8 GPUs)
- **Learning Rate**: 1e-4 with cosine schedule
- **Regularization**: Label smoothing (0.1), weight decay (0.05)

## G-code Output

The model generates G-code compatible with:
- Polargraph/drawbot machines
- Pen plotters
- Any G-code compatible CNC

Example output:
```gcode
G21 ; mm
G90 ; absolute
M280 P0 S90 ; pen up
G28 ; home

G0 X-200.00 Y100.00 F1000
M280 P0 S40 ; pen down
G1 X-180.00 Y120.00 F500
G1 X-160.00 Y115.00 F500
...
```

## Machine Specs

Default work area (configurable):
- Width: 841mm
- Height: 1189mm (A0 paper)
- Pen servo: 40° down, 90° up

## Project

Full project documentation, hardware build guide, and source code:

**🔗 [teddywarner.org/Projects/Polargraph/#dcode](https://teddywarner.org/Projects/Polargraph/#dcode)**

**GitHub**: [github.com/Twarner491/dcode](https://github.com/Twarner491/dcode)

## Citation

```bibtex
@misc{dcode2024,
  author = {Teddy Warner},
  title = {dcode: Text-to-Gcode Diffusion Model},
  year = {2024},
  url = {https://teddywarner.org/Projects/Polargraph/#dcode}
}
```

## License

MIT License
'''


def upload_model(
    checkpoint_dir: str = "checkpoints/sd_gcode_v3/final",
    repo_id: str = "twarner/dcode-sd-gcode-v3",
):
    """Upload trained model to HuggingFace."""
    from huggingface_hub import HfApi, create_repo
    
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    
    # Create repo
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"Created/verified repo: {repo_id}")
    
    checkpoint = Path(checkpoint_dir)
    
    if not checkpoint.exists():
        # Try fallback paths
        fallbacks = [
            Path("checkpoints/sd_gcode_v3/final"),
            Path("checkpoints/sd_gcode_v3"),
            Path("checkpoints/sd_gcode_v3_imagenet/final"),
            Path("checkpoints/sd_gcode_v3_imagenet"),
        ]
        for fb in fallbacks:
            if fb.exists():
                checkpoint = fb
                print(f"Using checkpoint: {checkpoint}")
                break
        else:
            print(f"Checkpoint not found: {checkpoint_dir}")
            print(f"Also tried: {[str(f) for f in fallbacks]}")
            return
    
    # Create temp upload directory
    upload_dir = Path("upload_model_temp")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    upload_dir.mkdir()
    
    # Copy model files
    for f in checkpoint.iterdir():
        if f.is_file():
            shutil.copy2(f, upload_dir / f.name)
        elif f.is_dir():
            shutil.copytree(f, upload_dir / f.name)
    
    # Write model card
    (upload_dir / "README.md").write_text(MODEL_CARD)
    
    # Upload
    print("Uploading to HuggingFace...")
    api.upload_folder(
        folder_path=str(upload_dir),
        repo_id=repo_id,
        repo_type="model",
    )
    
    print(f"Done! https://huggingface.co/{repo_id}")
    
    # Cleanup
    shutil.rmtree(upload_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload dcode model to HuggingFace")
    parser.add_argument("--checkpoint", "-c", default="checkpoints/sd_gcode_v3/final",
                        help="Path to checkpoint directory")
    parser.add_argument("--repo", "-r", default="twarner/dcode-sd-gcode-v3",
                        help="HuggingFace repo ID")
    args = parser.parse_args()
    upload_model(checkpoint_dir=args.checkpoint, repo_id=args.repo)
