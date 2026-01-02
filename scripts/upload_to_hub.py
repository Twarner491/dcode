"""Upload models and dataset to HuggingFace Hub."""

import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

api = HfApi()

# SD-Gcode v2 model (correct training)
SD_GCODE_MODEL = "checkpoints/sd_gcode_v2/final"
SD_GCODE_REPO = "twarner/dcode-sd-gcode"

# Legacy diffusion model
DIFFUSION_MODEL = "checkpoints/latent_gcode/final"
DIFFUSION_REPO = "twarner/dcode-latent-gcode"

# Text-to-text model (legacy)
BEST_MODEL = "checkpoints/flan-t5-base_seed42/final"
MODEL_REPO = "twarner/dcode-flan-t5-base"

# Dataset
DATASET_REPO = "twarner/dcode-polargraph-gcode"

MODEL_CARD = """---
license: mit
language:
- en
library_name: transformers
tags:
- gcode
- polargraph
- pen-plotter
- text-to-gcode
- flan-t5
base_model: google/flan-t5-base
datasets:
- twarner/dcode-polargraph-gcode
pipeline_tag: text-generation
---

# dcode-flan-t5-base

Flan-T5-base finetuned to generate polargraph-compatible gcode from text prompts.

## Model Description

This model converts natural language descriptions into gcode for polargraph pen plotters. It was trained on 175,952 image-caption-gcode triplets generated from art images.

## Usage

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("twarner/dcode-flan-t5-base")
tokenizer = AutoTokenizer.from_pretrained("twarner/dcode-flan-t5-base")

prompt = "drawing of a cat"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.8)
gcode = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(gcode)
```

## Training

- **Base model**: google/flan-t5-base
- **Dataset**: 175,952 image-gcode pairs with BLIP captions
- **Epochs**: 20
- **Batch size**: 32
- **Learning rate**: 5e-5
- **Final loss**: 0.257
- **Hardware**: NVIDIA H100

## Other Variants Tested

| Model | Seed | Epochs | Loss |
|-------|------|--------|------|
| flan-t5-base | 42 | 20 | 0.257 (this model) |
| flan-t5-base | 353 | 10 | 0.314 |
| flan-t5-base | 9340 | 10 | 0.310 |
| gpt2 | 5409 | 5 | 0.363 |
| gpt2 | 6573 | 5 | 0.360 |

## Machine Limits

The generated gcode is validated for these polargraph bounds:
- X: -420.5 to 420.5 mm
- Y: -594.5 to 594.5 mm
- Pen up: 90 deg, Pen down: 40 deg

## License

MIT

## Citation

```bibtex
@misc{dcode2025,
  author = {Teddy Warner},
  title = {dcode: Text to Polargraph Gcode},
  year = {2025},
  url = {https://github.com/Twarner491/dcode}
}
```
"""

DATASET_CARD = """---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- gcode
- polargraph
- pen-plotter
- art
size_categories:
- 100K<n<1M
---

# dcode-polargraph-gcode

Dataset of art image captions paired with polargraph-compatible gcode for training text-to-gcode models.

## Dataset Description

175,952 samples of (caption, gcode) pairs generated from art images. Each image was processed through multiple gcode conversion algorithms (spiral, crosshatch, pulse, squares, wander) to create diverse training data.

## Dataset Structure

```json
{
  "prompt": "a painting of a woman in a blue dress",
  "gcode": "G0 X0.00 Y0.00 F1000\\nM280 P0 S40\\nG1 X10.50 Y20.30 F500\\n..."
}
```

### Fields

- `prompt`: BLIP-generated caption describing the source artwork
- `gcode`: Polargraph-compatible gcode with:
  - G0/G1 movement commands
  - M280 servo commands for pen up/down
  - Coordinates within machine bounds

## Source Data

Images sourced from [Art Images: Drawing/Painting/Sculptures/Engravings](https://www.kaggle.com/datasets/thedownhill/art-images-drawings-painting-sculpture-engraving) on Kaggle.

**Attribution**: Dataset compiled by [thedownhill](https://www.kaggle.com/thedownhill) on Kaggle, containing drawings, paintings, sculptures, and engravings from various art collections.

## Gcode Generation

Each image was converted using a custom Python converter (ported from JavaScript) with 5 algorithms:
- Spiral
- Crosshatch  
- Pulse
- Squares
- Wander

## Machine Specifications

Target machine: Polargraph pen plotter
- Work area: 841mm x 1189mm (A0)
- X bounds: -420.5 to 420.5 mm
- Y bounds: -594.5 to 594.5 mm
- Pen servo: 40 deg (down) to 90 deg (up)

## License

MIT (dataset compilation and gcode generation)

Source images: See original Kaggle dataset license.

## Citation

```bibtex
@misc{dcode-dataset2025,
  author = {Teddy Warner},
  title = {dcode Polargraph Gcode Dataset},
  year = {2025},
  url = {https://huggingface.co/datasets/twarner/dcode-polargraph-gcode}
}
```
"""


SD_GCODE_CARD = """---
license: mit
language:
- en
library_name: diffusers
tags:
- gcode
- polargraph
- pen-plotter
- text-to-gcode
- stable-diffusion
- diffusion
datasets:
- twarner/dcode-polargraph-gcode
pipeline_tag: text-generation
---

# dcode-sd-gcode

Text-to-gcode model using Stable Diffusion latents for polargraph pen plotters.

## Model Description

This model combines a frozen Stable Diffusion pipeline with a trained gcode decoder:
1. **Text -> Latent**: Stable Diffusion generates image latents from text
2. **Latent -> Gcode**: Custom transformer decoder converts latents to gcode

## Architecture

- **Base**: runwayml/stable-diffusion-v1-5 (frozen)
- **Gcode Decoder**: 6-layer transformer, 768-dim, 170M params
- **Max gcode length**: 1024 tokens

## Usage

```python
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer
import torch

# Load the model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
# Load trained decoder from this repo
# See spaces/dcode for full inference code
```

Or try the [HuggingFace Space](https://huggingface.co/spaces/twarner/dcode).

## Training

- **Dataset**: 175,952 VAE-encoded image latents with gcode
- **Epochs**: 10
- **Hardware**: NVIDIA H100 (80GB)
- **Method**: Decoder trained on deterministic VAE latents (not diffusion noise)

## Machine Limits

Generated gcode is validated for polargraph bounds:
- X: -420.5 to 420.5 mm
- Y: -594.5 to 594.5 mm
- Pen up: 90 deg, Pen down: 40 deg

## License

MIT
"""

DIFFUSION_CARD = """---
license: mit
language:
- en
library_name: transformers
tags:
- gcode
- polargraph
- pen-plotter
- text-to-gcode
- diffusion
- latent
datasets:
- twarner/dcode-polargraph-gcode
pipeline_tag: text-generation
---

# dcode-latent-gcode (legacy)

Latent-to-gcode transformer model. See [dcode-sd-gcode](https://huggingface.co/twarner/dcode-sd-gcode) for the improved version.
"""


def upload_sd_gcode():
    """Upload SD-Gcode v2 model to HuggingFace."""
    print(f"Uploading SD-Gcode model to {SD_GCODE_REPO}...")
    
    # Create repo
    create_repo(SD_GCODE_REPO, exist_ok=True, repo_type="model")
    
    # Write model card
    model_path = Path(SD_GCODE_MODEL)
    readme_path = model_path / "README.md"
    readme_path.write_text(SD_GCODE_CARD, encoding="utf-8")
    
    # Upload all files
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=SD_GCODE_REPO,
        repo_type="model",
    )
    print(f"SD-Gcode model uploaded: https://huggingface.co/{SD_GCODE_REPO}")


def upload_diffusion_model():
    """Upload legacy diffusion model to HuggingFace."""
    print(f"Uploading diffusion model to {DIFFUSION_REPO}...")
    
    # Create repo
    create_repo(DIFFUSION_REPO, exist_ok=True, repo_type="model")
    
    # Write model card
    model_path = Path(DIFFUSION_MODEL)
    readme_path = model_path / "README.md"
    readme_path.write_text(DIFFUSION_CARD, encoding="utf-8")
    
    # Upload all files
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=DIFFUSION_REPO,
        repo_type="model",
    )
    print(f"Diffusion model uploaded: https://huggingface.co/{DIFFUSION_REPO}")


def upload_model():
    """Upload best model to HuggingFace."""
    print(f"Uploading model to {MODEL_REPO}...")
    
    # Create repo
    create_repo(MODEL_REPO, exist_ok=True, repo_type="model")
    
    # Write model card
    model_path = Path(BEST_MODEL)
    readme_path = model_path / "README.md"
    readme_path.write_text(MODEL_CARD, encoding="utf-8")
    
    # Upload all files
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=MODEL_REPO,
        repo_type="model",
    )
    print(f"Model uploaded: https://huggingface.co/{MODEL_REPO}")


def upload_dataset():
    """Upload dataset to HuggingFace."""
    print(f"Uploading dataset to {DATASET_REPO}...")
    
    # Create repo
    create_repo(DATASET_REPO, exist_ok=True, repo_type="dataset")
    
    # Load manifest - format is {"pairs": [...], "stats": {...}}
    manifest_path = Path("data/processed/captioned.json")
    with open(manifest_path) as f:
        data = json.load(f)
    
    pairs = data.get("pairs", data)  # Handle both formats
    
    # Convert to JSONL for HF datasets
    jsonl_path = Path("data/processed/train.jsonl")
    count = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in pairs:
            gcode_path = Path(item["gcode"])
            if gcode_path.exists():
                try:
                    gcode = gcode_path.read_text(encoding="utf-8", errors="ignore")
                    # Use caption field, fall back to prompt
                    prompt = item.get("caption", item.get("prompt", "abstract art"))
                    record = {"prompt": prompt, "gcode": gcode}
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                except Exception as e:
                    print(f"Skipping {gcode_path}: {e}")
    
    print(f"Created {count} records in train.jsonl")
    
    # Write dataset card
    readme_path = Path("data/processed/README.md")
    readme_path.write_text(DATASET_CARD, encoding="utf-8")
    
    # Upload
    api.upload_file(
        path_or_fileobj=str(jsonl_path),
        path_in_repo="train.jsonl",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Dataset uploaded: https://huggingface.co/datasets/{DATASET_REPO}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "sd-gcode":
            upload_sd_gcode()
        elif sys.argv[1] == "model":
            upload_model()
        elif sys.argv[1] == "diffusion":
            upload_diffusion_model()
        elif sys.argv[1] == "dataset":
            upload_dataset()
        elif sys.argv[1] == "all":
            upload_sd_gcode()
            upload_model()
            upload_dataset()
    else:
        upload_sd_gcode()  # Default to v2 model
