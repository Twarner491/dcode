#!/usr/bin/env python3
"""Upload SD-Gcode model to HuggingFace Hub."""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

TOKEN = os.environ.get("HF_TOKEN")
CHECKPOINT_PATH = Path("checkpoints/sd_gcode/final")
REPO_ID = "twarner/dcode-sd-gcode"


def main():
    if not TOKEN:
        print("Set HF_TOKEN environment variable")
        return
    
    if not CHECKPOINT_PATH.exists():
        print(f"Checkpoint not found: {CHECKPOINT_PATH}")
        return
    
    api = HfApi(token=TOKEN)
    
    # Create repo
    try:
        create_repo(REPO_ID, token=TOKEN, exist_ok=True)
        print(f"Created/verified repo: {REPO_ID}")
    except Exception as e:
        print(f"Repo creation: {e}")
    
    # Upload all files
    print(f"Uploading {CHECKPOINT_PATH} to {REPO_ID}...")
    api.upload_folder(
        folder_path=str(CHECKPOINT_PATH),
        repo_id=REPO_ID,
        token=TOKEN,
    )
    
    # Create model card
    model_card = """---
license: mit
tags:
- gcode
- polargraph
- diffusion
- stable-diffusion
- text-to-gcode
pipeline_tag: text-to-image
datasets:
- twarner/dcode-polargraph-gcode
---

# dcode SD-Gcode

**Single end-to-end text→gcode diffusion model.**

This model post-trains Stable Diffusion 1.5 to generate polargraph-compatible gcode directly from text prompts. Instead of decoding to images, the diffusion latent is decoded to gcode tokens via a learned transformer decoder.

## Architecture

```
text prompt → CLIP text encoder → UNet diffusion → latent [4,64,64] → GcodeDecoder → gcode tokens
```

All components trained end-to-end:
- **Text Encoder**: CLIP from SD 1.5 (frozen initially, then finetuned)
- **UNet**: Stable Diffusion 1.5 UNet (finetuned)
- **Gcode Decoder**: 6-layer transformer decoder (trained from scratch)

## Training

- Dataset: 175,952 image-gcode pairs from art images
- Gcode generated via 5 algorithms: spiral, crosshatch, pulse, squares, wander
- 10 epochs on H100 GPU
- Loss: cross-entropy on gcode token prediction

## Usage

```python
from dcode.sd_gcode import SDGcodeModel

model = SDGcodeModel.from_pretrained("twarner/dcode-sd-gcode")
model.to("cuda")

gcode = model.generate("line drawing of a cat", num_diffusion_steps=20)
print(gcode)
```

## Machine Specs

- Work area: 841mm x 1189mm (A0)
- Bounds: X [-420.5, 420.5], Y [-594.5, 594.5]
- Pen servo: 40 deg (down), 90 deg (up)

## License

MIT

## Links

- [GitHub](https://github.com/Twarner491/dcode)
- [Dataset](https://huggingface.co/datasets/twarner/dcode-polargraph-gcode)
- [Space](https://huggingface.co/spaces/twarner/dcode)
"""
    
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        token=TOKEN,
    )
    
    print(f"Done! https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
