# dcode

Text prompt → polargraph gcode via Stable Diffusion + trained decoder.

## Live Demo

[HuggingFace Space](https://huggingface.co/spaces/twarner/dcode) | [Model](https://huggingface.co/twarner/dcode-sd-gcode) | [Dataset](https://huggingface.co/datasets/twarner/dcode-polargraph-gcode)

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev,ray]"
huggingface-cli login
```

## Quick Start

### Train SD-Gcode v3 (recommended)

Single GPU:
```bash
uv run --active dcode train-sd-gcode-v3 \
  -m data/processed/captioned.json \
  -e 20 \
  -b 16 \
  --grad-accum 2 \
  --lr 3e-4
```

Multi-GPU (2x/4x/8x H100s):
```bash
torchrun --nproc_per_node=4 \
  -m dcode.train_sd_gcode_v3 \
  data/processed/captioned.json
```

### Inference

```bash
uv run --active dcode infer-sd-gcode "line drawing of a horse" \
  -m checkpoints/sd_gcode_v3/final \
  -o horse.gcode
```

## Training Options

```bash
uv run --active dcode train-sd-gcode-v3 --help
```

Key options:
- `--epochs`: More epochs for larger model (default: 20)
- `--batch-size`: Per-GPU batch (16 fits H100 80GB)
- `--grad-accum`: Gradient accumulation steps
- `--lr`: Learning rate (3e-4 with warmup works well)
- `--max-len`: Max gcode tokens (default: 2048)
- `--text-latents`: Generate text-derived latents for alignment (slower but better)
- `--num-gpus`: Number of GPUs (auto-detect if not set)

## Architecture (v3)

```
Text → SD Text Encoder → SD UNet (diffusion) → Latent (4×64×64)
                                                    ↓
                                           CNN Projector → 16 memory tokens
                                                    ↓
                                           Transformer Decoder (12 layers, 1024-dim)
                                                    ↓
                                           Gcode Tokens → Gcode
```

- **SD Components**: Frozen (pretrained)
- **Decoder**: Trained (~200M params)
- **Tokenizer**: Custom gcode-specific (8192 vocab, preserves newlines)

## Dataset Generation

```bash
# Download art images from Kaggle
uv run --active python scripts/download_data.py

# Generate gcode from images (5 algorithms)
uv run --active python scripts/generate_dataset.py

# Add BLIP captions
uv run --active dcode caption -m data/processed/manifest.json -o data/processed/captioned.json
```

## Upload to HuggingFace

```bash
export HF_TOKEN="your_token"

# Upload v3 model
uv run --active python scripts/upload_to_hub.py sd-gcode-v3

# Upload dataset
uv run --active python scripts/upload_to_hub.py dataset
```

## Legacy Models

For text-to-text models (Flan-T5, GPT-2):

```bash
uv run --active dcode train -m data/processed/captioned.json --model flan-t5-base -e 10
uv run --active dcode infer "drawing of a goat" -m checkpoints/flan-t5-base_seed42/final -o goat.gcode
```

## Machine Config

`configs/machine.json` - polargraph limits:
- Work area: 841×1189mm (A0)
- X: -420.5 to 420.5 mm
- Y: -594.5 to 594.5 mm
- Pen servo: 40° (down) to 90° (up)

## License

MIT
