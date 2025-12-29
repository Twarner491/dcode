# dcode

Text prompt → gcode via finetuned models. Experimental.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev,ray]"
huggingface-cli login
```

## Workflow

### 1. Generate Dataset

```bash
uv run --active python scripts/download_data.py
uv run --active python scripts/generate_dataset.py
```

### 2. Add Captions

```bash
uv run --active dcode caption -m data/processed/manifest.json -o data/processed/captioned.json
```

### 3. Train

```bash
# Single model
uv run --active dcode train -m data/processed/captioned.json --model flan-t5-base -e 10

# Sweep across models (flan-t5, gpt2, codegen)
uv run --active dcode sweep -m data/processed/captioned.json -t 8
```

### 4. Inference

```bash
uv run --active dcode infer "drawing of a goat" -m checkpoints/flan-t5-base_seed42/final -o goat.gcode
```

### 5. Validate

```bash
uv run --active dcode validate goat.gcode --fix --stats -o goat_safe.gcode
```

## Models

```bash
uv run --active dcode models
```

- `flan-t5-small/base/large` - seq2seq
- `gpt2/gpt2-medium` - causal
- `codegen-350m` - code-focused causal

## Config

`configs/machine.json` - your polargraph limits
