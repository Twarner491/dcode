# dcode

Diffusion models for gcode generation.

## Setup

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

# or w/ ray
uv pip install -e ".[dev,ray]"
```

## Usage

### 1. Download Data

```bash
# Requires ~/.kaggle/kaggle.json
uv run --active python scripts/download_data.py
```

### 2. Generate Dataset

Generates multiple algorithm/setting permutations per image:

```bash
uv run --active python scripts/generate_dataset.py -n 1000  # limit to 1000 images
```

Algorithms: spiral, crosshatch, pulse, squares, wander (14 total permutations per image)

### 3. Train

```bash
uv run --active dcode train -m data/processed/manifest.json -e 10 -s 42
uv run --active dcode sweep -m data/processed/manifest.json -t 8
```

### 4. Inference

```bash
uv run --active dcode infer "abstract spiral pattern" -o output.gcode
```

### 5. Validate

```bash
uv run --active dcode validate output.gcode --fix -o fixed.gcode
```

## Config

Edit `configs/machine.json` for your plotter limits.
