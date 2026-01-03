#!/usr/bin/env python3
"""Upload dcode dataset to HuggingFace with proper structure.

Structure:
    images/
        {synset_id}/           # ImageNet class folders
            {image}.jpg        # Original images (no duplicates)
    gcode/
        {synset_id}/           # ImageNet class folders
            {algorithm}/       # Algorithm subfolders
                {image}_{algo}_{idx}.gcode

Dataset card includes link to project writeup.
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dcode.imagenet_classes import synset_to_name


DATASET_CARD = '''---
license: mit
task_categories:
  - text-to-image
  - image-to-text
language:
  - en
tags:
  - gcode
  - cnc
  - plotter
  - polargraph
  - line-art
  - sketch
  - stable-diffusion
size_categories:
  - 100K<n<1M
---

# dcode: ImageNet-Sketch to G-code Dataset

A dataset of **ImageNet-Sketch images paired with generated G-code** for training text-to-gcode diffusion models.

## Overview

This dataset enables training models that convert text descriptions directly into G-code for CNC machines, plotters, and polargraph drawing robots.

| Feature | Value |
|---------|-------|
| Source Images | [ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch) |
| Classes | 1,000 ImageNet categories |
| Images | ~50,000 black/white sketches |
| G-code Files | ~200,000 (4 algorithms × images) |
| Algorithms | spiral, crosshatch, squares, trace |

## Structure

```
images/
    n01440764/              # ImageNet synset ID
        ILSVRC2012_val_00000293.JPEG
        ...
    n01443537/
        ...
gcode/
    n01440764/
        spiral/
            ILSVRC2012_val_00000293_spiral_0.gcode
            ILSVRC2012_val_00000293_spiral_1.gcode
        crosshatch/
            ...
        squares/
            ...
        trace/
            ...
```

## Algorithms

| Algorithm | Description |
|-----------|-------------|
| **spiral** | Concentric spiral from center, density varies with brightness |
| **crosshatch** | Multi-angle hatching lines at configurable angles |
| **squares** | Concentric squares sized by local brightness |
| **trace** | Binary edge detection with scan-line tracing |

## Usage

```python
from datasets import load_dataset

# Load the dataset
ds = load_dataset("twarner/dcode-imagenet-sketch")

# Access image and corresponding gcode
sample = ds["train"][0]
print(sample["image_path"])
print(sample["gcode_path"])
print(sample["caption"])  # "a sketch of a goldfish"
```

## Training

This dataset was used to train [dcode-sd-gcode-v3](https://huggingface.co/twarner/dcode-sd-gcode-v3), an end-to-end text-to-gcode diffusion model.

## Project

Full project documentation, hardware build guide, and interactive demo:

**🔗 [teddywarner.org/Projects/Polargraph/#dcode](https://teddywarner.org/Projects/Polargraph/#dcode)**

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


def upload_dataset(
    processed_dir: str = "data/processed",
    raw_sketch_dir: str = "data/raw/sketch",
    repo_id: str = "twarner/dcode-imagenet-sketch",
):
    """Upload dataset to HuggingFace with proper structure."""
    from huggingface_hub import HfApi, create_repo
    
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    
    # Create repo
    create_repo(repo_id, repo_type="dataset", exist_ok=True)
    print(f"Created/verified repo: {repo_id}")
    
    processed = Path(processed_dir)
    raw_sketch = Path(raw_sketch_dir)
    
    # Load manifest to understand structure
    manifest_path = processed / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    pairs = manifest.get("pairs", [])
    print(f"Found {len(pairs)} pairs in manifest")
    
    # Build structure: track unique images and their gcode files
    # images: synset -> set of original image paths
    # gcode: synset -> algorithm -> list of gcode paths
    images_by_synset = defaultdict(set)
    gcode_by_synset_algo = defaultdict(lambda: defaultdict(list))
    
    for pair in tqdm(pairs, desc="Analyzing structure"):
        gcode_path = Path(pair["gcode"])
        source = Path(pair.get("source", pair["image"]))
        algorithm = pair.get("algorithm", "unknown")
        
        # Extract synset from source path (e.g., .../sketch/n01440764/image.JPEG)
        synset = source.parent.name
        
        # Track unique source images
        if source.exists():
            images_by_synset[synset].add(source)
        elif (raw_sketch / synset / source.name).exists():
            images_by_synset[synset].add(raw_sketch / synset / source.name)
        
        # Track gcode files
        if gcode_path.exists():
            gcode_by_synset_algo[synset][algorithm].append(gcode_path)
    
    print(f"Found {len(images_by_synset)} synsets")
    print(f"Found {sum(len(imgs) for imgs in images_by_synset.values())} unique images")
    print(f"Found {sum(sum(len(g) for g in algos.values()) for algos in gcode_by_synset_algo.values())} gcode files")
    
    # Create temp upload directory
    upload_dir = Path("upload_temp")
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    
    images_dir = upload_dir / "images"
    gcode_dir = upload_dir / "gcode"
    
    # Copy images (no duplicates)
    print("Copying images...")
    for synset, image_paths in tqdm(images_by_synset.items(), desc="Images"):
        synset_dir = images_dir / synset
        synset_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in image_paths:
            dst = synset_dir / img_path.name
            if not dst.exists():
                shutil.copy2(img_path, dst)
    
    # Copy gcode with algorithm subfolders
    print("Copying gcode...")
    for synset, algos in tqdm(gcode_by_synset_algo.items(), desc="Gcode"):
        for algorithm, gcode_paths in algos.items():
            algo_dir = gcode_dir / synset / algorithm
            algo_dir.mkdir(parents=True, exist_ok=True)
            
            for gcode_path in gcode_paths:
                dst = algo_dir / gcode_path.name
                if not dst.exists():
                    shutil.copy2(gcode_path, dst)
    
    # Write dataset card
    (upload_dir / "README.md").write_text(DATASET_CARD)
    
    # Create metadata JSON
    metadata = {
        "synsets": {s: synset_to_name(s) for s in images_by_synset.keys()},
        "algorithms": ["spiral", "crosshatch", "squares", "trace"],
        "total_images": sum(len(imgs) for imgs in images_by_synset.values()),
        "total_gcode": sum(sum(len(g) for g in algos.values()) for algos in gcode_by_synset_algo.values()),
    }
    (upload_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    
    # Upload using large folder method
    print("Uploading to HuggingFace...")
    api.upload_large_folder(
        folder_path=str(upload_dir),
        repo_id=repo_id,
        repo_type="dataset",
        num_workers=8,
    )
    
    print(f"Done! https://huggingface.co/datasets/{repo_id}")
    
    # Cleanup
    shutil.rmtree(upload_dir)


if __name__ == "__main__":
    upload_dataset()
