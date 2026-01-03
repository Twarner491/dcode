"""Dataset preparation for text-to-gcode training."""

import json
from pathlib import Path

import torch
from datasets import Dataset
from PIL import Image
from tqdm import tqdm


from .imagenet_classes import synset_to_name


def get_imagenet_class_name(folder_name: str) -> str:
    """Get human-readable class name from ImageNet folder (WordNet ID)."""
    return synset_to_name(folder_name)


def caption_imagenet_sketch(manifest_path: Path, output_path: Path) -> int:
    """Add captions to ImageNet-Sketch manifest using folder names as class labels.
    
    ImageNet-Sketch structure: sketch/{wordnet_id}/{image}.JPEG
    Caption = "a sketch of a {class_name}"
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    captioned = []
    
    for pair in tqdm(manifest["pairs"], desc="Captioning from folder names"):
        img_path = Path(pair.get("image", pair.get("source", "")))
        
        # Extract class from folder structure: .../sketch/n01440764/image.JPEG
        folder_name = img_path.parent.name
        class_name = get_imagenet_class_name(folder_name)
        
        # Create BLIP-style caption for consistency
        caption = f"a sketch of a {class_name}"
        
        captioned.append({**pair, "caption": caption})
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"pairs": captioned, "stats": manifest.get("stats", {})}, f, indent=2)
    
    return len(captioned)


def caption_images(manifest_path: Path, output_path: Path, device: str = "cuda"):
    """Generate BLIP captions for all images in manifest."""
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large",
        torch_dtype=torch.float16,
    ).to(device)

    with open(manifest_path) as f:
        manifest = json.load(f)

    captioned = []
    seen_sources = {}

    for pair in tqdm(manifest["pairs"], desc="Captioning"):
        source = pair.get("source", pair["image"])

        # Cache captions per source image
        if source not in seen_sources:
            img_path = Path(pair["image"])
            if img_path.exists():
                try:
                    image = Image.open(img_path).convert("RGB")
                    inputs = processor(image, return_tensors="pt").to(device, torch.float16)
                    out = model.generate(**inputs, max_new_tokens=50)
                    caption = processor.decode(out[0], skip_special_tokens=True)
                    seen_sources[source] = caption
                except Exception:
                    seen_sources[source] = "abstract art"
            else:
                seen_sources[source] = "abstract art"

        captioned.append({**pair, "caption": seen_sources[source]})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"pairs": captioned, "stats": manifest.get("stats", {})}, f, indent=2)

    return len(captioned)


def _load_single_pair(args):
    """Load single gcode file (for multiprocessing)."""
    pair, base_dir, max_lines, use_captions = args
    
    gcode_path = Path(pair["gcode"])
    if not gcode_path.is_absolute():
        gcode_path = base_dir / gcode_path.name
    if not gcode_path.exists():
        gcode_path = base_dir / "gcode" / Path(pair["gcode"]).name
    if not gcode_path.exists():
        return None

    try:
        gcode = gcode_path.read_text()
        lines = gcode.split("\n")[:max_lines]
        gcode = "\n".join(lines)

        if use_captions and "caption" in pair:
            prompt = pair["caption"]
        else:
            algo = pair.get("algorithm", "art")
            prompt = f"drawing in {algo} style"

        return {"prompt": prompt, "gcode": gcode}
    except Exception:
        return None


def load_dataset(
    manifest_path: Path,
    max_gcode_lines: int = 300,
    use_captions: bool = True,
    num_workers: int = 24,  # Use most CPUs
) -> Dataset:
    """Load manifest as HuggingFace Dataset with multiprocessing."""
    import multiprocessing as mp
    from functools import partial

    manifest_path = Path(manifest_path).resolve()
    base_dir = manifest_path.parent

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Prepare args for parallel loading
    args_list = [
        (pair, base_dir, max_gcode_lines, use_captions)
        for pair in manifest["pairs"]
    ]

    # Load in parallel
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_load_single_pair, args_list, chunksize=500),
            total=len(args_list),
            desc="Loading gcode",
        ))

    # Filter None and build dataset
    data = {"prompt": [], "gcode": []}
    for r in results:
        if r:
            data["prompt"].append(r["prompt"])
            data["gcode"].append(r["gcode"])

    return Dataset.from_dict(data)


def create_splits(
    manifest_path: Path,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Create train/eval splits."""
    dataset = load_dataset(manifest_path)
    splits = dataset.train_test_split(test_size=1 - train_ratio, seed=seed)
    return splits["train"], splits["test"]
