"""Dataset generation: images -> gcode pairs with multiprocessing."""

import json
import multiprocessing as mp
from functools import partial
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .config import Config
from .converter import ALGORITHM_PERMUTATIONS, ImageConverter
from .gcode import GcodeExporter


def _process_image(args, config_dict, output_dir):
    """Process single image with all algorithms (worker function)."""
    img_path, algorithms = args
    config = Config(**config_dict)
    converter = ImageConverter(config)
    exporter = GcodeExporter(config)

    images_dir = Path(output_dir) / "images"
    gcode_dir = Path(output_dir) / "gcode"

    results = []
    stem = img_path.stem

    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((512, 512), Image.Resampling.LANCZOS)
    except Exception as e:
        return [], [{"path": str(img_path), "error": str(e)}]

    for algo in algorithms:
        permutations = ALGORITHM_PERMUTATIONS.get(algo, [{}])
        for perm_idx, options in enumerate(permutations):
            suffix = f"{algo}_{perm_idx}"
            out_img = images_dir / f"{stem}_{suffix}.png"
            out_gcode = gcode_dir / f"{stem}_{suffix}.gcode"

            try:
                img.save(out_img)
                turtle = converter.convert(img, algo, options)
                # No metadata comment - keep gcode clean for training
                gcode = exporter.export(turtle, comment="")
                out_gcode.write_text(gcode)

                results.append({
                    "image": str(out_img),
                    "gcode": str(out_gcode),
                    "algorithm": algo,
                    "options": options,
                    "source": str(img_path),
                })
            except Exception:
                pass

    return results, []


class DataGenerator:
    """Generates image-gcode pairs using multiple algorithms (parallelized)."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()

    def generate_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        max_samples: int | None = None,
        algorithms: list[str] | None = None,
        num_workers: int | None = None,
    ) -> dict:
        """Generate dataset with multiprocessing."""
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        gcode_dir = output_dir / "gcode"
        images_dir.mkdir(exist_ok=True)
        gcode_dir.mkdir(exist_ok=True)

        # Find images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(input_dir.glob(f"**/{ext}"))

        if max_samples:
            image_files = image_files[:max_samples]

        algos = algorithms or list(ALGORITHM_PERMUTATIONS.keys())
        num_workers = num_workers or mp.cpu_count()

        # Prepare args
        args_list = [(img, algos) for img in image_files]
        config_dict = self.config.model_dump()

        # Process in parallel
        worker_fn = partial(_process_image, config_dict=config_dict, output_dir=str(output_dir))

        all_pairs = []
        all_failed = []

        with mp.Pool(num_workers) as pool:
            results = list(tqdm(
                pool.imap(worker_fn, args_list),
                total=len(args_list),
                desc=f"Converting ({num_workers} workers)",
            ))

        for pairs, failed in results:
            all_pairs.extend(pairs)
            all_failed.extend(failed)

        # Build manifest
        manifest = {
            "pairs": all_pairs,
            "failed": all_failed,
            "stats": {
                "total_pairs": len(all_pairs),
                "total_failed": len(all_failed),
                "algorithms": {},
            },
        }
        for p in all_pairs:
            algo = p["algorithm"]
            manifest["stats"]["algorithms"][algo] = manifest["stats"]["algorithms"].get(algo, 0) + 1

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest
