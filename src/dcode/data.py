"""Dataset generation: images -> gcode pairs with multiple algorithm permutations."""

import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .config import Config
from .converter import ALGORITHM_PERMUTATIONS, ImageConverter
from .gcode import GcodeExporter


class DataGenerator:
    """Generates image-gcode pairs using multiple algorithms and settings."""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.converter = ImageConverter(self.config)
        self.exporter = GcodeExporter(self.config)

    def generate_dataset(
        self,
        input_dir: Path,
        output_dir: Path,
        max_samples: int | None = None,
        algorithms: list[str] | None = None,
    ) -> dict:
        """Generate dataset with multiple permutations per image."""
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        gcode_dir = output_dir / "gcode"
        images_dir.mkdir(exist_ok=True)
        gcode_dir.mkdir(exist_ok=True)

        # Find all images
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_files.extend(input_dir.glob(f"**/{ext}"))

        if max_samples:
            image_files = image_files[:max_samples]

        # Use specified algorithms or all
        algos = algorithms or list(ALGORITHM_PERMUTATIONS.keys())

        manifest = {"pairs": [], "failed": [], "stats": {"algorithms": {}}}

        for img_path in tqdm(image_files, desc="Converting"):
            try:
                img = Image.open(img_path).convert("RGB")
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            except Exception as e:
                manifest["failed"].append({"path": str(img_path), "error": str(e)})
                continue

            stem = img_path.stem

            # Generate all algorithm/setting permutations
            for algo in algos:
                permutations = ALGORITHM_PERMUTATIONS.get(algo, [{}])

                for perm_idx, options in enumerate(permutations):
                    suffix = f"{algo}_{perm_idx}"
                    out_img = images_dir / f"{stem}_{suffix}.png"
                    out_gcode = gcode_dir / f"{stem}_{suffix}.gcode"

                    try:
                        # Save normalized image
                        img.save(out_img)

                        # Convert to paths
                        turtle = self.converter.convert(img, algo, options)

                        # Export gcode
                        comment = f"Source: {img_path.name} | Algorithm: {algo} | Options: {options}"
                        gcode = self.exporter.export(turtle, comment)

                        out_gcode.write_text(gcode)

                        manifest["pairs"].append({
                            "image": str(out_img),
                            "gcode": str(out_gcode),
                            "algorithm": algo,
                            "options": options,
                            "source": str(img_path),
                        })

                        # Track stats
                        if algo not in manifest["stats"]["algorithms"]:
                            manifest["stats"]["algorithms"][algo] = 0
                        manifest["stats"]["algorithms"][algo] += 1

                    except Exception as e:
                        manifest["failed"].append({
                            "path": str(img_path),
                            "algorithm": algo,
                            "error": str(e),
                        })

        manifest["stats"]["total_pairs"] = len(manifest["pairs"])
        manifest["stats"]["total_failed"] = len(manifest["failed"])

        manifest_path = output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest
