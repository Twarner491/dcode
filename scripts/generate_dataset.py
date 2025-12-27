"""Generate gcode dataset from downloaded images."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dcode.data import DataGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, default=Path("data/raw"))
    parser.add_argument("-o", "--output", type=Path, default=Path("data/processed"))
    parser.add_argument("-n", "--max-samples", type=int, help="Max images to process")
    parser.add_argument("-a", "--algorithms", nargs="+", help="Specific algorithms to use")
    args = parser.parse_args()

    if not args.input.exists():
        print("Run scripts/download_data.py first")
        return

    gen = DataGenerator()
    manifest = gen.generate_dataset(
        args.input,
        args.output,
        max_samples=args.max_samples,
        algorithms=args.algorithms,
    )

    print(f"Done: {manifest['stats']['total_pairs']} pairs, {manifest['stats']['total_failed']} failed")
    print(f"By algorithm: {manifest['stats']['algorithms']}")
    print(f"Manifest: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()

