"""Run training sweep with different configurations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dcode.train import run_sweep


def main():
    manifest = Path("data/processed/manifest.json")
    if not manifest.exists():
        print("Generate dataset first: python scripts/generate_dataset.py")
        return

    best_config = run_sweep(str(manifest), num_trials=8)
    print(f"Best config: {best_config}")


if __name__ == "__main__":
    main()

