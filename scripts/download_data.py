"""Download art dataset from Kaggle."""

import os
import zipfile
from pathlib import Path

DATASET = "thedownhill/art-images-drawings-painting-sculpture-engraving"
OUTPUT_DIR = Path("data/raw")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("Setup: Place kaggle.json in ~/.kaggle/")
        print("Get it from: https://www.kaggle.com/settings -> API -> Create New Token")
        return

    # Use kaggle Python API
    os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_json.parent)
    from kaggle import api

    print(f"Downloading {DATASET}...")
    api.dataset_download_files(DATASET, path=str(OUTPUT_DIR), unzip=False)

    # Extract
    zip_file = OUTPUT_DIR / "art-images-drawings-painting-sculpture-engraving.zip"
    if zip_file.exists():
        print("Extracting...")
        with zipfile.ZipFile(zip_file, "r") as z:
            z.extractall(OUTPUT_DIR)
        zip_file.unlink()
        print(f"Done: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

