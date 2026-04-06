"""
data/scripts/download_airfrans.py

Downloads the AirfRANS dataset from HuggingFace using the `datasets` library
and saves raw files to data/raw/airfrans/.

AirfRANS — 1000 incompressible RANS simulations over NACA 4/5-digit airfoils.
HuggingFace repo: jdunstan/Airfrans
Splits: scarce (200 train cases), full (800 train cases)

Usage (Colab)
-------------
    python3 data/scripts/download_airfrans.py --out /content/drive/MyDrive/cfd-ald-app/data/raw/airfrans --split scarce
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out",       default="data/raw/airfrans")
    p.add_argument("--split",     default="scarce",
                   choices=["scarce", "full"],
                   help="scarce=200 cases (fast), full=800 cases")
    p.add_argument("--max_cases", type=int, default=None)
    return p.parse_args()


def download(out_dir: str, split: str, max_cases: Optional[int]):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []

    for is_train, subset_name in [(True, "train"), (False, "test")]:
        print(f"\n── AirfRANS  split={split}  subset={subset_name} ──")

        # HuggingFace dataset name and config
        ds = load_dataset(
            "jdunstan/Airfrans",
            name=split,
            split="train" if is_train else "test",
            trust_remote_code=True,
        )

        if max_cases:
            ds = ds.select(range(min(max_cases, len(ds))))

        subset_dir = out / split / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        for i, sample in enumerate(ds):
            # Each sample has keys: x [N,5], y [N,4], surf [N], simulation_id
            name      = sample.get("simulation_id", f"{split}_{subset_name}_{i:05d}")
            x         = np.array(sample["x"],    dtype=np.float32)   # [N, 5]
            y         = np.array(sample["y"],    dtype=np.float32)   # [N, 4]
            surf      = np.array(sample["surf"], dtype=bool)          # [N]

            save_path = subset_dir / f"{name}.npz"
            np.savez_compressed(save_path, x=x, y=y, surf=surf)

            manifest.append({
                "name":  name,
                "split": f"{split}/{subset_name}",
                "file":  str(save_path.relative_to(out)),
            })

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  saved {i+1}/{len(ds)}  shape x={x.shape} y={y.shape}")

        print(f"  ✓  {len(ds)} cases → {subset_dir}")

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}  ({len(manifest)} total cases)")


if __name__ == "__main__":
    args = parse_args()
    download(args.out, args.split, args.max_cases)
