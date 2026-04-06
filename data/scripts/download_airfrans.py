"""
data/scripts/download_airfrans.py

Downloads the AirfRANS dataset from HuggingFace and saves the raw files
to data/raw/airfrans/.

AirfRANS — 1000 incompressible RANS simulations over NACA 4/5-digit airfoils.
Each case: ~200 K mesh points with fields u_x, u_y, p, nu_t (turbulent viscosity).
HuggingFace repo: jdunstan/Airfrans

Usage (local Mac)
-----------------
    python3 data/scripts/download_airfrans.py --out data/raw/airfrans

Usage (Colab — called from 01_data_pipeline.ipynb)
----------------------------------------------------
    python3 data/scripts/download_airfrans.py --out /content/drive/MyDrive/cfd-ald-app/data/raw/airfrans
"""

import argparse
import json
import os
from pathlib import Path

from typing import Optional
import numpy as np


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out",    default="data/raw/airfrans", help="Output directory")
    p.add_argument("--splits", nargs="+", default=["scarce", "full"],
                   help="Which splits to download: scarce (200 cases) | full (1000 cases)")
    p.add_argument("--max_cases", type=int, default=None,
                   help="Cap number of cases per split (useful for quick tests)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def download(out_dir: str, splits: list, max_cases: Optional[int]):
    try:
        import airfrans as af
    except ImportError:
        raise ImportError(
            "airfrans package not found. Install with:\n"
            "  pip install airfrans"
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # airfrans.load() returns (dataset, sample_list)
    # dataset is a list of dicts with keys:
    #   'x'         — node features: [N, 5]  (x, y, inlet_u, inlet_v, sdf)
    #   'y'         — targets:       [N, 4]  (u_x, u_y, p, nu_t)
    #   'surf'      — surface mask:  [N]     bool
    #   'name'      — case name string

    manifest = []

    for split in splits:
        print(f"\n── Downloading split: {split} ──")
        dataset, sample_names = af.load(
            root=str(out / "raw_airfrans"),
            task=split,
            train=True,
        )
        val_dataset, val_names = af.load(
            root=str(out / "raw_airfrans"),
            task=split,
            train=False,
        )

        # Combine train + val then re-split by index (we do our own split later)
        all_cases   = list(dataset)   + list(val_dataset)
        all_names   = list(sample_names) + list(val_names)

        if max_cases:
            all_cases = all_cases[:max_cases]
            all_names = all_names[:max_cases]

        split_dir = out / split
        split_dir.mkdir(exist_ok=True)

        for i, (case, name) in enumerate(zip(all_cases, all_names)):
            save_path = split_dir / f"{name}.npz"
            np.savez_compressed(
                save_path,
                x    = case["x"],        # [N, 5]  node input features
                y    = case["y"],         # [N, 4]  u_x, u_y, p, nu_t
                surf = case["surf"],      # [N]     surface mask
            )
            manifest.append({"name": name, "split": split, "file": str(save_path.relative_to(out))})
            if (i + 1) % 50 == 0:
                print(f"  saved {i+1}/{len(all_cases)}")

        print(f"  ✓  {len(all_cases)} cases saved to {split_dir}")

    # Write manifest
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {manifest_path}  ({len(manifest)} total cases)")


if __name__ == "__main__":
    args = parse_args()
    download(args.out, args.splits, args.max_cases)
