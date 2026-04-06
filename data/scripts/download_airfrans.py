"""
data/scripts/download_airfrans.py

Downloads AirfRANS using the official airfrans library and saves
each case as a .npz file for the HDF5 conversion pipeline.

AirfRANS — 1000 incompressible RANS simulations over NACA airfoils.
Fields per node: x [N,5] (coords + BCs), y [N,4] (u_x, u_y, p, nu_t)

Usage
-----
    python3 data/scripts/download_airfrans.py \
        --out /content/drive/MyDrive/cfd-ald-app/data/raw/airfrans \
        --task scarce
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out",       default="data/raw/airfrans")
    p.add_argument("--task",      default="scarce",
                   choices=["scarce", "full", "reynolds", "aoa"],
                   help="scarce=200, full=800, reynolds/aoa=specific splits")
    p.add_argument("--max_cases", type=int, default=None)
    return p.parse_args()


def download(out_dir: str, task: str, max_cases: Optional[int]):
    try:
        import airfrans as af
    except ImportError:
        raise ImportError("pip install airfrans")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: download raw dataset (skips if already downloaded)
    raw_dir = out / "raw_airfrans"
    if not raw_dir.exists():
        print("Downloading AirfRANS dataset (~3 GB)...")
        af.dataset.download(root=str(out), unzip=True)
    else:
        print(f"Raw data already at {raw_dir} — skipping download.")

    manifest = []

    # Step 2: load train + test splits
    for is_train, subset_name in [(True, "train"), (False, "test")]:
        print(f"\n── task={task}  subset={subset_name} ──")

        dataset, dataname = af.dataset.load(
            root=str(out),
            task=task,
            train=is_train,
        )

        if max_cases:
            dataset  = dataset[:max_cases]
            dataname = dataname[:max_cases]

        subset_dir = out / task / subset_name
        subset_dir.mkdir(parents=True, exist_ok=True)

        for i, (case, name) in enumerate(zip(dataset, dataname)):
            x    = np.array(case["x"],    dtype=np.float32)   # [N, 5]
            y    = np.array(case["y"],    dtype=np.float32)   # [N, 4]
            surf = np.array(case["surf"], dtype=bool)          # [N]

            save_path = subset_dir / f"{name}.npz"
            np.savez_compressed(save_path, x=x, y=y, surf=surf)

            manifest.append({
                "name":  name,
                "split": f"{task}/{subset_name}",
                "file":  str(save_path.relative_to(out)),
            })

            if (i + 1) % 50 == 0 or i == 0:
                print(f"  saved {i+1}/{len(dataset)}  x={x.shape} y={y.shape}")

        print(f"  ✓  {len(dataset)} cases → {subset_dir}")

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}  ({len(manifest)} total cases)")


if __name__ == "__main__":
    args = parse_args()
    download(args.out, args.task, args.max_cases)
