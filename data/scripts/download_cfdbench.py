"""
data/scripts/download_cfdbench.py

Downloads CFDBench from HuggingFace and saves the raw numpy arrays
to data/raw/cfdbench/.

CFDBench — 4 classic CFD problems on 64×64 structured grids.
  cavity   : lid-driven cavity flow
  tube     : laminar pipe/tube flow
  dam      : dam-break / backward-facing step
  cylinder : flow past a cylinder (Karman vortex)

HuggingFace repo: luo-yining/CFDBench
Each case: [T, C, H, W] tensors — T time steps, C channels (u, v, p), H=W=64

Usage (local)
-------------
    python3 data/scripts/download_cfdbench.py --out data/raw/cfdbench

Usage (Colab)
-------------
    python3 data/scripts/download_cfdbench.py --out /content/drive/MyDrive/cfd-ald-app/data/raw/cfdbench
"""

import argparse
import json
import os
from pathlib import Path

from typing import Optional
import numpy as np


PROBLEMS = ["cavity", "tube", "dam", "cylinder"]

HF_REPO   = "chen-yingfa/CFDBench"
HF_SUBSETS = {
    "cavity":   "cavity",
    "tube":     "tube",
    "dam":      "dam",
    "cylinder": "cylinder",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out",      default="data/raw/cfdbench", help="Output directory")
    p.add_argument("--problems", nargs="+", default=PROBLEMS,
                   choices=PROBLEMS, help="Which problems to download")
    p.add_argument("--max_cases", type=int, default=None,
                   help="Cap cases per problem (useful for quick tests)")
    return p.parse_args()


def download(out_dir: str, problems: list, max_cases: Optional[int]):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install with: pip install datasets")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = []

    for prob in problems:
        print(f"\n── Downloading: {prob} ──")
        prob_dir = out / prob
        prob_dir.mkdir(exist_ok=True)

        ds = load_dataset(HF_REPO, name=HF_SUBSETS[prob])

        for split_name, split_ds in ds.items():
            cases = split_ds
            if max_cases:
                cases = cases.select(range(min(max_cases, len(cases))))

            split_dir = prob_dir / split_name
            split_dir.mkdir(exist_ok=True)

            for i, sample in enumerate(cases):
                # CFDBench samples contain numpy arrays under keys like
                # 'u', 'v', 'p', 'bc_params', 'case_params'
                # Shape of each field: [T, H, W]
                arrays = {k: np.array(v) for k, v in sample.items()
                          if isinstance(v, (list, np.ndarray))}
                scalars = {k: v for k, v in sample.items()
                           if not isinstance(v, (list, np.ndarray))}

                name = f"{prob}_{split_name}_{i:05d}"
                save_path = split_dir / f"{name}.npz"
                np.savez_compressed(save_path, **arrays)

                manifest.append({
                    "name":    name,
                    "problem": prob,
                    "split":   split_name,
                    "file":    str(save_path.relative_to(out)),
                    "meta":    scalars,
                })

                if (i + 1) % 100 == 0:
                    print(f"  {split_name}: saved {i+1}/{len(cases)}")

            print(f"  ✓  {prob}/{split_name}: {len(cases)} cases → {split_dir}")

    manifest_path = out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {manifest_path}  ({len(manifest)} total cases)")


if __name__ == "__main__":
    args = parse_args()
    download(args.out, args.problems, args.max_cases)
