"""
data/scripts/convert_to_hdf5.py

Converts raw AirfRANS (.npz pointcloud) and CFDBench (.npz grid) files into
the unified HDF5 schema used throughout the cfd-ald-app pipeline.

Unified HDF5 schema
────────────────────
Pointcloud / mesh cases (AirfRANS, future OpenFOAM cases):
    /coords              [N, 3]   node coordinates (z=0 for 2-D)
    /inputs/node_features [N, F_in] per-node inputs (inlet BCs, SDF, …)
    /inputs/global        [G]     case-level scalars (Re, AoA, …)
                                  dimensionless groups added by compute_dimensionless.py
    /outputs/node_fields  [N, F_out] target fields (u_x, u_y, p, nu_t, …)
    /mask/surface         [N]     bool — surface nodes

Grid cases (CFDBench):
    /inputs/bc_field      [C_in, H, W]  boundary-condition channels
    /inputs/global        [G]           case-level scalars + dimensionless groups
    /outputs/fields       [T, C_out, H, W]  time-step field snapshots

Metadata (metadata.json per dataset)
──────────────────────────────────────
Written once at the dataset level; describes field names, units, splits.

Usage
-----
    # AirfRANS
    python3 data/scripts/convert_to_hdf5.py airfrans \
        --raw  data/raw/airfrans \
        --out  data/processed/airfrans

    # CFDBench
    python3 data/scripts/convert_to_hdf5.py cfdbench \
        --raw  data/raw/cfdbench \
        --out  data/processed/cfdbench
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", choices=["airfrans", "cfdbench"],
                   help="Which dataset to convert")
    p.add_argument("--raw",  required=True, help="Path to raw data directory")
    p.add_argument("--out",  required=True, help="Output directory for HDF5 files")
    p.add_argument("--max_cases", type=int, default=None,
                   help="Process only first N cases (for testing)")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# AirfRANS converter  →  pointcloud HDF5
# ══════════════════════════════════════════════════════════════════════════

# AirfRANS node-feature layout (from the airfrans library):
#   col 0-1 : x, y  coordinates [m]
#   col 2   : inlet u_x [m/s]  (uniform far-field x-velocity)
#   col 3   : inlet u_y [m/s]  (uniform far-field y-velocity, ~0 unless AoA)
#   col 4   : signed distance function (SDF) to airfoil surface

# AirfRANS target layout:
#   col 0 : u_x  [m/s]
#   col 1 : u_y  [m/s]
#   col 2 : p    [Pa] (actually p/rho, kinematic pressure)
#   col 3 : nu_t [m²/s] turbulent viscosity

AIRFRANS_INPUT_COLS  = ["x", "y", "inlet_ux", "inlet_uy", "sdf"]
AIRFRANS_OUTPUT_COLS = ["u_x", "u_y", "p_over_rho", "nu_t"]

# Reference fluid properties for Re calculation (air at ~20 °C, 1 atm)
AIRFRANS_RHO = 1.2      # kg/m³
AIRFRANS_MU  = 1.81e-5  # Pa·s
AIRFRANS_CHORD = 1.0    # normalised chord length used in AirfRANS


def _airfrans_global_features(x: np.ndarray) -> np.ndarray:
    """Extract case-level scalars: [inlet_ux, inlet_uy, Re_chord, AoA_rad]"""
    inlet_ux = float(x[0, 2])
    inlet_uy = float(x[0, 3])
    V_inf    = float(np.sqrt(inlet_ux**2 + inlet_uy**2))
    Re       = float(AIRFRANS_RHO * V_inf * AIRFRANS_CHORD / AIRFRANS_MU)
    aoa_rad  = float(np.arctan2(inlet_uy, inlet_ux))
    return np.array([inlet_ux, inlet_uy, V_inf, Re, aoa_rad], dtype=np.float32)


def convert_airfrans(raw_dir: str, out_dir: str, max_cases: Optional[int]):
    raw  = Path(raw_dir)
    out  = Path(out_dir)

    manifest_path = raw / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in {raw}. Run download_airfrans.py first.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    if max_cases:
        manifest = manifest[:max_cases]

    # Group by split for output dirs
    splits_seen = set()
    case_index  = []

    for entry in tqdm(manifest, desc="AirfRANS → HDF5"):
        npz_path = raw / entry["file"]
        if not npz_path.exists():
            print(f"  ⚠  Missing: {npz_path}  — skipping")
            continue

        data = np.load(npz_path)
        x    = data["x"].astype(np.float32)   # [N, 5]
        y    = data["y"].astype(np.float32)    # [N, 4]
        surf = data["surf"].astype(bool)        # [N]

        # coords = first two columns of x (x, y); pad z=0
        coords = np.hstack([x[:, :2], np.zeros((len(x), 1), dtype=np.float32)])  # [N, 3]
        node_features = x[:, 2:]  # [N, 3]  inlet_ux, inlet_uy, sdf
        global_feats  = _airfrans_global_features(x)  # [5]

        split    = entry["split"]
        name     = entry["name"]
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        splits_seen.add(split)

        h5_path = split_dir / f"{name}.h5"
        with h5py.File(h5_path, "w") as h5:
            h5.create_dataset("coords",                    data=coords,        compression="gzip")
            h5.create_dataset("inputs/node_features",      data=node_features, compression="gzip")
            h5.create_dataset("inputs/global",             data=global_feats)
            h5.create_dataset("outputs/node_fields",       data=y,             compression="gzip")
            h5.create_dataset("mask/surface",              data=surf)

            # Store field names as attributes for introspection
            h5["inputs/node_features"].attrs["columns"] = ["inlet_ux", "inlet_uy", "sdf"]
            h5["inputs/global"].attrs["columns"]        = ["inlet_ux", "inlet_uy", "V_inf", "Re", "AoA_rad"]
            h5["outputs/node_fields"].attrs["columns"]  = AIRFRANS_OUTPUT_COLS

        case_index.append({
            "name":  name,
            "split": split,
            "file":  str(h5_path.relative_to(out)),
            "n_nodes": int(len(coords)),
        })

    _write_metadata(out, "airfrans_pretrain", "pointcloud",
                    fields=AIRFRANS_OUTPUT_COLS,
                    inputs=["node_features", "global"],
                    units={"u_x": "m/s", "u_y": "m/s", "p_over_rho": "m2/s2", "nu_t": "m2/s"},
                    splits={s: f"{s}/" for s in splits_seen},
                    case_index=case_index)

    print(f"\n✓  AirfRANS: {len(case_index)} cases written to {out}")


# ══════════════════════════════════════════════════════════════════════════
# CFDBench converter  →  grid HDF5
# ══════════════════════════════════════════════════════════════════════════

# CFDBench field layout (after HF download):
#   'u'  [T, H, W]  x-velocity
#   'v'  [T, H, W]  y-velocity
#   'p'  [T, H, W]  pressure
#
# Case-level parameters differ by problem; we store whatever scalars exist
# and compute Re from velocity + geometry when possible.

CFDBENCH_OUTPUT_COLS = ["u", "v", "p"]


def convert_cfdbench(raw_dir: str, out_dir: str, max_cases: Optional[int]):
    raw = Path(raw_dir)
    out = Path(out_dir)

    manifest_path = raw / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {raw}. Run download_cfdbench.py first.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    if max_cases:
        manifest = manifest[:max_cases]

    splits_seen = set()
    case_index  = []

    for entry in tqdm(manifest, desc="CFDBench → HDF5"):
        npz_path = raw / entry["file"]
        if not npz_path.exists():
            print(f"  ⚠  Missing: {npz_path}  — skipping")
            continue

        data    = np.load(npz_path)
        problem = entry["problem"]
        meta    = entry.get("meta", {})

        # Stack u, v, p into [T, 3, H, W] output tensor
        fields = []
        for key in ["u", "v", "p"]:
            if key in data:
                arr = data[key].astype(np.float32)  # [T, H, W]
                if arr.ndim == 2:
                    arr = arr[np.newaxis]            # treat as single timestep
                fields.append(arr)

        if not fields:
            print(f"  ⚠  No u/v/p in {npz_path}  — skipping")
            continue

        output_fields = np.stack(fields, axis=1).astype(np.float32)  # [T, C, H, W]

        # BC field: first time-step of each field as boundary snapshot [C, H, W]
        bc_field = output_fields[0]   # [C, H, W]

        # Global features: problem-specific scalars from meta dict
        global_keys = sorted(meta.keys())
        global_vals = np.array([float(meta[k]) for k in global_keys], dtype=np.float32)

        split     = entry["split"]
        name      = entry["name"]
        split_dir = out / problem / split
        split_dir.mkdir(parents=True, exist_ok=True)
        splits_seen.add(f"{problem}/{split}")

        h5_path = split_dir / f"{name}.h5"
        with h5py.File(h5_path, "w") as h5:
            h5.create_dataset("inputs/bc_field",  data=bc_field,      compression="gzip")
            h5.create_dataset("inputs/global",    data=global_vals)
            h5.create_dataset("outputs/fields",   data=output_fields, compression="gzip")

            h5["inputs/bc_field"].attrs["channels"]  = CFDBENCH_OUTPUT_COLS
            h5["inputs/global"].attrs["columns"]     = global_keys
            h5["outputs/fields"].attrs["channels"]   = CFDBENCH_OUTPUT_COLS
            h5.attrs["problem"] = problem

        T, C, H, W = output_fields.shape
        case_index.append({
            "name":    name,
            "problem": problem,
            "split":   split,
            "file":    str(h5_path.relative_to(out)),
            "shape":   [T, C, H, W],
        })

    _write_metadata(out, "cfdbench_pretrain", "grid",
                    fields=CFDBENCH_OUTPUT_COLS,
                    inputs=["bc_field", "global"],
                    units={"u": "m/s", "v": "m/s", "p": "Pa"},
                    splits={s: f"{s}/" for s in splits_seen},
                    case_index=case_index)

    print(f"\n✓  CFDBench: {len(case_index)} cases written to {out}")


# ══════════════════════════════════════════════════════════════════════════
# Shared metadata writer
# ══════════════════════════════════════════════════════════════════════════

def _write_metadata(out_dir: Path, dataset_name: str, representation: str,
                    fields: List[str], inputs: List[str], units: dict,
                    splits: dict, case_index: list):
    meta = {
        "dataset_name":    dataset_name,
        "representation":  representation,       # "pointcloud" | "grid"
        "fields":          fields,
        "inputs":          inputs,
        "dimensionless":   [],                   # filled in by compute_dimensionless.py
        "units":           units,
        "splits":          splits,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(out_dir / "case_index.json", "w") as f:
        json.dump(case_index, f, indent=2)

    print(f"metadata.json + case_index.json written to {out_dir}")


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    if args.dataset == "airfrans":
        convert_airfrans(args.raw, args.out, args.max_cases)
    else:
        convert_cfdbench(args.raw, args.out, args.max_cases)
