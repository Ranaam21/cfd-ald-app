"""
data/scripts/compute_dimensionless.py

Reads processed HDF5 cases, computes the project's standard dimensionless
numbers per case, and writes them back into inputs/global so every training
notebook has physics features ready without recomputing.

After this script runs, each HDF5 file gains:
    /inputs/global         updated array  [G + D]
    /inputs/global         attrs["columns"] updated list

And metadata.json gains:
    "dimensionless": ["Re", "Pr", "Sc", "Pe_h", "Pe_m", "Ma", "Da", ...]

Fluid property defaults
───────────────────────
AirfRANS   — air at 20 °C, 1 atm  (Pr, Sc, Ma are constants per case)
CFDBench   — air at 20 °C (same defaults; Da = 0 since non-reactive)
OpenFOAM showerhead cases — per-case properties read from HDF5 attrs

Usage
-----
    python3 data/scripts/compute_dimensionless.py \
        --processed data/processed/airfrans  --dataset airfrans

    python3 data/scripts/compute_dimensionless.py \
        --processed data/processed/cfdbench  --dataset cfdbench
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from physics.calculator import (
    reynolds, prandtl, schmidt, mach,
    peclet_heat, peclet_mass, damkohler,
)


# ── Default fluid properties (air at 20 °C, 1 atm) ───────────────────────
AIR = {
    "rho":      1.204,      # kg/m³
    "mu":       1.81e-5,    # Pa·s
    "cp":       1005.0,     # J/(kg·K)
    "k_fluid":  0.0257,     # W/(m·K)
    "D_m":      2.0e-5,     # m²/s  (binary diffusivity O2–N2 as proxy)
    "a":        343.0,      # m/s  speed of sound
    "L":        1.0,        # m    chord (AirfRANS normalised)
}

# N2 carrier at 120 °C (ALD showerhead default)
N2_ALD = {
    "rho":      1.12,
    "mu":       2.0e-5,
    "cp":       1040.0,
    "k_fluid":  0.031,
    "D_m":      2.5e-5,     # TMA in N2
    "a":        380.0,
    "L":        0.002,      # 2 mm nozzle diameter default
}


# ══════════════════════════════════════════════════════════════════════════
# Per-dataset compute functions
# ══════════════════════════════════════════════════════════════════════════

def _dim_numbers_air(V: float, L: float, fp: dict) -> dict:
    """Compute all applicable numbers for a non-reactive air case."""
    Re   = reynolds(fp["rho"], V, L, fp["mu"])
    Pr   = prandtl(fp["cp"], fp["mu"], fp["k_fluid"])
    Sc   = schmidt(fp["mu"], fp["rho"], fp["D_m"])
    Ma   = mach(V, fp["a"])
    Pe_h = peclet_heat(Re, Pr)
    Pe_m = peclet_mass(Re, Sc)
    Da   = 0.0      # non-reactive
    return dict(Re=Re, Pr=Pr, Sc=Sc, Ma=Ma, Pe_h=Pe_h, Pe_m=Pe_m, Da=Da)

DIM_COLS = ["Re", "Pr", "Sc", "Ma", "Pe_h", "Pe_m", "Da"]


# ── AirfRANS ───────────────────────────────────────────────────────────────

def process_airfrans(processed_dir: Path):
    """
    AirfRANS global array layout (after convert_to_hdf5.py):
        [0] inlet_ux  [1] inlet_uy  [2] V_inf  [3] Re  [4] AoA_rad
    We recompute Re cleanly and append Pr, Sc, Ma, Pe_h, Pe_m, Da.
    """
    h5_files = list(processed_dir.rglob("*.h5"))
    for h5_path in tqdm(h5_files, desc="AirfRANS dimensionless"):
        with h5py.File(h5_path, "a") as h5:
            g        = h5["inputs/global"][:]               # [5]
            V_inf    = float(g[2])
            L        = AIR["L"]

            nums = _dim_numbers_air(max(V_inf, 1e-6), L, AIR)
            dim_vals = np.array([nums[k] for k in DIM_COLS], dtype=np.float32)

            # Rebuild global: original 5 + 7 dimensionless
            existing_cols = list(h5["inputs/global"].attrs.get("columns", []))
            # Remove any previously written dim cols to avoid duplication
            keep_mask = [c not in DIM_COLS for c in existing_cols]
            kept_vals = g[np.array(keep_mask, dtype=bool)] if any(keep_mask) else g

            new_global = np.concatenate([kept_vals, dim_vals]).astype(np.float32)
            new_cols   = [c for c, k in zip(existing_cols, keep_mask) if k] + DIM_COLS

            del h5["inputs/global"]
            h5.create_dataset("inputs/global", data=new_global)
            h5["inputs/global"].attrs["columns"] = new_cols

    _update_metadata(processed_dir, DIM_COLS)
    print(f"✓  AirfRANS: dimensionless numbers written to {len(h5_files)} files")


# ── CFDBench ───────────────────────────────────────────────────────────────

def process_cfdbench(processed_dir: Path):
    """
    CFDBench: estimate V from max(|u|) in first timestep bc_field.
    L defaults to domain height (normalised 1.0).
    """
    h5_files = list(processed_dir.rglob("*.h5"))
    for h5_path in tqdm(h5_files, desc="CFDBench dimensionless"):
        with h5py.File(h5_path, "a") as h5:
            bc = h5["inputs/bc_field"][:]           # [C, H, W]

            # channel 0 = u, channel 1 = v
            u_field = bc[0] if bc.shape[0] >= 1 else np.zeros((1,))
            v_field = bc[1] if bc.shape[0] >= 2 else np.zeros_like(u_field)
            speed   = np.sqrt(u_field**2 + v_field**2)
            V       = float(np.percentile(speed, 95))   # robust characteristic V
            V       = max(V, 1e-4)
            L       = 1.0                               # normalised domain length

            nums     = _dim_numbers_air(V, L, AIR)
            dim_vals = np.array([nums[k] for k in DIM_COLS], dtype=np.float32)

            existing_g    = h5["inputs/global"][:]
            existing_cols = list(h5["inputs/global"].attrs.get("columns", []))
            keep_mask     = [c not in DIM_COLS for c in existing_cols]
            kept_vals     = existing_g[np.array(keep_mask, dtype=bool)] if any(keep_mask) else existing_g

            new_global = np.concatenate([kept_vals, dim_vals]).astype(np.float32)
            new_cols   = [c for c, k in zip(existing_cols, keep_mask) if k] + DIM_COLS

            del h5["inputs/global"]
            h5.create_dataset("inputs/global", data=new_global)
            h5["inputs/global"].attrs["columns"] = new_cols

    _update_metadata(processed_dir, DIM_COLS)
    print(f"✓  CFDBench: dimensionless numbers written to {len(h5_files)} files")


# ── OpenFOAM showerhead (future) ───────────────────────────────────────────

def process_openfoam(processed_dir: Path):
    """
    For OpenFOAM-generated showerhead cases the fluid properties are stored
    per-case in HDF5 root attrs (written by openfoam/postprocess.py).
    This function reads those attrs and computes the full dimensionless set
    including Da (reactive cases).
    """
    h5_files = list(processed_dir.rglob("*.h5"))
    for h5_path in tqdm(h5_files, desc="OpenFOAM dimensionless"):
        with h5py.File(h5_path, "a") as h5:
            # Read per-case physics from root attrs (set by postprocess.py)
            fp = {
                "rho":     float(h5.attrs.get("rho",     N2_ALD["rho"])),
                "mu":      float(h5.attrs.get("mu",      N2_ALD["mu"])),
                "cp":      float(h5.attrs.get("cp",      N2_ALD["cp"])),
                "k_fluid": float(h5.attrs.get("k_fluid", N2_ALD["k_fluid"])),
                "D_m":     float(h5.attrs.get("D_m",     N2_ALD["D_m"])),
                "a":       float(h5.attrs.get("a",       N2_ALD["a"])),
            }
            V    = float(h5.attrs.get("V_mean", 5.0))
            L    = float(h5.attrs.get("L",      N2_ALD["L"]))
            k_rxn = float(h5.attrs.get("k_rxn", 1.8))  # m/s default (beta=0.05, v_th=144)

            Re   = reynolds(fp["rho"], V, L, fp["mu"])
            Pr   = prandtl(fp["cp"], fp["mu"], fp["k_fluid"])
            Sc   = schmidt(fp["mu"], fp["rho"], fp["D_m"])
            Ma   = mach(V, fp["a"])
            Pe_h = peclet_heat(Re, Pr)
            Pe_m = peclet_mass(Re, Sc)
            Da   = damkohler(k_rxn, L, V)

            dim_vals = np.array([Re, Pr, Sc, Ma, Pe_h, Pe_m, Da], dtype=np.float32)

            if "inputs/global" in h5:
                existing_g    = h5["inputs/global"][:]
                existing_cols = list(h5["inputs/global"].attrs.get("columns", []))
                keep_mask     = [c not in DIM_COLS for c in existing_cols]
                kept_vals     = existing_g[np.array(keep_mask, dtype=bool)] if any(keep_mask) else existing_g
                new_global    = np.concatenate([kept_vals, dim_vals]).astype(np.float32)
                new_cols      = [c for c, k in zip(existing_cols, keep_mask) if k] + DIM_COLS
                del h5["inputs/global"]
            else:
                new_global = dim_vals
                new_cols   = DIM_COLS

            h5.create_dataset("inputs/global", data=new_global)
            h5["inputs/global"].attrs["columns"] = new_cols

    _update_metadata(processed_dir, DIM_COLS)
    print(f"✓  OpenFOAM: dimensionless numbers written to {len(h5_files)} files")


# ══════════════════════════════════════════════════════════════════════════
# Metadata updater
# ══════════════════════════════════════════════════════════════════════════

def _update_metadata(processed_dir: Path, dim_cols: list):
    meta_path = processed_dir / "metadata.json"
    if not meta_path.exists():
        return
    with open(meta_path) as f:
        meta = json.load(f)
    meta["dimensionless"] = dim_cols
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

DATASET_FN = {
    "airfrans":  process_airfrans,
    "cfdbench":  process_cfdbench,
    "openfoam":  process_openfoam,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--processed", required=True,
                   help="Path to processed HDF5 directory (output of convert_to_hdf5.py)")
    p.add_argument("--dataset",   required=True, choices=list(DATASET_FN.keys()))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATASET_FN[args.dataset](Path(args.processed))
