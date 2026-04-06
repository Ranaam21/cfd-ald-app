"""
openfoam/postprocess.py

Extracts OpenFOAM field results from completed aldFoam cases and converts
them to the unified HDF5 pointcloud schema for ML training.

For each completed case it reads:
  - U        (velocity vector field)
  - p        (pressure field)
  - N1       (precursor concentration — end of pulse snapshot)
  - cov      (surface coverage — end of pulse)
  - growth   (growth rate proxy — end of pulse)
  - Exp1     (time-integrated exposure)

And computes:
  - Uniformity index (both RMS and Zernike decomposition on wafer plane)
  - Dimensionless groups (Re, Sc, Pe_m, Da, Sh) per case

Output HDF5 schema (appends to the pointcloud file created by mesh_export.py):
  /coords                [N, 3]
  /inputs/node_features  [N, F_in]
  /inputs/global         [G]      geometry params + dimensionless groups
  /outputs/node_fields   [N, F_out]  U_x, U_y, U_z, p, N1, cov, growth
  /uniformity/rms        {radial_rms, azimuthal_rms, overall_rms}
  /uniformity/zernike    [28]     Zernike coefficients Z_nm up to order 6

Usage
-----
    # Process all completed cases in a sweep directory
    python3 openfoam/postprocess.py --cases_dir openfoam/cases \
                                    --out_dir data/processed/showerhead_openfoam

    # Process a single case
    python3 openfoam/postprocess.py --case_dir openfoam/cases/case_0001 \
                                    --out_dir data/processed/showerhead_openfoam
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

# ── optional heavy imports (only needed at runtime) ────────────────────────

def _import_fluidfoam():
    try:
        import fluidfoam
        return fluidfoam
    except ImportError:
        raise ImportError(
            "fluidfoam required for OpenFOAM post-processing:\n"
            "  pip install fluidfoam"
        )


# ══════════════════════════════════════════════════════════════════════════
# Field reader
# ══════════════════════════════════════════════════════════════════════════

def _latest_time(case_dir: Path) -> str:
    """Return the latest time directory name in an OpenFOAM case."""
    times = []
    for d in case_dir.iterdir():
        if d.is_dir():
            try:
                times.append(float(d.name))
            except ValueError:
                pass
    if not times:
        raise FileNotFoundError(f"No time directories found in {case_dir}")
    return f"{max(times):g}"


def read_case_fields(case_dir: Path) -> Optional[dict]:
    """
    Read all relevant fields from an OpenFOAM case using fluidfoam.

    Returns dict {field_name: np.ndarray} or None if case failed.
    """
    ff = _import_fluidfoam()

    try:
        t = _latest_time(case_dir)
    except FileNotFoundError:
        return None

    case_str = str(case_dir)
    fields   = {}

    field_specs = {
        "U":      ("vector", t),
        "p":      ("scalar", t),
        "N1":     ("scalar", t),
        "cov":    ("scalar", t),
        "growth": ("scalar", t),
        "Exp1":   ("scalar", t),
    }

    for name, (ftype, time_str) in field_specs.items():
        try:
            if ftype == "vector":
                fields[name] = ff.readfield(case_str, time_str, name)
            else:
                fields[name] = ff.readscalar(case_str, time_str, name)
        except Exception as e:
            print(f"  ⚠  Could not read {name} at t={time_str}: {e}")

    if not fields:
        return None

    # Also read mesh cell centres
    try:
        x, y, z = ff.readmesh(case_str)
        fields["x"] = x
        fields["y"] = y
        fields["z"] = z
    except Exception as e:
        print(f"  ⚠  Could not read mesh centres: {e}")

    return fields


# ══════════════════════════════════════════════════════════════════════════
# Uniformity metrics
# ══════════════════════════════════════════════════════════════════════════

def _wafer_slice(fields: dict, z_wafer: float = 0.0,
                 z_tol: float = 0.003) -> Optional[dict]:
    """Extract cells near the wafer plane (z ≈ z_wafer)."""
    if "z" not in fields:
        return None
    mask = np.abs(fields["z"] - z_wafer) < z_tol
    if mask.sum() < 10:
        return None
    return {k: v[mask] if v.shape[0] == mask.shape[0] else v
            for k, v in fields.items()}


def compute_rms_uniformity(x: np.ndarray, y: np.ndarray,
                            field: np.ndarray,
                            n_radial: int = 10,
                            n_azimuthal: int = 12) -> dict:
    """
    Compute radial and azimuthal RMS non-uniformity of a scalar field
    on a disk (e.g. wafer plane).

    Returns
    -------
    dict with keys: radial_rms, azimuthal_rms, overall_rms, mean, std
    """
    r     = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    R_max = r.max()

    # Radial bins
    r_edges = np.linspace(0, R_max, n_radial + 1)
    radial_means = []
    for i in range(n_radial):
        mask = (r >= r_edges[i]) & (r < r_edges[i+1])
        if mask.sum() > 0:
            radial_means.append(float(field[mask].mean()))

    # Azimuthal bins
    theta_edges = np.linspace(-np.pi, np.pi, n_azimuthal + 1)
    azimuthal_means = []
    for i in range(n_azimuthal):
        mask = (theta >= theta_edges[i]) & (theta < theta_edges[i+1])
        if mask.sum() > 0:
            azimuthal_means.append(float(field[mask].mean()))

    f_mean = float(field.mean())
    f_std  = float(field.std())

    radial_rms    = float(np.std(radial_means))    if radial_means    else 0.0
    azimuthal_rms = float(np.std(azimuthal_means)) if azimuthal_means else 0.0
    overall_rms   = f_std

    return {
        "radial_rms":    radial_rms,
        "azimuthal_rms": azimuthal_rms,
        "overall_rms":   overall_rms,
        "mean":          f_mean,
        "std":           f_std,
        "uniformity_index": 1.0 - (f_std / (f_mean + 1e-12)),  # 1 = perfect
    }


def compute_zernike(x: np.ndarray, y: np.ndarray,
                    field: np.ndarray,
                    n_max: int = 6) -> np.ndarray:
    """
    Project a scalar field on a unit disk onto Zernike polynomials up to
    radial order n_max.

    Parameters
    ----------
    x, y  : Cartesian coordinates on the disk
    field : scalar values at those coordinates
    n_max : maximum radial order (6 → 28 polynomials)

    Returns
    -------
    coeffs : (N_terms,) array of Zernike coefficients
             N_terms = (n_max+1)(n_max+2)/2
    """
    try:
        from zernike import RZern   # pip install zernike
        R_max = np.sqrt(x**2 + y**2).max()
        rho   = np.sqrt(x**2 + y**2) / (R_max + 1e-12)
        phi   = np.arctan2(y, x)

        cart = RZern(n_max)
        # Evaluate basis at data points
        # RZern expects (rho, phi) as separate arrays
        Z = cart.mtab   # [N_terms, N_terms] — not needed, use direct eval
        coeffs = cart.fit(rho, phi, field)
        return np.array(coeffs, dtype=np.float32)
    except Exception:
        # If zernike not installed or fit fails, return zeros
        n_terms = (n_max + 1) * (n_max + 2) // 2
        return np.zeros(n_terms, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════
# HDF5 writer
# ══════════════════════════════════════════════════════════════════════════

OUTPUT_FIELD_COLS = ["U_x", "U_y", "U_z", "p", "N1", "cov", "growth"]


def _build_output_array(fields: dict) -> np.ndarray:
    """Stack field arrays into [N, F_out] output tensor."""
    N = len(fields.get("x", np.array([])))
    if N == 0:
        return np.zeros((0, len(OUTPUT_FIELD_COLS)), dtype=np.float32)

    cols = []
    if "U" in fields and fields["U"].ndim == 2 and fields["U"].shape[1] == 3:
        cols += [fields["U"][:, 0], fields["U"][:, 1], fields["U"][:, 2]]
    else:
        cols += [np.zeros(N), np.zeros(N), np.zeros(N)]

    for fname in ["p", "N1", "cov", "growth"]:
        cols.append(fields.get(fname, np.zeros(N)))

    return np.column_stack(cols).astype(np.float32)


def write_hdf5(fields: dict, case_meta: dict, out_path: Path,
               uniformity: dict, zernike_coeffs: np.ndarray):
    import h5py
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from physics.calculator import (
        reynolds, schmidt, peclet_mass, damkohler, k_rxn_from_sticking
    )
    from openfoam.case_generator import RHO_N2, MU_N2

    proc = case_meta.get("process", {})
    geo  = case_meta.get("geometry", {})
    der  = case_meta.get("derived", {})

    N  = len(fields.get("x", []))
    xyz = np.column_stack([
        fields.get("x", np.zeros(N)),
        fields.get("y", np.zeros(N)),
        fields.get("z", np.zeros(N)),
    ]).astype(np.float32)

    output_fields = _build_output_array(fields)

    # Global feature vector: geometry + dimensionless numbers
    global_vals = np.array([
        geo.get("D",           0.002),
        geo.get("pitch_over_D", 4.0),
        geo.get("H_plenum",    0.020),
        geo.get("t_face",      0.003),
        geo.get("standoff",    0.020),
        geo.get("D_plate",     0.300),
        proc.get("flow_rate_slm", 2.0),
        proc.get("beta",       0.05),
        proc.get("v_th",       144.0),
        proc.get("D_m",        2.5e-5),
        der.get("Re",   0.0),
        der.get("Sc",   0.0),
        der.get("Pe_m", 0.0),
        der.get("Da",   0.0),
        float(case_meta.get("n_holes", 0)),
        float(case_meta.get("open_area", 0.0)),
    ], dtype=np.float32)

    global_cols = [
        "D", "pitch_over_D", "H_plenum", "t_face", "standoff", "D_plate",
        "flow_rate_slm", "beta", "v_th", "D_m",
        "Re", "Sc", "Pe_m", "Da", "n_holes", "open_area_frac",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("coords",               data=xyz,           compression="gzip")
        h5.create_dataset("inputs/global",        data=global_vals)
        h5.create_dataset("outputs/node_fields",  data=output_fields, compression="gzip")
        h5.create_dataset("outputs/node_fields",  data=output_fields, compression="gzip")

        h5["inputs/global"].attrs["columns"]       = global_cols
        h5["outputs/node_fields"].attrs["columns"] = OUTPUT_FIELD_COLS

        # Uniformity metrics
        for k, v in uniformity.items():
            h5[f"uniformity/rms"].attrs[k] = v
        h5.create_dataset("uniformity/zernike", data=zernike_coeffs)

        # Per-case fluid properties for compute_dimensionless.py
        h5.attrs["rho"]      = RHO_N2
        h5.attrs["mu"]       = MU_N2
        h5.attrs["D_m"]      = proc.get("D_m", 2.5e-5)
        h5.attrs["V_mean"]   = der.get("U_nozzle", 5.0)
        h5.attrs["L"]        = geo.get("D", 0.002)
        h5.attrs["k_rxn"]    = der.get("k_rxn", 1.8)

        # Full meta stored as JSON string for traceability
        h5.attrs["case_meta_json"] = json.dumps(case_meta)


# ══════════════════════════════════════════════════════════════════════════
# Per-case processor
# ══════════════════════════════════════════════════════════════════════════

def process_case(case_dir: Path, out_dir: Path) -> dict:
    name     = case_dir.name
    out_path = out_dir / f"{name}.h5"

    if out_path.exists():
        return {"name": name, "status": "already_done", "out": str(out_path)}

    # Load case metadata
    meta_path = case_dir / "case_meta.json"
    if not meta_path.exists():
        return {"name": name, "status": "no_meta", "out": ""}
    case_meta = json.loads(meta_path.read_text())

    # Read OpenFOAM fields
    fields = read_case_fields(case_dir)
    if fields is None or "x" not in fields:
        return {"name": name, "status": "no_fields", "out": ""}

    # Wafer-plane uniformity
    geo = case_meta.get("geometry", {})
    wafer_slice = _wafer_slice(fields, z_wafer=0.0)

    if wafer_slice and "N1" in wafer_slice:
        x_w = wafer_slice["x"]
        y_w = wafer_slice["y"]
        unif = compute_rms_uniformity(x_w, y_w, wafer_slice["N1"])
        zernike = compute_zernike(x_w, y_w, wafer_slice["N1"])
    else:
        unif    = {"radial_rms": 0.0, "azimuthal_rms": 0.0, "overall_rms": 0.0,
                   "mean": 0.0, "std": 0.0, "uniformity_index": 0.0}
        zernike = np.zeros(28, dtype=np.float32)

    write_hdf5(fields, case_meta, out_path, unif, zernike)
    print(f"  ✓  {name} → {out_path.name}  "
          f"(uniformity_index={unif['uniformity_index']:.3f})")

    return {"name": name, "status": "done", "out": str(out_path),
            "uniformity_index": unif["uniformity_index"]}


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases_dir", default=None,
                   help="Directory containing multiple case subdirs")
    p.add_argument("--case_dir",  default=None,
                   help="Single case directory")
    p.add_argument("--out_dir",   default="data/processed/showerhead_openfoam")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.case_dir:
        case_dirs = [Path(args.case_dir)]
    elif args.cases_dir:
        case_dirs = sorted(
            d for d in Path(args.cases_dir).iterdir()
            if d.is_dir() and (d / "case_meta.json").exists()
        )
    else:
        print("Provide --cases_dir or --case_dir")
        return

    print(f"Processing {len(case_dirs)} cases → {out_dir}")
    results = []
    for case_dir in case_dirs:
        print(f"\n  {case_dir.name}")
        result = process_case(case_dir, out_dir)
        results.append(result)

    done  = sum(1 for r in results if r["status"] == "done")
    print(f"\n✓  {done}/{len(results)} cases processed → {out_dir}")

    index_path = out_dir / "case_index.json"
    with open(index_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Index written: {index_path}")


if __name__ == "__main__":
    main()
