"""
openfoam/postprocess.py

Converts completed reactingFoam cases to the unified HDF5 pointcloud schema
for ML training.

Coordinate system (from case_generator.py):
    z = 0          : wafer plane  (deposition surface)
    z = z_total    : inlet plane  (gas enters)
    z_total        = standoff + t_face + H_plenum

Fields read from latest time directory:
    U    [N,3]  velocity vector
    p    [N]    pressure
    T    [N]    temperature
    TMA  [N]    TMA mass fraction

HDF5 output schema (CLAUDE.md canonical format):
    /coords                  [N, 3]        cell-centre x,y,z  [m]
    /inputs/node_features    [N, 4]        r_norm, z_norm, region_tag, |U|_BC=0
    /inputs/global           [F]           Re, Pr, Sc, Ma, Pe_h, Pe_m, Da,
                                           D, pitch_over_D, H_plenum, t_face,
                                           standoff, flow_rate_slm, beta,
                                           n_holes, open_area_frac
    /outputs/node_fields     [N, 6]        Ux, Uy, Uz, p, T, TMA
    /uniformity/rms_attrs                  TMA uniformity at wafer (scalars)
    /uniformity/zernike      [28]          Zernike decomposition of TMA on wafer

Usage
-----
    # All completed cases in a sweep directory
    python3 openfoam/postprocess.py --cases_dir openfoam/cases \
                                    --out_dir data/processed/showerhead_openfoam

    # Single case
    python3 openfoam/postprocess.py --case_dir openfoam/cases/case_0001 \
                                    --out_dir data/processed/showerhead_openfoam
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# OpenFOAM field reader
# ══════════════════════════════════════════════════════════════════════════

def _import_fluidfoam():
    try:
        import fluidfoam
        return fluidfoam
    except ImportError:
        raise ImportError(
            "fluidfoam not found. Install it:\n"
            "  pip install fluidfoam"
        )


def _latest_time(case_dir: Path) -> str:
    """Return the latest non-zero time directory name."""
    times = []
    for d in case_dir.iterdir():
        if d.is_dir():
            try:
                t = float(d.name)
                if t > 0:
                    times.append(t)
            except ValueError:
                pass
    if not times:
        raise FileNotFoundError(f"No completed time directories in {case_dir}")
    return f"{max(times):g}"


def read_case_fields(case_dir: Path) -> Optional[dict]:
    """
    Read U, p, T, TMA and mesh cell centres from an OpenFOAM case.
    Returns dict or None if the case has not completed.
    """
    ff = _import_fluidfoam()

    try:
        t = _latest_time(case_dir)
    except FileNotFoundError:
        return None

    case_str = str(case_dir)
    fields   = {}

    # Mesh cell centres
    try:
        x, y, z = ff.readmesh(case_str)
        fields["x"] = np.asarray(x, dtype=np.float32)
        fields["y"] = np.asarray(y, dtype=np.float32)
        fields["z"] = np.asarray(z, dtype=np.float32)
    except Exception as e:
        print(f"    WARN: could not read mesh centres: {e}")
        return None

    # Velocity (vector)
    try:
        U = ff.readfield(case_str, t, "U")
        fields["U"] = np.asarray(U, dtype=np.float32)   # [N, 3]
    except Exception as e:
        print(f"    WARN: could not read U: {e}")

    # Scalar fields
    for fname in ["p", "T", "TMA"]:
        try:
            val = ff.readscalar(case_str, t, fname)
            fields[fname] = np.asarray(val, dtype=np.float32)
        except Exception as e:
            print(f"    WARN: could not read {fname}: {e}")

    return fields


# ══════════════════════════════════════════════════════════════════════════
# Node feature engineering
# ══════════════════════════════════════════════════════════════════════════

def _region_tag(z: np.ndarray, z_total: float,
                standoff: float, t_face: float) -> np.ndarray:
    """
    Encode domain region as integer per cell:
        0 = standoff  (between wafer and faceplate)
        1 = faceplate (nozzle bores + solid thickness)
        2 = plenum    (above faceplate)
    """
    tags = np.zeros(len(z), dtype=np.float32)
    tags[z > standoff]                     = 1.0   # faceplate region
    tags[z > (standoff + t_face)]          = 2.0   # plenum region
    return tags


def build_node_features(fields: dict, geo: dict) -> np.ndarray:
    """
    Build [N, 4] node feature matrix:
        col 0: r_norm  — radial distance from plate centre / (D_plate/2)
        col 1: z_norm  — height from wafer / z_total   (0=wafer, 1=inlet)
        col 2: region  — 0=standoff, 1=faceplate, 2=plenum
        col 3: reserved (zeros — for BC mask, set externally if needed)
    """
    x, y, z = fields["x"], fields["y"], fields["z"]

    D_plate  = geo.get("D_plate",  0.300)
    standoff = geo.get("standoff", 0.020)
    t_face   = geo.get("t_face",   0.003)
    H_plenum = geo.get("H_plenum", 0.020)
    z_total  = standoff + t_face + H_plenum

    r      = np.sqrt(x**2 + y**2)
    r_norm = (r / (D_plate / 2.0)).astype(np.float32)
    z_norm = (z / (z_total + 1e-12)).astype(np.float32)
    region = _region_tag(z, z_total, standoff, t_face)
    zeros  = np.zeros(len(z), dtype=np.float32)

    return np.column_stack([r_norm, z_norm, region, zeros])


# ══════════════════════════════════════════════════════════════════════════
# Output field assembly
# ══════════════════════════════════════════════════════════════════════════

OUTPUT_COLS = ["Ux", "Uy", "Uz", "p", "T", "TMA"]


def build_output_array(fields: dict) -> np.ndarray:
    """Stack field arrays into [N, 6] output tensor."""
    N = len(fields["x"])

    U = fields.get("U")
    if U is not None and U.ndim == 2 and U.shape[1] == 3:
        Ux, Uy, Uz = U[:, 0], U[:, 1], U[:, 2]
    else:
        Ux = Uy = Uz = np.zeros(N, dtype=np.float32)

    p   = fields.get("p",   np.zeros(N, dtype=np.float32))
    T   = fields.get("T",   np.zeros(N, dtype=np.float32))
    TMA = fields.get("TMA", np.zeros(N, dtype=np.float32))

    # Trim all arrays to the smallest size (fluidfoam occasionally returns
    # slightly different cell counts for vector vs scalar fields)
    N = min(len(Ux), len(p), len(T), len(TMA), N)
    return np.column_stack([
        Ux[:N], Uy[:N], Uz[:N], p[:N], T[:N], TMA[:N]
    ]).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════
# Global feature vector
# ══════════════════════════════════════════════════════════════════════════

GLOBAL_COLS = [
    "Re", "Pr", "Sc", "Ma", "Pe_h", "Pe_m", "Da",
    "D", "pitch_over_D", "H_plenum", "t_face", "standoff",
    "flow_rate_slm", "beta", "v_th", "D_m",
    "n_holes", "open_area_frac",
]


def build_global_features(case_meta: dict) -> np.ndarray:
    """Build [18] global feature vector from case_meta.json."""
    dim = case_meta.get("dimensionless", {})
    geo = case_meta.get("geometry", {})
    proc = case_meta.get("process", {})

    vals = [
        dim.get("Re",   0.0),
        dim.get("Pr",   0.0),
        dim.get("Sc",   0.0),
        dim.get("Ma",   0.0),
        dim.get("Pe_h", 0.0),
        dim.get("Pe_m", 0.0),
        dim.get("Da",   0.0),
        geo.get("D",            0.002),
        geo.get("pitch_over_D", 4.0),
        geo.get("H_plenum",     0.020),
        geo.get("t_face",       0.003),
        geo.get("standoff",     0.020),
        proc.get("flow_rate_slm", 2.0),
        proc.get("beta",          0.05),
        proc.get("v_th",         144.0),
        proc.get("D_m",          2.5e-5),
        float(case_meta.get("n_holes",    0)),
        float(case_meta.get("open_area",  0.0)),
    ]
    return np.array(vals, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════
# Uniformity metrics
# ══════════════════════════════════════════════════════════════════════════

def _wafer_cells(fields: dict, z_tol: float = 0.002) -> Optional[dict]:
    """Select cells in the first z_tol metres above the wafer (z ≈ 0)."""
    z = fields.get("z")
    if z is None:
        return None
    mask = z < z_tol
    if mask.sum() < 10:
        return None
    return {k: v[mask] if (isinstance(v, np.ndarray) and v.shape[0] == len(z))
            else v for k, v in fields.items()}


def compute_rms_uniformity(x: np.ndarray, y: np.ndarray,
                            field: np.ndarray,
                            n_radial: int = 10,
                            n_azimuthal: int = 12) -> dict:
    """Radial and azimuthal RMS non-uniformity of a scalar on a disk."""
    r     = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    R_max = r.max() if r.max() > 0 else 1.0

    radial_means    = []
    azimuthal_means = []

    r_edges = np.linspace(0, R_max, n_radial + 1)
    for i in range(n_radial):
        mask = (r >= r_edges[i]) & (r < r_edges[i+1])
        if mask.sum() > 0:
            radial_means.append(float(field[mask].mean()))

    theta_edges = np.linspace(-np.pi, np.pi, n_azimuthal + 1)
    for i in range(n_azimuthal):
        mask = (theta >= theta_edges[i]) & (theta < theta_edges[i+1])
        if mask.sum() > 0:
            azimuthal_means.append(float(field[mask].mean()))

    f_mean = float(field.mean())
    f_std  = float(field.std())

    return {
        "radial_rms":       float(np.std(radial_means))    if radial_means    else 0.0,
        "azimuthal_rms":    float(np.std(azimuthal_means)) if azimuthal_means else 0.0,
        "overall_rms":      f_std,
        "mean_TMA_wafer":   f_mean,
        "uniformity_index": 1.0 - (f_std / (f_mean + 1e-12)),
    }


def compute_zernike(x: np.ndarray, y: np.ndarray,
                    field: np.ndarray, n_max: int = 6) -> np.ndarray:
    """Project scalar field onto Zernike polynomials (up to order n_max)."""
    n_terms = (n_max + 1) * (n_max + 2) // 2
    try:
        from zernike import RZern
        R_max = np.sqrt(x**2 + y**2).max()
        rho   = np.sqrt(x**2 + y**2) / (R_max + 1e-12)
        phi   = np.arctan2(y, x)
        cart  = RZern(n_max)
        coeffs = cart.fit(rho, phi, field)
        return np.array(coeffs, dtype=np.float32)
    except Exception:
        return np.zeros(n_terms, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════
# HDF5 writer
# ══════════════════════════════════════════════════════════════════════════

def write_hdf5(fields: dict, case_meta: dict, out_path: Path,
               uniformity: dict, zernike_coeffs: np.ndarray):
    import h5py

    geo = case_meta.get("geometry", {})
    N   = len(fields["x"])

    output_fields  = build_output_array(fields)
    N              = len(output_fields)   # canonical size after trimming
    # Trim coords and node features to same N
    fields["x"] = fields["x"][:N]
    fields["y"] = fields["y"][:N]
    fields["z"] = fields["z"][:N]
    coords         = np.column_stack([fields["x"], fields["y"], fields["z"]]).astype(np.float32)
    node_features  = build_node_features(fields, geo)
    global_feats   = build_global_features(case_meta)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5:
        # Canonical schema (CLAUDE.md)
        h5.create_dataset("coords",                data=coords,        compression="gzip")
        h5.create_dataset("inputs/node_features",  data=node_features, compression="gzip")
        h5.create_dataset("inputs/global",         data=global_feats)
        h5.create_dataset("outputs/node_fields",   data=output_fields, compression="gzip")

        # Column name metadata
        h5["inputs/node_features"].attrs["columns"] = ["r_norm", "z_norm", "region", "bc_mask"]
        h5["inputs/global"].attrs["columns"]        = GLOBAL_COLS
        h5["outputs/node_fields"].attrs["columns"]  = OUTPUT_COLS

        # Uniformity metrics (scalar attributes on uniformity group)
        grp = h5.require_group("uniformity")
        for k, v in uniformity.items():
            grp.attrs[k] = v
        h5.create_dataset("uniformity/zernike", data=zernike_coeffs)

        # Case-level metadata for traceability
        h5.attrs["case_name"]      = case_meta.get("case_name", out_path.stem)
        h5.attrs["solver"]         = case_meta.get("solver", "reactingFoam")
        h5.attrs["turbulence"]     = case_meta.get("turbulence", "laminar")
        h5.attrs["n_cells"]        = N
        h5.attrs["case_meta_json"] = json.dumps(case_meta)


# ══════════════════════════════════════════════════════════════════════════
# Per-case processor
# ══════════════════════════════════════════════════════════════════════════

def process_case(case_dir: Path, out_dir: Path) -> dict:
    name     = case_dir.name
    out_path = out_dir / f"{name}.h5"

    if out_path.exists():
        print(f"  [SKIP] {name}: already converted")
        return {"name": name, "status": "already_done", "out": str(out_path)}

    # Load case metadata
    meta_path = case_dir / "case_meta.json"
    if not meta_path.exists():
        print(f"  [SKIP] {name}: no case_meta.json")
        return {"name": name, "status": "no_meta", "out": ""}

    case_meta = json.loads(meta_path.read_text())
    case_meta["case_name"] = name

    # Read OpenFOAM fields
    print(f"  [READ] {name} ...")
    fields = read_case_fields(case_dir)
    if fields is None or "x" not in fields:
        print(f"  [FAIL] {name}: could not read fields")
        return {"name": name, "status": "no_fields", "out": ""}

    N = len(fields["x"])
    print(f"         {N:,} cells, latest t={_latest_time(case_dir)}")

    # Uniformity of TMA at wafer plane
    wafer = _wafer_cells(fields)
    if wafer is not None and "TMA" in wafer and wafer["TMA"].sum() > 0:
        unif    = compute_rms_uniformity(wafer["x"], wafer["y"], wafer["TMA"])
        zernike = compute_zernike(wafer["x"], wafer["y"], wafer["TMA"])
    else:
        unif    = {"radial_rms": 0.0, "azimuthal_rms": 0.0, "overall_rms": 0.0,
                   "mean_TMA_wafer": 0.0, "uniformity_index": 0.0}
        zernike = np.zeros(28, dtype=np.float32)

    # Write HDF5
    write_hdf5(fields, case_meta, out_path, unif, zernike)
    ui = unif["uniformity_index"]
    print(f"  [DONE] {name}  →  {out_path.name}  "
          f"(uniformity_index={ui:.4f}, N={N:,})")

    return {
        "name":              name,
        "status":            "done",
        "out":               str(out_path),
        "n_cells":           N,
        "uniformity_index":  ui,
    }


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cases_dir", default=None,
                   help="Directory containing multiple case subdirs")
    p.add_argument("--case_dir",  default=None,
                   help="Single case directory")
    p.add_argument("--out_dir",
                   default="data/processed/showerhead_openfoam",
                   help="Output directory for HDF5 files")
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

    print(f"\nPost-processing {len(case_dirs)} cases → {out_dir}\n")

    results = []
    for case_dir in case_dirs:
        result = process_case(case_dir, out_dir)
        results.append(result)

    # Summary
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    print(f"\n── Summary ──")
    for k, v in sorted(counts.items()):
        print(f"  {k:20s}: {v}")

    done_results = [r for r in results if r["status"] == "done"]
    if done_results:
        uis = [r["uniformity_index"] for r in done_results]
        print(f"\n  Uniformity index  mean={np.mean(uis):.4f}  "
              f"min={np.min(uis):.4f}  max={np.max(uis):.4f}")

    index_path = out_dir / "case_index.json"
    with open(index_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nIndex → {index_path}")


if __name__ == "__main__":
    main()
