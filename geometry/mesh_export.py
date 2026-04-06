"""
geometry/mesh_export.py

Exports a ShowerheadGeometry to two formats:

1. STL files  — for OpenFOAM snappyHexMesh
   Writes one STL per domain region (plenum, nozzles, standoff, wafer_plane)
   plus a combined "all_surfaces.stl" for quick visualisation.

2. Pointcloud HDF5  — for direct ML use (same schema as AirfRANS processed data)
   Samples surface points from the geometry and stores:
     /coords              [N, 3]
     /inputs/node_features [N, F]   (bc_type, sdf_to_wall, region_id)
     /inputs/global        [G]      (geometry params + dimensionless placeholders)
     /outputs/node_fields  [N, 0]   (empty — filled after CFD run)

Usage
-----
    from geometry.parametric import build_showerhead
    from geometry.mesh_export import export_stl, export_pointcloud_hdf5

    geo = build_showerhead({"D": 0.002, "pitch_over_D": 4.0})
    export_stl(geo, out_dir="openfoam/showerhead_3d/geometry")
    export_pointcloud_hdf5(geo, out_path="data/processed/showerhead_designs/case_001.h5")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from geometry.parametric import ShowerheadGeometry


# ══════════════════════════════════════════════════════════════════════════
# STL export — for OpenFOAM
# ══════════════════════════════════════════════════════════════════════════

def export_stl(geo: ShowerheadGeometry,
               out_dir: str,
               combined: bool = True) -> dict:
    """
    Write STL files from geometry meshes.

    Parameters
    ----------
    geo      : ShowerheadGeometry (must have meshes != None)
    out_dir  : directory to write STL files into
    combined : also write a combined all_surfaces.stl

    Returns
    -------
    dict {name: path}  of written files
    """
    try:
        import trimesh
    except ImportError:
        raise ImportError("trimesh required for STL export: pip install trimesh")

    if geo.meshes is None:
        raise ValueError(
            "ShowerheadGeometry has no mesh data. "
            "Ensure trimesh is installed when calling build_showerhead()."
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written = {}
    all_meshes = []

    for name, mesh in geo.meshes.items():
        path = out / f"{name}.stl"
        mesh.export(str(path))
        written[name] = str(path)
        all_meshes.append(mesh)

    if combined and all_meshes:
        combined_mesh = trimesh.util.concatenate(all_meshes)
        cpath = out / "all_surfaces.stl"
        combined_mesh.export(str(cpath))
        written["all_surfaces"] = str(cpath)

    # Write a companion JSON with nozzle positions for OpenFOAM case setup
    nozzle_json = out / "nozzle_positions.json"
    with open(nozzle_json, "w") as f:
        json.dump({
            "n_holes":      geo.n_holes,
            "D":            geo.params["D"],
            "pattern":      geo.topology.pattern.value,
            "nozzle_xy":    geo.nozzle_xy.tolist(),
            "open_area_frac": geo.open_area_frac,
            "domain_bounds": geo.domain_bounds,
        }, f, indent=2)
    written["nozzle_positions"] = str(nozzle_json)

    print(f"STL export: {len(written)} files written to {out_dir}")
    return written


# ══════════════════════════════════════════════════════════════════════════
# Pointcloud HDF5 export — for ML
# ══════════════════════════════════════════════════════════════════════════

# Region IDs used as node features (matches OpenFOAM patch naming convention)
REGION_IDS = {
    "plenum_wall":    0,
    "nozzle_wall":    1,
    "standoff_wall":  2,
    "wafer_plane":    3,
    "inlet":          4,
    "outlet":         5,
}

# BC type encoding
BC_TYPES = {
    "wall":     0,
    "inlet":    1,
    "outlet":   2,
    "symmetry": 3,
}


def export_pointcloud_hdf5(geo: ShowerheadGeometry,
                            out_path: str,
                            n_points_per_region: int = 2000) -> str:
    """
    Sample surface points and export as pointcloud HDF5 (ML schema).

    Schema matches data/processed/airfrans/ so the same DataLoader works.

    Parameters
    ----------
    geo                  : ShowerheadGeometry
    out_path             : path for the output .h5 file
    n_points_per_region  : how many surface points to sample per domain region

    Returns
    -------
    str  path to written file
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required: pip install h5py")

    coords_list  = []
    features_list = []

    p = geo.params
    D        = p["D"]
    t_face   = p["t_face"]
    H_plenum = p["H_plenum"]
    standoff = p["standoff"]
    D_plate  = p["D_plate"]
    R_plate  = D_plate / 2.0

    # ── Sample each domain region ─────────────────────────────────────────
    rng = np.random.default_rng(seed=42)

    def _cylinder_surface_pts(r, z_bot, z_top, n, region_id, bc_type):
        """Sample points on the lateral surface of a cylinder."""
        angles = rng.uniform(0, 2 * np.pi, n)
        zs     = rng.uniform(z_bot, z_top, n)
        xs     = r * np.cos(angles)
        ys     = r * np.sin(angles)
        pts    = np.column_stack([xs, ys, zs])
        feat   = np.column_stack([
            np.full(n, bc_type),
            np.zeros(n),          # sdf placeholder (0 = on surface)
            np.full(n, region_id),
        ])
        return pts, feat

    def _disk_pts(r, z, n, region_id, bc_type):
        """Sample points on a disk (top/bottom cap)."""
        # Rejection sampling for uniform disk distribution
        pts_xy = []
        while len(pts_xy) < n:
            candidates = rng.uniform(-r, r, (n * 4, 2))
            inside = candidates[np.linalg.norm(candidates, axis=1) <= r]
            pts_xy.append(inside[:n - len(pts_xy)])
        pts_xy = np.vstack(pts_xy)[:n]
        pts = np.column_stack([pts_xy, np.full(n, z)])
        feat = np.column_stack([
            np.full(n, bc_type),
            np.zeros(n),
            np.full(n, region_id),
        ])
        return pts, feat

    z_plate_bot  = standoff
    z_plate_top  = standoff + t_face
    z_plenum_top = z_plate_top + H_plenum
    n = n_points_per_region

    # Plenum lateral wall
    pts, feat = _cylinder_surface_pts(R_plate, z_plate_top, z_plenum_top,
                                       n, REGION_IDS["plenum_wall"], BC_TYPES["wall"])
    coords_list.append(pts); features_list.append(feat)

    # Plenum top (gas inlet)
    pts, feat = _disk_pts(R_plate, z_plenum_top, n, REGION_IDS["inlet"], BC_TYPES["inlet"])
    coords_list.append(pts); features_list.append(feat)

    # Standoff lateral wall
    pts, feat = _cylinder_surface_pts(R_plate, 0.0, z_plate_bot,
                                       n, REGION_IDS["standoff_wall"], BC_TYPES["wall"])
    coords_list.append(pts); features_list.append(feat)

    # Wafer plane (outlet / deposition surface)
    pts, feat = _disk_pts(R_plate, 0.0, n, REGION_IDS["wafer_plane"], BC_TYPES["outlet"])
    coords_list.append(pts); features_list.append(feat)

    # Nozzle walls — sample a subset of nozzles to keep file size reasonable
    max_nozzles = min(geo.n_holes, 50)
    idx = rng.choice(geo.n_holes, max_nozzles, replace=False)
    r_nozzle = D / 2.0
    for ci, (cx, cy) in enumerate(geo.nozzle_xy[idx]):
        n_nz = max(20, n // max_nozzles)
        angles = rng.uniform(0, 2 * np.pi, n_nz)
        zs     = rng.uniform(z_plate_bot, z_plate_top, n_nz)
        xs     = cx + r_nozzle * np.cos(angles)
        ys     = cy + r_nozzle * np.sin(angles)
        pts    = np.column_stack([xs, ys, zs])
        feat   = np.column_stack([
            np.full(n_nz, BC_TYPES["wall"]),
            np.zeros(n_nz),
            np.full(n_nz, REGION_IDS["nozzle_wall"]),
        ])
        coords_list.append(pts); features_list.append(feat)

    coords   = np.vstack(coords_list).astype(np.float32)
    features = np.vstack(features_list).astype(np.float32)
    N        = len(coords)

    # ── Global features: geometry params ─────────────────────────────────
    global_vals = np.array([
        p["D"],
        p["pitch_over_D"],
        p["H_plenum"],
        p["t_face"],
        p["standoff"],
        p["D_plate"],
        p["theta_deg"],
        geo.open_area_frac,
        float(geo.n_holes),
        float(geo.pitch_actual),
    ], dtype=np.float32)

    global_cols = [
        "D", "pitch_over_D", "H_plenum", "t_face", "standoff",
        "D_plate", "theta_deg", "open_area_frac", "n_holes", "pitch_actual",
    ]

    # ── Write HDF5 ────────────────────────────────────────────────────────
    import h5py
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("coords",               data=coords,      compression="gzip")
        h5.create_dataset("inputs/node_features", data=features,    compression="gzip")
        h5.create_dataset("inputs/global",        data=global_vals)
        # outputs/node_fields is empty at design time — filled after CFD run
        h5.create_dataset("outputs/node_fields",  data=np.zeros((N, 0), dtype=np.float32))

        h5["inputs/node_features"].attrs["columns"] = ["bc_type", "sdf", "region_id"]
        h5["inputs/global"].attrs["columns"]        = global_cols
        h5["outputs/node_fields"].attrs["columns"]  = []   # filled by postprocess.py

        # Store full params dict as root attributes for traceability
        for k, v in p.items():
            h5.attrs[k] = v
        h5.attrs["n_holes"]       = geo.n_holes
        h5.attrs["pattern"]       = geo.topology.pattern.value
        h5.attrs["open_area_frac"] = geo.open_area_frac

    print(f"Pointcloud HDF5: {N} nodes → {out_path}")
    return str(out_path)
