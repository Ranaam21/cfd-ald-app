"""
geometry/vices/showerhead.py

Builds an ALD showerhead geometry using the VICES CSG pipeline.

CSG tree structure (n_nozzles + 1 primitives):
    plenum_body − union(nozzle_0, nozzle_1, ..., nozzle_N)

This is the simplest valid CSG tree for a showerhead.
For Track 2 topology exploration, the optimizer varies which primitives
to union/subtract and their parameters — governed by Catalan(n-1) tree topologies.

Returns a VICESResult matching the PCGMResult interface so the rest of
the pipeline (guardrails → GNN → optimizer) is identical for both tracks.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

from geometry.vices.primitives import Cylinder, Box
from geometry.vices.csg        import CSGTree, CSGNode, CSGLeaf, Op
from geometry.vices.voxelizer  import build_voxel_grid, marching_cubes, VoxelGrid
from geometry.vices.tagger     import tag_points, REGION_WALL, REGION_INLET, REGION_NOZZLE, REGION_WAFER


@dataclass
class VICESResult:
    accepted:       bool
    reason:         str = ''
    mesh:           object = None          # trimesh.Trimesh
    point_cloud:    Optional[np.ndarray] = None   # [N, 3]
    node_tags:      Optional[np.ndarray] = None   # [N] region labels
    node_features:  Optional[np.ndarray] = None   # [N, 4]
    global_features: Optional[np.ndarray] = None  # [18]
    params:         dict = field(default_factory=dict)
    csg_description: str = ''


def _hex_nozzle_centers(D_plate: float, pitch: float,
                         n_rings: int = 3) -> List[Tuple[float, float]]:
    """Generate hex-pattern nozzle centres within the plate radius."""
    centers = [(0.0, 0.0)]
    for ring in range(1, n_rings + 1):
        for i in range(6 * ring):
            angle = 2 * np.pi * i / (6 * ring)
            x = ring * pitch * np.cos(angle)
            y = ring * pitch * np.sin(angle)
            if np.sqrt(x**2 + y**2) < D_plate / 2 - pitch * 0.3:
                centers.append((x, y))
    return centers


def build_showerhead_csg(
    D_mm:        float = 2.0,
    pitch_D:     float = 4.0,
    H_plenum_mm: float = 20.0,
    t_face_mm:   float = 3.0,
    D_plate_mm:  float = 300.0,
    n_rings:     int   = 3,
    resolution:  int   = 64,
) -> VICESResult:
    """
    Build showerhead via CSG:
      1. Plenum body   = Cylinder(D_plate, H_plenum + t_face)
      2. Nozzle holes  = union of Cylinders at hex positions
      3. Final solid   = plenum_body − nozzle_union

    Parameters are in mm internally, converted to metres for output.
    resolution: voxel grid resolution per axis (64 = fast, 128 = finer mesh)
    """
    D     = D_mm     / 1000.0
    H     = H_plenum_mm / 1000.0
    tf    = t_face_mm   / 1000.0
    Dp    = D_plate_mm  / 1000.0
    pitch = D * pitch_D

    total_h = H + tf
    z_bot   = 0.0
    z_top   = total_h

    # ── CSG tree ────────────────────────────────────────────────────────────
    # Plenum body: large cylinder
    plenum = CSGLeaf(Cylinder(
        center=(0.0, 0.0, total_h / 2.0),
        radius=Dp / 2.0,
        height=total_h,
    ))

    # Nozzle holes: small cylinders through the faceplate
    nozzle_centers = _hex_nozzle_centers(Dp, pitch, n_rings)
    if not nozzle_centers:
        return VICESResult(accepted=False, reason='No nozzles fit within plate radius')

    nozzle_leaves = [
        CSGLeaf(Cylinder(
            center=(cx, cy, tf / 2.0),
            radius=D / 2.0,
            height=tf * 1.1,   # slightly taller to ensure clean boolean
        ))
        for cx, cy in nozzle_centers
    ]

    # build union tree of all nozzles (left-associative)
    nozzle_union = nozzle_leaves[0]
    for nl in nozzle_leaves[1:]:
        nozzle_union = CSGNode(Op.UNION, nozzle_union, nl)

    # final: plenum minus nozzle union
    root = CSGNode(Op.SUBTRACT, plenum, nozzle_union)
    tree = CSGTree(root)

    csg_desc = (f'Subtract(Cylinder(Dp={D_plate_mm}mm, H={total_h*1000:.1f}mm), '
                f'Union({len(nozzle_centers)} × Cylinder(D={D_mm}mm)))')

    # ── Voxelise ─────────────────────────────────────────────────────────────
    margin = D * 2
    bounds = (
        (-Dp / 2 - margin, Dp / 2 + margin),
        (-Dp / 2 - margin, Dp / 2 + margin),
        (z_bot - margin,   z_top + margin),
    )
    vgrid = build_voxel_grid(tree, bounds, resolution=resolution)

    # ── Marching Cubes → mesh ─────────────────────────────────────────────────
    mesh = marching_cubes(vgrid)
    if len(mesh.vertices) < 10:
        return VICESResult(accepted=False, reason='Marching Cubes produced empty mesh',
                           csg_description=csg_desc)

    # ── Sample point cloud from mesh surface ──────────────────────────────────
    N_pts = 80_000
    pts, face_idx = trimesh_sample(mesh, N_pts)

    # ── Tag regions ───────────────────────────────────────────────────────────
    tag_bounds = dict(
        z_inlet=z_top * 0.95,
        z_wafer=z_bot + tf * 0.05,
        z_tol=D * 0.5,
        nozzle_r=D / 2.0,
        nozzle_centers=nozzle_centers,
        r_plate=Dp / 2.0,
    )
    tags = tag_points(pts, tag_bounds)

    # ── Node features [N, 4]: [wall, inlet, nozzle, wafer] one-hot ───────────
    nf = np.zeros((len(pts), 4), dtype=np.float32)
    for i, region in enumerate([REGION_WALL, REGION_INLET, REGION_NOZZLE, REGION_WAFER]):
        nf[tags == region, i] = 1.0

    # ── Global features [18]: same keys as Track 1 ───────────────────────────
    # (physics calculator fills these in properly; zeros here as placeholders)
    gf = np.zeros(18, dtype=np.float32)

    params = dict(
        D=D, pitch_over_D=pitch_D, H_plenum=H, t_face=tf,
        D_plate=Dp, n_nozzles=len(nozzle_centers),
        D_mm=D_mm, pitch_D=pitch_D, H_plenum_mm=H_plenum_mm,
        t_face_mm=t_face_mm,
    )

    return VICESResult(
        accepted=True,
        mesh=mesh,
        point_cloud=pts,
        node_tags=tags,
        node_features=nf,
        global_features=gf,
        params=params,
        csg_description=csg_desc,
    )


def trimesh_sample(mesh, n_points: int):
    """Sample n_points uniformly from mesh surface."""
    import trimesh as _tr
    pts, face_idx = _tr.sample.sample_surface(mesh, n_points)
    return pts.astype(np.float32), face_idx
