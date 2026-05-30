"""
geometry/vices/voxelizer.py

Evaluates a CSG tree on a 3D voxel grid → SDF volume → Marching Cubes mesh.
"""

import numpy as np
import mcubes
import trimesh
from dataclasses import dataclass


@dataclass
class VoxelGrid:
    sdf:     np.ndarray   # [Nx, Ny, Nz] signed distance values
    origin:  np.ndarray   # [3] world coords of voxel (0,0,0)
    spacing: np.ndarray   # [3] per-axis voxel spacing (NOT isotropic in general)


def build_voxel_grid(csg_tree, bounds, resolution: int = 64) -> VoxelGrid:
    """
    Evaluate csg_tree.evaluate() on a uniform grid inside bounds.

    bounds: ((xmin,xmax), (ymin,ymax), (zmin,zmax))
    resolution: number of voxels per axis (64 → 64³ grid)
    """
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = bounds

    xs = np.linspace(xmin, xmax, resolution)
    ys = np.linspace(ymin, ymax, resolution)
    zs = np.linspace(zmin, zmax, resolution)

    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    # per-axis spacing — critical for non-cubic bounds (e.g. tall thin showerheads)
    spacing = np.array([
        (xmax - xmin) / (resolution - 1),
        (ymax - ymin) / (resolution - 1),
        (zmax - zmin) / (resolution - 1),
    ], dtype=np.float32)

    chunk = 200_000
    sdf_flat = np.empty(len(pts), dtype=np.float32)
    for i in range(0, len(pts), chunk):
        sdf_flat[i:i+chunk] = csg_tree.evaluate(pts[i:i+chunk])

    sdf_vol = sdf_flat.reshape(resolution, resolution, resolution)
    return VoxelGrid(
        sdf=sdf_vol,
        origin=np.array([xmin, ymin, zmin], dtype=np.float32),
        spacing=spacing,
    )


def marching_cubes(vgrid: VoxelGrid) -> trimesh.Trimesh:
    """
    Run Marching Cubes on the SDF volume (iso-surface at 0).
    Returns a trimesh.Trimesh with world-space vertex coordinates.
    Applies per-axis spacing so non-cubic domains are correctly scaled.
    """
    verts, faces = mcubes.marching_cubes(-vgrid.sdf, 0.0)
    # verts are in voxel index space [0, resolution-1] per axis
    # convert to world coords using per-axis spacing
    verts = verts * vgrid.spacing + vgrid.origin
    mesh  = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return mesh
