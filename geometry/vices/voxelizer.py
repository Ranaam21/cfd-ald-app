"""
geometry/vices/voxelizer.py

Evaluates a CSG tree on a 3D voxel grid → SDF volume → Marching Cubes mesh.
"""

import numpy as np
import mcubes
import trimesh
from dataclasses import dataclass
from typing import Tuple


@dataclass
class VoxelGrid:
    sdf:        np.ndarray   # [Nx, Ny, Nz] signed distance values
    origin:     np.ndarray   # [3] world coords of voxel (0,0,0)
    spacing:    float        # voxel size (isotropic)


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

    # build flat list of all grid points [N, 3]
    gx, gy, gz = np.meshgrid(xs, ys, zs, indexing='ij')
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    spacing = (xmax - xmin) / (resolution - 1)

    # evaluate SDF in chunks to avoid memory spikes on large grids
    chunk = 200_000
    sdf_flat = np.empty(len(pts), dtype=np.float32)
    for i in range(0, len(pts), chunk):
        sdf_flat[i:i+chunk] = csg_tree.evaluate(pts[i:i+chunk])

    sdf_vol = sdf_flat.reshape(resolution, resolution, resolution)
    return VoxelGrid(sdf=sdf_vol,
                     origin=np.array([xmin, ymin, zmin], dtype=np.float32),
                     spacing=spacing)


def marching_cubes(vgrid: VoxelGrid) -> trimesh.Trimesh:
    """
    Run Marching Cubes on the SDF volume (iso-surface at 0).
    Returns a trimesh.Trimesh with world-space vertex coordinates.
    """
    verts, faces = mcubes.marching_cubes(-vgrid.sdf, 0.0)
    # verts are in voxel index space — convert to world coords
    verts = verts * vgrid.spacing + vgrid.origin
    mesh  = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    return mesh
