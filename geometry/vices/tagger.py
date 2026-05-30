"""
geometry/vices/tagger.py

Tags mesh faces and sampled points with region labels:
    0 — wall        (faceplate / plenum sidewalls)
    1 — inlet       (top face — gas enters here)
    2 — nozzle      (inside nozzle holes)
    3 — wafer       (bottom face — deposition surface)

Tags are used as node features in the GNN (same as Track 1 PCGM).
"""

import numpy as np
import trimesh


REGION_WALL   = 0
REGION_INLET  = 1
REGION_NOZZLE = 2
REGION_WAFER  = 3


def tag_points(pts: np.ndarray, bounds: dict) -> np.ndarray:
    """
    Tag each point [N, 3] with a region label based on its position.

    bounds dict keys: z_inlet, z_wafer, z_nozzle_top, z_nozzle_bot,
                      r_plate, nozzle_centers [(x,y)], nozzle_r
    """
    tags    = np.full(len(pts), REGION_WALL, dtype=np.int32)
    z       = pts[:, 2]
    z_tol   = bounds.get('z_tol', 0.001)

    # inlet: top face
    z_inlet = bounds['z_inlet']
    tags[z > z_inlet - z_tol] = REGION_INLET

    # wafer: bottom face
    z_wafer = bounds['z_wafer']
    tags[z < z_wafer + z_tol] = REGION_WAFER

    # nozzle: points inside any nozzle cylinder
    nozzle_r    = bounds.get('nozzle_r', 0.001)
    nozzle_ctrs = bounds.get('nozzle_centers', [])
    for (nx, ny) in nozzle_ctrs:
        r2 = (pts[:, 0] - nx)**2 + (pts[:, 1] - ny)**2
        in_nozzle = r2 < (nozzle_r * 1.2)**2
        tags[in_nozzle] = REGION_NOZZLE

    return tags


def tag_mesh_faces(mesh: trimesh.Trimesh, bounds: dict) -> np.ndarray:
    """Tag face centroids and return per-face region labels."""
    centroids = mesh.triangles_center
    return tag_points(centroids, bounds)
