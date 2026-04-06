"""
geometry/parametric.py

Parametric 3D showerhead geometry generator.

Takes a parameter dict (from grammar.PARAM_SCHEMA) and builds:
  - Nozzle hole positions (ring / hex / square patterns)
  - 3D mesh of the fluid domain using trimesh
  - A ShowerheadGeometry object carrying meshes + metadata

The fluid domain geometry (what OpenFOAM sees) consists of:
  ┌─ plenum cylinder (gas above plate)
  ├─ nozzle cylinders (N holes through plate, fluid inside)
  └─ standoff cylinder (space between plate exit and wafer)

The plate solid is NOT meshed here — OpenFOAM's snappyHexMesh handles
the background mesh. We only export boundary surfaces as STL.

Usage
-----
    from geometry.grammar import default_params, NozzlePattern, DEFAULT_TOPOLOGY
    from geometry.parametric import build_showerhead

    params = default_params()
    params["D"] = 0.002          # 2 mm nozzles
    params["pitch_over_D"] = 4.0
    geo = build_showerhead(params)
    print(geo.summary())
"""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Tuple

import numpy as np

from geometry.grammar import (
    DEFAULT_TOPOLOGY, NozzlePattern, PARAM_SCHEMA,
    ShowerheadTopology, default_params,
)


# ══════════════════════════════════════════════════════════════════════════
# Output data structure
# ══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class ShowerheadGeometry:
    """
    All geometric data for one showerhead design.

    Attributes
    ----------
    params          : the input parameter dict (copy)
    topology        : ShowerheadTopology used
    nozzle_xy       : (N, 2) array of nozzle centre positions [m]
    n_holes         : actual number of holes placed
    open_area_frac  : nozzle open area / plate area  [-]
    pitch_actual    : mean centre-to-centre distance  [m]
    meshes          : dict of trimesh.Trimesh objects keyed by component name
                      (None if trimesh not installed — geometry is numpy-only)
    domain_bounds   : ((x_min,y_min,z_min), (x_max,y_max,z_max)) of fluid domain
    """
    params:          dict
    topology:        ShowerheadTopology
    nozzle_xy:       np.ndarray          # [N, 2]
    n_holes:         int
    open_area_frac:  float
    pitch_actual:    float
    meshes:          Optional[dict]
    domain_bounds:   Tuple[Tuple, Tuple]

    def summary(self) -> str:
        p = self.params
        lines = [
            "── ShowerheadGeometry ──────────────────────",
            f"  D            = {p['D']*1e3:.2f} mm",
            f"  Pitch/D      = {p['pitch_over_D']:.2f}  "
            f"(pitch = {self.pitch_actual*1e3:.2f} mm)",
            f"  Pattern      = {self.topology.pattern.value}",
            f"  N holes      = {self.n_holes}",
            f"  Open area    = {self.open_area_frac*100:.1f} %",
            f"  Plate dia    = {p['D_plate']*1e3:.0f} mm",
            f"  Faceplate t  = {p['t_face']*1e3:.1f} mm",
            f"  Plenum H     = {p['H_plenum']*1e3:.1f} mm",
            f"  Standoff     = {p['standoff']*1e3:.1f} mm",
            f"  Diffuser θ   = {p['theta_deg']:.1f}°",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Nozzle position generators
# ══════════════════════════════════════════════════════════════════════════

def _hex_positions(D_plate: float, pitch: float) -> np.ndarray:
    """
    Hexagonal close-packed grid of nozzle centres, clipped to circular plate.

    Returns (N, 2) array in metres, centred at (0, 0).
    """
    R_plate = D_plate / 2.0
    # Hex grid basis vectors
    a1 = np.array([pitch, 0.0])
    a2 = np.array([pitch * 0.5, pitch * np.sqrt(3) / 2.0])

    # Range of integer indices to cover the plate
    n_max = int(np.ceil(R_plate / pitch)) + 2
    positions = []
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            pos = i * a1 + j * a2
            if np.linalg.norm(pos) <= R_plate - pitch * 0.4:  # keep away from edge
                positions.append(pos)

    if not positions:
        # Fall back to single centre hole if plate is too small
        return np.array([[0.0, 0.0]])
    return np.array(positions)


def _ring_positions(D_plate: float, pitch: float) -> np.ndarray:
    """
    Concentric rings of nozzles: 1 centre + rings at r = pitch, 2·pitch, …

    Returns (N, 2) array in metres.
    """
    R_plate = D_plate / 2.0
    positions = [[0.0, 0.0]]   # centre hole always present
    r = pitch
    while r <= R_plate - pitch * 0.4:
        # number of holes in this ring: circumference / pitch
        n_ring = max(1, int(round(2 * np.pi * r / pitch)))
        angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
        for a in angles:
            positions.append([r * np.cos(a), r * np.sin(a)])
        r += pitch
    return np.array(positions)


def _square_positions(D_plate: float, pitch: float) -> np.ndarray:
    """
    Square grid of nozzles, clipped to circular plate.

    Returns (N, 2) array in metres.
    """
    R_plate = D_plate / 2.0
    n_max   = int(np.ceil(R_plate / pitch)) + 1
    positions = []
    for i in range(-n_max, n_max + 1):
        for j in range(-n_max, n_max + 1):
            pos = np.array([i * pitch, j * pitch])
            if np.linalg.norm(pos) <= R_plate - pitch * 0.4:
                positions.append(pos)
    if not positions:
        return np.array([[0.0, 0.0]])
    return np.array(positions)


PATTERN_FN = {
    NozzlePattern.HEX:    _hex_positions,
    NozzlePattern.RING:   _ring_positions,
    NozzlePattern.SQUARE: _square_positions,
}


def generate_nozzle_positions(pattern: NozzlePattern,
                               D_plate: float,
                               pitch: float) -> np.ndarray:
    """
    Dispatch to the correct pattern generator.

    Parameters
    ----------
    pattern : NozzlePattern
    D_plate : plate diameter [m]
    pitch   : centre-to-centre distance [m]

    Returns
    -------
    xy : (N, 2) float array of hole centres [m]
    """
    return PATTERN_FN[pattern](D_plate, pitch)


# ══════════════════════════════════════════════════════════════════════════
# Mesh builders (require trimesh)
# ══════════════════════════════════════════════════════════════════════════

def _build_meshes(params: dict, nozzle_xy: np.ndarray,
                  topology: ShowerheadTopology) -> Optional[dict]:
    """
    Build trimesh objects for each domain region.

    Returns None if trimesh is not installed (geometry-only mode).
    """
    try:
        import trimesh
    except ImportError:
        return None

    D        = params["D"]
    H_plenum = params["H_plenum"]
    t_face   = params["t_face"]
    standoff = params["standoff"]
    D_plate  = params["D_plate"]
    theta    = np.radians(params["theta_deg"])
    N_seg    = 32   # cylinder resolution

    R_plate  = D_plate / 2.0
    r_nozzle = D / 2.0

    # Z extents (bottom = 0 = wafer plane)
    z_wafer       = 0.0
    z_plate_bot   = standoff
    z_plate_top   = standoff + t_face
    z_plenum_top  = z_plate_top + H_plenum

    meshes = {}

    # ── Plenum: cylinder above plate ──────────────────────────────────────
    plenum = trimesh.creation.cylinder(
        radius=R_plate, height=H_plenum, sections=N_seg,
        transform=trimesh.transformations.translation_matrix(
            [0, 0, z_plate_top + H_plenum / 2.0])
    )
    meshes["plenum"] = plenum

    # ── Standoff: cylinder between plate exit and wafer ───────────────────
    standoff_mesh = trimesh.creation.cylinder(
        radius=R_plate, height=standoff, sections=N_seg,
        transform=trimesh.transformations.translation_matrix(
            [0, 0, standoff / 2.0])
    )
    meshes["standoff"] = standoff_mesh

    # ── Nozzle cylinders: one per hole through the plate ──────────────────
    nozzle_meshes = []
    for (cx, cy) in nozzle_xy:
        if params["theta_deg"] > 0.5:
            # Slightly tapered nozzle (diffuser)
            r_exit = r_nozzle + t_face * np.tan(theta)
            nz = trimesh.creation.cone(
                radius=r_exit, height=t_face, sections=N_seg,
                transform=trimesh.transformations.translation_matrix(
                    [cx, cy, z_plate_bot])
            )
        else:
            nz = trimesh.creation.cylinder(
                radius=r_nozzle, height=t_face, sections=N_seg,
                transform=trimesh.transformations.translation_matrix(
                    [cx, cy, z_plate_bot + t_face / 2.0])
            )
        nozzle_meshes.append(nz)

    if nozzle_meshes:
        meshes["nozzles"] = trimesh.util.concatenate(nozzle_meshes)

    # ── Wafer plane: thin disk for surface sampling ───────────────────────
    wafer = trimesh.creation.cylinder(
        radius=R_plate, height=0.0001, sections=N_seg,
        transform=trimesh.transformations.translation_matrix([0, 0, 0.00005])
    )
    meshes["wafer_plane"] = wafer

    return meshes


# ══════════════════════════════════════════════════════════════════════════
# Main builder
# ══════════════════════════════════════════════════════════════════════════

def build_showerhead(params: Optional[dict] = None,
                     topology: ShowerheadTopology = DEFAULT_TOPOLOGY
                     ) -> ShowerheadGeometry:
    """
    Build a complete ShowerheadGeometry from a parameter dict.

    Parameters
    ----------
    params   : dict with keys from grammar.PARAM_SCHEMA.
               Missing keys are filled from defaults.
    topology : ShowerheadTopology — pattern and optional features.

    Returns
    -------
    ShowerheadGeometry
    """
    # Fill in any missing keys from defaults
    full_params = default_params()
    if params:
        full_params.update(params)
    p = full_params

    D        = p["D"]
    pitch    = p["pitch_over_D"] * D
    D_plate  = p["D_plate"]
    t_face   = p["t_face"]
    H_plenum = p["H_plenum"]
    standoff = p["standoff"]

    # ── Generate nozzle positions ─────────────────────────────────────────
    nozzle_xy = generate_nozzle_positions(topology.pattern, D_plate, pitch)
    n_holes   = len(nozzle_xy)

    # ── Derived geometry metrics ──────────────────────────────────────────
    plate_area    = np.pi * (D_plate / 2.0) ** 2
    nozzle_area   = n_holes * np.pi * (D / 2.0) ** 2
    open_area_frac = nozzle_area / plate_area

    # Actual mean pitch (nearest-neighbour distance)
    if n_holes > 1:
        from scipy.spatial import cKDTree
        tree = cKDTree(nozzle_xy)
        dists, _ = tree.query(nozzle_xy, k=2)  # k=2: self + nearest neighbour
        pitch_actual = float(np.mean(dists[:, 1]))
    else:
        pitch_actual = pitch

    # ── Build 3D meshes ───────────────────────────────────────────────────
    meshes = _build_meshes(p, nozzle_xy, topology)

    # ── Domain bounding box ───────────────────────────────────────────────
    R = D_plate / 2.0
    z_total = standoff + t_face + H_plenum
    domain_bounds = ((-R, -R, 0.0), (R, R, z_total))

    return ShowerheadGeometry(
        params        = dict(p),
        topology      = topology,
        nozzle_xy     = nozzle_xy,
        n_holes       = n_holes,
        open_area_frac = open_area_frac,
        pitch_actual  = pitch_actual,
        meshes        = meshes,
        domain_bounds = domain_bounds,
    )


# ══════════════════════════════════════════════════════════════════════════
# Parameter sweep helper
# ══════════════════════════════════════════════════════════════════════════

def sweep_params(sweep: dict,
                 base_params: Optional[dict] = None,
                 topology: ShowerheadTopology = DEFAULT_TOPOLOGY
                 ) -> List[ShowerheadGeometry]:
    """
    Generate a list of ShowerheadGeometry objects from a parameter sweep dict.

    Parameters
    ----------
    sweep       : {param_name: [val1, val2, …]}  — axes to sweep
    base_params : base parameter dict (defaults filled in automatically)

    Returns
    -------
    List of ShowerheadGeometry, one per combination.

    Example
    -------
    geos = sweep_params({
        "D":            [0.001, 0.002, 0.003],
        "pitch_over_D": [3.0, 4.0, 5.0],
    })
    """
    import itertools

    keys   = list(sweep.keys())
    values = list(sweep.values())
    base   = base_params or default_params()

    results = []
    for combo in itertools.product(*values):
        p = dict(base)
        p.update(dict(zip(keys, combo)))
        results.append(build_showerhead(p, topology))

    return results


# ══════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════

def _run_tests():
    from geometry.grammar import NozzlePattern, ShowerheadTopology

    # Default build
    geo = build_showerhead()
    assert geo.n_holes > 0,        "Should have at least 1 hole"
    assert 0 < geo.open_area_frac < 1, f"Open area fraction out of range: {geo.open_area_frac}"
    assert geo.nozzle_xy.shape[1] == 2

    # All three patterns
    for pat in NozzlePattern:
        topo = ShowerheadTopology(pattern=pat)
        g = build_showerhead(topology=topo)
        assert g.n_holes > 0, f"No holes for pattern {pat}"

    # Small plate → still produces at least 1 hole
    g = build_showerhead({"D": 0.002, "D_plate": 0.010, "pitch_over_D": 3.0})
    assert g.n_holes >= 1

    # Open area within physical range (5%–80%)
    g = build_showerhead({"D": 0.002, "pitch_over_D": 4.0})
    assert 0.005 < g.open_area_frac < 0.8, f"Open area out of range: {g.open_area_frac:.3f}"

    # Sweep
    geos = sweep_params({"D": [0.001, 0.002], "pitch_over_D": [3.0, 4.0]})
    assert len(geos) == 4, f"Expected 4 sweep cases, got {len(geos)}"

    print("All geometry tests passed.")
    print(build_showerhead().summary())


if __name__ == "__main__":
    _run_tests()
