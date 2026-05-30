"""
geometry/vices/variants.py

All 4 Track 2 geometry types for the ALD showerhead VICES sweep.

Type A — Baffled plenum:
    Cylinder − baffle_disk − nozzles
    Internal baffle redistributes flow before reaching nozzle plate.
    CSG topology: 3 primitive groups, 2 subtract ops.

Type B — Conical diffuser:
    Cylinder − cone_indent − nozzles
    Central cone spreads inlet jet radially before it hits nozzles.
    CSG topology: 3 primitive groups, 2 subtract ops.

Type C — Annular ring nozzles:
    Cylinder − union(ring_0_nozzles, ring_1_nozzles, ...)
    Nozzles arranged in concentric rings (not hex packing).
    Different spatial distribution → different flow pattern.

Type D — Two-zone plenum:
    Cylinder − divider_ring − nozzles
    Annular divider splits plenum into inner/outer gas zones.
    CSG topology: 3 primitive groups, 2 subtract ops.

All types return a VICESResult with the same interface as Track 1 PCGMResult.
"""

import numpy as np
from typing import List, Tuple
from geometry.vices.primitives import Cylinder, Box, Cone, Sphere
from geometry.vices.csg        import CSGTree, CSGNode, CSGLeaf, Op
from geometry.vices.voxelizer  import build_voxel_grid, marching_cubes
from geometry.vices.tagger     import tag_points, REGION_WALL, REGION_INLET, REGION_NOZZLE, REGION_WAFER
from geometry.vices.showerhead import VICESResult, trimesh_sample, _hex_nozzle_centers


# ── Shared helpers ────────────────────────────────────────────────────────────

def _ring_nozzle_centers(radii: List[float],
                          nozzles_per_ring: List[int]) -> List[Tuple[float, float]]:
    """Nozzle centres arranged in concentric rings."""
    centers = [(0.0, 0.0)]   # centre nozzle always present
    for r, n in zip(radii, nozzles_per_ring):
        for i in range(n):
            angle = 2 * np.pi * i / n
            centers.append((r * np.cos(angle), r * np.sin(angle)))
    return centers


def _build_nozzle_union(centers, D, tf):
    """Build CSG union of all nozzle cylinders."""
    leaves = [
        CSGLeaf(Cylinder(center=(cx, cy, tf / 2.0),
                         radius=D / 2.0, height=tf * 1.1))
        for cx, cy in centers
    ]
    node = leaves[0]
    for l in leaves[1:]:
        node = CSGNode(Op.UNION, node, l)
    return node


def _finalise(tree, bounds_tuple, tag_bounds, params, csg_desc,
              resolution=64) -> VICESResult:
    """Voxelise → Marching Cubes → tag → point cloud → VICESResult."""
    vgrid = build_voxel_grid(tree, bounds_tuple, resolution=resolution)
    mesh  = marching_cubes(vgrid)
    if len(mesh.vertices) < 10:
        return VICESResult(accepted=False,
                           reason='Marching Cubes produced empty mesh',
                           csg_description=csg_desc)
    pts, _ = trimesh_sample(mesh, 80_000)
    tags   = tag_points(pts, tag_bounds)
    nf     = np.zeros((len(pts), 4), dtype=np.float32)
    for i, region in enumerate([REGION_WALL, REGION_INLET,
                                 REGION_NOZZLE, REGION_WAFER]):
        nf[tags == region, i] = 1.0
    gf = np.zeros(18, dtype=np.float32)
    # store nozzle_centers in params so case_generator can extract nozzle wall STL
    params['nozzle_centers'] = tag_bounds.get('nozzle_centers', [])
    return VICESResult(accepted=True, mesh=mesh, point_cloud=pts,
                       node_tags=tags, node_features=nf,
                       global_features=gf, params=params,
                       csg_description=csg_desc)


# ── Type A — Baffled plenum ───────────────────────────────────────────────────

def build_type_a_baffled(
    D_mm=2.0, pitch_D=4.0, H_plenum_mm=20.0, t_face_mm=3.0,
    D_plate_mm=150.0, n_rings=2,
    baffle_frac=0.5,       # baffle z position as fraction of plenum height
    baffle_thick_mm=1.5,   # baffle thickness
    baffle_hole_frac=0.4,  # baffle hole radius as fraction of plate radius
    resolution=64,
) -> VICESResult:
    """
    CSG tree:
        SUBTRACT(
            SUBTRACT(plenum_cylinder, baffle_annulus),
            nozzle_union
        )
    Baffle is a solid disk with a central hole — forces gas to redistribute
    radially before reaching the nozzle plate.
    """
    D  = D_mm / 1000.0;  H = H_plenum_mm / 1000.0
    tf = t_face_mm / 1000.0;  Dp = D_plate_mm / 1000.0
    pitch = D * pitch_D;  bt = baffle_thick_mm / 1000.0
    total_h = H + tf
    baffle_z = tf + H * baffle_frac   # z from bottom

    # plenum
    plenum = CSGLeaf(Cylinder((0, 0, total_h / 2), Dp / 2, total_h))

    # baffle = full disk − central hole  (annular ring)
    baffle_full = CSGLeaf(Cylinder((0, 0, baffle_z), Dp / 2 * 0.95, bt))
    baffle_hole = CSGLeaf(Cylinder((0, 0, baffle_z),
                                    Dp / 2 * baffle_hole_frac, bt * 1.2))
    baffle = CSGNode(Op.SUBTRACT, baffle_full, baffle_hole)

    # nozzles
    ctrs = _hex_nozzle_centers(Dp, pitch, n_rings)
    nozzle_union = _build_nozzle_union(ctrs, D, tf)

    root = CSGNode(Op.SUBTRACT,
                   CSGNode(Op.SUBTRACT, plenum, baffle),
                   nozzle_union)

    tag_bounds = dict(z_inlet=total_h * 0.95, z_wafer=tf * 0.05,
                      z_tol=D * 0.5, nozzle_r=D / 2,
                      nozzle_centers=ctrs, r_plate=Dp / 2)
    margin = D * 2
    bounds = ((-Dp/2-margin, Dp/2+margin),
              (-Dp/2-margin, Dp/2+margin),
              (-margin, total_h+margin))
    params = dict(type='A_baffled', D_mm=D_mm, pitch_D=pitch_D,
                  H_plenum_mm=H_plenum_mm, t_face_mm=t_face_mm,
                  baffle_frac=baffle_frac, baffle_thick_mm=baffle_thick_mm,
                  n_nozzles=len(ctrs))
    csg_desc = (f'TypeA: Subtract(Subtract(Cylinder, BaffleAnnulus), '
                f'Union({len(ctrs)}×Cylinder))')
    return _finalise(CSGTree(root), bounds, tag_bounds, params,
                     csg_desc, resolution)


# ── Type B — Conical diffuser ─────────────────────────────────────────────────

def build_type_b_conical(
    D_mm=2.0, pitch_D=4.0, H_plenum_mm=20.0, t_face_mm=3.0,
    D_plate_mm=150.0, n_rings=2,
    cone_r_frac=0.35,    # cone base radius as fraction of plate radius
    cone_h_frac=0.45,    # cone height as fraction of plenum height
    resolution=64,
) -> VICESResult:
    """
    CSG tree:
        SUBTRACT(
            SUBTRACT(plenum_cylinder, cone_indent),
            nozzle_union
        )
    Cone protrudes downward from the inlet face, deflecting the inlet jet
    radially outward for more uniform distribution.
    """
    D  = D_mm / 1000.0;  H = H_plenum_mm / 1000.0
    tf = t_face_mm / 1000.0;  Dp = D_plate_mm / 1000.0
    pitch = D * pitch_D;  total_h = H + tf
    cone_r = Dp / 2 * cone_r_frac
    cone_h = H * cone_h_frac
    cone_z = total_h - cone_h / 2  # cone apex points down from top

    plenum = CSGLeaf(Cylinder((0, 0, total_h / 2), Dp / 2, total_h))
    cone   = CSGLeaf(Cone((0, 0, cone_z), cone_r, cone_h))
    ctrs   = _hex_nozzle_centers(Dp, pitch, n_rings)
    nozzle_union = _build_nozzle_union(ctrs, D, tf)

    root = CSGNode(Op.SUBTRACT,
                   CSGNode(Op.SUBTRACT, plenum, cone),
                   nozzle_union)

    tag_bounds = dict(z_inlet=total_h * 0.95, z_wafer=tf * 0.05,
                      z_tol=D * 0.5, nozzle_r=D / 2,
                      nozzle_centers=ctrs, r_plate=Dp / 2)
    margin = D * 2
    bounds = ((-Dp/2-margin, Dp/2+margin),
              (-Dp/2-margin, Dp/2+margin),
              (-margin, total_h+margin))
    params = dict(type='B_conical', D_mm=D_mm, pitch_D=pitch_D,
                  H_plenum_mm=H_plenum_mm, t_face_mm=t_face_mm,
                  cone_r_frac=cone_r_frac, cone_h_frac=cone_h_frac,
                  n_nozzles=len(ctrs))
    csg_desc = (f'TypeB: Subtract(Subtract(Cylinder, Cone(r={cone_r*1000:.1f}mm)), '
                f'Union({len(ctrs)}×Cylinder))')
    return _finalise(CSGTree(root), bounds, tag_bounds, params,
                     csg_desc, resolution)


# ── Type C — Annular ring nozzles ─────────────────────────────────────────────

def build_type_c_annular(
    D_mm=2.0, pitch_D=4.0, H_plenum_mm=20.0, t_face_mm=3.0,
    D_plate_mm=150.0,
    n_rings=3,             # number of concentric nozzle rings
    resolution=64,
) -> VICESResult:
    """
    CSG tree:
        SUBTRACT(plenum_cylinder, union(ring_0_nozzles, ring_1_nozzles, ...))
    Nozzles in concentric rings — equal angular spacing within each ring.
    Different spatial distribution to hex → different flow uniformity profile.
    """
    D  = D_mm / 1000.0;  H = H_plenum_mm / 1000.0
    tf = t_face_mm / 1000.0;  Dp = D_plate_mm / 1000.0
    total_h = H + tf

    # ring radii: evenly spaced from 0 to 70% of plate radius
    max_r = Dp / 2 * 0.70
    radii = [max_r * (i + 1) / n_rings for i in range(n_rings)]
    # nozzles per ring scales with circumference
    nozzles_per_ring = [max(4, int(2 * np.pi * r / (D * pitch_D)))
                        for r in radii]
    ctrs = _ring_nozzle_centers(radii, nozzles_per_ring)

    plenum = CSGLeaf(Cylinder((0, 0, total_h / 2), Dp / 2, total_h))
    nozzle_union = _build_nozzle_union(ctrs, D, tf)
    root = CSGNode(Op.SUBTRACT, plenum, nozzle_union)

    tag_bounds = dict(z_inlet=total_h * 0.95, z_wafer=tf * 0.05,
                      z_tol=D * 0.5, nozzle_r=D / 2,
                      nozzle_centers=ctrs, r_plate=Dp / 2)
    margin = D * 2
    bounds = ((-Dp/2-margin, Dp/2+margin),
              (-Dp/2-margin, Dp/2+margin),
              (-margin, total_h+margin))
    params = dict(type='C_annular', D_mm=D_mm, pitch_D=pitch_D,
                  H_plenum_mm=H_plenum_mm, t_face_mm=t_face_mm,
                  n_rings=n_rings, n_nozzles=len(ctrs),
                  radii_mm=[r*1000 for r in radii],
                  nozzles_per_ring=nozzles_per_ring)
    csg_desc = (f'TypeC: Subtract(Cylinder, '
                f'Union({len(ctrs)}×Cylinder rings={n_rings}))')
    return _finalise(CSGTree(root), bounds, tag_bounds, params,
                     csg_desc, resolution)


# ── Type D — Two-zone plenum ──────────────────────────────────────────────────

def build_type_d_twozone(
    D_mm=2.0, pitch_D=4.0, H_plenum_mm=20.0, t_face_mm=3.0,
    D_plate_mm=150.0, n_rings=2,
    divider_r_frac=0.45,    # divider inner radius as fraction of plate radius
    divider_h_frac=0.55,    # divider height as fraction of plenum height
    divider_thick_mm=1.5,
    resolution=64,
) -> VICESResult:
    """
    CSG tree:
        SUBTRACT(
            SUBTRACT(plenum_cylinder, divider_ring),
            nozzle_union
        )
    Annular divider ring splits plenum into inner zone (centre nozzles)
    and outer zone (peripheral nozzles). Two separate flow paths with
    different residence times — useful for ALD precursor isolation.
    """
    D  = D_mm / 1000.0;  H = H_plenum_mm / 1000.0
    tf = t_face_mm / 1000.0;  Dp = D_plate_mm / 1000.0
    pitch = D * pitch_D;  total_h = H + tf
    dt    = divider_thick_mm / 1000.0
    div_r_outer = Dp / 2 * divider_r_frac
    div_r_inner = div_r_outer - dt
    div_h       = H * divider_h_frac
    div_z       = tf + div_h / 2   # sits on faceplate, extends up

    plenum = CSGLeaf(Cylinder((0, 0, total_h / 2), Dp / 2, total_h))

    # divider ring = outer cylinder − inner cylinder
    div_outer = CSGLeaf(Cylinder((0, 0, div_z), div_r_outer, div_h))
    div_inner = CSGLeaf(Cylinder((0, 0, div_z), div_r_inner, div_h * 1.1))
    divider   = CSGNode(Op.SUBTRACT, div_outer, div_inner)

    ctrs = _hex_nozzle_centers(Dp, pitch, n_rings)
    nozzle_union = _build_nozzle_union(ctrs, D, tf)

    root = CSGNode(Op.SUBTRACT,
                   CSGNode(Op.SUBTRACT, plenum, divider),
                   nozzle_union)

    tag_bounds = dict(z_inlet=total_h * 0.95, z_wafer=tf * 0.05,
                      z_tol=D * 0.5, nozzle_r=D / 2,
                      nozzle_centers=ctrs, r_plate=Dp / 2)
    margin = D * 2
    bounds = ((-Dp/2-margin, Dp/2+margin),
              (-Dp/2-margin, Dp/2+margin),
              (-margin, total_h+margin))
    params = dict(type='D_twozone', D_mm=D_mm, pitch_D=pitch_D,
                  H_plenum_mm=H_plenum_mm, t_face_mm=t_face_mm,
                  divider_r_frac=divider_r_frac, divider_h_frac=divider_h_frac,
                  n_nozzles=len(ctrs))
    csg_desc = (f'TypeD: Subtract(Subtract(Cylinder, DividerRing(r={div_r_outer*1000:.1f}mm)), '
                f'Union({len(ctrs)}×Cylinder))')
    return _finalise(CSGTree(root), bounds, tag_bounds, params,
                     csg_desc, resolution)


# ── Builder registry ──────────────────────────────────────────────────────────

VARIANT_BUILDERS = {
    'A': build_type_a_baffled,
    'B': build_type_b_conical,
    'C': build_type_c_annular,
    'D': build_type_d_twozone,
}
