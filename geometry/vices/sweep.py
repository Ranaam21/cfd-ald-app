"""
geometry/vices/sweep.py

Generates the Track 2 parameter sweep: 40 cases across all 4 geometry types.

Each case:
  - Builds VICES geometry (CSG → Marching Cubes → point cloud)
  - Exports STL for OpenFOAM snappyHexMesh
  - Saves metadata JSON

Sweep design:
  Types A, B, C, D × D=[1.5, 2.0, 2.5]mm × Q=[1.0, 3.0, 5.0]slm = 9 cases/type = 36
  + 4 special cases (one per type with distinct type-specific param)  = 4
  Total: 40 cases

Usage:
    python3 geometry/vices/sweep.py --out_dir openfoam/track2_cases --resolution 64
"""

import argparse
import json
import shutil
from itertools import product
from pathlib import Path

import numpy as np

from geometry.vices.variants import VARIANT_BUILDERS


# ── Sweep parameter grid ──────────────────────────────────────────────────────

D_MM_VALS   = [1.5, 2.0, 2.5]
Q_SLM_VALS  = [1.0, 3.0, 5.0]
PITCH_D     = 4.0
H_PLENUM_MM = 20.0
T_FACE_MM   = 3.0
D_PLATE_MM  = 150.0

# Type-specific extra variants (one special case per type)
SPECIAL_CASES = [
    dict(variant='A', D_mm=2.0, Q_slm=3.0, baffle_frac=0.3, baffle_thick_mm=2.0),
    dict(variant='B', D_mm=2.0, Q_slm=3.0, cone_r_frac=0.50, cone_h_frac=0.60),
    dict(variant='C', D_mm=2.0, Q_slm=3.0, n_rings=4),
    dict(variant='D', D_mm=2.0, Q_slm=3.0, divider_r_frac=0.55, divider_h_frac=0.65),
]


def _base_kwargs(D_mm, Q_slm, variant):
    return dict(D_mm=D_mm, pitch_D=PITCH_D, H_plenum_mm=H_PLENUM_MM,
                t_face_mm=T_FACE_MM, D_plate_mm=D_PLATE_MM)


def generate_sweep(out_dir: Path, resolution: int = 64, dry_run: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    stl_dir  = out_dir / 'stl'
    meta_dir = out_dir / 'metadata'
    stl_dir.mkdir(exist_ok=True)
    meta_dir.mkdir(exist_ok=True)

    cases = []
    case_id = 0

    # ── Base grid: 4 types × 3D × 3Q = 36 cases ─────────────────────────────
    for variant in ['A', 'B', 'C', 'D']:
        for D_mm, Q_slm in product(D_MM_VALS, Q_SLM_VALS):
            kwargs = _base_kwargs(D_mm, Q_slm, variant)
            kwargs['resolution'] = resolution
            cases.append(dict(case_id=case_id, variant=variant,
                              Q_slm=Q_slm, **kwargs))
            case_id += 1

    # ── Special cases: 4 extra ────────────────────────────────────────────────
    for sp in SPECIAL_CASES:
        v = sp.pop('variant')
        Q_slm = sp.pop('Q_slm')
        D_mm  = sp.pop('D_mm')
        kwargs = _base_kwargs(D_mm, Q_slm, v)
        kwargs.update(sp)
        kwargs['resolution'] = resolution
        cases.append(dict(case_id=case_id, variant=v,
                          Q_slm=Q_slm, **kwargs))
        case_id += 1

    print(f'Track 2 sweep: {len(cases)} cases across types A/B/C/D')
    print(f'Output dir: {out_dir}')
    if dry_run:
        for c in cases:
            print(f"  [{c['case_id']:03d}] Type {c['variant']}  "
                  f"D={c['D_mm']}mm  Q={c['Q_slm']}slm")
        return cases

    failed = []
    for c in cases:
        cid     = c['case_id']
        variant = c['variant']
        Q_slm   = c['Q_slm']
        kwargs  = {k: v for k, v in c.items()
                   if k not in ('case_id', 'variant', 'Q_slm')}

        tag = (f"track2_{cid:04d}_type{variant}"
               f"_D{c['D_mm']}mm_Q{Q_slm}slm")
        print(f'  [{cid:3d}/{len(cases)}] {tag} ...', end=' ', flush=True)

        try:
            result = VARIANT_BUILDERS[variant](**kwargs)
            if not result.accepted:
                print(f'REJECTED: {result.reason}')
                failed.append(tag)
                continue

            # Export STL
            stl_path = stl_dir / f'{tag}.stl'
            result.mesh.export(str(stl_path))

            # Save metadata
            meta = dict(
                case_id=cid, tag=tag, variant=variant, Q_slm=Q_slm,
                n_vertices=len(result.mesh.vertices),
                n_faces=len(result.mesh.faces),
                n_nozzles=result.params.get('n_nozzles', 0),
                csg=result.csg_description,
                params=result.params,
                stl=str(stl_path),
            )
            with open(meta_dir / f'{tag}.json', 'w') as f:
                json.dump(meta, f, indent=2)

            print(f'OK  verts={len(result.mesh.vertices)}  '
                  f'nozzles={result.params.get("n_nozzles",0)}')

        except Exception as e:
            print(f'ERROR: {e}')
            failed.append(tag)

    print(f'\nDone. {len(cases)-len(failed)}/{len(cases)} succeeded.')
    if failed:
        print('Failed:', failed)

    # Write sweep index
    index_path = out_dir / 'sweep_index.json'
    with open(index_path, 'w') as f:
        json.dump(cases, f, indent=2)
    print(f'Sweep index: {index_path}')
    return cases


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir',    default='openfoam/track2_cases')
    p.add_argument('--resolution', type=int, default=64)
    p.add_argument('--dry_run',    action='store_true')
    args = p.parse_args()
    generate_sweep(Path(args.out_dir), args.resolution, args.dry_run)
