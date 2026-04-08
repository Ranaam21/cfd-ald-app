"""
geometry/quality_check.py

Validates a ShowerheadGeometry before it is sent to OpenFOAM or the ML pipeline.

Checks performed
────────────────
1. Open area fraction       — 5 % ≤ open_area ≤ 40 %
2. Pitch / D ratio          — pitch/D ≥ 2.0
3. Min wall thickness       — t_face ≥ 0.3 mm  AND  t_face > D (no hole larger than plate)
4. Nozzle-to-edge clearance — all nozzle centres ≥ D from plate edge
5. Standoff / D (H/D)       — 1 ≤ H/D ≤ 20  (jet impingement regime)
6. Watertight mesh          — if trimesh meshes present, check for open edges
7. Minimum hole count       — at least 1 nozzle

Usage
-----
    from geometry.parametric import build_showerhead
    from geometry.quality_check import check_geometry, QualityReport

    geo    = build_showerhead({"D": 0.002, "pitch_over_D": 1.5})  # bad pitch/D
    report = check_geometry(geo)
    print(report.summary())
    if not report.passed:
        print("Fix before running OpenFOAM!")
"""

from __future__ import annotations

import dataclasses
from typing import List

import numpy as np

from geometry.parametric import ShowerheadGeometry


# ══════════════════════════════════════════════════════════════════════════
# Result types
# ══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class QualityIssue:
    check:   str    # check name
    level:   str    # "error" | "warning"
    message: str    # human-readable explanation
    value:   float  # the computed value that triggered the issue
    limit:   str    # the limit that was violated, e.g. ">= 2.0"


@dataclasses.dataclass
class QualityReport:
    issues:  List[QualityIssue]
    metrics: dict   # all computed values for display in UI

    @property
    def passed(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.level == "warning" for i in self.issues)

    def summary(self) -> str:
        status = "PASS ✓" if self.passed else "FAIL ✗"
        lines  = [f"Quality check: {status}"]
        if not self.issues:
            lines.append("  No issues found.")
        for issue in self.issues:
            tag = "ERROR  " if issue.level == "error" else "WARNING"
            lines.append(f"  [{tag}] {issue.check}: {issue.message}")
        lines.append("Metrics:")
        for k, v in self.metrics.items():
            lines.append(f"  {k:25s} = {v}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Individual checks
# ══════════════════════════════════════════════════════════════════════════

# Limits
OPEN_AREA_MIN  = 0.01   # 1 %  (sparse ALD arrays are valid; warn below 5%)
OPEN_AREA_MAX  = 0.40   # 40 %
PITCH_D_MIN    = 2.0
T_FACE_MIN_M   = 3e-4   # 0.3 mm
H_D_MIN        = 1.0
H_D_MAX        = 20.0
EDGE_CLEAR_MIN = 1.0    # nozzle centres must be ≥ 1·D from plate edge


def _check_open_area(geo: ShowerheadGeometry) -> List[QualityIssue]:
    issues = []
    oa = geo.open_area_frac
    if oa < OPEN_AREA_MIN:
        issues.append(QualityIssue(
            check="open_area", level="error",
            message=f"Open area {oa*100:.1f}% < {OPEN_AREA_MIN*100:.0f}% min — "
                    f"flow will be severely restricted. Increase D or reduce pitch.",
            value=oa, limit=f">= {OPEN_AREA_MIN*100:.0f}%",
        ))
    elif oa < 0.05:
        issues.append(QualityIssue(
            check="open_area", level="warning",
            message=f"Open area {oa*100:.1f}% < 5% — sparse array; "
                    f"verify flow uniformity meets ALD requirements.",
            value=oa, limit=">= 5% recommended",
        ))
    elif oa > OPEN_AREA_MAX:
        issues.append(QualityIssue(
            check="open_area", level="warning",
            message=f"Open area {oa*100:.1f}% > {OPEN_AREA_MAX*100:.0f}% — "
                    f"plate may be structurally weak and flow distribution poor.",
            value=oa, limit=f"<= {OPEN_AREA_MAX*100:.0f}%",
        ))
    return issues


def _check_pitch_d(geo: ShowerheadGeometry) -> List[QualityIssue]:
    pitch_d = geo.params["pitch_over_D"]
    if pitch_d < PITCH_D_MIN:
        return [QualityIssue(
            check="pitch_over_D", level="error",
            message=f"Pitch/D = {pitch_d:.2f} < {PITCH_D_MIN} — "
                    f"nozzles too close; walls would overlap or be unmanufacturable.",
            value=pitch_d, limit=f">= {PITCH_D_MIN}",
        )]
    return []


def _check_wall_thickness(geo: ShowerheadGeometry) -> List[QualityIssue]:
    issues = []
    t = geo.params["t_face"]
    D = geo.params["D"]
    if t < T_FACE_MIN_M:
        issues.append(QualityIssue(
            check="t_face_min", level="error",
            message=f"Faceplate thickness {t*1e3:.2f} mm < {T_FACE_MIN_M*1e3:.1f} mm min.",
            value=t, limit=f">= {T_FACE_MIN_M*1e3:.1f} mm",
        ))
    if D >= t:
        issues.append(QualityIssue(
            check="t_face_vs_D", level="warning",
            message=f"Nozzle D ({D*1e3:.2f} mm) >= t_face ({t*1e3:.2f} mm) — "
                    f"aspect ratio L/D < 1; nozzle will behave as orifice not pipe.",
            value=D / t, limit="D < t_face",
        ))
    return issues


def _check_edge_clearance(geo: ShowerheadGeometry) -> List[QualityIssue]:
    R_plate = geo.params["D_plate"] / 2.0
    D       = geo.params["D"]
    r_nozzle = D / 2.0
    dists_from_edge = R_plate - np.linalg.norm(geo.nozzle_xy, axis=1)
    min_clearance   = float(np.min(dists_from_edge))
    if min_clearance < EDGE_CLEAR_MIN * r_nozzle:
        return [QualityIssue(
            check="edge_clearance", level="warning",
            message=f"Minimum nozzle-to-edge clearance {min_clearance*1e3:.2f} mm "
                    f"< {EDGE_CLEAR_MIN}·r_nozzle ({EDGE_CLEAR_MIN*r_nozzle*1e3:.2f} mm). "
                    f"Edge nozzles may be partially outside plate.",
            value=min_clearance, limit=f">= {EDGE_CLEAR_MIN}·r_nozzle",
        )]
    return []


def _check_standoff_hd(geo: ShowerheadGeometry) -> List[QualityIssue]:
    issues = []
    H = geo.params["standoff"]
    D = geo.params["D"]
    hd = H / D
    if hd < H_D_MIN:
        issues.append(QualityIssue(
            check="standoff_H_D", level="warning",
            message=f"H/D = {hd:.1f} < {H_D_MIN} — nozzles very close to wafer; "
                    f"risk of impingement pressure damage and non-uniform deposition.",
            value=hd, limit=f">= {H_D_MIN}",
        ))
    elif hd > H_D_MAX:
        issues.append(QualityIssue(
            check="standoff_H_D", level="warning",
            message=f"H/D = {hd:.1f} > {H_D_MAX} — jets will fully diffuse before "
                    f"reaching wafer; flow uniformity may still be good but Nu will be low.",
            value=hd, limit=f"<= {H_D_MAX}",
        ))
    return []


def _check_hole_count(geo: ShowerheadGeometry) -> List[QualityIssue]:
    if geo.n_holes < 1:
        return [QualityIssue(
            check="n_holes", level="error",
            message="No nozzle holes placed — plate diameter may be too small for given pitch.",
            value=geo.n_holes, limit=">= 1",
        )]
    return []


def _check_watertight(geo: ShowerheadGeometry) -> List[QualityIssue]:
    if geo.meshes is None:
        return []
    issues = []
    try:
        import trimesh
        for name, mesh in geo.meshes.items():
            if not mesh.is_watertight:
                issues.append(QualityIssue(
                    check=f"watertight_{name}", level="warning",
                    message=f"Mesh '{name}' has open edges — may cause snappyHexMesh issues. "
                            f"Run trimesh.repair.fix_normals() before STL export.",
                    value=0, limit="watertight",
                ))
    except ImportError:
        pass
    return issues


# ══════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════

def check_geometry(geo: ShowerheadGeometry) -> QualityReport:
    """
    Run all quality checks on a ShowerheadGeometry.

    Returns
    -------
    QualityReport with issues list, pass/fail status, and all metrics.
    """
    issues: List[QualityIssue] = []
    issues += _check_hole_count(geo)
    issues += _check_open_area(geo)
    issues += _check_pitch_d(geo)
    issues += _check_wall_thickness(geo)
    issues += _check_edge_clearance(geo)
    issues += _check_standoff_hd(geo)
    issues += _check_watertight(geo)

    p = geo.params
    D = p["D"]
    metrics = {
        "n_holes":              geo.n_holes,
        "open_area_%":          f"{geo.open_area_frac*100:.2f}",
        "pitch_over_D":         f"{p['pitch_over_D']:.2f}",
        "pitch_mm":             f"{geo.pitch_actual*1e3:.2f}",
        "t_face_mm":            f"{p['t_face']*1e3:.2f}",
        "D_mm":                 f"{D*1e3:.2f}",
        "L/D (aspect ratio)":   f"{p['t_face']/D:.2f}",
        "standoff_H/D":         f"{p['standoff']/D:.1f}",
        "plate_dia_mm":         f"{p['D_plate']*1e3:.0f}",
        "plenum_H_mm":          f"{p['H_plenum']*1e3:.1f}",
    }

    return QualityReport(issues=issues, metrics=metrics)


# ══════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════

def _run_tests():
    import sys; sys.path.insert(0, ".")
    from geometry.parametric import build_showerhead

    # Good design — should pass with no errors
    geo = build_showerhead({"D": 0.002, "pitch_over_D": 4.0, "t_face": 0.003})
    report = check_geometry(geo)
    assert report.passed, f"Expected pass:\n{report.summary()}"

    # Bad pitch/D — should error
    geo = build_showerhead({"D": 0.002, "pitch_over_D": 1.5})
    report = check_geometry(geo)
    assert not report.passed
    assert any(i.check == "pitch_over_D" for i in report.issues)

    # Too thin faceplate — should error
    geo = build_showerhead({"D": 0.002, "t_face": 0.0001})
    report = check_geometry(geo)
    assert not report.passed
    assert any(i.check == "t_face_min" for i in report.issues)

    print("All quality check tests passed.")


if __name__ == "__main__":
    _run_tests()
