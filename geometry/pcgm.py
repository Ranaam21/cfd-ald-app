"""
geometry/pcgm.py

Physics-Constrained Geometric Morphogenesis (PCGM) — Track 1 geometry module.

Takes a set of design parameters, runs physics guardrail checks BEFORE
generating any mesh, then produces a plenum point-cloud in the exact HDF5
(Hierarchical Data Format 5) format the MultiHeadMGN (MeshGraphNet)
surrogate expects.

Pipeline
--------
  Design params
       │
       ▼
  Pre-mesh guardrail check   ← rejects physically impossible designs early,
       │                        before wasting time on mesh generation
       ▼
  build_showerhead()         ← geometry/parametric.py (hole positions, n_holes)
       │
       ▼
  Plenum point-cloud         ← N random points inside the plenum cylinder,
       │                        with node features matching postprocess.py
       ▼
  Global feature vector      ← 18-element vector: Re, Pr, Sc, Ma, …
       │
       ▼
  PCGMResult                 ← accepted/rejected flag + inference-ready dict

Usage
-----
    from geometry.pcgm import generate, ProcessParams, PCGMResult
    from geometry.grammar import NozzlePattern, ShowerheadTopology

    # Minimal: use all defaults (N2 at 300 K, hex pattern, 300 mm plate)
    result = generate({"D": 0.002, "pitch_over_D": 3.0}, {"flow_rate_slm": 2.0})
    if result.accepted:
        coords       = result.inference_data["coords"]          # [N, 3]
        node_feats   = result.inference_data["node_features"]   # [N, 4]
        global_feats = result.inference_data["global_features"] # [18]
    else:
        print("Rejected:", result.reason)

    # Full control
    from physics.guardrails import GuardrailBounds
    bounds = GuardrailBounds(Re=(1.0, 2000.0), Da=(0.01, 50.0))
    result = generate(
        geo_params     = {"D": 0.001, "pitch_over_D": 4.0, "H_plenum": 0.025},
        process_params = {"flow_rate_slm": 5.0, "beta": 0.08},
        bounds         = bounds,
        pattern        = NozzlePattern.RING,
        n_points       = 80_000,
    )
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np

from geometry.grammar import (
    DEFAULT_TOPOLOGY, NozzlePattern, ShowerheadTopology, default_params,
)
from geometry.parametric import ShowerheadGeometry, build_showerhead
from physics.calculator import compute_all, k_rxn_from_sticking
from physics.guardrails import CheckResult, GuardrailBounds, GuardrailEngine


# ══════════════════════════════════════════════════════════════════════════
# Default fluid properties — N2 (nitrogen) at 300 K, 1 atm
# ══════════════════════════════════════════════════════════════════════════

N2_DEFAULTS = {
    "rho":     1.145,    # density                [kg/m³]
    "mu":      1.76e-5,  # dynamic viscosity       [Pa·s]
    "cp":      1040.0,   # specific heat           [J/(kg·K)]
    "k_fluid": 0.026,    # thermal conductivity    [W/(m·K)]
    "a":       353.0,    # speed of sound          [m/s]  (N2 at 300 K)
}

# TMA (trimethylaluminium) in N2 at 300 K
TMA_DEFAULTS = {
    "D_m":  2.5e-5,   # mass diffusivity of TMA in N2  [m²/s]
    "beta": 0.05,     # sticking coefficient            [-]
    "v_th": 144.0,    # TMA mean thermal velocity       [m/s]
}

# Global feature column order — must match postprocess.py GLOBAL_COLS exactly
GLOBAL_COLS = [
    "Re", "Pr", "Sc", "Ma", "Pe_h", "Pe_m", "Da",
    "D", "pitch_over_D", "H_plenum", "t_face", "standoff",
    "flow_rate_slm", "beta", "v_th", "D_m",
    "n_holes", "open_area_frac",
]


# ══════════════════════════════════════════════════════════════════════════
# Result data structure
# ══════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class PCGMResult:
    """
    Output of the PCGM pipeline for one design candidate.

    Attributes
    ----------
    accepted         : True if guardrail checks passed and mesh was generated
    reason           : Human-readable reason for rejection (empty if accepted)
    geometry         : ShowerheadGeometry with hole positions and mesh
                       (None if rejected at pre-mesh stage)
    dim_nums         : Dict of all computed dimensionless numbers
                       (Re, Pr, Sc, Ma, Pe_h, Pe_m, Da, …)
    guardrail_result : Full CheckResult from GuardrailEngine
    confidence       : Guardrail confidence score 0–1
    inference_data   : Dict ready for MultiHeadMGN inference:
                         "coords"          [N, 3]  float32
                         "node_features"   [N, 4]  float32
                         "global_features" [18]    float32
                       (None if rejected)
    params_used      : Final merged geometry + process parameter dict

    Notes
    -----
    node_features columns (matching postprocess.py build_node_features):
        col 0: r_norm  — radial distance / (D_plate/2)      range [0, 1]
        col 1: z_norm  — height from wafer / z_total        range [0, 1]
        col 2: region  — 0=standoff, 1=faceplate, 2=plenum
        col 3: bc_mask — boundary-condition flag (0 = interior)
    """
    accepted:         bool
    reason:           str
    geometry:         Optional[ShowerheadGeometry]
    dim_nums:         dict
    guardrail_result: Optional[CheckResult]
    confidence:       float
    inference_data:   Optional[dict]
    params_used:      dict

    def summary(self) -> str:
        lines = ["── PCGMResult " + ("✓ ACCEPTED" if self.accepted else "✗ REJECTED") + " ──"]
        if not self.accepted:
            lines.append(f"  Reason: {self.reason}")
        lines.append(f"  Confidence: {self.confidence:.3f}")
        if self.geometry:
            lines.append(self.geometry.summary())
        lines.append("  Dimensionless numbers:")
        for k, v in self.dim_nums.items():
            lines.append(f"    {k:6s} = {v:.4g}")
        if self.guardrail_result and self.guardrail_result.violations:
            lines.append("  Violations:")
            for v in self.guardrail_result.violations:
                lines.append(f"    [{v.reason_code}] {v.message}")
        if self.inference_data:
            N = len(self.inference_data["coords"])
            lines.append(f"  Point cloud: {N:,} nodes")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# Physics computation helpers
# ══════════════════════════════════════════════════════════════════════════

def _nozzle_velocity(flow_rate_slm: float, D: float, n_holes: int) -> float:
    """
    Mean nozzle exit velocity from volumetric flow rate.

    V = Q / (n_holes × π × (D/2)²)

    Parameters
    ----------
    flow_rate_slm : flow rate in standard litres per minute  [slm]
    D             : nozzle hole diameter                      [m]
    n_holes       : number of nozzle holes                    [-]

    Returns
    -------
    V : mean nozzle exit velocity  [m/s]
    """
    Q_m3s   = flow_rate_slm * 1.667e-5        # slm → m³/s
    A_total = n_holes * np.pi * (D / 2.0) ** 2
    if A_total <= 0:
        raise ValueError("Total nozzle area is zero — check D and n_holes.")
    return float(Q_m3s / A_total)


def _compute_dim_nums(geo_params: dict, process_params: dict,
                      n_holes: int) -> dict:
    """
    Compute all applicable dimensionless numbers for a design.

    Returns a flat dict {symbol: value}.
    """
    D            = geo_params["D"]
    flow_slm     = process_params["flow_rate_slm"]
    rho          = process_params["rho"]
    mu           = process_params["mu"]
    cp           = process_params["cp"]
    k_fluid      = process_params["k_fluid"]
    D_m          = process_params["D_m"]
    beta         = process_params["beta"]
    v_th         = process_params["v_th"]
    a            = process_params.get("a", 353.0)

    V     = _nozzle_velocity(flow_slm, D, n_holes)
    k_rxn = k_rxn_from_sticking(beta, v_th)

    calc_params = {
        "rho":     rho,
        "V":       V,
        "L":       D,       # characteristic length = nozzle diameter
        "mu":      mu,
        "cp":      cp,
        "k_fluid": k_fluid,
        "D_m":     D_m,
        "a":       a,
        "k_rxn":   k_rxn,
    }
    dim = compute_all(calc_params)
    dim["V_nozzle"] = V  # store for reference (not a dimensionless number)
    return dim


def _build_global_features(geo: ShowerheadGeometry,
                            process_params: dict,
                            dim_nums: dict) -> np.ndarray:
    """
    Build the 18-element global feature vector matching GLOBAL_COLS / postprocess.py.

    Returns
    -------
    gf : [18] float32 array
    """
    p = geo.params
    vals = [
        dim_nums.get("Re",   0.0),
        dim_nums.get("Pr",   0.0),
        dim_nums.get("Sc",   0.0),
        dim_nums.get("Ma",   0.0),
        dim_nums.get("Pe_h", 0.0),
        dim_nums.get("Pe_m", 0.0),
        dim_nums.get("Da",   0.0),
        p["D"],
        p["pitch_over_D"],
        p["H_plenum"],
        p["t_face"],
        p["standoff"],
        process_params["flow_rate_slm"],
        process_params["beta"],
        process_params["v_th"],
        process_params["D_m"],
        float(geo.n_holes),
        float(geo.open_area_frac),
    ]
    return np.array(vals, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════
# Point-cloud generator
# ══════════════════════════════════════════════════════════════════════════

def _sample_plenum(geo: ShowerheadGeometry,
                   n_points: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample `n_points` random points inside the PLENUM region of the showerhead.

    The plenum is the cylindrical volume between the top face of the faceplate
    and the gas inlet plane:
        z ∈ [standoff + t_face,  standoff + t_face + H_plenum]
        r ≤ D_plate / 2

    Node features (4 columns, matching postprocess.py build_node_features):
        col 0: r_norm  = r / (D_plate/2)
        col 1: z_norm  = z / z_total          where z_total = standoff+t_face+H_plenum
        col 2: region  = 2.0                   (all points are in the plenum)
        col 3: bc_mask = 0.0                   (interior, no boundary condition)

    Parameters
    ----------
    geo      : ShowerheadGeometry
    n_points : target number of sample points
    rng      : numpy random Generator

    Returns
    -------
    coords        : [N, 3]  float32   (x, y, z) in metres
    node_features : [N, 4]  float32
    """
    p        = geo.params
    D_plate  = p["D_plate"]
    R_plate  = D_plate / 2.0
    standoff = p["standoff"]
    t_face   = p["t_face"]
    H_plenum = p["H_plenum"]

    z_bot   = standoff + t_face            # bottom of plenum (top of faceplate)
    z_top   = z_bot + H_plenum             # top of plenum (inlet plane)
    z_total = z_top                        # = standoff + t_face + H_plenum

    # ── Rejection sampling: uniform in bounding box, keep points inside cylinder ──
    # We need exactly n_points inside the circle; oversample and trim.
    oversample = int(n_points * (4 / np.pi) * 1.1) + 100   # ~40% overhead
    x_raw = rng.uniform(-R_plate, R_plate, oversample)
    y_raw = rng.uniform(-R_plate, R_plate, oversample)
    z_raw = rng.uniform(z_bot, z_top, oversample)

    r_raw = np.sqrt(x_raw ** 2 + y_raw ** 2)
    inside = r_raw <= R_plate
    x = x_raw[inside][:n_points]
    y = y_raw[inside][:n_points]
    z = z_raw[inside][:n_points]
    r = r_raw[inside][:n_points]

    # If rejection sampling didn't yield enough points, fill with extra
    while len(x) < n_points:
        need = n_points - len(x)
        x2 = rng.uniform(-R_plate, R_plate, need * 2)
        y2 = rng.uniform(-R_plate, R_plate, need * 2)
        z2 = rng.uniform(z_bot, z_top, need * 2)
        r2 = np.sqrt(x2 ** 2 + y2 ** 2)
        ok = r2 <= R_plate
        x = np.concatenate([x, x2[ok][:need]])
        y = np.concatenate([y, y2[ok][:need]])
        z = np.concatenate([z, z2[ok][:need]])
        r = np.concatenate([r, r2[ok][:need]])

    N      = len(x)
    coords = np.column_stack([x, y, z]).astype(np.float32)

    # Node features
    r_norm  = (r / R_plate).astype(np.float32)
    z_norm  = (z / (z_total + 1e-12)).astype(np.float32)
    region  = np.full(N, 2.0, dtype=np.float32)    # 2 = plenum
    bc_mask = np.zeros(N, dtype=np.float32)         # 0 = interior

    node_features = np.column_stack([r_norm, z_norm, region, bc_mask])
    return coords, node_features


# ══════════════════════════════════════════════════════════════════════════
# Main public API
# ══════════════════════════════════════════════════════════════════════════

def generate(
    geo_params:     Optional[dict] = None,
    process_params: Optional[dict] = None,
    bounds:         Optional[GuardrailBounds] = None,
    pattern:        NozzlePattern = NozzlePattern.HEX,
    n_points:       int = 80_000,
    rng_seed:       int = 42,
) -> PCGMResult:
    """
    Full PCGM pipeline: design parameters → guardrail check → geometry →
    plenum point-cloud → HDF5-ready inference dict.

    Parameters
    ----------
    geo_params : dict, optional
        Geometry parameters.  Missing keys filled from grammar.default_params().
        Keys: D, pitch_over_D, H_plenum, t_face, standoff, D_plate, theta_deg, Cd.

    process_params : dict, optional
        Process / fluid parameters.  Missing keys filled from N2_DEFAULTS + TMA_DEFAULTS.
        Keys: flow_rate_slm, rho, mu, cp, k_fluid, D_m, beta, v_th, a.

    bounds : GuardrailBounds, optional
        Physics guardrail bounds.  Uses GuardrailEngine defaults if None.

    pattern : NozzlePattern
        Hole-array pattern: NozzlePattern.HEX (default), RING, or SQUARE.

    n_points : int
        Number of sample points to generate in the plenum (default 80,000).
        The MultiHeadMGN surrogate was trained on ~250,000 points per case;
        80,000–150,000 gives good inference quality without excessive memory.

    rng_seed : int
        Random seed for reproducible point-cloud sampling.

    Returns
    -------
    PCGMResult
        .accepted        — True if guardrails pass
        .reason          — rejection reason (empty string if accepted)
        .geometry        — ShowerheadGeometry
        .dim_nums        — computed dimensionless numbers
        .guardrail_result — full CheckResult
        .confidence      — guardrail confidence score
        .inference_data  — {"coords", "node_features", "global_features"}
        .params_used     — merged parameter dict

    Examples
    --------
    # Basic usage
    result = generate({"D": 0.002, "pitch_over_D": 3.0}, {"flow_rate_slm": 2.0})

    # From optimizer Pareto candidate
    candidate = {"D_mm": 2.0, "pitch_over_D": 3.0, "Q_slm": 1.0}
    result = generate(
        geo_params     = {"D": candidate["D_mm"] / 1000, "pitch_over_D": candidate["pitch_over_D"]},
        process_params = {"flow_rate_slm": candidate["Q_slm"]},
    )
    """
    rng = np.random.default_rng(rng_seed)

    # ── Merge parameters with defaults ────────────────────────────────────
    geo_full  = default_params()
    if geo_params:
        geo_full.update(geo_params)

    proc_full = {}
    proc_full.update(N2_DEFAULTS)
    proc_full.update(TMA_DEFAULTS)
    if process_params:
        proc_full.update(process_params)

    # Ensure flow_rate_slm is present
    if "flow_rate_slm" not in proc_full:
        proc_full["flow_rate_slm"] = 2.0

    params_used = {**geo_full, **proc_full}

    # ── Step 1: preliminary geometry (need n_holes for physics) ───────────
    topology = ShowerheadTopology(pattern=pattern)
    try:
        geo = build_showerhead(geo_full, topology)
    except Exception as exc:
        return PCGMResult(
            accepted=False,
            reason=f"Geometry build failed: {exc}",
            geometry=None,
            dim_nums={},
            guardrail_result=None,
            confidence=0.0,
            inference_data=None,
            params_used=params_used,
        )

    # ── Step 2: compute dimensionless numbers ─────────────────────────────
    try:
        dim_nums = _compute_dim_nums(geo_full, proc_full, geo.n_holes)
    except Exception as exc:
        return PCGMResult(
            accepted=False,
            reason=f"Physics computation failed: {exc}",
            geometry=geo,
            dim_nums={},
            guardrail_result=None,
            confidence=0.0,
            inference_data=None,
            params_used=params_used,
        )

    # ── Step 3: pre-mesh guardrail check ──────────────────────────────────
    engine = GuardrailEngine(bounds or GuardrailBounds())
    # Pass only the dimensionless numbers the engine knows about
    check_vals = {k: v for k, v in dim_nums.items()
                  if k in {"Re", "Ma", "Eu", "Pr", "Nu", "Bi", "Sc", "Sh", "Pe_h", "Pe_m", "Da"}}
    gr_result = engine.check(check_vals)

    if not gr_result.passed:
        violation_summary = "; ".join(
            f"{v.symbol}={v.value:.4g} (allowed [{v.lo:.4g},{v.hi:.4g}])"
            for v in gr_result.violations
        )
        return PCGMResult(
            accepted=False,
            reason=f"Guardrail violation — {violation_summary}",
            geometry=geo,
            dim_nums=dim_nums,
            guardrail_result=gr_result,
            confidence=gr_result.confidence,
            inference_data=None,
            params_used=params_used,
        )

    # ── Step 4: generate plenum point cloud ───────────────────────────────
    coords, node_features = _sample_plenum(geo, n_points, rng)

    # ── Step 5: assemble global feature vector ────────────────────────────
    global_features = _build_global_features(geo, proc_full, dim_nums)

    inference_data = {
        "coords":          coords,           # [N, 3]  float32
        "node_features":   node_features,    # [N, 4]  float32
        "global_features": global_features,  # [18]    float32
    }

    return PCGMResult(
        accepted=True,
        reason="",
        geometry=geo,
        dim_nums=dim_nums,
        guardrail_result=gr_result,
        confidence=gr_result.confidence,
        inference_data=inference_data,
        params_used=params_used,
    )


# ══════════════════════════════════════════════════════════════════════════
# Batch helper: generate multiple designs from optimizer Pareto candidates
# ══════════════════════════════════════════════════════════════════════════

def from_optimizer_candidates(
    candidates:     list[dict],
    process_params: Optional[dict] = None,
    bounds:         Optional[GuardrailBounds] = None,
    pattern:        NozzlePattern = NozzlePattern.HEX,
    n_points:       int = 80_000,
) -> list[PCGMResult]:
    """
    Convert a list of optimizer Pareto candidates (from 07_optimizer.ipynb)
    to PCGMResults.

    Each candidate dict is expected to have keys matching the optimizer output:
        D_mm         : nozzle diameter  [mm]
        pitch_over_D : pitch / D        [-]
        Q_slm        : flow rate        [slm]

    Parameters
    ----------
    candidates     : list of dicts from optimizer_results.json top_candidates
    process_params : override fluid/process properties (uses N2+TMA defaults)
    bounds         : guardrail bounds (uses defaults if None)
    pattern        : nozzle array pattern
    n_points       : point-cloud size per design

    Returns
    -------
    list of PCGMResult, one per candidate

    Example
    -------
    import json
    with open("optimizer_results.json") as f:
        opt = json.load(f)
    results = from_optimizer_candidates(opt["top_candidates"])
    accepted = [r for r in results if r.accepted]
    print(f"{len(accepted)}/{len(results)} designs passed guardrails")
    """
    results = []
    for i, cand in enumerate(candidates):
        # Convert optimizer output format → PCGM geo_params
        D_m = cand.get("D_mm", cand.get("D", 2.0))
        if D_m > 0.1:          # value is in mm, convert to m
            D_m = D_m / 1000.0

        geo_p = {
            "D":            float(D_m),
            "pitch_over_D": float(cand.get("pitch_over_D", 4.0)),
        }
        proc_p = dict(process_params or {})
        if "Q_slm" in cand:
            proc_p["flow_rate_slm"] = float(cand["Q_slm"])

        result = generate(
            geo_params=geo_p, process_params=proc_p,
            bounds=bounds, pattern=pattern,
            n_points=n_points, rng_seed=i,
        )
        results.append(result)
        status = "PASS" if result.accepted else "FAIL"
        print(f"  [{status}] Candidate {i+1}: D={D_m*1e3:.1f}mm "
              f"pitch/D={geo_p['pitch_over_D']:.1f} "
              f"Q={proc_p.get('flow_rate_slm', 2.0):.1f}slm "
              f"confidence={result.confidence:.3f}")
    return results


# ══════════════════════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════════════════════

def _run_tests():
    print("Running PCGM (Physics-Constrained Geometric Morphogenesis) tests...")

    # 1. Default generate — should be accepted
    r = generate()
    assert r.accepted, f"Default design should be accepted, got: {r.reason}"
    assert r.geometry is not None
    assert r.geometry.n_holes > 0
    assert r.inference_data is not None
    coords = r.inference_data["coords"]
    nf     = r.inference_data["node_features"]
    gf     = r.inference_data["global_features"]
    assert coords.shape[1] == 3,  f"coords should be [N,3], got {coords.shape}"
    assert nf.shape[1] == 4,      f"node_features should be [N,4], got {nf.shape}"
    assert len(gf) == 18,         f"global_features should be [18], got {len(gf)}"
    assert coords.dtype == np.float32
    assert nf.dtype == np.float32
    print(f"  [OK] Default design: {r.geometry.n_holes} holes, {len(coords):,} points")

    # 2. Node feature ranges
    r_norm = nf[:, 0]; z_norm = nf[:, 1]
    region = nf[:, 2]; bc     = nf[:, 3]
    assert r_norm.min() >= 0 and r_norm.max() <= 1.01, "r_norm out of range"
    assert z_norm.min() >= 0 and z_norm.max() <= 1.01, "z_norm out of range"
    assert np.all(region == 2.0), "All plenum points should have region=2"
    assert np.all(bc == 0.0),     "bc_mask should be 0 for interior points"
    print("  [OK] Node feature ranges valid")

    # 3. Global features — dimensionless numbers are positive
    for i, col in enumerate(GLOBAL_COLS[:7]):   # Re, Pr, Sc, Ma, Pe_h, Pe_m, Da
        assert gf[i] > 0, f"{col} should be > 0, got {gf[i]}"
    print(f"  [OK] Global features: Re={gf[0]:.1f} Da={gf[6]:.4f} n_holes={gf[16]:.0f}")

    # 4. Rejection test — set an impossibly tight Re window so default design fails
    r_reject = generate(
        bounds=GuardrailBounds(Re=(0.001, 0.01)),  # default Re ≈ 1.1 >> 0.01
    )
    assert not r_reject.accepted, "Design should be rejected by tight Re bounds"
    assert len(r_reject.guardrail_result.violations) > 0
    print(f"  [OK] Tight-bounds rejection works: {r_reject.reason[:60]}")

    # 5. Pattern variants
    for pat in [NozzlePattern.HEX, NozzlePattern.RING, NozzlePattern.SQUARE]:
        r_pat = generate({"D": 0.002, "pitch_over_D": 4.0},
                         {"flow_rate_slm": 2.0}, pattern=pat, n_points=5_000)
        assert r_pat.accepted, f"Pattern {pat.value} should be accepted"
        print(f"  [OK] Pattern {pat.value}: {r_pat.geometry.n_holes} holes")

    # 6. from_optimizer_candidates
    fake_candidates = [
        {"D_mm": 2.0, "pitch_over_D": 3.0, "Q_slm": 1.0, "pareto_rank": 1},
        {"D_mm": 1.5, "pitch_over_D": 4.0, "Q_slm": 3.0, "pareto_rank": 2},
    ]
    results = from_optimizer_candidates(fake_candidates, n_points=5_000)
    assert len(results) == 2
    print(f"  [OK] from_optimizer_candidates: {sum(r.accepted for r in results)}/2 accepted")

    print("\nAll PCGM tests passed.")
    print(generate().summary())


if __name__ == "__main__":
    _run_tests()
