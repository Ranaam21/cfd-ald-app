"""
openfoam/param_sweep.py

Generates and (optionally) runs a parametric sweep of aldFoam showerhead cases.

Default sweep: 5 geometry × 4 process parameters = 80 cases
Start small with --dry_run or --max_cases 5 to verify the setup first.

Usage
-----
    # Preview what would be generated (no files written)
    python3 openfoam/param_sweep.py --dry_run

    # Generate case directories only (no OpenFOAM run)
    python3 openfoam/param_sweep.py --generate_only --out_dir openfoam/cases

    # Generate + run all cases sequentially (requires OpenFOAM installed)
    python3 openfoam/param_sweep.py --out_dir openfoam/cases --n_parallel 4

    # Quick 5-case test
    python3 openfoam/param_sweep.py --max_cases 5 --generate_only
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry.grammar import DEFAULT_TOPOLOGY, NozzlePattern, ShowerheadTopology
from geometry.parametric import build_showerhead
from geometry.quality_check import check_geometry
from openfoam.case_generator import generate_case


# ══════════════════════════════════════════════════════════════════════════
# Sweep parameter grid
# ══════════════════════════════════════════════════════════════════════════

# Geometry axes — vary showerhead design
GEOM_SWEEP = {
    "D":            [0.001, 0.0015, 0.002, 0.0025, 0.003],   # nozzle diameters [m]
    "pitch_over_D": [3.0, 4.0, 5.0, 6.0],                    # pitch / D
}

# Shared geometry params (fixed across sweep)
GEOM_BASE = {
    "H_plenum":  0.020,   # 20 mm plenum
    "t_face":    0.003,   # 3 mm faceplate
    "standoff":  0.020,   # 20 mm standoff
    "D_plate":   0.300,   # 300 mm wafer
    "theta_deg": 0.0,     # straight nozzles
}

# Process axes — vary operating conditions
PROCESS_SWEEP = {
    "flow_rate_slm": [1.0, 2.0, 5.0, 10.0],
    "beta":          [0.01, 0.05, 0.1, 0.5],
}

# Fixed process params
PROCESS_BASE = {
    "v_th":       144.0,    # TMA thermal velocity [m/s]
    "D_m":        2.5e-5,   # TMA diffusivity in N2 [m²/s]
    "pulse_time": 0.10,     # 100 ms pulse
    "purge_time": 0.10,     # 100 ms purge
    "dt":         1e-4,
}


# ══════════════════════════════════════════════════════════════════════════
# Case list builder
# ══════════════════════════════════════════════════════════════════════════

def build_case_list(max_cases: int = None) -> list:
    """
    Returns a list of dicts, one per case:
        {"case_name": str, "geo_params": dict, "process_params": dict}
    """
    cases = []
    g_keys = list(GEOM_SWEEP.keys())
    g_vals = list(GEOM_SWEEP.values())
    p_keys = list(PROCESS_SWEEP.keys())
    p_vals = list(PROCESS_SWEEP.values())

    idx = 0
    for g_combo in itertools.product(*g_vals):
        geo_params = dict(GEOM_BASE)
        geo_params.update(dict(zip(g_keys, g_combo)))

        for p_combo in itertools.product(*p_vals):
            proc_params = dict(PROCESS_BASE)
            proc_params.update(dict(zip(p_keys, p_combo)))

            D_mm   = geo_params["D"] * 1e3
            pitch  = geo_params["pitch_over_D"]
            Q      = proc_params["flow_rate_slm"]
            b      = proc_params["beta"]
            name   = f"case_{idx:04d}_D{D_mm:.1f}mm_p{pitch:.1f}_Q{Q:.1f}slm_b{b:.3f}"

            cases.append({
                "case_name":     name,
                "case_idx":      idx,
                "geo_params":    geo_params,
                "process_params": proc_params,
            })
            idx += 1

    if max_cases:
        cases = cases[:max_cases]

    return cases


# ══════════════════════════════════════════════════════════════════════════
# Single case runner
# ══════════════════════════════════════════════════════════════════════════

def run_single_case(case_info: dict, out_dir: Path,
                    generate_only: bool = False) -> dict:
    """
    Generate + optionally run one OpenFOAM case.

    Returns a status dict with keys: name, status, case_dir, error.
    """
    name         = case_info["case_name"]
    geo_params   = case_info["geo_params"]
    proc_params  = case_info["process_params"]
    case_dir     = out_dir / name

    result = {"name": name, "case_dir": str(case_dir), "status": "pending", "error": ""}

    # ── Build geometry ────────────────────────────────────────────────────
    try:
        topology = ShowerheadTopology(pattern=NozzlePattern.HEX)
        geo      = build_showerhead(geo_params, topology)
    except Exception as e:
        result["status"] = "failed"
        result["error"]  = f"Geometry build failed: {e}"
        return result

    # ── Quality check ─────────────────────────────────────────────────────
    report = check_geometry(geo)
    if not report.passed:
        errors = [i.message for i in report.issues if i.level == "error"]
        result["status"] = "skipped_quality"
        result["error"]  = " | ".join(errors)
        print(f"  [SKIP] {name}: quality check failed — {result['error']}")
        return result

    # ── Generate case directory ───────────────────────────────────────────
    try:
        generate_case(
            geo=geo,
            case_dir=str(case_dir),
            flow_rate_slm=proc_params["flow_rate_slm"],
            beta=proc_params["beta"],
            v_th=proc_params["v_th"],
            D_m=proc_params["D_m"],
            pulse_time=proc_params["pulse_time"],
            purge_time=proc_params["purge_time"],
            dt=proc_params["dt"],
        )
        result["status"] = "generated"
    except Exception as e:
        result["status"] = "failed"
        result["error"]  = f"Case generation failed: {e}"
        return result

    if generate_only:
        return result

    # ── Run OpenFOAM ─────────────────────────────────────────────────────
    try:
        proc = subprocess.run(
            ["bash", "run.sh"],
            cwd=str(case_dir),
            capture_output=True,
            text=True,
            timeout=3600,   # 1 hour timeout per case
        )
        if proc.returncode == 0:
            result["status"] = "completed"
        else:
            result["status"] = "foam_failed"
            result["error"]  = proc.stderr[-500:]   # last 500 chars of stderr
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["error"]  = "Case exceeded 1 hour timeout."
    except FileNotFoundError:
        result["status"] = "no_openfoam"
        result["error"]  = (
            "OpenFOAM not found. Install via Homebrew:\n"
            "  brew install --cask openfoam"
        )

    return result


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",       default="openfoam/cases")
    p.add_argument("--max_cases",     type=int, default=None,
                   help="Limit total cases (useful for testing)")
    p.add_argument("--dry_run",       action="store_true",
                   help="Print case list only, write nothing")
    p.add_argument("--generate_only", action="store_true",
                   help="Write case dirs but do not run OpenFOAM")
    p.add_argument("--n_parallel",    type=int, default=1,
                   help="Number of cases to run in parallel (requires GNU parallel)")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    cases   = build_case_list(args.max_cases)

    print(f"Sweep: {len(cases)} cases  →  {out_dir}")
    print(f"Mode : {'dry_run' if args.dry_run else ('generate_only' if args.generate_only else 'run')}")

    if args.dry_run:
        for c in cases:
            print(f"  {c['case_name']}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    statuses = []
    for i, case_info in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] {case_info['case_name']}")
        status = run_single_case(case_info, out_dir, args.generate_only)
        statuses.append(status)
        print(f"  → {status['status']}"
              + (f": {status['error']}" if status["error"] else ""))

    # ── Summary ───────────────────────────────────────────────────────────
    counts = {}
    for s in statuses:
        counts[s["status"]] = counts.get(s["status"], 0) + 1

    print("\n── Sweep summary ──")
    for k, v in sorted(counts.items()):
        print(f"  {k:20s}: {v}")

    summary_path = out_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"cases": statuses, "counts": counts}, f, indent=2)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
