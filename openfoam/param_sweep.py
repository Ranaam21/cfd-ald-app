"""
openfoam/param_sweep.py

Generates and runs a parametric sweep of reactingFoam showerhead cases via Docker.

Default first run: 80 cases (5D × 4pitch × 4Q, beta fixed at 0.05)
Full sweep:        320 cases (add --max_cases 320 --beta_values 0.01,0.05,0.1,0.5)

Usage
-----
    # Preview what would be generated (no files written)
    python3 openfoam/param_sweep.py --dry_run

    # Generate case directories only (no OpenFOAM run)
    python3 openfoam/param_sweep.py --generate_only --out_dir openfoam/cases

    # Generate + run 80 cases in parallel (6 Docker containers)
    python3 openfoam/param_sweep.py --out_dir openfoam/cases --n_parallel 6

    # Full 320-case sweep (2 overnight runs)
    python3 openfoam/param_sweep.py --out_dir openfoam/cases --n_parallel 6 --max_cases 320
"""

import argparse
import itertools
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from geometry.grammar import DEFAULT_TOPOLOGY, NozzlePattern, ShowerheadTopology
from geometry.parametric import build_showerhead
from geometry.quality_check import check_geometry
from openfoam.case_generator import generate_case


# ══════════════════════════════════════════════════════════════════════════
# Sweep parameter grid
# ══════════════════════════════════════════════════════════════════════════

# Geometry axes
GEOM_SWEEP = {
    "D":            [0.001, 0.0015, 0.002, 0.0025, 0.003],   # nozzle diameters [m]
    "pitch_over_D": [3.0, 4.0, 5.0, 6.0],                    # pitch / D
}

# Shared geometry params (fixed across sweep)
GEOM_BASE = {
    "H_plenum":  0.020,
    "t_face":    0.003,
    "standoff":  0.020,
    "D_plate":   0.300,
    "theta_deg": 0.0,
}

# Process axes — beta fixed at 0.05 for first 80-case run; expand later
PROCESS_SWEEP = {
    "flow_rate_slm": [1.0, 2.0, 5.0, 10.0],
    "beta":          [0.05],
}

# Fixed process params
PROCESS_BASE = {
    "v_th":       144.0,
    "D_m":        2.5e-5,
    "pulse_time": 0.10,
    "purge_time": 0.10,
    "dt":         1e-4,
}

DOCKER_IMAGE = "opencfd/openfoam-default:latest"
CASE_TIMEOUT = 14400  # 4 hours per case (high-Q cases need longer)


# ══════════════════════════════════════════════════════════════════════════
# Case list builder
# ══════════════════════════════════════════════════════════════════════════

def build_case_list(max_cases: int = None) -> list:
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

            D_mm  = geo_params["D"] * 1e3
            pitch = geo_params["pitch_over_D"]
            Q     = proc_params["flow_rate_slm"]
            b     = proc_params["beta"]
            name  = f"case_{idx:04d}_D{D_mm:.1f}mm_p{pitch:.1f}_Q{Q:.1f}slm_b{b:.3f}"

            cases.append({
                "case_name":      name,
                "case_idx":       idx,
                "geo_params":     geo_params,
                "process_params": proc_params,
            })
            idx += 1

    if max_cases:
        cases = cases[:max_cases]

    return cases


# ══════════════════════════════════════════════════════════════════════════
# Case generation
# ══════════════════════════════════════════════════════════════════════════

def generate_one(case_info: dict, out_dir: Path) -> dict:
    name       = case_info["case_name"]
    geo_params = case_info["geo_params"]
    proc       = case_info["process_params"]
    case_dir   = out_dir / name

    result = {"name": name, "case_dir": str(case_dir), "status": "pending", "error": ""}

    try:
        topology = ShowerheadTopology(pattern=NozzlePattern.HEX)
        geo      = build_showerhead(geo_params, topology)
    except Exception as e:
        result.update(status="failed", error=f"Geometry build failed: {e}")
        return result

    report = check_geometry(geo)
    if not report.passed:
        errors = [i.message for i in report.issues if i.level == "error"]
        result.update(status="skipped_quality", error=" | ".join(errors))
        return result

    try:
        generate_case(
            geo=geo,
            case_dir=str(case_dir),
            flow_rate_slm=proc["flow_rate_slm"],
            beta=proc["beta"],
            v_th=proc["v_th"],
            D_m=proc["D_m"],
            pulse_time=proc["pulse_time"],
            purge_time=proc["purge_time"],
            dt=proc["dt"],
        )
        result["status"] = "generated"
    except Exception as e:
        result.update(status="failed", error=f"Case generation failed: {e}")

    return result


# ══════════════════════════════════════════════════════════════════════════
# Docker runner (single case)
# ══════════════════════════════════════════════════════════════════════════

def _is_completed(case_dir: Path) -> bool:
    """Case is done if a non-zero time directory exists with a U field."""
    for d in case_dir.iterdir():
        if d.is_dir() and d.name != "0":
            try:
                t = float(d.name)
                if t > 0 and (d / "U").exists():
                    return True
            except ValueError:
                pass
    return False


def run_case_docker(case_dir: Path, verbose: bool = False) -> dict:
    """Run a single case inside a Docker container. Skips if already done."""
    name = case_dir.name

    if not case_dir.exists():
        return {"name": name, "status": "not_found", "error": "Case directory missing"}

    if _is_completed(case_dir):
        print(f"  [SKIP] {name}: already completed")
        return {"name": name, "status": "completed_cached"}

    lock = case_dir / ".running"
    lock.touch()

    print(f"  [RUN ] {name}")
    try:
        proc = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{case_dir.resolve()}:/case",
                DOCKER_IMAGE,
                "bash", "-c", "cd /case && bash run.sh"
            ],
            capture_output=not verbose,
            text=True,
            timeout=CASE_TIMEOUT,
        )
        lock.unlink(missing_ok=True)

        if proc.returncode == 0:
            print(f"  [DONE] {name}")
            return {"name": name, "status": "completed"}
        else:
            err = (proc.stderr or "")[-800:]
            print(f"  [FAIL] {name}: {err[-200:]}")
            return {"name": name, "status": "foam_failed", "error": err}

    except subprocess.TimeoutExpired:
        lock.unlink(missing_ok=True)
        print(f"  [TIMEOUT] {name}")
        return {"name": name, "status": "timeout", "error": "Exceeded 2h timeout"}

    except Exception as e:
        lock.unlink(missing_ok=True)
        return {"name": name, "status": "error", "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
# Parallel sweep runner
# ══════════════════════════════════════════════════════════════════════════

def run_sweep_parallel(case_dirs: list, n_parallel: int = 6) -> list:
    """Run all cases with up to n_parallel Docker containers at once."""
    results = []
    print(f"\nRunning {len(case_dirs)} cases with {n_parallel} parallel Docker containers\n")

    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = {executor.submit(run_case_docker, d): d for d in case_dirs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    return results


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",       default="openfoam/cases")
    p.add_argument("--max_cases",     type=int, default=80,
                   help="Cases to run (default 80; use 320 for full sweep)")
    p.add_argument("--n_parallel",    type=int, default=6,
                   help="Parallel Docker containers (default 6)")
    p.add_argument("--dry_run",       action="store_true",
                   help="Print case list only, write nothing")
    p.add_argument("--generate_only", action="store_true",
                   help="Write case dirs but do not run OpenFOAM")
    p.add_argument("--retry_failed",  action="store_true",
                   help="Only re-run timeout/failed cases from sweep_summary.json")
    p.add_argument("--verbose",       action="store_true",
                   help="Stream Docker output to console")
    return p.parse_args()


def _load_failed_cases(out_dir: Path) -> list:
    """Read sweep_summary.json and return names of timeout/failed cases."""
    summary_path = out_dir / "sweep_summary.json"
    if not summary_path.exists():
        print(f"ERROR: no sweep_summary.json in {out_dir}")
        return []
    data = json.load(open(summary_path))
    failed = [c["name"] for c in data["cases"]
              if c["status"] in ("timeout", "foam_failed", "error", "no_fields")]
    print(f"Found {len(failed)} failed/timeout case(s) to retry:")
    for name in failed:
        print(f"  {name}")
    return failed


def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)

    # ── Retry-failed mode: skip generation, just re-run specific cases ────
    if args.retry_failed:
        failed_names = _load_failed_cases(out_dir)
        if not failed_names:
            print("Nothing to retry.")
            return
        case_dirs = [out_dir / name for name in failed_names
                     if (out_dir / name).exists()]
        run_results = run_sweep_parallel(case_dirs, n_parallel=args.n_parallel)
        _write_summary(out_dir, run_results)
        return

    cases = build_case_list(args.max_cases)
    total = len(cases)
    print(f"Sweep : {total} cases  →  {out_dir}")
    print(f"Mode  : {'dry_run' if args.dry_run else ('generate_only' if args.generate_only else f'run ({args.n_parallel} parallel)')}")

    if args.dry_run:
        for c in cases:
            print(f"  {c['case_name']}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: generate all case directories ────────────────────────────
    print(f"\n── Generating {total} case directories ──")
    gen_results = []
    for i, case_info in enumerate(cases):
        r = generate_one(case_info, out_dir)
        gen_results.append(r)
        status_icon = "✓" if r["status"] == "generated" else "✗"
        print(f"  [{i+1:3d}/{total}] {status_icon} {r['name']}"
              + (f"  ERR: {r['error']}" if r["error"] else ""))

    generated = [r for r in gen_results if r["status"] == "generated"]
    print(f"\nGenerated {len(generated)}/{total} cases")

    if args.generate_only:
        _write_summary(out_dir, gen_results)
        return

    # ── Step 2: run in parallel via Docker ───────────────────────────────
    case_dirs = [out_dir / r["name"] for r in generated]
    run_results = run_sweep_parallel(case_dirs, n_parallel=args.n_parallel)

    # Merge gen + run results
    run_map = {r["name"]: r for r in run_results}
    for r in gen_results:
        if r["name"] in run_map:
            r.update(run_map[r["name"]])

    _write_summary(out_dir, gen_results)


def _write_summary(out_dir: Path, statuses: list):
    counts = {}
    for s in statuses:
        counts[s["status"]] = counts.get(s["status"], 0) + 1

    print("\n── Sweep summary ──")
    for k, v in sorted(counts.items()):
        print(f"  {k:25s}: {v}")

    summary_path = out_dir / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"cases": statuses, "counts": counts}, f, indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
