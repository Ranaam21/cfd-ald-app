# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Physics + Deep Learning CFD application** for ALD (Atomic Layer Deposition) showerhead / faceplate / nozzle-array design. Trains ML surrogates on public CFD data, finetunes on self-generated ALD reactingFoam cases, enforces physics guardrails, and runs multi-objective optimization. Reference plan: `CFD Example Plan.docx`.

---

## Two Design Tracks

Both tracks share the **same physics, data, ML, guardrail, optimizer, and verification backbone**. They differ only in **Module 1 — Geometry Creation**.

### Track 1 — PCGM (Physics-Constrained Geometric Morphogenesis) ✓ BUILT
Geometry grown via topology grammar + parametric rules → point cloud → k-NN graph → GNN surrogate.
- No mesh framework at inference time — k-NN graph built on-the-fly from point cloud coordinates.
- Design space: discrete parameters (D, pitch, H_plenum, t_face, standoff). Fixed topology (hex nozzle array).

### Track 2 — VICES (Voxel-Implicit Computational Engineering Synthesis) ⬜ TO BUILD
Geometry synthesized via **PicoGK** (open-source C# voxel/SDF kernel, basis of LEAP71/Noyron stack) using boolean ops → **Marching Cubes** triangulation → mesh → point cloud → same GNN pipeline.
- Enables topology variation: diffusers, baffles, lattice internals, freeform channels — anything expressible as CSG.
- Design space: CSG tree topology + continuous dimensions → much larger than Track 1.

**Noyron** (proprietary, LEAP71) is the reference implementation: it uses PicoGK for geometry creation and feeds observed behavior back to improve iterations.

---

## Full System Architecture — 8 Modules

```
User Inputs (geometry knobs + process knobs + guardrail ranges)
       │
       ▼
┌─────────────────────┐    ┌──────────────────────────────────┐
│  MODULE 0           │    │  MODULE 1 (Track-specific)        │
│  Requirements &     │───▶│  Geometry Creation                │
│  Physics Guardrails │    │  Track 1: PCGM grammar            │
└─────────────────────┘    │  Track 2: PicoGK voxel + CSG     │
                           └─────────────┬────────────────────┘
                                         │
                                         ▼
                           ┌─────────────────────────────┐
                           │  MODULE 2 + 3               │
                           │  Data Layer + Standardize   │
                           │  (shared)                   │
                           └─────────────┬───────────────┘
                                         │
                                         ▼
                           ┌─────────────────────────────┐
                           │  MODULE 4                   │
                           │  Multi-Head GNN Surrogate   │
                           │  flow / heat / species      │
                           │  (shared)                   │
                           └─────────────┬───────────────┘
                                         │
                                         ▼
                           ┌─────────────────────────────┐
                           │  MODULE 5                   │
                           │  Guardrail Engine           │
                           │  (shared)                   │
                           └──────┬──────────────────────┘
                                  │
                     ┌────────────┴──────────────┐
                     ▼                           ▼
           ┌──────────────────┐       ┌─────────────────────┐
           │  MODULE 6        │       │  MODULE 7           │
           │  Optimizer       │       │  Verification Loop  │
           │  Pareto front    │       │  Targeted CFD runs  │
           │  (shared)        │       │  (shared)           │
           └──────────────────┘       └─────────────────────┘
```

---

## Module-by-Module Detail

### Module 0 — Requirements & Physics Guardrails (shared) ✓
Computes dimensionless groups as explicit physics feature vector + pass/fail thresholds (user-settable sliders).

| Group | Formula | Protects Against |
|-------|---------|-----------------|
| Re | ρVD/μ | Regime mismatch (laminar/turbulent) |
| Ma | V/a | Compressibility effects |
| Eu | Δp/(½ρV²) | Pressure drop sanity — ALD creeping flow gives Eu~1e5–1e7 naturally (set max=1e8) |
| Pr | cpμ/k | Heat boundary layer scaling |
| Nu | hL/k | Heat transfer coefficient bounds |
| Bi | hL/k_s | Lumped vs. internal temperature gradient |
| Sc | μ/(ρD_m) | Diffusion scaling consistency |
| Sh | k_m L/D_m | Mass transfer coefficient bounds |
| Pe_h | Re·Pr | Advection vs. diffusion (heat) |
| Pe_m | Re·Sc | Advection vs. diffusion (mass) |
| Da | k_rxn L/V | Reaction-limited vs. transport-limited (critical for ALD) |

Guardrail behavior: violation → "Regime violation" warning + reduced confidence score + recommend CFD refinement.

---

### Module 1A — Geometry Creation: Track 1 PCGM ✓ BUILT

**Inputs:** Nozzle diameter D, pitch/D, H_plenum, t_face, standoff, flow_rate_slm, NozzlePattern.HEX
**Processing:**
- Topology grammar (hex nozzle array) — fixed topology, vary parameters
- Manufacturability + regime pre-checks (Re, Ma, Da) before generating
- Outputs 80,000-point cloud over fluid domain

**Outputs:** coords [N,3], node_features [N,4], global_features [18]
**No mesh framework** — point cloud directly built from parametric rules.

---

### Module 1B — Geometry Creation: Track 2 VICES ⬜ TO BUILD

**Inputs:** CSG primitive set + boolean tree topology + dimension parameters + voxel resolution
**Processing — 3 steps:**

**Step 1 — PicoGK Voxel/SDF Synthesis**
- PicoGK is a C# open-source geometry kernel; Python interface via subprocess or .NET binding
- Geometry represented as a **Signed Distance Field (SDF)** / voxel grid
- Boolean operations: union ∪, subtract −, intersect ∩ applied to primitive shapes (cylinder, box, cone, sphere, torus)
- Supports offsets, shells, lattice infill

**Step 2 — CSG Tree Search (Catalan Numbers)**
- For n primitive shapes, the number of distinct binary CSG tree topologies = **(n−1)th Catalan number**:

| n primitives | Catalan C(n-1) | Distinct tree topologies |
|---|---|---|
| 2 | 1 | 1 |
| 3 | 2 | 2 |
| 4 | 5 | 5 |
| 5 | 14 | 14 |
| 6 | 42 | 42 |
| 7 | 132 | 132 |

- Each node in the tree is one of {∪, −, ∩} × which primitive → full design space = tree_topologies × 3^(n-1) ops × continuous dimensions
- Optimizer must search this joint space (topology + dimensions) — use **genetic algorithm** or **Bayesian optimization over tree encoding**

**Step 3 — Implicit → Mesh via Marching Cubes**
- Convert SDF voxel grid to triangle mesh using **Marching Cubes** algorithm
- Each cube of 8 voxels has 2^8=256 configurations → reduced to 15 unique surface cases by symmetry
- Library: **PyMCubes** (lightweight, pure Python) or **VTK** (full-featured)
- After triangulation: tag mesh regions (inlet/outlet/wall/wafer) → sample as point cloud → same HDF5 → same GNN

**Outputs:** mesh (STL/VTK) + region tags → point cloud [N,3] → feeds Module 2+

**Key difference from Track 1:** Track 1 topology is fixed (hex array). Track 2 topology is a variable — the optimizer searches over both topology AND dimensions simultaneously.

---

### Module 2 — Data Layer: Public CFD Datasets (shared) ✓ (partially)

| Dataset | Physics | Fields | Status |
|---------|---------|--------|--------|
| reactingFoam (OpenFOAM) | ALD reactive transport | U, p, T, TMA | ✓ 83 cases |
| AirfRANS | Flow pretraining | u, p, ν_t | ⬜ not ingested |
| CFDBench | Flow pretraining | u, p | ⬜ not ingested |
| TNF Piloted Jet Flames | Heat + species | T, species, velocity | ⬜ not ingested |
| JHTDB | Turbulence | u, p, derivatives | ⬜ optional |
| Jet impingement papers | Heat (Nu/h) | Nu vs Re, H/D, spacing | ⬜ correlations only |
| Sherwood dataset (Mendeley) | Mass transfer | Sh vs Re, Sc | ⬜ not ingested |
| Da design charts paper | ALD kinetics | Da-based design reasoning | ⬜ reference only |

---

### Module 3 — Data Standardization & Feature Engineering (shared) ✓

**HDF5 per case + metadata.json index.**

For mesh/pointcloud datasets:
```
/coords                → [N, 3]
/inputs/node_features  → [N, Fin]   # BC tags, distances, wall proximity
/inputs/global         → [F]        # Re, Pr, Sc, Pe_h, Pe_m, Da, Bi, Eu, Ma
/outputs/node_fields   → [N, Fout]  # Ux, Uy, Uz, p, T, TMA
```

For grid datasets (CFDBench-style):
```
/inputs/bc             → [C_in, H, W]
/inputs/global         → [F]
/outputs/fields        → [C_out, H, W]
```

Dimensionless groups stored as `inputs/global` — used at both train-time (loss penalties) and inference-time (guardrail checks).

---

### Module 4 — ML Surrogate: Multi-Head MeshGraphNet (shared) ✓

**Architecture:** Shared encoder + 15× MGNProcessor + 3 separate decoder heads

| Head | Outputs | Trained On |
|------|---------|-----------|
| Flow | Ux, Uy, Uz, p [N,4] | reactingFoam cases |
| Heat | T [N,1] | reactingFoam cases |
| Species | TMA [N,1] | reactingFoam cases |

**Config:** hidden=256, n_layers=15, k_neighbors=6, node_input_dim=22 (4 node + 18 global), edge_input_dim=4
**Backend:** PyTorch + PyG (torch_geometric). PhysicsNeMo MeshGraphNet was original plan but DGL incompatible with PyTorch 2.10+cu128 on Colab A100.
**Checkpoint format:** `{'model': state_dict, 'cfg': dict, 'norm': dict}` — exact keys required by app.py
**Inference:** `fp, hp, sp = model(x, ei, ea)` — model returns 3 separate tensors; preds denormalized before use

**Staged training plan (from doc):**
1. Pretrain flow head on AirfRANS + CFDBench ⬜
2. Domain adapt on jet impingement (Nu, H/D effects) ⬜
3. Finetune all heads on reactingFoam ALD cases ✓ (150 epochs, 83 cases)
4. Optional hybrid CFD init (PhysicsNeMo-CFD pattern) ⬜

ML engine is **swappable** — PhysicsNeMo (MeshGraphNet/DoMINO) OR plain PyTorch. NeMo is a plugin, not the pipeline.

---

### Module 5 — Guardrail Engine (shared) ✓

Physics authority layer. Computes regime checks from predicted fields + user thresholds.
- Violation → warn + reduce confidence score + recommend CFD refinement
- Ma > threshold → flag compressibility regime
- Da mismatch → flag reaction-limited vs transport-limited inconsistency
- Eu guardrail: max=1e8 for ALD (creeping flow gives Eu~1e5–1e7 naturally; not a problem)

---

### Module 6 — Multi-Objective Optimizer (shared) ✓

**Track 1:** BoTorch Pareto optimization over continuous parameters (D, pitch, H_plenum, t_face, standoff).
**Track 2:** Must also search over CSG tree topology (discrete) + dimensions (continuous) jointly.
- Recommend: **genetic algorithm** for tree topology search + local Bayesian optimization for dimensions
- Constraint repair loop: geometry module receives "why failed" signal and regenerates

Objectives: minimize Δp, maximize TMA uniformity index, maximize T uniformity index, minimize Da risk.

---

### Module 7 — Verification Loop (shared) ✓ (partial)

Targeted CFD/CHT/reactive runs on top Pareto candidates → compare surrogate vs CFD → finetune heads on errors.
- Noyron-style: "observed behavior fed back to improve iterations"
- Currently implemented as CFD refinement hook in app.py; finetune-on-error loop not yet automated

---

## Track Comparison Summary

| Element | Track 1 PCGM | Track 2 VICES |
|---------|-------------|---------------|
| Geometry representation | Point cloud from parametric rules | SDF/voxel from PicoGK boolean CSG tree |
| Generator mechanics | Topology grammar + parametric edits | PicoGK primitives + boolean ops (union/subtract/intersect) |
| Mesh step | None (point cloud directly) | Marching Cubes: SDF voxel → triangle mesh → point cloud |
| Design space | Fixed topology, vary ~5 parameters | Variable topology (Catalan tree enumeration) + continuous dims |
| Optimizer type | Continuous Pareto (BoTorch) | Genetic algorithm (tree) + Bayesian (dims) |
| External dependency | None | PicoGK (C#, open-source) |
| Status | ✓ Built | ⬜ To build |

---

## CFD Solver

**reactingFoam** (standard OpenFOAM, Docker `opencfd/openfoam-default`). Chosen over aldFoam (Argonne) because:
- Handles flow + heat + species + turbulence in one solver
- Auto turbulence: laminar (Re<2300) or k-ω SST (Re≥2300)
- Time-varying BCs for ALD pulse/purge (`uniformFixedValue` table)
- No compilation needed, actively maintained

---

## Key Geometry Parameters (User Controls)

- **Geometry**: nozzle/hole diameter, pitch, pattern (ring/hex), plenum height, faceplate thickness, diffuser angle, restrictor Cd
- **Process**: fluid properties (ρ, μ, k, cp), precursor diffusivity D_m, pulse/purge timing, sticking coefficient / k_rxn
- **Guardrail ranges**: min/max for each dimensionless number; incompressible/compressible toggle; reaction/transport-limited toggle

---

## Track 2 Build Plan (next steps)

1. **PicoGK Python bridge** — subprocess wrapper or pythonnet .NET binding to call C# PicoGK from Python
2. **Primitive library** — cylinder, box, cone, torus → SDF representations
3. **CSG tree encoder** — represent tree as list of (op, primitive_id, params); enumerate up to n=6 (42 topologies)
4. **Marching Cubes export** — PyMCubes: `verts, faces = mcubes.marching_cubes(sdf_grid, 0.0)` → STL
5. **Region tagger** — label mesh faces as inlet/outlet/wall/wafer using geometric rules
6. **Point cloud sampler** — sample N points from mesh surface + interior → HDF5 → same pipeline
7. **Track 2 optimizer** — genetic algorithm over CSG tree topology + BoTorch over continuous dims
8. **app.py Track 2 tab** — geometry synthesis UI with primitive controls + boolean tree visualizer

---

## Current Status

| Module | Track 1 | Track 2 |
|--------|---------|---------|
| 0 — Guardrails | ✓ | ✓ (shared) |
| 1 — Geometry | ✓ PCGM | ⬜ VICES |
| 2 — Data Layer | ✓ 83 cases | ⬜ (will reuse) |
| 3 — Standardization | ✓ HDF5 | ✓ (shared) |
| 4 — Surrogate | ✓ 150 epochs | ✓ (shared) |
| 5 — Guardrail Engine | ✓ | ✓ (shared) |
| 6 — Optimizer | ✓ Pareto | ⬜ tree+dims |
| 7 — Verification | ✓ partial | ✓ (shared) |
| App UI | ✓ 5 tabs | ⬜ Track 2 tab |
