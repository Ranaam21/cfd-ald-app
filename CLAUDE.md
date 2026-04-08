# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Physics + Deep Learning CFD application** for ALD (Atomic Layer Deposition) showerhead / faceplate / nozzle-array design. The project trains ML surrogates on public CFD data, then finetunes on ALD-specific data, enforcing physics guardrails throughout. The planning document is `CFD Track Wise Planning.docx`.

## Two Design Tracks

**Track 1 — PCGM (Physics-Constrained Geometric Morphogenesis):** Geometry generated via topology grammar + parametric rules → mesh/graph representation.

**Track 2 — VICES (Voxel-Implicit Computational Engineering Synthesis):** Geometry synthesized via PicoGK (open-source voxel/SDF kernel with boolean ops) → export to mesh for CFD. Noyron (proprietary, built on PicoGK) is the reference implementation.

Both tracks share the same physics, data, ML, and optimization backbone — they differ only in the geometry creation module.

## System Architecture (7 Modules)

```
User Inputs → Geometry Module → Physics Calculator → Data Layer
                                                          ↓
                                              ML Surrogate (plug-in)
                                                          ↓
                                              Guardrail Engine
                                                ↙          ↘
                                         Optimizer    CFD Refinement Hook
```

| Module | Purpose |
|--------|---------|
| **0 — Requirements & Guardrails** | Compute dimensionless groups as physics features + pass/fail thresholds |
| **1 — Geometry Creator** | PCGM grammar OR PicoGK voxel synthesis; outputs mesh + semantic tags |
| **2 — Data Layer** | Ingest public CFD datasets (AirfRANS, CFDBench, TNF, JHTDB, reactingFoam) |
| **3 — Data Standardization** | Convert to HDF5 with canonical keys; embed dimensionless features |
| **4 — ML Surrogate (multi-head)** | Separate flow/heat/species heads → merge with shared encoder + masked losses |
| **5 — Guardrail Engine** | Physics authority: regime checks → confidence scores + reason codes |
| **6 — Optimizer** | Multi-objective Pareto optimization; constraint repair feedback to Geometry |
| **7 — Verification Loop** | Targeted CFD/CHT runs on best candidates; finetune heads on errors |

## Physics Guardrails to Implement

These are user-settable sliders/ranges, computed per design:

| Group | Formula | Protects Against |
|-------|---------|-----------------|
| Re | ρVD/μ | Regime mismatch (laminar/turbulent) |
| Ma | V/a | Compressibility effects |
| Eu | Δp/(½ρV²) | Unrealistic pressure drop predictions |
| Pr | cpμ/k | Heat boundary layer scaling |
| Nu | hL/k | Heat transfer coefficient bounds |
| Bi | hL/k_s | Lumped vs. internal temperature gradients |
| Sc | μ/(ρD_m) | Diffusion scaling consistency |
| Sh | k_m L/D_m | Mass transfer coefficient bounds |
| Pe_h | Re·Pr | Advection vs. diffusion dominance (heat) |
| Pe_m | Re·Sc | Advection vs. diffusion dominance (mass) |
| Da | k_rxn L/V | Reaction-limited vs. transport-limited (critical for ALD) |

Guardrail behavior: violation → "Regime violation" warning + reduced confidence score + recommend CFD refinement.

## ML Training Pipeline (Staged)

1. **Pretrain** on AirfRANS + CFDBench (generic flow/geometry physics)
2. **Domain adapt** on jet/nozzle-array impingement datasets (Nu, spacing/jet-count effects)
3. **Finetune** on reactingFoam-generated ALD-specific data (species, Da, residence time τ)
4. **Optional hybrid init**: use PhysicsNeMo-CFD to initialize classical CFD from surrogate predictions

ML engine is **swappable**: PhysicsNeMo (MeshGraphNet/DoMINO/operators) OR plain PyTorch. NeMo is a plugin, not the pipeline itself.

## Dataset Formats

**HDF5 per case + `metadata.json` index** (PhysicsNeMo datapipe compatible).

For mesh/pointcloud datasets:
```
/coords          → [N, 3]
/inputs/node_features → [N, Fin]   # BC tags, distances, etc.
/inputs/global   → [F]             # Re, Pr, Sc, Pe_h, Pe_m, Da, Bi, ...
/outputs/node_fields → [N, Fout]
```

For grid datasets (CFDBench-style):
```
/inputs/bc       → [C_in, H, W]
/inputs/global   → [F]
/outputs/fields  → [C_out, H, W]
```

Dimensionless groups (Re, Pr, Sc, Pe_h, Pe_m, Da, Bi, Eu, Ma) must be computed and stored as `inputs/global` channels per case — used for both train-time physics loss penalties and inference-time guardrail checks.

## Public Data Sources

| Dataset | Physics | Key fields |
|---------|---------|-----------|
| AirfRANS | Flow pretraining | u, p, ν_t; RANS over NACA airfoils |
| CFDBench | Flow pretraining | u, p; cavity/tube/dam/cylinder; HuggingFace download |
| TNF Piloted Jet Flames | Heat + species | T, major/minor species, velocity |
| JHTDB | Turbulence | u, p, derivatives; HDF5 cutout service |
| reactingFoam (OpenFOAM) | ALD reactive transport | Self-generated runs; flow + heat + species + k-ω SST turbulence; replaces aldFoam |
| Jet impingement papers | Heat (Nu/h) | Nu vs Re, H/D, spacing correlations |
| Sherwood dataset (Mendeley) | Mass transfer | Sh vs Re, Sc numeric data |
| Da design charts paper | ALD kinetics | Da-based design reasoning |

## CFD Solver Architecture

**Solver: `reactingFoam` (standard OpenFOAM, Docker image `opencfd/openfoam-default`)**

Chosen over aldFoam (Argonne custom solver) because:
- Handles all physics in one solver: flow + heat (T, Nu, h) + species (TMA/N2) + turbulence
- Auto turbulence selection: laminar (Re < 2300) or k-ω SST (Re ≥ 2300)
- Time-varying BCs for ALD pulse/purge cycle (`uniformFixedValue` table)
- Standard OpenFOAM — no compilation needed, actively maintained, community support
- Gives T field → real Nu/h/Bi for heat head training (no synthetic correlations needed)

**ML Backend: PyTorch + PyG (torch_geometric)**

PhysicsNeMo MeshGraphNet was the original plan but uses DGL which is incompatible with PyTorch 2.10+cu128 on Colab A100. Custom MeshGraphNet implemented directly in PyG with identical architecture (encoder / 15× MGNProcessor / decoder, hidden=256, SiLU, LayerNorm, residuals).

## Key Geometry Parameters (User Controls)

- **Geometry**: nozzle/hole diameter, pitch, pattern (ring/hex), plenum height, faceplate thickness, diffuser angle, restrictor Cd
- **Process**: fluid properties (ρ, μ, k, cp), precursor diffusivity D_m, pulse/purge timing, sticking coefficient / k_rxn
- **Guardrail ranges**: min/max for each dimensionless number; incompressible vs. compressible toggle; reaction-limited vs. transport-limited mode toggle
