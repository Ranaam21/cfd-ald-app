# cfd-ald-app

Physics + Deep Learning CFD application for ALD (Atomic Layer Deposition) showerhead / faceplate / nozzle-array design. Trains a multi-head GNN surrogate on reactingFoam CFD data, enforces physics guardrails (Re, Pr, Sc, Ma, Pe, Da, ...), and runs multi-objective geometry optimization behind a Streamlit UI.

See `CLAUDE.md` for full architecture details (the 8-module pipeline, Track 1 PCGM vs. Track 2 VICES geometry generation, dataset/training notes).

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

By default the app looks for checkpoints/results under the repo root (`checkpoints/multihead/multihead_final.pt`, `checkpoints/optimizer/`, `checkpoints/guardrail/`, `checkpoints/geometry_loop/` — all included in this repo), so it runs standalone right after cloning. Override the data root with an env var if needed:

```bash
CFD_BASE=/path/to/cfd-ald-app streamlit run app.py
```

## Layout

- `app.py` — Streamlit UI (predictions, guardrails, optimizer, geometry views)
- `geometry/`, `physics/`, `optimization/`, `optimizer/` — core pipeline modules
- `training/` — notebooks used to pretrain/finetune the GNN (00–11, run on Colab)
- `openfoam/` — reactingFoam case generation + postprocessing scripts (raw case outputs are not stored here — see below)
- `checkpoints/` — trained model + optimizer/guardrail/geometry-loop result artifacts (small files only; per-epoch training snapshots are gitignored)
- `data/sample/` — 2 representative cases per track + full `case_index.json` per track (documents every parameter combination actually run). The full dataset (~32GB, 123 reactingFoam cases) is archived outside this repo.
- `Paper/` — manuscript drafts and submission packages

## Data

Full raw/processed CFD data is intentionally excluded from git (see `.gitignore`) — it's large and not needed to run the app. `data/sample/` has enough to see the HDF5 schema and the parameter sweep that was run.
