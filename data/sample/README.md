# Sample data

This folder holds a small reference sample of the training data, not the full dataset.

- `ald_hdf5/` — Track 1 (PCGM). 2 of 83 reactingFoam cases, plus `case_index.json` listing all 83 cases' parameters (D, pitch, Q, etc.).
- `track2_hdf5/` — Track 2 (VICES). 2 of 40 reactingFoam cases, plus `case_index.json` listing all 40 cases' parameters.

Each `.h5` file follows the schema in `data/scripts/convert_to_hdf5.py`: `/coords`, `/inputs/node_features`, `/inputs/global`, `/outputs/node_fields`.

The full ~32GB dataset (all 123 raw + processed cases) is archived separately and is not stored in this repo.
