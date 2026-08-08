# solid_solutions_li

Machine-learning pipeline for **Li vacancy energetics** in dilute Li–X solid solutions: vacancy–solute **binding** energy \(E_b\), **solution** energy \(E_s\), and derived effective energy \(E_{\mathrm{eff}}\).

DFT targets are combined with Materials Project phase metadata, matminer structure descriptors, Hume–Rothery-style elemental deltas vs Li, and Li–X endmember features, then used for feature-importance analysis and multi-model regression.

## Data pools: `_pre` vs `_opt`

| Suffix | Meaning |
|--------|---------|
| **`_pre`** | Full dopant set — all available DFT vacancy energies (~50 dopants). |
| **`_opt`** | Lattice-optimized subset (~19 dopants), including **2NN neighbour** positions for solute–vacancy **binding** energy (`vac_position`: `near` / `2NN`). |

Energy CSVs live under `data/`:

- `vac_binding_energy_pre.csv` / `vac_binding_energy_opt.csv`
- `vac_solution_energy_pre.csv` / `vac_solution_energy_opt.csv`

Figures and importance tables are tagged the same way (`*_pre` / `*_opt`). ML figures go to `figures/ml_pre_trained/` or `figures/ml_opt_trained/`; Hume–Rothery figures go to `figures/h_r_rules/`.

## Workflow

1. **`create_data.ipynb`** — fetch MP phases / structures, build endmember tables, merge features → `data/datasheet.csv`
2. **`feature_importance.ipynb`** — RF permutation importance for \(E_b\), \(E_s\), \(E_s-E_b\) on `_pre` or `_opt`
3. **`model_train.ipynb`** (or `model_train.py`) — tune Ridge / Lasso / ElasticNet / SVR / RF / GBM / XGBoost / CatBoost

Default hold-out: train on the full `_pre` pool plus remaining `_opt` dopants; **test only on `_opt`**.

## Repository layout

```
├── create_data.ipynb          # feature / datasheet construction
├── feature_importance.ipynb   # permutation importance
├── model_train.ipynb          # model training notebook
├── model_train.py             # script export of model training
├── build_datasheet.py         # merge energies + features
├── mp_stable_phases.py        # Materials Project unary phases
├── mp_structure_features.py   # matminer structure featurization
├── endmember_phases.py        # Li–X endmember phase map
├── solution_thermo.py         # chemical potentials / thermo features
├── solution_features_best.py  # pinned E_s feature list
├── data/                      # CSVs (energies, features, metrics)
└── figures/
    ├── h_r_rules/             # Hume–Rothery rule plots
    ├── ml_pre_trained/        # ML figures for `_pre` / pre-train runs
    └── ml_opt_trained/        # ML figures for `_opt` / opt-only runs
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Materials Project queries in `create_data.ipynb`, set an API key:

```bash
export MP_API_KEY="your_key"
```

Get a key at [materialsproject.org/api](https://materialsproject.org/api).

Precomputed tables under `data/` are already included, so importance / training notebooks can be run without re-querying MP.

## Targets

| Symbol | Column / definition |
|--------|---------------------|
| \(E_b\) | `vac_binding_energy_eV` |
| \(E_s\) | `solution_energy_eV` |
| \(E_s - E_b\) | difference of the above |
| \(E_{\mathrm{eff}}\) | piecewise: \(E_s-E_b\) if \(E_s>0\), else \(-E_b\) (derived from best \(E_s\) / \(E_b\) models) |

## Citation / notes

DFT vacancy energies and chemical potentials are project inputs. Feature tables may be regenerated against the current Materials Project release; energies in `data/vac_*_*.csv` are the fixed labels for ML.
