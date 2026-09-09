# solid_solutions_li

Machine-learning pipeline for **Li vacancy energetics** in dilute Li–X solid solutions: vacancy–solute binding energy \(E_b\), solution energy \(E_s\), and derived effective energy \(E_{\mathrm{eff}}\).

DFT targets are combined with Materials Project phase metadata, matminer structure descriptors, Hume–Rothery elemental properties / deltas vs Li, and Li–X endmember features, then used for feature-importance analysis and multi-model regression.

## Contents

- [Data pools: `_pre` vs `_opt`](#data-pools-_pre-vs-_opt)
- [Workflow](#workflow)
- [Hume-Rothery rules](#hume-rothery-rules)
- [Repository layout](#repository-layout)
- [Data files (`data/`)](#data-files-data)
- [Setup](#setup)
- [Targets](#targets)
- [Figures](#figures)

## Data pools: `_pre` vs `_opt`

| Suffix | Meaning |
|--------|---------|
| **`_pre`** | Full dopant set — all available DFT vacancy energies (~50 dopants). |
| **`_opt`** | Lattice-optimized subset (~19 dopants), including **2NN neighbour** positions for solute–vacancy binding (`vac_position`: `near` / `2NN`). |

Primary energy inputs:

- [`data/vac_binding_energy_pre.csv`](data/vac_binding_energy_pre.csv) / [`data/vac_binding_energy_opt.csv`](data/vac_binding_energy_opt.csv)
- [`data/vac_solution_energy_pre.csv`](data/vac_solution_energy_pre.csv) / [`data/vac_solution_energy_opt.csv`](data/vac_solution_energy_opt.csv)

Importance tables, metrics, and ML figures are tagged `_pre` / `_opt` the same way.

## Workflow

1. [`create_data.ipynb`](create_data.ipynb) — MP phases / structures, endmember tables, merge features → [`data/datasheet.csv`](data/datasheet.csv)
2. [`hume_rules.ipynb`](hume_rules.ipynb) — Hume–Rothery rule checks vs Li (optional, see below)
3. [`feature_importance.ipynb`](feature_importance.ipynb) — RF permutation importance for \(E_b\), \(E_s\), \(E_s-E_b\)
4. [`model_train.ipynb`](model_train.ipynb) / [`model_train.py`](model_train.py) — tune Ridge / Lasso / ElasticNet / SVR / RF / GBM / XGBoost / CatBoost

Hold-out options in model training:

- **Opt-only** (current defaults: all `*_OPT_ONLY=True`): train and test on the `_opt` split only (**9 train / 10 test**).
- **Pre-train / opt-test**: train on `_pre` + remaining `_opt`, test on `_opt` only.

Current training defaults (see `model_train.py` / [`data/model_train_results.csv`](data/model_train_results.csv)):

- Hyperparameter search: **randomized** (`HYPERPARAM_SEARCH='random'`)
- Model selection: minimal **cross-validation** RMSE (`SELECT_BEST_BY='cv'`)
- Features: significance + correlation filtering, capped at 25 → **17** features per target in the latest run
- CV-best models (opt-only hold-out): **CatBoost** (\(E_s\)), **Ridge** (\(E_b\)), **SVR** (direct \(E_{\mathrm{eff}}\)); derived piecewise \(E_{\mathrm{eff}}\) from the best \(E_s\)/\(E_b\) models outperforms direct prediction

## Hume-Rothery rules

Classical solid-solution criteria are implemented in [`hume_rules.ipynb`](hume_rules.ipynb) and the elemental table [`hume_rothery_data.py`](hume_rothery_data.py) → [`data/hume_rothery_data.csv`](data/hume_rothery_data.csv).

For host **Li** and dopant \(X\), the notebook evaluates four rules (defaults: size tolerance 15 %, electronegativity tolerance \(|\Delta\chi|\le 0.4\)):

| Rule | Criterion | Implementation notes |
|------|-----------|----------------------|
| **1. Atomic size** | Relative radius mismatch \(\lvert r_X-r_{\mathrm{Li}}\rvert / \max(r_X,r_{\mathrm{Li}}) \le 15\%\) | Metallic / XRD radii in pm (`r_atomic_pm`) |
| **2. Crystal structure** | Same structure type as Li (bcc), or treated as compatible (fcc↔bcc allowed) | Labels: `bcc`, `fcc`, `hcp`, `diamond`, … |
| **3. Valency** | Same preferred valency as Li (1) is ideal; otherwise reported as partial | Integer valency in the table |
| **4. Electronegativity** | \(\lvert\chi_X-\chi_{\mathrm{Li}}\rvert\le 0.4\) (Pauling) | Absolute \(\Delta\chi\) (also stored as %) |

Outputs:

- Rule-enriched table: [`figures/h_r_rules/elements_data.csv`](figures/h_r_rules/elements_data.csv) (and a copy under [`data/elements_data.csv`](data/elements_data.csv) if regenerated there)
- Plots: [`figures/h_r_rules/`](figures/h_r_rules/) (`hr_rule_1`–`4`, metallic radii analysis)

For the ML pipeline, Hume–Rothery-style **deltas vs Li** used as features are in [`data/elemental_props_vs_li.csv`](data/elemental_props_vs_li.csv) (`delta_r_vs_Li_pm`, `delta_valency_vs_Li`, `delta_chi_vs_Li`), built by [`create_data.ipynb`](create_data.ipynb).

## Repository layout

```
├── create_data.ipynb          # feature / datasheet construction
├── hume_rules.ipynb           # Hume–Rothery rule analysis
├── hume_rothery_data.py       # elemental HR table + CSV export
├── feature_importance.ipynb   # permutation importance
├── model_train.ipynb          # model training notebook
├── model_train.py             # script export of model training
├── build_datasheet.py         # merge energies + features
├── mp_stable_phases.py        # Materials Project unary phases
├── mp_structure_features.py   # matminer structure featurization
├── endmember_phases.py        # Li–X endmember phase map
├── solution_thermo.py         # chemical potentials / thermo features
├── solution_features_best.py  # pinned E_s feature list
├── data/                      # CSVs (see below)
└── figures/
    ├── h_r_rules/             # Hume–Rothery plots + rule table
    ├── ml_pre_trained/        # ML figures for `_pre` runs
    └── ml_opt_trained/        # ML figures for `_opt` runs
```

## Data files (`data/`)

Suffixes: `_pre` / `_opt` mark the energy pool; `_train` is importance computed on a train split only. Untagged copies of some importance/metrics files are legacy duplicates of `_opt` runs.

### DFT targets and thermo inputs

| File | Description |
|------|-------------|
| [`vac_binding_energy_pre.csv`](data/vac_binding_energy_pre.csv) | Vacancy–solute binding energies \(E_b\) (eV), full dopant set |
| [`vac_binding_energy_opt.csv`](data/vac_binding_energy_opt.csv) | \(E_b\) for lattice-optimized set; includes `vac_position` (`near` / `2NN`) |
| [`vac_solution_energy_pre.csv`](data/vac_solution_energy_pre.csv) | Solution energies \(E_s\) (eV), full set |
| [`vac_solution_energy_opt.csv`](data/vac_solution_energy_opt.csv) | \(E_s\) for optimized set |
| [`chem_pots_pre.csv`](data/chem_pots_pre.csv) | Dopant chemical potentials for the `_pre` set (`chem_pot_eV`, `delta_eV`) |
| [`chem_pots_opt.csv`](data/chem_pots_opt.csv) | Same for the `_opt` set |

### Feature tables (MP / matminer / elemental)

| File | Description |
|------|-------------|
| [`elemental_props_vs_li.csv`](data/elemental_props_vs_li.csv) | Hume–Rothery-style deltas vs Li for ML (`delta_r`, `delta_valency`, `delta_chi`) |
| [`hume_rothery_data.csv`](data/hume_rothery_data.csv) | Elemental HR properties: \(Z\), radius (pm), crystal, valency, \(\chi\) |
| [`elements_data.csv`](data/elements_data.csv) | HR table plus Rule 1–4 columns (from [`hume_rules.ipynb`](hume_rules.ipynb)) |
| [`dopant_stable_experimental_phases_mp.csv`](data/dopant_stable_experimental_phases_mp.csv) | Selected unary MP phases (host + dopants) |
| [`dopant_endmember_phases_mp.csv`](data/dopant_endmember_phases_mp.csv) | Li–X supercell endmember phase metadata (`endmember_*`) |
| [`structure_matminer_features.csv`](data/structure_matminer_features.csv) | Magpie / density / symmetry features for unary MP structures |
| [`structure_endmember_matminer_features.csv`](data/structure_endmember_matminer_features.csv) | Same for endmember phases (`endmember_` prefix) |
| [`datasheet.csv`](data/datasheet.csv) | Merged ML-ready table (energies + MP + elemental + endmember; matminer attached at load time) |

### Feature importance and correlations

| File | Description |
|------|-------------|
| [`binding_feature_importance_{pre,opt,train}.csv`](data/binding_feature_importance_opt.csv) | RF MDI + permutation importance for \(E_b\) |
| [`solution_feature_importance_{pre,opt,train}.csv`](data/solution_feature_importance_opt.csv) | Same for \(E_s\) |
| [`solution_minus_binding_feature_importance_{pre,opt,train}.csv`](data/solution_minus_binding_feature_importance_opt.csv) | Same for \(E_s-E_b\) |
| [`binding_feature_importance.csv`](data/binding_feature_importance.csv) (and solution / solution_minus_binding untagged) | Legacy untagged copies (typically equal to `_opt`) |
| [`top_feature_pearson_correlations_{pre,opt}.csv`](data/top_feature_pearson_correlations_opt.csv) | Pearson \(r\) of top important features vs all three targets |
| [`top_feature_pearson_correlations.csv`](data/top_feature_pearson_correlations.csv) | Untagged / legacy pearson table |

### Model metrics and predictions

| File | Description |
|------|-------------|
| [`model_metrics_binding_{pre,opt}.csv`](data/model_metrics_binding_opt.csv) | Train / CV / test metrics for all models (\(E_b\)) |
| [`model_metrics_solution_{pre,opt}.csv`](data/model_metrics_solution_opt.csv) | Same for \(E_s\) |
| [`model_metrics_solution_minus_binding_{pre,opt}.csv`](data/model_metrics_solution_minus_binding_opt.csv) | Same for \(E_{\mathrm{eff}}\) (direct) |
| [`model_train_results.csv`](data/model_train_results.csv) | Full training summary (legacy untagged name; newer runs may write `model_train_results_{pre\|opt}.csv`) |
| [`derived_effective_energies_opt.csv`](data/derived_effective_energies_opt.csv) | Per-dopant DFT vs ML \(E_s\), \(E_b\), piecewise \(E_{\mathrm{eff}}\) |
| [`derived_solution_minus_binding_metrics_{pre,opt}.csv`](data/derived_solution_minus_binding_metrics_opt.csv) | Metrics for derived \(E_{\mathrm{eff}}\) from best \(E_s\) / \(E_b\) models |
| [`derived_solution_minus_binding_metrics.csv`](data/derived_solution_minus_binding_metrics.csv) | Untagged / legacy derived metrics |
| [`binding_prediction_errors.csv`](data/binding_prediction_errors.csv) | Optional per-dopant \(E_b\) prediction errors (legacy export) |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Materials Project queries in [`create_data.ipynb`](create_data.ipynb):

```bash
export MP_API_KEY="your_key"
```

Get a key at [materialsproject.org/api](https://materialsproject.org/api).

Precomputed tables under [`data/`](data/) are included, so importance / training notebooks can run without re-querying MP.

## Targets

| Symbol | Column / definition |
|--------|---------------------|
| \(E_b\) | `vac_binding_energy_eV` |
| \(E_s\) | `solution_energy_eV` |
| \(E_s - E_b\) | difference of the above |
| \(E_{\mathrm{eff}}\) | piecewise: \(E_s-E_b\) if \(E_s>0\), else \(-E_b\) (optionally derived from best \(E_s\) / \(E_b\) models) |

## Figures

| Directory | Contents |
|-----------|----------|
| [`figures/h_r_rules/`](figures/h_r_rules/) | Hume–Rothery rule plots and rule-enriched CSV |
| [`figures/ml_pre_trained/`](figures/ml_pre_trained/) | Feature importance, correlations, parity plots for `_pre` |
| [`figures/ml_opt_trained/`](figures/ml_opt_trained/) | Same for `_opt` / opt-only training |

## Notes

DFT vacancy energies and chemical potentials are project inputs. Feature tables may be regenerated against the current Materials Project release; energies in `data/vac_*_*.csv` are the fixed labels for ML.
