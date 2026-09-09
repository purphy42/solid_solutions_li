#!/usr/bin/env python
# coding: utf-8

# Train and tune regression models for $E_{\mathrm{b}}$, $E_{\mathrm{s}}$, and $E_{\mathrm{s}} - E_{\mathrm{b}}$.
# 
# Thermodynamic features (`mu_dopant_eV`, MP hull/selection metadata) are included when present in the datasheet — re-run `create_data.ipynb` after updating chemical potentials.
# 
# - **$E_{\mathrm{b}}$, $E_{\mathrm{s}}$, $E_{\mathrm{s}}-E_{\mathrm{b}}$**: same hold-out — train on pre+opt by default; **test is opt-only**. Size = 80/20 of the pre pool (`HOLDOUT_TEST_FRACTION`), or set `HOLDOUT_N_TEST` to fix it. Set `EB_OPT_ONLY` / `ES_OPT_ONLY` / `ES_MINUS_EB_OPT_ONLY` for opt-only train/test per target.
# - **Features**: `FEATURE_IMPORTANCE_ON_TRAIN_ONLY` (True = train-split permutation importance; False = load tagged `*_feature_importance_{pre,opt}.csv`). Cap with `TOP_N_FEATURES` / `MAX_FEATURES`.
# - **Tuning**: `HYPERPARAM_SEARCH` = `'random'` (`N_ITER_SEARCH`) or `'grid'`. Scoring follows `BEST_MODEL_CRITERION`.
# - **Model selection**: `SELECT_BEST_BY` = `'cv'` or `'test'`; both `cv_*` and `test_*` metrics are stored.
# - Derived plot on `_opt`: piecewise $E_{\mathrm{eff}}$ ($E_s-E_b$ if $E_s>0$, else $-E_b$) from best $E_{\mathrm{s}}$ / $E_{\mathrm{b}}$ models.
# 
# Plot toggles: `PLOT_SHOW_GRID` / `PLOT_SHOW_TITLES`, `PLOT_SHOW_GRID_ALL_MODELS` / `PLOT_SHOW_TITLES_ALL_MODELS`, `PLOT_SHOW_TEST_INSET`.
# 

# In[ ]:


# Install optional gradient-boosting backends (run once per environment)
# !pip install -q xgboost catboost


# In[ ]:


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    RandomizedSearchCV,
    cross_val_predict,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from build_datasheet import (
    dedupe_correlated_features,
    discover_ml_feature_columns,
    feature_dedup_family,
    load_ml_frame,
    split_pre_train_opt_test,
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

plt.rcParams.update({'figure.dpi': 120, 'font.size': 10})
sns.set_theme(style='white')



# In[ ]:


from matplotlib import font_manager as fm
from pathlib import Path as _Path
for _fp in (
    _Path('/home/a.burov/fonts/ARIAL.TTF'),
    _Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
    _Path('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'),
):
    if _fp.is_file():
        fm.fontManager.addfont(str(_fp))
        break


# In[ ]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[ ]:


np.set_printoptions(precision=5)


# In[ ]:


# %matplotlib inline
plt.rcParams['figure.dpi'] = 450


# In[ ]:





# In[ ]:


# Resolve ml_part/data even if kernel cwd is the repo root.
_candidates = []
try:
    _candidates.append(Path(__vsc_ipynb_file__).resolve().parent)  # VS Code / Cursor
except Exception:
    pass
_candidates.extend([Path.cwd().resolve(), Path.cwd().resolve() / 'ml_part'])
ML_DIR = next((p for p in _candidates if (p / 'data' / 'vac_solution_energy_opt.csv').is_file()), Path.cwd().resolve())
if not (ML_DIR / 'data' / 'vac_solution_energy_opt.csv').is_file() and (ML_DIR / 'ml_part' / 'data' / 'vac_solution_energy_opt.csv').is_file():
    ML_DIR = ML_DIR / 'ml_part'
REPO_ROOT = ML_DIR.parent if ML_DIR.name == 'ml_part' else ML_DIR
DATA_DIR = ML_DIR / 'data'
FIGURES_DIR = ML_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
print(f'ML_DIR={ML_DIR}')
print(f'DATA_DIR={DATA_DIR}')

merged_opt = load_ml_frame(REPO_ROOT, energy_pool='opt')
if merged_opt.empty:
    raise ValueError('merged_opt is empty — check vac_*_opt.csv and create_data.ipynb outputs')
print(f'opt merge: {len(merged_opt)} rows (reference _opt set)')

RANDOM_STATE = 42
OPT_HOLDOUT_TARGETS = {'binding', 'solution', 'solution_minus_binding'}
# Hold-out: test is always drawn from _opt only; train = remaining opt + all pre-only.
# Default: 80/20 of the full pre pool (n_test = round(n_pre * fraction), capped so ≥1 opt stays in train).
HOLDOUT_TEST_FRACTION = 0.2
# Optional override of the 80/20 size. None → use HOLDOUT_TEST_FRACTION; e.g. 11 → fixed test size.
HOLDOUT_N_TEST = None  # → n_test=10 with 50-pre pool (opt-only: 9 train / 10 test)
# If True: that target is trained/tested on _opt only (same opt split; no pre-only rows).
# Default False: pre-train / opt-test (more train data; recommended).
EB_OPT_ONLY = True
ES_OPT_ONLY = True
ES_MINUS_EB_OPT_ONLY = True


def use_opt_only_for(target_key: str) -> bool:
    return {
        'binding': EB_OPT_ONLY,
        'solution': ES_OPT_ONLY,
        'solution_minus_binding': ES_MINUS_EB_OPT_ONLY,
    }.get(target_key, False)

holdout_df = load_ml_frame(REPO_ROOT, energy_pool='pre', verbose=False)
merged_holdout_train, merged_holdout_test = split_pre_train_opt_test(
    holdout_df,
    merged_opt,
    test_fraction=HOLDOUT_TEST_FRACTION,
    n_test=HOLDOUT_N_TEST,
    random_state=RANDOM_STATE,
)
opt_elements = set(merged_opt['element'].astype(str))
merged_opt_holdout_train = merged_holdout_train[
    merged_holdout_train['element'].astype(str).isin(opt_elements)
].copy()
merged_opt_holdout_test = merged_holdout_test.copy()  # already opt-only

TARGETS = {
    'binding': 'vac_binding_energy_eV',
    'solution': 'solution_energy_eV',
    'solution_minus_binding': 'E_solution_minus_binding_eV',
}
TARGET_LABELS = {
    'binding': r'$E_{\mathrm{b}}$',
    'solution': r'$E_{\mathrm{s}}$',
    'solution_minus_binding': r'$E_{\mathrm{eff}}$',
}

for frame in (
    merged_opt,
    holdout_df,
    merged_holdout_train,
    merged_holdout_test,
    merged_opt_holdout_train,
    merged_opt_holdout_test,
):
    frame['E_solution_minus_binding_eV'] = (
        frame['solution_energy_eV'].astype(float) - frame['vac_binding_energy_eV'].astype(float)
    )

n_opt_train = int(merged_holdout_train['element'].astype(str).isin(opt_elements).sum())
n_pre_only_train = len(merged_holdout_train) - n_opt_train
n_opt_test = int(merged_holdout_test['element'].astype(str).isin(opt_elements).sum())
holdout_size_note = (
    f'n_test={HOLDOUT_N_TEST} (fixed)'
    if HOLDOUT_N_TEST is not None
    else f'fraction={HOLDOUT_TEST_FRACTION} (~80/20 of pre pool)'
)

print(
    f'Hold-out ({holdout_size_note}): pool={len(holdout_df)} pre, '
    f'train={len(merged_holdout_train)} ({n_opt_train} opt + {n_pre_only_train} pre-only), '
    f'test={len(merged_holdout_test)} (opt-only={n_opt_test}/{len(merged_holdout_test)})'
)
print(
    f'E_b mode: {"opt-only train/test" if EB_OPT_ONLY else "pre-train / opt-test"} '
    f'(opt train={len(merged_opt_holdout_train)}, opt test={len(merged_opt_holdout_test)})'
)
print(
    f'E_s mode: {"opt-only train/test" if ES_OPT_ONLY else "pre-train / opt-test"} '
    f'(opt train={len(merged_opt_holdout_train)}, opt test={len(merged_opt_holdout_test)})'
)
print(
    f'E_s-E_b mode: {"opt-only train/test" if ES_MINUS_EB_OPT_ONLY else "pre-train / opt-test"} '
    f'(opt train={len(merged_opt_holdout_train)}, opt test={len(merged_opt_holdout_test)})'
)
print(f'hold-out train dopants: {sorted(merged_holdout_train["element"].astype(str).tolist())}')
print(f'hold-out test dopants (opt-only): {sorted(merged_holdout_test["element"].astype(str).tolist())}')



# In[ ]:


# Feature selection.
# True  → permutation importance on the train split only (no test leakage).
# False → load DATA_DIR / '{target}_feature_importance[_opt|_pre].csv' from feature_importance.ipynb.
FEATURE_IMPORTANCE_ON_TRAIN_ONLY = False  # best: load CSV, do not recompute on n≈9
# Used only when FEATURE_IMPORTANCE_ON_TRAIN_ONLY is False (must match feature_importance.ipynb OPT_ONLY).
FEATURE_IMPORTANCE_POOL = 'opt'  # 'opt' | 'pre' | 'train'
IMPORTANCE_THRESHOLD = 1e-3
REQUIRE_PERM_GT_STD = True
MIN_FEATURES = 5
MAX_FEATURES = 50
TOP_N_FEATURES = 25  # None = all significant up to MAX_FEATURES
CORR_DEDUP_THRESHOLD = 0.99
PERM_N_ESTIMATORS = 500
PERM_N_REPEATS = 25


def metrics_pool_tag() -> str:
    """Suffix for metric CSVs: '_opt' / '_pre' (same idea as feature_importance tags)."""
    if FEATURE_IMPORTANCE_POOL in {'opt', 'pre'}:
        return FEATURE_IMPORTANCE_POOL
    if EB_OPT_ONLY and ES_OPT_ONLY and ES_MINUS_EB_OPT_ONLY:
        return 'opt'
    return 'pre'


METRICS_POOL_TAG = metrics_pool_tag()
METRICS_SUFFIX = f'_{METRICS_POOL_TAG}'
print(f'Metrics outputs tagged: *{METRICS_SUFFIX}.csv')


def mark_significant(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'significant' not in out.columns:
        out['significant'] = out['importance_perm_mean'] > out['importance_perm_std']
    return out


def select_significant_features(
    importance_df: pd.DataFrame,
    available_cols: set[str],
    top_n: int | None = TOP_N_FEATURES,
) -> tuple[list[str], int]:
    """Return feature names and count of significant features before top-N cap."""
    df = mark_significant(importance_df).sort_values('importance_perm_mean', ascending=False)
    mask = df['importance_perm_mean'] >= IMPORTANCE_THRESHOLD
    if REQUIRE_PERM_GT_STD:
        mask &= df['significant']
    significant = df[mask]
    n_significant = len(significant)

    if top_n is not None and top_n > 0:
        n_use = min(top_n, n_significant) if n_significant else min(top_n, len(df))
        sel = significant.head(n_use) if n_significant else df.head(n_use)
    else:
        sel = significant
        if len(sel) < MIN_FEATURES:
            sel = df.head(MIN_FEATURES)
        elif len(sel) > MAX_FEATURES:
            sel = significant.head(MAX_FEATURES)

    feats = [f for f in sel['feature'].astype(str).tolist() if f in available_cols]
    if len(feats) < MIN_FEATURES:
        for feat in df['feature'].astype(str):
            if feat in available_cols and feat not in feats:
                feats.append(feat)
            if len(feats) >= MIN_FEATURES:
                break
    cap = top_n if top_n is not None and top_n > 0 else MAX_FEATURES
    return feats[:cap], n_significant


def usable_feature_cols(X: pd.DataFrame, cols: list[str]) -> list[str]:
    usable = []
    for col in cols:
        if col not in X.columns:
            continue
        if pd.to_numeric(X[col], errors='coerce').notna().any():
            usable.append(col)
    return usable


def compute_feature_importance_train(
    X: pd.DataFrame,
    y,
    cols: list[str],
    *,
    n_estimators: int = PERM_N_ESTIMATORS,
    n_repeats: int = PERM_N_REPEATS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, float]:
    fit_cols = usable_feature_cols(X, cols)
    if not fit_cols:
        raise ValueError('No features with observed values for train-only importance.')
    X_fit = X[fit_cols].apply(pd.to_numeric, errors='coerce')
    y_arr = np.asarray(y, dtype=float)
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('rf', RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            max_features='sqrt',
        )),
    ])
    n_splits = min(5, max(2, len(y_arr)))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_r2 = float(np.mean(cross_val_score(model, X_fit, y_arr, cv=cv, scoring='r2')))
    model.fit(X_fit, y_arr)
    perm = permutation_importance(
        model, X_fit, y_arr,
        n_repeats=n_repeats,
        random_state=random_state,
        scoring='r2',
        n_jobs=-1,
    )
    df = pd.DataFrame({
        'feature': fit_cols,
        'importance_mdi': model.named_steps['rf'].feature_importances_,
        'importance_perm_mean': perm.importances_mean,
        'importance_perm_std': perm.importances_std,
    })
    df = mark_significant(df).sort_values('importance_perm_mean', ascending=False).reset_index(drop=True)
    return df, cv_r2


def train_frame_for_target(target_key: str) -> pd.DataFrame:
    if target_key in OPT_HOLDOUT_TARGETS:
        return merged_opt_holdout_train if use_opt_only_for(target_key) else merged_holdout_train
    return merged_opt


features_by_target = {}
importance_by_target = {}
for target_key, target_col in TARGETS.items():
    available_cols = set(merged_opt.columns)
    top_note = f'top {TOP_N_FEATURES}' if TOP_N_FEATURES else 'all significant'

    if FEATURE_IMPORTANCE_ON_TRAIN_ONLY:
        train_df = train_frame_for_target(target_key)
        candidate_cols = discover_ml_feature_columns(train_df, ML_DIR)
        X_imp = train_df[candidate_cols].copy() if candidate_cols else pd.DataFrame(index=train_df.index)
        y_imp = train_df[target_col].astype(float)
        ok = y_imp.notna()
        X_imp, y_imp = X_imp.loc[ok], y_imp.loc[ok]

        full_df, _ = compute_feature_importance_train(X_imp, y_imp, candidate_cols)
        kept, dropped = dedupe_correlated_features(
            X_imp, full_df, threshold=CORR_DEDUP_THRESHOLD, across_families=True,
        )
        imp_df, cv_r2 = compute_feature_importance_train(X_imp[kept], y_imp, kept)
        importance_by_target[target_key] = imp_df
        path = DATA_DIR / f'{target_key}_feature_importance_train.csv'
        imp_df.to_csv(path, index=False)

        available_cols = set(train_df.columns) | set(merged_opt.columns)
        feats, n_sig = select_significant_features(imp_df, available_cols)
        features_by_target[target_key] = feats
        print(
            f"{TARGET_LABELS[target_key]} [train-only importance, n={len(y_imp)}]: "
            f"{len(feats)} features ({top_note}, n_significant={n_sig}, "
            f"dedup {len(kept)}/{len(candidate_cols)}, dropped {len(dropped)}, "
            f"CV R2={cv_r2:.3f}) -> {path}"
        )
    else:
        path = DATA_DIR / f'{target_key}_feature_importance_{FEATURE_IMPORTANCE_POOL}.csv'
        if not path.is_file():
            raise FileNotFoundError(
                f'{path} missing. Run feature_importance.ipynb '
                f'with OPT_ONLY matching FEATURE_IMPORTANCE_POOL={FEATURE_IMPORTANCE_POOL!r}.'
            )
        imp_df = pd.read_csv(path)
        importance_by_target[target_key] = imp_df
        feats, n_sig = select_significant_features(imp_df, available_cols)
        features_by_target[target_key] = feats
        print(
            f"{TARGET_LABELS[target_key]} [CSV importance]: "
            f"{len(feats)} features ({top_note}, n_significant={n_sig}, "
            f"importance rows={len(imp_df)}) <- {path}"
        )



# In[ ]:





# In[ ]:





# ## Regularization and model selection
# 
# All models use median imputation + `StandardScaler`. Set `HYPERPARAM_SEARCH` (`'random'` / `'grid'`), `SELECT_BEST_BY` (`'cv'` / `'test'`), and `BEST_MODEL_CRITERION`. Both train-CV and holdout metrics are stored.
# 

# In[ ]:





# In[ ]:


RANDOM_STATE = 42
N_SPLITS = 3  # better fold size for n≈9–19 than 5
CV = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Best-model metric: RMSE | MAE | R2 | MaxAE.
BEST_MODEL_CRITERION = 'RMSE'
# Select winner by train CV ('cv') or holdout test ('test').
SELECT_BEST_BY = 'cv'  # 'cv' | 'test'
# Hyperparameter search: 'random' (RandomizedSearchCV) or 'grid' (GridSearchCV).
HYPERPARAM_SEARCH = 'random'  # 'random' | 'grid'
# Used only when HYPERPARAM_SEARCH == 'random' (capped by each model's grid size).
N_ITER_SEARCH = 40

if SELECT_BEST_BY not in {'cv', 'test'}:
    raise ValueError(f"SELECT_BEST_BY must be 'cv' or 'test', got {SELECT_BEST_BY!r}")
if HYPERPARAM_SEARCH not in {'random', 'grid'}:
    raise ValueError(f"HYPERPARAM_SEARCH must be 'random' or 'grid', got {HYPERPARAM_SEARCH!r}")

SEARCH_SCORING = {
    'RMSE': 'neg_root_mean_squared_error',
    'MAE': 'neg_mean_absolute_error',
    'MaxAE': 'neg_max_error',
    'R2': 'r2',
}

# Regularization grid: keep mild–moderate strength, but do NOT include alphas that
# force a null (mean) predictor (e.g. Ridge alpha=1e5 → train R²≈0).
ALPHAS = np.logspace(-3, 2, 16).tolist()
L1_RATIOS = np.round(np.linspace(0.05, 0.95, 10), 3).tolist()
# Lasso / ElasticNet: weak alpha + many features can need more iterations.
LINEAR_MAX_ITER = 100_000
LINEAR_TOL = 1e-3
SVR_C = np.logspace(-2, 1, 10).tolist()  # cap C; large C memorizes tiny n
SVR_GAMMA = ['scale', 'auto'] + np.logspace(-4, 0, 6).tolist()
SVR_EPSILON = np.logspace(-3, -1, 6).tolist()


def svr_param_grid() -> list[dict]:
    """Kernel-specific grids (avoid invalid Cartesian products)."""
    return [
        {
            'model__kernel': ['rbf'],
            'model__C': SVR_C,
            'model__gamma': SVR_GAMMA,
            'model__epsilon': SVR_EPSILON,
        },
        {
            'model__kernel': ['linear'],
            'model__C': SVR_C,
            'model__epsilon': SVR_EPSILON,
        },
        {
            'model__kernel': ['poly'],
            'model__C': SVR_C,
            'model__gamma': SVR_GAMMA,
            'model__epsilon': SVR_EPSILON,
            'model__degree': [2, 3],
            'model__coef0': [0.0, 0.5, 1.0],
        },
    ]


def _grid_size(param_grid) -> int:
    if isinstance(param_grid, list):
        return sum(_grid_size(pg) for pg in param_grid)
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size


def build_model_specs():
    specs = {
        'Ridge': {
            'estimator': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', Ridge()),
            ]),
            'param_grid': {'model__alpha': ALPHAS},
        },
        'Lasso': {
            'estimator': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', Lasso(max_iter=LINEAR_MAX_ITER, tol=LINEAR_TOL, selection='random', random_state=RANDOM_STATE)),
            ]),
            'param_grid': {'model__alpha': ALPHAS},
        },
        'ElasticNet': {
            'estimator': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', ElasticNet(max_iter=LINEAR_MAX_ITER, tol=LINEAR_TOL, selection='random', random_state=RANDOM_STATE)),
            ]),
            'param_grid': {
                'model__alpha': ALPHAS,
                'model__l1_ratio': L1_RATIOS,
            },
        },
        'SVR': {
            'estimator': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler()),
                ('model', SVR()),
            ]),
            'param_grid': svr_param_grid(),
        },
        'RandomForest': {
            'estimator': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
            ]),
            'param_grid': {
                'model__n_estimators': [100, 200, 300],
                'model__max_depth': [2, 3, 4],
                'model__min_samples_leaf': [2, 3],
                'model__min_samples_split': [2, 4],
                'model__max_features': ['sqrt', 0.5],
            },
        },
        'GradientBoosting': {
            'estimator': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]),
            'param_grid': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [1, 2],
                'model__min_samples_leaf': [2, 3],
                'model__subsample': [0.6, 0.8],
                'model__max_features': ['sqrt', 0.5],
            },
        },
        'XGBoost': {
            'estimator': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', XGBRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=0,
                    objective='reg:squarederror',
                )),
            ]),
            'param_grid': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__max_depth': [1, 2],
                'model__min_child_weight': [2, 3],
                'model__subsample': [0.6, 0.8],
                'model__colsample_bytree': [0.5, 0.8],
                'model__reg_lambda': [1.0, 10.0],
                'model__reg_alpha': [0.1, 1.0],
            },
        },
        'CatBoost': {
            'estimator': Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('model', CatBoostRegressor(
                    random_state=RANDOM_STATE,
                    verbose=0,
                    allow_writing_files=False,
                    thread_count=1,  # search uses n_jobs=-1; avoid thread oversubscription
                    bootstrap_type='Bernoulli',  # required when tuning subsample (< 1)
                )),
            ]),
            'param_grid': {
                'model__iterations': [50, 100, 200],
                'model__learning_rate': [0.01, 0.05, 0.1],
                'model__depth': [2, 3],
                'model__min_data_in_leaf': [2, 3],
                'model__subsample': [0.6, 0.8, 1.0],
                'model__l2_leaf_reg': [1.0, 3.0, 10.0],
            },
        },
    }
    for name, spec in specs.items():
        spec['n_grid'] = _grid_size(spec['param_grid'])
    return specs


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        'R2': float(r2_score(y_true, y_pred)),
        'MaxAE': float(max_error(y_true, y_pred)),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def effective_es_minus_eb(e_s, e_b) -> float:
    """Piecewise E_eff used in best_candidates: E_s - E_b if E_s > 0, else -E_b."""
    e_s = float(e_s)
    e_b = float(e_b)
    if e_s > 0.0:
        return e_s - e_b
    return -e_b


def metrics_from_results_row(row: pd.Series, split: str) -> dict[str, float]:
    return {
        'R2': float(row[f'{split}_R2']),
        'MaxAE': float(row[f'{split}_MaxAE']),
        'MAE': float(row[f'{split}_MAE']),
        'RMSE': float(row[f'{split}_RMSE']),
    }


def select_best_row(
    subset: pd.DataFrame,
    criterion: str = BEST_MODEL_CRITERION,
    *,
    by: str = SELECT_BEST_BY,
) -> pd.Series:
    """Pick best model by train-CV (`cv_*`) or holdout test (`test_*`)."""
    prefix = 'cv' if by == 'cv' else 'test'
    col = f'{prefix}_{criterion}'
    if col not in subset.columns:
        raise KeyError(
            f'Missing {col} in results; re-run training so metrics are stored.'
        )
    if criterion == 'R2':
        return subset.loc[subset[col].idxmax()]
    return subset.loc[subset[col].idxmin()]



SCORERS = {
    'R2': 'r2',
    'neg_MaxAE': 'neg_max_error',
    'neg_MAE': 'neg_mean_absolute_error',
    'neg_RMSE': 'neg_root_mean_squared_error',
}


def cv_metrics_from_scores(scores: dict) -> dict[str, float]:
    return {
        'R2': float(np.mean(scores['test_R2'])),
        'MaxAE': float(-np.mean(scores['test_neg_MaxAE'])),
        'MAE': float(-np.mean(scores['test_neg_MAE'])),
        'RMSE': float(-np.mean(scores['test_neg_RMSE'])),
    }


EV_TO_MEV = 1000
PLOT_SHOW_GRID = False  # best-model / derived / inset grids
PLOT_SHOW_TITLES = False  # best-model / derived titles
PLOT_SHOW_GRID_ALL_MODELS = True  # pred_vs_actual_*_all_models.pdf only
PLOT_SHOW_TITLES_ALL_MODELS = True  # pred_vs_actual_*_all_models.pdf only
PLOT_SHOW_TEST_INSET = False  # zoom inset on test region for E_s / E_s-E_b
PLOT_TEST_INSET_TARGETS = ('solution', 'solution_minus_binding')  # inset only for E_s best-model plot
PLOT_INSET_PAD = 0.15  # padding as fraction of test-range span
# Inset box in parent-axes coordinates: (x0, y0, width, height).
# Increase x0 → right; increase y0 → higher.
PLOT_INSET_BOUNDS = (0.11, 0.58, 0.40, 0.40)
PLOT_INSET_TICK_FONTSIZE = 9  # inset tick labels
PLOT_SPINE_COLOR = 'black'
PLOT_TRAIN_COLOR = 'darkblue'
PLOT_TEST_COLOR = 'crimson'
PLOT_LEGEND_FONTSIZE = 12
PLOT_ANNOTATION_FONTSIZE = 12
PLOT_LABEL_FONTSIZE = 16  # x/y axis label fontsize
PLOT_LEGEND_EDGECOLOR = 'black'
TICK_LENGTH = 6
TICK_WIDTH = 1.0

sns.set_theme(style='white')  # grid is applied explicitly per axes


def rmse_to_meV(rmse_ev: float) -> float:
    return float(rmse_ev) * EV_TO_MEV


def format_metric_for_plot(metric: str, value: float) -> str:
    if metric == 'R2':
        return f'{value:.4f}'
    return f'{value * EV_TO_MEV:.1f} meV'


def annotate_train_test_metric(
    ax,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    criterion: str = BEST_MODEL_CRITERION,
) -> None:
    ax.text(
        0.98,
        0.02,
        f"train {criterion} = {format_metric_for_plot(criterion, train_metrics[criterion])}\n"
        f"test {criterion} = {format_metric_for_plot(criterion, test_metrics[criterion])}",
        transform=ax.transAxes,
        va='bottom',
        ha='right',
        fontsize=PLOT_ANNOTATION_FONTSIZE,
        bbox={'boxstyle': 'round', 'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'},
        clip_on=False,
        zorder=5,
    )


def will_show_test_inset(target_key: str | None = None) -> bool:
    if not PLOT_SHOW_TEST_INSET:
        return False
    if target_key is None:
        return False
    return target_key in PLOT_TEST_INSET_TARGETS


def apply_axes_grid(ax, show_grid: bool) -> None:
    """Apply grid consistently on main axes and insets."""
    ax.grid(bool(show_grid), which='major', axis='both')
    ax.set_axisbelow(True)


def add_train_test_legend(ax, *, loc: str = 'upper left') -> None:
    leg = ax.legend(
        loc=loc,
        fontsize=PLOT_LEGEND_FONTSIZE,
        framealpha=0.9,
        borderpad=0.25,       # space between content and frame
        labelspacing=0.35,    # vertical gap between entries
        handletextpad=0.5,    # gap between marker and text
        borderaxespad=0.4,    # gap between legend and axes edge
        edgecolor=PLOT_LEGEND_EDGECOLOR,
        fancybox=True,
    )
    frame = leg.get_frame()
    frame.set_edgecolor(PLOT_LEGEND_EDGECOLOR)
    frame.set_linewidth(1.0)
    # Keep corners rounded without inflating the frame around the text.
    frame.set_boxstyle('round', pad=0.1, rounding_size=0.4)


def style_parity_axes(
    ax,
    *,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    criterion: str | None = None,
    target_key: str | None = None,
    show_grid: bool | None = None,
    with_inset: bool | None = None,
) -> None:
    annotate_train_test_metric(
        ax, train_metrics, test_metrics, criterion or BEST_MODEL_CRITERION,
    )
    if with_inset is None:
        with_inset = will_show_test_inset(target_key)
    legend_loc = 'center right' if with_inset else 'upper left'
    add_train_test_legend(ax, loc=legend_loc)
    # seaborn themes set xtick.bottom/ytick.left to False (labels only); re-enable tick marks.
    ax.tick_params(
        axis='both',
        which='major',
        direction='out',
        length=TICK_LENGTH,
        width=TICK_WIDTH,
        labelsize=PLOT_LABEL_FONTSIZE - 1,
        bottom=True,
        left=True,
        top=False,
        right=False,
    )
    ax.tick_params(
        axis='both',
        which='minor',
        labelsize=PLOT_LABEL_FONTSIZE - 3,
    )
    apply_axes_grid(ax, PLOT_SHOW_GRID if show_grid is None else show_grid)
    for spine in ax.spines.values():
        spine.set_color(PLOT_SPINE_COLOR)
        spine.set_linewidth(1.0)


def scatter_train_test(ax, y_train_pred, y_train_true, y_test_pred, y_test_true, *, s=55) -> None:
    ax.scatter(
        y_train_pred,
        y_train_true,
        s=s,
        alpha=0.85,
        facecolors=PLOT_TRAIN_COLOR,
        edgecolor='k',
        linewidth=0.4,
        label='train',
        zorder=3,
    )
    ax.scatter(
        y_test_pred,
        y_test_true,
        s=s,
        alpha=0.9,
        facecolors='none',
        edgecolors=PLOT_TEST_COLOR,
        linewidth=1.2,
        label='test',
        zorder=4,
    )


def _test_zoom_limits(y_test_pred, y_test_true, *, pad_frac: float = PLOT_INSET_PAD):
    x = np.asarray(y_test_pred, dtype=float)
    y = np.asarray(y_test_true, dtype=float)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    span = hi - lo if hi > lo else 1.0
    pad = pad_frac * span
    return (lo - pad, hi + pad)


def add_test_region_inset(
    ax,
    y_train_pred,
    y_train_true,
    y_test_pred,
    y_test_true,
    *,
    target_key: str | None = None,
    s: float = 35,
):
    """Optional zoom inset over the test-point region (E_s / E_s-E_b)."""
    if not PLOT_SHOW_TEST_INSET:
        return None
    if target_key is not None and target_key not in PLOT_TEST_INSET_TARGETS:
        return None
    y_test_pred = np.asarray(y_test_pred, dtype=float)
    y_test_true = np.asarray(y_test_true, dtype=float)
    if y_test_pred.size == 0:
        return None

    zoom = _test_zoom_limits(y_test_pred, y_test_true)
    axins = ax.inset_axes(PLOT_INSET_BOUNDS)
    scatter_train_test(axins, y_train_pred, y_train_true, y_test_pred, y_test_true, s=s)
    axins.plot(zoom, zoom, 'k--', lw=1, alpha=0.7)
    axins.set_xlim(zoom)
    axins.set_ylim(zoom)
    axins.set_aspect('equal', adjustable='box')
    axins.tick_params(
        axis='both',
        which='major',
        labelsize=PLOT_INSET_TICK_FONTSIZE,
        length=4,
        width=0.8,
        bottom=True,
        left=True,
        top=False,
        right=False,
        pad=2,
    )
    axins.tick_params(
        axis='both',
        which='minor',
        labelsize=max(6, PLOT_INSET_TICK_FONTSIZE - 2),
    )
    axins.locator_params(axis='x', nbins=6)  
    axins.locator_params(axis='y', nbins=6)  
    apply_axes_grid(axins, PLOT_SHOW_GRID)
    for spine in axins.spines.values():
        spine.set_color(PLOT_SPINE_COLOR)
        spine.set_linewidth(1.0)
    _rect, conn1, conn2 = mark_inset(
        ax, axins, loc1=2, loc2=4, fc='none', ec='0.35', lw=0.8,
    )
    # Zoom rectangle stays solid; connector lines are dashed.
    conn1.set_linestyle('--')
    conn2.set_linestyle('--')
    return axins



# In[ ]:





# In[ ]:


model_specs = build_model_specs()
print('Grid sizes (combinations per model):')
for name, spec in model_specs.items():
    print(f'  {name:18} {spec["n_grid"]:4d}')
print(f'  total per target   {sum(s["n_grid"] for s in model_specs.values())}')


def features_in_df(df: pd.DataFrame, feature_cols: list[str]) -> list[str]:
    return [f for f in feature_cols if f in df.columns]


def get_xy(df: pd.DataFrame, target_col: str, feature_cols: list[str]):
    if df.empty:
        raise ValueError(
            f'Empty dataframe for target {target_col}. Re-run the data-loading cell above.'
        )
    cols = features_in_df(df, feature_cols)
    if not cols:
        raise ValueError(
            f'No selected features found in dataframe columns for {target_col}. '
            f'Re-run feature_importance / check matminer coverage.'
        )
    work = df.dropna(subset=[target_col])
    if work.empty:
        raise ValueError(f'All rows have NaN target {target_col}')
    X = work[cols].apply(pd.to_numeric, errors='coerce')
    y = work[target_col].astype(float).values
    elements = work['element'].astype(str).values
    return X, y, elements


def cv_for_n(n_samples: int, *, max_splits: int = N_SPLITS) -> KFold:
    if n_samples < 2:
        raise ValueError(
            f'Need at least 2 training samples for CV, got {n_samples}. '
            f'Check merged_opt sizes in the data cell.'
        )
    n_splits = min(max_splits, n_samples)
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def catboost_subsample_grid(n_train: int, n_splits: int) -> list[float]:
    """Bernoulli subsample needs enough units in each CV training fold."""
    n_fold_train = max(1, int(np.floor(n_train * (n_splits - 1) / n_splits)))
    candidates = [0.5, 0.6, 0.8, 1.0]
    # CatBoost raises if subsample * n_fold_train is too small (~<5).
    safe = [s for s in candidates if s * n_fold_train >= 5]
    return safe if safe else [1.0]


def _clip_leaf_grid(values, *, n_train: int) -> list:
    """Keep leaf sizes that still allow splits on small training sets."""
    max_leaf = max(1, n_train // 4)
    clipped = [int(v) for v in values if int(v) <= max_leaf]
    return clipped if clipped else [1]


def param_grid_for_model(model_name: str, param_grid, *, n_train: int, n_splits: int):
    grid = param_grid
    if model_name == 'CatBoost':
        grid = {
            **param_grid,
            'model__subsample': catboost_subsample_grid(n_train, n_splits),
        }

    leaf_key = {
        'RandomForest': 'model__min_samples_leaf',
        'GradientBoosting': 'model__min_samples_leaf',
        'XGBoost': 'model__min_child_weight',
        'CatBoost': 'model__min_data_in_leaf',
    }.get(model_name)
    if leaf_key and isinstance(grid, dict) and leaf_key in grid:
        grid = {**grid, leaf_key: _clip_leaf_grid(grid[leaf_key], n_train=n_train)}
    return grid


def pack_predictions(*, holdout: bool, y_train, y_train_pred, y_test, y_test_pred, train_elements, test_elements):
    if holdout:
        return {
            'holdout': True,
            'y_train_true': y_train,
            'y_train_pred': y_train_pred,
            'y_test_true': y_test,
            'y_test_pred': y_test_pred,
            'train_elements': train_elements,
            'test_elements': test_elements,
        }
    return {
        'holdout': False,
        'y_true': y_train,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
        'elements': train_elements,
    }


results_rows = []


best_models = {}
predictions = {}

for target_key, target_col in TARGETS.items():
    feature_cols = features_by_target[target_key]
    holdout = target_key in OPT_HOLDOUT_TARGETS

    if holdout:
        use_opt_only = use_opt_only_for(target_key)
        train_df = merged_opt_holdout_train if use_opt_only else merged_holdout_train
        test_df = merged_opt_holdout_test if use_opt_only else merged_holdout_test
        X_train, y_train, train_elements = get_xy(train_df, target_col, feature_cols)
        X_test, y_test, test_elements = get_xy(test_df, target_col, feature_cols)
        target_cv = cv_for_n(len(y_train))
        mode = 'opt-only' if use_opt_only else 'pre-train/opt-test'
        search_label = (
            f'RandomizedSearchCV n_iter≤{N_ITER_SEARCH}'
            if HYPERPARAM_SEARCH == 'random'
            else 'GridSearchCV'
        )
        print('\n' + '=' * 72)
        print(
            f'Target: {TARGET_LABELS[target_key]} ({target_col}), features={X_train.shape[1]} '
            f'[{mode}: train n={len(y_train)}, test n={len(y_test)}, '
            f'{search_label}, {target_cv.n_splits}-fold on train]'
        )
        print('=' * 72)
    else:
        X_train, y_train, train_elements = get_xy(merged_opt, target_col, feature_cols)
        X_test = y_test = test_elements = None
        target_cv = cv_for_n(len(y_train))
        search_label = (
            f'RandomizedSearchCV n_iter≤{N_ITER_SEARCH}'
            if HYPERPARAM_SEARCH == 'random'
            else 'GridSearchCV'
        )
        print('\n' + '=' * 72)
        print(
            f'Target: {TARGET_LABELS[target_key]} ({target_col}), '
            f'features={X_train.shape[1]}, n={len(y_train)}, '
            f'{search_label}, {target_cv.n_splits}-fold'
        )
        print('=' * 72)

    search_scoring = SEARCH_SCORING[BEST_MODEL_CRITERION]

    for model_name, spec in model_specs.items():
        param_grid = param_grid_for_model(
            model_name, spec['param_grid'],
            n_train=len(y_train), n_splits=target_cv.n_splits,
        )
        n_iter = min(N_ITER_SEARCH, _grid_size(param_grid))
        if HYPERPARAM_SEARCH == 'random':
            search = RandomizedSearchCV(
                estimator=spec['estimator'],
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=target_cv,
                scoring=search_scoring,
                refit=True,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                error_score=np.nan,
            )
        else:
            search = GridSearchCV(
                estimator=spec['estimator'],
                param_grid=param_grid,
                cv=target_cv,
                scoring=search_scoring,
                refit=True,
                n_jobs=-1,
                error_score=np.nan,
            )
            n_iter = _grid_size(param_grid)
        search.fit(X_train, y_train)
        best_est = search.best_estimator_
        best_models[(target_key, model_name)] = best_est

        y_train_pred = best_est.predict(X_train)
        train_metrics = compute_metrics(y_train, y_train_pred)

        # Train-CV metrics (used when SELECT_BEST_BY == 'cv').
        cv_scores = cross_validate(
            clone(best_est), X_train, y_train, cv=target_cv, scoring=SCORERS,
            n_jobs=-1,
        )
        cv_metrics = cv_metrics_from_scores(cv_scores)

        if holdout:
            y_test_pred = best_est.predict(X_test)
            test_metrics = compute_metrics(y_test, y_test_pred)
            cv_note = 'Test (opt-only hold-out):'
        else:
            test_metrics = cv_metrics
            y_test_pred = cross_val_predict(
                clone(best_est), X_train, y_train, cv=target_cv, n_jobs=-1
            )
            cv_note = 'Test (out-of-fold = train CV):'

        predictions[(target_key, model_name)] = pack_predictions(
            holdout=holdout,
            y_train=y_train,
            y_train_pred=y_train_pred,
            y_test=y_test if holdout else y_train,
            y_test_pred=y_test_pred,
            train_elements=train_elements,
            test_elements=test_elements if holdout else train_elements,
        )

        row = {
            'target_key': target_key,
            'target_label': TARGET_LABELS[target_key],
            'model': model_name,
            'n_features': int(X_train.shape[1]),
            'n_train': len(y_train),
            'n_test': len(y_test) if holdout else len(y_train),
            'eval_mode': (
                'opt_only_holdout'
                if holdout and use_opt_only_for(target_key)
                else ('pre_train_opt_test_holdout' if holdout else 'cv')
            ),
            'hyperparam_search': HYPERPARAM_SEARCH,
            'select_best_by': SELECT_BEST_BY,
            'best_params': repr(search.best_params_),
            'n_iter_search': n_iter,
            'train_R2': train_metrics['R2'],
            'train_MaxAE': train_metrics['MaxAE'],
            'train_MAE': train_metrics['MAE'],
            'train_RMSE': train_metrics['RMSE'],
            'cv_R2': cv_metrics['R2'],
            'cv_MaxAE': cv_metrics['MaxAE'],
            'cv_MAE': cv_metrics['MAE'],
            'cv_RMSE': cv_metrics['RMSE'],
            'test_R2': test_metrics['R2'],
            'test_MaxAE': test_metrics['MaxAE'],
            'test_MAE': test_metrics['MAE'],
            'test_RMSE': test_metrics['RMSE'],
        }
        results_rows.append(row)

        print(f"\n--- {model_name} ---")
        print(f'Best parameters ({HYPERPARAM_SEARCH}, n_iter/grid={n_iter}):', search.best_params_)
        print(
            'Train: '
            f"R2={train_metrics['R2']:.4f}, MaxAE={train_metrics['MaxAE']:.4f}, "
            f"MAE={train_metrics['MAE']:.4f}, RMSE={rmse_to_meV(train_metrics['RMSE']):.1f} meV"
        )
        print(
            f'Train CV: '
            f"R2={cv_metrics['R2']:.4f}, MaxAE={cv_metrics['MaxAE']:.4f}, "
            f"MAE={cv_metrics['MAE']:.4f}, RMSE={rmse_to_meV(cv_metrics['RMSE']):.1f} meV"
        )
        print(
            f'{cv_note} '
            f"R2={test_metrics['R2']:.4f}, MaxAE={test_metrics['MaxAE']:.4f}, "
            f"MAE={test_metrics['MAE']:.4f}, RMSE={rmse_to_meV(test_metrics['RMSE']):.1f} meV"
        )


results_df = pd.DataFrame(results_rows)
results_path = DATA_DIR / f'model_train_results{METRICS_SUFFIX}.csv'
results_df.to_csv(results_path, index=False)
print(f'\nSaved summary to {results_path}')

METRIC_COLS = [
    'model',
    'train_R2', 'train_MaxAE', 'train_MAE', 'train_RMSE',
    'cv_R2', 'cv_MaxAE', 'cv_MAE', 'cv_RMSE',
    'test_R2', 'test_MaxAE', 'test_MAE', 'test_RMSE',
]
for target_key in TARGETS:
    per_target = results_df.loc[results_df['target_key'] == target_key, METRIC_COLS].copy()
    out = DATA_DIR / f'model_metrics_{target_key}{METRICS_SUFFIX}.csv'
    per_target.to_csv(out, index=False)
    print(f'Saved {out}')

_sel_prefix = 'cv' if SELECT_BEST_BY == 'cv' else 'test'
print(
    f'\nBest models by {SELECT_BEST_BY} {BEST_MODEL_CRITERION} '
    f'(FEATURE_IMPORTANCE_ON_TRAIN_ONLY={FEATURE_IMPORTANCE_ON_TRAIN_ONLY}, '
    f'HYPERPARAM_SEARCH={HYPERPARAM_SEARCH}):'
)
for target_key in TARGETS:
    best = select_best_row(results_df[results_df['target_key'] == target_key])
    print(
        f"  {TARGET_LABELS[target_key]}: {best['model']} "
        f"({_sel_prefix}_{BEST_MODEL_CRITERION}={best[f'{_sel_prefix}_{BEST_MODEL_CRITERION}']:.4f}, "
        f"test_{BEST_MODEL_CRITERION}={best[f'test_{BEST_MODEL_CRITERION}']:.4f})"
    )



# In[ ]:





# In[11]:



# In[ ]:





# In[ ]:


summary_df = results_df.copy()
summary_df['train_RMSE_meV'] = summary_df['train_RMSE'].map(rmse_to_meV)
summary_df['cv_RMSE_meV'] = summary_df['cv_RMSE'].map(rmse_to_meV)
summary_df['test_RMSE_meV'] = summary_df['test_RMSE'].map(rmse_to_meV)
display_cols = [
    'target_label', 'model', 'n_features',
    'train_R2', 'train_RMSE_meV',
    'cv_R2', 'cv_MaxAE', 'cv_MAE', 'cv_RMSE_meV',
    'test_R2', 'test_MaxAE', 'test_MAE', 'test_RMSE_meV',
]
_sort_col = f"{'cv' if SELECT_BEST_BY == 'cv' else 'test'}_{BEST_MODEL_CRITERION}"
# Sort on full summary_df (has cv_RMSE / test_RMSE), then show display columns.
display(
    summary_df.sort_values(
        ['target_label', _sort_col],
        ascending=[True, BEST_MODEL_CRITERION != 'R2'],
    )[display_cols]
)


# In[12]:


# In[ ]:


print('Best parameters per target and model:\n')
for target_key in TARGETS:
    print(f"{TARGET_LABELS[target_key]}:")
    subset = results_df[results_df['target_key'] == target_key]
    for _, row in subset.iterrows():
        model_name = row['model']
        params = best_models[(target_key, model_name)].get_params()
        tuned = {k.replace('model__', ''): v for k, v in params.items() if k.startswith('model__')}
        print(f"  {model_name}: {tuned}")
    print()


# In[ ]:


def _scatter_limits(y_true, y_train, y_test):
    lo = float(min(y_true.min(), y_train.min(), y_test.min()))
    hi = float(max(y_true.max(), y_train.max(), y_test.max()))
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    return (lo - pad, hi + pad)


for target_key in TARGETS:
    model_names = list(model_specs.keys())
    n_models = len(model_names)
    n_cols = 4
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, model_name in zip(axes, model_names):
        pred = predictions[(target_key, model_name)]
        if pred['holdout']:
            y_train_true = pred['y_train_true']
            y_test_true = pred['y_test_true']
        else:
            y_train_true = pred['y_true']
            y_test_true = pred['y_true']
        y_train = pred['y_train_pred']
        y_test = pred['y_test_pred']
        lim = _scatter_limits(
            np.concatenate([y_train_true, y_test_true]),
            y_train,
            y_test,
        )
        scatter_train_test(ax, y_train, y_train_true, y_test, y_test_true, s=40)
        ax.plot(lim, lim, 'k--', lw=1, alpha=0.7)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel('ML-predicted (eV)', fontsize=PLOT_LABEL_FONTSIZE)
        ax.set_ylabel('DFT-calculated (eV)', fontsize=PLOT_LABEL_FONTSIZE)
        ax.locator_params(axis='x', nbins=6)  
        ax.locator_params(axis='y', nbins=6)  
        row = results_df.query('target_key == @target_key and model == @model_name').iloc[0]
        if PLOT_SHOW_TITLES_ALL_MODELS:
            ax.set_title(model_name, pad=5, fontsize=PLOT_LABEL_FONTSIZE)
        style_parity_axes(
            ax,
            train_metrics=metrics_from_results_row(row, 'train'),
            test_metrics=metrics_from_results_row(row, 'test'),
            show_grid=PLOT_SHOW_GRID_ALL_MODELS,
            with_inset=False,
        )
    for ax in axes[n_models:]:
        ax.axis('off')
    if PLOT_SHOW_TITLES_ALL_MODELS:
        fig.suptitle(f'Predicted vs calculated : {TARGET_LABELS[target_key]}', y=0.98, fontsize=PLOT_LABEL_FONTSIZE+2)
        fig.subplots_adjust(hspace=0.32, wspace=0.32, top=0.90)
    else:
        fig.subplots_adjust(hspace=0.32, wspace=0.32)
    out = FIGURES_DIR / f'pred_vs_actual_{target_key}_all_models.pdf'
    fig.tight_layout()
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.show()
    print(f'Saved {out}')



# In[ ]:





# In[ ]:


for target_key in TARGETS:
    subset = results_df[results_df['target_key'] == target_key]
    best_row = select_best_row(subset)
    model_name = best_row['model']
    pred = predictions[(target_key, model_name)]
    if pred['holdout']:
        y_train_true = pred['y_train_true']
        y_test_true = pred['y_test_true']
    else:
        y_train_true = pred['y_true']
        y_test_true = pred['y_true']
    y_train = pred['y_train_pred']
    y_test = pred['y_test_pred']

    lim = _scatter_limits(
        np.concatenate([y_train_true, y_test_true]),
        y_train,
        y_test,
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    scatter_train_test(ax, y_train, y_train_true, y_test, y_test_true, s=55)
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.7)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('ML-predicted (eV)', fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel('DFT-calculated (eV)', fontsize=PLOT_LABEL_FONTSIZE)
    if PLOT_SHOW_TITLES:
        ax.set_title(f'{TARGET_LABELS[target_key]} — {model_name}')
    style_parity_axes(
        ax,
        train_metrics=metrics_from_results_row(best_row, 'train'),
        test_metrics=metrics_from_results_row(best_row, 'test'),
        target_key=target_key,
    )
    add_test_region_inset(
        ax, y_train, y_train_true, y_test, y_test_true,
        target_key=target_key, s=40,
    )
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout()
    out = FIGURES_DIR / f'best_model_{target_key}.pdf'
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.show()
    print(
        f"Best model for {TARGET_LABELS[target_key]} "
        f"(selected by {SELECT_BEST_BY} {BEST_MODEL_CRITERION}): {model_name} -> {out}"
    )



# In[ ]:


# Derived E_eff on _opt from best E_s / E_b models (piecewise: E_s-E_b if E_s>0 else -E_b).
best_binding_row = select_best_row(results_df[results_df['target_key'] == 'binding'])
best_solution_row = select_best_row(results_df[results_df['target_key'] == 'solution'])
binding_model_name = best_binding_row['model']
solution_model_name = best_solution_row['model']

pred_b = predictions[('binding', binding_model_name)]
pred_s = predictions[('solution', solution_model_name)]

def holdout_preds_by_element(pred) -> tuple[dict[str, float], dict[str, str]]:
    """Map opt dopants to hold-out train/test predictions (same split as training)."""
    by_el: dict[str, float] = {}
    split: dict[str, str] = {}
    if pred.get('holdout', False) or 'train_elements' in pred:
        for el, yp in zip(pred['train_elements'], pred['y_train_pred']):
            el = str(el)
            if el in opt_elements:
                by_el[el] = float(yp)
                split[el] = 'train'
        for el, yp in zip(pred['test_elements'], pred['y_test_pred']):
            el = str(el)
            by_el[el] = float(yp)
            split[el] = 'test'
        return by_el, split
    for el, yp in zip(pred['elements'], pred['y_test_pred']):
        el = str(el)
        by_el[el] = float(yp)
        split[el] = 'test'
    return by_el, split


es_by_el, es_split = holdout_preds_by_element(pred_s)
eb_by_el, _ = holdout_preds_by_element(pred_b)

opt_work = merged_opt.dropna(
    subset=['solution_energy_eV', 'vac_binding_energy_eV']
).copy()
opt_work['element'] = opt_work['element'].astype(str)
opt_work['E_s_dft'] = opt_work['solution_energy_eV'].astype(float)
opt_work['E_b_dft'] = opt_work['vac_binding_energy_eV'].astype(float)
opt_work['E_s_pred'] = opt_work['element'].map(es_by_el)
opt_work['E_b_pred'] = opt_work['element'].map(eb_by_el)
opt_work['y_true'] = [
    effective_es_minus_eb(es, eb)
    for es, eb in zip(opt_work['E_s_dft'], opt_work['E_b_dft'])
]
opt_work['y_pred'] = [
    effective_es_minus_eb(es, eb)
    for es, eb in zip(opt_work['E_s_pred'], opt_work['E_b_pred'])
]
opt_work['split'] = opt_work['element'].map(es_split)
opt_work = opt_work.dropna(subset=['E_s_pred', 'E_b_pred', 'split'])

train_mask = opt_work['split'] == 'train'
test_mask = opt_work['split'] == 'test'
y_train_true = opt_work.loc[train_mask, 'y_true'].to_numpy(dtype=float)
y_train_pred = opt_work.loc[train_mask, 'y_pred'].to_numpy(dtype=float)
y_test_true = opt_work.loc[test_mask, 'y_true'].to_numpy(dtype=float)
y_test_pred = opt_work.loc[test_mask, 'y_pred'].to_numpy(dtype=float)

derived_train_metrics = compute_metrics(y_train_true, y_train_pred)
derived_test_metrics = compute_metrics(y_test_true, y_test_pred)
derived_all_metrics = compute_metrics(
    opt_work['y_true'].to_numpy(dtype=float),
    opt_work['y_pred'].to_numpy(dtype=float),
)

print(
    f'Derived $E_{{\\mathrm{{eff}}}}$ on _opt from best models '
    f'(E_s>0: E_s-E_b; E_s<=0: -E_b): '
    f'E_s={solution_model_name}, E_b={binding_model_name}'
)
print(
    f"  train (n={len(y_train_true)}): "
    f"R2={derived_train_metrics['R2']:.4f}, MaxAE={derived_train_metrics['MaxAE']:.4f}, "
    f"MAE={derived_train_metrics['MAE']:.4f}, RMSE={rmse_to_meV(derived_train_metrics['RMSE']):.1f} meV"
)
print(
    f"  test  (n={len(y_test_true)}): "
    f"R2={derived_test_metrics['R2']:.4f}, MaxAE={derived_test_metrics['MaxAE']:.4f}, "
    f"MAE={derived_test_metrics['MAE']:.4f}, RMSE={rmse_to_meV(derived_test_metrics['RMSE']):.1f} meV"
)
print(
    f"  all   (n={len(opt_work)}): "
    f"R2={derived_all_metrics['R2']:.4f}, MaxAE={derived_all_metrics['MaxAE']:.4f}, "
    f"MAE={derived_all_metrics['MAE']:.4f}, RMSE={rmse_to_meV(derived_all_metrics['RMSE']):.1f} meV"
)

lim = _scatter_limits(
    np.concatenate([y_train_true, y_test_true]),
    y_train_pred,
    y_test_pred,
)
fig, ax = plt.subplots(figsize=(5, 5))
scatter_train_test(ax, y_train_pred, y_train_true, y_test_pred, y_test_true, s=55)
ax.plot(lim, lim, 'k--', lw=1, alpha=0.7)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel('ML-predicted (eV)', fontsize=PLOT_LABEL_FONTSIZE)
ax.set_ylabel('DFT-calculated (eV)', fontsize=PLOT_LABEL_FONTSIZE)
if PLOT_SHOW_TITLES:
    ax.set_title(
        rf'Derived $E_{{\mathrm{{eff}}}}$ '
        rf'({solution_model_name} / {binding_model_name})'
    )
style_parity_axes(
    ax,
    train_metrics=derived_train_metrics,
    test_metrics=derived_test_metrics,
    with_inset=False,
)
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()
out = FIGURES_DIR / 'best_model_solution_minus_binding_derived.pdf'
fig.savefig(out, dpi=600, bbox_inches='tight')
plt.show()
print(f'Saved {out}')

derived_energies_df = opt_work[
    [
        'element', 'split',
        'E_s_dft', 'E_b_dft', 'y_true',
        'E_s_pred', 'E_b_pred', 'y_pred',
    ]
].rename(columns={'y_true': 'E_eff_dft', 'y_pred': 'E_eff_pred'})
derived_energies_path = DATA_DIR / f'derived_effective_energies{METRICS_SUFFIX}.csv'
derived_energies_df.to_csv(derived_energies_path, index=False)
print(f'Saved energies {derived_energies_path}')

derived_metrics_df = pd.DataFrame(
    [
        {'split': 'train', 'n': len(y_train_true), **derived_train_metrics},
        {'split': 'test', 'n': len(y_test_true), **derived_test_metrics},
        {'split': 'all_opt', 'n': len(opt_work), **derived_all_metrics},
    ]
)
derived_path = DATA_DIR / f'derived_solution_minus_binding_metrics{METRICS_SUFFIX}.csv'
derived_metrics_df.to_csv(derived_path, index=False)
print(f'Saved metrics {derived_path}')
display(derived_energies_df)
display(derived_metrics_df)


# In[ ]:





# In[ ]:




