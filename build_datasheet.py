"""Build ML datasheet from vacancy-energy CSVs and MP stable-phase features."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from solution_thermo import SOLUTION_THERMO_FEATURE_COLS, add_solution_thermo_features

PathLike = Union[str, Path]

ELEMENTAL_FEATURE_COLS = ["delta_r_vs_Li_pm", "delta_valency_vs_Li", "delta_chi_vs_Li"]
_UNARY_MATMINER_META = {"element", "material_id", "role", "endmember_formula"}
_ENDMEMBER_MATMINER_META = {"element", "endmember_material_id", "role", "endmember_formula"}


def _read_energy_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Dopant" in df.columns:
        df = df.rename(columns={"Dopant": "element"})
    df["element"] = df["element"].astype(str).str.strip()
    return df


def optimized_dopant_list(
    repo_root: PathLike,
    *,
    binding_file: str = "vac_binding_energy_opt.csv",
    solution_file: str = "vac_solution_energy_opt.csv",
) -> list[str]:
    """Dopants present in both optimized binding and solution energy CSVs."""
    repo_root = Path(repo_root)
    data_dir = repo_root / "data"
    bind = set(_read_energy_csv(data_dir / binding_file)["element"])
    sol = set(_read_energy_csv(data_dir / solution_file)["element"])
    return sorted(bind & sol)


def build_datasheet(
    repo_root: PathLike,
    *,
    mp_phases_path: Optional[PathLike] = None,
    elemental_props_path: Optional[PathLike] = None,
    elemental_props_df: Optional[pd.DataFrame] = None,
    energy_pool: str = "opt",
    binding_energy_file: Optional[str] = None,
    solution_energy_file: Optional[str] = None,
) -> pd.DataFrame:
    """Merge vacancy energies with MP phase, thermo, and endmember metadata.

    ``energy_pool``:
    - ``"opt"`` (default): optimized binding + solution CSVs (~19 dopants)
    - ``"pre"``: pre-optimized CSVs (~50 dopants)
    """
    repo_root = Path(repo_root)
    data_dir = repo_root / "data"
    ml_dir = repo_root / "ml_part"

    if energy_pool == "opt":
        bind_name = binding_energy_file or "vac_binding_energy_opt.csv"
        sol_name = solution_energy_file or "vac_solution_energy_opt.csv"
    elif energy_pool == "pre":
        bind_name = binding_energy_file or "vac_binding_energy_pre.csv"
        sol_name = solution_energy_file or "vac_solution_energy_pre.csv"
    else:
        raise ValueError(f"energy_pool must be 'opt' or 'pre', got {energy_pool!r}")

    bind = _read_energy_csv(data_dir / bind_name)
    sol = _read_energy_csv(data_dir / sol_name)
    df = bind.merge(sol, on="element", how="inner", validate="one_to_one")

    if mp_phases_path is None:
        mp_phases_path = ml_dir / "data" / "dopant_stable_experimental_phases_mp.csv"
    phases = pd.read_csv(mp_phases_path)
    phases["element"] = phases["element"].astype(str).str.strip()
    phases = phases[phases["role"] == "dopant"].drop(columns=["role"])

    phase_cols = [c for c in phases.columns if c != "element"]
    rename = {c: c if c.startswith("mp_") else f"mp_{c}" for c in phase_cols}
    phases = phases.rename(columns=rename)

    df = df.merge(phases, on="element", how="left", validate="one_to_one")

    if elemental_props_df is not None:
        props = elemental_props_df.copy()
    else:
        if elemental_props_path is None:
            # Hume–Rothery-style deltas vs Li from create_data.ipynb
            elemental_props_path = ml_dir / "data" / "elemental_props_vs_li.csv"
        props = (
            pd.read_csv(elemental_props_path)
            if Path(elemental_props_path).is_file()
            else None
        )

    if props is not None:
        props["element"] = props["element"].astype(str).str.strip()
        # Keep only identity + delta_* ML features (drop raw valency/chi if present).
        keep = ["element", *ELEMENTAL_FEATURE_COLS]
        keep = [c for c in keep if c in props.columns]
        props = props[keep]
        df = df.merge(props, on="element", how="left", validate="one_to_one")

    df = add_solution_thermo_features(df, ml_dir=ml_dir)
    df = merge_endmember_phase_features(df, ml_dir=ml_dir)
    return drop_absolute_radius_columns(df)


def load_structure_matminer_tables(
    ml_dir: PathLike,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Load unary and endmember matminer feature tables from ``ml_part/data``."""
    ml_dir = Path(ml_dir)
    data_dir = ml_dir / "data"

    matminer_df = pd.read_csv(data_dir / "structure_matminer_features.csv")
    if "role" in matminer_df.columns:
        matminer_df = matminer_df[matminer_df["role"] == "dopant"].copy()

    endmember_path = data_dir / "structure_endmember_matminer_features.csv"
    endmember_matminer_df = pd.read_csv(endmember_path) if endmember_path.is_file() else None
    return matminer_df, endmember_matminer_df


def merge_structure_matminer(
    df: pd.DataFrame,
    ml_dir: PathLike,
    *,
    verbose: bool = True,
) -> pd.DataFrame:
    """Left-merge unary + endmember matminer columns onto ``df`` by ``element``."""
    matminer_df, endmember_matminer_df = load_structure_matminer_tables(ml_dir)
    merged = df.copy()
    for table in (matminer_df, endmember_matminer_df):
        if table is None:
            continue
        chunk = table.copy()
        drop_cols = [c for c in chunk.columns if c in merged.columns and c != "element"]
        chunk = chunk.drop(columns=drop_cols, errors="ignore")
        merged = merged.merge(chunk, on="element", how="left", validate="m:1")

    if len(merged) == 0:
        raise ValueError(f"structure-feature merge yielded 0 rows (input had {len(df)})")

    if verbose:
        mm_cols = [c for c in merged.columns if c != "element" and c not in df.columns]
        if mm_cols:
            n_mm = int(merged[mm_cols].notna().any(axis=1).sum())
            print(f"  structure features attached for {n_mm}/{len(merged)} rows")
    return merged


def load_ml_frame(
    repo_root: PathLike,
    *,
    energy_pool: str = "opt",
    attach_matminer: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build the datasheet and optionally attach matminer structure features."""
    df = build_datasheet(repo_root, energy_pool=energy_pool)
    if attach_matminer:
        df = merge_structure_matminer(df, Path(repo_root) / "ml_part", verbose=verbose)
    return df


def holdout_test_count(n_samples: int, test_fraction: float = 0.2) -> int:
    """Round hold-out size to the nearest integer (at least one test sample)."""
    if n_samples < 2:
        raise ValueError(f"Need at least 2 samples for a hold-out split, got {n_samples}")
    return max(1, round(n_samples * test_fraction))


def split_pre_train_opt_test(
    full_df: pd.DataFrame,
    opt_df: pd.DataFrame,
    *,
    test_fraction: float = 0.2,
    n_test: Optional[int] = None,
    random_state: int = 42,
    prefer_opt_energies: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold-out on the full dopant pool with an **opt-only** test set.

    Concept:
    - Use all pre-optimized dopants as the pool (``full_df``).
    - Draw the test set **only** from optimized dopants (``opt_df``).
    - Train = remaining opt dopants + all non-opt pre dopants.

    Test size defaults to ``round(len(full_df) * test_fraction)`` (capped so at least
    one opt dopant remains in train). With 50 pre / 19 opt and ``n_test=11``:
    test = 11 opt, train = 8 opt + 31 pre-only = 39.

    When ``prefer_opt_energies`` is True, opt dopants in both splits use energy /
    target columns from ``opt_df`` (pre rows supply features / non-opt energies).
    """
    from sklearn.model_selection import train_test_split

    energy_cols = [
        "vac_binding_energy_eV",
        "solution_energy_eV",
        "vac_position",
        "E_solution_minus_binding_eV",
    ]

    full = full_df.copy()
    opt = opt_df.copy()
    full["element"] = full["element"].astype(str).str.strip()
    opt["element"] = opt["element"].astype(str).str.strip()

    if prefer_opt_energies:
        opt_vals = opt.set_index("element")
        for col in energy_cols:
            if col in opt_vals.columns and col in full.columns:
                full.loc[full["element"].isin(opt_vals.index), col] = full["element"].map(
                    opt_vals[col]
                )
            elif col in opt_vals.columns:
                full[col] = full["element"].map(opt_vals[col])

    opt_elements = set(opt["element"])
    full_opt = full[full["element"].isin(opt_elements)].copy()
    full_non_opt = full[~full["element"].isin(opt_elements)].copy()

    if full_opt.empty:
        raise ValueError("No optimized dopants found inside the full hold-out pool")

    n_test_eff = holdout_test_count(len(full), test_fraction) if n_test is None else int(n_test)
    n_test_eff = min(max(1, n_test_eff), len(full_opt) - 1)

    opt_train, opt_test = train_test_split(
        full_opt,
        test_size=n_test_eff,
        random_state=random_state,
    )
    train = pd.concat([full_non_opt, opt_train], ignore_index=True)
    return train, opt_test.reset_index(drop=True)


def feature_dedup_family(name: str) -> str:
    """Feature family label (used for reporting; dedup may span families)."""
    feature = str(name)
    if feature.startswith("endmember_MagpieData"):
        return "endmember_matminer"
    if feature.startswith("MagpieData"):
        return "unary_matminer"
    if feature.startswith("endmember_"):
        return "endmember_meta"
    if feature.startswith(("mp_", "mu_")):
        return "thermo"
    if feature in ELEMENTAL_FEATURE_COLS:
        return "elemental"
    return "other"


def dedupe_correlated_features(
    X: pd.DataFrame,
    importance_df: pd.DataFrame,
    *,
    threshold: float = 0.99,
    importance_col: str = "importance_perm_mean",
    across_families: bool = True,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """
    Drop near-duplicates ranked by importance.

    By default (``across_families=True``), any pair with ``|r| > threshold`` is
    collapsed to the higher-importance feature — including unary vs endmember
    Magpie pairs (e.g. electronegativity with r = 1).
    Set ``across_families=False`` to only deduplicate within a feature family.
    """
    ranked = importance_df.sort_values(importance_col, ascending=False)

    def _pearson_r(a: str, b: str) -> float:
        pair = X[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
        return float(pair.corr().iloc[0, 1]) if len(pair) >= 2 else float("nan")

    kept: list[str] = []
    dropped: list[tuple[str, str, float]] = []
    for feat in ranked["feature"].astype(str):
        if feat not in X.columns:
            continue
        family = feature_dedup_family(feat)
        redundant_with: Optional[str] = None
        corr_value = float("nan")
        for keeper in kept:
            if not across_families and feature_dedup_family(keeper) != family:
                continue
            r = _pearson_r(feat, keeper)
            if np.isfinite(r) and abs(r) > threshold:
                redundant_with, corr_value = keeper, r
                break
        if redundant_with is None:
            kept.append(feat)
        else:
            dropped.append((feat, redundant_with, corr_value))
    return kept, dropped


def _has_observed_values(df: pd.DataFrame, col: str) -> bool:
    return pd.to_numeric(df[col], errors="coerce").notna().any()


def discover_ml_feature_columns(
    merged: pd.DataFrame,
    ml_dir: PathLike,
    *,
    drop_all_nan: bool = True,
) -> list[str]:
    """Return ordered ML feature column names for importance / model training."""
    matminer_df, endmember_matminer_df = load_structure_matminer_tables(ml_dir)

    matminer_feature_cols = [
        c
        for c in matminer_df.columns
        if c not in _UNARY_MATMINER_META and pd.api.types.is_numeric_dtype(matminer_df[c])
    ]
    endmember_feature_cols: list[str] = []
    if endmember_matminer_df is not None:
        endmember_feature_cols = [
            c
            for c in endmember_matminer_df.columns
            if c.startswith("endmember_")
            and c not in _ENDMEMBER_MATMINER_META
            and pd.api.types.is_numeric_dtype(endmember_matminer_df[c])
        ]

    thermo_feature_cols = [c for c in SOLUTION_THERMO_FEATURE_COLS if c in merged.columns]
    endmember_metadata_cols = [
        c
        for c in merged.columns
        if str(c).startswith("endmember_")
        and c not in endmember_feature_cols
        and pd.api.types.is_numeric_dtype(merged[c])
    ]
    feature_cols = (
        [c for c in ELEMENTAL_FEATURE_COLS if c in merged.columns]
        + thermo_feature_cols
        + matminer_feature_cols
        + endmember_metadata_cols
        + endmember_feature_cols
    )

    if drop_all_nan:
        feature_cols = [c for c in feature_cols if c in merged.columns and _has_observed_values(merged, c)]
    return feature_cols


def merge_endmember_phase_features(
    df: pd.DataFrame,
    *,
    ml_dir: PathLike,
    endmember_phases_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Left-merge ``dopant_endmember_phases_mp.csv`` (columns already ``endmember_*``)."""
    path = (
        Path(endmember_phases_path)
        if endmember_phases_path
        else Path(ml_dir) / "data" / "dopant_endmember_phases_mp.csv"
    )
    if not path.is_file():
        return df
    endmember = pd.read_csv(path)
    endmember["element"] = endmember["element"].astype(str).str.strip()
    drop_cols = [c for c in endmember.columns if c in df.columns and c != "element"]
    endmember = endmember.drop(columns=drop_cols, errors="ignore")
    return df.merge(endmember, on="element", how="left", validate="one_to_one")


def drop_absolute_radius_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove absolute radius columns; keep ``delta_r_vs_Li_pm`` only."""
    drop_cols = [
        "r_atomic_pm",
        "metallic_radius_pm",
        "mp_metallic_radius_pm",
        "mp_covalent_radius_pm",
        "mp_radius_pm",
        "mp_radius_source",
    ]
    present = [c for c in drop_cols if c in df.columns]
    if present:
        df = df.drop(columns=present)
    return df


def elemental_props_vs_host(
    elements: Sequence[str],
    host_element: str,
    elements_data: dict[str, Any],
    *,
    get_element_data: Optional[Any] = None,
) -> pd.DataFrame:
    """Build radius, valency, and chi differences relative to the host (Li)."""
    host_el = str(host_element).strip()
    if get_element_data is not None:
        host = get_element_data(host_el)
        lookup = get_element_data
    else:
        host = elements_data[host_el]
        lookup = lambda sym: elements_data[str(sym).strip()]

    rows: list[dict[str, Any]] = []
    for el in elements:
        sym = str(el).strip()
        if not sym or sym == host_el:
            continue
        d = lookup(sym)
        rows.append(
            {
                "element": sym,
                "delta_r_vs_Li_pm": d.r_atomic - host.r_atomic,
                "valency": d.valency,
                "delta_valency_vs_Li": d.valency - host.valency,
                "chi": d.chi,
                "delta_chi_vs_Li": d.chi - host.chi,
            }
        )
    return pd.DataFrame(rows)


def save_datasheet(
    df: pd.DataFrame,
    out_dir: PathLike,
    *,
    basename: str = "datasheet",
) -> Path:
    """Write ``datasheet.csv`` and remove legacy train/test exports."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{basename}.csv"
    df.to_csv(path, index=False)

    for legacy in (out_dir / f"{basename}_train.csv", out_dir / f"{basename}_test.csv"):
        if legacy.is_file():
            legacy.unlink()

    return path
