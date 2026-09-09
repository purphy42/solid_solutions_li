"""Thermodynamic features for dopant solution-energy ML (mu_dopant, MP hull metadata)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from pymatgen.core import Composition

PathLike = Union[str, Path]

# Used by feature_importance.ipynb and model_train.ipynb (via importance CSVs).
SOLUTION_THERMO_FEATURE_COLS = [
    "mu_dopant_eV",
    "mu_dopant_delta_vs_elemental_eV",
    "mp_energy_above_hull",
    "mp_n_experimental",
    "mp_n_candidates",
    "mp_n_near_hull",
    "mp_is_stable",
    "mp_theoretical",
    "mp_material_id_forced",
    "mp_is_theoretical_fallback",
    "mp_hull_unstable",
]

_BOOL_INT_COLS = [
    "mp_is_stable",
    "mp_theoretical",
    "mp_material_id_forced",
    "mp_is_theoretical_fallback",
    "mp_hull_unstable",
]


def parse_formula(formula: str) -> dict[str, float]:
    return Composition(formula).as_dict()


def calculate_chemical_potentials(
    dopant_data: dict[str, dict[str, tuple[str, float]]],
    en_endmembers: dict[str, float],
) -> dict[str, dict[str, float]]:
    """
    Chemical potentials from Li–X supercell end-member fractions and DFT energies.

    ``dopant_data`` maps dopant symbol to {species: (mp_id, fraction)}.
    ``en_endmembers`` maps species label (e.g. ``'Li'``, ``'Li3Tl'``) to eV/atom.
    """
    chemical_potentials: dict[str, dict[str, float]] = {}
    for _dopant, compositions in dopant_data.items():
        mu: dict[str, float] = {}
        elemental: list[tuple[str, dict[str, float]]] = []
        compounds: list[tuple[str, dict[str, float]]] = []

        for species, (_mp_id, _fraction) in compositions.items():
            comp_dict = parse_formula(species)
            if len(comp_dict) == 1:
                elemental.append((species, comp_dict))
            else:
                compounds.append((species, comp_dict))

        for species, comp_dict in elemental:
            element = list(comp_dict.keys())[0]
            mu[element] = float(en_endmembers[species])

        for species, comp_dict in compounds:
            e_per_atom = float(en_endmembers[species])
            total_atoms = sum(comp_dict.values())
            e_total = e_per_atom * total_atoms
            unknown = [el for el in comp_dict if el not in mu]
            known = [el for el in comp_dict if el in mu]
            if len(unknown) == 1:
                unknown_el = unknown[0]
                known_energy = sum(comp_dict[el] * mu[el] for el in known)
                mu[unknown_el] = (e_total - known_energy) / comp_dict[unknown_el]

        chemical_potentials[_dopant] = mu
    return chemical_potentials


def chemical_potentials_path(ml_dir: PathLike) -> Path:
    return Path(ml_dir) / "data" / "dopant_chemical_potentials_opt.csv"


def load_chemical_potentials(path: PathLike) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Dopant" in df.columns:
        df = df.rename(columns={"Dopant": "element"})
    if "chem_pot_eV" in df.columns:
        df = df.rename(columns={"chem_pot_eV": "mu_dopant_eV"})
    if "delta_eV" in df.columns:
        df = df.rename(columns={"delta_eV": "mu_dopant_delta_vs_elemental_eV"})
    df["element"] = df["element"].astype(str).str.strip()
    keep = ["element", "mu_dopant_eV", "mu_dopant_delta_vs_elemental_eV"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates("element")


def _as_bool_int(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    return series.fillna(False).astype(bool).astype(int)


def add_solution_thermo_features(
    df: pd.DataFrame,
    *,
    ml_dir: PathLike,
    chem_pot_path: Optional[PathLike] = None,
) -> pd.DataFrame:
    """Merge mu_dopant and encode MP phase-selection metadata as numeric ML features."""
    out = df.copy()
    path = Path(chem_pot_path) if chem_pot_path else chemical_potentials_path(ml_dir)
    chem_cols = ["mu_dopant_eV", "mu_dopant_delta_vs_elemental_eV"]
    if path.is_file():
        chem = load_chemical_potentials(path)
        out = out.drop(columns=[c for c in chem_cols if c in out.columns], errors="ignore")
        out = out.merge(chem, on="element", how="left", validate="one_to_one")
    else:
        for col in chem_cols:
            if col not in out.columns:
                out[col] = float("nan")

    if "mp_energy_above_hull" in out.columns:
        hull = pd.to_numeric(out["mp_energy_above_hull"], errors="coerce")
        out["mp_hull_unstable"] = (hull > 1e-12).fillna(False).astype(int)
    else:
        out["mp_hull_unstable"] = 0

    for col in ["mp_is_stable", "mp_theoretical", "mp_material_id_forced", "mp_is_theoretical_fallback"]:
        if col in out.columns:
            out[col] = _as_bool_int(out[col])
        elif col.startswith("mp_"):
            out[col] = 0

    return out


def available_thermo_features(df: pd.DataFrame) -> list[str]:
    return [c for c in SOLUTION_THERMO_FEATURE_COLS if c in df.columns]
