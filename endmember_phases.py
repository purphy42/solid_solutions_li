"""Li–X supercell endmember phases for ML features (endmember_* columns)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Composition

from mp_stable_phases import _row_from_material_id_override, mp_api_key

PathLike = Union[str, Path]

# Li–X supercell endmember references: {species: (mp_id_with_suffix, supercell_fraction)}.
OPT_DOPANT_ENDMEMBERS: dict[str, dict[str, tuple[str, float]]] = {
    "Be": {"Be": ("mp-87-GGA", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "B": {"LiB": ("mp-1001835-r2SCAN", 0.015625), "Li": ("mp-135-r2SCAN", 0.984375)},
    "Na": {"Li": ("mp-135-GGA", 0.9921875), "Na": ("mp-127-r2SCAN", 0.0078125)},
    "Mg": {"Mg": ("mp-153-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Al": {"Li9Al4": ("mp-568404-r2SCAN", 0.025390625), "Li": ("mp-135-r2SCAN", 0.974609375)},
    "Si": {"Si": ("mp-149-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "K": {"Li": ("mp-135-GGA", 0.9921875), "K": ("mp-58-r2SCAN", 0.0078125)},
    "Ca": {"Li": ("mp-135-GGA", 0.9921875), "Li2Ca": ("mp-570466-r2SCAN", 0.0078125)},
    "Sc": {"Sc": ("mp-67-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Ti": {"Ti": ("mp-72-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "V": {"V": ("mp-146-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Cr": {"Cr": ("mp-90-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Mn": {"Mn": ("mp-35-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Fe": {"Fe": ("mp-13-GGA", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Co": {"Co": ("mp-102-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Ni": {"Ni": ("mp-23-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Cu": {"Cu": ("mp-30-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Zn": {"Li": ("mp-135-GGA", 0.9921875), "LiZn": ("mp-1934-r2SCAN", 0.0078125)},
    "Ga": {"Li": ("mp-135-GGA", 0.9921875), "Li2Ga": ("mp-29210-r2SCAN", 0.0078125)},
    "Ge": {"Li": ("mp-135-GGA", 0.9921875), "Li15Ge4": ("mp-1777-r2SCAN", 0.0078125)},
    "Se": {"Li2Se": ("mp-2286-r2SCAN", 0.0234375), "Li": ("mp-135-r2SCAN", 0.9765625)},
    "Y": {"Y": ("mp-112-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Zr": {"Zr": ("mp-131-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Nb": {"Nb": ("mp-75-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Mo": {"Mo": ("mp-129-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Ru": {"Ru": ("mp-33-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Rh": {"LiRh": ("mp-600561-r2SCAN", 0.015625), "Li": ("mp-135-r2SCAN", 0.984375)},
    "Pd": {"Li15Pd4": ("mp-1197547-r2SCAN", 0.037109375), "Li": ("mp-135-r2SCAN", 0.962890625)},
    "Ag": {"LiAg": ("mp-2426-r2SCAN", 0.015625), "Li": ("mp-135-r2SCAN", 0.984375)},
    "Cd": {"LiCd": ("mp-1437-r2SCAN", 0.015625), "Li": ("mp-135-r2SCAN", 0.984375)},
    "In": {"Li13In3": ("mp-510430-r2SCAN", 0.041666666666666664), "Li": ("mp-135-r2SCAN", 0.9583333333333334)},
    "Sn": {"Li": ("mp-135-GGA", 0.9921875), "Li17Sn4": ("mp-573471-r2SCAN", 0.0078125)},
    "Sb": {"Li3Sb": ("mp-7955-r2SCAN", 0.03125), "Li": ("mp-135-r2SCAN", 0.96875)},
    "La": {"La": ("mp-26-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Ce": {"Ce": ("mp-28-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Pr": {"Pr": ("mp-97-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Nd": {"Nd": ("mp-123-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Sm": {"Sm": ("mp-69-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Hf": {"Hf": ("mp-103-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Ta": {"Ta": ("mp-569794-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "W": {"W": ("mp-91-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Re": {"Re": ("mp-8-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Os": {"Os": ("mp-49-r2SCAN", 0.0078125), "Li": ("mp-135-r2SCAN", 0.9921875)},
    "Ir": {"LiIr": ("mp-279-r2SCAN", 0.015625), "Li": ("mp-135-r2SCAN", 0.984375)},
    "Pt": {"Li2Pt": ("mp-2170-r2SCAN", 0.0234375), "Li": ("mp-135-r2SCAN", 0.9765625)},
    "Au": {"Li15Au4": ("mp-567395-r2SCAN", 0.037109375), "Li": ("mp-135-r2SCAN", 0.962890625)},
    "Hg": {"Li": ("mp-135-GGA", 0.9921875), "Li3Hg": ("mp-1646-r2SCAN", 0.0078125)},
    "Tl": {"Li3Tl": ("mp-7396-r2SCAN", 0.03125), "Li": ("mp-135-r2SCAN", 0.96875)},
    "Pb": {"Li": ("mp-135-GGA", 0.9921875), "Li17Pb4": ("mp-574275-r2SCAN", 0.0078125)},
    "Bi": {"Li3Bi": ("mp-23222-r2SCAN", 0.03125), "Li": ("mp-135-r2SCAN", 0.96875)},
}

_ENDMEMBER_META_KEEP = {"element"}


def normalize_mp_material_id(material_id: str) -> str:
    """``mp-573471-GGA`` -> ``mp-573471`` for MP API lookups."""
    s = str(material_id).strip()
    match = re.match(r"^(mp-\d+)", s)
    return match.group(1) if match else s


def primary_endmember_species(
    compositions: dict[str, tuple[str, float]],
) -> tuple[str, str, float]:
    """
    Pick the non-Li endmember species for a dopant supercell reference.

    Prefers Li–X compounds over unary dopant when both are present (e.g. LiZn vs Zn).
    """
    non_li = {k: v for k, v in compositions.items() if str(k).strip() != "Li"}
    if not non_li:
        raise ValueError("No non-Li endmember species in composition map")

    if len(non_li) == 1:
        species = next(iter(non_li))
        mp_id, frac = non_li[species]
        return str(species), str(mp_id), float(frac)

    compounds: list[tuple[str, str, float, int]] = []
    unaries: list[tuple[str, str, float]] = []
    for species, (mp_id, frac) in non_li.items():
        n_elements = len(Composition(species).elements)
        if n_elements > 1:
            compounds.append((species, mp_id, float(frac), n_elements))
        else:
            unaries.append((species, mp_id, float(frac)))

    if compounds:
        species, mp_id, frac, _ = max(compounds, key=lambda row: (row[3], row[2]))
        return str(species), str(mp_id), float(frac)

    species, mp_id, frac = max(unaries, key=lambda row: row[2])
    return str(species), str(mp_id), float(frac)


def build_endmember_table(
    endmembers_by_dopant: Optional[dict[str, dict[str, tuple[str, float]]]] = None,
    *,
    elements: Optional[list[str]] = None,
) -> pd.DataFrame:
    """One row per dopant with primary endmember formula, MP id, and supercell fraction."""
    mapping = endmembers_by_dopant or OPT_DOPANT_ENDMEMBERS
    keys = sorted(elements) if elements else sorted(mapping.keys())
    rows: list[dict[str, Any]] = []
    for element in keys:
        compositions = mapping.get(element)
        if not compositions:
            rows.append(
                {
                    "element": element,
                    "endmember_formula": None,
                    "endmember_material_id_raw": None,
                    "endmember_material_id": None,
                    "endmember_fraction": None,
                }
            )
            continue
        formula, mp_raw, frac = primary_endmember_species(compositions)
        rows.append(
            {
                "element": element,
                "endmember_formula": formula,
                "endmember_material_id_raw": mp_raw,
                "endmember_material_id": normalize_mp_material_id(mp_raw),
                "endmember_fraction": frac,
            }
        )
    return pd.DataFrame(rows)


def prefix_endmember_columns(
    df: pd.DataFrame,
    *,
    keep: Optional[set[str]] = None,
) -> pd.DataFrame:
    """Add ``endmember_`` to metadata columns (skip ``element`` and already-prefixed)."""
    keep = keep or _ENDMEMBER_META_KEEP
    rename = {
        col: f"endmember_{col}"
        for col in df.columns
        if col not in keep and not str(col).startswith("endmember_")
    }
    return df.rename(columns=rename)


def fetch_endmember_mp_metadata(
    endmember_table: pd.DataFrame,
    *,
    api_key: Optional[str] = None,
    e_hull_max: float = 0.1,
) -> pd.DataFrame:
    """Attach MP summary fields for each ``endmember_material_id`` (prefixed)."""
    key = mp_api_key(api_key)
    base = endmember_table.copy()
    base["element"] = base["element"].astype(str).str.strip()

    meta_field_map = {
        "material_id": "endmember_mp_material_id",
        "formula_pretty": "endmember_formula_pretty",
        "energy_per_atom": "endmember_energy_per_atom",
        "energy_above_hull": "endmember_energy_above_hull",
        "theoretical": "endmember_theoretical",
        "is_stable": "endmember_is_stable",
        "crystal_system": "endmember_crystal_system",
        "spacegroup_symbol": "endmember_spacegroup_symbol",
        "spacegroup_number": "endmember_spacegroup_number",
        "density": "endmember_density",
        "volume": "endmember_volume",
        "nsites": "endmember_nsites",
        "selection_rule": "endmember_selection_rule",
        "e_hull_max_eV_per_atom": "endmember_e_hull_max_eV_per_atom",
        "is_theoretical_fallback": "endmember_is_theoretical_fallback",
        "n_candidates": "endmember_n_candidates",
        "n_experimental": "endmember_n_experimental",
        "n_near_hull": "endmember_n_near_hull",
        "material_id_forced": "endmember_material_id_forced",
        "error": "endmember_error",
    }

    rows: list[dict[str, Any]] = []
    with MPRester(api_key=key) as mpr:
        for _, row in base.iterrows():
            out_row = row.to_dict()
            mp_id = row.get("endmember_material_id")
            if pd.isna(mp_id) or not str(mp_id).strip():
                out_row["endmember_error"] = "missing_material_id"
                rows.append(out_row)
                continue
            meta = _row_from_material_id_override(
                str(row["element"]),
                mpr,
                str(mp_id),
                role="endmember",
                e_hull_max=e_hull_max,
                material_id_forced=True,
                selection_rule="endmember_reference",
            )
            for src, dst in meta_field_map.items():
                if src in meta:
                    out_row[dst] = meta[src]
            rows.append(out_row)

    return pd.DataFrame(rows)


def default_endmember_table_path(ml_dir: PathLike) -> Path:
    return Path(ml_dir) / "data" / "dopant_endmember_phases_mp.csv"


def default_endmember_matminer_path(ml_dir: PathLike) -> Path:
    return Path(ml_dir) / "data" / "structure_endmember_matminer_features.csv"


def load_endmember_table(path: PathLike) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["element"] = df["element"].astype(str).str.strip()
    return df


def save_endmember_table(df: pd.DataFrame, path: PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
