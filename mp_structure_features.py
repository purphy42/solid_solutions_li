"""Fetch MP crystal structures as ASE Atoms and featurize with matminer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence, Union

import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from mp_stable_phases import manual_radii_for_element

try:
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install matminer: pip install matminer") from exc

if TYPE_CHECKING:
    from ase import Atoms

PathLike = Union[str, Path]


@dataclass(frozen=True)
class PhaseStructure:
    element: str
    material_id: str
    role: str
    structure: Structure
    atoms: Atoms


def fetch_ase_structures_for_phases(
    phases_df: pd.DataFrame,
    *,
    api_key: str,
) -> list[PhaseStructure]:
    """Download pymatgen/ASE structures for each ``material_id`` in ``phases_df``."""
    rows = phases_df.dropna(subset=["material_id"]).copy()
    rows["material_id"] = rows["material_id"].astype(str).str.strip()
    rows["element"] = rows["element"].astype(str).str.strip()
    role_col = "role" if "role" in rows.columns else None

    phases: list[PhaseStructure] = []
    with MPRester(api_key=api_key) as mpr:
        for _, row in rows.iterrows():
            mp_id = row["material_id"]
            structure = mpr.get_structure_by_material_id(mp_id)
            atoms = AseAtomsAdaptor.get_atoms(structure)
            phases.append(
                PhaseStructure(
                    element=row["element"],
                    material_id=mp_id,
                    role=str(row[role_col]) if role_col else "dopant",
                    structure=structure,
                    atoms=atoms,
                )
            )
    return phases


def add_manual_metallic_radii(
    df: pd.DataFrame,
    *,
    manual_radii: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Append ``metallic_radius_pm`` from the flat ``MANUAL_RADII`` table."""
    radii = df["element"].map(
        lambda el: manual_radii_for_element(el, manual_radii=manual_radii)[
            "metallic_radius_pm"
        ]
    )
    out = df.copy()
    out["metallic_radius_pm"] = radii
    return out


def featurize_with_matminer(
    phases: Sequence[PhaseStructure],
    *,
    manual_radii: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Extract composition and structure features with matminer."""
    if not phases:
        return pd.DataFrame(columns=["element", "material_id", "role"])

    df = pd.DataFrame(
        {
            "element": [p.element for p in phases],
            "material_id": [p.material_id for p in phases],
            "role": [p.role for p in phases],
            "structure": [p.structure for p in phases],
            "composition": [p.structure.composition for p in phases],
        }
    )
    meta = {"element", "material_id", "role", "structure", "composition"}

    df = ElementProperty.from_preset("magpie").featurize_dataframe(
        df, col_id="composition", ignore_errors=True
    )
    df = DensityFeatures().featurize_dataframe(
        df, col_id="structure", ignore_errors=True
    )
    df = GlobalSymmetryFeatures().featurize_dataframe(
        df, col_id="structure", ignore_errors=True
    )

    feature_cols = [c for c in df.columns if c not in meta]
    out = df[["element", "material_id", "role", *feature_cols]]
    if manual_radii is not None:
        out = add_manual_metallic_radii(out, manual_radii=manual_radii)
    return out


def fetch_and_featurize_phases(
    phases_df: pd.DataFrame,
    *,
    api_key: str,
    manual_radii: Optional[dict[str, float]] = None,
) -> tuple[list[PhaseStructure], pd.DataFrame]:
    """Fetch ASE structures from MP and return matminer feature table."""
    phases = fetch_ase_structures_for_phases(phases_df, api_key=api_key)
    features = featurize_with_matminer(phases, manual_radii=manual_radii)
    return phases, features


def save_structure_features(
    features_df: pd.DataFrame,
    path: PathLike,
    *,
    manual_radii: Optional[dict[str, float]] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = features_df
    if manual_radii is not None and "metallic_radius_pm" not in df.columns:
        df = add_manual_metallic_radii(df, manual_radii=manual_radii)
    if manual_radii is None and "metallic_radius_pm" in df.columns:
        df = df.drop(columns=["metallic_radius_pm"])
    df.to_csv(path, index=False)
    return path


def prefix_feature_columns(
    df: pd.DataFrame,
    prefix: str,
    *,
    id_cols: tuple[str, ...] = ("element", "material_id", "role"),
) -> pd.DataFrame:
    """Prefix matminer / structure feature columns (e.g. ``endmember_``)."""
    out = df.copy()
    rename: dict[str, str] = {}
    for col in out.columns:
        if col in id_cols:
            if col == "material_id":
                rename[col] = f"{prefix}material_id"
            continue
        if str(col).startswith(prefix):
            continue
        rename[col] = f"{prefix}{col}"
    return out.rename(columns=rename)


def featurize_phases_with_prefix(
    phases_df: pd.DataFrame,
    *,
    api_key: str,
    prefix: str,
    manual_radii: Optional[dict[str, float]] = None,
    extra_id_cols: tuple[str, ...] = (),
) -> tuple[list[PhaseStructure], pd.DataFrame]:
    """Fetch MP structures and return matminer features with a column prefix."""
    phases = fetch_ase_structures_for_phases(phases_df, api_key=api_key)
    features = featurize_with_matminer(phases, manual_radii=manual_radii)
    id_cols = ("element", "material_id", "role", *extra_id_cols)
    features = prefix_feature_columns(features, prefix, id_cols=id_cols)
    return phases, features
