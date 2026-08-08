"""Hume–Rothery elemental properties (radii, structure, valency, χ)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import pandas as pd

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ElementData:
    Z: int
    r_atomic: float  # pm (metallic radius from XRD where applicable)
    crystal: str
    valency: int
    chi: float  # Pauling electronegativity


ELEMENTS_DATA: dict[str, ElementData] = {
    "H": ElementData(1, 25, 'molecular', 1, 2.2),
    "Li": ElementData(3, 152, 'bcc', 1, 0.98),
    "Be": ElementData(4, 112, 'hcp', 2, 1.57),
    "B": ElementData(5, 85, 'beta_rhombohedral', 3, 2.04),
    "Na": ElementData(11, 186, 'bcc', 1, 0.93),
    "Mg": ElementData(12, 160, 'hcp', 2, 1.31),
    "Al": ElementData(13, 143, 'fcc', 3, 1.61),
    "Si": ElementData(14, 111, 'diamond', 4, 1.9),
    "K": ElementData(19, 227, 'bcc', 1, 0.82),
    "Ca": ElementData(20, 197, 'fcc', 2, 1.0),
    "Sc": ElementData(21, 162, 'hcp', 3, 1.36),
    "Ti": ElementData(22, 147, 'hcp', 4, 1.54),
    "V": ElementData(23, 134, 'bcc', 5, 1.63),
    "Cr": ElementData(24, 128, 'bcc', 3, 1.66),
    "Mn": ElementData(25, 127, 'complex', 2, 1.55),
    "Fe": ElementData(26, 126, 'bcc', 2, 1.83),
    "Co": ElementData(27, 125, 'hcp', 2, 1.88),
    "Ni": ElementData(28, 124, 'fcc', 2, 1.91),
    "Cu": ElementData(29, 128, 'fcc', 1, 1.9),
    "Zn": ElementData(30, 134, 'hcp', 2, 1.65),
    "Ga": ElementData(31, 135, 'ortho', 3, 1.81),
    "Ge": ElementData(32, 122, 'diamond', 4, 2.01),
    "As": ElementData(33, 119, 'rhombo', 5, 2.18),
    "Rb": ElementData(37, 248, 'bcc', 1, 0.82),
    "Sr": ElementData(38, 215, 'fcc', 2, 0.95),
    "Y": ElementData(39, 180, 'hcp', 3, 1.22),
    "Zr": ElementData(40, 160, 'hcp', 4, 1.33),
    "Nb": ElementData(41, 146, 'bcc', 5, 1.6),
    "Mo": ElementData(42, 139, 'bcc', 6, 2.16),
    "Tc": ElementData(43, 136, 'hcp', 7, 1.9),
    "Ru": ElementData(44, 134, 'hcp', 3, 2.2),
    "Rh": ElementData(45, 134, 'fcc', 3, 2.28),
    "Pd": ElementData(46, 137, 'fcc', 2, 2.2),
    "Ag": ElementData(47, 144, 'fcc', 1, 1.93),
    "Cd": ElementData(48, 151, 'hcp', 2, 1.69),
    "In": ElementData(49, 167, 'tetra', 3, 1.78),
    "Sn": ElementData(50, 140, 'tetra', 4, 1.96),
    "Sb": ElementData(51, 140, 'rhombo', 5, 2.05),
    "Cs": ElementData(55, 265, 'bcc', 1, 0.79),
    "Ba": ElementData(56, 222, 'bcc', 2, 0.89),
    "Hf": ElementData(72, 159, 'hcp', 4, 1.3),
    "Ta": ElementData(73, 146, 'bcc', 5, 1.5),
    "W": ElementData(74, 139, 'bcc', 6, 2.36),
    "Re": ElementData(75, 137, 'hcp', 7, 1.9),
    "Os": ElementData(76, 135, 'hcp', 4, 2.2),
    "Ir": ElementData(77, 136, 'fcc', 3, 2.2),
    "Pt": ElementData(78, 139, 'fcc', 2, 2.28),
    "Au": ElementData(79, 144, 'fcc', 1, 2.54),
    "Hg": ElementData(80, 151, 'rhombo', 2, 2.0),
    "Bi": ElementData(83, 155, 'rhombo', 5, 2.02),
    "La": ElementData(57, 187, 'dhcp', 3, 1.1),
    "Ce": ElementData(58, 182, 'fcc', 3, 1.12),
    "Pr": ElementData(59, 182, 'hcp', 3, 1.13),
    "Nd": ElementData(60, 181, 'hcp', 3, 1.14),
    "Sm": ElementData(62, 180, 'rhombo', 3, 1.17),
    "Eu": ElementData(63, 208, 'bcc', 2, 0.63),
    "Gd": ElementData(64, 180, 'hcp', 3, 1.2),
    "Tb": ElementData(65, 177, 'hcp', 3, 1.22),
    "Dy": ElementData(66, 178, 'hcp', 3, 1.22),
    "Ho": ElementData(67, 176, 'hcp', 3, 1.23),
    "Er": ElementData(68, 176, 'hcp', 3, 1.24),
    "Tm": ElementData(69, 176, 'hcp', 3, 1.25),
    "Yb": ElementData(70, 194, 'fcc', 2, 1.1),
    "Lu": ElementData(71, 174, 'hcp', 3, 1.27),
}


def hume_rothery_dataframe(data: dict[str, ElementData] | None = None) -> pd.DataFrame:
    """Return a dataframe of elemental Hume–Rothery properties."""
    table = data or ELEMENTS_DATA
    rows = []
    for element, props in table.items():
        rows.append(
            {
                "element": element,
                "Z": props.Z,
                "r_atomic_pm": props.r_atomic,
                "crystal": props.crystal,
                "valency": props.valency,
                "chi": props.chi,
            }
        )
    return pd.DataFrame(rows).sort_values("Z").reset_index(drop=True)


def save_hume_rothery_csv(
    path: PathLike | None = None,
    *,
    data: dict[str, ElementData] | None = None,
) -> Path:
    """Write ``hume_rothery_data.csv`` (default: ``data/hume_rothery_data.csv``)."""
    out = Path(path) if path is not None else Path(__file__).resolve().parent / "data" / "hume_rothery_data.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = hume_rothery_dataframe(data)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    written = save_hume_rothery_csv()
    print(f"Wrote {written} ({len(hume_rothery_dataframe())} elements)")

