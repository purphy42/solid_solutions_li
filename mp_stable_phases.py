"""Fetch room-temperature-like stable elemental phases from the Materials Project API."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import pandas as pd

try:
    from mp_api.client import MPRester
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Install mp-api: pip install mp-api"
    ) from exc

SelectionMode = Literal["min_nsites", "min_energy_above_hull"]

# Always applied (independent of automatic hull/min-nsites selection)
DEFAULT_FORCED_MP_IDS: dict[str, str] = {
    "Hg": "mp-10861",
    "K": "mp-58",
    "Pr": "mp-38",
}

# Optional manual pins (only when use_rt_mp_catalog=True)
ELEMENT_RT_MP_IDS: dict[str, str] = {
    "Li": "mp-135",
    "Na": "mp-127",
    "Ag": "mp-124",
    "Se": "mp-14",
    "Ta": "mp-50",
}


def _merged_forced_mp_ids(forced_mp_ids: Optional[dict[str, str]]) -> dict[str, str]:
    merged = dict(DEFAULT_FORCED_MP_IDS)
    if forced_mp_ids:
        merged.update({k.strip(): v.strip() for k, v in forced_mp_ids.items()})
    return merged


# Manual radii (pm) — same values as ``MANUAL_RADII`` in create_data.ipynb
METALLIC_RADII_PM: dict[str, float] = {
    # Alkali metals
    "Li": 152,
    "Na": 186,
    "K": 227,
    "Rb": 248,
    "Cs": 265,
    # Alkaline earth metals
    "Be": 112,
    "Mg": 160,
    "Ca": 197,
    "Sr": 215,
    "Ba": 222,
    # Transition metals — period 4
    "Sc": 162,
    "Ti": 147,
    "V": 134,
    "Cr": 128,
    "Mn": 127,
    "Fe": 126,
    "Co": 125,
    "Ni": 124,
    "Cu": 128,
    "Zn": 134,
    # Period 5
    "Y": 180,
    "Zr": 160,
    "Nb": 146,
    "Mo": 139,
    "Tc": 136,
    "Ru": 134,
    "Rh": 134,
    "Pd": 137,
    "Ag": 144,
    "Cd": 151,
    # Period 6
    "Lu": 174,
    "Hf": 159,
    "Ta": 146,
    "W": 139,
    "Re": 137,
    "Os": 135,
    "Ir": 136,
    "Pt": 139,
    "Au": 144,
    "Hg": 151,
    # p-block metals
    "Al": 143,
    "Ga": 135,
    "In": 167,
    "Tl": 170,
    "Sn": 140,
    "Pb": 175,
    "Bi": 155,
    # Lanthanides
    "La": 187,
    "Ce": 182,
    "Pr": 182,
    "Nd": 181,
    "Sm": 180,
    "Eu": 208,
    "Gd": 180,
    "Tb": 177,
    "Dy": 178,
    "Ho": 176,
    "Er": 176,
    "Tm": 176,
    "Yb": 194,
    # Actinides
    "Ac": 187,
    "Th": 179,
    "Pa": 163,
    "U": 156,
    "Np": 155,
    "Pu": 159,
    "Am": 173,
}

COVALENT_RADII_PM: dict[str, float] = {
    "He": 32,
    "Ne": 69,
    "Ar": 106,
    "Kr": 116,
    "Xe": 131,
    "Rn": 147,
    "F": 71,
    "Cl": 99,
    "Br": 114,
    "I": 133,
    "At": 150,
    "O": 66,
    "S": 105,
    "Se": 116,
    "Te": 140,
    "Po": 168,
    "N": 71,
    "P": 107,
    "As": 119,
    "Sb": 140,
    "C": 76,
    "Si": 111,
    "Ge": 122,
    "B": 85,
    "H": 31,
}

MANUAL_RADII: dict[str, float] = {**METALLIC_RADII_PM, **COVALENT_RADII_PM}

# Classify flat ``manual_radii`` entries into metallic vs covalent columns.
METALLIC_ELEMENT_SYMBOLS = frozenset(METALLIC_RADII_PM)
COVALENT_ELEMENT_SYMBOLS = frozenset(COVALENT_RADII_PM)


def manual_radii_for_element(
    symbol: str,
    *,
    manual_radii: Optional[dict[str, float]] = None,
) -> dict[str, Optional[float]]:
    """Look up radii (pm) from a flat ``manual_radii`` dict (default: ``MANUAL_RADII``)."""
    radii = manual_radii if manual_radii is not None else MANUAL_RADII
    sym = str(symbol).strip()
    r = radii.get(sym)
    r_met = r if sym in METALLIC_ELEMENT_SYMBOLS else None
    r_cov = r if sym in COVALENT_ELEMENT_SYMBOLS else None
    if r_met is not None:
        radius_source = "metallic"
    elif r_cov is not None:
        radius_source = "covalent"
    else:
        radius_source = None
    return {
        "metallic_radius_pm": r_met,
        "covalent_radius_pm": r_cov,
        "radius_pm": r,
        "radius_source": radius_source,
    }


def add_manual_radii_columns(
    df: pd.DataFrame,
    *,
    manual_radii: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Append radius columns from a flat ``manual_radii`` dict (no API calls)."""
    radii = df["element"].map(
        lambda el: manual_radii_for_element(el, manual_radii=manual_radii)
    )
    radii_df = pd.DataFrame(radii.tolist(), index=df.index)
    return pd.concat([df, radii_df], axis=1)


def check_manual_radii_coverage(
    elements: Sequence[str],
    *,
    manual_radii: Optional[dict[str, float]] = None,
) -> pd.DataFrame:
    """Report radius lookup for each element using ``manual_radii``."""
    rows: list[dict[str, Any]] = []
    for el in elements:
        sym = str(el).strip()
        if not sym:
            continue
        fields = manual_radii_for_element(sym, manual_radii=manual_radii)
        rows.append(
            {
                "element": sym,
                "metallic_radius_pm": fields["metallic_radius_pm"],
                "covalent_radius_pm": fields["covalent_radius_pm"],
                "radius_pm": fields["radius_pm"],
                "radius_source": fields["radius_source"],
                "missing": fields["radius_pm"] is None,
            }
        )
    return pd.DataFrame(rows)

SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "energy_per_atom",
    "energy_above_hull",
    "theoretical",
    "symmetry",
    "density",
    "volume",
    "nsites",
    "is_stable",
    "deprecated",
]


def _symmetry_fields(symmetry: Any) -> tuple[Optional[str], Optional[str], Optional[int]]:
    """Extract crystal system, space-group symbol, and number from MP ``symmetry``."""
    if symmetry is None:
        return None, None, None
    if isinstance(symmetry, dict):
        crystal_system = symmetry.get("crystal_system")
        symbol = symmetry.get("symbol")
        number = symmetry.get("number")
    else:
        crystal_system = getattr(symmetry, "crystal_system", None)
        symbol = getattr(symmetry, "symbol", None)
        number = getattr(symmetry, "number", None)
    if number is not None:
        number = int(number)
    return crystal_system, symbol, number


def mp_api_key(explicit: Optional[str] = None) -> str:
    """Resolve Materials Project API key from argument or MP_API_KEY env var."""
    key = explicit or os.environ.get("MP_API_KEY")
    if not key:
        raise EnvironmentError(
            "Set MP_API_KEY (https://materialsproject.org/api) or pass api_key=..."
        )
    return key


def _is_deprecated(doc: Any) -> bool:
    return bool(getattr(doc, "deprecated", False))


def _e_above_hull(doc: Any) -> float:
    val = getattr(doc, "energy_above_hull", None)
    if val is None:
        return float("nan")
    return float(val)


def _sort_key_min_nsites(
    doc: Any,
    *,
    prefer_is_stable: bool = False,
) -> tuple[float, float, float, float]:
    """Smallest unit cell first, then lower hull; ``is_stable`` only breaks ties."""
    nsites = getattr(doc, "nsites", None)
    ns = float(nsites) if nsites is not None else float("inf")
    e_hull = _e_above_hull(doc)
    if e_hull != e_hull:  # NaN
        e_hull = float("inf")
    e_atom = float(getattr(doc, "energy_per_atom", float("inf")))
    stable_rank = (
        0.0
        if (not prefer_is_stable or getattr(doc, "is_stable", False) is True)
        else 1.0
    )
    return (ns, e_hull, stable_rank, e_atom)


def _sort_key_min_hull(doc: Any) -> tuple[float, float, float]:
    e_hull = _e_above_hull(doc)
    if e_hull != e_hull:
        e_hull = float("inf")
    nsites = getattr(doc, "nsites", None)
    ns = float(nsites) if nsites is not None else float("inf")
    e_atom = float(getattr(doc, "energy_per_atom", float("inf")))
    return (e_hull, ns, e_atom)


def _pick_from_pool(
    pool: Sequence[Any],
    *,
    selection_mode: SelectionMode,
    prefer_is_stable: bool = False,
) -> Any:
    if selection_mode == "min_nsites":
        return min(
            pool,
            key=lambda d: _sort_key_min_nsites(d, prefer_is_stable=prefer_is_stable),
        )
    return min(pool, key=_sort_key_min_hull)


def _near_hull_pool(
    pool: Sequence[Any],
    e_hull_max: float,
    *,
    strict_less_than: bool = True,
) -> list[Any]:
    """Phases within the hull window (default: energy_above_hull < e_hull_max)."""
    out: list[Any] = []
    for d in pool:
        e = _e_above_hull(d)
        if e != e:  # NaN
            continue
        if strict_less_than:
            if e < e_hull_max - 1e-15:
                out.append(d)
        else:
            if e <= e_hull_max + 1e-12:
                out.append(d)
    return out


def _resolve_pinned_mp_id(
    symbol: str,
    *,
    material_id_override: Optional[str],
    forced_mp_ids: Optional[dict[str, str]],
    rt_mp_id_overrides: Optional[dict[str, str]],
    use_rt_mp_catalog: bool,
) -> tuple[Optional[str], bool, Optional[str]]:
    """
    Return (mp_id, material_id_forced, selection_rule) if the id is pinned; else (None, False, None).
    """
    if material_id_override:
        return str(material_id_override).strip(), True, "material_id_override"
    forced = _merged_forced_mp_ids(forced_mp_ids)
    if symbol in forced:
        return forced[symbol], True, "forced_mp_id"
    if use_rt_mp_catalog:
        merged = dict(ELEMENT_RT_MP_IDS)
        if rt_mp_id_overrides:
            merged.update({k.strip(): v.strip() for k, v in rt_mp_id_overrides.items()})
        if symbol in merged:
            return merged[symbol], True, "rt_mp_catalog"
    return None, False, None


def _select_stable_element_doc(
    docs: Sequence[Any],
    *,
    e_hull_max: float = 0.1,
    strict_less_than: bool = True,
    require_experimental: bool = True,
    prefer_is_stable: bool = False,
    allow_theoretical_fallback: bool = False,
    selection_mode: SelectionMode = "min_nsites",
) -> tuple[Optional[Any], str, bool, int, int, int]:
    """
    Choose one MP summary document for a unary element.

    Default rule (``min_nsites``):
      1. Drop deprecated entries.
      2. Keep experimental structures (``theoretical is False``) unless fallback is on.
      3. Keep phases with ``energy_above_hull < e_hull_max`` (default 0.1 eV/atom).
      4. Pick the **smallest ``nsites``** (then lowest ``energy_above_hull``).

    No space-group shortcuts — Ag mp-124, Se mp-14, Ta mp-50 win if they are in the hull window.
    """
    active = [d for d in docs if not _is_deprecated(d)]
    n_candidates = len(active)
    exp_docs = [d for d in active if d.theoretical is False]
    n_experimental = len(exp_docs)

    if require_experimental:
        pool = exp_docs
        is_theo_fallback = False
        if not pool and allow_theoretical_fallback:
            pool = list(active)
            is_theo_fallback = True
    else:
        pool = list(active)
        is_theo_fallback = False

    if not pool:
        return None, "no_pool", is_theo_fallback, n_candidates, n_experimental, 0

    near = _near_hull_pool(pool, e_hull_max, strict_less_than=strict_less_than)
    n_near_hull = len(near)

    if near:
        doc = _pick_from_pool(
            near,
            selection_mode=selection_mode,
            prefer_is_stable=prefer_is_stable,
        )
        rule = "hull_window_min_nsites"
        if strict_less_than:
            rule += "_lt"
        else:
            rule += "_le"
        if selection_mode == "min_energy_above_hull":
            rule = "hull_window_min_e_hull"
        return doc, rule, is_theo_fallback, n_candidates, n_experimental, n_near_hull

    if exp_docs and require_experimental and not is_theo_fallback:
        doc = _pick_from_pool(
            exp_docs,
            selection_mode=selection_mode,
            prefer_is_stable=prefer_is_stable,
        )
        return (
            doc,
            "fallback_experimental_min_nsites",
            is_theo_fallback,
            n_candidates,
            n_experimental,
            n_near_hull,
        )

    if is_theo_fallback:
        near_all = _near_hull_pool(active, e_hull_max, strict_less_than=strict_less_than)
        if near_all:
            doc = _pick_from_pool(
                near_all,
                selection_mode=selection_mode,
                prefer_is_stable=prefer_is_stable,
            )
            return (
                doc,
                "fallback_theoretical_hull_window",
                is_theo_fallback,
                n_candidates,
                n_experimental,
                n_near_hull,
            )
        doc = _pick_from_pool(
            active,
            selection_mode=selection_mode,
            prefer_is_stable=prefer_is_stable,
        )
        return (
            doc,
            "fallback_theoretical_min_nsites",
            is_theo_fallback,
            n_candidates,
            n_experimental,
            n_near_hull,
        )

    return None, "no_match", is_theo_fallback, n_candidates, n_experimental, n_near_hull


def list_element_phase_candidates(
    element: str,
    mpr: MPRester,
    *,
    e_hull_max: float = 0.1,
    strict_less_than: bool = True,
    require_experimental: bool = True,
) -> pd.DataFrame:
    """All non-deprecated unary MP entries in the hull window (for debugging selection)."""
    symbol = str(element).strip()
    docs = mpr.materials.summary.search(
        elements=[symbol],
        num_elements=1,
        fields=SUMMARY_FIELDS,
    )
    active = [d for d in docs if not _is_deprecated(d)]
    if require_experimental:
        pool = [d for d in active if d.theoretical is False]
    else:
        pool = list(active)
    near = _near_hull_pool(pool, e_hull_max, strict_less_than=strict_less_than)
    rows = []
    for d in near:
        cs, sg, sg_num = _symmetry_fields(getattr(d, "symmetry", None))
        rows.append(
            {
                "element": symbol,
                "material_id": str(d.material_id),
                "nsites": d.nsites,
                "energy_above_hull": _e_above_hull(d),
                "is_stable": getattr(d, "is_stable", None),
                "theoretical": d.theoretical,
                "spacegroup_symbol": sg,
                "spacegroup_number": sg_num,
                "crystal_system": cs,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "element",
                "material_id",
                "nsites",
                "energy_above_hull",
                "is_stable",
                "theoretical",
                "spacegroup_symbol",
                "spacegroup_number",
                "crystal_system",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["nsites", "energy_above_hull"], ascending=[True, True])
        .reset_index(drop=True)
    )


def _empty_row(symbol: str, error: str, *, role: str = "dopant", **counts: int) -> dict[str, Any]:
    return {
        "element": symbol,
        "role": role,
        "material_id": None,
        "formula_pretty": None,
        "energy_per_atom": None,
        "energy_above_hull": None,
        "theoretical": None,
        "crystal_system": None,
        "spacegroup_symbol": None,
        "spacegroup_number": None,
        "is_stable": None,
        "density": None,
        "volume": None,
        "nsites": None,
        "selection_rule": None,
        "e_hull_max_eV_per_atom": None,
        "is_theoretical_fallback": False,
        "n_candidates": counts.get("n_candidates", 0),
        "n_experimental": counts.get("n_experimental", 0),
        "n_near_hull": counts.get("n_near_hull", 0),
        "material_id_forced": False,
        "error": error,
    }


def get_experimental_ground_state_element(
    element: str,
    mpr: MPRester,
    *,
    e_hull_max: float = 0.1,
    strict_less_than: bool = True,
    require_experimental: bool = True,
    prefer_is_stable: bool = False,
    allow_theoretical_fallback: bool = False,
    selection_mode: SelectionMode = "min_nsites",
    role: str = "dopant",
    material_id_override: Optional[str] = None,
    use_rt_mp_catalog: bool = False,
    rt_mp_id_overrides: Optional[dict[str, str]] = None,
    forced_mp_ids: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """
    Return a unary phase: experimental, ``energy_above_hull < e_hull_max`` (default 0.1 eV/atom),
    then minimum ``nsites``.

    Pinned ids (``material_id_forced=True``): ``forced_mp_ids`` (includes Hg/K/Pr defaults),
    ``material_id_override``, or ``use_rt_mp_catalog`` entries.
    """
    symbol = str(element).strip()
    if not symbol:
        raise ValueError("element must be a non-empty symbol")

    pin_id, is_forced, pin_rule = _resolve_pinned_mp_id(
        symbol,
        material_id_override=material_id_override,
        forced_mp_ids=forced_mp_ids,
        rt_mp_id_overrides=rt_mp_id_overrides,
        use_rt_mp_catalog=use_rt_mp_catalog,
    )
    if pin_id:
        return _row_from_material_id_override(
            symbol,
            mpr,
            pin_id,
            role=role,
            e_hull_max=e_hull_max,
            material_id_forced=is_forced,
            selection_rule=pin_rule or "material_id_override",
        )

    docs = mpr.materials.summary.search(
        elements=[symbol],
        num_elements=1,
        fields=SUMMARY_FIELDS,
    )
    if not docs:
        row = _empty_row(symbol, "no_entries", role=role)
        row["e_hull_max_eV_per_atom"] = e_hull_max
        return row

    doc, rule, is_theo_fallback, n_cand, n_exp, n_near = _select_stable_element_doc(
        docs,
        e_hull_max=e_hull_max,
        strict_less_than=strict_less_than,
        require_experimental=require_experimental,
        prefer_is_stable=prefer_is_stable,
        allow_theoretical_fallback=allow_theoretical_fallback,
        selection_mode=selection_mode,
    )

    if doc is None:
        row = _empty_row(
            symbol,
            rule,
            role=role,
            n_candidates=n_cand,
            n_experimental=n_exp,
            n_near_hull=n_near,
        )
        row["e_hull_max_eV_per_atom"] = e_hull_max
        row["is_theoretical_fallback"] = is_theo_fallback
        return row

    return _row_from_summary_doc(
        symbol,
        doc,
        role=role,
        selection_rule=rule,
        e_hull_max=e_hull_max,
        is_theo_fallback=is_theo_fallback,
        n_cand=n_cand,
        n_exp=n_exp,
        n_near=n_near,
        material_id_forced=False,
    )


def _row_from_summary_doc(
    symbol: str,
    doc: Any,
    *,
    role: str,
    selection_rule: str,
    e_hull_max: float,
    is_theo_fallback: bool,
    n_cand: int,
    n_exp: int,
    n_near: int,
    material_id_forced: bool = False,
) -> dict[str, Any]:
    crystal_system, spacegroup_symbol, spacegroup_number = _symmetry_fields(
        getattr(doc, "symmetry", None)
    )

    return {
        "element": symbol,
        "role": role,
        "material_id": str(doc.material_id),
        "formula_pretty": doc.formula_pretty,
        "energy_per_atom": float(doc.energy_per_atom),
        "energy_above_hull": _e_above_hull(doc),
        "theoretical": bool(doc.theoretical),
        "is_stable": getattr(doc, "is_stable", None),
        "crystal_system": crystal_system,
        "spacegroup_symbol": spacegroup_symbol,
        "spacegroup_number": spacegroup_number,
        "density": doc.density,
        "volume": doc.volume,
        "nsites": doc.nsites,
        "selection_rule": selection_rule,
        "e_hull_max_eV_per_atom": e_hull_max,
        "is_theoretical_fallback": is_theo_fallback,
        "n_candidates": n_cand,
        "n_experimental": n_exp,
        "n_near_hull": n_near,
        "material_id_forced": material_id_forced,
        "error": None,
    }


def _row_from_material_id_override(
    symbol: str,
    mpr: MPRester,
    material_id: str,
    *,
    role: str,
    e_hull_max: float,
    material_id_forced: bool = True,
    selection_rule: str = "material_id_override",
) -> dict[str, Any]:
    """Load a specific mp-id instead of automatic selection."""
    mp_id = str(material_id).strip()
    docs = mpr.materials.summary.search(material_ids=[mp_id], fields=SUMMARY_FIELDS)
    if not docs:
        row = _empty_row(symbol, f"override_not_found:{mp_id}", role=role)
        row["material_id"] = mp_id
        row["e_hull_max_eV_per_atom"] = e_hull_max
        row["selection_rule"] = f"{selection_rule}_missing"
        row["material_id_forced"] = material_id_forced
        return row

    doc = docs[0]
    return _row_from_summary_doc(
        symbol,
        doc,
        role=role,
        selection_rule=selection_rule,
        e_hull_max=e_hull_max,
        is_theo_fallback=False,
        n_cand=1,
        n_exp=0 if doc.theoretical else 1,
        n_near=1 if _e_above_hull(doc) < e_hull_max else 0,
        material_id_forced=material_id_forced,
    )


def _elements_with_roles(
    elements: Sequence[str],
    host_element: Optional[str],
) -> list[tuple[str, str]]:
    """Build (element, role) list: host first, then dopants (no duplicate of host)."""
    out: list[tuple[str, str]] = []
    host = str(host_element).strip() if host_element else ""
    if host:
        out.append((host, "host"))
    seen = {host} if host else set()
    for el in elements:
        sym = str(el).strip()
        if not sym or sym in seen:
            continue
        seen.add(sym)
        out.append((sym, "dopant"))
    return out


def fetch_stable_phases_for_elements(
    elements: Sequence[str],
    *,
    api_key: Optional[str] = None,
    e_hull_max: float = 0.1,
    strict_less_than: bool = True,
    require_experimental: bool = True,
    prefer_is_stable: bool = False,
    allow_theoretical_fallback: bool = False,
    selection_mode: SelectionMode = "min_nsites",
    host_element: Optional[str] = None,
    host_mp_id_override: Optional[str] = None,
    use_rt_mp_catalog: bool = False,
    rt_mp_id_overrides: Optional[dict[str, str]] = None,
    forced_mp_ids: Optional[dict[str, str]] = None,
    li_reference_mp_id: str = "mp-135",
    manual_radii: Optional[dict[str, float]] = None,
    include_manual_radii: bool = True,
) -> pd.DataFrame:
    """
    Query MP for each element; pick experimental phase with
    ``energy_above_hull < e_hull_max`` and minimum ``nsites`` (default 0.1 eV/atom).
    """
    key = mp_api_key(api_key)
    rows: list[dict[str, Any]] = []
    pairs = _elements_with_roles(elements, host_element)

    with MPRester(api_key=key) as mpr:
        for el, role in pairs:
            override = host_mp_id_override if role == "host" else None
            row = get_experimental_ground_state_element(
                el,
                mpr,
                e_hull_max=e_hull_max,
                strict_less_than=strict_less_than,
                require_experimental=require_experimental,
                prefer_is_stable=prefer_is_stable,
                allow_theoretical_fallback=allow_theoretical_fallback,
                selection_mode=selection_mode,
                role=role,
                material_id_override=override,
                use_rt_mp_catalog=use_rt_mp_catalog,
                rt_mp_id_overrides=rt_mp_id_overrides,
                forced_mp_ids=forced_mp_ids,
            )
            rows.append(row)

    df = pd.DataFrame(rows)
    if include_manual_radii:
        df = add_manual_radii_columns(df, manual_radii=manual_radii)
    df.attrs["forced_mp_ids"] = _merged_forced_mp_ids(forced_mp_ids)
    df.attrs["host_element"] = host_element
    df.attrs["host_mp_id_override"] = host_mp_id_override
    df.attrs["use_rt_mp_catalog"] = use_rt_mp_catalog
    df.attrs["rt_mp_id_overrides"] = rt_mp_id_overrides
    df.attrs["li_reference_mp_id"] = li_reference_mp_id
    df.attrs["selection_mode"] = selection_mode
    df.attrs["e_hull_max_eV_per_atom"] = e_hull_max
    df.attrs["strict_less_than"] = strict_less_than
    return df


def save_stable_phases_table(
    df: pd.DataFrame,
    path: Path | str,
) -> Path:
    """Write stable-phase table to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
