"""Pinned E_s feature list that recovers SVR ~15/94 meV (opt-only 9/10 holdout).

Do not replace this with the current cluster feature_importance_*_opt/train.csv
selection (~28–38 features → SVR ~120/134 meV).
"""

from __future__ import annotations

SOLUTION_FEATURES_BEST: list[str] = [
    'endmember_MagpieData avg_dev CovalentRadius',
    'endmember_MagpieData avg_dev NdValence',
    'endmember_MagpieData mean NsValence',
    'endmember_MagpieData avg_dev MendeleevNumber',
    'endmember_MagpieData avg_dev Column',
    'MagpieData mode Electronegativity',
    'endmember_MagpieData mode NUnfilled',
    'endmember_MagpieData mode SpaceGroupNumber',
    'endmember_MagpieData avg_dev NsValence',
    'endmember_MagpieData mean CovalentRadius',
    'endmember_MagpieData mean NUnfilled',
    'endmember_MagpieData mean NsUnfilled',
    'endmember_MagpieData maximum CovalentRadius',
    'endmember_MagpieData range Electronegativity',
    'endmember_MagpieData range NValence',
    'endmember_MagpieData mean NdValence',
    'endmember_MagpieData mean Column',
    'endmember_MagpieData mean Row',
    'mp_n_candidates',
    'endmember_spacegroup_number',
    'endmember_MagpieData range CovalentRadius',
    'MagpieData minimum NValence',
    'MagpieData minimum Number',
    'endmember_MagpieData mean Number',
    'MagpieData maximum CovalentRadius',
    'endmember_packing fraction',
    'endmember_fraction',
    'endmember_MagpieData avg_dev Electronegativity',
    'MagpieData mean Column',
    'endmember_MagpieData avg_dev GSvolume_pa',
    'packing fraction',
    'endmember_MagpieData mean AtomicWeight',
    'endmember_MagpieData range AtomicWeight',
    'endmember_volume',
    'endmember_MagpieData avg_dev Row',
    'endmember_MagpieData mean GSvolume_pa',
    'endmember_MagpieData mean MeltingT',
    'MagpieData mean NUnfilled',
    'endmember_MagpieData range MendeleevNumber',
    'endmember_density',
    'mp_n_experimental',
    'MagpieData mode NpValence',
    'endmember_MagpieData avg_dev NUnfilled',
    'MagpieData minimum MeltingT',
    'vpa',
    'density',
    'endmember_n_symmetry_ops',
    'delta_valency_vs_Li',
    'endmember_MagpieData mean NValence',
    'endmember_MagpieData avg_dev SpaceGroupNumber',
]


def resolve_solution_features(available_columns, *, min_features: int = 40) -> list[str]:
    """Return pinned E_s columns present in ``available_columns``."""
    avail = set(available_columns)
    pinned = [f for f in SOLUTION_FEATURES_BEST if f in avail]
    if len(pinned) < min_features:
        raise RuntimeError(
            f'E_s feature pin failed: only {len(pinned)}/{len(SOLUTION_FEATURES_BEST)} '
            f'features found in datasheet (need >= {min_features}). '
            f'Check matminer/endmember merge and that you load energy_pool="opt" columns.'
        )
    return pinned
