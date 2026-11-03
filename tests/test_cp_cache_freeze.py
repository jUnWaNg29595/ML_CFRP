# -*- coding: utf-8 -*-
"""Executable compatibility and adversarial tests for cache contract."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Test works both when installed at project root and when run in this bundle.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.component_physics import compute_component_physics, enrich_narrow_table
from core.crosslink_physics import compute_crosslink_features
from core.cp_cache_integrity import (
    formulation_input_hashes, invalid_cp_cache_mask,
)

DGEBA = 'CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'
EDA = 'NCCN'
MONOEPOXY = 'CC1CO1'


def formulation(**overrides):
    base = {
        'resin_1_structure': DGEBA,
        'resin_1_phr': 100.0,
        'resin_1_molecular_weight_g_mol': 340.419,
        'resin_1_equivalent_weight_g_eq': 170.2095,
        'curing_agent_1_structure': EDA,
        'curing_agent_1_phr': 8.83,
        'curing_agent_1_molecular_weight_g_mol': 60.1,
        'curing_agent_1_equivalent_weight_g_eq': 15.025,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def pair(df, *, strict=True):
    cp = compute_component_physics(df, assume_stoichiometric_if_missing=not strict)
    fresh = compute_crosslink_features(df, assume_stoichiometric_if_missing=not strict)
    cached = compute_crosslink_features(df, cp=cp, assume_stoichiometric_if_missing=not strict)
    return cp, fresh, cached


def same_physics(a, b):
    for name in ('xl_r_value', 'xl_W_g_per_epoxy', 'xl_epoxy_conc_mol_m3',
                 'xl_nu_theory_mol_m3', 'xl_nu_junction_mol_m3',
                 'xl_f_r', 'xl_f_h', 'xl_f_avg', 'xl_alpha_gel'):
        x, y = a.iloc[0][name], b.iloc[0][name]
        if pd.isna(x):
            assert pd.isna(y), name
        else:
            assert y == pytest.approx(x, rel=1e-9, abs=1e-9), name
    assert a.iloc[0]['xl_r_source'] == b.iloc[0]['xl_r_source']


def test_identical_precomputed_cp_gives_identical_physics():
    _, fresh, cached = pair(formulation())
    same_physics(fresh, cached)
    assert fresh.iloc[0]['xl_r_source'] == 'calculated_from_mass'


@pytest.mark.parametrize('col,new', [
    ('curing_agent_1_phr', 6.0),
    ('curing_agent_1_phr', 8.835),
    ('curing_agent_1_equivalent_weight_g_eq', 20.0),
    ('resin_1_equivalent_weight_g_eq', 190.0),
    ('curing_agent_1_molecular_weight_g_mol', 64.0),
    ('resin_1_structure', MONOEPOXY),
    ('curing_agent_1_structure', 'NCCCN'),
    ('curing_agent_1_active_hydrogen_equivalent_count', 3.0),
    ('formulation_resin_total_eew_g_eq', 200.0),
])
def test_change_any_physics_source_invalidates_and_matches_fresh(col, new):
    raw = formulation()
    old = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    new_df = raw.copy()
    new_df.loc[0, col] = new
    assert invalid_cp_cache_mask(new_df, old, assume_stoichiometric_if_missing=False)[0]
    direct = compute_crosslink_features(new_df, assume_stoichiometric_if_missing=False)
    with_old_cp = compute_crosslink_features(new_df, cp=old, assume_stoichiometric_if_missing=False)
    same_physics(direct, with_old_cp)


def test_old_cp_without_hash_is_recomputed():
    raw = formulation()
    old = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    old = old.drop(columns=['cp_cache_input_hash'])
    updated = formulation(curing_agent_1_phr=6.0)
    assert invalid_cp_cache_mask(updated, old, assume_stoichiometric_if_missing=False)[0]
    same_physics(
        compute_crosslink_features(updated, assume_stoichiometric_if_missing=False),
        compute_crosslink_features(updated, cp=old, assume_stoichiometric_if_missing=False),
    )


def test_cache_mode_change_invalidates():
    raw = formulation()
    cp_lenient = compute_component_physics(raw, assume_stoichiometric_if_missing=True)
    assert invalid_cp_cache_mask(raw, cp_lenient, assume_stoichiometric_if_missing=False)[0]
    cp_strict = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    assert not invalid_cp_cache_mask(raw, cp_strict, assume_stoichiometric_if_missing=False)[0]


def test_valid_cache_does_not_reparse(monkeypatch):
    raw = formulation()
    cached_cp = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    import core.component_physics as m

    def crash(*args, **kwargs):
        raise AssertionError('valid cp must avoid costly component recomputation')

    monkeypatch.setattr(m, 'compute_component_physics', crash)
    result = compute_crosslink_features(raw, cp=cached_cp, assume_stoichiometric_if_missing=False)
    assert result.iloc[0]['xl_r_value'] > 0


def test_density_change_not_affected_by_cp_epoxy_density():
    raw = formulation()
    cache = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    for density in [1.0, 1.2, 1.4]:
        a = compute_crosslink_features(raw, rho_g_cm3=density, assume_stoichiometric_if_missing=False)
        b = compute_crosslink_features(raw, cp=cache, rho_g_cm3=density, assume_stoichiometric_if_missing=False)
        same_physics(a, b)


def test_multiple_rows_reorder_detected():
    a = formulation()
    b = formulation(curing_agent_1_phr=6.0)
    raw = pd.concat([a, b], ignore_index=True)
    cp = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    assert not invalid_cp_cache_mask(raw, cp, assume_stoichiometric_if_missing=False).any()
    shuffled = raw.iloc[::-1].reset_index(drop=True)
    assert invalid_cp_cache_mask(shuffled, cp, assume_stoichiometric_if_missing=False).all()


def test_duplicate_dataframe_index_keeps_position_hash():
    raw = pd.concat([formulation(), formulation(curing_agent_1_phr=6.0)])
    assert list(raw.index) == [0, 0]
    cp = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    assert list(cp.index) == [0, 0]
    assert not invalid_cp_cache_mask(raw, cp, assume_stoichiometric_if_missing=False).any()


def test_float_string_roundtrip_hash_stable():
    df = formulation()
    series = formulation_input_hashes(df, assume_stoichiometric_if_missing=False)
    csv_roundtrip = pd.read_csv(pd.io.common.StringIO(df.to_csv(index=False)))
    assert list(series) == list(formulation_input_hashes(csv_roundtrip, assume_stoichiometric_if_missing=False))


def test_columns_reordered_hash_stable():
    df = formulation()
    rev = df[list(reversed(df.columns))]
    assert list(formulation_input_hashes(df)) == list(formulation_input_hashes(rev))


def test_new_source_column_invalidates_even_if_mw_same():
    raw = formulation()
    cp = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    raw['resin_2_structure'] = 'BAD'
    assert invalid_cp_cache_mask(raw, cp, assume_stoichiometric_if_missing=False)[0]


def test_partial_still_remains_nan_in_strict_after_enrich():
    d = formulation(resin_2_structure='BAD', resin_2_phr=np.nan,
                    xl_r_source='literature')
    enhanced = enrich_narrow_table(d)
    result = compute_crosslink_features(enhanced, assume_stoichiometric_if_missing=False)
    assert pd.isna(result.iloc[0]['xl_r_value'])
    assert result.iloc[0]['xl_r_source'] == 'missing_stoichiometry'


def test_legacy_unknown_r_source_safe():
    d = pd.DataFrame({'resin_1_structure':[DGEBA],
                      'curing_agent_1_structure':[EDA], 'cp_r_value':[0.84]})
    en = enrich_narrow_table(d)
    strict = compute_crosslink_features(en, assume_stoichiometric_if_missing=False)
    assert pd.isna(strict.iloc[0]['xl_r_value'])


def test_real_literature_r_survives_old_cache():
    raw = formulation()
    cp = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    changed = raw.assign(formulation_r_value=0.75, formulation_r_source='literature')
    direct = compute_crosslink_features(changed, assume_stoichiometric_if_missing=False)
    with_cp = compute_crosslink_features(changed, cp=cp, assume_stoichiometric_if_missing=False)
    same_physics(direct, with_cp)
    assert with_cp.iloc[0]['xl_r_value'] == pytest.approx(0.75)
    assert with_cp.iloc[0]['xl_r_source'] == 'literature'


def test_csv_cache_hash_roundtrip_reuse():
    raw = formulation()
    cp = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    cp_after = pd.read_csv(pd.io.common.StringIO(cp.to_csv(index=False)))
    assert not invalid_cp_cache_mask(raw, cp_after, assume_stoichiometric_if_missing=False).any()
    same_physics(
        compute_crosslink_features(raw, assume_stoichiometric_if_missing=False),
        compute_crosslink_features(raw, cp=cp_after, assume_stoichiometric_if_missing=False),
    )


def test_label_mismatch_forces_refresh():
    raw = formulation()
    cp = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    cp.index = pd.Index(['wrong'])
    assert invalid_cp_cache_mask(raw, cp, assume_stoichiometric_if_missing=False)[0]
    same_physics(
        compute_crosslink_features(raw, assume_stoichiometric_if_missing=False),
        compute_crosslink_features(raw, cp=cp, assume_stoichiometric_if_missing=False),
    )


def test_third_component_change_invalidates():
    raw = formulation(resin_2_structure='BAD', resin_2_phr=np.nan)
    cp = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    newer = raw.assign(resin_2_structure=MONOEPOXY, resin_2_phr=12.0)
    assert invalid_cp_cache_mask(newer, cp, assume_stoichiometric_if_missing=False)[0]
    same_physics(
        compute_crosslink_features(newer, assume_stoichiometric_if_missing=False),
        compute_crosslink_features(newer, cp=cp, assume_stoichiometric_if_missing=False),
    )


def test_parquet_cache_hash_roundtrip_reuse(tmp_path):
    pytest.importorskip('pyarrow')
    raw = formulation()
    cp = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    path = tmp_path / 'cp.parquet'
    cp.to_parquet(path)
    restored = pd.read_parquet(path)
    assert not invalid_cp_cache_mask(raw, restored, assume_stoichiometric_if_missing=False).any()
    same_physics(
        compute_crosslink_features(raw, assume_stoichiometric_if_missing=False),
        compute_crosslink_features(raw, cp=restored, assume_stoichiometric_if_missing=False),
    )


def test_input_mode_from_csv_string_false_is_false():
    raw = formulation()
    cp = compute_component_physics(raw, assume_stoichiometric_if_missing=False)
    cp['cp_cache_assume_stoich'] = 'False'
    assert not invalid_cp_cache_mask(raw, cp, assume_stoichiometric_if_missing=False)[0]
    assert invalid_cp_cache_mask(raw, cp, assume_stoichiometric_if_missing=True)[0]


def test_registry_feature_set_still_unchanged():
    from core.feature_registry import get_feature_registry, FEATURE_SETS
    reg = get_feature_registry()
    assert len(reg) == 77
    # No new cache fields are registered training features.
    assert 'cp_cache_input_hash' not in reg
    assert 'cp_cache_max_components' not in reg
