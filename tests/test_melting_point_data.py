import pandas as pd
import pytest

from core.melting_point_data import (
    canonicalize_smiles,
    deduplicate_melting_point_records,
    parse_melting_point_text,
    prepare_melting_point_dataset,
    summarize_melting_point_dataset,
    persist_melting_point_dataset,
    load_persisted_melting_point_dataset,
    normalize_melting_point_units,
)


def test_parse_celsius_single_value():
    result = parse_melting_point_text('126 °C')

    assert result['mp_c'] == 126.0
    assert result['mp_quality'] == 'high'


def test_parse_fahrenheit_and_kelvin():
    assert parse_melting_point_text('212 °F')['mp_c'] == 100.0
    assert parse_melting_point_text('373.15 K')['mp_c'] == 100.0


def test_normalize_numeric_fahrenheit_column_to_celsius():
    result = normalize_melting_point_units(pd.DataFrame([
        {"smiles": "CCO", "mp_c": 212.0, "mp_unit_raw": "°F"},
    ]))

    assert result.loc[0, "mp_c"] == 100.0
    assert result.loc[0, "mp_unit_normalized"] == "C"
    assert result.loc[0, "mp_unit"] == "°C"


def test_normalize_kelvin_annotation_to_celsius():
    result = normalize_melting_point_units(pd.DataFrame([
        {"smiles": "CCO", "mp_raw": "373.15 K"},
    ]))

    assert result.loc[0, "mp_c"] == 100.0
    assert result.loc[0, "mp_unit_raw"] == "K"


def test_parse_range_keeps_bounds_without_training_value():
    result = parse_melting_point_text('120-130 °C')

    assert result['mp_c'] is None
    assert result['mp_lower_c'] == 120.0
    assert result['mp_upper_c'] == 130.0
    assert result['mp_quality'] == 'range'


@pytest.mark.parametrize('text', ['120 °C-130 °C', '120 °C to 130 °C'])
def test_parse_ranges_with_units_on_each_bound(text):
    result = parse_melting_point_text(text)

    assert result['mp_c'] is None
    assert result['mp_lower_c'] == 120.0
    assert result['mp_upper_c'] == 130.0
    assert result['mp_quality'] == 'range'


def test_parse_decomposition_is_low_quality():
    result = parse_melting_point_text('240 °C (decomposes)')

    assert result['mp_c'] == 240.0
    assert result['mp_quality'] == 'decomp'


def test_unparsed_text_never_becomes_training_value():
    result = parse_melting_point_text('softens above room temperature')

    assert result['mp_c'] is None
    assert result['mp_quality'] == 'unparsed'


def test_canonical_smiles_deduplicates_equivalent_structures():
    canonical_left = canonicalize_smiles('C(C)O')
    canonical_right = canonicalize_smiles('CCO')
    if canonical_left is None or canonical_right is None:
        pytest.skip('RDKit is unavailable; canonicalization cannot be verified')
    assert canonical_left == canonical_right


def test_deduplication_preserves_same_structure_across_roles_and_classes():
    frame = pd.DataFrame([
        {'smiles': 'CCO', 'mp_c': 10.0, 'mp_quality': 'high', 'component_role': 'resin'},
        {'smiles': 'CCO', 'mp_c': 20.0, 'mp_quality': 'high', 'component_role': 'hardener', 'hardener_class': '胺'},
        {'smiles': 'CCO', 'mp_c': 30.0, 'mp_quality': 'high', 'component_role': 'hardener', 'hardener_class': '酚'},
        {'smiles': 'CCO', 'mp_c': 40.0, 'mp_quality': 'estimated', 'component_role': 'hardener', 'hardener_class': '胺'},
    ])
    result = deduplicate_melting_point_records(frame)
    assert len(result) == 3
    assert set(zip(result['component_role'], result.get('hardener_class', pd.Series('', index=result.index)))) == {
        ('resin', ''), ('hardener', '胺'), ('hardener', '酚')
    }
    amine = result[(result['component_role'] == 'hardener') & (result['hardener_class'] == '胺')]
    assert amine['mp_c'].tolist() == [20.0]

def test_deduplication_prefers_high_quality_record():
    frame = pd.DataFrame([
        {'smiles': 'CCO', 'mp_c': -114.0, 'mp_quality': 'estimated', 'source': 'a'},
        {'smiles': 'CCO', 'mp_c': -114.1, 'mp_quality': 'high', 'source': 'b'},
    ])

    result = deduplicate_melting_point_records(frame)

    assert len(result) == 1
    assert result.iloc[0]['source'] == 'b'


def test_deduplication_handles_missing_quality_column():
    frame = pd.DataFrame([{'smiles': 'CCO', 'mp_c': -114.0}])

    result = deduplicate_melting_point_records(frame)

    assert len(result) == 1


def test_deduplication_does_not_group_invalid_smiles():
    frame = pd.DataFrame([
        {'smiles': 'not-a-smiles', 'mp_c': 10.0, 'mp_quality': 'high', 'source': 'a'},
        {'smiles': None, 'mp_c': 20.0, 'mp_quality': 'high', 'source': 'b'},
    ])

    result = deduplicate_melting_point_records(frame)

    assert len(result) == 2
    assert set(result['source']) == {'a', 'b'}


def test_prepare_dataset_excludes_low_quality_by_default():
    frame = pd.DataFrame([
        {'smiles': 'CCO', 'mp_raw': '-114 °C', 'component_role': 'other'},
        {'smiles': 'CCN', 'mp_raw': 'about 80 °C', 'component_role': 'hardener'},
    ])

    result = prepare_melting_point_dataset(frame)

    assert set(result['mp_quality']) == {'high'}
    summary = summarize_melting_point_dataset(result)
    assert summary['high_quality_count'] == 1


def test_prepare_dataset_excludes_invalid_and_missing_smiles():
    frame = pd.DataFrame([
        {'smiles': 'CCO', 'mp_raw': '-114 °C'},
        {'smiles': 'not-a-smiles', 'mp_raw': '20 °C'},
        {'smiles': None, 'mp_raw': '30 °C'},
    ])

    result = prepare_melting_point_dataset(frame)

    assert len(result) == 1
    assert result.iloc[0]['canonical_smiles'] is not None


def test_parse_mixture_is_distinct_from_estimated_quality():
    result = parse_melting_point_text('120 °C (mixture)')

    assert result['mp_quality'] == 'mixture'


def test_prepare_dataset_includes_numeric_mixture_only_when_low_quality_enabled():
    frame = pd.DataFrame([
        {'smiles': 'CCO', 'mp_raw': '120 °C (mixture)'},
    ])

    assert prepare_melting_point_dataset(frame).empty
    result = prepare_melting_point_dataset(frame, include_low_quality=True)

    assert len(result) == 1
    assert result.iloc[0]['mp_quality'] == 'mixture'


def test_deduplication_prefers_mixture_over_unparsed_record():
    frame = pd.DataFrame([
        {'smiles': 'CCO', 'mp_c': 10.0, 'mp_quality': 'unparsed', 'source': 'unparsed'},
        {'smiles': 'CCO', 'mp_c': 20.0, 'mp_quality': 'mixture', 'source': 'mixture'},
    ])

    result = deduplicate_melting_point_records(frame)

    assert len(result) == 1
    assert result.iloc[0]['source'] == 'mixture'


def test_persist_and_load_melting_point_dataset(tmp_path):
    raw = pd.DataFrame([{"smiles": "CCO", "mp_raw": "-114 °C", "source": "test"}])
    cleaned = prepare_melting_point_dataset(raw)
    paths = persist_melting_point_dataset(raw, cleaned, output_dir=tmp_path)
    loaded_raw, loaded_cleaned, summary = load_persisted_melting_point_dataset(output_dir=tmp_path)

    assert set(paths) == {"raw_path", "cleaned_path", "summary_path"}
    assert loaded_raw is not None and len(loaded_raw) == 1
    assert loaded_cleaned is not None and len(loaded_cleaned) == 1
    assert summary["row_count"] == 1
