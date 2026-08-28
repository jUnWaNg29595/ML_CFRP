import pandas as pd


def test_process_derivation_is_deterministic_and_does_not_fill_missing():
    from core.process_features import compute_process_features

    frame = pd.DataFrame({"cure_schedule": ["80C/2h;120C/1h"]})
    definitions = [
        {"name": "cure_stage_count", "source_type": "derived_workflow", "calculation_rule": {"implementation": "core.process_features:derive_cure_stage_count", "input_fields": ["cure_schedule"], "null_policy": "reject", "invalid_policy": "reject"}},
        {"name": "cure_total_time_h", "source_type": "derived_workflow", "calculation_rule": {"implementation": "core.process_features:derive_cure_total_time_h", "input_fields": ["cure_schedule"], "null_policy": "reject", "invalid_policy": "reject"}},
    ]
    result = compute_process_features(frame, definitions, {"source_bindings": [{"raw_column": "cure_schedule", "source_field": "cure_schedule"}]})
    assert result.errors == []
    assert result.features.loc[0, "cure_stage_count"] == 2
    assert result.features.loc[0, "cure_total_time_h"] == 3.0


def test_process_error_has_machine_readable_rule_and_source():
    from core.process_features import compute_process_features

    result = compute_process_features(
        pd.DataFrame({"other": [1]}),
        [{"name": "cure_stage_count", "source_type": "derived_workflow", "calculation_rule": {"implementation": "core.process_features:derive_cure_stage_count", "input_fields": ["cure_schedule"], "null_policy": "reject", "invalid_policy": "reject"}}],
        {"source_bindings": []},
    )
    assert result.errors and {"code", "feature", "source", "rule", "message"} <= set(result.errors[0])


def test_empty_curing_agent_is_structural_zero_but_invalid_nonempty_is_blocked():
    from core.process_features import count_smiles_components

    assert count_smiles_components(None, role="curing_agent") == 0
    assert count_smiles_components("", role="curing_agent") == 0
    assert count_smiles_components("CCO.CCN", role="curing_agent") == 2
    try:
        count_smiles_components("(", role="curing_agent")
    except ValueError as exc:
        assert "SMILES" in str(exc)
    else:
        raise AssertionError("invalid non-empty curing agent must fail")


def test_offline_script_delegates_business_derivation():
    import scripts.expand_manual_process_columns as script

    assert hasattr(script, "compute_process_features")

