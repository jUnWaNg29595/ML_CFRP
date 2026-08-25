import pandas as pd

from core.melting_point_screening import (
    build_melting_point_artifact_extra,
    is_melting_point_artifact,
    validate_melting_point_artifact,
)


def test_melting_point_artifact_requires_task_and_celsius_unit():
    artifact = {
        "target_col": "mp_c",
        "extra": {
            "task_kind": "melting_point",
            "target_unit": "C",
        },
    }

    assert is_melting_point_artifact(artifact)
    report = validate_melting_point_artifact(artifact)
    assert report["ok"]
    assert report["task_kind"] == "melting_point"
    assert report["target_unit"] == "C"
    assert report["target_col"] == "mp_c"


def test_explicit_non_default_target_column_is_preserved():
    artifact = {
        "target_col": "melting_point",
        "feature_cols": ["feature_a"],
        "extra": {
            "task_kind": "melting_point",
            "target_unit": "C",
            "target_col": "melting_point",
            "molecular_feature_workflow": {"workflow_hash": "workflow-a"},
        },
    }

    assert is_melting_point_artifact(artifact)
    assert validate_melting_point_artifact(artifact)["target_col"] == "melting_point"


def test_tg_artifact_is_not_accepted_as_melting_point_model():
    artifact = {
        "target_col": "tg_c",
        "extra": {"task_kind": "tg", "target_unit": "C"},
    }

    assert not is_melting_point_artifact(artifact)
    report = validate_melting_point_artifact(artifact)
    assert not report["ok"]
    assert "task_kind" in report["error_codes"]


def test_wrong_target_unit_is_rejected():
    artifact = {
        "target_col": "mp_c",
        "extra": {"task_kind": "melting_point", "target_unit": "F"},
    }

    report = validate_melting_point_artifact(artifact)

    assert not report["ok"]
    assert "target_unit" in report["error_codes"]


def test_artifact_extra_records_dataset_fingerprint_roles_and_workflow_hash():
    dataset = pd.DataFrame(
        [
            {"smiles": "CCO", "mp_c": 10.0, "mp_quality": "high", "component_role": "resin"},
            {"smiles": "CCN", "mp_c": 20.0, "mp_quality": "high", "component_role": "hardener"},
            {"smiles": "CCC", "mp_c": 30.0, "mp_quality": "estimated", "component_role": "hardener"},
        ]
    )

    extra = build_melting_point_artifact_extra(dataset, workflow_hash="workflow-a")

    assert extra["task_kind"] == "melting_point"
    assert extra["target_unit"] == "C"
    assert extra["target_col"] == "mp_c"
    assert extra["dataset_row_count"] == 3
    assert len(extra["dataset_fingerprint"]) == 64
    assert extra["role_counts"] == {"resin": 1, "hardener": 2}
    assert extra["workflow_hash"] == "workflow-a"
    assert extra["quality_policy"] == "high_quality_only"


def test_dataset_fingerprint_changes_when_dataset_changes():
    dataset = pd.DataFrame(
        [{"smiles": "CCO", "mp_c": 10.0, "component_role": "resin"}]
    )

    first = build_melting_point_artifact_extra(dataset)
    changed = build_melting_point_artifact_extra(dataset.assign(mp_c=[11.0]))

    assert first["dataset_fingerprint"] != changed["dataset_fingerprint"]
