import copy

from core.dataset_manifest import (
    compute_dataset_manifest_hash,
    normalize_dataset_manifest,
    resolve_dataset_feature_bindings,
    validate_dataset_manifest,
)


def registry():
    features = [
        {"feature_id": "stage_count", "name": "cure_stage_count", "source_type": "derived_workflow", "status": "approved", "unit": "stage"},
        {"feature_id": "total_time", "name": "cure_total_time_h", "source_type": "derived_workflow", "status": "approved", "unit": "h"},
        {"feature_id": "temp", "name": "temperature", "source_type": "manual_input", "status": "approved", "unit": "C"},
    ]
    return {"schema_version": 1, "registry_version": "1", "features": features, "model_profiles": {"p": {"feature_ids": ["stage_count", "total_time", "temp"], "status": "approved", "target_col": "tg", "allow_feature_subset": True}}, "approval": {"status": "approved"}}


def manifest(bindings=None):
    if bindings is None:
        bindings = [{"feature_id": "stage_count", "raw_columns": ["schedule"], "source_role": "derived_workflow", "unit": "stage"}, {"feature_id": "total_time", "raw_columns": ["schedule"], "source_role": "derived_workflow", "unit": "h"}]
    return {"schema_version": 1, "dataset_id": "d", "model_profile_id": "p", "source_bindings": [{"raw_column": "schedule", "source_field": "cure_schedule", "parse_rule_version": "v1"}], "feature_bindings": bindings, "status": "approved"}


def test_one_source_column_can_feed_multiple_derived_features():
    assert validate_dataset_manifest(manifest(), registry(), ["schedule"], require_approved=True)["ok"]


def test_manifest_rejects_duplicate_feature_binding_and_missing_required():
    duplicate = manifest([{"feature_id": "stage_count", "raw_columns": ["schedule"]}, {"feature_id": "stage_count", "raw_columns": ["schedule"]}])
    assert not validate_dataset_manifest(duplicate, registry())["ok"]
    strict_registry = registry()
    strict_registry["model_profiles"]["p"]["allow_feature_subset"] = False
    assert not validate_dataset_manifest(manifest([]), strict_registry, ["schedule"], require_approved=True)["ok"]


def test_manifest_hash_changes_on_mapping_and_normalization_is_stable():
    base = normalize_dataset_manifest(manifest())
    changed = copy.deepcopy(base)
    changed["feature_bindings"][0]["unit"] = "count"
    assert compute_dataset_manifest_hash(base) != compute_dataset_manifest_hash(changed)
    assert normalize_dataset_manifest(base) == base


def test_manifest_rejects_unknown_raw_column_and_resolves_bindings():
    report = validate_dataset_manifest(manifest(), registry(), ["other"])
    assert not report["ok"]
    resolved = resolve_dataset_feature_bindings(manifest(), registry(), "p")
    assert resolved["stage_count"]["raw_columns"] == ["schedule"]


def test_approved_manifest_rejects_unapproved_feature_and_alias_candidate():
    bad_registry = registry()
    bad_registry["features"][0]["status"] = "blocked"
    assert not validate_dataset_manifest(manifest(), bad_registry, ["schedule"], require_approved=True)["ok"]
    alias_manifest = manifest()
    alias_manifest["aliases"] = [{"raw_column": "sched", "candidate_feature_id": "stage_count", "status": "approved"}]
    report = validate_dataset_manifest(alias_manifest, registry(), ["schedule"], require_approved=True)
    assert not report["ok"]
