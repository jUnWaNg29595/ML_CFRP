import json


def test_training_run_persists_contract_summary_and_excludes_arbitrary_secrets(tmp_path):
    from core.training_runs import TrainingRunManager

    context = {
        "prediction_contract": {"schema_version": 2, "contract_hash": "c1"},
        "registry_snapshot": {"registry_version": "v1", "registry_hash": "r1"},
        "dataset_manifest": {"dataset_id": "d1", "manifest_hash": "m1"},
        "feature_audit": {"canonical_feature_cols": ["x", "y"], "effective_feature_cols": ["x"], "removed_feature_cols": ["y"]},
        "secret": "do-not-persist",
    }
    manager = TrainingRunManager(str(tmp_path))
    summary = manager.save_run("demo", {"status": "completed"}, contract_context=context)
    run_dir = tmp_path / summary.run_id
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    contract = json.loads((run_dir / "contract.json").read_text(encoding="utf-8"))
    assert metadata["feature_registry_version"] == "v1"
    assert metadata["feature_registry_hash"] == "r1"
    assert metadata["dataset_id"] == "d1"
    assert metadata["dataset_manifest_hash"] == "m1"
    assert metadata["canonical_feature_count"] == 2
    assert metadata["effective_feature_count"] == 1
    assert metadata["removed_feature_cols"] == ["y"]
    assert contract["prediction_contract"]["contract_hash"] == "c1"
    assert "do-not-persist" not in (run_dir / "metadata.json").read_text(encoding="utf-8")
    assert "do-not-persist" not in (run_dir / "contract.json").read_text(encoding="utf-8")
