from core.prediction_portal import build_prediction_contract, compute_contract_hash, validate_publication_artifact
from core.model_io import create_model_artifact, dumps_artifact, loads_artifact


class Model:
    feature_names_in_ = ["molecular_x", "derived_temperature", "manual_pressure"]
    n_features_in_ = 3


def test_contract_v2_keeps_feature_partitions_and_hash():
    artifact = {"model": Model(), "pipeline": None, "feature_cols": ["molecular_x", "derived_temperature", "manual_pressure"], "target_col": "tg_c", "extra": {}}
    snapshot = {"registry_version": "v1", "registry_hash": "r1", "features": [{"feature_id": "m", "name": "molecular_x", "source_type": "molecular_workflow", "status": "approved"}, {"feature_id": "d", "name": "derived_temperature", "source_type": "derived_workflow", "status": "approved"}, {"feature_id": "p", "name": "manual_pressure", "source_type": "manual_input", "status": "approved"}]}
    contract = build_prediction_contract(artifact=artifact, feature_cols=artifact["feature_cols"], target_col="tg_c", registry_snapshot=snapshot, dataset_manifest={"manifest_hash": "m1"}, model_profile_id="p", canonical_feature_cols=artifact["feature_cols"], effective_feature_cols=artifact["feature_cols"], removed_feature_cols=[], removed_feature_reasons={})
    assert contract["schema_version"] == 2
    assert contract["workflow_feature_cols"] == ["molecular_x", "derived_temperature"]
    assert contract["manual_input_feature_cols"] == ["manual_pressure"]
    assert contract["contract_hash"] == compute_contract_hash(contract)


def test_contract_context_round_trip():
    context = {"prediction_contract": {"schema_version": 2, "contract_hash": "x"}, "registry_snapshot": {"registry_hash": "r1"}, "dataset_manifest": {"manifest_hash": "m1"}, "feature_audit": {"effective_feature_cols": ["x"]}}
    artifact = create_model_artifact(model_name="demo", target_col="tg_c", feature_cols=["x"], model=Model(), contract_context=context)
    restored = loads_artifact(dumps_artifact(artifact))
    assert restored["extra"]["registry_snapshot"]["registry_hash"] == "r1"
    assert restored["extra"]["dataset_manifest"]["manifest_hash"] == "m1"
