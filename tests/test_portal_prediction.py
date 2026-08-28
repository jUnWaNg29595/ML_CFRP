import numpy as np
import pandas as pd
import pytest
import hashlib
from pathlib import Path
import tempfile

import core.portal_prediction as portal_prediction
from core.molecular_feature_workflow import (
    WorkflowExecutionResult,
    materialize_workflow_source_columns,
)
from core.dataset_manifest import compute_dataset_manifest_hash
from core.prediction_portal import compute_contract_hash, validate_publication_artifact
from core.feature_registry import compute_registry_hash
import json
from core.prediction_molecular_baseline import collect_workflow_source_columns


class FakeModel:
    feature_names_in_ = np.asarray(['x'])
    n_features_in_ = 1

    def __init__(self):
        self.calls = 0

    def predict(self, values):
        self.calls += 1
        assert list(values.columns) == ['x']
        return np.asarray([42.5] * len(values))


class FakePipeline(FakeModel):
    steps = [('model', object())]



def _workflow():
    return {
        'schema_version': 3,
        'workflow_hash': 'workflow-1',
        'steps': [
            {'step_id': 'resin', 'role': 'resin', 'source_columns': ['resin_smiles'], 'order': 1},
        ],
        'merge_order': ['resin'],
        'final_feature_names': ['x'],
        'feature_source_map': {'x': 'resin'},
    }


def _v2_context(workflow, feature_cols, feature_definitions):
    """Build the smallest complete v2 publication context for portal tests."""
    feature_definitions = [
        {
            **dict(item),
            'data_type': item.get('data_type', 'float'),
            'unit': item.get('unit', 'unknown'),
            'default_policy': item.get(
                'default_policy',
                'workflow_only' if item.get('source_type') in {'molecular_workflow', 'derived_workflow'} else 'explicit_only',
            ),
            'required_for_prediction': item.get('required_for_prediction', True),
            'nullable': item.get('nullable', False),
            **({'calculation_rule': item.get('calculation_rule') or {
                'input_fields': ['resin_smiles'],
                'implementation': 'fixture.workflow:derive',
                'null_policy': 'reject',
                'invalid_policy': 'reject',
            }} if item.get('source_type') in {'molecular_workflow', 'derived_workflow'} else {}),
        }
        for item in feature_definitions
    ]
    source_columns = collect_workflow_source_columns(workflow)
    profile = {
        'profile_id': 'fixture-profile',
        'status': 'approved',
        'feature_ids': [item.get('feature_id') for item in feature_definitions],
        'target_col': 'Tg',
    }
    registry_payload = {
        'schema_version': 1,
        'registry_version': 'fixture-v1',
        'features': feature_definitions,
        'model_profiles': {'fixture-profile': profile},
        'approval': {'status': 'approved'},
    }
    registry_payload['approval']['approved_hash'] = compute_registry_hash(registry_payload)
    registry_snapshot = {
        'schema_version': 1,
        'registry_version': 'fixture-v1',
        'registry_hash': registry_payload['approval']['approved_hash'],
        'profile_id': 'fixture-profile',
        'model_profile': profile,
        'features': feature_definitions,
        'registry_payload': registry_payload,
    }
    selected_payload = {'profile_id': registry_snapshot['profile_id'], 'model_profile': registry_snapshot['model_profile'], 'features': feature_definitions}
    registry_snapshot['selected_features_hash'] = hashlib.sha256(json.dumps(selected_payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    dataset_manifest = {
        'schema_version': 1,
        'dataset_id': 'portal-fixture',
        'model_profile_id': 'fixture-profile',
        'source_bindings': [],
        'feature_bindings': [
            {
                'feature_id': item.get('feature_id'),
                'raw_columns': ['resin_smiles'],
                'source_role': item.get('source_type'),
                'unit': item.get('unit', 'unknown'),
            }
            for item in feature_definitions
        ],
        'status': 'approved',
    }
    dataset_manifest['manifest_hash'] = compute_dataset_manifest_hash(dataset_manifest)
    contract = {
        'schema_version': 2,
        'feature_cols': list(feature_cols),
        'canonical_feature_cols': list(feature_cols),
        'effective_feature_cols': list(feature_cols),
        'removed_feature_cols': [],
        'removed_feature_reasons': {},
        'feature_registry_version': registry_snapshot['registry_version'],
        'feature_registry_hash': registry_snapshot['registry_hash'],
        'dataset_manifest_hash': dataset_manifest['manifest_hash'],
        'model_profile_id': registry_snapshot['profile_id'],
        'workflow_feature_cols': [
            item['name'] for item in feature_definitions
            if item.get('source_type') in {'molecular_workflow', 'derived_workflow'}
        ],
        'molecular_workflow_feature_cols': [
            item['name'] for item in feature_definitions
            if item.get('source_type') == 'molecular_workflow'
        ],
        'derived_feature_cols': [
            item['name'] for item in feature_definitions
            if item.get('source_type') == 'derived_workflow'
        ],
        'manual_input_feature_cols': [
            item['name'] for item in feature_definitions
            if item.get('source_type') == 'manual_input'
        ],
        'feature_definitions': [dict(item) for item in feature_definitions],
        'target_col': 'Tg',
        'workflow_hash': workflow.get('workflow_hash'),
        'workflow_schema_version': workflow.get('schema_version'),
        'source_columns': source_columns,
        'workflow_source_columns': [dict(item) for item in source_columns],
        'workflow_source_fields': [dict(item) for item in source_columns],
        'workflow_present': True,
        'molecular_features_indicated': True,
        'pipeline_present': True,
        'imputer_present': False,
        'scaler_present': False,
        'numeric_ranges': {},
        'contract_hash': '',
    }
    contract['contract_hash'] = compute_contract_hash(contract)
    return contract, registry_snapshot, dataset_manifest


def _artifact(model=None):
    model = model or FakePipeline()
    workflow = _workflow()
    contract, registry_snapshot, dataset_manifest = _v2_context(
        workflow,
        ['x'],
        [{'feature_id': 'x', 'name': 'x', 'source_type': 'molecular_workflow', 'status': 'approved'}],
    )
    return {
        'model': model,
        'pipeline': model,
        'feature_cols': ['x'],
        'target_col': 'Tg',
        'extra': {
            'prediction_contract': contract,
            'molecular_feature_workflow': workflow,
            'registry_snapshot': registry_snapshot,
            'dataset_manifest': dataset_manifest,
        },
    }


def test_v2_contract_requires_canonical_workflow_source_fields_but_aliases_are_optional():
    artifact = _artifact()
    manifest = artifact["extra"]["dataset_manifest"]
    manifest["feature_bindings"] = [{
        "feature_id": "x",
        "raw_columns": ["resin_smiles"],
        "source_role": "molecular_workflow",
        "unit": "unknown",
    }]
    manifest["manifest_hash"] = compute_dataset_manifest_hash(manifest)
    contract = artifact["extra"]["prediction_contract"]
    contract["dataset_manifest_hash"] = manifest["manifest_hash"]
    contract.pop("source_columns", None)
    contract.pop("workflow_source_columns", None)
    contract["contract_hash"] = compute_contract_hash(contract)

    report = validate_publication_artifact(artifact)

    assert report["ok"] is True, report["errors"]


def test_v2_contract_rejects_inconsistent_source_aliases_when_present():
    artifact = _artifact()
    contract = artifact["extra"]["prediction_contract"]
    contract["source_columns"] = [{"column": "tampered", "roles": ["resin"]}]
    contract["contract_hash"] = compute_contract_hash(contract)

    report = validate_publication_artifact(artifact)

    assert report["ok"] is False
    assert any("source" in error.lower() and "equal" in error.lower() for error in report["errors"])


def test_compact_snapshot_requires_contract_profile_id_to_match_registry_profile():
    artifact = _artifact()
    contract = artifact["extra"]["prediction_contract"]
    contract["model_profile_id"] = "different-profile"
    contract["contract_hash"] = compute_contract_hash(contract)

    report = validate_publication_artifact(artifact)

    assert report["ok"] is False
    assert any("profile" in error.lower() for error in report["errors"])


def test_resolved_manifest_must_be_approved_and_registry_consistent():
    artifact = _artifact()
    manifest = dict(artifact["extra"]["dataset_manifest"])
    manifest["status"] = "draft"
    manifest["manifest_hash"] = compute_dataset_manifest_hash(manifest)
    artifact["extra"]["dataset_manifest"] = manifest
    contract = artifact["extra"]["prediction_contract"]
    contract["dataset_manifest_hash"] = manifest["manifest_hash"]
    contract["contract_hash"] = compute_contract_hash(contract)

    report = validate_publication_artifact(artifact)

    assert report["ok"] is False
    assert any("manifest is not approved" in error.lower() for error in report["errors"])


def test_resolved_manifest_rejects_tampered_bindings_even_with_recomputed_hash():
    artifact = _artifact()
    manifest = dict(artifact["extra"]["dataset_manifest"])
    manifest["feature_bindings"] = [{
        "feature_id": "not-in-registry",
        "raw_columns": ["resin_smiles"],
        "source_role": "molecular_workflow",
        "unit": "unknown",
    }]
    manifest["manifest_hash"] = compute_dataset_manifest_hash(manifest)
    artifact["extra"]["dataset_manifest"] = manifest
    contract = artifact["extra"]["prediction_contract"]
    contract["dataset_manifest_hash"] = manifest["manifest_hash"]
    contract["contract_hash"] = compute_contract_hash(contract)

    report = validate_publication_artifact(artifact)

    assert report["ok"] is False
    assert any("unknown feature_id" in error for error in report["errors"])


def _config(artifact=None, *, enabled=True, status='published'):
    artifact = artifact or _artifact()
    fixture_path = Path(tempfile.gettempdir()) / 'cfrp_portal_fixture.joblib'
    fixture_path.write_bytes(b'portal-fixture-artifact')
    entry = {
        'id': 'tg-v1',
        'version': 'v1',
        'label': 'Tg model',
        'enabled': enabled,
        'publication_status': status,
        'unit': '°C',
        'target_col': 'Tg',
        'contract': artifact['extra']['prediction_contract'],
        '_artifact': artifact,
        'artifact_path': str(fixture_path),
        'artifact_hash': hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        'gate_report': {'ok': True, 'status': 'valid', 'errors': []},
    }
    return {'materials': {'epoxy_resin': {'targets': {'tg': {'models': [entry]}}}}}


def _request(**updates):
    request = {
        'material_type': 'epoxy_resin',
        'target': 'tg',
        'inputs': {'resin_smiles': 'C1CC1'},
        'confirmed_by_user': True,
    }
    request.update(updates)
    return request


def test_prediction_requires_confirmation_and_published_workflow(monkeypatch):
    config = _config()
    with pytest.raises(ValueError, match='确认'):
        portal_prediction.run_confirmed_prediction(
            _request(confirmed_by_user=False), config=config
        )

    artifact = _artifact()
    monkeypatch.setattr('core.portal_prediction._load_artifact', lambda entry: artifact)
    monkeypatch.setattr(
        'core.portal_prediction.execute_molecular_feature_workflow',
        lambda *args, **kwargs: WorkflowExecutionResult(
            features=pd.DataFrame({'x': [3.0]}),
            step_trace=[], warnings=[], workflow_hash='workflow-1', valid_row_indices={'resin': [0]}
        ),
    )
    result = portal_prediction.run_confirmed_prediction(
        _request(), config=config, progress=lambda *args, **kwargs: None
    )
    assert result.prediction == 42.5
    assert result.model_version == 'v1'
    assert result.feature_workflow_id == 'workflow-1'


def test_disabled_or_unpublished_release_is_rejected():
    with pytest.raises(ValueError, match='启用'):
        portal_prediction.load_published_portal_model(_config(enabled=False), 'epoxy_resin', 'tg')
    with pytest.raises(ValueError, match='已发布'):
        portal_prediction.load_published_portal_model(_config(status='draft'), 'epoxy_resin', 'tg')


def test_multiple_enabled_releases_are_rejected():
    config = _config()
    first = config['materials']['epoxy_resin']['targets']['tg']['models'][0]
    second = dict(first)
    second['id'] = 'tg-v2'
    second['version'] = 'v2'
    config['materials']['epoxy_resin']['targets']['tg']['models'].append(second)
    with pytest.raises(ValueError, match='歧义'):
        portal_prediction.load_published_portal_model(config, 'epoxy_resin', 'tg')


def test_unknown_columns_and_forbidden_ai_values_are_rejected():
    config = _config()
    errors = portal_prediction.validate_prediction_request(
        _request(inputs={'resin_smiles': 'C1CC1', 'unexpected': 1}), config
    )
    assert any('未知列' in error for error in errors)
    errors = portal_prediction.validate_prediction_request(
        _request(inputs={'resin_smiles': 'C1CC1'}, ai_feature_vector=[1, 2]), config
    )
    assert any('AI' in error for error in errors)
    errors = portal_prediction.validate_prediction_request(
        _request(inputs={'resin_smiles': 'C1CC1', 'resin_smiles_2': lambda: None}), config
    )
    assert any('callable' in error for error in errors)


def test_missing_and_malformed_molecular_sources_are_rejected():
    config = _config()
    errors = portal_prediction.validate_prediction_request(
        _request(inputs={'resin_smiles': '(C'}), config
    )
    assert any('括号' in error for error in errors)
    errors = portal_prediction.validate_prediction_request(
        _request(inputs={'other': 1}), config
    )
    assert any('缺少分子源列' in error for error in errors)


def test_canonical_smiles_are_expanded_to_declared_numbered_sources():
    frame = materialize_workflow_source_columns(
        pd.DataFrame({
            'resin_smiles': ['CCO.CCN'],
            'hardener_smiles': ['NCCN'],
        }),
        resin_columns=['resin_smiles_1', 'resin_smiles_2', 'resin_smiles_3'],
        hardener_columns=['curing_agent_smiles_1', 'curing_agent_smiles_2'],
    )
    assert frame.loc[0, 'resin_smiles_1'] == 'CCO'
    assert frame.loc[0, 'resin_smiles_2'] == 'CCN'
    assert pd.isna(frame.loc[0, 'resin_smiles_3'])
    assert frame.loc[0, 'curing_agent_smiles_1'] == 'NCCN'
    assert pd.isna(frame.loc[0, 'curing_agent_smiles_2'])


def test_realistic_feature_gap_is_blocked_without_derived_workflow(monkeypatch):
    artifact = _artifact()
    artifact['feature_cols'] = ['x', 'degree_of_cure_pct']
    artifact['model'].n_features_in_ = 2
    artifact['model'].feature_names_in_ = np.asarray(['x', 'degree_of_cure_pct'])
    artifact['pipeline'] = artifact['model']
    workflow = artifact['extra']['molecular_feature_workflow']
    workflow['final_feature_names'] = ['x', 'degree_of_cure_pct']
    workflow['feature_source_map'] = {'x': 'resin', 'degree_of_cure_pct': 'resin'}
    contract, registry_snapshot, dataset_manifest = _v2_context(
        workflow,
        ['x', 'degree_of_cure_pct'],
        [
            {'feature_id': 'x', 'name': 'x', 'source_type': 'molecular_workflow', 'status': 'approved'},
            {'feature_id': 'degree', 'name': 'degree_of_cure_pct', 'source_type': 'derived_workflow', 'status': 'approved'},
        ],
    )
    artifact['extra']['prediction_contract'] = contract
    artifact['extra']['registry_snapshot'] = registry_snapshot
    artifact['extra']['dataset_manifest'] = dataset_manifest
    config = _config(artifact)
    config['materials']['epoxy_resin']['targets']['tg']['models'][0]['contract'] = contract
    monkeypatch.setattr(
        'core.portal_prediction.execute_molecular_feature_workflow',
        lambda *args, **kwargs: WorkflowExecutionResult(
            features=pd.DataFrame({'x': [3.0]}),
            step_trace=[], warnings=[], workflow_hash='workflow-1', valid_row_indices={'resin': [0]}
        ),
    )
    with pytest.raises(ValueError, match='degree_of_cure_pct'):
        portal_prediction.run_confirmed_prediction(_request(), config=config)


def test_explicit_non_workflow_features_must_be_supplied():
    contract = {
        'feature_cols': ['x', 'degree_of_cure_pct'],
    }
    with pytest.raises(ValueError, match='显式工艺/实验特征'):
        portal_prediction._merge_explicit_model_features(
            pd.DataFrame({'x': [3.0]}),
            pd.DataFrame({'resin_smiles': ['C1CC1']}),
            contract,
            {'final_feature_names': ['x']},
        )
