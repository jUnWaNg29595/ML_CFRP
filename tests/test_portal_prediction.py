import numpy as np
import pandas as pd
import pytest

import core.portal_prediction as portal_prediction
from core.molecular_feature_workflow import WorkflowExecutionResult


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


def _artifact(model=None):
    model = model or FakePipeline()
    workflow = _workflow()
    contract = {
        'schema_version': 1,
        'feature_cols': ['x'],
        'target_col': 'Tg',
        'workflow_hash': 'workflow-1',
        'workflow_schema_version': 3,
        'source_columns': [{'column': 'resin_smiles', 'roles': ['resin']}],
        'workflow_source_columns': [{'column': 'resin_smiles', 'roles': ['resin']}],
        'workflow_present': True,
        'molecular_features_indicated': True,
        'pipeline_present': True,
        'imputer_present': False,
        'scaler_present': False,
        'numeric_ranges': {},
    }
    return {
        'model': model,
        'pipeline': model,
        'feature_cols': ['x'],
        'target_col': 'Tg',
        'extra': {
            'prediction_contract': contract,
            'molecular_feature_workflow': workflow,
        },
    }


def _config(artifact=None, *, enabled=True, status='published'):
    entry = {
        'id': 'tg-v1',
        'version': 'v1',
        'label': 'Tg model',
        'enabled': enabled,
        'publication_status': status,
        'unit': '°C',
        'target_col': 'Tg',
        'contract': _artifact(artifact)['extra']['prediction_contract'],
        '_artifact': artifact or _artifact(),
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
