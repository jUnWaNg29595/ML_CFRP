import pandas as pd
import pytest

from core.melting_point_training import (
    build_melting_point_split_metrics,
    collect_workflow_source_columns,
    prepare_melting_point_source_frame,
)


def test_collect_workflow_source_columns_preserves_declared_order():
    workflow = {
        'steps': [
            {'source_columns': ['resin_smiles_1', 'resin_smiles_2']},
            {'source_columns': ['resin_smiles_2', 'resin_smiles_3']},
        ],
        'input_contract': {'smiles_col': 'smiles'},
    }
    assert collect_workflow_source_columns(workflow) == [
        'resin_smiles_1',
        'resin_smiles_2',
        'resin_smiles_3',
        'smiles',
    ]


def test_prepare_single_role_numbered_sources_from_smiles():
    dataset = pd.DataFrame({'smiles': ['CCO', 'CCN'], 'mp_c': [10.0, 20.0]})
    frame, report = prepare_melting_point_source_frame(
        dataset,
        {'steps': [{'source_columns': ['resin_smiles_1', 'resin_smiles_2']}]},
    )
    assert frame['resin_smiles_1'].tolist() == ['CCO', 'CCN']
    assert frame['resin_smiles_2'].tolist() == ['CCO', 'CCN']
    assert report['source_role'] == 'resin'


def test_prepare_rejects_mixed_resin_and_hardener_workflow():
    dataset = pd.DataFrame({'smiles': ['CCO'], 'mp_c': [10.0]})
    with pytest.raises(ValueError, match='同时包含树脂和固化剂'):
        prepare_melting_point_source_frame(
            dataset,
            {
                'steps': [
                    {'source_columns': ['resin_smiles_1']},
                    {'source_columns': ['curing_agent_smiles_1']},
                ]
            },
        )


def test_prepare_requires_smiles_column():
    with pytest.raises(ValueError, match='缺少 smiles'):
        prepare_melting_point_source_frame(
            pd.DataFrame({'mp_c': [10.0]}),
            {'steps': [{'source_columns': ['smiles']}]},
        )


def test_split_metrics_stratifies_roles_and_hardener_classes():
    dataset = pd.DataFrame({
        'mp_c': [100.0, 110.0, 120.0, 130.0],
        'component_role': ['resin', 'hardener', 'hardener', 'resin'],
        'hardener_class': ['', '胺', '酸酐', ''],
    })
    result = build_melting_point_split_metrics(
        dataset,
        train_indices=[0, 1],
        test_indices=[2, 3],
        y_train=[100.0, 112.0],
        y_pred_train=[101.0, 110.0],
        y_test=[120.0, 130.0],
        y_pred_test=[118.0, 133.0],
    )

    assert result['train']['roles']['resin']['n'] == 1
    assert result['train']['hardener_classes']['胺']['n'] == 1
    assert result['test']['hardener_classes']['酸酐']['n'] == 1


def test_split_metrics_marks_single_sample_r2_as_not_evaluable():
    dataset = pd.DataFrame({
        'mp_c': [100.0],
        'component_role': ['resin'],
        'hardener_class': [''],
    })
    result = build_melting_point_split_metrics(
        dataset,
        train_indices=[0],
        y_train=[100.0],
        y_pred_train=[105.0],
    )

    assert result['train']['metrics']['r2'] is None
    assert result['train']['metrics']['r2_status'] == 'insufficient_samples'


def test_split_metrics_marks_constant_target_r2_as_not_evaluable():
    dataset = pd.DataFrame({
        'mp_c': [100.0, 100.0],
        'component_role': ['resin', 'resin'],
        'hardener_class': ['', ''],
    })
    result = build_melting_point_split_metrics(
        dataset,
        test_indices=[0, 1],
        y_test=[100.0, 100.0],
        y_pred_test=[99.0, 101.0],
    )

    assert result['test']['metrics']['r2'] is None
    assert result['test']['metrics']['r2_status'] == 'constant_target'
