from core.ui_config import (
    ANN_DEFAULTS,
    MANUAL_TUNING_PARAMS,
    RECOMMENDED_PRESETS,
    prepare_manual_training_params,
)


def test_ann_defaults_include_professional_controls():
    required = {
        'hidden_layer_sizes',
        'activation',
        'dropout_rate',
        'optimizer',
        'learning_rate',
        'weight_decay',
        'batch_size',
        'epochs',
        'validation_split',
        'early_stopping',
        'patience',
        'min_delta',
        'lr_scheduler',
        'scheduler_factor',
        'min_learning_rate',
        'gradient_clip',
        'device',
        'use_data_parallel',
        'use_amp',
        'scaler_type',
        'normalize_target',
        'random_state',
        'verbose',
    }

    assert set(ANN_DEFAULTS) == required
    assert ANN_DEFAULTS == {
        'hidden_layer_sizes': '256,128,64',
        'activation': 'relu',
        'dropout_rate': 0.2,
        'optimizer': 'adamw',
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'batch_size': 512,
        'epochs': 150,
        'validation_split': 0.15,
        'early_stopping': True,
        'patience': 30,
        'min_delta': 0.0001,
        'lr_scheduler': 'reduce_on_plateau',
        'scheduler_factor': 0.5,
        'min_learning_rate': 0.000001,
        'gradient_clip': 1.0,
        'device': 'auto',
        'use_data_parallel': True,
        'use_amp': False,
        'scaler_type': 'standard',
        'normalize_target': False,
        'random_state': 42,
        'verbose': True,
    }


def test_ann_schema_contains_each_new_control():
    configs = MANUAL_TUNING_PARAMS['人工神经网络']
    names = {item['name'] for item in configs}

    assert names == set(ANN_DEFAULTS)

    activation = next(item for item in configs if item['name'] == 'activation')
    assert activation['args']['options'] == [
        'relu',
        'gelu',
        'silu',
        'elu',
        'tanh',
        'leaky_relu',
    ]

    dropout = next(item for item in configs if item['name'] == 'dropout_rate')
    assert dropout['args']['min_value'] == 0.0
    assert dropout['args']['max_value'] == 0.8

    validation_split = next(item for item in configs if item['name'] == 'validation_split')
    assert validation_split['args']['min_value'] == 0.0
    assert validation_split['args']['max_value'] == 0.4

    scheduler = next(item for item in configs if item['name'] == 'lr_scheduler')
    assert scheduler['option_labels'] == {
        'none': '无',
        'reduce_on_plateau': 'ReduceLROnPlateau',
        'cosine_annealing': 'CosineAnnealing',
    }

    sections = {item['name']: item['section'] for item in configs}
    assert sections['hidden_layer_sizes'] == '网络结构'
    assert sections['validation_split'] == '训练稳定性/验证'
    assert sections['device'] == '设备与性能'


def test_ann_presets_are_independent_dictionaries():
    presets = RECOMMENDED_PRESETS['人工神经网络']
    assert set(presets) == {'稳健小样本', '快速 GPU', '高容量网络', '恢复默认'}

    params = [presets[name]['params'] for name in presets]
    assert len({id(item) for item in params}) == len(params)
    assert all(set(item) == set(ANN_DEFAULTS) for item in params)
    assert all(isinstance(presets[name]['desc'], str) and presets[name]['desc'] for name in presets)
    assert presets['恢复默认']['params'] == ANN_DEFAULTS
    assert presets['恢复默认']['params'] is not ANN_DEFAULTS
    assert presets['稳健小样本']['params'] is not presets['恢复默认']['params']

    original = dict(presets['稳健小样本']['params'])
    candidate = dict(presets['稳健小样本']['params'])
    candidate['epochs'] = -1
    assert presets['稳健小样本']['params'] == original
    assert ANN_DEFAULTS['epochs'] != -1


def test_ann_pipeline_parameters_are_explicitly_compatible():
    configs = {item['name']: item for item in MANUAL_TUNING_PARAMS['人工神经网络']}

    assert 'pipeline' in configs['scaler_type']['help'].lower()
    assert 'pipeline' in configs['normalize_target']['help'].lower()


def test_manual_training_params_do_not_duplicate_global_random_state():
    params = {'random_state': 7, 'epochs': 3}

    prepared = prepare_manual_training_params(params)

    assert prepared == {'epochs': 3}
    assert params == {'random_state': 7, 'epochs': 3}
