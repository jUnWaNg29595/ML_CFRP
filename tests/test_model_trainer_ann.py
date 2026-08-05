from core.ann_model import ANNRegressor
from core.model_trainer import EnhancedModelTrainer


def test_trainer_passes_ann_controls_instead_of_dropping_them():
    trainer = EnhancedModelTrainer(use_gpu=False)

    model = trainer._get_model(
        '人工神经网络',
        hidden_layer_sizes='32,16',
        activation='silu',
        dropout_rate=0.2,
        optimizer='adamw',
        learning_rate=0.002,
        weight_decay=0.0001,
        validation_split=0.2,
        early_stopping=True,
        patience=5,
        min_delta=1e-4,
        lr_scheduler='reduce_on_plateau',
        scheduler_factor=0.5,
        min_learning_rate=1e-6,
        gradient_clip=1.0,
        use_amp=False,
        device='cpu',
        use_data_parallel=False,
    )

    assert isinstance(model, ANNRegressor)
    assert model.hidden_layer_sizes_str == '32,16'
    assert model.activation == 'silu'
    assert model.dropout_rate == 0.2
    assert model.optimizer == 'adamw'
    assert model.gradient_clip == 1.0


def test_trainer_accepts_legacy_use_gpu_parameter():
    trainer = EnhancedModelTrainer(use_gpu=False)
    model = trainer._get_model(
        '人工神经网络',
        hidden_layer_sizes='8',
        use_gpu=False,
        epochs=1,
        device='cpu',
    )

    assert isinstance(model, ANNRegressor)
    assert model.device == 'cpu'
