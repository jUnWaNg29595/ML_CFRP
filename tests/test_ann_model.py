import joblib
import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.base import clone

from core.ann_model import ANNRegressor, FFN


def test_ffn_builds_gelu_and_dropout():
    model = FFN(4, [8, 4], activation='gelu', dropout_rate=0.25)
    modules = list(model.network.children())

    assert any(isinstance(module, nn.GELU) for module in modules)
    assert any(isinstance(module, nn.Dropout) for module in modules)


def test_ffn_omits_dropout_when_rate_is_zero():
    model = FFN(4, [8], activation='relu', dropout_rate=0.0)

    assert not any(isinstance(module, nn.Dropout) for module in model.network.children())


def test_ann_preserves_constructor_parameters_for_sklearn_clone():
    raw_layers = ["8", 4]
    model = ANNRegressor(
        hidden_layer_sizes_str=raw_layers,
        activation="GELU",
        learning_rate="0.001",
        batch_size=8,
    )

    assert model.get_params()["hidden_layer_sizes_str"] is raw_layers
    assert model.activation == "GELU"
    assert model.learning_rate == "0.001"
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()


def test_ann_rejects_invalid_configuration_at_fit():
    X = np.ones((4, 2))
    y = np.ones(4)
    with pytest.raises(ValueError, match='dropout'):
        ANNRegressor(dropout_rate=0.9).fit(X, y)

    with pytest.raises(ValueError, match='min_learning_rate'):
        ANNRegressor(learning_rate=1e-3, min_learning_rate=2e-3).fit(X, y)


def test_ann_trains_without_validation_and_records_metadata():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 5))
    y = X[:, 0] * 0.5 + X[:, 1] * -0.2
    model = ANNRegressor(
        hidden_layer_sizes_str='8,4',
        epochs=2,
        batch_size=8,
        validation_split=0.0,
        early_stopping=False,
        device='cpu',
        use_amp=False,
        verbose=False,
    )

    model.fit(X, y)

    assert len(model.train_loss_history) == 2
    assert model.validation_loss_history == []
    assert model.training_metadata_['device'] == 'cpu'
    assert model.training_metadata_['effective_batch_size'] >= 1


def test_ann_can_be_serialized_after_training(tmp_path):
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = X[:, 0] + X[:, 1]
    model = ANNRegressor(
        hidden_layer_sizes_str='4',
        epochs=1,
        batch_size=4,
        device='cpu',
        use_amp=False,
        verbose=False,
    ).fit(X, y)
    path = tmp_path / 'ann.joblib'

    joblib.dump(model, path)
    restored = joblib.load(path)

    np.testing.assert_allclose(model.predict(X[:3]), restored.predict(X[:3]))


def test_ann_data_parallel_initialization_failure_falls_back(monkeypatch):
    def fail_wrap(*args, **kwargs):
        raise RuntimeError("simulated DataParallel failure")

    monkeypatch.setattr("core.ann_model.wrap_model_for_multi_gpu", fail_wrap)
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = X[:, 0] - X[:, 1]

    model = ANNRegressor(
        hidden_layer_sizes_str="4",
        epochs=1,
        batch_size=4,
        device="cpu",
        use_data_parallel=True,
        use_amp=False,
        verbose=False,
    ).fit(X, y)

    assert model.training_metadata_["device"] == "cpu"
    assert model.training_metadata_["data_parallel_enabled"] is False
    assert any("simulated DataParallel failure" in reason for reason in model.training_metadata_["fallback_reasons"])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_ann_nonzero_cuda_device_disables_data_parallel_without_crashing():
    X = np.arange(24, dtype=float).reshape(12, 2)
    y = X[:, 0] - X[:, 1]
    model = ANNRegressor(
        hidden_layer_sizes_str="4",
        epochs=1,
        batch_size=4,
        device="cuda:1",
        use_data_parallel=True,
        use_amp=False,
        verbose=False,
    ).fit(X, y)

    assert model.training_metadata_["device"] == "cuda:1"
    assert model.training_metadata_["data_parallel_enabled"] is False
    assert any(
        "non-zero CUDA" in reason
        for reason in model.training_metadata_["fallback_reasons"]
    )
