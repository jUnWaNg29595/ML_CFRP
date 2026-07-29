import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from core.model_trainer import EnhancedModelTrainer
from core.transformer_bnn_model import TransformerBNNRegressor
from core.ui_config import TRANSFORMER_BNN_DEFAULTS


def test_transformer_bnn_defaults_to_mse_for_regression():
    model = TransformerBNNRegressor()

    assert model.loss_name == "mse"
    assert TRANSFORMER_BNN_DEFAULTS["loss_name"] == "mse"


def test_transformer_bnn_predict_is_deterministic_by_default():
    model = TransformerBNNRegressor.__new__(TransformerBNNRegressor)
    calls = []

    def fake_predict_raw(X, mc_samples=None):
        calls.append(mc_samples)
        return np.array([1.0]), np.array([0.1])

    model._predict_raw = fake_predict_raw

    prediction = model.predict([[1.0]])

    assert prediction.tolist() == [1.0]
    assert calls == [1]


def test_transformer_bnn_keeps_missing_values_for_internal_masking(monkeypatch):
    class FakeTransformerBNN(BaseEstimator, RegressorMixin):
        def __init__(self):
            self.validation_data = None
            self.fit_input = None
            self.predict_inputs = []

        def fit(self, X, y):
            self.fit_input = np.asarray(X)
            return self

        def predict(self, X):
            self.predict_inputs.append(np.asarray(X))
            return np.zeros(len(X), dtype=float)

        def predict_with_uncertainty(self, X, n_samples=None):
            self.predict_inputs.append(np.asarray(X))
            return np.zeros(len(X), dtype=float), np.ones(len(X), dtype=float)

    fake_model = FakeTransformerBNN()
    trainer = EnhancedModelTrainer(use_gpu=False)
    monkeypatch.setattr(
        trainer,
        "_get_model",
        lambda model_name, **params: fake_model,
    )

    X = pd.DataFrame(
        {
            "dense_feature": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "sparse_feature": [np.nan, np.nan, np.nan, 1.0, np.nan, 2.0],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    trainer.train_model(
        X,
        y,
        model_name="Transformer + BNN",
        test_size=0.33,
        random_state=42,
    )

    assert np.isnan(fake_model.fit_input).any()
    assert all(np.isnan(values).any() for values in fake_model.predict_inputs)


def test_transformer_bnn_does_not_use_holdout_test_as_validation(monkeypatch):
    class FakeTransformerBNN(BaseEstimator, RegressorMixin):
        def __init__(self):
            self.validation_data = None

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=float)

    fake_model = FakeTransformerBNN()
    trainer = EnhancedModelTrainer(use_gpu=False)
    monkeypatch.setattr(
        trainer,
        "_get_model",
        lambda model_name, **params: fake_model,
    )

    X = pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "feature_b": [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    trainer.train_model(
        X,
        y,
        model_name="Transformer + BNN",
        test_size=0.25,
        random_state=42,
    )

    assert fake_model.validation_data is None


def test_transformer_bnn_emits_postprocessing_status():
    source = (
        TransformerBNNRegressor.__module__.replace(".", "/")
    )
    source_path = __import__("pathlib").Path(__file__).resolve().parents[1] / f"{source}.py"
    source_text = source_path.read_text(encoding="utf-8")

    assert '"postprocessing"' in source_text
