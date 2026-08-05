import numpy as np
import pandas as pd

from core import model_trainer as trainer_module
from core.model_trainer import AutoGluonWrapper


class _FakePrediction:
    def __init__(self, values):
        self.values = np.asarray(values)


class _FakePredictor:
    created_paths = []

    def __init__(self, *, label, path, verbosity):
        del label, verbosity
        if path in self.created_paths:
            raise RuntimeError('Learner is already fit.')
        self.created_paths.append(path)
        self.path = path

    def fit(self, train_data, **kwargs):
        del kwargs
        self.train_columns = [column for column in train_data.columns if column != 'target']
        return self

    def predict(self, test_data):
        return _FakePrediction(np.zeros(len(test_data)))


def test_autogluon_wrappers_get_unique_paths_with_same_timestamp(monkeypatch):
    _FakePredictor.created_paths = []
    monkeypatch.setattr(trainer_module, 'TabularPredictor', _FakePredictor)
    monkeypatch.setattr(trainer_module.time, 'time', lambda: 1_750_000_000)

    first = AutoGluonWrapper()
    second = AutoGluonWrapper()

    assert first.save_path != second.save_path


def test_autogluon_wrapper_refreshes_path_before_each_fit(monkeypatch):
    _FakePredictor.created_paths = []
    monkeypatch.setattr(trainer_module, 'TabularPredictor', _FakePredictor)
    monkeypatch.setattr(trainer_module.time, 'time', lambda: 1_750_000_000)

    wrapper = AutoGluonWrapper()
    wrapper.fit(pd.DataFrame({'feature': [1.0, 2.0]}), np.asarray([0.0, 1.0]))
    first_path = wrapper.save_path

    wrapper.fit(pd.DataFrame({'feature': [3.0, 4.0]}), np.asarray([1.0, 0.0]))

    assert wrapper.save_path != first_path
    assert len(_FakePredictor.created_paths) == 2
