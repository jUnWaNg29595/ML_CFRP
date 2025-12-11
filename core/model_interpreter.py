# -*- coding: utf-8 -*-
"""模型解释模块"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
import contextlib

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')


class ModelInterpreter:
    """基础模型解释器"""

    def __init__(self, model, background_data, model_type: str):
        self.model = model
        self.background_data = background_data
        self.model_type = model_type
        self.shap_values = None

        tree_models = ["随机森林", "Extra Trees", "梯度提升树", "XGBoost", "LightGBM", "CatBoost"]

        try:
            if model_type in tree_models:
                self.explainer = shap.TreeExplainer(model)
            else:
                sample = shap.sample(background_data, min(100, len(background_data)))
                self.explainer = shap.KernelExplainer(model.predict, sample)
        except:
            sample = shap.sample(background_data, min(100, len(background_data)))
            self.explainer = shap.KernelExplainer(model.predict, sample)

    def plot_summary(self, data, plot_type="bar", max_display=15):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(None):
                self.shap_values = self.explainer.shap_values(data)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(self.shap_values, data, plot_type=plot_type, max_display=max_display, show=False)
        plt.tight_layout()
        return fig


class EnhancedModelInterpreter:
    """增强版模型解释器"""

    def __init__(self, model, X_train, y_train, X_test, y_test, model_name, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.feature_names = feature_names or (X_train.columns.tolist() if hasattr(X_train, 'columns') 
                                                else [f"Feature_{i}" for i in range(X_train.shape[1])])
        self._shap_values = None

    def compute_shap_values(self):
        if self._shap_values is not None:
            return self._shap_values

        try:
            if self.model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
                explainer = shap.TreeExplainer(self.model)
            else:
                background = shap.sample(self.X_train, min(100, len(self.X_train)))
                explainer = shap.KernelExplainer(self.model.predict, background)

            X_sample = shap.sample(self.X_test, min(100, len(self.X_test)))
            self._shap_values = explainer.shap_values(X_sample)
            return self._shap_values
        except:
            return None

    def plot_shap_summary(self, plot_type='bar', max_display=15):
        shap_values = self.compute_shap_values()
        if shap_values is None:
            return None

        X_sample = shap.sample(self.X_test, min(100, len(self.X_test)))
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type=plot_type, max_display=max_display, show=False)
        plt.tight_layout()
        return fig
