# -*- coding: utf-8 -*-
"""模型解释模块"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
import matplotlib

matplotlib.use('Agg')

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')


class ModelInterpreter:
    """基础模型解释器 (兼容旧代码)"""

    def __init__(self, model, background_data, model_type: str):
        pass


class EnhancedModelInterpreter:
    """增强版模型解释器 - 修复版"""

    def __init__(self, model, X_train, y_train, X_test, y_test, model_name, feature_names=None):
        self.model = model
        # 确保保存特征名
        # NOTE:
        # X_train / X_test 在训练器中通常被保存为 DataFrame(列名为 0..n-1)。
        # 旧实现使用 pd.DataFrame(X_train, columns=feature_names) 会触发“按列名重索引”，
        # 结果把整张表变成 NaN，导致 beeswarm 退化成灰色竖条。
        # 这里统一先取数值矩阵，再显式赋予 feature_names。
        self.feature_names = list(feature_names) if feature_names is not None else (
            [f"Feature_{i}" for i in range(np.asarray(X_train).shape[1])]
        )

        X_train_arr = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
        X_test_arr = X_test.values if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)

        # 转换为 DataFrame 以便 SHAP 识别列名
        self.X_train = pd.DataFrame(X_train_arr, columns=self.feature_names)
        self.X_test = pd.DataFrame(X_test_arr, columns=self.feature_names)

        self.model_name = model_name
        self._shap_values = None
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is not None:
            return self._explainer

        try:
            # 树模型使用 TreeExplainer
            tree_models = ['XGBoost', 'LightGBM', 'CatBoost', '随机森林', 'Extra Trees', '梯度提升树']
            # 检查模型是否是树模型或有 feature_importances_ 属性
            if self.model_name in tree_models or hasattr(self.model, 'feature_importances_'):
                self._explainer = shap.TreeExplainer(self.model)
            # 线性模型
            elif self.model_name in ['线性回归', 'Ridge回归', 'Lasso回归', 'ElasticNet']:
                background = shap.sample(self.X_train, min(100, len(self.X_train)))
                self._explainer = shap.LinearExplainer(self.model, background)
            # 其他模型
            else:
                background = shap.sample(self.X_train, min(50, len(self.X_train)))
                self._explainer = shap.KernelExplainer(self.model.predict, background)
        except:
            # 兜底方案
            background = shap.sample(self.X_train, min(20, len(self.X_train)))
            self._explainer = shap.KernelExplainer(self.model.predict, background)

        return self._explainer

    def compute_shap_values(self):
        if self._shap_values is not None:
            return self._shap_values

        explainer = self._get_explainer()
        # 采样测试集，防止计算太慢
        # 使用 pandas.sample 保证与后续作图的样本一致且可复现
        n = min(200, len(self.X_test))
        self._X_sample = self.X_test.sample(n=n, random_state=42) if len(self.X_test) > n else self.X_test.copy()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # check_additivity=False 主要用于树模型，防止微小误差导致报错。
                # 部分 Explainer（如 LinearExplainer）不支持该参数，这里做兼容处理。
                try:
                    shap_values = explainer.shap_values(self._X_sample, check_additivity=False)
                except TypeError:
                    shap_values = explainer.shap_values(self._X_sample)

            # 处理 list (多分类) 和 array (回归) 的区别
            if isinstance(shap_values, list):
                self._shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                self._shap_values = shap_values

            return self._shap_values
        except Exception as e:
            print(f"SHAP 计算错误: {e}")
            return None

    def plot_summary(self, plot_type='bar', max_display=20):
        # 确保 SHAP 值与用于作图的样本完全一致
        shap_values = self.compute_shap_values()
        if shap_values is None:
            return None, None

        if plot_type == 'beeswarm':
            plot_type = 'dot'

        vals = np.array(shap_values)

        # self._X_sample 已在 compute_shap_values 中创建（默认 200 条）
        X_plot_full = self._X_sample.copy()

        # 2) 只在“作图阶段”过滤掉在采样集中为常数的列（否则会灰 + 竖条）
        nunique = X_plot_full.nunique(dropna=False)
        valid_cols = nunique[nunique > 1].index.tolist()

        if len(valid_cols) >= 2:
            idx = [self.feature_names.index(c) for c in valid_cols]
            vals_plot = vals[:, idx]
            X_plot = X_plot_full[valid_cols]
            feature_names_plot = valid_cols
        else:
            # 实在都常数，就退回原始（至少能画出来），但 beeswarm 仍会退化
            vals_plot = vals
            X_plot = X_plot_full
            feature_names_plot = self.feature_names

        fig = plt.figure(figsize=(10, max(6, max_display * 0.4)))
        shap.summary_plot(
            vals_plot,
            X_plot,
            plot_type=plot_type,
            max_display=min(max_display, len(feature_names_plot)),
            feature_names=feature_names_plot,
            show=False
        )
        plt.tight_layout()

        export_df = pd.DataFrame(vals, columns=self.feature_names)
        return fig, export_df