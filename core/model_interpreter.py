# -*- coding: utf-8 -*-
"""模型解释模块"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
import contextlib

# 设置绘图后端，防止在无头服务器上报错
import matplotlib

matplotlib.use('Agg')

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings('ignore')


class ModelInterpreter:
    """基础模型解释器 (旧版兼容)"""

    def __init__(self, model, background_data, model_type: str):
        pass  # 占位，建议使用 EnhancedModelInterpreter


class EnhancedModelInterpreter:
    """增强版模型解释器"""

    def __init__(self, model, X_train, y_train, X_test, y_test, model_name, feature_names=None):
        self.model = model
        self.X_train = pd.DataFrame(X_train, columns=feature_names) if feature_names else pd.DataFrame(X_train)
        self.X_test = pd.DataFrame(X_test, columns=feature_names) if feature_names else pd.DataFrame(X_test)
        self.model_name = model_name
        self.feature_names = feature_names or self.X_train.columns.tolist()
        self._shap_values = None
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is not None:
            return self._explainer

        try:
            # 树模型使用 TreeExplainer (速度快)
            if self.model_name in ['XGBoost', 'LightGBM', 'CatBoost', '随机森林', 'Extra Trees', '梯度提升树']:
                self._explainer = shap.TreeExplainer(self.model)
            # 线性模型使用 LinearExplainer
            elif self.model_name in ['线性回归', 'Ridge回归', 'Lasso回归', 'ElasticNet']:
                # 采样背景数据
                background = shap.sample(self.X_train, min(100, len(self.X_train)))
                self._explainer = shap.LinearExplainer(self.model, background)
            # 其他模型使用通用的 KernelExplainer (速度慢)
            else:
                background = shap.sample(self.X_train, min(50, len(self.X_train)))
                self._explainer = shap.KernelExplainer(self.model.predict, background)
        except Exception as e:
            print(f"SHAP Explainer 初始化回退到 KernelExplainer: {e}")
            background = shap.sample(self.X_train, min(50, len(self.X_train)))
            self._explainer = shap.KernelExplainer(self.model.predict, background)

        return self._explainer

    def compute_shap_values(self):
        if self._shap_values is not None:
            return self._shap_values

        explainer = self._get_explainer()

        # 对测试集进行采样计算 (全量计算太慢)
        X_sample = shap.sample(self.X_test, min(200, len(self.X_test)))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 部分模型 check_additivity 会报错，即使计算是正确的
            shap_values = explainer.shap_values(X_sample, check_additivity=False)

        # 处理 shap_values 的格式差异
        # 此时 shap_values 可能是 list (多分类) 或 array (回归/二分类)
        if isinstance(shap_values, list):
            self._shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            self._shap_values = shap_values

        self._X_sample = X_sample  # 保存采样后的数据用于绘图
        return self._shap_values

    def plot_summary(self, plot_type='bar', max_display=20):
        """生成 SHAP 摘要图"""
        shap_values = self.compute_shap_values()
        if shap_values is None:
            return None, None

        # 创建一个新的 Figure
        fig = plt.figure(figsize=(10, max(6, max_display * 0.4)))

        # 绘图
        # feature_names 参数确保显示真实特征名
        shap.summary_plot(
            shap_values,
            self._X_sample,
            plot_type=plot_type,
            max_display=max_display,
            feature_names=self.feature_names,
            show=False
        )

        plt.tight_layout()

        # 生成导出数据
        # 将 SHAP 值转换为 DataFrame，列名为特征名
        export_df = pd.DataFrame(shap_values, columns=self.feature_names)
        # 添加原始特征值 (可选，方便对比)
        # export_df = pd.concat([export_df.add_suffix('_SHAP'), self._X_sample.reset_index(drop=True)], axis=1)

        return fig, export_df