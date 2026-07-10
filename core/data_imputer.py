# -*- coding: utf-8 -*-
"""使用训练好的模型补齐缺失数据"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from .model_io import loads_artifact


class ModelBasedImputer:
    """基于训练模型的数据补齐器"""

    def __init__(self, model_artifact_bytes: bytes):
        """
        初始化补齐器

        参数:
            model_artifact_bytes: 模型文件的字节数据
        """
        self.artifact = loads_artifact(model_artifact_bytes)
        self.pipeline = self.artifact.get('pipeline')
        self.model = self.artifact.get('model')
        self.target_col = self.artifact.get('target_col', '')
        self.feature_cols = self.artifact.get('feature_cols', [])

        if self.pipeline is None and self.model is None:
            raise ValueError("模型文件无效：缺少 pipeline 或 model")

    def impute(self, df: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """
        补齐数据框中的缺失值

        参数:
            df: 待补齐的数据框
            target_col: 目标列名（如果为None则使用模型保存的列名）

        返回:
            补齐后的数据框副本
        """
        df_result = df.copy()
        target = target_col or self.target_col

        if target not in df_result.columns:
            raise ValueError(f"目标列 '{target}' 不存在于数据框中")

        # 找出缺失值的行
        missing_mask = df_result[target].isna()
        missing_count = missing_mask.sum()

        if missing_count == 0:
            return df_result

        # 检查特征列
        missing_features = [col for col in self.feature_cols if col not in df_result.columns]
        if missing_features:
            raise ValueError(f"缺少特征列: {missing_features}")

        # 提取缺失行的特征
        X_missing = df_result.loc[missing_mask, self.feature_cols]

        # 使用模型预测
        predictor = self.pipeline if self.pipeline is not None else self.model
        predictions = predictor.predict(X_missing)

        # 填充预测值
        df_result.loc[missing_mask, target] = predictions

        return df_result

    def get_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': self.artifact.get('model_name', 'Unknown'),
            'target_col': self.target_col,
            'feature_cols': self.feature_cols,
            'metrics': self.artifact.get('metrics', {}),
            'created_at': self.artifact.get('created_at', 0)
        }


def impute_with_model(
    df: pd.DataFrame,
    model_path: str,
    target_col: Optional[str] = None
) -> pd.DataFrame:
    """
    使用保存的模型文件补齐数据

    参数:
        df: 待补齐的数据框
        model_path: 模型文件路径
        target_col: 目标列名

    返回:
        补齐后的数据框
    """
    with open(model_path, 'rb') as f:
        model_bytes = f.read()

    imputer = ModelBasedImputer(model_bytes)
    return imputer.impute(df, target_col)
