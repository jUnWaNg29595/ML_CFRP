"""
独立特征对比与机器学习评估模块 (Comparison & ML Evaluation)
核心科学原则：
1. 严禁在原系统上修改旧模型，所有评估均在独立训练副本中执行；
2. 四组阶梯特征集对比：
   - A 组：现有快速机理特征基线 (Baseline Mechanism Features)
   - B 组：A + 网络拓扑特征 (Topology Features)
   - C 组：B + 自由体积与堆积特征 (Free Volume & Packing)
   - D 组：C + 相互作用能量密度与氢键特征 (Energy Density & HBond)
3. 严格 GroupKFold 分组划分（按配方 ID / 文献来源分组），杜绝同一文献或同一配方跨折数据泄漏；
4. 模拟失败样本严禁以 0 填充；对比必须在相同有效子集上对齐评估；
5. 科学诚实性铁律：若小试样本量小（如 6 个配方），主要结论应是“物理特征可计算性、数值稳定度与化学区分度”，明确标注小样本不足以宣称整体泛化 R2 显著提升，绝不伪造虚高指标。
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

class FeatureComparisonEngine:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def evaluate_feature_groups(
        self,
        df: pd.DataFrame,
        target_col: str,
        group_col: str,
        baseline_feature_cols: List[str],
        topology_cols: List[str],
        free_volume_cols: List[str],
        interaction_cols: List[str],
        n_splits: int = 3
    ) -> Dict[str, Any]:
        """
        在相同样本子集和相同分组折叠下评估四组特征模型表现
        """
        # 1. 过滤有效样本：目标列与所有待测特征不能全为空
        all_cols = list(set([target_col, group_col] + baseline_feature_cols + topology_cols + free_volume_cols + interaction_cols))
        valid_cols = [c for c in all_cols if c in df.columns]
        sub_df = df.dropna(subset=[target_col, group_col]).copy()

        n_samples = len(sub_df)
        unique_groups = sub_df[group_col].nunique()

        # 科学边界检查：若有效分组数过少，不能做可信 CV
        if unique_groups < 2 or n_samples < 4:
            return {
                "status": "INSUFFICIENT_SAMPLE_SIZE_FOR_CV",
                "sample_count": n_samples,
                "group_count": unique_groups,
                "message": "独立配方/分组样本量过少，不足以进行统计严密的交叉验证评估。结论主要依据计算可行性与物理特征区分度。",
                "results": {}
            }

        actual_splits = min(n_splits, unique_groups)
        gkf = GroupKFold(n_splits=actual_splits)

        groups_dict = {
            "A_Baseline": baseline_feature_cols,
            "B_Plus_Topology": baseline_feature_cols + topology_cols,
            "C_Plus_FreeVolume": baseline_feature_cols + topology_cols + free_volume_cols,
            "D_Plus_Interactions": baseline_feature_cols + topology_cols + free_volume_cols + interaction_cols
        }

        results = {}

        for grp_name, feat_cols in groups_dict.items():
            # 仅保留在数据集中存在的特征
            active_cols = [c for c in feat_cols if c in sub_df.columns]
            if not active_cols:
                results[grp_name] = {"status": "NO_FEATURES_AVAILABLE"}
                continue

            y_true_all = []
            y_pred_all = []

            for train_idx, test_idx in gkf.split(sub_df, sub_df[target_col], sub_df[group_col]):
                train_data = sub_df.iloc[train_idx]
                test_data = sub_df.iloc[test_idx]

                X_train = train_data[active_cols].fillna(train_data[active_cols].median())
                y_train = train_data[target_col]

                X_test = test_data[active_cols].fillna(train_data[active_cols].median())
                y_test = test_data[target_col]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = RandomForestRegressor(n_estimators=50, random_state=self.random_state, max_depth=5)
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)

                y_true_all.extend(y_test.values)
                y_pred_all.extend(preds)

            y_true_arr = np.array(y_true_all)
            y_pred_arr = np.array(y_pred_all)

            mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
            rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
            r2 = float(r2_score(y_true_arr, y_pred_arr)) if len(y_true_arr) > 1 else 0.0

            results[grp_name] = {
                "features_used_count": len(active_cols),
                "features_list": active_cols,
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "R2": round(r2, 4)
            }

        return {
            "status": "PASS",
            "sample_count": n_samples,
            "unique_groups": unique_groups,
            "cross_validation_splits": actual_splits,
            "results": results,
            "honest_assessment_note": (
                "小规模小试中，各模型组差异主要用于检验物理特征加入后的数值数值稳定性与多重共线性风险。"
                "不应把单一小样本上的指标变动夸大为生产级泛化定论。"
            )
        }
