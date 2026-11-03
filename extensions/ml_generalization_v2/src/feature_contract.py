"""
特征契约与防静默退化控制模块 (Feature Contract & Anti-Degradation Controller)
核心职责：
1. 定义阶梯特征组契约 (Tiered Feature Contract A, B, C, D)；
2. 彻底堵塞实测漏洞 2 (all_md_columns_missing 静默退化漏洞)：
   - 当请求 B、C、D 组时，必须严格核验该组声明的增量特征列 (incremental features)；
   - 如果某一新增特征组的增量列全部或部分缺失，该组必须明确被阻断并标记为 BLOCKED / NOT_RUN，
     详细记录缺失列清单，绝不允许悄悄缩减列名退化成 A 组旧特征并返回假 PASS！
3. 缺失值防护铁律：
   - 严禁将 MD 列的缺失值用全局中位数或 0 伪造填充为“物理特征”；
   - 必须保持各组间比较样本相同、折叠相同。
"""

import hashlib
import json
from typing import Dict, Any, List, Set, Tuple, Optional
import pandas as pd
import numpy as np

# 标准阶梯特征组定义
CONTRACT_GROUPS_DEF = {
    "A_Baseline": {
        "description": "基础文献与现有快速机理特征基线",
        "incremental_category": "baseline_only"
    },
    "B_Plus_Topology": {
        "description": "基线特征 + 真实交联网络拓扑特征 (加成转化率, 最大渗流团簇, 节点密度, 环秩)",
        "incremental_category": "topology"
    },
    "C_Plus_FreeVolume": {
        "description": "B组 + 真实平衡密度与周期网格 Bondi 自由体积特征",
        "incremental_category": "free_volume"
    },
    "D_Plus_Interactions": {
        "description": "C组 + 真实范德华/长程静电能量密度与共价氢键特征",
        "incremental_category": "interactions"
    }
}

class FeatureContractValidator:
    def __init__(self):
        pass

    def validate_group_contract(
        self,
        df_columns: List[str],
        grp_name: str,
        declared_base_cols: List[str],
        declared_incremental_cols: List[str]
    ) -> Dict[str, Any]:
        """
        审核单组特征契约完整性：
        :param df_columns: 数据集中实际存在的列名列表
        :param grp_name: 特征组名称 (如 'B_Plus_Topology')
        :param declared_base_cols: 基础依赖特征列
        :param declared_incremental_cols: 该组声明的核心增量特征列
        :return: 契约审核结果字典
        """
        existing_cols_set = set(df_columns)

        # 1. 检查基础特征列
        missing_base = [c for c in declared_base_cols if c not in existing_cols_set]

        # 2. 检查增量特征列 (防御静默退化核心！)
        missing_incremental = [c for c in declared_incremental_cols if c not in existing_cols_set]
        present_incremental = [c for c in declared_incremental_cols if c in existing_cols_set]

        # 如果是 A 组基线
        if grp_name == "A_Baseline":
            if missing_base:
                return {
                    "group_name": grp_name,
                    "contract_status": "BLOCKED",
                    "reason": f"基线特征存在缺失: {missing_base}",
                    "active_features": [],
                    "has_valid_increment": False
                }
            active_cols = [c for c in declared_base_cols if c in existing_cols_set]
            return {
                "group_name": grp_name,
                "contract_status": "PASS",
                "reason": "基线特征契约核验通过",
                "active_features": active_cols,
                "has_valid_increment": True
            }

        # 如果是 B, C, D 增量组
        # 铁律：增量特征若完全缺失，必须判定为 BLOCKED，严禁静默退化！
        if not present_incremental:
            return {
                "group_name": grp_name,
                "contract_status": "BLOCKED",
                "reason": (
                    f"特征组 {grp_name} 声明的所有增量特征全部缺失: {missing_incremental}！"
                    f"严禁静默退化为旧基线并伪造模型评估指标。"
                ),
                "active_features": [],
                "missing_columns": missing_incremental,
                "has_valid_increment": False
            }

        # 增量特征若部分缺失，同样拒绝放行（要求严格契约）
        if missing_incremental:
            return {
                "group_name": grp_name,
                "contract_status": "BLOCKED",
                "reason": f"特征组 {grp_name} 缺少部分必要增量特征: {missing_incremental}",
                "active_features": [],
                "missing_columns": missing_incremental,
                "has_valid_increment": False
            }

        if missing_base:
            return {
                "group_name": grp_name,
                "contract_status": "BLOCKED",
                "reason": f"特征组 {grp_name} 缺少基础依赖特征: {missing_base}",
                "active_features": [],
                "missing_columns": missing_base,
                "has_valid_increment": False
            }

        # 全部通过
        all_active = declared_base_cols + declared_incremental_cols
        return {
            "group_name": grp_name,
            "contract_status": "PASS",
            "reason": f"特征组 {grp_name} 包含完整声明的基础与增量特征",
            "active_features": all_active,
            "missing_columns": [],
            "has_valid_increment": True
        }

    def compute_features_sha256(self, df: pd.DataFrame, feature_cols: List[str]) -> str:
        """计算特定特征子集的数据内容 SHA256 哈希"""
        if df.empty or not feature_cols:
            return ""
        sub = df[feature_cols].copy()
        # 排序并转为二进制
        data_bytes = sub.to_csv(index=False).encode("utf-8")
        return hashlib.sha256(data_bytes).hexdigest()
