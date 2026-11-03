"""针对模型组分与输入组分数量不匹配的自适应回归测试。

背景与业务痛点
--------------
真实工业模型（如基于宽表训练的 TabPFN、XGBoost 多槽位模型）通常在训练时声明了
多个树脂、固化剂或助剂槽位（例如 resin_1, resin_2, resin_3, curing_agent_1,
curing_agent_2, small_additive_1 等）。

但在单点预测或虚拟筛选时，用户通常只提供单树脂（resin_1）+ 单固化剂（curing_agent_1）。
旧架构下存在四大连环断点：
  1. KeyError: 'workflow step requires missing source columns: resin_2_structure'
  2. 预测门禁拦截：'分子源列不能为空：resin_2_structure'
  3. 分子特征空帧全为 NaN → 虚拟筛选有效行判定被误杀清空全部候选
  4. 辅助 phr 缺失导致 '缺少必填手工特征：resin_2_amount_phr'

本测试套件锁定：
  - 主/辅槽位正确分级：序号 >= 2 的槽位与添加剂槽位默认可选，不强求输入
  - 空辅助槽位特征安全填补 0.0，不产生 NaN 穿透
  - 真实 10 步 381 维多槽位模型（TabPFN）在纯单树脂单固化剂输入下端到端预测成功
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.molecular_feature_workflow import (
    MolecularFeatureWorkflow,
    execute_molecular_feature_workflow,
    is_auxiliary_component_column,
    materialize_workflow_source_columns,
)
from core.portal_prediction import (
    _normalize_portal_frame,
    _source_is_required,
    is_auxiliary_phr_column,
)


# ---------------------------------------------------------------------------
# 1. 槽位主辅分类与 PHR 判定规则
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "column,expected",
    [
        ("resin_1_structure", False),
        ("resin_2_structure", True),
        ("resin_3_structure", True),
        ("resin_smiles", False),
        ("resin_smiles_1", False),
        ("resin_smiles_2", True),
        ("curing_agent_1_structure", False),
        ("curing_agent_2_structure", True),
        ("curing_agent_3_structure", True),
        ("hardener_smiles", False),
        ("hardener_smiles_2", True),
        ("small_additive_1_structure", True),
        ("small_additive_2_structure", True),
        ("accelerator_1_structure", True),
        ("initiator_1_structure", True),
        ("reactive_diluent_1_structure", True),
        ("filler_1_structure", True),
    ],
)
def test_is_auxiliary_component_column(column, expected):
    assert is_auxiliary_component_column(column) is expected


@pytest.mark.parametrize(
    "column,expected",
    [
        ("resin_1_amount_phr", False),
        ("resin_2_amount_phr", True),
        ("resin_3_amount_phr", True),
        ("curing_agent_1_amount_phr", False),
        ("curing_agent_2_amount_phr", True),
        ("curing_agent_3_amount_phr", True),
        ("small_additive_1_amount_phr", True),
        ("reactive_toughener_total_phr", True),
        ("accelerator_total_phr", True),
        ("phr", False),
        ("curing_temperature_c", False),
    ],
)
def test_is_auxiliary_phr_column(column, expected):
    assert is_auxiliary_phr_column(column) is expected


def test_source_is_required_handles_auxiliary_slots():
    """主槽位必填，辅助槽位默认可选。"""
    assert _source_is_required({"column": "resin_1_structure"}) is True
    assert _source_is_required({"column": "curing_agent_1_structure"}) is True
    assert _source_is_required({"column": "resin_2_structure"}) is False
    assert _source_is_required({"column": "curing_agent_2_structure"}) is False
    assert _source_is_required({"column": "small_additive_1_structure"}) is False

    # 若契约显式声明了 required 或 nullable，依然尊重显式声明
    assert _source_is_required({"column": "resin_2_structure", "required": True}) is True
    assert _source_is_required({"column": "resin_1_structure", "nullable": True}) is False


# ---------------------------------------------------------------------------
# 2. 结构物化展开与缺失源列自适应
# ---------------------------------------------------------------------------
def test_materialize_workflow_source_columns_pads_auxiliary_columns():
    """物化展开时支持单树脂扩展到多槽位，未匹配到的槽位以 None 占位。"""
    cand = pd.DataFrame({
        "resin_smiles": ["CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"],
        "hardener_smiles": ["Nc1ccc(Cc2ccc(N)cc2)cc1"],
    })
    res = materialize_workflow_source_columns(
        cand,
        resin_columns=["resin_1_structure", "resin_2_structure", "resin_3_structure"],
        hardener_columns=["curing_agent_1_structure", "curing_agent_2_structure"],
        auxiliary_columns=["small_additive_1_structure"],
    )
    assert res.loc[0, "resin_1_structure"] == "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
    assert res.loc[0, "resin_2_structure"] is None
    assert res.loc[0, "resin_3_structure"] is None
    assert res.loc[0, "curing_agent_1_structure"] == "Nc1ccc(Cc2ccc(N)cc2)cc1"
    assert res.loc[0, "curing_agent_2_structure"] is None
    assert res.loc[0, "small_additive_1_structure"] is None


def test_prepare_step_inputs_allows_missing_auxiliary_columns():
    """当数据完全缺少辅助槽位列时，不抛 KeyError，平滑返回 None 样本。"""
    from core.molecular_feature_workflow import prepare_step_inputs

    data = pd.DataFrame({"resin_1_structure": ["CCO"]})
    step = {
        "step_id": "step_r2",
        "method": "标准版",
        "source_columns": ["resin_2_structure"],
    }
    # 不应抛出 KeyError
    smiles, valid_indices, metadata = prepare_step_inputs(data, step)
    assert smiles == [None]
    assert valid_indices == []


# ---------------------------------------------------------------------------
# 3. 工作流执行特征补零与安全填充
# ---------------------------------------------------------------------------
def test_execute_workflow_neutral_fills_absent_auxiliary_features():
    """辅助槽位无分子输入时，其所有分子特征安全补 0.0，不产生 NaN。"""
    wf_dict = {
        "schema_version": 1,
        "workflow_hash": "test_wf_aux",
        "merge_order": ["step_r1", "step_r2"],
        "steps": [
            {
                "step_id": "step_r1",
                "method": "标准版 (RDKit 2D/3D 基础描述符)",
                "role": "resin",
                "source_columns": ["resin_1_structure"],
                "prefix": "resin_1_",
                "feature_names": ["resin_1_MolWt"],
                "params": {},
            },
            {
                "step_id": "step_r2",
                "method": "标准版 (RDKit 2D/3D 基础描述符)",
                "role": "resin",
                "source_columns": ["resin_2_structure"],
                "prefix": "resin_2_",
                "feature_names": ["resin_2_MolWt"],
                "params": {},
            },
        ],
        "final_feature_names": ["resin_1_MolWt", "resin_2_MolWt"],
    }
    wf = MolecularFeatureWorkflow.from_dict(wf_dict)
    data = pd.DataFrame({"resin_1_structure": ["c1ccc(OCC2CO2)cc1"]})

    res = execute_molecular_feature_workflow(data, wf)
    assert len(res.features) == 1
    assert "resin_1_MolWt" in res.features.columns
    assert "resin_2_MolWt" in res.features.columns
    assert np.isfinite(res.features["resin_1_MolWt"].iloc[0])
    assert res.features["resin_2_MolWt"].iloc[0] == 0.0
    assert not res.features.isna().any().any()


# ---------------------------------------------------------------------------
# 4. 门户输入规范化（自动补全 0.0 phr）
# ---------------------------------------------------------------------------
def test_normalize_portal_frame_auto_populates_auxiliary_phr():
    contract = {
        "feature_cols": ["resin_1_amount_phr", "resin_2_amount_phr", "curing_agent_2_amount_phr"],
        "manual_input_feature_cols": ["resin_1_amount_phr", "resin_2_amount_phr", "curing_agent_2_amount_phr"],
    }
    frame = pd.DataFrame({"resin_1_amount_phr": [100.0]})
    norm = _normalize_portal_frame(frame, contract, None)
    assert norm.loc[0, "resin_1_amount_phr"] == 100.0
    assert norm.loc[0, "resin_2_amount_phr"] == 0.0
    assert norm.loc[0, "curing_agent_2_amount_phr"] == 0.0


# ---------------------------------------------------------------------------
# 5. 真实模型端到端回归测试
# ---------------------------------------------------------------------------
def test_real_multislot_tabpfn_model_predicts_with_single_component_input():
    """真实 381 维多槽位 TabPFN 模型，纯单树脂单固化剂输入端到端成功预测。"""
    import os
    from core.model_io import load_model_artifact_bytes
    from core.virtual_screening import (
        build_feature_matrix,
        predict_with_model,
        resolve_molecular_feature_workflow,
        resolve_workflow_source_columns_by_role,
    )
    from core.screening_feature_derivation import derive_computable_candidate_columns
    from core.computable_features import apply_schedule_derivations

    path = "results/training_runs/20260923_032815_TabPFN/model_TabPFN.joblib"
    if not os.path.exists(path):
        pytest.skip("测试模型不存在")

    art = load_model_artifact_bytes(open(path, "rb").read())
    fc = [str(c) for c in art["feature_cols"]]
    wf_dict = (art.get("extra") or {}).get("molecular_feature_workflow")
    wf = resolve_molecular_feature_workflow({"molecular_feature_workflow": wf_dict}, model_feature_cols=fc)
    resin_cols, hard_cols = resolve_workflow_source_columns_by_role(wf, {})

    # 纯单树脂 + 单固化剂输入
    cand = pd.DataFrame({
        "resin_smiles": ["CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"],
        "hardener_smiles": ["Nc1ccc(Cc2ccc(N)cc2)cc1"],
        "resin_1_amount_phr": [100.0],
        "curing_agent_1_amount_phr": [33.0],
        "cure_schedule": ["130 °C/2 h + 180 °C/2 h"],
    })

    cand = materialize_workflow_source_columns(cand, resin_columns=resin_cols, hardener_columns=hard_cols)
    cand, _ = apply_schedule_derivations(cand, feature_cols=fc, schedule_column="cure_schedule")
    cand, _ = derive_computable_candidate_columns(cand, feature_cols=fc)

    res = execute_molecular_feature_workflow(cand, wf)
    base_row = pd.Series({c: 0.0 for c in fc})
    X = build_feature_matrix(fc, res.features, base_row=base_row)

    preds = predict_with_model(art["model"], X, pipeline=art["pipeline"])
    assert len(preds) == 1
    val = float(preds[0])
    assert np.isfinite(val)
    assert val > 0.0, f"预测值异常：{val}"
