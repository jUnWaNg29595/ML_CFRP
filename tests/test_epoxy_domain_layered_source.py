# -*- coding: utf-8 -*-
"""环氧树脂反应特征的分层数据源契约测试。

分层优先级：窄表优化列 (cp_* / *_resolved) → 宽表文献值 → 结构直算

覆盖：
1. 宽表命中时 EEW/AHEW 与文献值一致（修正结构直算的系统偏差）
2. 无数据源时降级为纯结构口径，且不报错
3. 行数不匹配时安全降级
4. 窄表优先级高于宽表
5. 酸酐双口径（f_stoich=1 / f_network=2）
6. 单线程与多进程结果一致（row_idx 对齐正确）
7. 新增特征存在且数值合理
"""
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

pytest.importorskip("rdkit")

from core.molecular_features import EpoxyDomainFeatureExtractor

STOICH_MODE = "Resin/Hardener (总质量比, R/H)"

# 真实体系：DGEBA + DDS（胺类）、DGEBA + MHHPA（酸酐类）
DGEBA = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
DDS = "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1"
MHHPA = "CC1CCC2C(=O)OC(=O)C2C1"


def _wide(rows):
    """构造行序对齐的宽表。"""
    return pd.DataFrame(rows)


def test_wide_table_supplies_literature_equivalent_weight():
    """宽表提供文献 EEW 时，应直接采用而非结构直算。

    DGEBA 结构 EEW = 340.4/2 = 170.2，但商用树脂文献值通常 187~208
    （含低聚物），两者差异显著，可用于区分数据源是否命中。
    """
    wide = _wide([{
        "resin_1_structure": DGEBA,
        "curing_agent_1_structure": DDS,
        "resin_1_equivalent_weight_g_eq": 190.0,
        "curing_agent_1_equivalent_weight_g_eq": 62.0,
        "resin_1_amount_phr": 100.0,
        "curing_agent_1_amount_phr": 33.0,
    }])
    ext = EpoxyDomainFeatureExtractor(enable_reaction_simulation=False, wide_df=wide)
    feat, valid = ext.extract_features([DGEBA], [DDS], n_jobs=1, stoich_mode=STOICH_MODE)

    assert len(feat) == 1, "应成功提取 1 行"
    assert feat.loc[0, "EEW"] == pytest.approx(190.0), "应使用文献 EEW"
    assert feat.loc[0, "AHEW"] == pytest.approx(62.0), "应使用文献 AHEW"
    assert feat.loc[0, "Physics_Source_Resin"] == "wide_table"
    assert feat.loc[0, "Physics_Source_Hardener"] == "wide_table"


def test_structure_fallback_when_no_source():
    """无任何数据源时降级为结构直算，且特征完整。"""
    ext = EpoxyDomainFeatureExtractor(enable_reaction_simulation=False)
    feat, _ = ext.extract_features([DGEBA], [DDS], n_jobs=1, stoich_mode=STOICH_MODE)

    assert len(feat) == 1
    # 结构直算：MW/f = 340.42/2
    assert feat.loc[0, "EEW"] == pytest.approx(170.2, abs=1.0)
    assert feat.loc[0, "Physics_Source_Resin"] == "structure"
    assert feat.loc[0, "Resin_Functionality"] == 2


def test_row_count_mismatch_degrades_safely():
    """宽表行数与输入不匹配时不得按位置取错值，应整体降级。"""
    wide = _wide([
        {"resin_1_structure": DGEBA, "curing_agent_1_structure": DDS,
         "resin_1_equivalent_weight_g_eq": 999.0},
    ])  # 只有 1 行，但输入 2 条
    ext = EpoxyDomainFeatureExtractor(enable_reaction_simulation=False, wide_df=wide)
    feat, _ = ext.extract_features(
        [DGEBA, DGEBA], [DDS, DDS], n_jobs=1, stoich_mode=STOICH_MODE
    )
    # 不得出现 999（那是错位取值），必须退回结构值 170.2
    assert not np.isclose(feat["EEW"].to_numpy(), 999.0).any()
    assert np.allclose(feat["EEW"].to_numpy(), 170.2, atol=1.0)


def test_narrow_table_takes_priority_over_wide():
    """窄表优化列优先级高于宽表。"""
    wide = _wide([{
        "resin_1_structure": DGEBA, "curing_agent_1_structure": DDS,
        "resin_1_equivalent_weight_g_eq": 190.0,
        "curing_agent_1_equivalent_weight_g_eq": 62.0,
    }])
    narrow = _wide([{
        "resin_1_mw_resolved": 380.0,
        "resin_1_ew_resolved": 205.0,
        "curing_agent_1_mw_resolved": 248.0,
        "curing_agent_1_ew_resolved": 62.0,
        "resin_1_f_stoich": 2.0,
        "curing_agent_1_f_stoich": 4.0,
    }])
    ext = EpoxyDomainFeatureExtractor(
        enable_reaction_simulation=False, wide_df=wide, narrow_df=narrow
    )
    feat, _ = ext.extract_features([DGEBA], [DDS], n_jobs=1, stoich_mode=STOICH_MODE)

    assert feat.loc[0, "EEW"] == pytest.approx(205.0), "窄表应覆盖宽表"
    assert feat.loc[0, "Physics_Source_Resin"] == "narrow_table"


def test_anhydride_dual_functionality():
    """酸酐必须给出两套口径：f_stoich=1（1:1 消耗）、f_network=2（桥接 2 条链）。

    并且 Hardener_Functionality（主字段）必须等于**网络口径**——
    该字段供 Flory 凝胶点 / Mc / 交联密度使用，用 f_stoich=1 代入 (f−2)
    项会得到负交联密度。
    """
    ext = EpoxyDomainFeatureExtractor(enable_reaction_simulation=False)
    feat, _ = ext.extract_features([DGEBA], [MHHPA], n_jobs=1, stoich_mode=STOICH_MODE)

    assert feat.loc[0, "Curer_Type_Anhydride"] == 1
    assert feat.loc[0, "Hardener_Functionality_Stoich"] == pytest.approx(1.0)
    assert feat.loc[0, "Hardener_Functionality_Network"] == pytest.approx(2.0)
    # 主字段已切到网络口径
    assert feat.loc[0, "Hardener_Functionality"] == pytest.approx(2.0)
    assert feat.loc[0, "Hardener_Functionality"] == feat.loc[0, "Hardener_Functionality_Network"]


def test_amine_dual_functionality_identical():
    """胺的两套口径应相同（伯胺 2 氢既消耗 2 环氧也形成 2 个分支）。"""
    ext = EpoxyDomainFeatureExtractor(enable_reaction_simulation=False)
    feat, _ = ext.extract_features([DGEBA], [DDS], n_jobs=1, stoich_mode=STOICH_MODE)

    assert feat.loc[0, "Curer_Type_Amine"] == 1
    fs = feat.loc[0, "Hardener_Functionality_Stoich"]
    fn = feat.loc[0, "Hardener_Functionality_Network"]
    assert fs == pytest.approx(fn)
    assert fs == pytest.approx(4.0), "DDS 两个伯胺 → 4 个活性氢"
    assert feat.loc[0, "Hardener_Functionality"] == pytest.approx(4.0)


def test_anhydride_gel_point_uses_network_functionality():
    """凝胶点必须用网络口径，否则酸酐体系会算出 alpha_gel=1（即“永不凝胶”）。

    用三官能树脂 + 酸酐可区分两种口径：
        f_network=2 → alpha_gel = 1/sqrt((3-1)(2-1)) = 0.707
        f_stoich=1  → 分母 (f-1)=0 → 被 clip 成 1.0（错误地表示“不凝胶”）
    """
    import math

    TGMDA = "C(OC1CO1)c1cc(COC2CO2)cc(COC3CO3)c1"  # 苯三酚三缩水甘油醚（三官能）
    ext = EpoxyDomainFeatureExtractor(enable_reaction_simulation=False)
    feat, _ = ext.extract_features([TGMDA], [MHHPA], n_jobs=1, stoich_mode=STOICH_MODE)

    f_r = feat.loc[0, "Resin_Functionality"]
    f_h = feat.loc[0, "Hardener_Functionality"]
    assert f_r >= 3.0, f"测试前提：树脂应为多官能 (实际 f_r={f_r})"
    assert f_h == pytest.approx(2.0), "酸酐网络口径"

    expected = min(1.0, 1.0 / math.sqrt((f_r - 1.0) * (f_h - 1.0)))
    assert feat.loc[0, "Gel_Point_Conversion"] == pytest.approx(expected)
    # 关键断言：用 f_stoich=1 会得到 1.0（永不凝胶），网络口径必须 < 1
    assert feat.loc[0, "Gel_Point_Conversion"] < 1.0


def test_new_crosslink_features_present_and_sane():
    """新增的物理特征应存在且量级合理。"""
    ext = EpoxyDomainFeatureExtractor(enable_reaction_simulation=False)
    feat, _ = ext.extract_features([DGEBA], [DDS], n_jobs=1, stoich_mode=STOICH_MODE)

    for col in (
        "Hardener_Functionality_Stoich",
        "Hardener_Functionality_Network",
        "Resin_Functionality_Network",
        "W_g_per_epoxy",
        "Epoxy_Conc_mol_m3",
        "Crosslink_Density_Network_mol_m3",
        "Mc_g_mol",
        "Physics_Source_Resin",
        "Physics_Source_Hardener",
    ):
        assert col in feat.columns, f"缺少新增特征 {col}"

    # DGEBA+DDS 等当量体系：ν 应落在 1e3~1e4 mol/m³ 量级
    nu = feat.loc[0, "Crosslink_Density_Network_mol_m3"]
    assert 100.0 < nu < 1.0e4, f"交联密度量级异常: {nu}"
    # Mc 与 ν 互为倒数关系（ρ=1.2e6 g/m³）
    mc = feat.loc[0, "Mc_g_mol"]
    assert mc == pytest.approx(1.2e6 / nu, rel=1e-3)


def test_multiprocess_matches_singleprocess():
    """多进程必须与单线程逐行一致（row_idx 在 chunk 内定位，不能错位）。"""
    n = 40
    wide = _wide([
        {
            "resin_1_structure": DGEBA,
            "curing_agent_1_structure": DDS if i % 2 == 0 else MHHPA,
            "resin_1_equivalent_weight_g_eq": 185.0 + i,
            "curing_agent_1_equivalent_weight_g_eq": 60.0 + i,
        }
        for i in range(n)
    ])
    rs = [DGEBA] * n
    hs = [DDS if i % 2 == 0 else MHHPA for i in range(n)]

    e1 = EpoxyDomainFeatureExtractor(enable_reaction_simulation=False, wide_df=wide)
    f1, vi1 = e1.extract_features(rs, hs, n_jobs=1, stoich_mode=STOICH_MODE)

    e2 = EpoxyDomainFeatureExtractor(enable_reaction_simulation=False, wide_df=wide)
    f2, vi2 = e2.extract_features(rs, hs, n_jobs=2, stoich_mode=STOICH_MODE)

    assert vi1 == vi2, "有效索引顺序应一致"
    assert len(f1) == len(f2)

    f2s = f2.set_index(pd.Index(vi2)).reindex(vi1)
    for col in ("EEW", "AHEW", "Crosslink_Density_Network_mol_m3", "Epoxy_Conc_mol_m3"):
        a = pd.to_numeric(f1[col], errors="coerce").to_numpy()
        b = pd.to_numeric(f2s[col], errors="coerce").to_numpy()
        assert np.allclose(a, b, equal_nan=True), f"{col} 在单线程/多进程间不一致"


def test_missing_hardener_returns_none():
    """缺固化剂结构应返回 None（保持原有行为）。"""
    ext = EpoxyDomainFeatureExtractor(enable_reaction_simulation=False)
    feat, valid = ext.extract_features([DGEBA], [None], n_jobs=1, stoich_mode=STOICH_MODE)
    assert len(feat) == 0
    assert valid == []


# --------------------------------------------------------------------------
# 模型补齐数据页面走的链路：workflow → virtual_screening → extractor
# --------------------------------------------------------------------------
def test_extract_features_from_config_forwards_source_df():
    """virtual_screening 适配器应把 _source_df 透传给环氧提取器。"""
    from core.virtual_screening import extract_features_from_config

    src = pd.DataFrame({
        "resin_1_equivalent_weight_g_eq": [195.0],
        "curing_agent_1_equivalent_weight_g_eq": [81.0],
    })
    cfg = {
        "method": "环氧树脂反应特征",
        "params": {"enable_reaction_simulation": False},
        "_source_df": src,
    }
    feat, err = extract_features_from_config([DGEBA], [MHHPA], cfg)
    assert err is None
    assert feat.loc[0, "EEW"] == pytest.approx(195.0), "应使用数据源中的文献 EEW"
    assert feat.loc[0, "AHEW"] == pytest.approx(81.0)
    assert feat.loc[0, "Physics_Source_Resin"] == "wide_table"


def test_extract_features_from_config_without_source_uses_structure():
    from core.virtual_screening import extract_features_from_config

    cfg = {"method": "环氧树脂反应特征", "params": {"enable_reaction_simulation": False}}
    feat, err = extract_features_from_config([DGEBA], [MHHPA], cfg)
    assert err is None
    assert feat.loc[0, "EEW"] == pytest.approx(170.2, abs=1.0)
    assert feat.loc[0, "Physics_Source_Resin"] == "structure"


def test_extract_features_from_config_rejects_misaligned_source():
    """数据源行数与输入不符时必须丢弃，不得按位置错位取值。"""
    from core.virtual_screening import extract_features_from_config

    src = pd.DataFrame({"resin_1_equivalent_weight_g_eq": [195.0, 196.0]})  # 2 行 vs 1 条
    cfg = {
        "method": "环氧树脂反应特征",
        "params": {"enable_reaction_simulation": False},
        "_source_df": src,
    }
    feat, err = extract_features_from_config([DGEBA], [MHHPA], cfg)
    assert err is None
    assert feat.loc[0, "EEW"] == pytest.approx(170.2, abs=1.0), "应降级为结构口径"


def test_workflow_executor_passes_source_df(monkeypatch):
    """execute_molecular_feature_workflow 应把 data 作为数据源传给步骤。"""
    from core.molecular_feature_workflow import (
        MolecularFeatureWorkflow,
        execute_molecular_feature_workflow,
    )

    captured = {}

    def spy_step(smiles, step, device=None, source_df=None):
        captured["source_df"] = source_df
        return pd.DataFrame({"single_value": [1.0]}), [0], []

    monkeypatch.setattr(
        "core.molecular_feature_workflow.execute_feature_step", spy_step
    )

    data = pd.DataFrame({"smiles": [DGEBA], "resin_1_equivalent_weight_g_eq": [195.0]})
    wf = MolecularFeatureWorkflow.from_dict({
        "schema_version": 2,
        "steps": [{
            "step_id": "single",
            "source_columns": ["smiles"],
            "method": "环氧树脂反应特征",
            "prefix": "single",
            "feature_names": ["single_value"],
        }],
        "merge_order": ["single"],
        "final_feature_names": ["single_value"],
    })

    execute_molecular_feature_workflow(data, wf)
    assert captured["source_df"] is not None, "应默认传入数据源"
    assert "resin_1_equivalent_weight_g_eq" in captured["source_df"].columns

    # use_data_source=False 时应显式关闭
    captured.clear()
    execute_molecular_feature_workflow(data, wf, use_data_source=False)
    assert captured["source_df"] is None


def test_resolver_hardener_functionality_uses_network_scope():
    """auto_feature_resolver（总表查不到特征时的现场计算路径）口径必须一致。"""
    from core.auto_feature_resolver import compute_formulation_feature

    df = pd.DataFrame({
        "resin_1_structure": [DGEBA, DGEBA],
        "curing_agent_1_structure": [MHHPA, DDS],
    })
    hf = compute_formulation_feature(df, "hardener_functionality")
    assert hf.iloc[0] == pytest.approx(2.0), "酸酐网络口径 = 2"
    assert hf.iloc[1] == pytest.approx(4.0), "DDS = 4"

    # 当量重仍用化学计量口径（酸酐 f=1）
    ahew = compute_formulation_feature(df, "ahew")
    assert ahew.iloc[0] == pytest.approx(168.19, abs=0.5), "酸酐 AHEW 用 f_stoich=1"


# --------------------------------------------------------------------------
# 窄表补齐：当量重原位填补 + 元数据列不进表
# --------------------------------------------------------------------------
def _narrow_fixture():
    """两行：一行有实测 EEW/AHEW，一行只有结构（需补齐）。"""
    return pd.DataFrame({
        "resin_1_structure": [DGEBA, DGEBA],
        "curing_agent_1_structure": [DDS, DDS],
        "formulation_resin_total_eew_g_eq": [208.0, np.nan],
        "formulation_hardener_total_ahew_g_eq": [62.0, np.nan],
        "formulation_r_value": [1.0, np.nan],
    })


def test_enrich_fills_eew_in_place_and_keeps_measured_values():
    """实测 EEW/AHEW 必须原样保留，只在空位填补——不得被结构直算覆盖。

    实测依据（受控子集 n=352，仅替换 EEW/AHEW，目标=实测 ν）：
        原表实测值 spearman=+0.236   结构直算 spearman=+0.080
    同一 DGEBA 结构在原表中有 105 个不同 EEW（134~288），反映真实低聚物
    分布；结构直算恒为单体值 170.21。
    """
    from core.component_physics import enrich_narrow_table

    df = _narrow_fixture()
    out = enrich_narrow_table(df)

    # 第 0 行有实测值 → 必须原样保留
    assert out.loc[0, "formulation_resin_total_eew_g_eq"] == pytest.approx(208.0)
    assert out.loc[0, "formulation_hardener_total_ahew_g_eq"] == pytest.approx(62.0)
    # 第 1 行缺失 → 用结构直算填补（DGEBA: 340.42/2 = 170.21）
    assert out.loc[1, "formulation_resin_total_eew_g_eq"] == pytest.approx(170.21, abs=1.0)
    assert out.loc[1, "formulation_hardener_total_ahew_g_eq"] == pytest.approx(62.08, abs=1.0)


def test_enrich_does_not_duplicate_eew_columns():
    """补齐不得另起 cp_eew/cp_ahew 列——那会与原表列 100% 重复。"""
    from core.component_physics import enrich_narrow_table

    out = enrich_narrow_table(_narrow_fixture())
    assert "cp_eew" not in out.columns, "cp_eew 与原表 EEW 重复，应写回原列"
    assert "cp_ahew" not in out.columns
    assert "cp_r_value" not in out.columns, "cp_r_value 与原表 r 值重复"


def test_enrich_output_has_no_string_metadata_columns():
    """元数据列（来源/机制/覆盖度标记）不得进入数据集。"""
    from core.component_physics import enrich_narrow_table, is_metadata_column

    df = _narrow_fixture()
    out = enrich_narrow_table(df)
    new = [c for c in out.columns if c not in df.columns]
    bad = [c for c in new if not pd.api.types.is_numeric_dtype(out[c])
           and not pd.api.types.is_bool_dtype(out[c])]
    assert not bad, f"新增列中不应有字符串列: {bad}"

    # keep_metadata=True 时才保留（供调试）
    dbg = enrich_narrow_table(df, keep_metadata=True)
    assert any(is_metadata_column(c) for c in dbg.columns)


def test_metadata_column_classifier():
    from core.component_physics import is_metadata_column

    for name in ("resin_1_mw_source", "curing_agent_1_ew_source",
                 "resin_1_f_source", "resin_1_mw_trust",
                 "curing_agent_1_mechanism", "cp_mechanism",
                 "cp_coverage", "cp_eew_source", "cp_ahew_source"):
        assert is_metadata_column(name), f"{name} 应判定为元数据列"
    for name in ("resin_1_mw_resolved", "cp_eew", "cp_ahew", "cp_f_r",
                 "formulation_resin_total_eew_g_eq", "cp_epoxy_conc_mol_m3"):
        assert not is_metadata_column(name), f"{name} 是数值特征，不应被剔除"


def test_crosslink_physics_prefers_measured_eew():
    """crosslink_physics 不得用结构直算覆盖实测 EEW。"""
    from core.crosslink_physics import compute_crosslink_features

    df = _narrow_fixture()
    out = compute_crosslink_features(df)
    assert len(out) == 2
    # 有实测值的行，其环氧浓度应与用 208 算的一致，而非 170.21
    r = 1.0
    W_measured = 208.0 + r * 62.0
    W_struct = 170.21 + r * 62.08
    conc = pd.to_numeric(out["xl_epoxy_conc_mol_m3"], errors="coerce")
    assert conc.notna().any(), "应能算出环氧浓度"
    c0 = conc.iloc[0]
    assert abs(c0 - 1.2e6 / W_measured) < abs(c0 - 1.2e6 / W_struct), \
        f"应优先使用实测 EEW（得到 {c0:.1f}）"
    # 第 1 行无实测值 → 结构直算
    c1 = conc.iloc[1]
    assert abs(c1 - 1.2e6 / W_struct) < abs(c1 - 1.2e6 / W_measured)
