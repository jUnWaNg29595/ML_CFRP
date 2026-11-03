"""
强制训练准入与特征契约测试套件 (Test Admission & Feature Contract)
严格覆盖独立审计发现的两大核心漏洞与对抗性测试 (源自 ml_adversarial.json 与 audit.py)：
1. no_origin_no_receipts_no_hashes: 仅设置 eligible=True 和 gate=PASS，但缺少 data_origin、network_id、计算证据的 fixture，正式 MD 准入必须坚决拒绝 (admitted=False, ADMISSION_REJECTED)；
2. missing_physical_values: MD 物理特征列存在缺失值时，严格禁止中位数插补，该组必须坚决标记为 BLOCKED；
3. inf_physical_values: MD 物理特征列存在 Inf/-Inf 时，坚决标记为 BLOCKED；
4. fake_pass_gate: scientific_gate_status 虽然写着 PASS，但缺少合法 calculation_receipt_hash 时坚决拒绝；
5. duplicate_seed_expansion: 同一配方在同一 seed 下重复 join 导致行数虚假膨胀，必须被拦截；
6. conflicting_protocol: 混入未知或冲突的 protocol_id 时坚决拒绝；
7. all_md_columns_missing: 缺失全部 MD 列时，B/C/D 组必须明确阻断 (BLOCKED)，严禁静默退化成 A 组旧特征；
8. 严格布尔解析与路径模式隔离。
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

from src.admission import (
    strict_parse_bool,
    TrainingAdmissionValidator,
    PATH_BASELINE_LITERATURE_ONLY,
    PATH_FORMAL_MD_ABLATION,
    PATH_DIAGNOSTIC_FIXTURE_TEST
)
from src.feature_contract import FeatureContractValidator
from src.safe_comparison import SafeFeatureComparisonEngine

def test_audit_adversarial_case_1_no_origin_no_receipts_no_hashes():
    """
    独立验收反例 1 (ml_adversarial.json: no_origin_no_receipts_no_hashes):
    在完全没有 data_origin、原始文件、执行回执、哈希和 protocol 信息的 fixture 中，
    仅设置 eligible=True 和 gate=PASS，正式 MD 入口必须坚决拒绝 (admitted=False, ADMISSION_REJECTED)！
    """
    df_no_evidence = pd.DataFrame({
        'y': [1., 2., 3., 4., 5., 6.],
        'group': ['a', 'a', 'b', 'b', 'c', 'c'],
        'x': [1., 3., 2., 4., 5., 6.],
        'md': [.3, .2, .4, .5, .6, .7],
        'eligible_for_training': [True] * 6,
        'scientific_gate_status': ['PASS'] * 6
        # 故意不提供 data_origin, network_id, formulation_id
    })

    # 1. 验证准入审核器拒绝
    admit_validator = TrainingAdmissionValidator(path_mode=PATH_FORMAL_MD_ABLATION)
    is_admitted, _, audit_report = admit_validator.validate_dataset_admission(df_no_evidence)
    assert is_admitted is False
    assert audit_report["status"] == "ADMISSION_REJECTED"
    assert "missing_data_origin_evidence" in audit_report["rejection_reasons"] or "missing_network_id_column" in audit_report["rejection_reasons"]

    # 2. 验证端到端评估引擎阻断
    engine = SafeFeatureComparisonEngine(path_mode=PATH_FORMAL_MD_ABLATION)
    res = engine.evaluate_feature_groups(
        df=df_no_evidence,
        target_col='y',
        group_col='group',
        baseline_feature_cols=['x'],
        topology_cols=['md'],
        free_volume_cols=[],
        interaction_cols=[],
        n_splits=3
    )
    assert res["status"] == "ADMISSION_REJECTED"
    assert res["sample_count"] == 0

def test_audit_adversarial_case_2_missing_physical_values_blocked():
    """
    独立验收反例 2 (ml_adversarial.json: missing_physical_values):
    在物理特征存在缺失值 (NaN) 时，严格禁止普通中位数插补掩盖真实缺失！
    B_Plus_Topology 必须被坚决标记为 BLOCKED！
    """
    df_with_nans = pd.DataFrame({
        'y': [1., 2., 3., 4., 5., 6.],
        'group': ['a', 'a', 'b', 'b', 'c', 'c'],
        'x': [1., 3., 2., 4., 5., 6.],
        'md': [.3, np.nan, .4, .5, .6, .7],  # 物理值缺失！
        'eligible_for_training': [True] * 6,
        'scientific_gate_status': ['PASS'] * 6,
        'data_origin': ['REAL_MD_SIMULATION'] * 6,
        'network_id': ['net_1', 'net_2', 'net_3', 'net_4', 'net_5', 'net_6'],
        'formulation_id': ['f1', 'f1', 'f2', 'f2', 'f3', 'f3']
    })

    engine = SafeFeatureComparisonEngine(path_mode=PATH_FORMAL_MD_ABLATION)
    res = engine.evaluate_feature_groups(
        df=df_with_nans,
        target_col='y',
        group_col='group',
        baseline_feature_cols=['x'],
        topology_cols=['md'],
        free_volume_cols=[],
        interaction_cols=[],
        n_splits=3
    )

    assert res["admission_audit"]["status"] == "PASS"
    assert res["results"]["A_Baseline"]["status"] == "PASS"
    # B_Plus_Topology 包含物理缺失列，必须为 BLOCKED，严禁静默中位数插补并输出虚假 MAE/R2！
    assert res["results"]["B_Plus_Topology"]["status"] == "BLOCKED"
    assert "缺失值" in res["results"]["B_Plus_Topology"]["reason"]
    assert "md" in res["results"]["B_Plus_Topology"]["missing_columns"]

def test_audit_adversarial_case_3_inf_physical_values_blocked():
    """
    反例 3: 物理特征包含 Inf / -Inf 异常值时，必须阻断为 BLOCKED，严禁插补！
    """
    df_with_inf = pd.DataFrame({
        'y': [1., 2., 3., 4., 5., 6.],
        'group': ['a', 'a', 'b', 'b', 'c', 'c'],
        'x': [1., 3., 2., 4., 5., 6.],
        'md': [.3, np.inf, .4, .5, .6, .7],  # 出现未收敛 Inf！
        'eligible_for_training': [True] * 6,
        'scientific_gate_status': ['PASS'] * 6,
        'data_origin': ['REAL_MD_SIMULATION'] * 6,
        'network_id': ['net_1', 'net_2', 'net_3', 'net_4', 'net_5', 'net_6'],
        'formulation_id': ['f1', 'f1', 'f2', 'f2', 'f3', 'f3']
    })

    engine = SafeFeatureComparisonEngine(path_mode=PATH_FORMAL_MD_ABLATION)
    res = engine.evaluate_feature_groups(
        df=df_with_inf,
        target_col='y',
        group_col='group',
        baseline_feature_cols=['x'],
        topology_cols=['md'],
        free_volume_cols=[],
        interaction_cols=[],
        n_splits=3
    )

    assert res["results"]["B_Plus_Topology"]["status"] == "BLOCKED"
    assert "Inf" in res["results"]["B_Plus_Topology"]["reason"] or "缺失值" in res["results"]["B_Plus_Topology"]["reason"]

def test_audit_adversarial_case_4_fake_pass_without_hash_blocked():
    """
    反例 4: 假 PASS 门禁 —— 伪造 scientific_gate_status='PASS' 但 calculation_receipt_hash 缺失或伪造
    """
    df_fake_pass = pd.DataFrame({
        'y': [1., 2., 3., 4., 5., 6.],
        'group': ['a', 'a', 'b', 'b', 'c', 'c'],
        'x': [1., 3., 2., 4., 5., 6.],
        'md': [.3, .2, .4, .5, .6, .7],
        'eligible_for_training': [True] * 6,
        'scientific_gate_status': ['PASS'] * 6,
        'data_origin': ['REAL_MD_SIMULATION'] * 6,
        'network_id': ['net_1', 'net_2', 'net_3', 'net_4', 'net_5', 'net_6'],
        'formulation_id': ['f1', 'f1', 'f2', 'f2', 'f3', 'f3'],
        'calculation_receipt_hash': ['bad_hash', '', None, 'short', 'fake', 'none']  # 非法哈希
    })

    validator = TrainingAdmissionValidator(path_mode=PATH_FORMAL_MD_ABLATION)
    is_admitted, _, audit_report = validator.validate_dataset_admission(df_fake_pass)
    assert is_admitted is False
    assert audit_report["status"] == "ADMISSION_REJECTED"
    assert "missing_or_invalid_receipt_hash" in audit_report["rejection_reasons"]

def test_audit_adversarial_case_5_duplicate_seed_expansion_blocked():
    """
    反例 5: 多 seed 重复 join 导致单配方行数膨胀
    """
    df_dup_seed = pd.DataFrame({
        'y': [1., 2., 3., 4., 5., 6.],
        'group': ['a', 'a', 'b', 'b', 'c', 'c'],
        'x': [1., 3., 2., 4., 5., 6.],
        'md': [.3, .2, .4, .5, .6, .7],
        'eligible_for_training': [True] * 6,
        'scientific_gate_status': ['PASS'] * 6,
        'data_origin': ['REAL_MD_SIMULATION'] * 6,
        'network_id': ['net_1', 'net_2', 'net_3', 'net_4', 'net_5', 'net_6'],
        'formulation_id': ['f1', 'f1', 'f2', 'f2', 'f3', 'f3'],
        'seed': [1, 1, 1, 1, 1, 1]  # 在同一配方 f1 下出现两个 seed=1 的重复行！
    })

    validator = TrainingAdmissionValidator(path_mode=PATH_FORMAL_MD_ABLATION)
    is_admitted, _, audit_report = validator.validate_dataset_admission(df_dup_seed)
    assert is_admitted is False
    assert audit_report["status"] == "ADMISSION_REJECTED"
    assert "duplicate_formulation_seed_expansion" in audit_report["rejection_reasons"]

def test_audit_adversarial_case_6_conflicting_protocol_blocked():
    """
    反例 6: 混入未知或冲突的 protocol_id
    """
    df_conflicting_proto = pd.DataFrame({
        'y': [1., 2., 3., 4., 5., 6.],
        'group': ['a', 'a', 'b', 'b', 'c', 'c'],
        'x': [1., 3., 2., 4., 5., 6.],
        'md': [.3, .2, .4, .5, .6, .7],
        'eligible_for_training': [True] * 6,
        'scientific_gate_status': ['PASS'] * 6,
        'data_origin': ['REAL_MD_SIMULATION'] * 6,
        'network_id': ['net_1', 'net_2', 'net_3', 'net_4', 'net_5', 'net_6'],
        'formulation_id': ['f1', 'f1', 'f2', 'f2', 'f3', 'f3'],
        'protocol_id': ['UNKNOWN_PROTOCOL_123'] * 6  # 未认证协议！
    })

    validator = TrainingAdmissionValidator(path_mode=PATH_FORMAL_MD_ABLATION)
    is_admitted, _, audit_report = validator.validate_dataset_admission(df_conflicting_proto)
    assert is_admitted is False
    assert audit_report["status"] == "ADMISSION_REJECTED"
    assert "unrecognized_or_conflicting_protocol" in audit_report["rejection_reasons"]

def test_audit_adversarial_case_7_all_md_columns_missing():
    """
    当请求的 MD 列完全不存在于数据集中时，B/C/D 组必须明确为 BLOCKED，严禁静默退化！
    """
    df_clean = pd.DataFrame({
        'y': [1., 2., 3., 4., 5., 6.],
        'group': ['a', 'a', 'b', 'b', 'c', 'c'],
        'x': [1., 3., 2., 4., 5., 6.],
        'eligible_for_training': [True] * 6,
        'scientific_gate_status': ['PASS'] * 6,
        'data_origin': ['REAL_MD_SIMULATION'] * 6,
        'network_id': ['net_1', 'net_2', 'net_3', 'net_4', 'net_5', 'net_6'],
        'formulation_id': ['f1', 'f1', 'f2', 'f2', 'f3', 'f3']
    })

    engine = SafeFeatureComparisonEngine(path_mode=PATH_FORMAL_MD_ABLATION)
    res = engine.evaluate_feature_groups(
        df=df_clean,
        target_col='y',
        group_col='group',
        baseline_feature_cols=['x'],
        topology_cols=['non_existent_md_col'],
        free_volume_cols=[],
        interaction_cols=[],
        n_splits=3
    )

    assert res["results"]["A_Baseline"]["status"] == "PASS"
    assert res["results"]["B_Plus_Topology"]["status"] == "BLOCKED"
    assert res["results"]["C_Plus_FreeVolume"]["status"] == "BLOCKED"
    assert res["results"]["D_Plus_Interactions"]["status"] == "BLOCKED"

def test_string_false_bool_parsing():
    """核验字符串 'False' 绝对不能被解析为 True"""
    assert strict_parse_bool(True) is True
    assert strict_parse_bool(False) is False
    assert strict_parse_bool("True") is True
    assert strict_parse_bool("true") is True
    assert strict_parse_bool("1") is True

    # 关键防守：字符串 'False' 绝对不能被 bool() 解释为 True！
    assert strict_parse_bool("False") is False
    assert strict_parse_bool("false") is False
    assert strict_parse_bool("0") is False
    assert strict_parse_bool("None") is False
    assert strict_parse_bool("") is False
    assert strict_parse_bool(None) is False
    assert strict_parse_bool(np.nan) is False

def test_path_modes_isolation():
    """路径模式隔离核验"""
    df_lit = pd.DataFrame({
        'target': [100.0, 120.0, 110.0, 130.0],
        'group': ['p1', 'p1', 'p2', 'p2'],
        'feat_lit': [1.0, 2.0, 1.5, 2.5]
    })

    # 1. 仅文献基线路径：无需 MD 证据即可通过
    engine_lit = SafeFeatureComparisonEngine(path_mode=PATH_BASELINE_LITERATURE_ONLY)
    res_lit = engine_lit.evaluate_feature_groups(
        df=df_lit,
        target_col='target',
        group_col='group',
        baseline_feature_cols=['feat_lit'],
        topology_cols=[],
        free_volume_cols=[],
        interaction_cols=[],
        n_splits=2
    )
    assert res_lit['status'] == 'PASS'
    assert res_lit['sample_count'] == 4

    # 2. 正式 MD 消融路径：缺少 MD 证据链直接拒绝
    engine_md = SafeFeatureComparisonEngine(path_mode=PATH_FORMAL_MD_ABLATION)
    res_md = engine_md.evaluate_feature_groups(
        df=df_lit,
        target_col='target',
        group_col='group',
        baseline_feature_cols=['feat_lit'],
        topology_cols=['md_top'],
        free_volume_cols=[],
        interaction_cols=[],
        n_splits=2
    )
    assert res_md['status'] == 'ADMISSION_REJECTED'
