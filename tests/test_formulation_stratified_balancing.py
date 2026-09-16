import numpy as np
import pandas as pd
import pytest

from core.data_processor import AdvancedDataCleaner


def test_formulation_stratified_basic_within_group():
    # 模拟数据：E51+DDS 有 80 条，E51+DDM 有 25 条，RareResin 有 5 条
    n_dds, n_ddm, n_rare = 80, 25, 5
    df = pd.DataFrame({
        "resin": ["E51"] * n_dds + ["E51"] * n_ddm + ["RareResin"] * n_rare,
        "hardener": ["DDS"] * n_dds + ["DDM"] * n_ddm + ["RareHardener"] * n_rare,
        "Tg": np.concatenate([
            np.linspace(120, 220, n_dds),
            np.linspace(100, 160, n_ddm),
            np.linspace(180, 240, n_rare),
        ]),
        "id": range(n_dds + n_ddm + n_rare),
    })

    cleaner = AdvancedDataCleaner(df)
    cleaned, stats = cleaner.balance_formulation_stratified(
        group_cols=["resin", "hardener"],
        max_samples_per_group=15,
        target_col="Tg",
        n_bins=5,
        stratify_mode="within_group",
        random_state=42,
    )

    # E51+DDS 应被限制为 15，E51+DDM 为 15，RareResin 保留 5，总计 35
    assert len(cleaned) == 15 + 15 + 5
    assert stats["removed_rows"] == (80 + 25 + 5) - 35
    assert stats["n_groups_before"] == 3
    assert stats["n_groups_after"] == 3
    assert stats["max_group_count_after"] == 15

    # 验证 DDS 内部各个分箱的代表性（全温区覆盖）
    dds_subset = cleaned[cleaned["hardener"] == "DDS"]
    assert len(dds_subset) == 15
    # 最低温和最高温样本都在其中
    assert dds_subset["Tg"].min() < 135
    assert dds_subset["Tg"].max() > 200


def test_formulation_stratified_by_target_with_group_cap():
    # 模式 B：按目标分箱平衡，同时限制单一配方在每个分箱中的上限
    n_dds, n_ddm, n_rare = 60, 30, 10
    df = pd.DataFrame({
        "resin": ["E51"] * n_dds + ["E51"] * n_ddm + ["RareResin"] * n_rare,
        "hardener": ["DDS"] * n_dds + ["DDM"] * n_ddm + ["RareHardener"] * n_rare,
        "class_label": ["High"] * 40 + ["Medium"] * 20 + ["Low"] * 30 + ["High"] * 10,
        "feat": range(100),
    })

    cleaner = AdvancedDataCleaner(df)
    cleaned, stats = cleaner.balance_formulation_stratified(
        group_cols=["resin", "hardener"],
        target_col="class_label",
        stratify_mode="by_target_with_group_cap",
        max_per_group_per_bin=8,  # 每类单一配方最多 8 个
        max_samples_per_target_class=20,  # 每个类别总样本最多 20 个
        random_state=42,
    )

    # 检查每个类别中的单一配方数量均不超过 8
    grouped = cleaned.groupby(["class_label", "resin", "hardener"]).size()
    assert (grouped <= 8).all()

    # 检查每个类别的总数均不超过 20
    class_counts = cleaned["class_label"].value_counts()
    assert (class_counts <= 20).all()


def test_formulation_stratified_reproducibility():
    df = pd.DataFrame({
        "resin": ["E51"] * 50 + ["CYD128"] * 20,
        "hardener": ["DDS"] * 50 + ["DDM"] * 20,
        "Tg": np.linspace(100, 200, 70),
    })

    cleaner1 = AdvancedDataCleaner(df)
    res1, _ = cleaner1.balance_formulation_stratified(
        group_cols=["resin", "hardener"],
        max_samples_per_group=10,
        target_col="Tg",
        random_state=123,
    )

    cleaner2 = AdvancedDataCleaner(df)
    res2, _ = cleaner2.balance_formulation_stratified(
        group_cols=["resin", "hardener"],
        max_samples_per_group=10,
        target_col="Tg",
        random_state=123,
    )

    pd.testing.assert_frame_equal(res1, res2)


def test_formulation_stratified_invalid_inputs():
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    cleaner = AdvancedDataCleaner(df)

    with pytest.raises(ValueError, match="指定的配方分组列均不在数据集中"):
        cleaner.balance_formulation_stratified(group_cols=["non_existent"])

    with pytest.raises(ValueError, match="必须指定有效的 target_col"):
        cleaner.balance_formulation_stratified(
            group_cols=["A"],
            stratify_mode="by_target_with_group_cap",
            target_col=None,
        )


def test_combo_protection_rescues_rare_combination():
    """单列(仅固化剂)平衡 + 组合多样性保护：
    高频固化剂 DDS 内部包含大量 E51 与少量 RareResin；RareResin 只与 DDS 搭配出现。
    无保护时随机抽删可能把 RareResin 组合整体清除；开启保护后必须至少保留 min_samples_per_combo 条。"""
    n_e51_dds, n_rare_dds, n_e51_ddm = 60, 20, 10
    df = pd.DataFrame({
        "resin": ["E51"] * n_e51_dds + ["RareResin"] * n_rare_dds + ["E51"] * n_e51_ddm,
        "hardener": ["DDS"] * (n_e51_dds + n_rare_dds) + ["DDM"] * n_e51_ddm,
        "Tg": np.concatenate([
            np.linspace(120, 220, n_e51_dds),
            np.linspace(180, 240, n_rare_dds),
            np.linspace(100, 150, n_e51_ddm),
        ]),
        "id": range(n_e51_dds + n_rare_dds + n_e51_ddm),
    })

    cleaner = AdvancedDataCleaner(df)
    cleaned, stats = cleaner.balance_formulation_stratified(
        group_cols=["hardener"],  # 仅固化剂
        max_samples_per_group=15,
        target_col="Tg",
        n_bins=5,
        stratify_mode="within_group",
        random_state=42,
        protect_combo_cols=["resin"],
        min_samples_per_combo=1,
    )

    # 保护已生效：RareResin 组合仅存在于被削减的 DDS 组中，属于风险组合并获预留
    cp = stats["combo_protection"]
    assert cp["enabled"] is True
    assert cp["n_at_risk_combos"] == 1  # 仅 RareResin；E51 在未削减的 DDM 组中有完整保留
    assert "RareResin" in cp["protected_combos"]
    assert cp["reserved_samples"] >= 1

    # 名额不超标：DDS 组总量仍为 cap=15，DDM 组完整保留
    dds_kept = cleaned[cleaned["hardener"] == "DDS"]
    assert len(dds_kept) == 15
    assert len(cleaned[cleaned["hardener"] == "DDM"]) == n_e51_ddm
    # 关键断言：稀有组合未被误伤灭绝
    assert len(cleaned[cleaned["resin"] == "RareResin"]) >= 1

    # 同种子可复现
    cleaner2 = AdvancedDataCleaner(df)
    cleaned2, _ = cleaner2.balance_formulation_stratified(
        group_cols=["hardener"],
        max_samples_per_group=15,
        target_col="Tg",
        n_bins=5,
        stratify_mode="within_group",
        random_state=42,
        protect_combo_cols=["resin"],
        min_samples_per_combo=1,
    )
    pd.testing.assert_frame_equal(cleaned, cleaned2)


def test_combo_protection_budget_shortfall_recorded():
    """削减名额不足以覆盖全部风险组合时：按全局最稀有优先预留，其余记入 unprotected_combos。"""
    n_e51_dds, n_rare, n_e51_ddm = 40, 2, 3  # DDM 组 3 条 ≤ cap，不削减 → E51 非风险组合
    rare_resins = [f"R{i}" for i in range(5)]  # 5 个稀有树脂各 2 条，均只与 DDS 搭配
    df = pd.DataFrame({
        "resin": ["E51"] * n_e51_dds + rare_resins * n_rare + ["E51"] * n_e51_ddm,
        "hardener": ["DDS"] * (n_e51_dds + 5 * n_rare) + ["DDM"] * n_e51_ddm,
        "id": range(n_e51_dds + 5 * n_rare + n_e51_ddm),
    })

    cleaner = AdvancedDataCleaner(df)
    cleaned, stats = cleaner.balance_formulation_stratified(
        group_cols=["hardener"],
        max_samples_per_group=3,  # cap=3 < 5 个风险组合，名额不够
        stratify_mode="within_group",
        random_state=42,
        protect_combo_cols=["resin"],
        min_samples_per_combo=1,
    )

    cp = stats["combo_protection"]
    assert cp["enabled"] is True
    assert cp["n_at_risk_combos"] == 5
    assert cp["reserved_samples"] == 3  # 名额全部分配给最稀有的前 3 个组合
    assert len(cp["protected_combos"]) == 3
    assert len(cp["unprotected_combos"]) == 2  # 剩余 2 个记入未保护清单
    assert len(cleaned[cleaned["hardener"] == "DDS"]) == 3


def test_combo_protection_ignored_for_duplicate_cols_or_mode():
    """保护列与分组列重复、或使用 by_target_with_group_cap 模式时，保护自动忽略并在 stats 中给出说明。"""
    df = pd.DataFrame({
        "resin": ["E51"] * 50 + ["CYD128"] * 10,
        "hardener": ["DDS"] * 50 + ["DDM"] * 10,
        "Tg": np.linspace(100, 200, 60),
    })

    cleaner = AdvancedDataCleaner(df)
    _, stats = cleaner.balance_formulation_stratified(
        group_cols=["hardener"],
        max_samples_per_group=10,
        stratify_mode="within_group",
        protect_combo_cols=["hardener"],  # 与分组列重复 → 无组合差异
    )
    assert stats["combo_protection"]["enabled"] is False
    assert "note" in stats["combo_protection"]

    _, stats2 = cleaner.balance_formulation_stratified(
        group_cols=["hardener"],
        target_col="Tg",
        stratify_mode="by_target_with_group_cap",
        max_per_group_per_bin=5,
        protect_combo_cols=["resin"],  # 仅 within_group 模式支持
    )
    assert stats2["combo_protection"]["enabled"] is False
    assert "note" in stats2["combo_protection"]
