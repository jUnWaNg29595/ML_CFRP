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
