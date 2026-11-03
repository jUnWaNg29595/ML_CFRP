"""
测试分子设计引擎的高性能执行、上限提高、进度回调及自适应提前收敛。
"""
import pytest
from rdkit import Chem
from core.molecule_design import (
    parse_and_extract_custom_precursors,
    run_combinatorial_monomer_design,
    PrecursorCore,
    _process_single_precursor_reactions,
    SYNTHETIC_REACTION_TEMPLATES,
    ROUTE_CHAINS,
)
from rdkit.Chem import rdChemReactions


def test_progress_callback_and_early_exit():
    """验证进度回调能够正确被触发，且自适应提前收敛能够在目标达成后立即停止计算。"""
    raw_records = [
        {"smiles": "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1", "name": "DGEBA"},
        {"smiles": "Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1", "name": "DDS"},
    ]
    custom_precursors, _ = parse_and_extract_custom_precursors(raw_records, "测试模板")
    assert len(custom_precursors) > 0

    progress_reports = []

    def on_progress(done, total, prods, msg):
        progress_reports.append((done, total, prods))

    # 设置小目标 100 个，测试其能够迅速触发提前收敛，而不需要等待几千个前驱体算完
    target_prods = 100
    df, logs = run_combinatorial_monomer_design(
        custom_precursors=custom_precursors,
        enable_scaffold_fission=True,
        max_total_products=target_prods,
        progress_callback=on_progress,
    )

    assert not df.empty
    assert len(progress_reports) > 0
    # 验证提前收敛生效
    has_early_exit_log = any("提前收敛" in l for l in logs)
    assert has_early_exit_log
    # 产物数符合目标设定
    usable = df[df["role"] != "intermediate"]
    assert len(usable) <= target_prods


def test_oversized_precursor_filtered_early():
    """验证分子量过大或重原子超标的前驱体母核在前置阶段被快速过滤，避免组合爆炸。"""
    reactions = [r for r in SYNTHETIC_REACTION_TEMPLATES if r.default_enabled]
    compiled_reactions = [(r, rdChemReactions.ReactionFromSmarts(r.reaction_smarts)) for r in reactions if r.reaction_smarts]

    # 超大高分子片段母核 (MW > 1500 * 1.05)
    huge_smi = "Cc1cc(Cc2cc(Cc3cc(Cc4cc(Cc5cc(Cc6cc(Cc7cc(Cc8cc(Cc9cc(Cc1cc(C)c(O)cc1)c(O)cc9)c(O)cc8)c(O)cc7)c(O)cc6)c(O)cc5)c(O)cc4)c(O)cc3)c(O)cc2)c(O)cc1O" * 3
    huge_mol = Chem.MolFromSmiles(huge_smi)
    assert huge_mol is not None

    huge_p = PrecursorCore("p_huge", "超大前驱体", "多酚", "resin", huge_smi, "超大前驱体测试")
    prods = _process_single_precursor_reactions(
        huge_p,
        compiled_reactions,
        reactions,
        min_functionality=2,
        max_sa_score=6.0,
        mw_range=(140.0, 500.0), # 设定分子量上限 500
        enable_route_chains=True,
    )
    # 应该在前置过滤阶段直接返回空列表，绝不进行耗时的组合穷举
    assert len(prods) == 0
