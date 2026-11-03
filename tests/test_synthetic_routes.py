"""路线感知体系：ROUTE_CHAINS 注册表、四羟基环氧化路线、端到端等效。"""
import pytest
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from core.molecule_design import ROUTE_CHAINS


def test_routes_registered():
    ids = {r.route_id for r in ROUTE_CHAINS}
    assert {"P1_phenol_ech", "P2_allyl_epoxidation", "P3_chlorohydrin", "P4_sharpless"} <= ids


def test_route_smarts_compile():
    for route in ROUTE_CHAINS:
        for step in route.steps:
            if step.reaction_smarts:
                rxn = rdChemReactions.ReactionFromSmarts(step.reaction_smarts)
                assert rxn is not None, f"{route.route_id}.{step.step_index} 编译失败"


from core.molecule_design import (
    _expand_route_chain,
    PrecursorCore,
    ROUTE_CHAINS,
)


def test_p3_chain_end_to_end_equals_p1():
    """苯酚 --P3两步--> 与 P1 单步产物 canonical 等价；中间体以 intermediate 角色入库。"""
    phenol = PrecursorCore(core_id="test_phenol", name="苯酚", category="测试",
                           role="resin", smiles="Oc1ccccc1", description="", default_selected=True)
    p1 = next(r for r in ROUTE_CHAINS if r.route_id == "P1_phenol_ech")
    p3 = next(r for r in ROUTE_CHAINS if r.route_id == "P3_chlorohydrin")
    # 官能度=1 为刻意选择：苯→单环氧化末端产物官能度为 1，本测试目标
    # 是验证链路等价性（P3 两步 == P1 单步），而非门禁行为，故显式放宽门槛。
    out1 = _expand_route_chain(Chem.MolFromSmiles("Oc1ccccc1"), p1, phenol,
                                mw_range=(50.0, 1500.0), min_functionality=1)
    out3 = _expand_route_chain(Chem.MolFromSmiles("Oc1ccccc1"), p3, phenol,
                                mw_range=(50.0, 1500.0), min_functionality=1)
    end1 = [p for p in out1 if p.role != "intermediate"]
    end3 = [p for p in out3 if p.role != "intermediate"]
    assert end1 and end3
    assert end1[0].product_smiles == end3[0].product_smiles == "c1ccc(OCC2CO2)cc1"
    mids = [p for p in out3 if p.role == "intermediate"]
    assert mids, "P3 应产出氯代醇中间体"
    assert mids[0].product_smiles == "OC(CCl)COc1ccccc1"


def test_route_chain_functionality_gate_rejects_mono_functional_end_product():
    """默认 min_functionality=2 时，单官能末产物被拒，但中间体仍应入库。

    回归背景：路线链末步曾只查 functionality >= 1，导致单官能产物（如苯基缩水
    甘油醚）挤占配方候选池却无法交联。修复后官能度门禁与平铺模板路径一致；
    SA 分数豁免保留。中间体不受门禁影响，必须存活。
    """
    phenol = PrecursorCore(core_id="test_phenol", name="苯酚", category="测试",
                           role="resin", smiles="Oc1ccccc1", description="", default_selected=True)
    p3 = next(r for r in ROUTE_CHAINS if r.route_id == "P3_chlorohydrin")
    # 默认 min_functionality=2：苯→苯基缩水甘油醚（官能度 1）应被拒
    out3 = _expand_route_chain(Chem.MolFromSmiles("Oc1ccccc1"), p3, phenol,
                               mw_range=(50.0, 1500.0))
    ends = [p for p in out3 if p.role != "intermediate"]
    assert not ends, f"单官能末产物应被官能度门禁拒绝: {[e.product_smiles for e in ends]}"
    mids = [p for p in out3 if p.role == "intermediate"]
    assert mids, "末产物被拒时氯代醇中间体仍应入库"
    assert mids[0].product_smiles == "OC(CCl)COc1ccccc1"


import pandas as pd


def test_route_accumulation_columns_exist():
    """产物表必须带路线列：route_ids/route_redundancy/route_steps_min/green_score_best。"""
    from core.molecule_design import run_combinatorial_monomer_design
    df, _ = run_combinatorial_monomer_design(
        enable_scaffold_fission=False, min_functionality=2,
        max_total_products=100, n_jobs=1,
    )
    assert not df.empty
    for col in ("route_ids", "route_names", "route_redundancy", "route_steps_min", "green_score_best"):
        assert col in df.columns, f"缺少路线列 {col}"


def test_same_product_different_routes_accumulate():
    """同一产物若可经多条路线合成，route_ids 必须累加而非丢弃。"""
    import json
    from core.molecule_design import run_combinatorial_monomer_design, PrecursorCore
    phenol = PrecursorCore(core_id="tp", name="苯酚", category="测试", role="resin",
                           smiles="Oc1ccccc1", description="", default_selected=True)
    # 官能度=1 为刻意选择：苯酚→苯基缩水甘油醚官能度为 1，本测试目标
    # 是验证路线累加机制（P1+P3 路线合并），而非门禁行为，故显式放宽门槛。
    df, _ = run_combinatorial_monomer_design(
        custom_precursors=[phenol], enable_scaffold_fission=False,
        enable_route_chains=True, min_functionality=1, max_total_products=200, n_jobs=1,
    )
    target = df.loc[df["product_smiles"] == "c1ccc(OCC2CO2)cc1"]
    assert not target.empty, "苯基缩水甘油醚必须在产物中"
    routes = str(target.iloc[0]["route_ids"])
    assert "P1_phenol_ech" in routes and "P3_chlorohydrin" in routes, \
        f"路线累加失败: {routes}"


def test_route_serialization_sorted_and_paired():
    """序列化时路线必须按 route_id 成对排序：乱序到达也输出恒定顺序，且 id↔名称对应不被破坏。"""
    from core.molecule_design import _serialize_route_columns

    ids, names = _serialize_route_columns(
        ["P3_chlorohydrin", "P1_phenol_ech", "P2_allyl_epoxidation"],
        ["氯代醇法闭环", "P1 直接ECH醚化", "烯丙基化-环氧化"],
    )
    assert ids == "P1_phenol_ech;P2_allyl_epoxidation;P3_chlorohydrin"
    assert names == "P1 直接ECH醚化;烯丙基化-环氧化;氯代醇法闭环"

    # 防御：名称比 id 短时缺失项回退为 route_id 本身，绝不出现空名称，也不抛异常
    ids2, names2 = _serialize_route_columns(
        ["P3_chlorohydrin", "P1_phenol_ech"], ["氯代醇法闭环"]
    )
    assert ids2 == "P1_phenol_ech;P3_chlorohydrin"
    assert names2 == "P1_phenol_ech;氯代醇法闭环"


def test_route_columns_deterministic_across_concurrent_runs():
    """并发分支两次运行：PGE 行的路线列必须字节级一致，且等于按 route_id 排序后的顺序。

    并发分支要求 precursors > 10 且 workers > 1：这里用自建苯酚 + 骨架裂变
    （限制环/桥联/R 基以控制规模）凑足 48 个前驱体，两次运行均为多线程并发。
    """
    import re
    from core.molecule_design import run_combinatorial_monomer_design, PrecursorCore

    phenol = PrecursorCore(core_id="tp", name="苯酚", category="测试", role="resin",
                           smiles="Oc1ccccc1", description="", default_selected=True)
    kw = dict(
        custom_precursors=[phenol],
        enable_scaffold_fission=True,
        selected_rings=["benzene"],
        selected_linkers=["direct_bond"],
        selected_r_groups=["H"],
        selected_reaction_ids=["glycidyl_etherification_phenol"],
        # 官能度=1 为刻意选择：苯→苯基缩水甘油醚官能度为 1，本测试目标
        # 是验证路线列并发可复现性，而非门禁行为，故显式放宽门槛。
        min_functionality=1,
        max_total_products=200,
        n_jobs=-1,
    )

    df1, logs1 = run_combinatorial_monomer_design(**kw)
    df2, _ = run_combinatorial_monomer_design(**kw)

    # 确认并发分支前提成立：前驱体数 > 10（n_jobs=-1 时 workers = 逻辑核心数）
    m = re.search(r"共计\s*(\d+)\s*个前驱体", "\n".join(logs1))
    assert m, f"未找到前驱体计数日志: {logs1}"
    assert int(m.group(1)) > 10, "前驱体数不足 10，未触发并发分支"

    def _pge_route_strings(df):
        row = df.loc[df["product_smiles"] == "c1ccc(OCC2CO2)cc1"]
        assert not row.empty, "苯基缩水甘油醚必须在产物中"
        return str(row.iloc[0]["route_ids"]), str(row.iloc[0]["route_names"])

    ids1, names1 = _pge_route_strings(df1)
    ids2, names2 = _pge_route_strings(df2)
    assert ids1 == ids2, f"并发模式 route_ids 不可复现: {ids1!r} vs {ids2!r}"
    assert names1 == names2, f"并发模式 route_names 不可复现: {names1!r} vs {names2!r}"
    # 序列化结果必须已是按 route_id 排序的顺序；id 与名称数量保持一一对应
    assert ids1 == ";".join(sorted(ids1.split(";"))), f"路线未按 route_id 排序: {ids1!r}"
    assert len(names1.split(";")) == len(ids1.split(";")), f"id/名称未成对: {ids1!r} vs {names1!r}"


def test_intermediates_do_not_consume_max_total_products_quota():
    """中间体不得挤占 max_total_products 名额（该参数是用户对可用单体数的预期）。

    回归背景：路线链开启后曾出现 1748 个中间体吃掉名额，可用单体 3000 -> 1252（-58%）。
    修复后：resin+hardener 恰为 max_total_products（或全部，若不足），中间体额外保留。
    """
    from core.molecule_design import run_combinatorial_monomer_design

    budget = 300
    df, logs = run_combinatorial_monomer_design(
        enable_scaffold_fission=True,
        min_functionality=2, max_sa_score=6.0, mw_range=(140.0, 1500.0),
        max_total_products=budget, n_jobs=-1, enable_route_chains=True,
    )
    assert not df.empty
    usable = df[df["role"].isin(["resin", "hardener"])]
    # 可用单体数必须被 budget 约束，且应正好填满（生成量远大于 budget）
    assert len(usable) <= budget, f"可用单体 {len(usable)} 超过 budget {budget}"
    assert len(usable) == budget, f"可用单体 {len(usable)} 未填满 budget {budget}（被中间体挤占）"
    # 中间体额外保留，不参与配方筛选
    mids = df[df["role"] == "intermediate"]
    if not mids.empty:
        assert mids["equivalent_weight"].eq(0.0).all(), "中间体当量必须为 0"


def test_route_chain_substitutes_all_sites_on_multifunctional_substrate():
    """路线链每步必须有界穷举：双酚A 经 P1/P3 须得到全取代 DGEBA（func=2）。

    回归背景：执行器原先每步只应用一次 SMARTS，双酚A 只得单缩水甘油醚（func=1），
    被末步官能度门禁整体拒收，导致默认参数下 route_* 列永远为空（tagged=0）。
    DGEBA 的 canonical SMILES 与平铺模板路径产物一致，故可直接断言。
    """
    from core.molecule_design import _expand_route_chain, ROUTE_CHAINS

    dgeba = "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"
    bpa = PrecursorCore(
        core_id="test_bpa_chain", name="双酚A", category="测试", role="resin",
        smiles="CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", description="", default_selected=True,
    )
    bpa_mol = Chem.MolFromSmiles(bpa.smiles)
    for rid in ("P1_phenol_ech", "P3_chlorohydrin"):
        route = next(r for r in ROUTE_CHAINS if r.route_id == rid)
        out = _expand_route_chain(bpa_mol, route, bpa, mw_range=(50.0, 1500.0))
        ends = [p for p in out if p.role != "intermediate"]
        assert ends, f"{rid} 未产出末产物（可能退化为单遍应用）"
        assert any(p.product_smiles == dgeba for p in ends), \
            f"{rid} 未得到全取代 DGEBA: {[p.product_smiles for p in ends]}"
        assert all(p.functionality >= 2 for p in ends), \
            f"{rid} 末产物官能度虚低: {[p.functionality for p in ends]}"


def test_route_tags_reach_usable_monomers_by_default():
    """默认门禁下可用单体必须真正带上路线标记（Part B 核心价值的回归钉）。

    回归背景：单遍应用时代 tagged=0 —— 路线累加/冗余度/绿色评分对可用单体完全不可见。
    """
    from core.molecule_design import run_combinatorial_monomer_design

    bpa = PrecursorCore(
        core_id="test_bpa_tag", name="双酚A", category="测试", role="resin",
        smiles="CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", description="", default_selected=True,
    )
    df, _ = run_combinatorial_monomer_design(
        custom_precursors=[bpa], enable_scaffold_fission=False,
        min_functionality=2, max_total_products=200, n_jobs=1,
        enable_route_chains=True,
    )
    usable = df[df["role"].isin(["resin", "hardener"])]
    tagged = usable[usable["route_ids"].astype(str).str.len() > 0]
    assert not tagged.empty, "可用单体中无任何路线标记（路线特征空转）"
    dgeba_rows = tagged[tagged["product_smiles"] == "CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1"]
    assert not dgeba_rows.empty, "DGEBA 未带路线标记"
    routes = str(dgeba_rows.iloc[0]["route_ids"])
    assert "P1_phenol_ech" in routes and "P3_chlorohydrin" in routes, \
        f"DGEBA 未累加 P1+P3 路线: {routes}"
