"""UI 契约：路线列缺失时旧缓存可读；路线链开关在函数签名中存在。"""
import inspect

import pandas as pd


TEST_LEGACY_ROUTE_COLUMNS = ("route_ids", "route_redundancy", "route_steps_min", "green_score_best")


def test_run_signature_has_route_chain_params():
    from core.molecule_design import run_combinatorial_monomer_design
    sig = inspect.signature(run_combinatorial_monomer_design)
    assert "enable_route_chains" in sig.parameters
    assert "route_chain_ids" in sig.parameters


def test_legacy_parquet_columns_backfilled():
    """旧 parquet（无路线列）经真实回填函数处理后必须补齐默认列，而非抛 KeyError。

    这里调用的是 core.molecule_design.ensure_route_columns 本体（生产代码），
    不是把回填逻辑抄进测试——抄一遍只能证明测试自己写得对。
    """
    from core.molecule_design import ensure_route_columns

    legacy = pd.DataFrame({
        "product_smiles": ["c1ccc(OCC2CO2)cc1", "Oc1ccccc1"],
        "role": ["resin", "hardener"],
    })
    out = ensure_route_columns(legacy)
    for col in TEST_LEGACY_ROUTE_COLUMNS:
        assert col in out.columns, f"旧缓存缺列 {col} 未被回填"
    assert out.loc[0, "route_ids"] == ""
    assert out.loc[0, "route_redundancy"] == 1
    assert out.loc[0, "route_steps_min"] == 1
    assert out.loc[0, "green_score_best"] == 0.0
    # 原有列与数据不得被破坏
    assert out["role"].tolist() == ["resin", "hardener"]
    assert out["product_smiles"].tolist() == ["c1ccc(OCC2CO2)cc1", "Oc1ccccc1"]


def test_route_columns_preserved_when_already_present():
    """新数据（已有路线列）不得被回填覆盖。"""
    from core.molecule_design import ensure_route_columns

    fresh = pd.DataFrame({
        "product_smiles": ["c1ccc(OCC2CO2)cc1"],
        "role": ["resin"],
        "route_ids": ["P1_phenol_ech;P3_chlorohydrin"],
        "route_redundancy": [2],
        "route_steps_min": [1],
        "green_score_best": [6.0],
    })
    out = ensure_route_columns(fresh)
    assert out.loc[0, "route_ids"] == "P1_phenol_ech;P3_chlorohydrin"
    assert out.loc[0, "route_redundancy"] == 2
    assert out.loc[0, "green_score_best"] == 6.0


def test_backfill_after_cache_roundtrip_without_route_columns():
    """旧持久化缓存（仅 product_smiles/role）回填后必须补齐 4 个路线列。

    对应 app_lib.py 配方页与设计页两处旧缓存读取点：读盘后的帧直接进 session_state，
    必须经 ensure_route_columns 补齐，否则结果表与后续 concat 会缺列/出现 NaN。
    本环境无 parquet 引擎（pyarrow/fastparquet 均缺失），故用等价的帧复制模拟
    "读盘得到旧格式帧"这一数据形态，验证生产回填函数的契约。
    """
    from core.molecule_design import ensure_route_columns

    on_disk_legacy = pd.DataFrame({
        "product_smiles": ["c1ccc(OCC2CO2)cc1"],
        "role": ["resin"],
        "reaction_id": ["glycidyl_etherification"],
    })
    loaded = on_disk_legacy.copy(deep=True)   # 模拟 read_parquet 得到的新帧
    out = ensure_route_columns(loaded)
    for col in TEST_LEGACY_ROUTE_COLUMNS:
        assert col in out.columns, f"旧缓存回填后仍缺列 {col}"
    assert out.loc[0, "route_ids"] == ""
    assert out.loc[0, "route_redundancy"] == 1
    assert out.loc[0, "route_steps_min"] == 1
    assert out.loc[0, "green_score_best"] == 0.0
    # 原盘上已有列不得被改动
    assert out.loc[0, "reaction_id"] == "glycidyl_etherification"
    assert out["role"].tolist() == ["resin"]
