from pathlib import Path

import pandas as pd


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app_source() -> str:
    return APP_PATH.read_text(encoding="utf-8-sig")


def test_training_dataset_contract_has_target_and_provenance_columns():
    required_columns = {
        "smiles",
        "mp_c",
        "mp_raw",
        "mp_quality",
        "component_role",
        "hardener_class",
        "cid",
        "source_url",
        "canonical_smiles",
    }
    frame = pd.DataFrame(columns=sorted(required_columns))

    assert required_columns.issubset(set(frame.columns))
    assert frame.empty


def test_virtual_screening_has_melting_point_collection_panel_contract():
    source = _app_source()

    assert "def render_melting_point_dataset_panel()" in source
    assert "熔点数据集采集" in source
    assert "树脂 SMARTS" in source
    assert "固化剂类别" in source
    assert "树脂 CID 上限" in source
    assert "固化剂 CID 上限" in source
    assert "fetch_melting_point_records_by_smarts" in source
    assert "prepare_melting_point_dataset" in source


def test_melting_point_collection_persists_raw_clean_summary_and_supports_resume():
    source = _app_source()

    for key in (
        "melting_point_raw_records",
        "melting_point_dataset",
        "melting_point_dataset_summary",
        "melting_point_collection_progress",
    ):
        assert key in source
    assert "缓存" in source
    assert "续采" in source or "resume" in source.lower()
    assert "st.progress" in source or "st.status" in source
    assert "下载原始熔点记录" in source
    assert "下载熔点训练数据" in source


def test_melting_point_collection_has_explicit_training_handoff():
    source = _app_source()

    assert "载入模型训练" in source
    assert "melting_point_training_dataset" in source
    assert "target_col" in source
    assert '"mp_c"' in source or "'mp_c'" in source


def test_model_training_handoff_defaults_to_finite_high_quality_mp_rows():
    source = _app_source()

    assert "melting_point_training_dataset" in source
    assert "低质量" in source
    assert "include_low_quality" in source
    assert "isfinite" in source
    assert "mp_quality" in source
    assert "melting_point_training_handoff" in source or "熔点训练数据已载入" in source


def test_virtual_screening_page_renders_collection_panel_before_design_engine():
    source = _app_source()
    page_start = source.rfind("def page_virtual_screening():")
    assert page_start >= 0
    page_source = source[page_start:]

    assert "render_melting_point_dataset_panel()" in page_source
    assert "_render_molecule_design_engine()" in page_source

def test_virtual_screening_result_table_exposes_melting_point_columns_in_chinese():
    source = _app_source()

    required_columns = (
        "resin_mp_predicted_c",
        "resin_mp_std_c",
        "resin_mp_ad_score",
        "resin_mp_filter_status",
        "resin_mp_filter_reason",
        "hardener_mp_predicted_c",
        "hardener_mp_std_c",
        "hardener_mp_ad_score",
        "hardener_mp_filter_status",
        "hardener_mp_filter_reason",
        "mp_filter_reason",
    )
    for column in required_columns:
        assert f'"{column}"' in source

    for label in (
        "树脂预测熔点（°C）",
        "固化剂预测熔点（°C）",
        "树脂熔点筛选状态",
        "固化剂熔点筛选状态",
        "熔点综合筛选原因",
    ):
        assert label in source
