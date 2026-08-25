from pathlib import Path
import re


APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'


def _app_source() -> str:
    return APP_PATH.read_text(encoding='utf-8-sig')


def test_virtual_screening_entry_points_are_unique_and_split_by_mode():
    source = _app_source()

    assert len(re.findall(r'(?m)^def page_virtual_screening\(', source)) == 1
    assert len(re.findall(r'(?m)^def _page_virtual_screening_formula\(', source)) == 1
    assert len(re.findall(r'(?m)^def _render_molecule_design_engine\(', source)) == 1
    entry = source[source.rfind('def page_virtual_screening():'):]
    assert '配方级高通量筛选' in entry
    assert '分子设计引擎' in entry
    assert 'render_melting_point_dataset_panel()' in entry


def test_melting_point_screening_predicts_resin_and_hardener_separately():
    source = _app_source()
    predictor_start = source.index('def _predict_melting_point_role(')
    predictor_end = source.index('st.markdown("### 5) 联合评分")', predictor_start)
    predictor = source[predictor_start:predictor_end]

    assert "role_column = 'resin_smiles' if role == 'resin' else 'hardener_smiles'" in predictor
    assert "for melting_point_role in ('resin', 'hardener')" in source
    assert 'apply_melting_point_gate(' in source
    assert "mode=melting_point_mode_applied" in source


def test_melting_point_model_import_requires_artifact_workflow_compatibility():
    source = _app_source()
    import_start = source.index("with st.expander('🌡️ 独立熔点模型")
    import_end = source.index("# [关键修复] 检查模型是否存在", import_start)
    import_block = source[import_start:import_end]

    assert 'validate_melting_point_artifact(' in import_block
    assert 'validate_melting_point_artifact_for_screening(' in import_block
    assert 'melting_point_model_feature_cols' in import_block
    assert 'validate_melting_point_artifact_for_screening' in import_block


def test_melting_point_artifact_export_keeps_task_metadata_and_workflow_fields():
    source = _app_source()
    export_start = source.index('def _current_melting_point_artifact_extra(')
    training_start = source.index('def page_model_training(')
    export_source = source[export_start:training_start]
    assert 'build_melting_point_artifact_extra' in export_source
    assert 'target != "mp_c"' in export_source

    model_export_start = source.index('create_model_artifact_bytes(')
    model_export_end = source.index('process_payload = {', model_export_start)
    model_export = source[model_export_start:model_export_end]
    assert 'model=current_model' in model_export
    assert '_extra.update(_current_melting_point_artifact_extra(target_col))' in source
    assert "'molecular_feature_workflow': st.session_state.get('molecular_feature_workflow')" in source

    process_start = source.index('process_payload = {')
    process_end = source.index('process_bytes = json.dumps', process_start)
    process_export = source[process_start:process_end]
    assert '_current_melting_point_artifact_extra(target_col)' in process_export
    assert "'molecular_feature_workflow'" in process_export
