def test_feature_registry_ui_exports_minimal_renderer():
    from core.feature_registry_ui import render_feature_registry_page

    assert callable(render_feature_registry_page)


def test_app_dispatch_exposes_feature_management_page():
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "app.py"
    text = source.read_text(encoding="utf-8")
    assert "🧩 特征管理" in text
    assert "render_feature_registry_page" in text


def test_feature_review_ui_persists_ai_and_local_decision_records():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "feature_registry_ui.py").read_text(encoding="utf-8")
    assert "save_feature_review_record" in source
    assert "feature_reviews" in source
    assert "ai_response" in source
    assert "local_decision" in source
