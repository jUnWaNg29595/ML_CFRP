from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "局域网访问配置.bat"


def test_main_launcher_cleans_only_previous_app_py_streamlit_on_8501():
    source = LAUNCHER.read_text(encoding="utf-8")
    normalized = source.lower()

    assert "get-nettcpconnection" in normalized
    assert "localport 8501" in normalized
    assert "streamlit" in normalized
    assert "app\\.py" in normalized or "app.py" in normalized
    assert "--server.port 8501" in normalized
    assert "stop-process" in normalized or "taskkill" in normalized
    assert "--server.port 8555" not in normalized


def test_main_launcher_waits_for_port_release_before_starting_streamlit():
    source = LAUNCHER.read_text(encoding="utf-8").lower()
    cleanup_pos = source.index("get-nettcpconnection")
    start_pos = source.index("streamlit run app.py")
    assert cleanup_pos < start_pos
    assert "start-sleep" in source or "timeout" in source
