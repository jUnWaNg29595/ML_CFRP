from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "局域网访问配置.bat"
PORTAL_LAUNCHER = Path(__file__).resolve().parents[1] / "启动预测平台.bat"


def test_main_launcher_cleans_only_previous_app_py_streamlit_on_8501():
    source = LAUNCHER.read_text(encoding="utf-8")
    normalized = source.lower()

    assert "get-nettcpconnection" in normalized
    # 主平台端口由变量 MAIN_PORT 承载，默认值为 8501
    assert 'if "%main_port%"=="" set "main_port=8501"' in normalized
    assert "-localport %main_port%" in normalized
    assert "streamlit" in normalized
    assert "app\\.py" in normalized or "app.py" in normalized
    assert "--server.port %main_port%" in normalized
    assert "stop-process" in normalized or "taskkill" in normalized
    assert "--server.port 8555" not in normalized


def test_main_launcher_waits_for_port_release_before_starting_streamlit():
    source = LAUNCHER.read_text(encoding="utf-8").lower()
    cleanup_pos = source.index("get-nettcpconnection")
    start_pos = source.index("streamlit run app.py")
    assert cleanup_pos < start_pos
    assert "start-sleep" in source or "timeout" in source


def test_main_launcher_refuses_to_kill_foreign_process():
    """外部进程占用端口时必须提示且不得 Stop-Process（验收：不误杀其他程序）。"""
    source = LAUNCHER.read_text(encoding="utf-8").lower()
    # elseif 分支：命令行不匹配 streamlit/app.py 时只提示占用，不关闭
    assert "elseif($proc)" in source
    assert "非本项目进程" in source
    # 该分支显式 exit 1，bat 侧检测后中止启动
    assert "exit 1" in source
    assert "端口 %main_port% 无法释放" in source


def test_main_launcher_supports_custom_port_argument():
    source = LAUNCHER.read_text(encoding="utf-8").lower()
    assert 'set "main_port=%~1"' in source


def test_portal_launcher_uses_8555_and_safe_defaults():
    source = PORTAL_LAUNCHER.read_text(encoding="utf-8").lower()
    assert "userprediction.py" in source
    assert "8555" in source
    assert "--server.port" in source
