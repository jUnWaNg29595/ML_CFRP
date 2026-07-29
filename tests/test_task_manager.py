import time
from pathlib import Path

from core.task_manager import BackgroundTaskManager, TaskStatus, is_cancelled


def test_acquire_task_is_idempotent_for_active_task_key():
    manager = BackgroundTaskManager()
    manager.reset()

    first_id, first_created = manager.acquire_task(
        name='虚拟筛选',
        task_type='virtual_screening',
        task_key='virtual_screening',
    )
    second_id, second_created = manager.acquire_task(
        name='虚拟筛选',
        task_type='virtual_screening',
        task_key='virtual_screening',
    )

    assert first_created is True
    assert second_created is False
    assert second_id == first_id
    assert manager.get_task_snapshot(first_id).status == TaskStatus.PENDING

    manager.reset()


def test_completed_task_keeps_result_for_later_rerun():
    manager = BackgroundTaskManager()
    manager.reset()

    task_id, created = manager.acquire_task(
        name='测试任务',
        task_type='test',
        task_key='test_result',
    )
    assert created is True
    manager.start_task(task_id)
    manager.complete_task(task_id, success=True, result={'rows': 3})

    snapshot = manager.get_task_snapshot(task_id)
    assert snapshot.status == TaskStatus.COMPLETED
    assert snapshot.result == {'rows': 3}
    assert manager.get_task_by_key('test_result').task_id == task_id

    manager.reset()


def test_sidebar_initializes_task_lock_before_widget_uses_it():
    app_source = (
        Path(__file__).resolve().parents[1] / 'app.py'
    ).read_text(encoding='utf-8')
    lock_assignment = 'active_task_lock = bool(get_task_manager().get_active_tasks())'
    widget_usage = 'disabled=active_task_lock'

    assert app_source.index(lock_assignment) < app_source.index(widget_usage)


def test_virtual_screening_restores_saved_result_before_run_controls():
    app_source = (
        Path(__file__).resolve().parents[1] / 'app.py'
    ).read_text(encoding='utf-8')
    result_state = 'saved_vs_formula_result = st.session_state.get("vs_formula_result_df")'
    run_button = '"🚀 开始配方级高通量筛选"'

    assert app_source.index(result_state) < app_source.index(run_button)


def test_main_dispatch_has_global_task_lock_before_page_controls():
    app_source = (
        Path(__file__).resolve().parents[1] / 'app.py'
    ).read_text(encoding='utf-8')
    lock_gate = 'if _render_global_task_lock(page):'
    first_dispatch = 'if page == "🏠 首页":'

    assert app_source.index(lock_gate) < app_source.index(first_dispatch)


def test_acquiring_new_task_clears_cancel_flag_after_previous_run_stops():
    manager = BackgroundTaskManager()
    manager.reset()

    task_id, created = manager.acquire_task(
        name='可取消任务',
        task_type='test',
        task_key='cancel_then_retry',
    )
    assert created is True
    manager.start_task(task_id)
    manager.cancel_all_tasks(force=False)
    assert is_cancelled() is True

    retry_id, retry_created = manager.acquire_task(
        name='可取消任务',
        task_type='test',
        task_key='cancel_then_retry',
    )
    assert retry_created is True
    assert retry_id != task_id
    assert is_cancelled() is False

    manager.reset()


def test_virtual_screening_guard_catches_streamlit_control_flow_exceptions():
    app_source = (
        Path(__file__).resolve().parents[1] / 'app.py'
    ).read_text(encoding='utf-8')

    assert 'except BaseException as exc:' in app_source
