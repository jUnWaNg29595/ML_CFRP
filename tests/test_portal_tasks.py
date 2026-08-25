import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

for _name, _value in {
    "long": np.int64,
    "ulong": np.uint64,
    "uintc": np.uint32,
}.items():
    if _name not in np.__dict__:
        setattr(np, _name, _value)

from core.portal_tasks import PortalTaskManager


def _request(tmp_path: Path, **updates):
    request = {
        "request": {
            "material_type": "epoxy_resin",
            "target": "tg",
            "inputs": {"x": 1.0},
            "confirmed_by_user": True,
        },
        "config": {"materials": {}},
    }
    request.update(updates)
    return request


def _result(value=42.5):
    return SimpleNamespace(
        prediction=value,
        unit="°C",
        warnings=[],
        model_version="v1",
        feature_workflow_id="workflow-1",
        summary={"rows": 1},
    )


def test_task_lifecycle_persists_completed_result(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.portal_tasks.run_confirmed_prediction",
        lambda request, config, progress: _result(),
    )
    manager = PortalTaskManager(tmp_path)

    task_id = manager.create_task(_request(tmp_path))
    snapshot = manager.wait_for_task(task_id, timeout=2)

    assert snapshot["status"] == "completed"
    assert snapshot["progress"] == 100
    assert snapshot["result"]["prediction"] == 42.5
    assert (tmp_path / "prediction_portal" / "tasks" / f"{task_id}.json").exists()


def test_cancelled_task_does_not_publish_later_prediction(tmp_path, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def predict(request, config, progress):
        started.set()
        release.wait(2)
        progress({"stage": "predicting"})
        return _result()

    monkeypatch.setattr("core.portal_tasks.run_confirmed_prediction", predict)
    manager = PortalTaskManager(tmp_path)
    task_id = manager.create_task(_request(tmp_path))
    assert started.wait(2)

    cancelled = manager.cancel_task(task_id)
    release.set()
    snapshot = manager.wait_for_task(task_id, timeout=2)

    assert cancelled["status"] == "cancelled"
    assert snapshot["status"] == "cancelled"
    assert snapshot.get("result") is None


def test_retry_creates_new_task_and_reuses_persisted_request(tmp_path, monkeypatch):
    calls = []

    def predict(request, config, progress):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("temporary failure")
        return _result(43.0)

    monkeypatch.setattr("core.portal_tasks.run_confirmed_prediction", predict)
    manager = PortalTaskManager(tmp_path)
    first_id = manager.create_task(_request(tmp_path))
    first = manager.wait_for_task(first_id, timeout=2)
    retry_id = manager.retry_task(first_id)
    second = manager.wait_for_task(retry_id, timeout=2)

    assert first["status"] == "failed"
    assert retry_id != first_id
    assert second["status"] == "completed"
    assert calls[1]["material_type"] == "epoxy_resin"


def test_restart_marks_active_snapshot_failed(tmp_path):
    task_dir = tmp_path / "prediction_portal" / "tasks"
    task_dir.mkdir(parents=True)
    (task_dir / "stale.json").write_text(
        json.dumps({"task_id": "stale", "status": "running", "progress": 20}),
        encoding="utf-8",
    )

    manager = PortalTaskManager(tmp_path)
    snapshot = manager.get_task_snapshot("stale")

    assert snapshot["status"] == "failed"
    assert "重启" in snapshot["error"]


def test_explanation_failure_keeps_python_prediction_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "core.portal_tasks.run_confirmed_prediction",
        lambda request, config, progress: _result(),
    )
    monkeypatch.setattr(
        "core.portal_tasks.run_explanation",
        lambda request, result: (_ for _ in ()).throw(RuntimeError("AI unavailable")),
    )
    manager = PortalTaskManager(tmp_path)
    task_id = manager.create_task(_request(tmp_path, explain=True))
    snapshot = manager.wait_for_task(task_id, timeout=2)

    assert snapshot["status"] == "completed"
    assert snapshot["result"]["prediction"] == 42.5
    assert "AI unavailable" in snapshot["explanation_error"]

