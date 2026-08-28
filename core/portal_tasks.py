"""Reload-safe background tasks for the prediction portal."""

from __future__ import annotations

import json
import hashlib
import os
import re
import threading
import time
import uuid
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .portal_prediction import run_confirmed_prediction


_ACTIVE_STATUSES = {"queued", "validating", "featuring", "predicting", "explaining", "pending", "running"}
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
_STAGE_PROGRESS = {
    "queued": 0,
    "validated": 10,
    "validation": 5,
    "workflow": 50,
    "workflow_step": 60,
    "predicting": 85,
    "explanation": 95,
    "explaining": 95,
    "completed": 100,
}


class _TaskCancelled(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


_SENSITIVE_ERROR_PATTERN = re.compile(
    r"(?i)(api[_ -]?key|token|secret|password|authorization)\s*[:=]\s*[^\s,;]+"
)


def _safe_error(value: Any) -> str:
    text = str(value).replace("\x00", " ").strip()
    text = _SENSITIVE_ERROR_PATTERN.sub(r"\1=[redacted]", text)
    return text[:1000]


_SECRET_ERROR_PATTERN = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|passwd|authorization|x-api-key|key)"
    r"(\s*[:=]\s*|\s+)([\"']?)([^\s,;}&\"']+)\3?"
)
_BEARER_PATTERN = re.compile(r"(?i)(\bbearer\s+)[^\s,;}&]+")
_QUERY_SECRET_PATTERN = re.compile(
    r"(?i)([?&](?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|key)=)[^&#\s]+"
)


def _safe_error(value: Any) -> str:
    message = str(value)
    message = _QUERY_SECRET_PATTERN.sub(r"\1[REDACTED]", message)
    message = _BEARER_PATTERN.sub(r"\1[REDACTED]", message)
    return _SECRET_ERROR_PATTERN.sub(r"\1\2[REDACTED]", message)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _jsonable(value.to_dict(orient="records"))
        except TypeError:
            return _jsonable(value.to_dict())
    if hasattr(value, "item") and callable(value.item):
        try:
            return _jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        return _jsonable(value.tolist())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported task value: {type(value).__name__}")


def run_explanation(request: Mapping[str, Any], result: Mapping[str, Any]) -> Any:
    """Run optional AI explanation without importing or calling Streamlit."""

    client = request.get("ai_client")
    if client is not None and callable(getattr(client, "explain_result", None)):
        return client.explain_result(result)
    ai_config = request.get("ai_config")
    if isinstance(ai_config, Mapping):
        from .portal_ai import PortalAIClient
        from .portal_ai_config import AIServiceConfig

        return PortalAIClient(AIServiceConfig(**dict(ai_config))).explain_result(result)
    raise RuntimeError("AI explanation is not configured")


class PortalTaskManager:
    """Execute confirmed predictions in bounded workers with durable snapshots."""

    def __init__(self, root: Path, executor: Any = None):
        self.root = Path(root)
        self.task_dir = self.root / "prediction_portal" / "tasks"
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conditions: dict[str, threading.Condition] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._futures: dict[str, Future[Any]] = {}
        self._snapshots: dict[str, dict[str, Any]] = {}
        self._runtime_requests: dict[str, Mapping[str, Any]] = {}
        self._owns_executor = executor is None
        self._executor = executor or ThreadPoolExecutor(max_workers=2, thread_name_prefix="portal-task")
        self._load_snapshots()

    def _path(self, task_id: str) -> Path:
        return self.task_dir / f"{task_id}.json"

    def _condition(self, task_id: str) -> threading.Condition:
        condition = self._conditions.get(task_id)
        if condition is None:
            condition = threading.Condition(self._lock)
            self._conditions[task_id] = condition
        return condition

    def _persist(self, snapshot: dict[str, Any]) -> None:
        payload = _jsonable(snapshot)
        destination = self._path(str(snapshot["task_id"]))
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            try:
                temporary.chmod(0o600)
            except OSError:
                if os.name != "nt":
                    raise
            os.replace(temporary, destination)
            try:
                destination.chmod(0o600)
            except OSError:
                if os.name != "nt":
                    raise
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _publish(self, task_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            current = self._snapshots[task_id]
            updated = dict(current)
            updated.update(updates)
            updated["updated_at"] = _now()
            self._persist(updated)
            self._snapshots[task_id] = updated
            self._condition(task_id).notify_all()
            return dict(updated)

    def _load_snapshots(self) -> None:
        for path in sorted(self.task_dir.glob("*.json")):
            try:
                snapshot = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            task_id = snapshot.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                continue
            self._snapshots[task_id] = snapshot
            self._condition(task_id)
        for task_id, snapshot in list(self._snapshots.items()):
            if snapshot.get("status") in _ACTIVE_STATUSES:
                self._publish(
                    task_id,
                    status="failed",
                    error="任务在服务重启后失效。",
                    finished_at=_now(),
                )

    def _split_request(self, envelope: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        if "request" in envelope or "prediction_request" in envelope:
            prediction_request = envelope.get("request", envelope.get("prediction_request"))
        else:
            prediction_request = {
                key: value
                for key, value in envelope.items()
                if key not in {"config", "ai_config", "explain", "explanation_requested"}
            }
        if not isinstance(prediction_request, Mapping):
            raise ValueError("task request must contain a prediction request object")
        config = envelope.get("config", {})
        if not isinstance(config, Mapping):
            raise ValueError("task config must be an object")
        return prediction_request, config

    def create_task(self, request: Mapping[str, Any]) -> str:
        if not isinstance(request, Mapping):
            raise ValueError("task request must be an object")
        task_id = uuid.uuid4().hex
        request_payload = _jsonable(request)
        request_hash = hashlib.sha256(json.dumps(request_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        request_summary = request_payload if isinstance(request_payload, Mapping) else {}
        snapshot = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "stage": "queued",
            "request_summary_hash": request_hash,
            "request_fields": sorted(str(key) for key in request_summary.keys()),
            "result": None,
            "explanation": None,
            "explanation_error": "",
            "error": "",
            "created_at": _now(),
            "updated_at": _now(),
        }
        with self._lock:
            self._snapshots[task_id] = snapshot
            self._runtime_requests[task_id] = dict(request)
            self._cancel_events[task_id] = threading.Event()
            self._condition(task_id)
            self._persist(snapshot)
            future = self._executor.submit(self._run_task, task_id)
            self._futures[task_id] = future
        return task_id

    def _check_cancelled(self, task_id: str) -> None:
        if self._cancel_events[task_id].is_set():
            raise _TaskCancelled("任务已取消。")

    def _progress_callback(self, task_id: str, payload: Any = None, *args: Any, **kwargs: Any) -> None:
        self._check_cancelled(task_id)
        details = payload if isinstance(payload, Mapping) else {}
        stage = str(details.get("stage") or (args[0] if args else "running"))
        progress = _STAGE_PROGRESS.get(stage, 0)
        status = {
            "validated": "validating", "validation": "validating",
            "workflow": "featuring", "workflow_step": "featuring",
            "predicting": "predicting", "explanation": "explaining",
            "explaining": "explaining",
        }.get(stage, "validating")
        self._publish(task_id, status=status, stage=stage, progress=progress)
        self._check_cancelled(task_id)

    def _run_task(self, task_id: str) -> None:
        try:
            self._publish(task_id, status="validating", stage="validation", progress=5)
            self._check_cancelled(task_id)
            envelope = self._runtime_requests.get(task_id)
            if not isinstance(envelope, Mapping):
                raise ValueError("任务快照不包含可重放的原始请求，请由用户重新提交输入。")
            prediction_request, config = self._split_request(envelope)
            result = run_confirmed_prediction(
                prediction_request,
                config=config,
                progress=lambda payload=None, *args, **kwargs: self._progress_callback(
                    task_id, payload, *args, **kwargs
                ),
            )
            self._check_cancelled(task_id)
            serialized_result = _jsonable(result)
            explanation = None
            explanation_error = ""
            if envelope.get("explain") is True or envelope.get("explanation_requested") is True:
                self._publish(task_id, stage="explanation", progress=95)
                self._check_cancelled(task_id)
                try:
                    explanation = _jsonable(run_explanation(envelope, serialized_result))
                except Exception as exc:
                    explanation_error = _safe_error(exc)
            self._check_cancelled(task_id)
            self._publish(
                task_id,
                status="completed",
                stage="completed",
                progress=100,
                result=serialized_result,
                explanation=explanation,
                explanation_error=explanation_error,
                finished_at=_now(),
            )
        except _TaskCancelled as exc:
            self._publish(task_id, status="cancelled", stage="cancelled", error=_safe_error(exc), finished_at=_now())
        except Exception as exc:
            self._publish(task_id, status="failed", stage="failed", error=_safe_error(exc), finished_at=_now())

    def get_task_snapshot(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            snapshot = self._snapshots.get(task_id)
            if snapshot is None:
                path = self._path(task_id)
                if not path.is_file():
                    raise KeyError(task_id)
                snapshot = json.loads(path.read_text(encoding="utf-8"))
                self._snapshots[task_id] = snapshot
                self._condition(task_id)
            return dict(snapshot)

    def cancel_task(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            snapshot = self.get_task_snapshot(task_id)
            if snapshot.get("status") in _ACTIVE_STATUSES:
                self._cancel_events.setdefault(task_id, threading.Event()).set()
                future = self._futures.get(task_id)
                if future is not None:
                    future.cancel()
                return self._publish(
                    task_id,
                    status="cancelled",
                    stage="cancelled",
                    error="任务已取消。",
                    finished_at=_now(),
                )
            return snapshot

    def retry_task(self, task_id: str, request: Mapping[str, Any] | None = None) -> str:
        snapshot = self.get_task_snapshot(task_id)
        if snapshot.get("status") not in {"failed", "cancelled"}:
            raise ValueError("只有失败或取消的任务可以重试。")
        if not isinstance(request, Mapping):
            raise ValueError("重试任务必须由用户重新提交原始输入。")
        return self.create_task(request)

    def wait_for_task(self, task_id: str, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            condition = self._condition(task_id)
            start = time.monotonic()
            while True:
                snapshot = self.get_task_snapshot(task_id)
                if snapshot.get("status") in _TERMINAL_STATUSES:
                    return snapshot
                if timeout is not None:
                    remaining = float(timeout) - (time.monotonic() - start)
                    if remaining <= 0:
                        return snapshot
                    condition.wait(remaining)
                else:
                    condition.wait()

    def shutdown(self, wait: bool = True) -> None:
        if self._owns_executor:
            self._executor.shutdown(wait=wait)


__all__ = ["PortalTaskManager", "run_explanation"]
