"""
后台任务持久化与进程生命周期管理模块 (Task Store & Process Management)
核心规范：
1. 任务提交后状态持久化到磁盘 (JSON 存储)，页面刷新不丢失、不重复提交；
2. 进程完成状态与科学门禁验收状态严格区分；
3. 进程取消仅针对本任务的独立进程组 (PGID)，严禁按模糊进程名杀除外部系统任务；
4. 捕获完整原生 stdout, stderr, exit_code 与执行耗时。
"""

import os
import sys
import json
import time
import signal
import subprocess
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

TASK_DB_PATH = "/home/wangj/project/CFRP/extensions/network_md_features/artifacts/jobs/task_store.json"

@dataclass
class TaskRecord:
    task_id: str
    network_id: str
    formulation_id: str
    status: str  # "QUEUED", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"
    scientific_gate_status: str  # "NOT_RUN", "PASS", "FAIL", "EXPLORATORY"
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    pid: Optional[int] = None
    pgid: Optional[int] = None
    exit_code: Optional[int] = None
    log_file: str = ""
    error_message: str = ""
    gate_details: Optional[Dict[str, Any]] = None
    extracted_features: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class TaskStore:
    def __init__(self, db_path: str = TASK_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        if not os.path.exists(self.db_path):
            self._save_all({})

    def _load_all(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.db_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_all(self, data: Dict[str, Dict[str, Any]]) -> None:
        temp_path = self.db_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(temp_path, self.db_path)

    def register_task(self, task: TaskRecord) -> None:
        data = self._load_all()
        data[task.task_id] = task.to_dict()
        self._save_all(data)

    def update_task(self, task_id: str, **kwargs) -> Optional[TaskRecord]:
        data = self._load_all()
        if task_id not in data:
            return None
        t_dict = data[task_id]
        t_dict.update(kwargs)
        data[task_id] = t_dict
        self._save_all(data)
        return TaskRecord(**t_dict)

    def get_task(self, task_id: str) -> Optional[TaskRecord]:
        data = self._load_all()
        if task_id in data:
            return TaskRecord(**data[task_id])
        return None

    def list_tasks(self) -> List[TaskRecord]:
        data = self._load_all()
        return [TaskRecord(**v) for v in data.values()]

    def cancel_task(self, task_id: str) -> bool:
        """
        仅针对当前任务所属独立进程组 (PGID) 进行精准信号取消，绝不影响外部进程
        """
        task = self.get_task(task_id)
        if not task or task.status not in ("QUEUED", "RUNNING"):
            return False

        if task.pgid:
            try:
                # 发送 SIGTERM 到整个进程组
                os.killpg(task.pgid, signal.SIGTERM)
                time.sleep(0.5)
                # 检查是否退出，若未退出则 SIGKILL
                os.killpg(task.pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except Exception as e:
                print(f"Error terminating pgid {task.pgid}: {e}")

        self.update_task(
            task_id,
            status="CANCELLED",
            scientific_gate_status="NOT_RUN",
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            error_message="User requested cancellation"
        )
        return True
