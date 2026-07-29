# -*- coding: utf-8 -*-
"""
后台任务管理模块 - 统一管理和控制所有后台计算任务

功能：
1. 注册和追踪所有后台任务（进程池、线程池、子进程等）
2. 提供一键终止所有后台任务的功能
3. 任务状态监控和日志记录
"""

import os
import signal
import threading
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from urllib.parse import parse_qs, urlparse
import weakref
import time
import secrets

# =============================================================================
# 任务状态枚举
# =============================================================================
class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


# =============================================================================
# 任务信息数据类
# =============================================================================
@dataclass
class TaskInfo:
    """任务信息"""
    task_id: str
    name: str
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    progress: float = 0.0  # 0-100
    total_items: int = 0
    processed_items: int = 0
    error_message: str = ""
    task_type: str = "unknown"  # 'process_pool', 'thread_pool', 'subprocess'
    task_key: str = ""
    message: str = ""
    result: Any = field(default=None, repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict, repr=False)
    heartbeat_at: Optional[datetime] = None
    

# =============================================================================
# 全局取消标志 - 用于跨进程通信
# =============================================================================
# ✅ 修复：使用共享内存 Value 替代 mp.Event()，确保在 spawn 模式下也能跨进程通信
# mp.Event() 在模块导入时创建，但在 spawn 模式下子进程无法继承该对象
_CANCEL_VALUE = mp.Value('i', 0)  # 使用共享内存整数：0=未取消，1=已取消
_CANCEL_LOCK = threading.Lock()
# 保留 Event 用于主进程内的线程同步（可选）
_CANCEL_EVENT_LOCAL = threading.Event()
_EMERGENCY_STOP_SERVER = None
_EMERGENCY_STOP_THREAD = None
_EMERGENCY_STOP_PORT = None
_EMERGENCY_STOP_HOST = "127.0.0.1"
_EMERGENCY_STOP_TOKEN = secrets.token_urlsafe(24)


def _looks_like_cancel_message(message: str) -> bool:
    text = str(message or "").strip().lower()
    if not text:
        return False
    cancel_markers = (
        "用户取消",
        "任务已被取消",
        "训练已停止",
        "cancelled",
        "canceled",
    )
    return any(marker in text for marker in cancel_markers)


def is_cancelled() -> bool:
    """检查是否已请求取消（可在任务内部调用，支持跨进程）"""
    try:
        # 优先检查共享内存值（支持跨进程）
        return _CANCEL_VALUE.value == 1
    except Exception:
        # 回退到本地 Event（仅主进程内有效）
        try:
            return _CANCEL_EVENT_LOCAL.is_set()
        except Exception:
            return False


def request_cancel():
    """请求取消所有任务（跨进程生效）"""
    with _CANCEL_LOCK:
        try:
            _CANCEL_VALUE.value = 1
        except Exception:
            pass
        try:
            _CANCEL_EVENT_LOCAL.set()
        except Exception:
            pass


def clear_cancel():
    """清除取消标志（重新开始任务前调用）"""
    with _CANCEL_LOCK:
        try:
            _CANCEL_VALUE.value = 0
        except Exception:
            pass
        try:
            _CANCEL_EVENT_LOCAL.clear()
        except Exception:
            pass


class _EmergencyStopRequestHandler(BaseHTTPRequestHandler):
    """Lightweight local endpoint for the in-page emergency-stop control."""

    server_version = "CFRPTrainingEmergencyStop/1.0"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        token = params.get("token", [""])[0]

        if token != _EMERGENCY_STOP_TOKEN:
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b"forbidden")
            return

        if parsed.path == "/status":
            try:
                active_tasks = len(get_task_manager().get_active_tasks())
            except Exception:
                active_tasks = 0
            self.send_response(200)
            self.end_headers()
            payload = f"active_tasks={active_tasks};cancelled={int(is_cancelled())}"
            self.wfile.write(payload.encode("utf-8"))
            return

        if parsed.path != "/cancel":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"not found")
            return

        try:
            result = cancel_all_background_tasks(force=False)
            self.send_response(200)
            self.end_headers()
            payload = (
                f"cancelled_tasks={int(result.get('cancelled_tasks', 0))};"
                f"terminated_executors={int(result.get('terminated_executors', 0))};"
                f"terminated_processes={int(result.get('terminated_processes', 0))}"
            )
            self.wfile.write(payload.encode("utf-8"))
        except Exception as exc:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(str(exc).encode("utf-8", errors="ignore"))


def ensure_emergency_stop_server() -> Dict[str, Any]:
    """Start or reuse a tiny localhost HTTP server for emergency-stop requests."""
    global _EMERGENCY_STOP_SERVER, _EMERGENCY_STOP_THREAD, _EMERGENCY_STOP_PORT

    with _CANCEL_LOCK:
        thread_alive = _EMERGENCY_STOP_THREAD is not None and _EMERGENCY_STOP_THREAD.is_alive()
        if _EMERGENCY_STOP_SERVER is None or not thread_alive:
            server = ThreadingHTTPServer((_EMERGENCY_STOP_HOST, 0), _EmergencyStopRequestHandler)
            server.daemon_threads = True
            thread = threading.Thread(
                target=server.serve_forever,
                name="cfrp-emergency-stop-server",
                daemon=True,
            )
            thread.start()
            _EMERGENCY_STOP_SERVER = server
            _EMERGENCY_STOP_THREAD = thread
            _EMERGENCY_STOP_PORT = int(server.server_port)

        return {
            "host": _EMERGENCY_STOP_HOST,
            "port": int(_EMERGENCY_STOP_PORT),
            "token": _EMERGENCY_STOP_TOKEN,
            "cancel_url": (
                f"http://{_EMERGENCY_STOP_HOST}:{int(_EMERGENCY_STOP_PORT)}"
                f"/cancel?token={_EMERGENCY_STOP_TOKEN}"
            ),
            "status_url": (
                f"http://{_EMERGENCY_STOP_HOST}:{int(_EMERGENCY_STOP_PORT)}"
                f"/status?token={_EMERGENCY_STOP_TOKEN}"
            ),
        }


def _list_child_processes() -> List[Dict[str, Any]]:
    """Best-effort listing of child processes for orphan detection."""
    processes: List[Dict[str, Any]] = []
    try:
        import psutil  # type: ignore
        current = psutil.Process()
        for child in current.children(recursive=True):
            try:
                processes.append({
                    "pid": child.pid,
                    "name": child.name(),
                    "status": child.status(),
                    "cmdline": " ".join(child.cmdline()[:3]) if child.cmdline() else "",
                })
            except Exception:
                continue
        return processes
    except Exception:
        pass

    try:
        for p in mp.active_children():
            processes.append({
                "pid": p.pid,
                "name": getattr(p, "name", "") if p is not None else "",
                "status": "alive" if p.is_alive() else "stopped",
                "cmdline": "",
            })
    except Exception:
        pass
    return processes


def _terminate_pid(pid: int, force: bool = False) -> bool:
    try:
        import psutil  # type: ignore
        proc = psutil.Process(int(pid))
        if force:
            proc.kill()
        else:
            proc.terminate()
        return True
    except Exception:
        try:
            sig = signal.SIGTERM
            if force and hasattr(signal, "SIGKILL"):
                sig = signal.SIGKILL
            os.kill(int(pid), sig)
            return True
        except Exception:
            return False


# =============================================================================
# 后台任务管理器 (单例模式)
# =============================================================================
class BackgroundTaskManager:
    """
    后台任务管理器
    
    功能：
    - 注册和追踪所有后台任务
    - 提供一键终止功能
    - 任务状态监控
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._tasks: Dict[str, TaskInfo] = {}
        self._executors: List[weakref.ref] = []  # 弱引用列表
        self._processes: List[mp.Process] = []
        self._futures: Dict[str, List[Future]] = {}  # task_id -> futures
        self._task_counter = 0
        self._manager_lock = threading.RLock()
        self._initialized = True
        
        # 清除之前的取消状态
        clear_cancel()
    
    def _generate_task_id(self) -> str:
        """生成唯一任务ID"""
        with self._manager_lock:
            self._task_counter += 1
            return f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._task_counter:04d}"
    
    def register_task(self, name: str, task_type: str = "unknown", 
                      total_items: int = 0,
                      task_key: Optional[str] = None) -> str:
        """
        注册新任务
        
        Parameters:
        -----------
        name : str
            任务名称
        task_type : str
            任务类型 ('process_pool', 'thread_pool', 'subprocess')
        total_items : int
            总任务项数（用于进度显示）
            
        Returns:
        --------
        str : 任务ID
        """
        task_id = self._generate_task_id()
        
        task_info = TaskInfo(
            task_id=task_id,
            name=name,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            task_type=task_type,
            total_items=total_items,
            task_key=str(task_key or "").strip(),
        )
        
        with self._manager_lock:
            self._tasks[task_id] = task_info
            self._futures[task_id] = []
        
        return task_id

    def acquire_task(
        self,
        name: str,
        task_type: str = "unknown",
        total_items: int = 0,
        task_key: Optional[str] = None,
    ):
        """Atomically reuse an active task instead of starting a duplicate run.

        Streamlit reruns the script after every widget interaction. A stable
        task key lets a page reconnect to the already-running task rather than
        submitting the same expensive operation again.
        """
        normalized_key = str(task_key or "").strip()
        with self._manager_lock:
            if normalized_key:
                active = [
                    task
                    for task in self._tasks.values()
                    if task.task_key == normalized_key
                    and task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
                ]
                if active:
                    active.sort(key=lambda task: task.created_at, reverse=True)
                    return active[0].task_id, False

            if not any(
                task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
                for task in self._tasks.values()
            ):
                clear_cancel()

            task_id = self._generate_task_id()
            task_info = TaskInfo(
                task_id=task_id,
                name=name,
                status=TaskStatus.PENDING,
                created_at=datetime.now(),
                task_type=task_type,
                total_items=total_items,
                task_key=normalized_key,
            )
            self._tasks[task_id] = task_info
            self._futures[task_id] = []
            return task_id, True

    def get_task_snapshot(self, task_id: str) -> Optional[TaskInfo]:
        """Return a stable snapshot that can be rendered after a rerun."""
        with self._manager_lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            return TaskInfo(
                task_id=task.task_id,
                name=task.name,
                status=task.status,
                created_at=task.created_at,
                started_at=task.started_at,
                finished_at=task.finished_at,
                progress=float(task.progress),
                total_items=int(task.total_items),
                processed_items=int(task.processed_items),
                error_message=str(task.error_message or ""),
                task_type=task.task_type,
                task_key=task.task_key,
                message=str(task.message or ""),
                result=task.result,
                metadata=dict(task.metadata or {}),
                heartbeat_at=task.heartbeat_at,
            )

    def get_task_by_key(self, task_key: str) -> Optional[TaskInfo]:
        """Return the newest task registered with ``task_key``."""
        normalized_key = str(task_key or "").strip()
        if not normalized_key:
            return None
        with self._manager_lock:
            matches = [
                task
                for task in self._tasks.values()
                if task.task_key == normalized_key
            ]
            if not matches:
                return None
            matches.sort(key=lambda task: task.created_at, reverse=True)
            task = matches[0]
            return self.get_task_snapshot(task.task_id)

    def update_task(
        self,
        task_id: str,
        *,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Update human-readable status without touching Streamlit from workers."""
        with self._manager_lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            if message is not None:
                task.message = str(message)
            if metadata:
                task.metadata.update(dict(metadata))
            task.heartbeat_at = datetime.now()
    
    def start_task(self, task_id: str):
        """标记任务开始"""
        with self._manager_lock:
            if task_id in self._tasks:
                self._tasks[task_id].status = TaskStatus.RUNNING
                self._tasks[task_id].started_at = datetime.now()
    
    def update_progress(self, task_id: str, processed: int):
        """更新任务进度"""
        with self._manager_lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.processed_items = processed
                if task.total_items > 0:
                    task.progress = (processed / task.total_items) * 100
                task.heartbeat_at = datetime.now()
    
    def complete_task(self, task_id: str, success: bool = True,
                      error_message: str = "", result: Any = None):
        """标记任务完成"""
        with self._manager_lock:
            if task_id in self._tasks:
                task = self._tasks[task_id]
                task.finished_at = datetime.now()
                cancelled = (not success) and (
                    task.status == TaskStatus.CANCELLED
                    or _looks_like_cancel_message(error_message)
                    or is_cancelled()
                )
                if cancelled:
                    task.status = TaskStatus.CANCELLED
                    task.error_message = error_message or task.error_message or "用户取消"
                    task.heartbeat_at = datetime.now()
                    return
                task.progress = 100.0 if success else task.progress
                task.status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                task.error_message = error_message
                if result is not None:
                    task.result = result
                task.heartbeat_at = datetime.now()
    
    def register_executor(self, executor):
        """注册执行器（ProcessPoolExecutor 或 ThreadPoolExecutor）"""
        with self._manager_lock:
            # 使用弱引用避免阻止垃圾回收
            self._executors.append(weakref.ref(executor))
            # 清理已失效的引用
            self._executors = [ref for ref in self._executors if ref() is not None]
    
    def register_future(self, task_id: str, future: Future):
        """注册 Future 对象"""
        with self._manager_lock:
            if task_id in self._futures:
                self._futures[task_id].append(future)
    
    def register_process(self, process: mp.Process):
        """注册子进程"""
        with self._manager_lock:
            self._processes.append(process)
            # 清理已结束的进程
            self._processes = [p for p in self._processes if p.is_alive()]
    
    def get_active_tasks(self) -> List[TaskInfo]:
        """获取所有活跃任务"""
        with self._manager_lock:
            return [
                task for task in self._tasks.values()
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
            ]

    def get_orphan_processes(self) -> List[Dict[str, Any]]:
        """检测未注册的子进程（可能是后台遗留任务）"""
        with self._manager_lock:
            registered_pids = set()
            for p in self._processes:
                try:
                    if p is not None and p.is_alive() and p.pid:
                        registered_pids.add(int(p.pid))
                except Exception:
                    continue

        children = _list_child_processes()
        orphans = []
        for info in children:
            try:
                pid = int(info.get("pid"))
            except Exception:
                continue
            if pid in registered_pids:
                continue
            orphans.append(info)
        return orphans

    def terminate_orphan_processes(self, force: bool = False) -> Dict[str, Any]:
        """终止未注册子进程"""
        orphans = self.get_orphan_processes()
        result = {"terminated": 0, "errors": []}
        for info in orphans:
            pid = info.get("pid")
            if pid is None:
                continue
            ok = _terminate_pid(int(pid), force=force)
            if ok:
                result["terminated"] += 1
            else:
                result["errors"].append(f"terminate pid {pid} failed")
        return result
    
    def get_all_tasks(self) -> List[TaskInfo]:
        """获取所有任务"""
        with self._manager_lock:
            return list(self._tasks.values())
    
    def get_task_count(self) -> Dict[str, int]:
        """获取任务计数统计"""
        with self._manager_lock:
            counts = {
                "total": len(self._tasks),
                "pending": 0,
                "running": 0,
                "completed": 0,
                "cancelled": 0,
                "failed": 0
            }
            for task in self._tasks.values():
                if task.status == TaskStatus.PENDING:
                    counts["pending"] += 1
                elif task.status == TaskStatus.RUNNING:
                    counts["running"] += 1
                elif task.status == TaskStatus.COMPLETED:
                    counts["completed"] += 1
                elif task.status == TaskStatus.CANCELLED:
                    counts["cancelled"] += 1
                elif task.status == TaskStatus.FAILED:
                    counts["failed"] += 1
            return counts
    
    def cancel_all_tasks(self, force: bool = False) -> Dict[str, Any]:
        """
        取消所有后台任务
        
        Parameters:
        -----------
        force : bool
            是否强制终止（True=强制kill，False=优雅终止）
            
        Returns:
        --------
        dict : 取消结果统计
        """
        result = {
            "cancelled_tasks": 0,
            "cancelled_futures": 0,
            "terminated_executors": 0,
            "terminated_processes": 0,
            "errors": []
        }
        
        # 1. 设置全局取消标志
        request_cancel()
        
        with self._manager_lock:
            # 2. 取消所有注册的 Future
            for task_id, futures in self._futures.items():
                for future in futures:
                    try:
                        if not future.done():
                            cancelled = future.cancel()
                            if cancelled:
                                result["cancelled_futures"] += 1
                    except Exception as e:
                        result["errors"].append(f"取消Future失败: {e}")
            
            # 3. 关闭所有执行器
            for ref in self._executors:
                executor = ref()
                if executor is not None:
                    try:
                        executor.shutdown(wait=False, cancel_futures=True)
                        result["terminated_executors"] += 1
                    except TypeError:
                        # Python 3.8 不支持 cancel_futures 参数
                        try:
                            executor.shutdown(wait=False)
                            result["terminated_executors"] += 1
                        except Exception as e:
                            result["errors"].append(f"关闭执行器失败: {e}")
                    except Exception as e:
                        result["errors"].append(f"关闭执行器失败: {e}")
            
            # 4. 终止所有子进程
            for process in self._processes:
                if process.is_alive():
                    try:
                        if force:
                            process.kill()
                        else:
                            process.terminate()
                        result["terminated_processes"] += 1
                    except Exception as e:
                        result["errors"].append(f"终止进程失败: {e}")
            
            # 5. 更新任务状态
            for task_id, task in self._tasks.items():
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    task.status = TaskStatus.CANCELLED
                    task.finished_at = datetime.now()
                    task.error_message = "用户取消"
                    result["cancelled_tasks"] += 1
            
            # 6. 清理
            self._executors.clear()
            self._processes.clear()
            self._futures.clear()
        
        return result
    
    def clear_completed_tasks(self):
        """清除已完成的任务记录"""
        with self._manager_lock:
            completed_ids = [
                task_id for task_id, task in self._tasks.items()
                if task.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED)
            ]
            for task_id in completed_ids:
                del self._tasks[task_id]
                if task_id in self._futures:
                    del self._futures[task_id]
    
    def reset(self):
        """重置任务管理器"""
        self.cancel_all_tasks(force=True)
        with self._manager_lock:
            self._tasks.clear()
            self._futures.clear()
            self._executors.clear()
            self._processes.clear()
        clear_cancel()


# =============================================================================
# 全局任务管理器实例
# =============================================================================
_task_manager: Optional[BackgroundTaskManager] = None


def get_task_manager() -> BackgroundTaskManager:
    """获取全局任务管理器实例"""
    global _task_manager
    if _task_manager is None:
        _task_manager = BackgroundTaskManager()
    return _task_manager


# =============================================================================
# 便捷函数
# =============================================================================
def cancel_all_background_tasks(force: bool = False) -> Dict[str, Any]:
    """一键取消所有后台任务"""
    manager = get_task_manager()
    return manager.cancel_all_tasks(force=force)


def get_active_task_count() -> int:
    """获取活跃任务数量"""
    manager = get_task_manager()
    return len(manager.get_active_tasks())


def get_task_summary() -> Dict[str, Any]:
    """获取任务摘要"""
    manager = get_task_manager()
    counts = manager.get_task_count()
    active_tasks = manager.get_active_tasks()
    
    return {
        "counts": counts,
        "active_tasks": [
            {
                "id": t.task_id,
                "name": t.name,
                "type": t.task_type,
                "progress": t.progress,
                "status": t.status.value
            }
            for t in active_tasks
        ]
    }


def stop_gpu_inference_tasks() -> Dict[str, Any]:
    """
    ★ 专门停止GPU推理任务
    
    这个函数会：
    1. 设置取消标志
    2. 清理CUDA缓存
    3. 终止相关任务
    """
    result = {
        "cancelled": 0,
        "cuda_cleared": False,
        "errors": []
    }
    
    # 1. 设置取消标志
    request_cancel()
    
    # 2. 尝试清理CUDA缓存
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            result["cuda_cleared"] = True
    except ImportError:
        pass
    except Exception as e:
        result["errors"].append(f"清理CUDA缓存失败: {e}")
    
    # 3. 取消相关任务
    manager = get_task_manager()
    with manager._manager_lock:
        for task_id, task in manager._tasks.items():
            if task.task_type in ('transformer_inference', 'gpu_inference', 'dl_correction'):
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    task.status = TaskStatus.CANCELLED
                    task.finished_at = datetime.now()
                    task.error_message = "GPU任务已停止"
                    result["cancelled"] += 1
    
    return result


def graceful_shutdown() -> Dict[str, Any]:
    """
    ★ 优雅关闭应用
    
    这个函数会：
    1. 停止所有后台任务
    2. 清理GPU资源
    3. 关闭所有执行器
    4. 返回关闭统计
    
    用于应用退出时调用。
    """
    result = {
        "tasks_cancelled": 0,
        "executors_closed": 0,
        "processes_terminated": 0,
        "cuda_cleared": False,
        "streamlit_stopped": False,
        "errors": []
    }
    
    print("\n" + "="*60)
    print("🛑 正在优雅关闭应用...")
    print("="*60)
    
    # 1. 停止所有后台任务
    try:
        manager = get_task_manager()
        cancel_result = manager.cancel_all_tasks(force=True)
        result["tasks_cancelled"] = cancel_result.get("cancelled_tasks", 0)
        result["executors_closed"] = cancel_result.get("terminated_executors", 0)
        result["processes_terminated"] = cancel_result.get("terminated_processes", 0)
        print(f"   ✓ 已取消 {result['tasks_cancelled']} 个任务")
        print(f"   ✓ 已关闭 {result['executors_closed']} 个执行器")
        print(f"   ✓ 已终止 {result['processes_terminated']} 个进程")
    except Exception as e:
        result["errors"].append(f"取消任务失败: {e}")
        print(f"   ✗ 取消任务失败: {e}")
    
    # 2. 清理GPU资源
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            result["cuda_cleared"] = True
            print("   ✓ 已清理GPU资源")
    except ImportError:
        pass
    except Exception as e:
        result["errors"].append(f"清理GPU失败: {e}")
    
    # 3. 尝试停止Streamlit（如果在Streamlit环境中）
    try:
        import streamlit as st
        if hasattr(st, 'stop'):
            # 注意：这只是标记，实际停止需要用户刷新页面
            result["streamlit_stopped"] = True
            print("   ✓ 已标记Streamlit停止")
    except ImportError:
        pass
    except Exception as e:
        result["errors"].append(f"停止Streamlit失败: {e}")
    
    # 4. 重置任务管理器
    try:
        manager = get_task_manager()
        manager.reset()
        print("   ✓ 已重置任务管理器")
    except Exception as e:
        result["errors"].append(f"重置任务管理器失败: {e}")
    
    print("="*60)
    print("✓ 关闭完成")
    print("="*60 + "\n")
    
    return result


def emergency_stop() -> Dict[str, Any]:
    """
    ★ 紧急停止 - 强制终止所有任务
    
    比graceful_shutdown更激进，用于进程卡死时。
    """
    import signal
    import sys
    
    result = {
        "force_stopped": True,
        "errors": []
    }
    
    print("\n" + "="*60)
    print("🚨 紧急停止 - 强制终止所有任务")
    print("="*60)
    
    # 1. 强制取消
    request_cancel()
    
    # 2. 强制清理GPU
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 重置所有CUDA设备
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
    except:
        pass
    
    # 3. 强制终止进程池
    try:
        manager = get_task_manager()
        manager.cancel_all_tasks(force=True)
    except:
        pass
    
        # 4. 杀死所有子进程
        try:
            killed = 0
            children = _list_child_processes()
            for child in children:
                pid = child.get("pid")
                if pid is None:
                    continue
                if _terminate_pid(int(pid), force=True):
                    killed += 1
        except Exception as e:
            result["errors"].append(f"杀死子进程失败: {e}")
    
    print("✓ 紧急停止完成")
    print("="*60 + "\n")
    
    return result


# =============================================================================
# 可取消的执行器包装器
# =============================================================================
class CancellableProcessPoolExecutor(ProcessPoolExecutor):
    """支持取消功能的 ProcessPoolExecutor 包装器"""
    
    def __init__(self, *args, task_name: str = "并行任务", **kwargs):
        super().__init__(*args, **kwargs)
        self._task_manager = get_task_manager()
        self._task_id = self._task_manager.register_task(
            name=task_name,
            task_type="process_pool"
        )
        self._task_manager.register_executor(self)
        self._task_manager.start_task(self._task_id)
    
    @property
    def task_id(self) -> str:
        return self._task_id
    
    def submit(self, fn, *args, **kwargs):
        if is_cancelled():
            raise RuntimeError("任务已被取消")
        future = super().submit(fn, *args, **kwargs)
        self._task_manager.register_future(self._task_id, future)
        return future
    
    def __exit__(self, *args):
        super().__exit__(*args)
        self._task_manager.complete_task(self._task_id)


class CancellableThreadPoolExecutor(ThreadPoolExecutor):
    """支持取消功能的 ThreadPoolExecutor 包装器"""
    
    def __init__(self, *args, task_name: str = "线程任务", **kwargs):
        super().__init__(*args, **kwargs)
        self._task_manager = get_task_manager()
        self._task_id = self._task_manager.register_task(
            name=task_name,
            task_type="thread_pool"
        )
        self._task_manager.register_executor(self)
        self._task_manager.start_task(self._task_id)
    
    @property
    def task_id(self) -> str:
        return self._task_id
    
    def submit(self, fn, *args, **kwargs):
        if is_cancelled():
            raise RuntimeError("任务已被取消")
        future = super().submit(fn, *args, **kwargs)
        self._task_manager.register_future(self._task_id, future)
        return future
    
    def __exit__(self, *args):
        super().__exit__(*args)
        self._task_manager.complete_task(self._task_id)


# =============================================================================
# Keras 取消回调（用于 TensorFlow 模型训练取消）
# =============================================================================
class KerasCancellationCallback:
    """
    Keras 训练取消回调
    
    在每个 epoch 结束时检查取消标志，如果被取消则停止训练。
    
    用法:
        from core.task_manager import KerasCancellationCallback, is_cancelled
        
        callback = KerasCancellationCallback()
        model.fit(X, y, callbacks=[callback])
    """
    
    def __init__(self, check_interval: int = 1):
        """
        Parameters:
        -----------
        check_interval : int
            每隔多少个 epoch 检查一次取消状态（默认每个 epoch 检查）
        """
        self.check_interval = check_interval
        self._epoch_count = 0
        
    def set_model(self, model):
        self.model = model
        
    def on_epoch_end(self, epoch, logs=None):
        self._epoch_count += 1
        if self._epoch_count % self.check_interval == 0:
            if is_cancelled():
                print("\n⚠️ 检测到取消请求，正在停止训练...")
                self.model.stop_training = True
                
    def on_train_begin(self, logs=None):
        self._epoch_count = 0
        
    def on_train_end(self, logs=None):
        pass
        
    def on_batch_end(self, batch, logs=None):
        pass


def get_keras_cancellation_callback():
    """获取 Keras 取消回调实例"""
    try:
        # 尝试创建一个兼容 Keras 的回调
        try:
            from tensorflow.keras.callbacks import Callback
        except ImportError:
            try:
                from keras.callbacks import Callback
            except ImportError:
                return None
        
        class _KerasCancellationCallback(Callback):
            """Keras 取消回调的正式实现"""
            
            def __init__(self):
                super().__init__()
                self._cancelled = False
                
            def on_epoch_end(self, epoch, logs=None):
                if is_cancelled():
                    print(f"\n⚠️ Epoch {epoch+1}: 检测到取消请求，正在停止训练...")
                    self.model.stop_training = True
                    self._cancelled = True
                    
            def was_cancelled(self) -> bool:
                return self._cancelled
                
        return _KerasCancellationCallback()
    except Exception:
        return None


# =============================================================================
# Streamlit UI 组件
# =============================================================================
def render_task_manager_ui():
    """
    渲染任务管理器 UI 组件（用于 Streamlit 侧边栏）
    
    ★ 优化版：添加GPU任务停止和紧急停止功能
    
    Returns:
    --------
    bool : 是否点击了取消按钮
    """
    import streamlit as st
    
    manager = get_task_manager()
    counts = manager.get_task_count()
    active_tasks = manager.get_active_tasks()
    orphan_processes = manager.get_orphan_processes()
    
    st.markdown("---")
    st.markdown("### 🔄 后台任务")
    
    # 任务计数
    running_count = counts["running"] + counts["pending"]
    
    if running_count > 0:
        st.warning(f"⚡ {running_count} 个任务运行中")
        
        # 显示活跃任务详情
        for task in active_tasks[:3]:  # 最多显示3个
            progress_text = f"{task.progress:.0f}%" if task.total_items > 0 else "进行中"
            task_icon = "🤖" if 'transformer' in task.task_type.lower() or 'gpu' in task.task_type.lower() else "🔄"
            st.caption(f"{task_icon} {task.name}: {progress_text}")
        
        if len(active_tasks) > 3:
            st.caption(f"... 还有 {len(active_tasks) - 3} 个任务")
        
        # ★ 停止按钮组
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏹️ 停止全部", key="btn_stop_all_tasks", type="primary",
                        help="优雅地停止所有后台任务"):
                result = manager.cancel_all_tasks(force=False)
                st.success(f"已停止 {result['cancelled_tasks']} 个任务")
                return True
        with col2:
            if st.button("🛑 强制终止", key="btn_force_stop_tasks",
                        help="强制终止所有后台任务（可能导致数据丢失）"):
                result = manager.cancel_all_tasks(force=True)
                st.warning(f"已强制终止 {result['cancelled_tasks']} 个任务")
                return True
        
        # ★ GPU专用停止按钮
        gpu_tasks = [t for t in active_tasks if 'transformer' in t.task_type.lower() 
                     or 'gpu' in t.task_type.lower() or 'dl' in t.task_type.lower()]
        if gpu_tasks:
            if st.button("🎮 停止GPU任务", key="btn_stop_gpu_tasks",
                        help="停止所有GPU/深度学习任务并清理显存"):
                result = stop_gpu_inference_tasks()
                st.info(f"已停止 {result['cancelled']} 个GPU任务")
                if result['cuda_cleared']:
                    st.caption("✓ 已清理GPU缓存")
                return True
    else:
        st.caption("✅ 无后台任务运行中")
        
        # 显示历史统计
        if counts["total"] > 0:
            st.caption(
                f"历史: {counts['completed']}完成 | "
                f"{counts['cancelled']}取消 | {counts['failed']}失败"
            )
            if st.button("🗑️ 清除历史", key="btn_clear_task_history"):
                manager.clear_completed_tasks()
                st.rerun()

    # --- orphan process detection ---
    if orphan_processes:
        st.warning(f"⚠️ 检测到 {len(orphan_processes)} 个未注册后台进程")
        with st.expander("查看未注册进程", expanded=False):
            for info in orphan_processes[:5]:
                pid = info.get("pid")
                name = info.get("name", "")
                status = info.get("status", "")
                cmdline = info.get("cmdline", "")
                st.caption(f"pid={pid} {name} {status} {cmdline}".strip())
            if len(orphan_processes) > 5:
                st.caption(f"... 还有 {len(orphan_processes) - 5} 个进程")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏹️ 停止未注册任务", key="btn_stop_orphans"):
                result = manager.terminate_orphan_processes(force=False)
                st.success(f"已终止 {result['terminated']} 个进程")
                st.rerun()
        with col2:
            if st.button("🛑 强制终止未注册任务", key="btn_force_stop_orphans"):
                result = manager.terminate_orphan_processes(force=True)
                st.warning(f"已强制终止 {result['terminated']} 个进程")
                st.rerun()
    
    # ★ 紧急停止按钮（始终显示）
    with st.expander("🚨 紧急控制", expanded=False):
        st.caption("用于任务卡死时的紧急操作")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重置管理器", key="btn_reset_manager"):
                manager.reset()
                st.info("已重置")
                st.rerun()
        with col2:
            if st.button("🚨 紧急停止", key="btn_emergency_stop", type="secondary",
                        help="强制终止所有任务和子进程"):
                result = emergency_stop()
                st.warning("已执行紧急停止")
                st.rerun()
    
    return False


def render_task_control_expander():
    """
    渲染任务控制展开面板（更详细的版本）
    """
    import streamlit as st
    
    manager = get_task_manager()
    
    with st.expander("🔧 后台任务控制", expanded=False):
        counts = manager.get_task_count()
        all_tasks = manager.get_all_tasks()
        
        # 统计信息
        col1, col2, col3 = st.columns(3)
        col1.metric("运行中", counts["running"])
        col2.metric("已完成", counts["completed"])
        col3.metric("已取消/失败", counts["cancelled"] + counts["failed"])
        
        # 任务列表
        if all_tasks:
            st.markdown("#### 任务列表")
            for task in all_tasks[-10:]:  # 显示最近10个
                status_emoji = {
                    TaskStatus.PENDING: "⏳",
                    TaskStatus.RUNNING: "🔄",
                    TaskStatus.COMPLETED: "✅",
                    TaskStatus.CANCELLED: "⏹️",
                    TaskStatus.FAILED: "❌"
                }.get(task.status, "❓")
                
                st.caption(
                    f"{status_emoji} [{task.task_id[-8:]}] {task.name} "
                    f"({task.task_type}) - {task.progress:.0f}%"
                )
        # orphan process snapshot
        orphan_processes = manager.get_orphan_processes()
        if orphan_processes:
            st.markdown("#### 未注册进程")
            for info in orphan_processes[:5]:
                pid = info.get("pid")
                name = info.get("name", "")
                status = info.get("status", "")
                st.caption(f"pid={pid} {name} {status}".strip())
            if len(orphan_processes) > 5:
                st.caption(f"... 还有 {len(orphan_processes) - 5} 个进程")
        
        # 控制按钮
        st.markdown("#### 控制")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("⏹️ 停止全部", key="exp_stop_all"):
                result = manager.cancel_all_tasks(force=False)
                st.success(f"已停止 {result['cancelled_tasks']} 个任务")
                st.rerun()
        
        with col2:
            if st.button("🛑 强制终止", key="exp_force_stop"):
                result = manager.cancel_all_tasks(force=True)
                st.warning(f"已强制终止")
                st.rerun()
        
        with col3:
            if st.button("🔄 重置管理器", key="exp_reset"):
                manager.reset()
                st.info("已重置")
                st.rerun()
