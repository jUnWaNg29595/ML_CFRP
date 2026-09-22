# -*- coding: utf-8 -*-
"""Model import/export utilities.

Goal:
- Export a trained model (or sklearn Pipeline) into a single portable file.
- Import it back for prediction without retraining.

Format:
- joblib-serialized dict (a.k.a. "artifact")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import contextlib
import io
import time

from .molecular_feature_workflow import MolecularFeatureWorkflow
from .process_pls import PROCESS_PLS_SCHEMA_VERSION

try:
    import joblib  # sklearn dependency, but import defensively
except Exception as e:  # pragma: no cover
    joblib = None  # type: ignore

ARTIFACT_VERSION = "1.0"


def workflow_to_artifact_extra(workflow: Any) -> Dict[str, Any]:
    """Return workflow metadata fields shared by model and process exports."""
    if workflow is None:
        return {}
    if isinstance(workflow, dict):
        workflow = MolecularFeatureWorkflow.from_dict(workflow)
    if not isinstance(workflow, MolecularFeatureWorkflow):
        raise TypeError("workflow must be a MolecularFeatureWorkflow or mapping")
    return {
        "molecular_feature_workflow": workflow.to_dict(),
        "final_feature_names": list(workflow.final_feature_names),
        "feature_source_map": dict(workflow.feature_source_map),
        "workflow_hash": workflow.workflow_hash,
        "workflow_schema_version": workflow.schema_version,
    }


def process_pls_to_artifact_extra(config: Any) -> Dict[str, Any]:
    """Return compact, versioned process PLS metadata for an artifact."""
    if not isinstance(config, dict):
        return {}
    workflow = dict(config)
    return {
        "process_pls_workflow": workflow,
        "process_pls_schema_version": workflow.get("schema_version"),
        "process_pls_workflow_hash": workflow.get("workflow_hash"),
    }


def restore_process_pls_metadata(payload: Any) -> Optional[Dict[str, Any]]:
    """Extract and validate process PLS metadata from an artifact or config payload."""
    if not isinstance(payload, dict):
        return None
    extra = payload.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    workflow = (
        extra.get("process_pls_workflow")
        if "process_pls_workflow" in extra
        else payload.get("process_pls_workflow")
    )
    if not isinstance(workflow, dict):
        return None
    try:
        schema_version = int(workflow.get("schema_version", -1))
    except (TypeError, ValueError):
        schema_version = -1
    if schema_version != PROCESS_PLS_SCHEMA_VERSION:
        raise ValueError("导入模型的工艺 PLS workflow 版本不受支持")
    return dict(workflow)


def create_model_artifact(
    *,
    model_name: str,
    target_col: str,
    feature_cols: List[str],
    model: Any = None,
    pipeline: Any = None,
    scaler: Any = None,
    imputer: Any = None,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    contract_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a serializable artifact dict."""
    merged_extra = dict(extra or {})
    if contract_context:
        for key in ("prediction_contract", "registry_snapshot", "dataset_manifest", "feature_audit"):
            value = contract_context.get(key)
            if value is None and key not in contract_context:
                continue
            if key in merged_extra and merged_extra[key] != value:
                raise ValueError(f"extra 与 contract_context 的 {key} 内容冲突")
            merged_extra[key] = value
    artifact: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": int(time.time()),
        "model_name": str(model_name),
        "target_col": str(target_col),
        "feature_cols": list(feature_cols) if feature_cols is not None else [],
        "metrics": metrics or {},
        "extra": merged_extra,
    }

    # Prefer saving the Pipeline if available (safer/complete: includes preprocessing)
    if pipeline is not None:
        artifact["pipeline"] = pipeline
        artifact["model"] = model  # keep for convenience
        artifact["scaler"] = None
        artifact["imputer"] = None
    else:
        artifact["pipeline"] = None
        artifact["model"] = model
        artifact["scaler"] = scaler
        artifact["imputer"] = imputer

    return artifact


def dumps_artifact(artifact: Dict[str, Any], *, compress: int = 3) -> bytes:
    """Serialize an artifact dict to bytes using joblib."""
    if joblib is None:
        raise ImportError("joblib not available. Please install joblib (or scikit-learn).")

    buf = io.BytesIO()
    joblib.dump(artifact, buf, compress=compress)
    return buf.getvalue()


def _cuda_device_usable(index: int) -> bool:
    """判断 ``cuda:{index}`` 在当前机器是否可用。"""
    try:
        import torch

        return bool(torch.cuda.is_available()) and 0 <= index < int(torch.cuda.device_count())
    except Exception:  # pragma: no cover - torch 为可选依赖
        return False


def _portable_map_location(storage, location):
    """torch.load 的 map_location 钩子：设备存在则原地恢复，不存在才留在 CPU。

    这是本函数与“一律 map to cpu”的关键区别：

    - 模型保存时权重在 ``cuda:0``，且当前机器有该设备 → 权重回 ``cuda:0``。
      这一点至关重要：模型 pickle 里通常还保存着 ``self.device = 'cuda:0'``
      这样的**普通字符串属性**（map_location 改不了它）。若把权重强制改到 CPU，
      而 forward 里 ``x.to(self.device)`` 仍把输入搬到 ``cuda:0``，就会报
      ``Expected all tensors to be on the same device, but found at least two
      devices, cpu and cuda:0!``（实测自建 NN 模型踩中）。
    - 模型保存在 ``cuda:1``，而当前机器只有 1 块 GPU → 返回原 storage
      （torch 重建 storage 时先在 CPU 上分配，返回它即留在 CPU），
      否则反序列化直接失败（用户实测报错）。

    注意：可调用版 map_location 在 torch 的 legacy/zip 两条路径里都要求
    **返回 storage 对象**（不是设备字符串）。进入本函数时 storage 已经在
    CPU 上重建完毕，因此“留在 CPU”就是原样返回。
    """
    text = str(location or "").strip().lower()
    if not text.startswith("cuda"):
        return storage
    raw = text.split(":", 1)[1] if ":" in text else "0"
    try:
        index = int(raw)
    except (TypeError, ValueError):
        index = 0
    if not _cuda_device_usable(index):
        return storage  # 已在 CPU：保持不变
    try:
        moved = storage.cuda(index)
        return moved if moved is not None else storage
    except Exception:  # pragma: no cover - 特殊 storage 类型不支持 .cuda()
        return storage


def _repair_stale_device_attributes(obj: Any, *, max_depth: int = 4) -> None:
    """尽力修复“device 字符串属性指向不存在的 CUDA 设备”的模型。

    背景：即使 tensor 被映射回 CPU，模型 pickle 里的 ``self.device = 'cuda:1'``
    仍是不变的普通属性。若该设备在当前机器不存在，forward 会因设备不一致失败。

    修复条件（全部满足才改，尽量保守）：
    1. 对象是 ``torch.nn.Module``；
    2. 它有名为 ``device`` 的字符串属性，形如 ``cuda:N`` 且该设备**不可用**；
    3. 它的全部参数与 buffer 都已在 CPU 上。

    只把属性改成 ``"cpu"``（这是唯一可行的目标设备）；不改动任何 tensor。
    """
    try:
        import torch
        import torch.nn as nn
    except Exception:  # pragma: no cover
        return
    if not isinstance(obj, nn.Module):
        return

    device_attr = getattr(obj, "device", None)
    if not (isinstance(device_attr, str) and device_attr.strip().lower().startswith("cuda")):
        return
    raw = device_attr.strip().lower().split(":", 1)[1] if ":" in device_attr else "0"
    try:
        index = int(raw)
    except (TypeError, ValueError):
        index = 0
    if _cuda_device_usable(index):
        return  # 设备可用，不动

    tensors = list(obj.parameters(recurse=True)) + list(obj.buffers(recurse=True))
    if tensors and all(t.device.type == "cpu" for t in tensors):
        try:
            obj.device = "cpu"
        except Exception:  # pragma: no cover - 属性只读等罕见情况
            return

    if max_depth <= 0:
        return
    for child in obj.children():
        _repair_stale_device_attributes(child, max_depth=max_depth - 1)


def _repair_loaded_object_devices(obj: Any) -> None:
    """遍历刚加载的 artifact，修复指向不存在 CUDA 设备的 ``device`` 属性。"""
    try:
        import torch.nn as nn
    except Exception:  # pragma: no cover
        return

    seen = 0
    limit = 512

    def visit(node, depth):
        nonlocal seen
        if seen >= limit or depth > 3:
            return
        if isinstance(node, nn.Module):
            seen += 1
            _repair_stale_device_attributes(node, max_depth=3)
            for child in node.children():
                visit(child, depth + 1)
            return
        if isinstance(node, dict):
            for value in list(node.values())[:64]:
                visit(value, depth + 1)
        elif isinstance(node, (list, tuple)):
            for value in list(node)[:64]:
                visit(value, depth + 1)

    visit(obj, 0)


@contextlib.contextmanager
def _torch_portable_load():
    """反序列化期间把 torch 的 storage 加载器改为“可迁移”的 map_location。

    问题
    ----
    模型若在 ``cuda:1`` 上训练并保存，artifact 里 pickle 了绑定该设备的
    tensor。在只有 1 块 GPU（或纯 CPU）的机器上反序列化时，PyTorch 的
    ``torch.storage._load_from_bytes`` 内部调用
    ``torch.load(io.BytesIO(b), weights_only=False)``，**未传 map_location**，
    于是尝试在 cuda:1 上重建 storage 并报错：

        Attempting to deserialize object on CUDA device 1 but
        torch.cuda.device_count() is 1. Please use torch.load with
        map_location to map your storages to an existing device.

    为什么必须打补丁
    ----------------
    ``joblib.load`` 不接受 ``map_location`` 参数，而错误发生在它内部调用的
    torch 反序列化钩子上，调用方无法直接传递。因此只能在反序列化期间把该钩子
    替换为带 ``map_location`` 的实现。

    ⚠️ 不能“一律映射到 CPU”（历史教训）
    ----------------------------------
    早期实现把所有 CUDA tensor 强制映射到 CPU，结果在**有 GPU 的机器**上引入了
    新回归：模型 pickle 里的 ``self.device = 'cuda:0'`` 是普通字符串属性，
    不会被映射；权重被改到 CPU 后，forward 里 ``x.to(self.device)`` 仍把输入
    搬到 ``cuda:0``，报
    ``Expected all tensors to be on the same device, but found at least two
    devices, cpu and cuda:0!``。

    因此现在用 :func:`_portable_map_location`（可调用版 map_location）：
    **设备存在则原地恢复（行为与未修复时一致），设备不存在才落 CPU**。

    影响面
    ------
    - 现有 GPU 上保存的模型：行为与修复前完全一致。
    - 保存自不可用设备的模型：能加载，且 ``device`` 属性会被
      :func:`_repair_loaded_object_devices` 同步修正为 cpu，避免 forward 报错。
    - 纯 CPU artifact 行为完全不变。
    - 补丁在 with 块结束时恢复，不污染全局 torch 状态。
    - torch 未安装或钩子不存在时静默跳过（不影响非 torch 模型的加载）。
    """
    try:
        import torch
        import torch.storage as torch_storage
    except Exception:  # pragma: no cover - torch 为可选依赖
        yield
        return

    original = getattr(torch_storage, "_load_from_bytes", None)
    if original is None:  # pragma: no cover - 旧版 torch 无此钩子
        yield
        return

    def _load_from_bytes_portable(b):
        return torch.load(
            io.BytesIO(b), map_location=_portable_map_location, weights_only=False
        )

    torch_storage._load_from_bytes = _load_from_bytes_portable
    try:
        yield
    finally:
        torch_storage._load_from_bytes = original


def loads_artifact(data: bytes) -> Dict[str, Any]:
    """Load artifact dict from bytes.

    反序列化时可迁移处理 CUDA tensor：设备存在则原地恢复（与训练时一致），
    设备不存在才落回 CPU，并同步修复指向不可用设备的 ``device`` 属性，
    使在别的 GPU 拓扑（如 ``cuda:1``）上保存的模型也能在当前机器正常预测
    （见 :func:`_torch_portable_load`）。
    """
    if joblib is None:
        raise ImportError("joblib not available. Please install joblib (or scikit-learn).")

    buf = io.BytesIO(data)
    with _torch_portable_load():
        obj = joblib.load(buf)
    _repair_loaded_object_devices(obj)

    # Backward compatibility:
    # - if user uploads a raw pipeline/model pickled by joblib, wrap it
    if isinstance(obj, dict) and ("pipeline" in obj or "model" in obj) and "artifact_version" in obj:
        return obj

    # raw sklearn Pipeline or estimator
    wrapped = create_model_artifact(
        model_name="ImportedModel",
        target_col="",
        feature_cols=[],
        model=obj,
        pipeline=obj if hasattr(obj, "predict") and hasattr(obj, "fit") and "Pipeline" in type(obj).__name__ else None,
    )
    return wrapped


def create_model_artifact_bytes(
    *,
    model_name: str,
    target_col: str,
    feature_cols: List[str],
    model: Any = None,
    pipeline: Any = None,
    scaler: Any = None,
    imputer: Any = None,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    contract_context: Optional[Dict[str, Any]] = None,
    compress: int = 3,
) -> bytes:
    artifact = create_model_artifact(
        model_name=model_name,
        target_col=target_col,
        feature_cols=feature_cols,
        model=model,
        pipeline=pipeline,
        scaler=scaler,
        imputer=imputer,
        metrics=metrics,
        extra=extra,
        contract_context=contract_context,
    )
    # artifact_hash 基于去除 hash 键后的规范化 payload 计算，写入 extra 供门禁核验。
    try:
        artifact.setdefault("extra", {})["artifact_hash"] = compute_artifact_hash(artifact)
    except (TypeError, ValueError):
        # joblib 对象无法 JSON 序列化时跳过内嵌 hash（发布层仍有文件 hash）。
        pass
    return dumps_artifact(artifact, compress=compress)


def _canonical_artifact_payload(artifact: Any) -> Dict[str, Any]:
    """Return a JSON-serializable payload without the artifact_hash key."""
    if isinstance(artifact, (bytes, bytearray)):
        from typing import cast
        payload = loads_artifact(bytes(artifact))
    else:
        payload = artifact
    if not isinstance(payload, dict):
        raise TypeError("artifact must be a dict or serialized bytes")
    cleaned = dict(payload)
    extra = cleaned.get("extra")
    if isinstance(extra, dict):
        extra = {key: value for key, value in extra.items() if key != "artifact_hash"}
        cleaned["extra"] = extra
    return cleaned


def compute_artifact_hash(artifact: Any) -> str:
    """sha256 of the canonical JSON (sorted keys) payload without artifact_hash.

    接受 artifact dict 或 joblib 序列化 bytes；对无法 JSON 序列化的对象（如
    sklearn 模型）以 repr 代替，保证 hash 仍可复现计算。
    """
    import hashlib
    import json as _json

    def _default(obj: Any) -> str:
        return f"<non-serializable:{type(obj).__name__}>"

    payload = _canonical_artifact_payload(artifact)
    encoded = _json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def artifact_hash_from_bytes(data: bytes) -> str:
    """Compute the artifact hash for joblib-serialized artifact bytes."""
    return compute_artifact_hash(data)


def load_model_artifact_bytes(data: bytes) -> Dict[str, Any]:
    return loads_artifact(data)
