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


def loads_artifact(data: bytes) -> Dict[str, Any]:
    """Load artifact dict from bytes."""
    if joblib is None:
        raise ImportError("joblib not available. Please install joblib (or scikit-learn).")

    buf = io.BytesIO(data)
    obj = joblib.load(buf)

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
