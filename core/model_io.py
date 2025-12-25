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

try:
    import joblib  # sklearn dependency, but import defensively
except Exception as e:  # pragma: no cover
    joblib = None  # type: ignore

ARTIFACT_VERSION = "1.0"

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
) -> Dict[str, Any]:
    """Create a serializable artifact dict."""
    artifact: Dict[str, Any] = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": int(time.time()),
        "model_name": str(model_name),
        "target_col": str(target_col),
        "feature_cols": list(feature_cols) if feature_cols is not None else [],
        "metrics": metrics or {},
        "extra": extra or {},
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
    )
    return dumps_artifact(artifact, compress=compress)


def load_model_artifact_bytes(data: bytes) -> Dict[str, Any]:
    return loads_artifact(data)
