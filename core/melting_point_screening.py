"""Core melting-point prediction gates for candidate screening."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


_STATUS_PASS = "pass"
_STATUS_FAIL = "fail"
_STATUS_UNKNOWN = "unknown"


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def melting_point_filter_status(
    prediction_c: float,
    std_c: float,
    ad_score: float,
    limit_c: float,
    max_std_c: float,
    min_ad_score: float,
    ad_in_domain: Any = None,
) -> tuple[str, str]:
    """Return ``pass``, ``fail`` or ``unknown`` and a reason code."""
    prediction = _finite_number(prediction_c)
    if prediction is None:
        return _STATUS_UNKNOWN, "non_finite_prediction"

    standard_deviation = _finite_number(std_c)
    if standard_deviation is None:
        return _STATUS_UNKNOWN, "non_finite_std"
    if standard_deviation < 0:
        return _STATUS_UNKNOWN, "negative_std"

    applicability_score = _finite_number(ad_score)
    if applicability_score is None:
        return _STATUS_UNKNOWN, "non_finite_ad_score"

    limit = _finite_number(limit_c)
    maximum_standard_deviation = _finite_number(max_std_c)
    minimum_applicability_score = _finite_number(min_ad_score)
    if limit is None:
        return _STATUS_UNKNOWN, "non_finite_limit"
    if maximum_standard_deviation is None:
        return _STATUS_UNKNOWN, "non_finite_max_std"
    if minimum_applicability_score is None:
        return _STATUS_UNKNOWN, "non_finite_min_ad_score"

    if standard_deviation > maximum_standard_deviation:
        return _STATUS_FAIL, "std_exceeds_limit"
    if ad_in_domain is not None:
        try:
            if bool(pd.isna(ad_in_domain)):
                return _STATUS_UNKNOWN, "unknown_ad_domain"
        except (TypeError, ValueError):
            pass
        if isinstance(ad_in_domain, str):
            normalized_domain = ad_in_domain.strip().lower()
            if normalized_domain in {"false", "0", "no", "out", "out_of_domain"}:
                return _STATUS_UNKNOWN, "ad_out_of_domain"
            if normalized_domain in {"", "none", "nan", "unknown"}:
                return _STATUS_UNKNOWN, "unknown_ad_domain"
        elif ad_in_domain is False:
            return _STATUS_UNKNOWN, "ad_out_of_domain"
        elif ad_in_domain is not True:
            try:
                if not bool(ad_in_domain):
                    return _STATUS_UNKNOWN, "ad_out_of_domain"
            except Exception:
                return _STATUS_UNKNOWN, "unknown_ad_domain"
    if applicability_score < minimum_applicability_score:
        return _STATUS_FAIL, "ad_below_minimum"
    if prediction + standard_deviation > limit:
        return _STATUS_FAIL, "prediction_plus_std_exceeds_limit"

    return _STATUS_PASS, "within_limits"


def _role_is_available(
    frame: pd.DataFrame,
    prediction_col: str,
    std_col: str,
    ad_col: str,
) -> bool:
    return any(column in frame.columns for column in (prediction_col, std_col, ad_col))


def _gate_role(
    frame: pd.DataFrame,
    *,
    role: str,
    prediction_col: str,
    std_col: str,
    ad_col: str,
    ad_in_domain_col: str | None,
    limit_c: float,
    max_std_c: float,
    min_ad_score: float,
) -> tuple[pd.Series, pd.Series]:
    statuses: list[str] = []
    reasons: list[str] = []
    missing = object()

    for row in frame.itertuples(index=False, name=None):
        values = dict(zip(frame.columns, row))
        status, reason = melting_point_filter_status(
            values.get(prediction_col, missing),
            values.get(std_col, missing),
            values.get(ad_col, missing),
            limit_c,
            max_std_c,
            min_ad_score,
            values.get(ad_in_domain_col, missing) if ad_in_domain_col else None,
        )
        statuses.append(status)
        reasons.append(reason)

    index = frame.index
    return (
        pd.Series(statuses, index=index, name=f"{role}_mp_filter_status"),
        pd.Series(reasons, index=index, name=f"{role}_mp_filter_reason"),
    )


def apply_melting_point_gate(
    df: pd.DataFrame,
    *,
    role_col: str = "component_role",
    resin_prediction_col: str = "resin_mp_predicted_c",
    hardener_prediction_col: str = "hardener_mp_predicted_c",
    resin_std_col: str = "resin_mp_std_c",
    hardener_std_col: str = "hardener_mp_std_c",
    resin_ad_col: str = "resin_mp_ad_score",
    hardener_ad_col: str = "hardener_mp_ad_score",
    resin_ad_in_domain_col: str = "resin_mp_ad_in_domain",
    hardener_ad_in_domain_col: str = "hardener_mp_ad_in_domain",
    resin_limit_c: float,
    hardener_limit_c: float,
    max_std_c: float,
    min_ad_score: float,
    mode: str = "annotate",
) -> pd.DataFrame:
    """Annotate or strictly filter rows using available component roles.

    A role is available when at least one of its configured columns exists.
    Missing columns within an available role become ``unknown``; a completely
    absent role is skipped.  Thus resin-only frames can be screened without
    fabricating a hardener result.
    """
    del role_col
    if mode not in {"annotate", "strict"}:
        raise ValueError("mode must be 'annotate' or 'strict'")
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    result = df.copy()
    role_results: list[tuple[str, pd.Series, pd.Series]] = []
    role_specs = (
        (
            "resin",
            resin_prediction_col,
            resin_std_col,
            resin_ad_col,
            resin_ad_in_domain_col,
            resin_limit_c,
        ),
        (
            "hardener",
            hardener_prediction_col,
            hardener_std_col,
            hardener_ad_col,
            hardener_ad_in_domain_col,
            hardener_limit_c,
        ),
    )

    for role, prediction_col, std_col, ad_col, ad_in_domain_col, limit_c in role_specs:
        if not _role_is_available(result, prediction_col, std_col, ad_col):
            continue
        status, reason = _gate_role(
            result,
            role=role,
            prediction_col=prediction_col,
            std_col=std_col,
            ad_col=ad_col,
            ad_in_domain_col=ad_in_domain_col,
            limit_c=limit_c,
            max_std_c=max_std_c,
            min_ad_score=min_ad_score,
        )
        result[status.name] = status
        result[reason.name] = reason
        role_results.append((role, status, reason))

    if role_results:
        result["mp_filter_reason"] = [
            ";".join(
                f"{role}:{reason.iloc[row_number]}"
                for role, _, reason in role_results
                if reason.iloc[row_number] != "within_limits"
            )
            or "within_limits"
            for row_number in range(len(result))
        ]

    if mode == "annotate" or not role_results:
        return result

    strict_mask = pd.Series(True, index=result.index)
    for _, status, _ in role_results:
        strict_mask &= status.eq(_STATUS_PASS)
    return result.loc[strict_mask]

def _artifact_extra(artifact: Any) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        return {}
    extra = artifact.get("extra")
    return extra if isinstance(extra, dict) else {}


def _artifact_target_col(artifact: Any) -> str:
    extra = _artifact_extra(artifact)
    return str(extra.get("target_col") or (artifact.get("target_col") if isinstance(artifact, dict) else "") or "")


def is_melting_point_artifact(artifact: Any) -> bool:
    """Return whether an artifact is explicitly marked as a Celsius MP model."""
    extra = _artifact_extra(artifact)
    task_kind = str(extra.get("task_kind") or "").strip().lower()
    target_unit = str(extra.get("target_unit") or "").strip().upper()
    target_col = _artifact_target_col(artifact)
    return task_kind == "melting_point" and target_unit == "C" and bool(target_col)


def validate_melting_point_artifact(artifact: Any) -> dict[str, Any]:
    """Validate the basic explicit metadata contract for a melting-point artifact."""
    extra = _artifact_extra(artifact)
    target_col = _artifact_target_col(artifact)
    target_unit = str(extra.get("target_unit") or "").strip().upper()
    task_kind = str(extra.get("task_kind") or "").strip().lower()
    error_codes: list[str] = []
    if task_kind != "melting_point":
        error_codes.append("task_kind")
    if target_unit != "C":
        error_codes.append("target_unit")
    if not target_col:
        error_codes.append("target_col")
    return {
        "ok": not error_codes,
        "error_codes": error_codes,
        "task_kind": task_kind,
        "target_unit": target_unit,
        "target_col": target_col,
        "dataset_row_count": extra.get("dataset_row_count"),
        "workflow_hash": extra.get("workflow_hash"),
    }


def validate_melting_point_artifact_for_screening(
    artifact: Any,
    model_feature_cols: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Apply the strict workflow/feature compatibility contract before screening."""
    report = validate_melting_point_artifact(artifact)
    error_codes = list(report.get("error_codes") or [])
    extra = _artifact_extra(artifact)
    workflow = extra.get("molecular_feature_workflow")
    if not isinstance(workflow, dict) and isinstance(artifact, dict):
        workflow = artifact.get("molecular_feature_workflow")
    if not isinstance(workflow, dict):
        error_codes.append("missing_workflow")
        report["workflow_report"] = None
    else:
        try:
            from core.molecular_feature_workflow import normalize_workflow_config, validate_workflow_config

            normalized = normalize_workflow_config(workflow)
            workflow_report = validate_workflow_config(
                workflow,
                model_feature_cols=list(model_feature_cols or (artifact.get("feature_cols", []) if isinstance(artifact, dict) else []) or []),
            )
            report["workflow_report"] = workflow_report
            stored_hash = str(workflow.get("workflow_hash") or "").strip()
            if not stored_hash:
                error_codes.append("missing_workflow_hash")
            elif stored_hash != str(normalized.get("workflow_hash") or "").strip():
                error_codes.append("workflow_hash_mismatch")
            for code in ("missing_steps", "missing_features", "order_mismatch"):
                if workflow_report.get(code):
                    error_codes.append(code)
        except Exception:
            report["workflow_report"] = None
            error_codes.append("invalid_workflow")
    report["error_codes"] = list(dict.fromkeys(error_codes))
    report["ok"] = not report["error_codes"]
    return report

def _dataset_fingerprint(dataset: pd.DataFrame) -> str:
    import hashlib

    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame")
    frame = dataset.copy()
    columns = sorted(str(column) for column in frame.columns)
    frame = frame.loc[:, [column for column in columns if column in frame.columns]]
    frame = frame.sort_index(axis=1)
    payload = frame.to_json(orient="split", date_format="iso", default_handler=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_melting_point_artifact_extra(
    dataset: pd.DataFrame,
    *,
    workflow_hash: str | None = None,
    quality_policy: str = "high_quality_only",
) -> dict[str, Any]:
    """Build auditable metadata to embed in a melting-point model artifact."""
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("dataset must be a pandas DataFrame")
    roles = dataset.get("component_role", pd.Series(dtype="object")).fillna("unknown").astype(str)
    qualities = dataset.get("mp_quality", pd.Series(dtype="object")).fillna("unknown").astype(str)
    role_counts = {str(key): int(value) for key, value in roles.value_counts().sort_index().items()}
    quality_counts = {str(key): int(value) for key, value in qualities.value_counts().sort_index().items()}
    return {
        "task_kind": "melting_point",
        "target_unit": "C",
        "target_col": "mp_c",
        "dataset_fingerprint": _dataset_fingerprint(dataset),
        "dataset_row_count": int(len(dataset)),
        "role_counts": role_counts,
        "quality_counts": quality_counts,
        "quality_policy": str(quality_policy),
        "workflow_hash": workflow_hash,
    }


__all__ = [
    "apply_melting_point_gate",
    "melting_point_filter_status",
    "is_melting_point_artifact",
    "validate_melting_point_artifact",
    "validate_melting_point_artifact_for_screening",
    "build_melting_point_artifact_extra",
]
