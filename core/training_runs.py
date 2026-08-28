# -*- coding: utf-8 -*-
"""训练记录（Run）管理

功能：
- 每次训练把关键指标、参数、训练曲线数据与图片落盘
- 方便在 Streamlit 中浏览历史训练记录
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import json
import os
import re

import pandas as pd


def _safe_name(name: str) -> str:
    name = str(name)
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]", "_", name)
    return name[:80] if len(name) > 80 else name


@dataclass
class TrainingRunSummary:
    run_id: str
    path: str
    model_name: str
    created_at: str
    task_kind: str = "regression"
    r2: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    roc_auc: Optional[float] = None


class TrainingRunManager:
    def __init__(self, base_dir: str = "results/training_runs"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        # 图像导出分辨率（像素密度）。默认 300dpi，可通过环境变量 CFRP_EXPORT_DPI 覆盖。
        # 例如：
        #   - Windows (PowerShell):  $env:CFRP_EXPORT_DPI = 600
        #   - Linux/macOS (bash):    export CFRP_EXPORT_DPI=600
        try:
            self.export_dpi = max(72, int(os.environ.get("CFRP_EXPORT_DPI", "300")))
        except Exception:
            self.export_dpi = 300

    def create_run_dir(self, model_name: str) -> Tuple[str, str]:
        # 同一秒内可能完成多次训练/优化，必须避免复用目录覆盖旧记录。
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model_name = _safe_name(model_name)
        base_run_id = f"{ts}_{safe_model_name}"
        run_id = base_run_id
        suffix = 1
        while os.path.exists(os.path.join(self.base_dir, run_id)):
            run_id = f"{base_run_id}_{suffix}"
            suffix += 1
        run_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(run_dir, exist_ok=False)
        return run_id, run_dir

    def save_run(
        self,
        model_name: str,
        metadata: Dict[str, Any],
        history_df: Optional[pd.DataFrame] = None,
        curve_fig: Any = None,
        extra_figs: Optional[Dict[str, Any]] = None,
        extra_tables: Optional[Dict[str, pd.DataFrame]] = None,
        model: Any = None,
        pipeline: Any = None,
        scaler: Any = None,
        imputer: Any = None,
        feature_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        contract_context: Optional[Dict[str, Any]] = None,
    ) -> TrainingRunSummary:
        run_id, run_dir = self.create_run_dir(model_name)

        meta = dict(metadata or {})
        meta.setdefault("model_name", model_name)
        meta.setdefault("run_id", run_id)
        meta.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))

        if isinstance(contract_context, dict):
            prediction_contract = contract_context.get("prediction_contract")
            registry_snapshot = contract_context.get("registry_snapshot")
            dataset_manifest = contract_context.get("dataset_manifest")
            feature_audit = contract_context.get("feature_audit")
            if isinstance(registry_snapshot, dict):
                meta.setdefault("feature_registry_version", registry_snapshot.get("registry_version"))
                meta.setdefault("feature_registry_hash", registry_snapshot.get("registry_hash"))
            meta.setdefault("dataset_id", (dataset_manifest or {}).get("dataset_id") if isinstance(dataset_manifest, dict) else contract_context.get("dataset_id"))
            if isinstance(dataset_manifest, dict):
                meta.setdefault("dataset_manifest_hash", dataset_manifest.get("manifest_hash"))
            elif contract_context.get("dataset_manifest_hash") is not None:
                meta.setdefault("dataset_manifest_hash", contract_context.get("dataset_manifest_hash"))
            if isinstance(feature_audit, dict):
                canonical = list(feature_audit.get("canonical_feature_cols") or [])
                effective = list(feature_audit.get("effective_feature_cols") or [])
                meta.setdefault("canonical_feature_count", len(canonical))
                meta.setdefault("effective_feature_count", len(effective))
                meta.setdefault("removed_feature_cols", list(feature_audit.get("removed_feature_cols") or []))
            # Persist only the contract-bearing keys. Arbitrary request/input fields
            # are intentionally excluded from the run directory.
            contract_payload = {
                key: contract_context[key]
                for key in ("prediction_contract", "registry_snapshot", "dataset_manifest", "feature_audit")
                if key in contract_context
            }
            with open(os.path.join(run_dir, "contract.json"), "w", encoding="utf-8") as f:
                json.dump(contract_payload, f, ensure_ascii=False, indent=2)

        with open(os.path.join(run_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if history_df is not None and not history_df.empty:
            history_df.to_csv(os.path.join(run_dir, "history.csv"), index=False, encoding="utf-8-sig")

        # 保存主曲线图
        if curve_fig is not None:
            try:
                curve_fig.savefig(
                    os.path.join(run_dir, "training_curve.png"),
                    dpi=self.export_dpi,
                    bbox_inches="tight",
                    pad_inches=0.05,
                )
            except Exception:
                pass

        if extra_figs:
            for name, fig in extra_figs.items():
                if fig is None:
                    continue
                try:
                    safe = _safe_name(name)
                    fig.savefig(
                        os.path.join(run_dir, f"{safe}.png"),
                        dpi=self.export_dpi,
                        bbox_inches="tight",
                        pad_inches=0.05,
                    )
                except Exception:
                    continue

        if extra_tables:
            for name, df in extra_tables.items():
                if df is None or getattr(df, "empty", True):
                    continue
                try:
                    safe = _safe_name(name)
                    df.to_csv(
                        os.path.join(run_dir, f"{safe}.csv"),
                        index=False,
                        encoding="utf-8-sig",
                    )
                except Exception:
                    continue

        # 保存模型文件（用于后续加载和SHAP分析）
        if model is not None or pipeline is not None:
            try:
                from .model_io import create_model_artifact_bytes

                model_bytes = create_model_artifact_bytes(
                    model_name=model_name,
                    target_col=target_col or "",
                    feature_cols=feature_cols or [],
                    model=model,
                    pipeline=pipeline,
                    scaler=scaler,
                    imputer=imputer,
                    metrics={
                        "r2": meta.get("r2"),
                        "rmse": meta.get("rmse"),
                        "mae": meta.get("mae"),
                        "accuracy": meta.get("accuracy"),
                        "f1": meta.get("f1"),
                        "roc_auc": meta.get("roc_auc"),
                    },
                    extra=extra,
                    contract_context=contract_context,
                )

                with open(os.path.join(run_dir, "model.pkl"), "wb") as f:
                    f.write(model_bytes)
            except Exception as e:
                # 模型保存失败不应阻止训练记录保存
                print(f"Warning: Failed to save model file: {e}")

        return TrainingRunSummary(
            run_id=run_id,
            path=run_dir,
            model_name=model_name,
            created_at=str(meta.get("created_at", "")),
            task_kind=str(meta.get("task_kind", "regression")),
            r2=_to_float(meta.get("r2")),
            rmse=_to_float(meta.get("rmse")),
            mae=_to_float(meta.get("mae")),
            accuracy=_to_float(meta.get("accuracy")),
            f1=_to_float(meta.get("f1")),
            roc_auc=_to_float(meta.get("roc_auc")),
        )

    def list_runs(self, limit: int = 200) -> List[TrainingRunSummary]:
        if not os.path.isdir(self.base_dir):
            return []

        run_ids = sorted(os.listdir(self.base_dir), reverse=True)
        out: List[TrainingRunSummary] = []

        for rid in run_ids:
            run_dir = os.path.join(self.base_dir, rid)
            if not os.path.isdir(run_dir):
                continue
            meta_path = os.path.join(run_dir, "metadata.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

            out.append(
                TrainingRunSummary(
                    run_id=rid,
                    path=run_dir,
                    model_name=str(meta.get("model_name", "")),
                    created_at=str(meta.get("created_at", "")),
                    task_kind=str(meta.get("task_kind", "regression")),
                    r2=_to_float(meta.get("r2")),
                    rmse=_to_float(meta.get("rmse")),
                    mae=_to_float(meta.get("mae")),
                    accuracy=_to_float(meta.get("accuracy")),
                    f1=_to_float(meta.get("f1")),
                    roc_auc=_to_float(meta.get("roc_auc")),
                )
            )
            if len(out) >= int(limit):
                break

        return out

    def load_metadata(self, run_id: str) -> Dict[str, Any]:
        """读取单条 Run 的轻量元数据，不加载曲线、CSV 或模型文件。"""
        run_dir = os.path.join(self.base_dir, str(run_id))
        meta_path = os.path.join(run_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                value = json.load(f)
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def load_run(self, run_id: str, load_model: bool = False) -> Dict[str, Any]:
        run_dir = os.path.join(self.base_dir, run_id)
        meta_path = os.path.join(run_dir, "metadata.json")
        history_path = os.path.join(run_dir, "history.csv")
        curve_path = os.path.join(run_dir, "training_curve.png")
        model_path = os.path.join(run_dir, "model.pkl")

        meta = {}
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}

        hist_df = None
        if os.path.isfile(history_path):
            try:
                hist_df = pd.read_csv(history_path)
            except Exception:
                hist_df = None

        curve_bytes = None
        if os.path.isfile(curve_path):
            try:
                with open(curve_path, "rb") as f:
                    curve_bytes = f.read()
            except Exception:
                curve_bytes = None

        # 加载模型（可选）
        model_artifact = None
        if load_model and os.path.isfile(model_path):
            try:
                from .model_io import load_model_artifact_bytes
                with open(model_path, "rb") as f:
                    model_bytes = f.read()
                model_artifact = load_model_artifact_bytes(model_bytes)
            except Exception as e:
                print(f"Warning: Failed to load model: {e}")
                model_artifact = None

        # 额外 png
        extra_pngs = {}
        try:
            for fn in os.listdir(run_dir):
                if fn.endswith(".png") and fn not in ("training_curve.png",):
                    with open(os.path.join(run_dir, fn), "rb") as f:
                        extra_pngs[fn] = f.read()
        except Exception:
            extra_pngs = {}

        extra_tables = {}
        try:
            for fn in os.listdir(run_dir):
                if fn.endswith(".csv") and fn not in ("history.csv",):
                    try:
                        extra_tables[fn] = pd.read_csv(os.path.join(run_dir, fn))
                    except Exception:
                        continue
        except Exception:
            extra_tables = {}

        return {
            "run_id": run_id,
            "path": run_dir,
            "metadata": meta,
            "history": hist_df,
            "training_curve_png": curve_bytes,
            "extra_pngs": extra_pngs,
            "extra_tables": extra_tables,
            "model_artifact": model_artifact,
        }


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None
