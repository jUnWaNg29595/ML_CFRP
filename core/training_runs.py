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
    r2: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None


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
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{ts}_{_safe_name(model_name)}"
        run_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        return run_id, run_dir

    def save_run(
        self,
        model_name: str,
        metadata: Dict[str, Any],
        history_df: Optional[pd.DataFrame] = None,
        curve_fig: Any = None,
        extra_figs: Optional[Dict[str, Any]] = None,
    ) -> TrainingRunSummary:
        run_id, run_dir = self.create_run_dir(model_name)

        meta = dict(metadata or {})
        meta.setdefault("model_name", model_name)
        meta.setdefault("run_id", run_id)
        meta.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))

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

        return TrainingRunSummary(
            run_id=run_id,
            path=run_dir,
            model_name=model_name,
            created_at=str(meta.get("created_at", "")),
            r2=_to_float(meta.get("r2")),
            rmse=_to_float(meta.get("rmse")),
            mae=_to_float(meta.get("mae")),
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
                    r2=_to_float(meta.get("r2")),
                    rmse=_to_float(meta.get("rmse")),
                    mae=_to_float(meta.get("mae")),
                )
            )
            if len(out) >= int(limit):
                break

        return out

    def load_run(self, run_id: str) -> Dict[str, Any]:
        run_dir = os.path.join(self.base_dir, run_id)
        meta_path = os.path.join(run_dir, "metadata.json")
        history_path = os.path.join(run_dir, "history.csv")
        curve_path = os.path.join(run_dir, "training_curve.png")

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

        # 额外 png
        extra_pngs = {}
        try:
            for fn in os.listdir(run_dir):
                if fn.endswith(".png") and fn not in ("training_curve.png",):
                    with open(os.path.join(run_dir, fn), "rb") as f:
                        extra_pngs[fn] = f.read()
        except Exception:
            extra_pngs = {}

        return {
            "run_id": run_id,
            "path": run_dir,
            "metadata": meta,
            "history": hist_df,
            "training_curve_png": curve_bytes,
            "extra_pngs": extra_pngs,
        }


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None
