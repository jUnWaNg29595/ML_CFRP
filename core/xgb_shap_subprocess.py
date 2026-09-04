# -*- coding: utf-8 -*-
"""Isolated XGBoost SHAP worker to protect the Streamlit process."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import traceback

import matplotlib.pyplot as plt

from .model_interpreter import EnhancedModelInterpreter
from .plot_utils import fig_to_png_bytes


def main() -> int:
    parser = argparse.ArgumentParser(description="Run XGBoost SHAP in a subprocess")
    parser.add_argument("--job", required=True, help="Path to the pickled job payload")
    args = parser.parse_args()

    with open(args.job, "rb") as f:
        payload = pickle.load(f)

    result = {
        "ok": False,
        "png_path": payload.get("png_path"),
        "csv_path": payload.get("csv_path"),
    }

    try:
        interp = EnhancedModelInterpreter(
            payload["model"],
            payload["X_train"],
            payload["y_train"],
            payload["X_test"],
            payload["y_test"],
            payload["model_name"],
            feature_names=payload.get("feature_names"),
            max_samples=payload.get("max_samples"),
            kernel_background=payload.get("kernel_background"),
            kernel_nsamples=payload.get("kernel_nsamples"),
            scaler=payload.get("scaler"),
        )

        fig, df_shap = interp.plot_summary(
            plot_type=payload.get("plot_type", "beeswarm"),
            max_display=payload.get("max_display", 20),
        )
        if fig is None:
            raise RuntimeError("plot_summary returned no figure")

        png_bytes = fig_to_png_bytes(fig)
        with open(payload["png_path"], "wb") as f:
            f.write(png_bytes)

        if df_shap is not None:
            df_shap.to_csv(payload["csv_path"], index=False, encoding="utf-8-sig")

            # 同步导出 Origin 专属的 Beeswarm 蜂群散点长表和 Bar 柱状汇总表
            job_dir = os.path.dirname(payload["csv_path"])
            origin_beeswarm_df = getattr(df_shap, "_origin_beeswarm_df", None)
            if origin_beeswarm_df is not None:
                origin_beeswarm_path = os.path.join(job_dir, "origin_shap_beeswarm_data.csv")
                origin_beeswarm_df.to_csv(origin_beeswarm_path, index=False, encoding="utf-8-sig")
                result["origin_beeswarm_path"] = origin_beeswarm_path

            importance_summary_df = getattr(df_shap, "_importance_summary_df", None)
            if importance_summary_df is not None:
                origin_bar_path = os.path.join(job_dir, "origin_shap_importance_ranking.csv")
                importance_summary_df.to_csv(origin_bar_path, index=False, encoding="utf-8-sig")
                result["origin_bar_path"] = origin_bar_path
        else:
            result["csv_path"] = None

        result["ok"] = True
    except Exception as exc:
        result["error"] = str(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        try:
            plt.close("all")
        except Exception:
            pass
        gc.collect()

    os.makedirs(os.path.dirname(payload["result_path"]), exist_ok=True)
    with open(payload["result_path"], "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
