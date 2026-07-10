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
