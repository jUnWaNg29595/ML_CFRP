# -*- coding: utf-8 -*-
"""分子特征页交互 rerun 基准：点击选择框/复选框后的开销。

度量两个指标（都在脚本线程内部采集，避免测试框架开销污染）：
  1. script_s  —— 页面脚本执行耗时
  2. payload_mb —— 本次 rerun 发给浏览器的 delta 负载（真正决定浏览器卡不卡）

关键陷阱（本项目踩过）：
  - 用 cProfile 包 AppTest.run() 只会抓到测试框架的 sleep 轮询。
  - 探针若在每次 rerun 重建宽表（如逐列赋值 2000 列），构造耗时会被算进测量，
    得出完全错误的结论。宽表必须只在首轮构造一次。

用法：
    python tools/bench_molecular_features_interaction.py
    python tools/bench_molecular_features_interaction.py --rows 3000 --feats 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

PAGE_SCRIPT = r"""
import sys, time
sys.path.insert(0, {root!r})
import pandas as pd
import numpy as np
import streamlit as st

import streamlit.runtime.scriptrunner_utils.script_run_context as SRC
_msgs = []
_orig_enqueue = SRC.ScriptRunContext.enqueue


def _patched_enqueue(self, msg, *a, **kw):
    try:
        _msgs.append(msg.ByteSize())
    except Exception:
        pass
    return _orig_enqueue(self, msg, *a, **kw)


SRC.ScriptRunContext.enqueue = _patched_enqueue

# 宽表只构造一次：每次 rerun 重建会把构造耗时算进测量
if '_setup_done' not in st.session_state:
    n = int(st.session_state.get('_rows', 3000))
    nfeat = int(st.session_state.get('_feats', 2000))
    rng = np.random.default_rng(7)
    df = pd.DataFrame({{
        'sample_id': [f'S{{i:05d}}' for i in range(n)],
        'resin_smiles': ['CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'] * n,
        'hardener_smiles': ['Nc1ccc(Cc2ccc(N)cc2)cc1'] * n,
        'tg': rng.normal(150, 20, n).round(2),
    }})
    if nfeat:
        block = pd.DataFrame(
            rng.normal(0, 1, (n, nfeat)).round(4),
            columns=[f'resin_smiles_fp_{{i}}' for i in range(nfeat)],
        )
        df = pd.concat([df, block], axis=1)
    feat = df[[c for c in df.columns if c.startswith('resin_smiles_fp_')]]
    st.session_state.data = df
    st.session_state.processed_data = df
    st.session_state.molecular_features = feat
    st.session_state.molecular_feature_names = list(feat.columns)
    st.session_state.latest_extracted_features_df = feat
    st.session_state.latest_extraction_summary = {{
        'n_samples': n, 'n_features': nfeat, 'dual': False, 'method': 'single',
    }}
    st.session_state['_setup_done'] = True

from app_lib import page_molecular_features, init_session_state
if '_inited' not in st.session_state:
    init_session_state()
    st.session_state['_inited'] = True

_msgs.clear()
_t0 = time.perf_counter()
page_molecular_features()
st.session_state['_last'] = {{
    'script_s': round(time.perf_counter() - _t0, 4),
    'payload_mb': round(sum(_msgs) / 1024 / 1024, 4),
    'n_msgs': len(_msgs),
}}
"""


def build_app(rows: int, feats: int, timeout: float = 1800.0) -> AppTest:
    at = AppTest.from_string(PAGE_SCRIPT.format(root=str(ROOT)), default_timeout=timeout)
    at.session_state["_rows"] = rows
    at.session_state["_feats"] = feats
    return at


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--feats", type=int, default=2000)
    parser.add_argument("--all-sizes", action="store_true")
    args = parser.parse_args()

    sizes = [(3000, 0), (3000, 500), (3000, 2000), (6474, 3000)] if args.all_sizes \
        else [(args.rows, args.feats)]

    for rows, feats in sizes:
        at = build_app(rows, feats)
        at.run()
        assert not at.exception, [e.value for e in at.exception]

        # 交互都取带 key 的控件（AppTest 无法操作 key=None 的控件）。
        # 切换“当前处理种类” radio 触发整页 rerun —— 与用户点击选择框等价。
        # 注意：Radio.value 返回的是原始状态值（如 'resin'），而 options 是
        # format_func 渲染后的标签（如 '树脂'）——set_value 必须用原始值。
        role_opts = ["resin", "hardener", "neutral"]
        results = []
        for i in range(3):
            at.radio(key="molecular_feature_component_role").set_value(
                role_opts[i % len(role_opts)]
            ).run()
            assert not at.exception, [e.value for e in at.exception]
            results.append(at.session_state["_last"])

        script = sorted(r["script_s"] for r in results)
        payload = sorted(r["payload_mb"] for r in results)
        print(
            f"rows={rows:5d} feats={feats:5d} | "
            f"script median={script[1]:.3f}s | "
            f"payload median={payload[1]:.3f}MB | msgs={results[0]['n_msgs']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
