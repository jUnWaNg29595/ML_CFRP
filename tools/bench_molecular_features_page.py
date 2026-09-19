# -*- coding: utf-8 -*-
"""分子特征工程页（app_lib.page_molecular_features）性能基准。

在脚本线程内部计时与采样，避免 AppTest 轮询噪声。

用法：
    python tools/bench_molecular_features_page.py                      # 基准 + 分阶段计时
    python tools/bench_molecular_features_page.py --rows 200 --profile # 附加 cProfile 热点
    python tools/bench_molecular_features_page.py --skip-expanders      # 对照：跳过两个重面板
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

OUT_DIR = ROOT / "tools" / "_bench_out"
OUT_DIR.mkdir(exist_ok=True)

PAGE_SCRIPT = r"""
import sys, json, time
sys.path.insert(0, {root!r})
import pandas as pd
import numpy as np
import streamlit as st

import core.formulation_fusion_ui as _ffu
import core.polymer_physics_ui as _ppu

_timings = {{}}
_skip = bool(st.session_state.get('_bench_skip_expanders', False))

_orig_fusion = _ffu.render_formulation_fusion_ui
_orig_physics = _ppu.render_polymer_physics_ui


def _timed_fusion(*a, **kw):
    t0 = time.perf_counter()
    try:
        return _orig_fusion(*a, **kw)
    finally:
        _timings['fusion_expander'] = time.perf_counter() - t0


def _timed_physics(*a, **kw):
    t0 = time.perf_counter()
    try:
        return _orig_physics(*a, **kw)
    finally:
        _timings['physics_expander'] = time.perf_counter() - t0


_ffu.render_formulation_fusion_ui = _timed_fusion
_ppu.render_polymer_physics_ui = _timed_physics

_io_log = []
_orig_read_csv = pd.read_csv


def _traced_read_csv(*a, **kw):
    _t0 = time.perf_counter()
    try:
        return _orig_read_csv(*a, **kw)
    finally:
        _io_log.append(['read_csv', str(a[0] if a else kw.get('filepath_or_buffer'))[:120],
                        round(time.perf_counter() - _t0, 4)])


pd.read_csv = _traced_read_csv
_orig_to_csv = pd.DataFrame.to_csv


def _traced_to_csv(self, *a, **kw):
    _t0 = time.perf_counter()
    try:
        return _orig_to_csv(self, *a, **kw)
    finally:
        _io_log.append(['to_csv', f'{{self.shape[0]}}x{{self.shape[1]}}',
                        round(time.perf_counter() - _t0, 4)])


pd.DataFrame.to_csv = _traced_to_csv

if _skip:
    # 对照实验：面板照常渲染，但内部立即返回（保留 import 与函数调用开销）
    def _noop_fusion(*a, **kw):
        t0 = time.perf_counter()
        _timings['fusion_expander'] = time.perf_counter() - t0
    def _noop_physics(*a, **kw):
        t0 = time.perf_counter()
        _timings['physics_expander'] = time.perf_counter() - t0
    _ffu.render_formulation_fusion_ui = _noop_fusion
    _ppu.render_polymer_physics_ui = _noop_physics

from app_lib import page_molecular_features, init_session_state
init_session_state()

n = int(st.session_state.get('_bench_rows', 40))
rng = np.random.default_rng(7)
st.session_state.data = pd.DataFrame({{
    'sample_id': [f'S{{i:04d}}' for i in range(n)],
    'resin_smiles': ['CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1'] * n,
    'hardener_smiles': ['Nc1ccc(Cc2ccc(N)cc2)cc1'] * n,
    'tg': rng.normal(150, 20, n).round(2),
}})
st.session_state.processed_data = st.session_state.data

_prof = None
if st.session_state.get('_bench_profile', False):
    import cProfile
    _prof = cProfile.Profile()
    _prof.enable()

_t_page = time.perf_counter()
page_molecular_features()
_timings['page_total'] = time.perf_counter() - _t_page

if _prof is not None:
    _prof.disable()
    import pstats
    with open(st.session_state['_bench_profile_path'], 'w', encoding='utf-8') as fh:
        pstats.Stats(_prof, stream=fh).sort_stats('cumulative').print_stats(60)

st.session_state['_bench_timings'] = _timings
st.session_state['_bench_io_log'] = _io_log
"""


def build_app(rows: int, skip_expanders: bool, profile: bool, timeout: float = 900.0) -> AppTest:
    at = AppTest.from_string(PAGE_SCRIPT.format(root=str(ROOT)), default_timeout=timeout)
    at.session_state["_bench_rows"] = rows
    at.session_state["_bench_skip_expanders"] = skip_expanders
    at.session_state["_bench_profile"] = profile
    at.session_state["_bench_profile_path"] = str(OUT_DIR / "profile.txt")
    return at


def run_once(rows: int, skip_expanders: bool, profile: bool) -> dict:
    at = build_app(rows, skip_expanders, profile)
    at.run()
    errors = [e.value for e in at.exception]
    if errors:
        raise AssertionError(f"页面渲染抛出异常: {errors}")
    return dict(at.session_state["_bench_timings"])


def summarize(label: str, runs: list[dict]) -> None:
    keys = ["fusion_expander", "physics_expander", "page_total"]
    print(f"\n=== {label} ===")
    for key in keys:
        vals = sorted(r.get(key, 0.0) for r in runs)
        if not vals:
            continue
        print(f"  {key:<20} min={vals[0]:6.3f}s  median={vals[len(vals) // 2]:6.3f}s  max={vals[-1]:6.3f}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    print(f"[bench] rows={args.rows} repeat={args.repeat}")

    # 预热（含全部重依赖 import）
    t0 = time.perf_counter()
    warm = run_once(args.rows, False, False)
    print(f"[bench] warmup 首轮(含 import): {time.perf_counter() - t0:.2f}s  page_total={warm.get('page_total', 0):.3f}s")

    runs = [run_once(args.rows, False, args.profile) for _ in range(args.repeat)]
    summarize("完整页面", runs)

    print("\n[bench] 对照：跳过两个重面板内部渲染")
    ctrl = [run_once(args.rows, True, False) for _ in range(args.repeat)]
    summarize("跳过重面板", ctrl)

    if args.profile:
        path = OUT_DIR / "profile.txt"
        print(f"\n[bench] cProfile 输出: {path}")

    (OUT_DIR / "timings.json").write_text(
        json.dumps({"full": runs, "skip_expanders": ctrl}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
