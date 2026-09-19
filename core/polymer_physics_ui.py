# -*- coding: utf-8 -*-
"""
高分子物理指数 UI
=================
在“分子特征”页提供独立的物理指数提取工具：
  - 单分子计算器：输入 SMILES → 立即得到全部 11 项物理指数
  - 批量提取：对当前数据 / 上传 CSV 提取 phys_* 列并导出
所有指数均由结构直接计算（零标签、零训练）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.polymer_physics import (
    ALL_PHYSICS_FEATURES,
    DEFAULT_PHYSICS_FEATURES,
    PHYSICS_FEATURE_LABELS,
    PHYSICS_PREFIX,
    compute_polymer_physics_indices,
    detect_structure_columns,
    molecule_physics_indices,
)

# 实测准入结果（Spearman）——用于在 UI 上标注哪些指数真的有用
_ACCURACY_NOTE = {
    "csp3": "→Tg −0.507 ✅",
    "f_ar": "→Tg +0.478 ✅",
    "bde_mean": "→Tg +0.453 / →Td5 +0.403 ✅",
    "flex": "→Tg −0.413 ✅",
    "delta": "区分力弱(变异6.6%) ⚠️",
    "bde_min": "→Tg −0.332 ⚠️",
    "bde_weak_frac": "弱 ⚠️",
    "ced": "区分力弱 ⚠️",
    "n_arom_ring": "→Tg +0.421 ✅",
    "rho_vdw": "堆砌代理，弱 ⚠️",
    "mw": "辅助信息",
}


def _fmt_val(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.4g}"


def _render_full_index_table(idx: dict) -> pd.DataFrame:
    rows = []
    for k in ALL_PHYSICS_FEATURES:
        rows.append({
            "指数": f"{PHYSICS_PREFIX}{k}",
            "含义": PHYSICS_FEATURE_LABELS.get(k, k),
            "数值": _fmt_val(idx.get(k)),
            "实测相关性": _ACCURACY_NOTE.get(k, ""),
        })
    return pd.DataFrame(rows)


def render_polymer_physics_ui(key_prefix: str = "pp"):
    """渲染高分子物理指数提取面板（@st.fragment 包裹在调用处）。"""
    st.markdown("#### 🧪 高分子物理指数（零标签 · 结构直算）")
    st.caption(
        "从结构 SMILES 直接计算物理指数，**无需任何训练或实测标签**。"
        "其中 csp3 / 芳香度 / 平均键解离能 / 柔性 已实测与 Tg 强相关（|Spearman| ≥ 0.41），"
        "可单独导出，也可由 PINN 自动注入。"
    )

    tab1, tab2 = st.tabs(["🔬 单分子计算器", "📊 批量提取"])

    # ------------------------------------------------ 单分子
    with tab1:
        c1, c2 = st.columns([3, 1])
        smi = c1.text_input(
            "SMILES / BigSMILES",
            value="CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1",
            key=f"{key_prefix}_smi",
            help="例如双酚A二缩水甘油醚 (DGEBA)。BigSMILES 的连接标记会自动清理。",
        )
        if c2.button("计算", key=f"{key_prefix}_calc", use_container_width=True):
            st.session_state[f"{key_prefix}_result"] = molecule_physics_indices(smi)

        res = st.session_state.get(f"{key_prefix}_result")
        if res is None and smi:
            res = molecule_physics_indices(smi)
            st.session_state[f"{key_prefix}_result"] = res

        if res:
            st.success("✅ 解析成功")
            dfr = _render_full_index_table(res)
            st.dataframe(dfr, use_container_width=True, hide_index=True,
                         height=min(460, 40 + 33 * len(dfr)))
            csv = dfr.to_csv(index=False).encode("utf-8-sig")
            st.download_button("💾 下载该分子指数 (CSV)", data=csv,
                               file_name="polymer_physics_single.csv", mime="text/csv",
                               key=f"{key_prefix}_dl1")
        elif smi:
            st.error("❌ SMILES 无法解析（RDKit 失败）。请检查语法，或改用规范化后的 SMILES。")

    # ------------------------------------------------ 批量
    with tab2:
        src_opts = ["使用当前数据表"]
        cur = st.session_state.get("data")
        if isinstance(cur, pd.DataFrame) and len(cur) > 0:
            st.caption(f"当前数据表：{len(cur)} 行 × {cur.shape[1]} 列")
        else:
            st.info("当前没有已加载的数据，可上传 CSV。")
        src_opts.append("上传 CSV")

        src = st.radio("数据来源", src_opts, horizontal=True, key=f"{key_prefix}_src")

        df_in = None
        if src == "使用当前数据表":
            if isinstance(cur, pd.DataFrame) and len(cur) > 0:
                df_in = cur
            else:
                st.warning("当前无数据，请先上传或切换到“上传 CSV”。")
        else:
            up = st.file_uploader("上传含结构列的 CSV", type=["csv"], key=f"{key_prefix}_up")
            if up is not None:
                try:
                    df_in = pd.read_csv(up, encoding="utf-8", encoding_errors="replace",
                                        low_memory=False)
                except Exception as e_up:
                    st.error(f"读取失败: {e_up}")

        if df_in is not None:
            detected = detect_structure_columns(df_in)
            auto_cols = [c for cols in detected.values() for c in cols]
            # 全对象列（供手工选择）
            obj_cols = [c for c in df_in.columns if not pd.api.types.is_numeric_dtype(df_in[c])]
            default_cols = auto_cols or [c for c in obj_cols
                                        if any(k in str(c).lower()
                                               for k in ("structure", "smiles"))]
            sel_cols = st.multiselect(
                "结构列（用于提取）", options=obj_cols or auto_cols,
                default=[c for c in default_cols if c in (obj_cols or auto_cols)],
                key=f"{key_prefix}_cols",
                help="自动探测到: " + (", ".join(auto_cols) if auto_cols else "无"),
            )
            sel_feats = st.multiselect(
                "要提取的指数",
                options=list(ALL_PHYSICS_FEATURES),
                default=[f for f in DEFAULT_PHYSICS_FEATURES],
                format_func=lambda k: f"{PHYSICS_FEATURE_LABELS.get(k, k)} ({PHYSICS_PREFIX}{k})",
                key=f"{key_prefix}_feats",
            )
            st.caption(
                "加权规则：优先用逐组分的 `*_amount_phr`（如 resin_1_amount_phr），"
                "其次角色总 phr，最后等权平均 —— 得到每个配方的加权指数。"
            )

            if st.button("🚀 提取物理指数", type="primary", use_container_width=True,
                         key=f"{key_prefix}_run"):
                if not sel_cols or not sel_feats:
                    st.warning("请至少选择一个结构列和一个指数。")
                else:
                    with st.spinner("提取中 ..."):
                        idx = compute_polymer_physics_indices(
                            df_in, features=sel_feats, structure_columns=sel_cols)
                    if idx is None or len(idx.columns) == 0:
                        st.error("未能提取到任何指数（结构列可能全部无法解析）。")
                    else:
                        cov = idx.notna().mean()
                        bad = int((~idx.notna()).all(axis=1).sum())
                        m1, m2, m3 = st.columns(3)
                        m1.metric("提取指数数", len(idx.columns))
                        m2.metric("平均覆盖率", f"{cov.mean()*100:.0f}%")
                        m3.metric("全部失败行数", bad)
                        st.dataframe(idx.head(200), use_container_width=True, height=260)
                        out = pd.concat([df_in.reset_index(drop=True),
                                         idx.reset_index(drop=True)], axis=1)
                        csv = out.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "💾 下载含物理指数的完整表 (CSV)", data=csv,
                            file_name="data_with_polymer_physics.csv", mime="text/csv",
                            key=f"{key_prefix}_dl2")
                        st.caption("💡 下载的 `phys_*` 列可直接作为任意模型（含 XGBoost）的输入特征。")
