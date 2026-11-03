"""
交联网络物理特征实验室（独立试验）Web 应用
端口隔离：默认运行于 8503 独立端口，绝不抢占 8501
数据隔离：已隔离历史假数据，默认仅读取 recovery_v2 真实模拟结果
安全准则：明确声明当前允许进入 ML 的记录数为 0
"""

import os
import sys
import json
import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from src.feature_schema import MD_FEATURE_REGISTRY
from src.formulation_adapter import CANONICAL_MONOMER_LIBRARY

st.set_page_config(
    page_title="交联网络物理特征实验室（真实性修复与参考闭环）",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 侧边栏标题与隔离声明
st.sidebar.title("🧪 物理特征实验室")
st.sidebar.markdown("**版本**：v2.0 真实性修复与单配方闭环")
st.sidebar.info(
    "🔒 **系统安全与真实性承诺**：\n"
    "- 原系统 8501 服务无干扰独立运行\n"
    "- 旧机理特征与历史模型 100% 字节级受保护\n"
    "- 历史合成数据已全部移入隔离区\n"
    "- 当前允许进入正式 ML 的记录数：0"
)

nav_mode = st.sidebar.radio(
    "功能导航",
    [
        "1. 历史伪造声明撤回与隔离审计",
        "2. 真实单配方 MD 模拟与原生文件",
        "3. 真实物理特征与科学门禁看板",
        "4. 单体与 DFT-RESP 映射审计",
        "5. ML 接口与前置准则说明"
    ]
)

@st.cache_data
def load_claims_audit():
    p = os.path.join(CURRENT_DIR, "reports/historical_claims_audit.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_real_features():
    p = os.path.join(CURRENT_DIR, "artifacts/recovery_v2/md_features_reference.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data
def load_gate_report():
    p = os.path.join(CURRENT_DIR, "reports/scientific_gate_report.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@st.cache_data
def load_charge_audit():
    p = os.path.join(CURRENT_DIR, "reports/charge_mapping_audit.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    return pd.DataFrame()

# ----------------- 页面 1: 历史伪造声明撤回与隔离审计 -----------------
if nav_mode == "1. 历史伪造声明撤回与隔离审计":
    st.header("🛡️ 历史伪造数据撤回与隔离审计公告")
    st.error(
        "⚠️ **严正声明**：前一轮小试脚本 (run_pilot_pipeline.py) 存在严重的纯数学随机数伪造行为，"
        "未真实调用底层 GROMACS 求解力场方程。现已将所有虚假数据与衍生报告全量移入隔离区，"
        "历史小试数据**绝对禁止**混入机器学习训练集！"
    )

    audit_info = load_claims_audit()
    if audit_info:
        st.subheader("撤回的虚假声明清单")
        claims = audit_info.get("retracted_claims", [])
        for c in claims:
            with st.expander(f"❌ [{c['claim_id']}] {c['statement']}"):
                st.markdown(f"**审计结论**：`{c['audit_verdict']}`")
                st.markdown(f"**证据文件**：`{c['evidence_file']}`")
                st.markdown(f"**排查事实**：{c['finding']}")

        st.subheader("隔离资产清单 (artifacts/quarantine_synthetic_v1/)")
        q_files = audit_info.get("quarantined_files", {})
        q_rows = []
        for k, v in q_files.items():
            q_rows.append({
                "文件/目录": k,
                "数据属性": v.get("data_origin"),
                "科学状态": v.get("scientific_status"),
                "是否允许训练": v.get("eligible_for_training"),
                "拒绝与隔离原因": v.get("rejection_reason")
            })
        st.dataframe(pd.DataFrame(q_rows), use_container_width=True)

        st.success("✅ **旧系统污染排查结论**：原系统核心源码、数据库与模型 100% CLEAN，未受到任何伪造数据污染。")

# ----------------- 页面 2: 真实单配方 MD 模拟与原生文件 -----------------
elif nav_mode == "2. 真实单配方 MD 模拟与原生文件":
    st.header("⚙️ 真实单配方 HTPolyNet / GROMACS 模拟现场")
    st.markdown(
        "本轮在独立环境中使用 HTPolyNet 2.15 + GROMACS 2026.3 + AmberTools 26.0 真实执行了 DGEBA / 4,4'-DDS "
        "单配方小体系构建、能量最小化、NVT/NPT 预平衡与交联反应。"
    )

    summary_file = os.path.join(CURRENT_DIR, "artifacts/recovery_v2/REAL_DGEBA_DDS_seed20261015/proj-0/profile.json")
    if os.path.exists(summary_file):
        with open(summary_file, "r") as f:
            prof = json.load(f)
        st.subheader("真实计算耗时剖析 (Seed 20261015, 原生 profile.json)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("总执行耗时", f"{prof.get('total_wall_seconds', 0):.2f} s")
        subp = prof.get("aggregate_subprocess_seconds", {})
        c2.metric("gmx-mdrun 耗时", f"{subp.get('gmx-mdrun', 0):.2f} s")
        c3.metric("gmx-grompp 耗时", f"{subp.get('gmx-grompp', 0):.2f} s")
        c4.metric("AmberTools 参数化", f"{subp.get('antechamber', 0)+subp.get('parmchk2', 0)+subp.get('tleap', 0):.2f} s")

    st.subheader("真实原生引擎输出文件目录 (artifacts/recovery_v2/)")
    st.code(
        "REAL_DGEBA_DDS_seed20261015/proj-0/systems/final-results/\n"
        "├── final.gro          # 真实最终结构 (262 原子, 盒长 2.02824 nm)\n"
        "├── final.top          # 真实系统拓扑 (54 KB, GAFF2 参数化与电荷)\n"
        "├── final.grx          # 扩展拓扑属性\n"
        "├── final.viz.psf      # VMD 结构拓扑\n"
        "├── final.viz.macros.tcl\n"
        "└── final.viz.tcl"
    )

# ----------------- 页面 3: 真实物理特征与科学门禁看板 -----------------
elif nav_mode == "3. 真实物理特征与科学门禁看板":
    st.header("📊 真实单配方物理特征与科学门禁看板")
    st.markdown(
        "展示从真实 GROMACS `final.gro` 与原生拓扑中提取的物理特征。"
        "**严格遵守高分子物理准则：未严格解耦的弹性链段、未做 3D 渗流绕行连通性的孔隙及缺乏解聚参考态的 CED 均诚实置为 null (None)。**"
    )

    df_real = load_real_features()
    if not df_real.empty:
        st.subheader("真实参考特征表 (md_features_reference.csv)")
        st.dataframe(df_real, use_container_width=True)

        st.subheader("严密 S0 ~ S6 科学门禁结果")
        gates = load_gate_report()
        st.json(gates)
    else:
        st.info("尚无导出的特征结果。")

# ----------------- 页面 4: 单体与 DFT-RESP 映射审计 -----------------
elif nav_mode == "4. 单体与 DFT-RESP 映射审计":
    st.header("🔬 单体真实 DFT / RESP 映射审计")
    st.markdown(
        "直接只读核验 `/home/wangj/epoxy_prod/rebuild_verified_v3_5/continuation_20261015/` 下真实产物：\n"
        "- **4,4'-DDS**: 真实原子数严格为 **29** (C12H12N2O2S)，净电荷严格为 0.00000000 e；\n"
        "- **DGEBA**: 真实原子数严格为 **49** (C21H24O4)，净电荷严格为 -0.00000000 e。"
    )

    df_charge = load_charge_audit()
    if not df_charge.empty:
        st.subheader("逐原子二阶段 RESP 映射表 (charge_mapping_audit.csv, 共 78 原子)")
        st.dataframe(df_charge, use_container_width=True)

# ----------------- 页面 5: ML 接口与前置准则说明 -----------------
elif nav_mode == "5. ML 接口与前置准则说明":
    st.header("🛑 机器学习接口与扩大准则")
    st.warning(
        "⚠️ **当前允许进入正式 ML 训练集的记录数：0 条**\n\n"
        "**为什么当前记录数是 0？**\n"
        "1. 当前完成的是单配方微小体系（262原子）的构建与交联链路闭环，属于受限诊断探索协议（scientific_gate_status = EXPLORATORY / FAIL）；\n"
        "2. 严禁把单配方诊断数据直接当成材料物性生产真值，绝不污染已有 2000 条文献模型；\n"
        "3. 在扩大至正式生产训练前，必须满足：重新适配 CUDA 版 GROMACS、大体系纳秒级热松弛达到宏观致密态（~1.2 g/cm³）且完成严密的键解聚 CED 理论推导。"
    )
