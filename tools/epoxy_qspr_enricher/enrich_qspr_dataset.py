# -*- coding: utf-8 -*-
"""
tools/epoxy_qspr_enricher/enrich_qspr_dataset.py

专用于 ml_qspr_model_*.csv 原始数据集的高分子交联机理与动力学特征增强工具。

核心功能：
1. 自动对齐读取多组分树脂与固化剂结构（支持 BigSMILES 与标准 SMILES）；
2. 自动融合真实化学计量比（formulation_r_value）、计算偏离度 |r-1.0| 与对数计量比；
3. 计算 Flory-Stockmayer 凝胶网络物理参数：理论交联点间分子量 Mc 与理论交联密度代理 Crosslink_Density_Proxy；
4. 计算多组分成对前线轨道能差动力学矩阵（Delta E: min, max, span, weighted）；
5. 自动加权合成配方物理混合物特征（极性 TPSA、分子量、芳环数等）；
6. 导出可直接用于 QSPR 机器学习训练的增强版 CSV 表格；
7. 忠实保留全部样本（不对负数 Tg 等指标做强制删除，保留给用户自行处置）。

使用方法：
    python enrich_qspr_dataset.py \
        --input C:\\Users\\wangj\\Desktop\\ml_dataset\\ml_qspr_model_tg_c.csv \
        --output C:\\Users\\wangj\\Desktop\\ml_dataset\\ml_qspr_model_tg_c_enhanced.csv
"""

import argparse
import os
import sys
import time

# 兼容 Windows GBK 控制台打印
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 将项目根目录添加到 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from core.epoxy_mechanism_features import EpoxyMechanismEngine


def enrich_qspr_file(
    input_path: str,
    output_path: str,
    wide_path: str = None,
    verbose: bool = True
) -> str:
    """对 QSPR 原始数据集进行机理特征增强并导出文件"""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"未找到输入数据文件: {input_path}")

    if verbose:
        print("=" * 70)
        print("🚀 正在启动环氧高分子机理与动力学特征增强处理...")
        print(f"📂 输入文件: {input_path}")

    # 读取主数据表
    t0 = time.perf_counter()
    df_main = pd.read_csv(input_path, low_memory=False)
    if verbose:
        print(f"📊 主数据形状: {df_main.shape[0]} 行, {df_main.shape[1]} 列")

    # 自动探测 wide 表
    df_wide = None
    if wide_path and os.path.exists(wide_path):
        target_wide = wide_path
    else:
        # 在主文件同级目录下自动搜寻 ml_wide_samples.csv
        candidate_wide = os.path.join(os.path.dirname(input_path), "ml_wide_samples.csv")
        target_wide = candidate_wide if os.path.exists(candidate_wide) else None

    if target_wide and os.path.exists(target_wide):
        if verbose:
            print(f"🔗 找到大宽表补充文件: {target_wide}")
        raw_wide = pd.read_csv(target_wide, low_memory=False)
        # 尝试通过 tg_c 或行数对齐
        if len(raw_wide) == len(df_main):
            df_wide = raw_wide
        elif "tg_c" in raw_wide.columns and raw_wide["tg_c"].notna().sum() == len(df_main):
            df_wide = raw_wide[raw_wide["tg_c"].notna()].reset_index(drop=True)
            if verbose:
                print("✓ 已按 tg_c 有效行自动完成主表与宽表的索引精准对齐！")
        else:
            if verbose:
                print("⚠️ 宽表行数与主表不一致，将采用主表自适应计算模式。")

    # 初始化计算引擎
    engine = EpoxyMechanismEngine(verbose=False)

    if verbose:
        print("⚙️ 正在计算高分子三维交联网络、理论Mc、交联密度与ΔE动力学矩阵...")

    enriched_df = engine.enrich_dataframe(
        df_main,
        wide_df=df_wide,
        progress_callback=lambda curr, total: print(f"  进度: {curr}/{total} ({curr/total*100:.1f}%)") if verbose else None
    )

    elapsed = time.perf_counter() - t0
    new_cols = [c for c in enriched_df.columns if c.startswith("mech_")]

    if verbose:
        print("-" * 70)
        print(f"✨ 特征计算完成！总耗时: {elapsed:.2f} 秒")
        print(f"📈 新增机理与动力学特征列数: {len(new_cols)} 列")
        print(f"📋 新增特征列表: {new_cols}")
        print(f"💾 正在保存增强数据集至: {output_path}")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    enriched_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    if verbose:
        print(f"✅ 文件已成功导出！最终数据形状: {enriched_df.shape[0]} 行, {enriched_df.shape[1]} 列")
        print("=" * 70)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="环氧 QSPR 数据集机理特征增强工具")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=r"C:\Users\wangj\Desktop\ml_dataset\ml_qspr_model_tg_c.csv",
        help="输入 QSPR 数据集路径 (默认: ml_qspr_model_tg_c.csv)"
    )
    parser.add_argument(
        "--wide",
        "-w",
        type=str,
        default=None,
        help="可选的大宽表路径 (ml_wide_samples.csv，默认自动寻找同目录)"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="输出增强版 CSV 文件路径 (默认在输入文件同目录下生成 *_enhanced.csv)"
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_enhanced{ext}"

    enrich_qspr_file(input_path, output_path, wide_path=args.wide, verbose=True)


if __name__ == "__main__":
    main()
