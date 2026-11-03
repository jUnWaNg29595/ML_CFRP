# -*- coding: utf-8 -*-
"""
Direction B 生产级多尺度环氧树脂反应模拟与机器学习特征提取全流程总控脚本 (run_prod_pipeline.py)
参数空间 2D 网格扫描：
- 化学计量比 r in [0.8, 1.0, 1.2]
- 固化转化率 alpha in [0.70, 0.80, 0.90]
生产级体系规模：
- N_DGEBA = 300 (~18,000 - 20,000 原子的微观聚集态晶胞)
- HTPolyNet 两步胺反应动力学 (k2/k1 = 0.45)
- OpenMP 16 线程并行 LAMMPS 退火松弛与物性评估
- 多微团簇代表性采样与 32 线程 PySCF 局域弱键计算
一键贯通全部 7 大阶段。
"""

import sys
import os
import time
import argparse

# 导入各阶段模块
from stage1_quantum import run_stage1
from stage2_assembly import run_stage2
from stage3_crosslink import run_stage3
from stage4_graph_nu import run_stage4
from stage5_cluster_td import run_stage5
from stage6_properties import run_stage6
from stage7_dataset_export import run_stage7

def run_pipeline(ratios=[0.8, 1.0, 1.2], alphas=[0.70, 0.80, 0.90], n_dgeba=300):
    t_start = time.time()
    print("=" * 85)
    print("      Direction B 生产级环氧树脂高通量配方模拟与特征提取全流程流水线      ")
    print(f"  体系基准: DGEBA (环氧) + 4,4'-DDS (固化剂), N_DGEBA = {n_dgeba}")
    print(f"  自变量网格: 化学计量比 r = {ratios} x 固化转化率 alpha = {alphas} (共 {len(ratios)*len(alphas)} 组样本)")
    print("=" * 85)
    
    # 阶段 1：单体量子化学特征提取 (单体本征性质不变，缓存复用)
    t0 = time.time()
    print("\n>>> [步骤 1/7] 执行阶段 1：单体级本征量子化学特征提取 (PySCF / RDKit)...")
    run_stage1(output_dir='/home/wangj/epoxy_prod/data')
    print(f"    -> 阶段 1 完成, 耗时 {time.time() - t0:.2f} 秒.")
    
    # 阶段 2 ~ 6：外层循环遍历不同配比 r
    for idx, r in enumerate(ratios, 1):
        phr = (62.08 / 170.21) * 100.0 * r
        print("\n" + "#" * 85)
        print(f"### [配比实验 {idx}/{len(ratios)}] 开始处理生产级配比 r = {r:.2f} (DDS phr: {phr:.2f}) ###")
        print("#" * 85)
        
        # 阶段 2：生产级晶胞组装与 LAMMPS 前平衡
        t0 = time.time()
        print(f"\n>>> [步骤 2/7 | r={r:.2f}] 组装生产级晶胞 (N_DGEBA={n_dgeba}) 并执行 OpenMP 并行平衡松弛...")
        run_stage2(r=r, n_dgeba=n_dgeba)
        print(f"    -> 阶段 2 (r={r:.2f}) 完成, 耗时 {time.time() - t0:.2f} 秒.")
        
        # 阶段 3：动力学动态交联与快照捕获
        t0 = time.time()
        print(f"\n>>> [步骤 3/7 | r={r:.2f}] LAMMPS 两步胺动力学动态交联与 alpha={alphas} 快照捕获...")
        run_stage3(r=r)
        print(f"    -> 阶段 3 (r={r:.2f}) 完成, 耗时 {time.time() - t0:.2f} 秒.")
        
        # 阶段 4：图论网络拓扑分析
        t0 = time.time()
        print(f"\n>>> [步骤 4/7 | r={r:.2f}] NetworkX 图论网络剪枝与交联密度分析...")
        run_stage4(r=r)
        print(f"    -> 阶段 4 (r={r:.2f}) 完成, 耗时 {time.time() - t0:.2f} 秒.")
        
        # 阶段 5：微团簇逆向切球与 Td 分析
        t0 = time.time()
        print(f"\n>>> [步骤 5/7 | r={r:.2f}] 多微团簇逆向切球与 PySCF 局域弱键/Td 计算...")
        run_stage5(r=r)
        print(f"    -> 阶段 5 (r={r:.2f}) 完成, 耗时 {time.time() - t0:.2f} 秒.")
        
        # 阶段 6：聚集态宏观物性评估
        t0 = time.time()
        print(f"\n>>> [步骤 6/7 | r={r:.2f}] LAMMPS 降温退火拟合 Tg、自由体积与力学物性...")
        run_stage6(r=r)
        print(f"    -> 阶段 6 (r={r:.2f}) 完成, 耗时 {time.time() - t0:.2f} 秒.")
    
    # 阶段 7：统一多尺度特征汇聚与导出
    t0 = time.time()
    print("\n>>> [步骤 7/7] 执行阶段 7：生产级多尺度特征融合与结构化 CSV 导出...")
    dataset_file = run_stage7(ratios=ratios)
    print(f"    -> 阶段 7 完成, 耗时 {time.time() - t0:.2f} 秒.")
    
    total_time = time.time() - t_start
    print("\n" + "=" * 85)
    print("【Direction B 生产级多尺度全流程仿真圆满完成】")
    print(f"  总耗时: {total_time:.2f} 秒 (~{total_time/60.0:.2f} 分钟)")
    print(f"  包含配比 r 梯度: {ratios}")
    print(f"  包含固化度 alpha 梯度: {alphas}")
    print(f"  结构化机器学习数据集路径: {dataset_file}")
    print("=" * 85)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Direction B 生产级全流程模拟")
    parser.add_argument("--r", type=float, default=None, help="指定单个配比 r (如 1.0)")
    parser.add_argument("--n_dgeba", type=int, default=300, help="DGEBA 分子数 (默认 300)")
    args = parser.parse_args()
    
    if args.r is not None:
        run_pipeline(ratios=[args.r], alphas=[0.70, 0.80, 0.90], n_dgeba=args.n_dgeba)
    else:
        run_pipeline(ratios=[0.8, 1.0, 1.2], alphas=[0.70, 0.80, 0.90], n_dgeba=args.n_dgeba)
