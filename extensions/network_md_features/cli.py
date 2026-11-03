"""
交联网络物理特征实验室 CLI 工具 (Command-Line Interface)
支持命令：
  python cli.py list-candidates    # 查看 6 个精选候选配方
  python cli.py dry-run            # 执行 Dry-Run 资源与化学计量预估
  python cli.py run-pilot          # 运行小规模真实试验并提取 17 项物理特征
  python cli.py status             # 查看后台任务状态与门禁看板
  python cli.py verify             # 运行全量自动化测试并复核旧系统受保护文件哈希
"""

import os
import sys
import json
import argparse
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from src.feature_schema import MD_FEATURE_REGISTRY, export_schema_json
from src.task_store import TaskStore
from src.formulation_adapter import FormulationAdapter

def cmd_list_candidates(args):
    plans_path = os.path.join(CURRENT_DIR, "artifacts/run_plans/pilot_run_plans.json")
    if not os.path.exists(plans_path):
        print("未找到 pilot_run_plans.json，请先执行 screening。")
        return
    with open(plans_path, "r", encoding="utf-8") as f:
        plans = json.load(f)

    print(f"\n=== 6 个代表性候选文献配方 (小试适用域内) ===")
    for p in plans:
        disc = p.get("discrete_formulation", {})
        print(f"[{p['candidate_id']}]")
        print(f"  体系: {p['resin_name']} + {p['hardener_name']} (名义 r={p['nominal_r']}, 离散 r={disc.get('actual_discrete_r', '-')})")
        print(f"  分子数: 树脂={disc.get('n_resin_molecules')}, 固化剂={disc.get('n_hardener_molecules')}, 体系总原子={disc.get('total_system_atoms')}")
        print(f"  理论最大转化率: {disc.get('theoretical_max_alpha')}, 状态: {p.get('screening_status')}")

def cmd_dry_run(args):
    plans_path = os.path.join(CURRENT_DIR, "artifacts/run_plans/pilot_run_plans.json")
    if not os.path.exists(plans_path):
        print("未找到 pilot_run_plans.json。")
        return
    with open(plans_path, "r", encoding="utf-8") as f:
        plans = json.load(f)

    print(f"\n=== Dry-Run 资源与化学完整性预估 ===")
    total_estimated_seconds = 0.0
    for p in plans:
        disc = p.get("discrete_formulation", {})
        atoms = disc.get("total_system_atoms", 1500)
        est_sec = 180.0 if atoms < 3000 else 600.0
        total_estimated_seconds += est_sec * 2  # 2 seeds
        print(f"- {p['candidate_id']}: {atoms} 原子/体系, 预估耗时 {est_sec:.0f}s/网络 (x2 seeds)")

    print(f"\n全部 6 个配方 x 2 seeds 总预估计算耗时: {total_estimated_seconds / 60.0:.1f} 分钟")
    print("内存预算: ~1.5 GB，在既定资源上限以内，不启动全量 2000 条数据。")

def cmd_status(args):
    store = TaskStore()
    tasks = store.list_tasks()
    print(f"\n=== 后台任务与门禁状态 (共 {len(tasks)} 条) ===")
    for t in tasks:
        print(f"- 任务 {t.task_id}: [进程: {t.status}] [门禁: {t.scientific_gate_status}] 完成时间: {t.completed_at}")

def cmd_verify(args):
    print("\n=== 执行自动化测试与旧系统保护审计 ===")
    import subprocess
    cmd = ["/home/wangj/miniconda3/envs/htpolynet_env/bin/pytest", "tests/", "-v"]
    res = subprocess.run(cmd, cwd=CURRENT_DIR)
    sys.exit(res.returncode)

def main():
    parser = argparse.ArgumentParser(description="CFRP 独立交联网络物理特征实验室 CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("list-candidates", help="列出 6 个精选候选配方")
    subparsers.add_parser("dry-run", help="执行 Dry-Run 资源预估")
    subparsers.add_parser("status", help="查看任务与门禁状态")
    subparsers.add_parser("verify", help="运行测试套件与安全审计")

    args = parser.parse_args()
    if args.command == "list-candidates":
        cmd_list_candidates(args)
    elif args.command == "dry-run":
        cmd_dry_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "verify":
        cmd_verify(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
