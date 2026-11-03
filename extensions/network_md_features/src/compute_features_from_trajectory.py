"""
compute_features_from_trajectory.py
独立免重跑特征复算 CLI 工具（与 postprocess_trajectory 完全共享同一真实后处理实现）
功能：
1. 直接委托调用 postprocess_trajectory.run_postprocessing，消除一切硬编码能量公式 (-15.2*(st+1))、硬编码氢键与球体简单求和；
2. 默认严格要求存在真实的 nvt.xtc、nvt.edr、nvt.tpr 与 system.prmtop，缺失任何轨迹或能量文件立即抛出 FileNotFoundError 阻断；
3. 输入目录与输出目录分离，绝不反写或污染输入 attempt 目录下的原生文件。
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd

try:
    from .postprocess_trajectory import run_postprocessing
except (ImportError, ValueError):
    _pkg_root = Path(__file__).resolve().parent.parent
    if str(_pkg_root) not in sys.path:
        sys.path.insert(0, str(_pkg_root))
    from src.postprocess_trajectory import run_postprocessing


def recompute_features_for_attempt(
    attempt_dir: str,
    output_dir: Optional[str] = None,
    t_start_ps: Optional[float] = None,
    t_end_ps: Optional[float] = None,
    stride: int = 1,
    grid_spacing_nm: float = 0.10,
    snapshot_only: bool = False,
    expected_hashes: Optional[Dict[str, str]] = None,
    required_energy_terms: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    从指定 attempt_dir 下的原生轨迹 (nvt.xtc) 和能量文件 (nvt.edr) 复算特征。
    若未显式提供 output_dir，则默认写入同级独立目录 `<attempt_dir>_recomputed_postprocess`，
    不修改或覆盖输入 attempt_dir。
    """
    in_path = Path(attempt_dir).resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Attempt directory not found: {attempt_dir}")

    if output_dir is None:
        out_path = in_path.parent / f"{in_path.name}_recomputed_postprocess"
    else:
        out_path = Path(output_dir).resolve()

    res = run_postprocessing(
        input_attempt_dir=str(in_path),
        output_dir=str(out_path),
        t_start_ps=t_start_ps,
        t_end_ps=t_end_ps,
        stride=stride,
        grid_spacing_nm=grid_spacing_nm,
        snapshot_only=snapshot_only,
        expected_hashes=expected_hashes,
        required_energy_terms=required_energy_terms,
    )
    return res["df_summary"]


def main():
    parser = argparse.ArgumentParser(
        description="Recompute MD diagnostic features from existing native XTC/EDR trajectories without running MD"
    )
    parser.add_argument(
        "--input-attempt",
        "--attempt_dir",
        dest="input_attempt",
        type=str,
        default="/home/wangj/project/CFRP/extensions/network_md_features/artifacts/attempt_20261019_01",
        help="Path to input attempt directory containing raw MD files",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default="/home/wangj/project/CFRP/extensions/network_md_features/artifacts/postprocess_verified_20261019_02",
        help="Path to separate output directory for recomputed features",
    )
    parser.add_argument("--t-start", dest="t_start_ps", type=float, default=None)
    parser.add_argument("--t-end", dest="t_end_ps", type=float, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--grid-spacing", dest="grid_spacing_nm", type=float, default=0.10)
    parser.add_argument("--snapshot-only", action="store_true")
    args = parser.parse_args()

    recompute_features_for_attempt(
        attempt_dir=args.input_attempt,
        output_dir=args.output_dir,
        t_start_ps=args.t_start_ps,
        t_end_ps=args.t_end_ps,
        stride=args.stride,
        grid_spacing_nm=args.grid_spacing_nm,
        snapshot_only=args.snapshot_only,
    )


if __name__ == "__main__":
    main()
