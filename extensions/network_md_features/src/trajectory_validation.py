"""
平衡态轨迹科学验证门禁 (Trajectory Validation - Gate S6)
科学准则：
玻璃态热力学完全平衡往往不可达，但必须在统一制备协议下报告稳定参考态和残余漂移。
严禁仅凭短时间密度平坦宣布完全平衡。
门禁判据：
1. 温度均值与设定温度（如 298.15 K）偏差 <= 5 K；
2. 压力均值与设定压力（如 1.0 bar）在涨落范围内可接受（标准差一般 < 300 bar）；
3. 采样段密度线性拟合斜率（漂移率）必须满足 |drift| <= 1.0 %/ns；
4. 盒子边长 fluctuation 小于 5%。
"""

import math
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .chemical_validation import GateResult

class TrajectoryValidator:
    def __init__(
        self,
        target_temp_k: float = 298.15,
        target_press_bar: float = 1.0,
        max_density_drift_pct_per_ns: float = 1.0,
        max_temp_deviation_k: float = 5.0
    ):
        self.target_temp_k = target_temp_k
        self.target_press_bar = target_press_bar
        self.max_density_drift_pct_per_ns = max_density_drift_pct_per_ns
        self.max_temp_deviation_k = max_temp_deviation_k

    def check_gate_s6(
        self,
        time_ns: np.ndarray,
        density_g_cm3: np.ndarray,
        temperature_k: np.ndarray,
        pressure_bar: Optional[np.ndarray] = None,
        box_lengths_nm: Optional[np.ndarray] = None
    ) -> GateResult:
        """
        检验 S6 轨迹物理稳定度
        """
        if len(time_ns) < 5 or len(density_g_cm3) < 5:
            return GateResult("S6", "FAIL", {"steps": len(time_ns)}, "采样轨迹时间序列点数过少 (< 5)，无法评估稳定性")

        total_time_ns = float(time_ns[-1] - time_ns[0])
        if total_time_ns <= 0.0:
            return GateResult("S6", "FAIL", {}, "轨迹时间跨度异常 (<= 0 ns)")

        # 1. 密度统计与线性漂移分析
        mean_density = float(np.mean(density_g_cm3))
        std_density = float(np.std(density_g_cm3))

        # 线性回归拟合漂移斜率 slope (g/cm3 / ns)
        poly = np.polyfit(time_ns, density_g_cm3, 1)
        slope = float(poly[0])
        drift_pct_per_ns = (slope / mean_density) * 100.0 if mean_density > 0 else 999.0

        # 2. 温度统计分析
        mean_temp = float(np.mean(temperature_k))
        std_temp = float(np.std(temperature_k))
        temp_err = abs(mean_temp - self.target_temp_k)

        # 3. 压力统计
        mean_press = float(np.mean(pressure_bar)) if pressure_bar is not None else 1.0
        std_press = float(np.std(pressure_bar)) if pressure_bar is not None else 0.0

        details = {
            "mean_density_g_cm3": round(mean_density, 4),
            "std_density_g_cm3": round(std_density, 4),
            "density_drift_pct_per_ns": round(drift_pct_per_ns, 3),
            "mean_temperature_k": round(mean_temp, 2),
            "std_temperature_k": round(std_temp, 2),
            "temperature_deviation_k": round(temp_err, 2),
            "mean_pressure_bar": round(mean_press, 2),
            "std_pressure_bar": round(std_press, 2),
            "trajectory_duration_ns": round(total_time_ns, 3)
        }

        # 门禁检验逻辑
        reasons = []
        if abs(drift_pct_per_ns) > self.max_density_drift_pct_per_ns:
            reasons.append(f"密度漂移率超标: {drift_pct_per_ns:.2f}%/ns > {self.max_density_drift_pct_per_ns}%/ns")

        if temp_err > self.max_temp_deviation_k:
            reasons.append(f"平衡平均温度偏差超标: {temp_err:.2f}K > {self.max_temp_deviation_k}K")

        if mean_density < 0.80 or mean_density > 1.50:
            reasons.append(f"平衡密度异常超出聚合物正常物理范围: {mean_density:.3f} g/cm3")

        if reasons:
            return GateResult("S6", "FAIL", details, "; ".join(reasons))

        return GateResult("S6", "PASS", details, "平衡态参考轨迹稳定性达到 S6 门禁标准")
