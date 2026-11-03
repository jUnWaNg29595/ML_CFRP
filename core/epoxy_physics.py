# -*- coding: utf-8 -*-
"""
epoxy_physics.py

物理约束与工程化特征工具（用于 Epoxy PINN / Physics-Guided 模型）

目标：
- 解析带单位/符号的数值（如 "80 wt%"、"25°C"、"—"）
- r-value（化学计量比）相关的硬约束：alpha_max = min(1, r, 1/r)
- DiBenedetto Tg-α 关系（可微）
- Halpin–Tsai 复合材料弹性模量（可微，工程近似）

注意：本模块不依赖外部网络；仅依赖 numpy/torch（可选）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

# 常用“缺失/无效”标记
_NA_TOKENS = {
    "", "—", "–", "-", "na", "n/a", "nan", "none", "null", "missing",
    "未测试", "无", "N/A", "NA",
}


def parse_first_number(x: Union[str, float, int, None]) -> float:
    """从字符串中提取第一个浮点数。

    Examples
    --------
    "80 wt%" -> 80.0
    "25°C" -> 25.0
    "1.2e3 MPa" -> 1200.0
    "—" -> np.nan
    """
    if x is None:
        return float("nan")
    if isinstance(x, (int, float, np.number)):
        try:
            return float(x)
        except Exception:
            return float("nan")
    s = str(x).strip()
    if not s:
        return float("nan")
    if s.lower() in _NA_TOKENS:
        return float("nan")
    m = _NUMBER_RE.search(s)
    if not m:
        return float("nan")
    try:
        return float(m.group(0))
    except Exception:
        return float("nan")


def parse_percent_to_fraction(x: Union[str, float, int, None]) -> float:
    """解析百分比/质量分数为 0-1 的分数。

    - "80 wt%" -> 0.8
    - "0.2" -> 0.2（若本身已是 0-1）
    - 80 -> 0.8（若 >1 认为是百分数）
    """
    v = parse_first_number(x)
    if not np.isfinite(v):
        return float("nan")
    if v > 1.0:
        return v / 100.0
    return float(v)


def clamp01_np(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def alpha_max_from_r_np(r: np.ndarray) -> np.ndarray:
    """基于化学计量比 r 的理论最大固化度上限（工程近似）。

    alpha_max = min(1, r, 1/r), r>0
    """
    r = np.asarray(r, dtype=float)
    out = np.ones_like(r, dtype=float)
    valid = np.isfinite(r) & (r > 0)
    rv = r[valid]
    out[valid] = np.minimum(1.0, np.minimum(rv, 1.0 / rv))
    return clamp01_np(out)


def alpha_max_from_r_torch(r: "torch.Tensor") -> "torch.Tensor":
    """torch 版本 alpha_max（尽量兼容较旧 torch 版本）。"""
    # 旧版本 torch 可能没有 (r==r)；用 (r==r) 过滤 NaN
    valid = (r > 0) & (r == r)
    r_safe = torch.where(valid, r, torch.ones_like(r))
    # 防止极端值影响（如 inf）
    r_safe = torch.clamp(r_safe, min=1e-12, max=1e6)
    inv = 1.0 / torch.clamp(r_safe, min=1e-12)
    return torch.clamp(torch.minimum(torch.ones_like(r_safe), torch.minimum(r_safe, inv)), 0.0, 1.0)
def dibenedetto_tg_np(
    tg0: np.ndarray,
    tginf: np.ndarray,
    lam: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """DiBenedetto Tg-α 方程（numpy）。"""
    tg0 = np.asarray(tg0, dtype=float)
    tginf = np.asarray(tginf, dtype=float)
    lam = np.asarray(lam, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    denom = 1.0 - (1.0 - lam) * alpha
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    frac = (lam * alpha) / denom
    return tg0 + (tginf - tg0) * frac


def dibenedetto_tg_torch(
    tg0: "torch.Tensor",
    tginf: "torch.Tensor",
    lam: "torch.Tensor",
    alpha: "torch.Tensor",
) -> "torch.Tensor":
    """torch 版本 DiBenedetto 方程（可微）。"""
    denom = 1.0 - (1.0 - lam) * alpha
    denom = torch.where(torch.abs(denom) < 1e-8, torch.full_like(denom, 1e-8), denom)
    frac = (lam * alpha) / denom
    return tg0 + (tginf - tg0) * frac


def generalized_multimechanism_dibenedetto_tg_torch(
    tg0_c: "torch.Tensor",
    nu_mmol_g: "torch.Tensor",
    rigid_carbon_ratio: "torch.Tensor",
    alpha: "torch.Tensor",
    lam: "torch.Tensor",
    t_cure_max_c: "torch.Tensor",
    prob_alkyne: "torch.Tensor",
    delta_e_fwd: "torch.Tensor",
    kappa_nu: "torch.Tensor",
    kappa_rigid: "torch.Tensor",
    gamma_cure: "torch.Tensor",
    t0_base: float = 80.0,
) -> "torch.Tensor":
    """广义多机理自适应 DiBenedetto (GMM-DiBenedetto) 方程（可微）。

    物理扩展特性：
    1. Tginf 物理展开：Tginf = T0_base + kappa_nu * nu + kappa_rigid * rigid_carbon_ratio
       彻底将理论耐热极限从常数拟合解放为真实高分子交联密度与主链杂化刚度驱动；
    2. 主交联开环项：遵循自适应 DiBenedetto 网链热力学；
    3. 次级高温自聚项：当固化温度 > 180°C 时，激活烯丙基/炔基自由基热加成引起的耐热激增项，
       完美覆盖超高温高 Tg (>300°C) 异构网络体系。
    """
    # 1. 自适应理论极限耐热温度 Tginf(x)
    # nu_mmol_g 通常在 [1.0, 10.0]，rigid_carbon_ratio 在 [0.3, 1.0]
    tginf_primary = t0_base + kappa_nu * torch.clamp(nu_mmol_g, min=0.0) * 15.0 + kappa_rigid * torch.clamp(rigid_carbon_ratio, 0.0, 1.0) * 120.0
    tginf_primary = torch.maximum(tginf_primary, tg0_c + 15.0)

    # 2. 主通道开环交联项 (DiBenedetto)
    lam_safe = torch.clamp(lam, 0.15, 0.85)
    alpha_safe = torch.clamp(alpha, 0.0, 0.999)
    denom = 1.0 - (1.0 - lam_safe) * alpha_safe
    denom = torch.where(torch.abs(denom) < 1e-6, torch.full_like(denom, 1e-6), denom)
    frac_crosslink = (lam_safe * alpha_safe) / denom
    delta_tg_primary = (tginf_primary - tg0_c) * frac_crosslink

    # 3. 次级高温热自聚激增项 (针对 T_cure > 180°C 且含双键/炔基/共轭位点)
    # 热活化温度驱动门: ReLU(T_max - 180°C)
    t_therm_excess = torch.relu(t_cure_max_c - 180.0) / 50.0  # 归一化缩放
    kinetic_gate = 1.0 / (1.0 + torch.exp(torch.clamp((delta_e_fwd - 7.5) / 1.2, -15.0, 15.0)))
    delta_tg_secondary = gamma_cure * (prob_alkyne * 1.5 + 0.2) * t_therm_excess * kinetic_gate * 80.0

    return tg0_c + delta_tg_primary + delta_tg_secondary


def fox_tg_torch(
    tg0_c: "torch.Tensor",
    tginf_c: "torch.Tensor",
    alpha: "torch.Tensor",
) -> "torch.Tensor":
    """热力学绝对温标 (Kelvin) 计算的 Fox 方程（可微）。
    
    输入输出均为摄氏度 (°C)。严格防止摄氏度除法在低温/负温时数值崩塌。
    1 / T_g(K) = (1 - alpha) / T_g0(K) + alpha / T_ginf(K)
    """
    tg0_k = torch.clamp(tg0_c + 273.15, min=150.0)
    tginf_k = torch.clamp(tginf_c + 273.15, min=tg0_k + 5.0)
    alpha_clip = torch.clamp(alpha, 0.0, 0.999)
    inv_tg = (1.0 - alpha_clip) / tg0_k + alpha_clip / tginf_k
    tg_k = 1.0 / torch.clamp(inv_tg, min=1e-6)
    return tg_k - 273.15


def fox_loshaek_tg_torch(
    tg0_c: "torch.Tensor",
    nu_mol_m3: "torch.Tensor",
    k_nu: "torch.Tensor",
) -> "torch.Tensor":
    """Fox–Loshaek 交联密度关系（可微）。
    
    Tg = Tg0 + k_nu * (nu / 1000)
    交联密度越高 (nu 越大)，Tg 单调递增。
    输入输出均为摄氏度 (°C)。
    """
    nu_scaled = torch.clamp(nu_mol_m3 / 1000.0, min=0.0)
    return tg0_c + k_nu * nu_scaled


def halpin_tsai_np(E_m: np.ndarray, E_f: np.ndarray, xi: np.ndarray, vf: np.ndarray) -> np.ndarray:
    """Halpin–Tsai 复合材料弹性模量（工程近似，numpy）。

    E_c = E_m * (1 + xi*eta*Vf) / (1 - eta*Vf)
    eta = (E_f/E_m - 1) / (E_f/E_m + xi)
    """
    E_m = np.asarray(E_m, dtype=float)
    E_f = np.asarray(E_f, dtype=float)
    xi = np.asarray(xi, dtype=float)
    vf = np.asarray(vf, dtype=float)

    E_m_safe = np.where(E_m <= 1e-12, 1e-12, E_m)
    ratio = E_f / E_m_safe
    eta = (ratio - 1.0) / (ratio + xi + 1e-12)
    denom = 1.0 - eta * vf
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    return E_m_safe * (1.0 + xi * eta * vf) / denom


def halpin_tsai_torch(E_m: "torch.Tensor", E_f: "torch.Tensor", xi: "torch.Tensor", vf: "torch.Tensor") -> "torch.Tensor":
    """torch 版本 Halpin–Tsai（可微）。"""
    E_m_safe = torch.clamp(E_m, min=1e-12)
    ratio = E_f / E_m_safe
    eta = (ratio - 1.0) / (ratio + xi + 1e-12)
    denom = 1.0 - eta * vf
    denom = torch.where(torch.abs(denom) < 1e-8, torch.full_like(denom, 1e-8), denom)
    return E_m_safe * (1.0 + xi * eta * vf) / denom


@dataclass
class FillerProps:
    name: str
    E_gpa: float
    rho_g_cm3: float


# 典型填料的工程近似值（用于 Vf 换算与 Halpin–Tsai）
_FILLER_DB = [
    (("silica", "sio2", "nano-silica", "nanosilica"), 70.0, 2.20),
    (("alumina", "al2o3"), 300.0, 3.95),
    (("glass", "glass fiber", "gf"), 70.0, 2.55),
    (("carbon nanotube", "cnt", "mwcnt", "swcnt"), 1000.0, 1.80),
    (("graphene", "go", "rgo"), 1000.0, 2.20),
    (("clay", "montmorillonite", "mmt"), 170.0, 2.60),
    (("silver", "ag", "silver flake"), 83.0, 10.49),
    (("copper", "cu"), 120.0, 8.96),
    (("aluminum", "al"), 69.0, 2.70),
    (("boron nitride", "bn"), 800.0, 2.10),
]


def get_filler_props(text: Optional[str]) -> FillerProps:
    """从 nanofiller_type 文本猜测填料性质（工程先验）。"""
    if not text:
        return FillerProps(name="unknown", E_gpa=200.0, rho_g_cm3=2.50)

    s = str(text).strip().lower()
    for keys, E, rho in _FILLER_DB:
        if any(k in s for k in keys):
            return FillerProps(name=keys[0], E_gpa=float(E), rho_g_cm3=float(rho))
    return FillerProps(name="unknown", E_gpa=200.0, rho_g_cm3=2.50)


def volume_fraction_from_wt_fraction(
    wt_frac: Union[float, np.ndarray],
    rho_f: Union[float, np.ndarray],
    rho_m: float = 1.20,
) -> Union[float, np.ndarray]:
    """质量分数 -> 体积分数（Vf）换算。

    Vf = (w/rho_f) / ((w/rho_f) + ((1-w)/rho_m))
    """
    w = np.asarray(wt_frac, dtype=float)
    rho_f = np.asarray(rho_f, dtype=float)
    rho_m = float(rho_m)

    num = w / np.where(rho_f <= 1e-12, 1e-12, rho_f)
    den = num + (1.0 - w) / rho_m
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    vf = num / den
    return clamp01_np(vf)


def volume_fraction_from_wt_fraction_torch(
    wt_frac: "torch.Tensor",
    rho_f: "torch.Tensor",
    rho_m: float = 1.20,
) -> "torch.Tensor":
    """torch 版本：质量分数 -> 体积分数（Vf）换算。"""
    rho_m_t = torch.full_like(wt_frac, float(rho_m))
    rho_f_safe = torch.clamp(rho_f, min=1e-12)
    num = wt_frac / rho_f_safe
    den = num + (1.0 - wt_frac) / rho_m_t
    den = torch.where(torch.abs(den) < 1e-12, torch.full_like(den, 1e-12), den)
    vf = num / den
    return torch.clamp(vf, 0.0, 1.0)
