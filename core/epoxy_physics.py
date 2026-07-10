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
