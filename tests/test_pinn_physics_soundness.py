"""
测试重构后的 Epoxy PINN 物理信息神经网络：
1. 验证化学计量比识别优先级（严禁将 cp_r_value 误当成 r_value）
2. 验证热力学 Fox 方程（Kelvin 绝对温标，低温/负温单体不崩塌）
3. 验证 Fox–Loshaek 方程（交联密度越大，Tg 严格单调递增，杜绝算成负几百度的恶性 Bug）
4. 验证 ResMLP 架构具备真实的高分子性能外推能力（消除后置 LayerNorm 造成的水平平顶）
"""
import pytest
import numpy as np
import pandas as pd
import torch

from core.pinn_model import EpoxyPINNRegressor
from core.epoxy_physics import (
    dibenedetto_tg_torch,
    fox_tg_torch,
    fox_loshaek_tg_torch,
    alpha_max_from_r_torch,
)


def test_stoichiometric_ratio_identification_bans_cp_r_value():
    """验证：当同时存在 formulation_r_value 与 cp_r_value 时，必须严格选择 formulation_r_value。"""
    df = pd.DataFrame(columns=[
        "formulation_r_value",
        "cp_r_value",
        "resin_smiles",
        "cure_temperature",
    ])
    pinn = EpoxyPINNRegressor()
    pinn._identify_special_columns(df)
    assert pinn._col_r_ == "formulation_r_value", f"识别错误: 选中了 {pinn._col_r_}"


def test_fox_kelvin_soundness_on_cold_monomers():
    """验证：Fox 方程必须采用开尔文计算，对常温液体/负温单体（Tg0 <= 0°C）计算稳定且合理。"""
    tg0 = torch.tensor([-15.0])    # 常见低熔点液体固化剂 (如 MTHPA)
    tginf = torch.tensor([200.0])  # 固化后树脂
    alpha = torch.tensor([0.9])    # 90% 固化度

    fox_pred = fox_tg_torch(tg0_c=tg0, tginf_c=tginf, alpha=alpha)
    pred_val = fox_pred.item()

    # 在 Kelvin 下计算出的正确预测应该在 150°C ~ 190°C 之间
    assert 140.0 < pred_val < 195.0, f"Fox 计算异常: {pred_val}°C"
    # 绝不能崩塌至 0°C 或负温
    assert pred_val > 50.0


def test_fox_loshaek_monotonicity_with_crosslink_density():
    """验证：Fox–Loshaek 方程中，交联密度越高，Tg 必须单调递增，绝不能输出负数。"""
    tg0 = torch.tensor([20.0])
    k_nu = torch.tensor([30.0])

    nu_low = torch.tensor([1000.0])   # 松散网络
    nu_mid = torch.tensor([4000.0])   # 普通交联网络
    nu_high = torch.tensor([8000.0])  # 超高密度耐热网络 (如 TGDDM, BMI)

    tg_low = fox_loshaek_tg_torch(tg0, nu_low, k_nu).item()
    tg_mid = fox_loshaek_tg_torch(tg0, nu_mid, k_nu).item()
    tg_high = fox_loshaek_tg_torch(tg0, nu_high, k_nu).item()

    assert tg_low < tg_mid < tg_high, f"违反物理单调性: {tg_low}, {tg_mid}, {tg_high}"
    # 超高交联网络应该显著升温 (Tg > 200°C)
    assert tg_high >= 250.0, f"超高交联耐热 Tg 过低: {tg_high}"
    # 彻底杜绝旧代码中算出 -674°C 的严重 Bug
    assert tg_low > 0.0 and tg_mid > 0.0 and tg_high > 0.0


def test_resmlp_extrapolation_capability():
    """验证：基于 ResMLP 与解封惩罚后的 EpoxyPINN，当面对高耐温工况时能够突破训练集上限实现外推。"""
    np.random.seed(42)
    n = 150
    X_train = pd.DataFrame({
        "resin_total_eew_g_eq": np.random.uniform(180, 350, n),
        "hardener_total_ahew_g_eq": np.random.uniform(40, 90, n),
        "formulation_r_value": np.random.uniform(0.9, 1.1, n),
        "cure_temp": np.random.uniform(120, 180, n),
    })
    # 目标真实规律：温度越高、EEW 越小 (交联越密) -> Tg 越高
    y_train = 40.0 + 0.5 * X_train["cure_temp"] + 120.0 * (200.0 / X_train["resin_total_eew_g_eq"])

    y_max = y_train.max()

    pinn = EpoxyPINNRegressor(mode="tg", epochs=60, seed=42, physics_weight=0.3, lr=0.003)
    pinn.fit(X_train, y_train.values)

    # 构造超出训练集温度和交联密度的极端耐热候选
    X_extrap = pd.DataFrame({
        "resin_total_eew_g_eq": [90.0],
        "hardener_total_ahew_g_eq": [25.0],
        "formulation_r_value": [1.0],
        "cure_temp": [260.0],
    })
    preds = pinn.predict(X_extrap)
    pred_val = float(preds[0])

    # 验证预测值成功超越训练集上限 y_max，具备真正的外推能力
    assert pred_val > y_max, f"预测值未实现外推: pred={pred_val:.1f}, y_max={y_max:.1f}"
    assert pred_val >= 260.0, f"外推响应不足: pred={pred_val:.1f}"
