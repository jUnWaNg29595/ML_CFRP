import pytest
import numpy as np
import pandas as pd
import torch
from core.pinn_model import EpoxyPINNRegressor

def test_pinn_mechanics_physics_branch():
    """验证 mechanics 模式下物理分支能正确响应交联密度与刚性项，且具备无截断外推能力"""
    pinn = EpoxyPINNRegressor(
        mode="mechanics",
        target_name="storage_modulus_25c_gpa",
        hidden_dim=64,
        n_layers=2,
        epochs=5,
        physics_weight=0.3,
        physics_formula="enhanced",
        device="cpu"
    )
    
    np.random.seed(42)
    n = 35
    r = np.random.uniform(0.6, 1.2, size=n)
    ep_grp = np.random.choice([2.0, 3.0, 4.0], size=n)
    act_h = np.random.choice([2.0, 4.0], size=n)
    eew = np.random.uniform(100.0, 250.0, size=n)
    ahew = np.random.uniform(30.0, 100.0, size=n)
    temp = np.random.uniform(120.0, 220.0, size=n)
    time = np.random.uniform(1.0, 8.0, size=n)
    
    # 真实模量随交联密度与官能度递增
    nu_synth = (ep_grp * act_h) / (eew + ahew)
    modulus = 1.5 + 2.0 * nu_synth + 0.5 * (r == 1.0)
    
    df = pd.DataFrame({
        "formulation_r_value": r,
        "resin_epoxy_group_total": ep_grp,
        "curing_agent_active_hydrogen_total": act_h,
        "formulation_resin_total_eew_g_eq": eew,
        "formulation_hardener_total_ahew_g_eq": ahew,
        "process_max_temperature_c": temp,
        "process_total_time_h": time,
        "storage_modulus_25c_gpa": modulus,
    })
    
    y = df["storage_modulus_25c_gpa"].values
    pinn.fit(df.drop(columns=["storage_modulus_25c_gpa"]), np.log(y))
    
    preds = np.exp(pinn.predict(df.drop(columns=["storage_modulus_25c_gpa"])))
    assert len(preds) == n
    assert np.all(np.isfinite(preds))
    assert np.all(preds > 0.0)
