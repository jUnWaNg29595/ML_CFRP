import pytest
import numpy as np
import pandas as pd

def test_high_residual_contributor_math():
    np.random.seed(42)
    n = 100
    y_true = np.random.uniform(50, 250, size=n)
    noise = np.random.normal(0, 3, size=n)
    
    # 注入 5 个极端离群点
    noise[10] = 85.0
    noise[25] = 92.0
    noise[50] = 78.0
    noise[75] = 65.0
    noise[90] = 70.0
    
    y_pred = y_true + noise
    
    resids = np.abs(y_true - y_pred)
    sq_err = resids ** 2
    total_sse = np.sum(sq_err)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    orig_r2 = 1.0 - (total_sse / sst)
    
    contrib_pct = (sq_err / total_sse) * 100.0
    order = np.argsort(sq_err)[::-1]
    
    top_culprits = []
    cum_sse = 0.0
    for idx in order:
        p_val = contrib_pct[idx]
        if p_val >= 3.0:
            top_culprits.append(idx)
            cum_sse += sq_err[idx]
        elif (cum_sse / total_sse < 0.25) and len(top_culprits) < 5:
            top_culprits.append(idx)
            cum_sse += sq_err[idx]
        else:
            break
            
    assert len(top_culprits) >= 5
    assert set([10, 25, 50, 75, 90]).issubset(set(top_culprits))
    
    cum_pct = (cum_sse / total_sse) * 100.0
    assert cum_pct > 80.0
    
    hypo_sse = total_sse - cum_sse
    hypo_r2 = 1.0 - (hypo_sse / sst)
    
    assert orig_r2 < hypo_r2
    assert hypo_r2 > 0.99

def test_multicomponent_and_exact_row_mapping():
    # 测试多组分树脂+固化剂配方解析
    row = {
        'resin_1_name': 'DGEBA (E-51)',
        'resin_1_structure': 'CC(C)(c1ccc(OCC2CO2)cc1)c1ccc(OCC2CO2)cc1',
        'resin_1_amount_phr': 80.0,
        'resin_2_name': 'TGDDM',
        'resin_2_structure': 'c1cc(N(CC2CO2)CC2CO2)ccc1Cc1ccc(N(CC2CO2)CC2CO2)cc1',
        'resin_2_amount_phr': 20.0,
        'curing_agent_1_name': "4,4'-DDS",
        'curing_agent_1_structure': 'Nc1ccc(S(=O)(=O)c2ccc(N)cc2)cc1',
        'curing_agent_1_amount_phr': 35.0,
        'curing_agent_2_name': '2-MI (2-甲基咪唑)',
        'curing_agent_2_structure': 'Cc1ncc[nH]1',
        'curing_agent_2_amount_phr': 1.5,
        'nanofiller_type': 'SiO2纳米二氧化硅',
        'nanofiller_content': 5.0,
    }
    
    resins = []
    for i in range(1, 6):
        s = row.get(f"resin_{i}_structure", "")
        n = row.get(f"resin_{i}_name", "")
        p = row.get(f"resin_{i}_amount_phr")
        if s or n:
            label = n if n else s
            phr_str = f" [{float(p):.1f} phr]" if p else ""
            resins.append(f"{label}{phr_str}")
            
    assert len(resins) == 2
    assert "DGEBA (E-51) [80.0 phr]" in resins[0]
    assert "TGDDM [20.0 phr]" in resins[1]
    
    curers = []
    for i in range(1, 6):
        s = row.get(f"curing_agent_{i}_structure", "")
        n = row.get(f"curing_agent_{i}_name", "")
        p = row.get(f"curing_agent_{i}_amount_phr")
        if s or n:
            label = n if n else s
            phr_str = f" [{float(p):.1f} phr]" if p else ""
            curers.append(f"{label}{phr_str}")
            
    assert len(curers) == 2
    assert "4,4'-DDS [35.0 phr]" in curers[0]
    assert "2-MI (2-甲基咪唑) [1.5 phr]" in curers[1]
