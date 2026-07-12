# -*- coding: utf-8 -*-
"""强制修复XGBoost早停 - 直接修改fit调用"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.model_trainer import ModelTrainer
import inspect

# 获取_safe_xgb_fit的源代码
print("="*60)
print("检查_safe_xgb_fit函数")
print("="*60)

from core.model_trainer import _safe_xgb_fit
print(inspect.getsource(_safe_xgb_fit))

print("\n" + "="*60)
print("检查XGBoost版本")
print("="*60)

try:
    import xgboost as xgb
    print(f"XGBoost版本: {xgb.__version__}")

    # 检查XGBRegressor的fit方法签名
    from xgboost import XGBRegressor
    model = XGBRegressor()
    sig = inspect.signature(model.fit)
    print(f"\nXGBRegressor.fit()参数:")
    for param_name, param in sig.parameters.items():
        print(f"  - {param_name}: {param.annotation if param.annotation != inspect.Parameter.empty else 'Any'}")

    # 检查是否支持early_stopping_rounds
    if 'early_stopping_rounds' in sig.parameters:
        print("\n✓ fit()方法支持early_stopping_rounds参数")
    else:
        print("\n❌ fit()方法不支持early_stopping_rounds参数")

    # 检查XGBRegressor初始化参数
    init_sig = inspect.signature(XGBRegressor.__init__)
    if 'early_stopping_rounds' in init_sig.parameters:
        print("✓ __init__()方法支持early_stopping_rounds参数")
    else:
        print("❌ __init__()方法不支持early_stopping_rounds参数")

except Exception as e:
    print(f"检查失败: {e}")
    import traceback
    traceback.print_exc()
