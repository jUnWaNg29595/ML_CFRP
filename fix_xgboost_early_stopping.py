# -*- coding: utf-8 -*-
"""XGBoost早停终极修复方案"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

print("="*60)
print("XGBoost早停诊断")
print("="*60)

# 检查XGBoost版本
try:
    import xgboost as xgb
    print(f"\nXGBoost版本: {xgb.__version__}")
    version_parts = xgb.__version__.split('.')
    major = int(version_parts[0])
    minor = int(version_parts[1]) if len(version_parts) > 1 else 0

    print(f"主版本: {major}, 次版本: {minor}")

    # 生成测试数据
    X, y = make_regression(n_samples=1000, n_features=20, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n数据集: {X_train.shape[0]} 训练, {X_test.shape[0]} 测试")

    # 测试不同的early_stopping方法
    print("\n" + "="*60)
    print("测试1: XGBoost 2.0+ 新API (callbacks)")
    print("="*60)

    if major >= 2:
        try:
            from xgboost.callback import EarlyStopping

            model = xgb.XGBRegressor(
                n_estimators=5000,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                callbacks=[EarlyStopping(rounds=50, save_best=True)]
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
            )

            print(f"✓ 成功!")
            print(f"  最佳迭代: {model.best_iteration}")
            print(f"  总迭代数: {model.n_estimators}")

            if model.best_iteration < model.n_estimators - 50:
                print(f"  ✓ 早停生效,节省了 {model.n_estimators - model.best_iteration} 轮")
            else:
                print(f"  ❌ 早停未生效")

        except Exception as e:
            print(f"❌ 失败: {e}")
    else:
        print("跳过(版本<2.0)")

    # 测试2: 传统方法
    print("\n" + "="*60)
    print("测试2: 传统方法 (fit参数)")
    print("="*60)

    try:
        model2 = xgb.XGBRegressor(
            n_estimators=5000,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

        model2.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )

        print(f"✓ 成功!")
        if hasattr(model2, 'best_iteration'):
            print(f"  最佳迭代: {model2.best_iteration}")
            print(f"  总迭代数: {model2.n_estimators}")

            if model2.best_iteration < model2.n_estimators - 50:
                print(f"  ✓ 早停生效,节省了 {model2.n_estimators - model2.best_iteration} 轮")
            else:
                print(f"  ❌ 早停未生效")
        else:
            print(f"  ❌ 没有best_iteration属性")

    except Exception as e:
        print(f"❌ 失败: {e}")

    # 测试3: 原生API
    print("\n" + "="*60)
    print("测试3: 原生API (xgb.train)")
    print("="*60)

    try:
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'seed': 42
        }

        evals = [(dtrain, 'train'), (dtest, 'test')]
        model3 = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )

        print(f"✓ 成功!")
        print(f"  最佳迭代: {model3.best_iteration}")
        print(f"  实际训练轮数: {model3.num_boosted_rounds()}")

        if model3.best_iteration < 4950:
            print(f"  ✓ 早停生效,节省了 {5000 - model3.best_iteration} 轮")
        else:
            print(f"  ❌ 早停未生效")

    except Exception as e:
        print(f"❌ 失败: {e}")

    # 给出建议
    print("\n" + "="*60)
    print("修复建议")
    print("="*60)

    if major >= 2:
        print("\n您的XGBoost版本>=2.0,推荐使用callbacks方式:")
        print("""
from xgboost.callback import EarlyStopping

model = XGBRegressor(
    n_estimators=5000,
    callbacks=[EarlyStopping(rounds=50, save_best=True)]
)
model.fit(X, y, eval_set=[(X_val, y_val)])
        """)
    else:
        print("\n您的XGBoost版本<2.0,使用传统方式:")
        print("""
model = XGBRegressor(n_estimators=5000)
model.fit(
    X, y,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=50
)
        """)

    print("\n如果以上都不行,建议:")
    print("1. 升级XGBoost: pip install --upgrade xgboost")
    print("2. 或降级到稳定版本: pip install xgboost==1.7.6")

except Exception as e:
    print(f"\n诊断失败: {e}")
    import traceback
    traceback.print_exc()
