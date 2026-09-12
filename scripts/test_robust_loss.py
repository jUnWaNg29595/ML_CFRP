# -*- coding: utf-8 -*-
"""验证鲁棒损失（Huber/MAE）选项真实生效。"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd


def make_data(n=220, seed=7, n_outliers=12):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 4))
    y = X[:, 0] * 3.0 + X[:, 1] * 1.5 + rng.normal(0, 0.3, n)
    idx = rng.choice(n, n_outliers, replace=False)
    y[idx] += rng.choice([-1, 1], n_outliers) * rng.uniform(15, 30, n_outliers)  # 大异常
    return pd.DataFrame(X, columns=[f"f{i}" for i in range(4)]), pd.Series(y)


def test_ann_loss_name():
    from core.ann_model import ANNRegressor
    m = ANNRegressor(hidden_layer_sizes_str="16,", epochs=3, loss_name="huber", verbose=False)
    assert m.loss_name == "huber"
    config = m._validate_params()
    assert config["loss_name"] == "huber"
    try:
        bad = ANNRegressor(loss_name="bogus"); bad._validate_params()
        raise AssertionError("非法 loss_name 应报错")
    except ValueError as e:
        assert "loss_name" in str(e)
    # 兼容旧实例（无 loss_name 属性）
    old = ANNRegressor.__new__(ANNRegressor)
    old.__dict__.update({k: v for k, v in ANNRegressor(verbose=False).__dict__.items()})
    # 模拟旧 pickle：删除 loss_name
    old.__dict__.pop("loss_name", None)
    old._ensure_compat_attributes()
    assert old.loss_name == "mse", "旧模型应回落到 mse"
    print("[PASS] ANN loss_name 参数/校验/兼容")


def test_ann_training_huber():
    from core.ann_model import ANNRegressor
    X, y = make_data()
    m = ANNRegressor(hidden_layer_sizes_str="32,16,", epochs=8, batch_size=64,
                     loss_name="huber", verbose=False, early_stopping=False, random_state=0)
    m.fit(X.values, y.values)
    assert np.isfinite(m.train_loss_history).all() and len(m.train_loss_history) == 8
    print(f"[PASS] ANN huber 训练收敛: final_train_loss={m.train_loss_history[-1]:.4f}")


def test_xgb_objective():
    from core.model_trainer import EnhancedModelTrainer
    X, y = make_data()
    trainer = EnhancedModelTrainer(use_gpu=False)
    res_mse = trainer.train_model(X, y, model_name="XGBoost", test_size=0.2, random_state=42,
                                  n_estimators=80, early_stopping_rounds=0, verbosity=0)
    model_mse = res_mse["model"]
    assert model_mse.get_params()["objective"] in ("reg:squarederror", None), model_mse.get_params()["objective"]
    res_hub = trainer.train_model(X, y, model_name="XGBoost", test_size=0.2, random_state=42,
                                  n_estimators=80, early_stopping_rounds=0, verbosity=0,
                                  objective="reg:pseudohubererror")
    assert res_hub["model"].get_params()["objective"] == "reg:pseudohubererror"
    print(f"[PASS] XGBoost objective 透传: huber RMSE={res_hub['rmse']:.3f} vs mse RMSE={res_mse['rmse']:.3f}")


def test_lgbm_objective():
    from core.model_trainer import EnhancedModelTrainer
    X, y = make_data()
    trainer = EnhancedModelTrainer(use_gpu=False)
    res = trainer.train_model(X, y, model_name="LightGBM", test_size=0.2, random_state=42,
                              n_estimators=80, objective="huber")
    assert res["model"].get_params()["objective"] == "huber"
    res2 = trainer.train_model(X, y, model_name="LightGBM", test_size=0.2, random_state=42,
                               n_estimators=80, objective="regression_l1")
    assert res2["model"].get_params()["objective"] == "regression_l1"
    print(f"[PASS] LightGBM objective 透传: huber RMSE={res['rmse']:.3f}, l1 RMSE={res2['rmse']:.3f}")


def test_catboost_loss_function():
    from core.model_trainer import EnhancedModelTrainer
    X, y = make_data()
    trainer = EnhancedModelTrainer(use_gpu=False)
    res = trainer.train_model(X, y, model_name="CatBoost", test_size=0.2, random_state=42,
                              iterations=80, loss_function="MAE", verbose=0)
    assert res["model"].get_params()["loss_function"] == "MAE"
    res2 = trainer.train_model(X, y, model_name="CatBoost", test_size=0.2, random_state=42,
                               iterations=80, loss_function="Huber:delta=1.0", verbose=0)
    assert res2["model"].get_params()["loss_function"] == "Huber:delta=1.0"
    print(f"[PASS] CatBoost loss_function 透传: MAE RMSE={res['rmse']:.3f}, Huber RMSE={res2['rmse']:.3f}")


def test_gbdt_loss():
    from core.model_trainer import EnhancedModelTrainer
    X, y = make_data()
    trainer = EnhancedModelTrainer(use_gpu=False)
    res = trainer.train_model(X, y, model_name="梯度提升树", test_size=0.2, random_state=42,
                              n_estimators=80, loss="huber", verbose=0)
    assert res["model"].loss == "huber"
    print(f"[PASS] 梯度提升树 loss=huber RMSE={res['rmse']:.3f}")


def test_ui_config_wiring():
    from core.ui_config import MANUAL_TUNING_PARAMS, prepare_manual_training_params
    assert any(c["name"] == "objective" for c in MANUAL_TUNING_PARAMS["XGBoost"])
    assert any(c["name"] == "objective" for c in MANUAL_TUNING_PARAMS["LightGBM"])
    assert any(c["name"] == "loss_function" for c in MANUAL_TUNING_PARAMS["CatBoost"])
    assert any(c["name"] == "loss" for c in MANUAL_TUNING_PARAMS["梯度提升树"])
    assert any(c["name"] == "loss_name" for c in MANUAL_TUNING_PARAMS["人工神经网络"])
    params = prepare_manual_training_params({"objective": "reg:pseudohubererror", "random_state": 1})
    assert params["objective"] == "reg:pseudohubererror" and "random_state" not in params
    print("[PASS] UI 参数配置齐全且透传正常")


def test_huber_beats_mse_on_outliers():
    """统计验证：含 12/220 大异常时，huber 目标训练的测试 RMSE 应普遍不差于 mse。"""
    from core.model_trainer import EnhancedModelTrainer
    X, y = make_data()
    trainer = EnhancedModelTrainer(use_gpu=False)
    wins, total = 0, 0
    for seed in (42, 7, 123):
        r_mse = trainer.train_model(X, y, model_name="XGBoost", test_size=0.2, random_state=seed,
                                    n_estimators=100, early_stopping_rounds=0, verbosity=0)
        r_hub = trainer.train_model(X, y, model_name="XGBoost", test_size=0.2, random_state=seed,
                                    n_estimators=100, early_stopping_rounds=0, verbosity=0,
                                    objective="reg:pseudohubererror")
        total += 1
        if r_hub["rmse"] <= r_mse["rmse"] * 1.02:  # 允许 2% 容差
            wins += 1
        print(f"    seed={seed}: mse={r_mse['rmse']:.3f}, huber={r_hub['rmse']:.3f}")
    assert wins >= total - 1, f"huber 胜率过低: {wins}/{total}"
    print(f"[PASS] Huber 抗异常对比: {wins}/{total} 组不劣于 MSE")


if __name__ == "__main__":
    test_ann_loss_name()
    test_ann_training_huber()
    test_xgb_objective()
    test_lgbm_objective()
    test_catboost_loss_function()
    test_gbdt_loss()
    test_ui_config_wiring()
    test_huber_beats_mse_on_outliers()
    print("\nALL TESTS PASSED")
