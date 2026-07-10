# -*- coding: utf-8 -*-
"""反向预测模块：根据目标性能反向推导特征值"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize


FAST_DE_DIM_THRESHOLD = 20
FAST_DE_MAXITER = 30
FAST_DE_POPSIZE = 5


def _safe_model_predict(model, X):
    """兼容部分 XGBoost 模型中异常的 eval_metric 配置。"""
    try:
        return model.predict(X)
    except Exception as e:
        params = getattr(model, "get_xgb_params", lambda: {})()
        eval_metric = params.get("eval_metric") if isinstance(params, dict) else None
        if "Unknown metric function" not in str(e) or not isinstance(eval_metric, list):
            raise
        model.set_params(eval_metric="rmse")
        return model.predict(X)


def _resolve_method(method, n_features):
    if method == 'de' and n_features > FAST_DE_DIM_THRESHOLD:
        return 'slsqp'
    return method


def inverse_predict(model, target_value, feature_ranges, feature_names,
                   scaler=None, categorical_features=None, n_solutions=5, method='de'):
    """
    反向预测：根据目标值优化特征

    参数:
        model: 训练好的模型
        target_value: 目标性能值
        feature_ranges: dict, 每个特征的取值范围 {feature_name: (min, max)}
        feature_names: list, 特征名列表
        scaler: 标准化器（如果训练时使用了）
        categorical_features: list, 类别特征名列表（需要取整）
        n_solutions: 返回的解决方案数量
        method: 优化算法 'de'(差分进化), 'pso'(粒子群), 'slsqp'(梯度优化)

    返回:
        list of dict: 多个解决方案，每个包含特征值和预测值
    """
    if categorical_features is None:
        categorical_features = []

    categorical_indices = [i for i, name in enumerate(feature_names) if name in categorical_features]
    bounds = [feature_ranges.get(name, (0, 1)) for name in feature_names]
    effective_method = _resolve_method(method, len(feature_names))

    def objective(x):
        """目标函数：最小化预测值与目标值的差异"""
        x_rounded = x.copy()
        for idx in categorical_indices:
            x_rounded[idx] = round(x_rounded[idx])

        X = np.array(x_rounded).reshape(1, -1)
        if scaler:
            X = scaler.transform(X)
        pred = _safe_model_predict(model, X)[0]
        return (pred - target_value) ** 2

    solutions = []

    if effective_method == 'de':
        # 差分进化算法
        for i in range(n_solutions):
            result = differential_evolution(
                objective, bounds, seed=i, maxiter=FAST_DE_MAXITER, popsize=FAST_DE_POPSIZE,
                atol=1.0, tol=0.1, workers=1, updating='deferred'
            )
            solutions.append(_process_result(result.x, categorical_indices, feature_names,
                                            model, scaler, target_value))

    elif effective_method == 'pso':
        # 粒子群优化
        solutions = _particle_swarm_optimization(
            objective, bounds, categorical_indices, feature_names,
            model, scaler, target_value, n_solutions
        )

    elif effective_method == 'slsqp':
        # 梯度优化（多次随机初始化）
        for i in range(n_solutions):
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
            result = minimize(objective, x0, method='SLSQP', bounds=bounds,
                            options={'maxiter': 100, 'ftol': 1.0})
            solutions.append(_process_result(result.x, categorical_indices, feature_names,
                                            model, scaler, target_value))

    solutions.sort(key=lambda x: x['error'])
    return solutions[:n_solutions]


def _process_result(x_opt, categorical_indices, feature_names, model, scaler, target_value):
    """处理优化结果"""
    x_opt = x_opt.copy()
    for idx in categorical_indices:
        x_opt[idx] = round(x_opt[idx])

    X_pred = np.array(x_opt).reshape(1, -1)
    if scaler:
        X_pred_scaled = scaler.transform(X_pred)
        pred_value = _safe_model_predict(model, X_pred_scaled)[0]
    else:
        pred_value = _safe_model_predict(model, X_pred)[0]

    return {
        'features': {name: val for name, val in zip(feature_names, x_opt)},
        'predicted_value': pred_value,
        'error': abs(pred_value - target_value)
    }


def _particle_swarm_optimization(objective, bounds, categorical_indices, feature_names,
                                model, scaler, target_value, n_solutions):
    """粒子群优化算法"""
    n_particles = 20
    n_iterations = 50
    w = 0.7  # 惯性权重
    c1 = 1.5  # 个体学习因子
    c2 = 1.5  # 社会学习因子

    dim = len(bounds)
    lower = np.array([b[0] for b in bounds])
    upper = np.array([b[1] for b in bounds])

    solutions = []

    for _ in range(n_solutions):
        # 初始化粒子
        particles = np.random.uniform(lower, upper, (n_particles, dim))
        velocities = np.random.uniform(-1, 1, (n_particles, dim))

        pbest = particles.copy()
        pbest_scores = np.array([objective(p) for p in particles])
        gbest = pbest[np.argmin(pbest_scores)]

        # 迭代优化
        for _ in range(n_iterations):
            for i in range(n_particles):
                r1, r2 = np.random.rand(2)
                velocities[i] = (w * velocities[i] +
                               c1 * r1 * (pbest[i] - particles[i]) +
                               c2 * r2 * (gbest - particles[i]))
                particles[i] += velocities[i]
                particles[i] = np.clip(particles[i], lower, upper)

                score = objective(particles[i])
                if score < pbest_scores[i]:
                    pbest[i] = particles[i]
                    pbest_scores[i] = score
                    if score < objective(gbest):
                        gbest = particles[i]

        solutions.append(_process_result(gbest, categorical_indices, feature_names,
                                        model, scaler, target_value))

    return solutions
