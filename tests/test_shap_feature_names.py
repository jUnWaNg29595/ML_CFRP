import numpy as np
import pandas as pd

from core.model_interpreter import EnhancedModelInterpreter, resolve_feature_names_for_matrix


def test_interpreter_prefers_real_training_dataframe_columns_over_stale_names():
    X_train = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        columns=["resin_xtb_gap", "curing_agent_xtb_gap"],
    )
    X_test = X_train.copy()

    interpreter = EnhancedModelInterpreter(
        model=None,
        X_train=X_train,
        y_train=np.array([1.0, 2.0]),
        X_test=X_test,
        y_test=np.array([1.5, 2.5]),
        model_name="test",
        feature_names=["Feature_0", "Feature_1"],
    )

    assert interpreter.feature_names == [
        "resin_xtb_gap",
        "curing_agent_xtb_gap",
    ]


def test_interpreter_uses_supplied_names_when_training_matrix_has_placeholders():
    X_train = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        columns=["feat_0", "feat_1"],
    )
    X_test = X_train.copy()

    interpreter = EnhancedModelInterpreter(
        model=None,
        X_train=X_train,
        y_train=np.array([1.0, 2.0]),
        X_test=X_test,
        y_test=np.array([1.5, 2.5]),
        model_name="test",
        feature_names=["resin_xtb_gap", "curing_agent_xtb_gap"],
    )

    assert interpreter.feature_names == [
        "resin_xtb_gap",
        "curing_agent_xtb_gap",
    ]


def test_interpreter_uses_supplied_names_when_training_columns_are_numeric():
    X_train = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]])
    X_test = X_train.copy()

    interpreter = EnhancedModelInterpreter(
        model=None,
        X_train=X_train,
        y_train=np.array([1.0, 2.0]),
        X_test=X_test,
        y_test=np.array([1.5, 2.5]),
        model_name="test",
        feature_names=["resin_xtb_gap", "curing_agent_xtb_gap"],
    )

    assert interpreter.feature_names == [
        "resin_xtb_gap",
        "curing_agent_xtb_gap",
    ]


def test_resolver_applies_training_feature_mask_before_shap_placeholder_fallback():
    original_names = [
        "feed_n2_fraction",
        "metal_ni",
        "metal_co_loading_wt_pct",
        "support_cao",
        "product_class",
        "metal_cu",
    ]
    feature_mask = [True, False, True, True, False, True]
    X_model = np.zeros((3, 4), dtype=float)

    resolved = resolve_feature_names_for_matrix(
        X_model,
        feature_names=original_names,
        feature_mask=feature_mask,
    )

    assert resolved == [
        "feed_n2_fraction",
        "metal_co_loading_wt_pct",
        "support_cao",
        "metal_cu",
    ]


def test_interpreter_fallback_feature_names_rescue_placeholder_resolution():
    """TabPFN 场景：X_train 为 ndarray、session 名单长度不匹配、模型无名字元数据。
    传入 train_result['feature_names'] 作为 fallback 时应恢复真实特征名。"""
    real_names = [f"mol_desc_{i:03d}" for i in range(6)]
    X_arr = np.arange(24, dtype=float).reshape(4, 6)

    interpreter = EnhancedModelInterpreter(
        model=None,
        X_train=X_arr,
        y_train=np.array([1.0, 2.0, 3.0, 4.0]),
        X_test=X_arr[:2].copy(),
        y_test=np.array([1.5, 2.5]),
        model_name="TabPFN",
        feature_names=[f"canonical_{i}" for i in range(8)],  # 长度不匹配
        fallback_feature_names=real_names,
    )

    assert interpreter.feature_names == real_names


def test_interpreter_resolves_names_from_model_feature_names_in_():
    """训练器注入 feature_names_in_ 后（TabPFN fit 后为 None 的场景），
    即使 X_train DataFrame 是占位列名也能解析出真实名。"""
    real_names = ["resin_total_phr", "mech_stoichiometry_r"]

    class FakeTabPFN:
        feature_names_in_ = None

    model = FakeTabPFN()
    model.feature_names_in_ = np.asarray(real_names, dtype=object)

    X_train = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["Feature_0", "Feature_1"])

    interpreter = EnhancedModelInterpreter(
        model=model,
        X_train=X_train,
        y_train=np.array([1.0, 2.0]),
        X_test=X_train.copy(),
        y_test=np.array([1.5, 2.5]),
        model_name="TabPFN",
        feature_names=[f"canonical_{i}" for i in range(9)],  # 长度不匹配
    )

    assert interpreter.feature_names == real_names
