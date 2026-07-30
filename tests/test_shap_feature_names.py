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
