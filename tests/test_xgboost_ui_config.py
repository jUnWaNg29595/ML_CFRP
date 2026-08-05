from core.ui_config import MANUAL_TUNING_PARAMS


def test_xgboost_tree_controls_allow_10000_estimators():
    for model_name in ("XGBoost", "XGBoost分类"):
        config = next(
            item
            for item in MANUAL_TUNING_PARAMS[model_name]
            if item["name"] == "n_estimators"
        )

        assert config["args"]["max_value"] == 10000
