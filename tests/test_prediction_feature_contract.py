import numpy as np

from core.prediction_contract import resolve_prediction_feature_contract


class _Model:
    n_features_in_ = 2
    feature_names_in_ = np.array(["resin_xtb_gap", "curing_agent_xtb_gap"])


def test_contract_uses_model_names_and_reports_extra_artifact_columns():
    report = resolve_prediction_feature_contract(
        model=_Model(),
        artifact={
            "feature_cols": [
                "resin_xtb_gap",
                "curing_agent_xtb_gap",
                "stale_extra_feature",
            ]
        },
    )

    assert report["ok"] is True
    assert report["feature_cols"] == [
        "resin_xtb_gap",
        "curing_agent_xtb_gap",
    ]
    assert report["extra_features"] == ["stale_extra_feature"]
    assert report["expected_count"] == 2


def test_contract_reports_missing_features_without_silent_padding():
    report = resolve_prediction_feature_contract(
        model=_Model(),
        artifact={"feature_cols": ["resin_xtb_gap"]},
    )

    assert report["ok"] is False
    assert report["missing_features"] == ["curing_agent_xtb_gap"]
    assert report["feature_cols"] == [
        "resin_xtb_gap",
        "curing_agent_xtb_gap",
    ]


def test_contract_reorders_artifact_columns_to_model_order():
    report = resolve_prediction_feature_contract(
        model=_Model(),
        artifact={
            "feature_cols": [
                "curing_agent_xtb_gap",
                "resin_xtb_gap",
            ]
        },
    )

    assert report["ok"] is True
    assert report["feature_cols"] == [
        "resin_xtb_gap",
        "curing_agent_xtb_gap",
    ]
    assert report["order_mismatch"] is True
