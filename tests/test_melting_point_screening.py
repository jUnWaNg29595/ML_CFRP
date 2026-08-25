import math

import pandas as pd
import pytest

from core.melting_point_screening import (
    apply_melting_point_gate,
    melting_point_filter_status,
)


def test_status_passes_only_when_prediction_plus_std_is_within_limit():
    assert melting_point_filter_status(120.0, 4.0, 80.0, 130.0, 10.0, 50.0) == (
        "pass",
        "within_limits",
    )


@pytest.mark.parametrize(
    ("prediction", "std", "ad_score", "expected_reason"),
    [
        (127.0, 4.0, 80.0, "prediction_plus_std_exceeds_limit"),
        (120.0, 11.0, 80.0, "std_exceeds_limit"),
        (120.0, 4.0, 49.0, "ad_below_minimum"),
    ],
)
def test_status_fails_when_a_finite_gate_constraint_is_not_met(
    prediction, std, ad_score, expected_reason
):
    status, reason = melting_point_filter_status(
        prediction, std, ad_score, 130.0, 10.0, 50.0
    )

    assert status == "fail"
    assert reason == expected_reason


@pytest.mark.parametrize(
    ("prediction", "std", "ad_score", "expected_reason"),
    [
        (math.nan, 4.0, 80.0, "non_finite_prediction"),
        (120.0, math.inf, 80.0, "non_finite_std"),
        (120.0, 4.0, math.nan, "non_finite_ad_score"),
    ],
)
def test_status_is_unknown_when_prediction_uncertainty_or_ad_is_not_finite(
    prediction, std, ad_score, expected_reason
):
    status, reason = melting_point_filter_status(
        prediction, std, ad_score, 130.0, 10.0, 50.0
    )

    assert status == "unknown"
    assert reason == expected_reason


def test_annotate_keeps_rows_and_applies_separate_role_limits():
    frame = pd.DataFrame(
        [
            {
                "resin_mp_predicted_c": 120.0,
                "resin_mp_std_c": 4.0,
                "resin_mp_ad_score": 80.0,
                "hardener_mp_predicted_c": 135.0,
                "hardener_mp_std_c": 4.0,
                "hardener_mp_ad_score": 80.0,
            }
        ]
    )

    result = apply_melting_point_gate(
        frame,
        resin_limit_c=130.0,
        hardener_limit_c=130.0,
        max_std_c=10.0,
        min_ad_score=50.0,
        mode="annotate",
    )

    assert len(result) == 1
    assert result.iloc[0]["resin_mp_filter_status"] == "pass"
    assert result.iloc[0]["resin_mp_filter_reason"] == "within_limits"
    assert result.iloc[0]["hardener_mp_filter_status"] == "fail"
    assert result.iloc[0]["hardener_mp_filter_reason"] == "prediction_plus_std_exceeds_limit"


def test_strict_keeps_only_rows_where_available_roles_pass():
    frame = pd.DataFrame(
        [
            {
                "resin_mp_predicted_c": 120.0,
                "resin_mp_std_c": 4.0,
                "resin_mp_ad_score": 80.0,
                "hardener_mp_predicted_c": 125.0,
                "hardener_mp_std_c": 4.0,
                "hardener_mp_ad_score": 80.0,
            },
            {
                "resin_mp_predicted_c": 120.0,
                "resin_mp_std_c": 4.0,
                "resin_mp_ad_score": 80.0,
                "hardener_mp_predicted_c": 135.0,
                "hardener_mp_std_c": 4.0,
                "hardener_mp_ad_score": 80.0,
            },
        ]
    )

    result = apply_melting_point_gate(
        frame,
        resin_limit_c=130.0,
        hardener_limit_c=130.0,
        max_std_c=10.0,
        min_ad_score=50.0,
        mode="strict",
    )

    assert result.index.tolist() == [0]


def test_missing_hardener_columns_only_gate_existing_resin_role():
    frame = pd.DataFrame(
        [
            {
                "resin_mp_predicted_c": 120.0,
                "resin_mp_std_c": 4.0,
                "resin_mp_ad_score": 80.0,
            },
            {
                "resin_mp_predicted_c": 140.0,
                "resin_mp_std_c": 4.0,
                "resin_mp_ad_score": 80.0,
            },
        ]
    )

    result = apply_melting_point_gate(
        frame,
        resin_limit_c=130.0,
        hardener_limit_c=130.0,
        max_std_c=10.0,
        min_ad_score=50.0,
        mode="strict",
    )

    assert result.index.tolist() == [0]
    assert "hardener_mp_filter_status" not in result


def test_missing_value_in_existing_hardener_role_is_unknown_and_not_strict_pass():
    frame = pd.DataFrame(
        [
            {
                "resin_mp_predicted_c": 120.0,
                "resin_mp_std_c": 4.0,
                "resin_mp_ad_score": 80.0,
                "hardener_mp_predicted_c": 125.0,
                "hardener_mp_std_c": None,
                "hardener_mp_ad_score": 80.0,
            }
        ]
    )

    annotated = apply_melting_point_gate(
        frame,
        resin_limit_c=130.0,
        hardener_limit_c=130.0,
        max_std_c=10.0,
        min_ad_score=50.0,
        mode="annotate",
    )
    strict = apply_melting_point_gate(
        frame,
        resin_limit_c=130.0,
        hardener_limit_c=130.0,
        max_std_c=10.0,
        min_ad_score=50.0,
        mode="strict",
    )

    assert annotated.iloc[0]["hardener_mp_filter_status"] == "unknown"
    assert strict.empty
