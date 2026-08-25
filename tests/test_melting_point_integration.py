import pandas as pd

from core.melting_point_screening import (
    apply_melting_point_gate,
    build_melting_point_artifact_extra,
    validate_melting_point_artifact,
)


def _gate_kwargs():
    return {
        "resin_limit_c": 130.0,
        "hardener_limit_c": 130.0,
        "max_std_c": 10.0,
        "min_ad_score": 50.0,
    }


def test_end_to_end_annotation_keeps_original_tg_prediction_and_all_candidates():
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "pass",
                "prediction": 185.0,
                "resin_mp_predicted_c": 118.0,
                "resin_mp_std_c": 4.0,
                "resin_mp_ad_score": 82.0,
                "hardener_mp_predicted_c": 124.0,
                "hardener_mp_std_c": 3.0,
                "hardener_mp_ad_score": 76.0,
            },
            {
                "candidate_id": "resin-limit",
                "prediction": 190.0,
                "resin_mp_predicted_c": 129.0,
                "resin_mp_std_c": 2.0,
                "resin_mp_ad_score": 82.0,
                "hardener_mp_predicted_c": 124.0,
                "hardener_mp_std_c": 3.0,
                "hardener_mp_ad_score": 76.0,
            },
            {
                "candidate_id": "hardener-unknown",
                "prediction": 195.0,
                "resin_mp_predicted_c": 118.0,
                "resin_mp_std_c": 4.0,
                "resin_mp_ad_score": 82.0,
                "hardener_mp_predicted_c": None,
                "hardener_mp_std_c": 3.0,
                "hardener_mp_ad_score": 76.0,
            },
        ]
    )

    annotated = apply_melting_point_gate(candidates, mode="annotate", **_gate_kwargs())

    assert len(annotated) == len(candidates)
    assert annotated["candidate_id"].tolist() == candidates["candidate_id"].tolist()
    assert annotated["prediction"].tolist() == candidates["prediction"].tolist()
    assert annotated["resin_mp_filter_status"].tolist() == ["pass", "fail", "pass"]
    assert annotated["hardener_mp_filter_status"].tolist() == ["pass", "pass", "unknown"]
    assert annotated["mp_filter_reason"].tolist() == [
        "within_limits",
        "resin:prediction_plus_std_exceeds_limit",
        "hardener:non_finite_prediction",
    ]


def test_end_to_end_strict_gate_excludes_unknown_and_all_failed_candidates():
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "pass",
                "resin_mp_predicted_c": 118.0,
                "resin_mp_std_c": 4.0,
                "resin_mp_ad_score": 82.0,
                "hardener_mp_predicted_c": 124.0,
                "hardener_mp_std_c": 3.0,
                "hardener_mp_ad_score": 76.0,
            },
            {
                "candidate_id": "std-fail",
                "resin_mp_predicted_c": 118.0,
                "resin_mp_std_c": 11.0,
                "resin_mp_ad_score": 82.0,
                "hardener_mp_predicted_c": 124.0,
                "hardener_mp_std_c": 3.0,
                "hardener_mp_ad_score": 76.0,
            },
            {
                "candidate_id": "ad-fail",
                "resin_mp_predicted_c": 118.0,
                "resin_mp_std_c": 4.0,
                "resin_mp_ad_score": 49.0,
                "hardener_mp_predicted_c": 124.0,
                "hardener_mp_std_c": 3.0,
                "hardener_mp_ad_score": 76.0,
            },
            {
                "candidate_id": "unknown",
                "resin_mp_predicted_c": 118.0,
                "resin_mp_std_c": None,
                "resin_mp_ad_score": 82.0,
                "hardener_mp_predicted_c": 124.0,
                "hardener_mp_std_c": 3.0,
                "hardener_mp_ad_score": 76.0,
            },
        ]
    )

    strict = apply_melting_point_gate(candidates, mode="strict", **_gate_kwargs())

    assert strict["candidate_id"].tolist() == ["pass"]
    assert "mp_filter_reason" in strict.columns


def test_no_melting_point_columns_preserves_original_screening_frame():
    candidates = pd.DataFrame(
        [
            {"candidate_id": "a", "prediction": 180.0},
            {"candidate_id": "b", "prediction": 210.0},
        ]
    )

    annotated = apply_melting_point_gate(candidates, mode="annotate", **_gate_kwargs())
    strict = apply_melting_point_gate(candidates, mode="strict", **_gate_kwargs())

    pd.testing.assert_frame_equal(annotated, candidates)
    pd.testing.assert_frame_equal(strict, candidates)


def test_melting_point_artifact_contract_survives_screening_handoff():
    dataset = pd.DataFrame(
        [
            {
                "smiles": "CCO",
                "mp_c": 10.0,
                "mp_quality": "high",
                "component_role": "resin",
            },
            {
                "smiles": "CCN",
                "mp_c": 25.0,
                "mp_quality": "high",
                "component_role": "hardener",
            },
        ]
    )
    artifact = {
        "target_col": "mp_c",
        "extra": build_melting_point_artifact_extra(
            dataset,
            workflow_hash="workflow-integration",
        ),
    }

    validation = validate_melting_point_artifact(artifact)

    assert validation["ok"] is True
    assert validation["target_col"] == "mp_c"
    assert validation["dataset_row_count"] == 2
    assert validation["workflow_hash"] == "workflow-integration"
