import pandas as pd

from core.prediction_molecular_baseline import (
    build_single_row_source_frame,
    collect_workflow_source_columns,
    validate_single_row_source_values,
    workflow_requires_manual_molecular_input,
)


def test_collect_source_columns_preserves_workflow_order_and_roles():
    workflow = {
        "steps": [
            {
                "step_id": "hardener_1",
                "order": 1,
                "role": "hardener",
                "source_columns": ["curing_agent_smiles_1"],
            },
            {
                "step_id": "resin_1",
                "order": 0,
                "role": "resin",
                "source_columns": ["resin_smiles_1", "resin_smiles_2"],
            },
            {
                "step_id": "resin_2",
                "order": 2,
                "role": "resin",
                "source_columns": ["resin_smiles_2", "resin_smiles_3"],
            },
        ],
        "merge_order": ["resin_1", "hardener_1", "resin_2"],
    }

    columns = collect_workflow_source_columns(workflow)

    assert columns == [
        {
            "column": "resin_smiles_1",
            "roles": ["resin"],
        },
        {
            "column": "resin_smiles_2",
            "roles": ["resin"],
        },
        {
            "column": "curing_agent_smiles_1",
            "roles": ["hardener"],
        },
        {
            "column": "resin_smiles_3",
            "roles": ["resin"],
        },
    ]


def test_build_single_row_source_frame_keeps_empty_values_as_empty_strings():
    frame = build_single_row_source_frame(
        [
            {"column": "resin_smiles_1", "roles": ["resin"]},
            {"column": "curing_agent_smiles_1", "roles": ["hardener"]},
        ],
        {
            "resin_smiles_1": "C1CO1",
            "curing_agent_smiles_1": None,
        },
    )

    assert isinstance(frame, pd.DataFrame)
    assert frame.to_dict(orient="records") == [
        {
            "resin_smiles_1": "C1CO1",
            "curing_agent_smiles_1": "",
        }
    ]


def test_validate_single_row_source_values_reports_missing_and_empty_sources():
    source_columns = [
        {"column": "resin_smiles_1", "roles": ["resin"]},
        {"column": "curing_agent_smiles_1", "roles": ["hardener"]},
    ]
    frame = build_single_row_source_frame(
        source_columns,
        {"resin_smiles_1": "C1CO1", "curing_agent_smiles_1": ""},
    )

    report = validate_single_row_source_values(frame, source_columns)

    assert report["ok"] is False
    assert report["missing_columns"] == []
    assert report["empty_columns"] == ["curing_agent_smiles_1"]


def test_workflow_with_declared_sources_requires_manual_molecular_input():
    workflow = {
        "steps": [
            {
                "step_id": "resin_1",
                "role": "resin",
                "source_columns": ["resin_smiles_1"],
            }
        ]
    }

    assert workflow_requires_manual_molecular_input(workflow) is True
    assert workflow_requires_manual_molecular_input({"steps": []}) is False
    assert workflow_requires_manual_molecular_input(None) is False
