import pandas as pd

import core.industrial_filter as industrial_filter


def test_industrial_filter_does_not_apply_hidden_2000_cap(monkeypatch):
    candidates = [f"candidate_{index}" for index in range(2501)]
    scored = pd.DataFrame(
        {
            "smiles": candidates,
            "avg_similarity": [float(index) for index in range(2501)],
        }
    )

    monkeypatch.setattr(
        industrial_filter,
        "stage1_industrial_filter",
        lambda *args, **kwargs: (
            candidates,
            {"total": len(candidates), "passed": len(candidates)},
            {},
        ),
    )
    monkeypatch.setattr(
        industrial_filter,
        "compute_similarity_scores",
        lambda *args, **kwargs: scored.copy(),
    )

    passed, stats, score_df = industrial_filter.pipeline_industrial_filter(
        candidates,
        {"known_smiles_1", "known_smiles_2", "known_smiles_3"},
    )

    assert len(passed) == 2501
    assert stats["stage2_kept"] == 2501
    assert len(score_df) == 2501
