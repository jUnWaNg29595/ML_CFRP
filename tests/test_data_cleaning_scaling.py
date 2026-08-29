import ast
from pathlib import Path


def test_app_imports_all_scalers_at_module_scope():
    app_path = Path(__file__).resolve().parents[1] / 'app.py'
    tree = ast.parse(app_path.read_text(encoding='utf-8-sig'))

    imported_scalers = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module != 'sklearn.preprocessing':
            continue
        imported_scalers.update(
            alias.name
            for alias in node.names
            if alias.name in {'StandardScaler', 'MinMaxScaler', 'RobustScaler'}
        )

    assert imported_scalers == {
        'StandardScaler',
        'MinMaxScaler',
        'RobustScaler',
    }


def test_numeric_target_bin_balancing_keeps_sparse_bins_and_missing_rows():
    import numpy as np
    import pandas as pd

    from core.data_processor import AdvancedDataCleaner

    frame = pd.DataFrame({
        'target': [0.1] * 12 + [5.1, 6.0] + [9.9] + [np.nan, np.inf],
        'feature': range(17),
    })
    cleaner = AdvancedDataCleaner(frame)
    cleaned, stats = cleaner.balance_numeric_target_bins(
        'target', n_bins=2, max_samples_per_bin=2, random_state=7
    )

    assert len(cleaned) == 2 + 2 + 2
    assert stats['counts_before'] == [12, 3]
    assert stats['counts_after'] == [2, 2]
    assert stats['removed_rows'] == 11
    assert stats['missing_rows'] == 2
    assert cleaned['target'].isna().sum() == 1
    assert np.isinf(cleaned['target']).sum() == 1

    all_nonfinite = pd.DataFrame({'target': [np.nan, np.inf], 'feature': [1, 2]})
    cleaned, stats = AdvancedDataCleaner(all_nonfinite).balance_numeric_target_bins(
        'target', n_bins=2, max_samples_per_bin=1, keep_missing=False
    )
    assert cleaned.empty
    assert stats['removed_rows'] == 2

    constant_with_missing = pd.DataFrame({
        'target': [3.0, 3.0, np.nan, np.inf],
        'feature': range(4),
    })
    cleaned, stats = AdvancedDataCleaner(constant_with_missing).balance_numeric_target_bins(
        'target', n_bins=2, max_samples_per_bin=1, keep_missing=False
    )
    assert len(cleaned) == 2
    assert stats['counts_before'] == [2]
    assert stats['counts_after'] == [2]
    assert stats['removed_rows'] == 2


def test_numeric_target_bin_balancing_is_reproducible_and_validates_inputs():
    import pandas as pd
    import pytest

    from core.data_processor import AdvancedDataCleaner

    frame = pd.DataFrame({'target': [0.0, 1.0, 2.0, 8.0, 9.0, 10.0]})
    first, _ = AdvancedDataCleaner(frame).balance_numeric_target_bins(
        'target', n_bins=3, max_samples_per_bin=1, random_state=11
    )
    second, _ = AdvancedDataCleaner(frame).balance_numeric_target_bins(
        'target', n_bins=3, max_samples_per_bin=1, random_state=11
    )
    pd.testing.assert_frame_equal(first, second)

    with pytest.raises(ValueError, match='至少为 2'):
        AdvancedDataCleaner(frame).balance_numeric_target_bins('target', n_bins=1)
    with pytest.raises(ValueError, match='至少为 1'):
        AdvancedDataCleaner(frame).balance_numeric_target_bins(
            'target', n_bins=2, max_samples_per_bin=0
        )
