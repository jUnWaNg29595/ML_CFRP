import numpy as np
import pandas as pd

from core.model_trainer import EnhancedModelTrainer


def test_regression_training_preserves_feature_mask_when_imputer_drops_train_only_empty_column():
    feature_data = {
        f'feature_{index}': np.linspace(0.0, 1.0, 20)
        for index in range(41)
    }
    frame = pd.DataFrame(feature_data)
    frame.iloc[:, 0] = np.nan
    frame.iloc[0, 0] = 123.0
    frame.iloc[:, 1] = 0.0
    target = pd.Series(np.linspace(0.0, 19.0, 20), name='target')

    trainer = EnhancedModelTrainer(use_gpu=False)

    result = trainer.train_model(
        frame,
        target,
        model_name='线性回归',
        test_size=0.2,
        random_state=42,
        target_balance_enabled=False,
    )

    assert result['feature_mask'].shape == (41,)
    assert result['feature_mask'].dtype == bool
    assert result['pipeline'].predict(frame.iloc[[0]]).shape == (1,)
