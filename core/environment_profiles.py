# -*- coding: utf-8 -*-
"""环境配置文件 - 定义不同的环境预设"""

# 环境配置文件定义
ENVIRONMENT_PROFILES = {
    "torch2_full": {
        "name": "Torch 2.x 完整环境（推荐）",
        "description": "支持最新的深度学习模型和AutoML工具",
        "torch_version": ">=2.6.0",
        "supported_models": [
            "XGBoost", "LightGBM", "CatBoost",
            "TabNet", "人工神经网络", "Epoxy PINN",
            "TensorFlow Sequential", "AutoGluon",
            "TabPFN", "传统模型"
        ],
        "unsupported_models": ["FT-Transformer"],
        "install_commands": [
            "pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124",
            "pip install pytorch-tabnet",
            "pip install autogluon",
        ]
    },

    "torch1_fttransformer": {
        "name": "Torch 1.x FT-Transformer环境",
        "description": "专门用于FT-Transformer的环境",
        "torch_version": ">=1.13.0,<2.0.0",
        "supported_models": [
            "XGBoost", "LightGBM", "CatBoost",
            "TabNet", "FT-Transformer",
            "传统模型"
        ],
        "unsupported_models": ["人工神经网络", "Epoxy PINN", "AutoGluon"],
        "install_commands": [
            "pip install torch==1.13.1",
            "pip install rtdl",
            "pip install pytorch-tabnet",
        ]
    },

    "minimal": {
        "name": "最小环境（无深度学习）",
        "description": "只使用树模型和传统机器学习",
        "torch_version": None,
        "supported_models": [
            "XGBoost", "LightGBM", "CatBoost",
            "传统模型"
        ],
        "unsupported_models": ["所有深度学习模型"],
        "install_commands": [
            "pip install xgboost lightgbm catboost scikit-learn",
        ]
    }
}


def get_recommended_profile():
    """获取推荐的环境配置"""
    return "torch2_full"


def get_profile_for_model(model_name: str) -> str:
    """根据模型名称推荐环境配置"""
    model_to_profile = {
        "FT-Transformer": "torch1_fttransformer",
        "Epoxy PINN": "torch2_full",
        "人工神经网络": "torch2_full",
        "AutoGluon": "torch2_full",
        "TabNet": "torch2_full",  # TabNet在两个环境都可用，推荐torch2
        "XGBoost": "minimal",
        "LightGBM": "minimal",
        "CatBoost": "minimal",
    }
    return model_to_profile.get(model_name, "torch2_full")


def print_profile_comparison():
    """打印环境配置对比"""
    print("\n" + "=" * 80)
    print("环境配置对比")
    print("=" * 80)

    for profile_id, profile in ENVIRONMENT_PROFILES.items():
        print(f"\n【{profile['name']}】")
        print(f"  描述: {profile['description']}")
        print(f"  Torch版本: {profile['torch_version'] or '不需要'}")
        print(f"  支持模型: {', '.join(profile['supported_models'][:5])}...")
        print(f"  不支持: {', '.join(profile['unsupported_models'])}")

    print("\n" + "=" * 80)
    print("推荐配置: torch2_full（支持最多模型）")
    print("=" * 80)


if __name__ == "__main__":
    print_profile_comparison()
