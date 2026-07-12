# -*- coding: utf-8 -*-
"""环境切换助手 - 帮助用户在不同环境配置间切换"""

import sys
import os

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.environment_manager import get_environment_manager
from core.environment_profiles import ENVIRONMENT_PROFILES, get_recommended_profile


def main():
    print("=" * 80)
    print("CFRP系统 - 环境切换助手")
    print("=" * 80)

    # 检查当前环境
    manager = get_environment_manager()
    print("\n【当前环境状态】")
    manager.print_environment_report()

    # 显示可用的环境配置
    print("\n【可用的环境配置】")
    for i, (profile_id, profile) in enumerate(ENVIRONMENT_PROFILES.items(), 1):
        print(f"\n{i}. {profile['name']}")
        print(f"   {profile['description']}")
        print(f"   支持模型: {', '.join(profile['supported_models'][:3])} 等")

    # 推荐配置
    print("\n" + "=" * 80)
    print("【推荐方案】")
    print("=" * 80)

    current_profile = manager.get_environment_profile()

    if current_profile == "torch2_profile":
        print("\n✓ 当前环境已是推荐配置（Torch 2.x）")
        print("\n支持的模型：")
        for model in manager.get_available_models():
            print(f"  • {model}")

        unavailable = manager.get_unavailable_models()
        if unavailable:
            print("\n不支持的模型：")
            for model, reason in unavailable.items():
                print(f"  • {model}: {reason}")

        print("\n如果需要使用FT-Transformer：")
        print("  1. 创建新的conda环境：")
        print("     conda create -n ft_transformer_env python=3.10")
        print("     conda activate ft_transformer_env")
        print("  2. 安装依赖：")
        print("     pip install torch==1.13.1 rtdl pytorch-tabnet scikit-learn pandas")
        print("  3. 在新环境中运行系统")
        print("\n  注意：新环境将不支持PINN、AutoGluon等模型")

    elif current_profile == "torch1_profile":
        print("\n当前环境：Torch 1.x")
        print("支持：FT-Transformer、TabNet、树模型")
        print("不支持：PINN、AutoGluon、人工神经网络")

        print("\n如果需要使用PINN/AutoGluon：")
        print("  1. 切换回主环境：")
        print("     conda activate CFRP_env")
        print("  2. 恢复Torch 2.x：")
        print("     pip uninstall rtdl torch -y")
        print("     pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124")

    else:
        print("\n当前环境：最小配置")
        print("建议安装Torch 2.x以使用更多模型")

    print("\n" + "=" * 80)
    print("【使用建议】")
    print("=" * 80)
    print("\n方案A：单环境策略（推荐）")
    print("  • 保持当前Torch 2.x环境")
    print("  • 使用TabNet替代FT-Transformer（性能相当）")
    print("  • 优势：支持最多模型，无需切换")
    print("  • 劣势：无法使用FT-Transformer")

    print("\n方案B：双环境策略")
    print("  • 主环境（CFRP_env）：Torch 2.x，用于日常工作")
    print("  • 辅助环境（ft_transformer_env）：Torch 1.x，专门用于FT-Transformer")
    print("  • 优势：可以使用所有模型")
    print("  • 劣势：需要手动切换环境")

    print("\n方案C：容器化部署（高级）")
    print("  • 使用Docker为不同模型创建独立容器")
    print("  • 通过API调用不同容器中的模型")
    print("  • 优势：完全隔离，无冲突")
    print("  • 劣势：配置复杂")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
