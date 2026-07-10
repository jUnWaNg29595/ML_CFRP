# -*- coding: utf-8 -*-
"""环境管理器 - 动态检测和管理模型依赖"""

import sys
import subprocess
import importlib.util
from typing import Dict, List, Tuple, Optional


class EnvironmentManager:
    """环境管理器：检测依赖、管理模型可用性"""

    def __init__(self):
        self.torch_version = None
        self.available_models = {}
        self._check_environment()

    def _check_environment(self):
        """检查当前环境"""
        # 检查torch版本
        try:
            import torch
            self.torch_version = torch.__version__
        except ImportError:
            self.torch_version = None

        # 检查各个模型的可用性
        self._check_model_availability()

    def _check_model_availability(self):
        """检查各个模型的可用性"""
        checks = {
            'TabNet': self._check_tabnet,
            'FT-Transformer': self._check_fttransformer,
            'PINN': self._check_pinn,
            'ANN': self._check_ann,
            'XGBoost': self._check_xgboost,
            'LightGBM': self._check_lightgbm,
            'AutoGluon': self._check_autogluon,
        }

        for model_name, check_func in checks.items():
            available, reason = check_func()
            self.available_models[model_name] = {
                'available': available,
                'reason': reason
            }

    def _check_tabnet(self) -> Tuple[bool, str]:
        """检查TabNet"""
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
            if self.torch_version and self.torch_version.startswith('1.'):
                return True, ""
            elif self.torch_version and self.torch_version.startswith('2.'):
                return True, ""
            return True, ""
        except ImportError:
            return False, "未安装pytorch-tabnet"

    def _check_fttransformer(self) -> Tuple[bool, str]:
        """检查FT-Transformer"""
        try:
            import rtdl
            if self.torch_version and self.torch_version.startswith('2.'):
                return False, "rtdl需要torch<2.0，当前torch版本为" + self.torch_version
            return True, ""
        except ImportError:
            return False, "未安装rtdl"

    def _check_pinn(self) -> Tuple[bool, str]:
        """检查PINN"""
        try:
            import torch
            if self.torch_version:
                major_version = int(self.torch_version.split('.')[0])
                if major_version >= 2:
                    return True, ""
                return False, f"PINN需要torch>=2.0，当前版本为{self.torch_version}"
            return False, "未安装torch"
        except Exception as e:
            return False, str(e)

    def _check_ann(self) -> Tuple[bool, str]:
        """检查ANN"""
        try:
            import torch
            return True, ""
        except ImportError:
            return False, "未安装torch"

    def _check_xgboost(self) -> Tuple[bool, str]:
        """检查XGBoost"""
        try:
            import xgboost
            return True, ""
        except ImportError:
            return False, "未安装xgboost"

    def _check_lightgbm(self) -> Tuple[bool, str]:
        """检查LightGBM"""
        try:
            import lightgbm
            return True, ""
        except ImportError:
            return False, "未安装lightgbm"

    def _check_autogluon(self) -> Tuple[bool, str]:
        """检查AutoGluon"""
        try:
            from autogluon.tabular import TabularPredictor
            if self.torch_version and self.torch_version.startswith('1.'):
                return False, f"AutoGluon需要torch>=2.6，当前版本为{self.torch_version}"
            return True, ""
        except ImportError:
            return False, "未安装autogluon"

    def get_available_models(self) -> List[str]:
        """获取可用的模型列表"""
        return [name for name, info in self.available_models.items() if info['available']]

    def get_unavailable_models(self) -> Dict[str, str]:
        """获取不可用的模型及原因"""
        return {name: info['reason'] for name, info in self.available_models.items() if not info['available']}

    def is_model_available(self, model_name: str) -> bool:
        """检查指定模型是否可用"""
        return self.available_models.get(model_name, {}).get('available', False)

    def get_environment_profile(self) -> str:
        """获取当前环境配置文件"""
        if self.torch_version:
            if self.torch_version.startswith('2.'):
                return "torch2_profile"  # 支持PINN、ANN、TabNet、AutoGluon
            elif self.torch_version.startswith('1.'):
                return "torch1_profile"  # 支持FT-Transformer、TabNet
        return "minimal_profile"  # 只支持树模型

    def suggest_environment_switch(self, target_model: str) -> Optional[str]:
        """建议切换环境以使用目标模型"""
        if self.is_model_available(target_model):
            return None  # 当前环境已支持

        suggestions = {
            'FT-Transformer': "需要torch<2.0环境。建议：创建独立环境或使用TabNet替代",
            'PINN': "需要torch>=2.0环境。当前环境不支持",
            'AutoGluon': "需要torch>=2.6环境。当前环境不支持",
            'TabNet': "需要安装pytorch-tabnet: pip install pytorch-tabnet",
        }

        return suggestions.get(target_model, "未知模型")

    def print_environment_report(self):
        """打印环境报告"""
        print("=" * 60)
        print("环境配置报告")
        print("=" * 60)
        print(f"Torch版本: {self.torch_version or '未安装'}")
        print(f"环境配置: {self.get_environment_profile()}")
        print(f"\n可用模型 ({len(self.get_available_models())}个):")
        for model in self.get_available_models():
            print(f"  ✓ {model}")

        unavailable = self.get_unavailable_models()
        if unavailable:
            print(f"\n不可用模型 ({len(unavailable)}个):")
            for model, reason in unavailable.items():
                print(f"  ✗ {model}: {reason}")
        print("=" * 60)


# 全局环境管理器实例
_env_manager = None

def get_environment_manager() -> EnvironmentManager:
    """获取环境管理器单例"""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager


if __name__ == "__main__":
    # 测试
    manager = get_environment_manager()
    manager.print_environment_report()
