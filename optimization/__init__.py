# coding=utf-8
"""
AI参数优化模块
提供基于Optuna的智能参数搜索功能
"""

import sys
import os

# 确保项目根目录在路径中（仅在此处设置一次）
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from .base import (
    OptimizationConfig,
    ParamSpace,
    OptimizationResult,
    BaseOptimizer
)
from .param_space import ParamSpaceManager
from .optuna_optimizer import OptunaOptimizer
from .config_applier import ConfigApplier

__all__ = [
    'OptimizationConfig',
    'ParamSpace',
    'OptimizationResult',
    'BaseOptimizer',
    'ParamSpaceManager',
    'OptunaOptimizer',
    'ConfigApplier',
]
