# coding=utf-8
"""
策略管理模块
自动发现和加载策略
"""

import os
import importlib
import importlib.util
from typing import Dict, List, Type
from .base import BaseStrategy, StrategyParam, Signal, get_strategy_info


# 已注册的策略
_strategies: Dict[str, Type[BaseStrategy]] = {}


def register_strategy(strategy_class: Type[BaseStrategy]):
    """注册策略"""
    _strategies[strategy_class.name] = strategy_class
    return strategy_class


def get_strategy(name: str) -> Type[BaseStrategy]:
    """获取策略类"""
    return _strategies.get(name)


def get_all_strategies() -> Dict[str, Type[BaseStrategy]]:
    """获取所有已注册策略"""
    return _strategies.copy()


def list_strategies() -> List[Dict]:
    """列出所有策略信息"""
    return [get_strategy_info(s) for s in _strategies.values()]


def load_strategy_from_file(filepath: str) -> Type[BaseStrategy]:
    """
    从外部文件加载策略

    filepath: 策略文件路径 (.py文件)
    返回: 策略类
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"策略文件不存在: {filepath}")

    # 动态导入模块
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 查找策略类
    strategy_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if (isinstance(obj, type) and
            issubclass(obj, BaseStrategy) and
            obj is not BaseStrategy):
            strategy_class = obj
            break

    if strategy_class is None:
        raise ValueError(f"文件中未找到有效策略类: {filepath}")

    # 注册策略
    register_strategy(strategy_class)
    return strategy_class


def discover_strategies():
    """
    自动发现并加载内置策略
    扫描strategies目录下的所有.py文件
    """
    strategies_dir = os.path.dirname(__file__)

    for filename in os.listdir(strategies_dir):
        if filename.startswith('_') or not filename.endswith('.py'):
            continue
        if filename == 'base.py':
            continue

        module_name = filename[:-3]
        try:
            module = importlib.import_module(f'.{module_name}', package='strategies')

            # 查找并注册策略类
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and
                    issubclass(obj, BaseStrategy) and
                    obj is not BaseStrategy and
                    obj.name not in _strategies):
                    register_strategy(obj)
        except Exception as e:
            print(f"加载策略 {module_name} 失败: {e}")


# 自动发现策略
discover_strategies()


# 导出
__all__ = [
    'BaseStrategy',
    'StrategyParam',
    'Signal',
    'get_strategy_info',
    'register_strategy',
    'get_strategy',
    'get_all_strategies',
    'list_strategies',
    'load_strategy_from_file',
    'discover_strategies',
]
