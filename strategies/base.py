# coding=utf-8
"""
策略基类
所有策略必须继承此类并实现相关方法
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import pandas as pd


@dataclass
class StrategyParam:
    """策略参数定义"""
    name: str           # 参数名
    label: str          # 显示名称
    default: Any        # 默认值
    min_val: Any = None # 最小值
    max_val: Any = None # 最大值
    step: Any = None    # 步长
    param_type: str = "float"  # 类型: int, float, bool, select
    options: List = None  # select类型的选项
    description: str = ""  # 参数说明


@dataclass
class Signal:
    """交易信号"""
    action: str         # "buy", "sell", "close"
    price: float
    volume: int = 1
    tag: str = ""       # 信号标签
    stop_loss: float = 0
    take_profit: float = 0


class BaseStrategy(ABC):
    """策略基类"""

    # 策略元信息
    name: str = "base"
    display_name: str = "基础策略"
    description: str = "策略基类，请继承实现"
    version: str = "1.0"
    author: str = ""

    # 预热K线数量
    warmup_num: int = 100

    def __init__(self, params: Dict = None):
        """
        初始化策略
        params: 策略参数字典
        """
        self.params = params or {}
        self._apply_default_params()
        self.position = 0  # 当前持仓 (1=多, -1=空, 0=空仓)
        self.entry_price = 0
        self.entry_time = None
        self.record_high = 0  # 持仓最高价
        self.record_low = float('inf')  # 持仓最低价

    def _apply_default_params(self):
        """应用默认参数"""
        for param in self.get_params():
            if param.name not in self.params:
                self.params[param.name] = param.default

    @classmethod
    @abstractmethod
    def get_params(cls) -> List[StrategyParam]:
        """
        获取策略参数定义
        子类必须实现
        """
        pass

    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        子类必须实现
        """
        pass

    @abstractmethod
    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """
        每根K线调用
        返回交易信号或None

        idx: 当前K线索引
        df: 包含指标的DataFrame
        capital: 当前可用资金
        """
        pass

    def on_trade(self, signal: Signal, is_entry: bool):
        """
        交易回调
        signal: 交易信号
        is_entry: True=开仓, False=平仓
        """
        pass

    def reset(self):
        """重置状态"""
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.record_high = 0
        self.record_low = float('inf')


def get_strategy_info(strategy_class) -> Dict:
    """获取策略信息"""
    return {
        'name': strategy_class.name,
        'display_name': strategy_class.display_name,
        'description': strategy_class.description,
        'version': strategy_class.version,
        'author': strategy_class.author,
        'params': [p.__dict__ for p in strategy_class.get_params()]
    }
