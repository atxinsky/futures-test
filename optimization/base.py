# coding=utf-8
"""
优化器基类与抽象接口
定义统一的优化API，支持多种优化算法
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import pandas as pd


@dataclass
class OptimizationConfig:
    """优化配置"""
    strategy_name: str                    # 策略名称
    symbols: List[str]                    # 品种列表
    train_start: str                      # 训练集开始
    train_end: str                        # 训练集结束
    val_start: str                        # 验证集开始
    val_end: str                          # 验证集结束
    timeframe: str = "1d"                 # 时间周期: 1m/5m/15m/30m/1h/4h/1d
    n_trials: int = 50                    # 优化轮数
    objective: str = "sharpe"             # 优化目标: sharpe/calmar/return/sortino
    initial_capital: float = 100000       # 初始资金
    min_trades: int = 5                   # 最小交易次数
    max_drawdown: float = 0.40            # 最大回撤限制
    per_symbol: bool = False              # 是否每品种独立优化


@dataclass
class ParamSpace:
    """参数空间定义"""
    name: str                             # 参数名
    low: float                            # 下界
    high: float                           # 上界
    step: Optional[float] = None          # 步长
    param_type: str = "float"             # 类型: int/float
    log_scale: bool = False               # 是否对数尺度
    default: Any = None                   # 默认值
    label: str = ""                       # 显示名称
    description: str = ""                 # 参数说明


@dataclass
class OptimizationResult:
    """优化结果"""
    strategy_name: str
    symbol: Optional[str]                 # None表示多品种综合优化
    best_params: Dict[str, Any]
    best_value: float
    train_metrics: Dict[str, float]       # 训练集指标
    val_metrics: Dict[str, float]         # 验证集指标
    param_importance: Dict[str, float]    # 参数重要性
    optimization_history: pd.DataFrame    # 优化历史
    created_at: datetime = field(default_factory=datetime.now)
    config: Optional[OptimizationConfig] = None


class BaseOptimizer(ABC):
    """
    优化器基类

    职责：
    1. 定义统一的优化接口
    2. 管理优化配置
    3. 协调数据加载、回测、评估
    4. 提供回调机制（进度、日志）
    """

    def __init__(self, config: OptimizationConfig):
        self.config = config
        self._progress_callback: Optional[Callable] = None
        self._log_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable[[float, str], None]):
        """设置进度回调 (progress: 0-1, message: str)"""
        self._progress_callback = callback

    def set_log_callback(self, callback: Callable[[str], None]):
        """设置日志回调"""
        self._log_callback = callback

    def _log(self, msg: str):
        """内部日志"""
        if self._log_callback:
            self._log_callback(msg)

    def _update_progress(self, progress: float, msg: str):
        """更新进度"""
        if self._progress_callback:
            self._progress_callback(progress, msg)

    @abstractmethod
    def optimize(self, param_spaces: Dict[str, ParamSpace]) -> OptimizationResult:
        """
        执行优化

        Args:
            param_spaces: 参数空间定义字典 {param_name: ParamSpace}

        Returns:
            OptimizationResult: 优化结果
        """
        pass

    @abstractmethod
    def optimize_per_symbol(self, param_spaces: Dict[str, ParamSpace]) -> Dict[str, OptimizationResult]:
        """
        每品种独立优化

        Returns:
            Dict[symbol, OptimizationResult]: 每个品种的优化结果
        """
        pass
