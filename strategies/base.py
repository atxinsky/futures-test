# coding=utf-8
"""
策略基类
支持回测和实盘两种模式
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import pandas as pd
import logging
import uuid

logger = logging.getLogger(__name__)


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
    """交易信号（回测模式）"""
    action: str         # "buy", "sell", "close", "close_long", "close_short"
    price: float
    volume: int = 1
    tag: str = ""       # 信号标签
    stop_loss: float = 0
    take_profit: float = 0


class BaseStrategy(ABC):
    """
    策略基类

    支持两种运行模式:
    1. 回测模式: 通过on_bar方法逐K线调用
    2. 实盘模式: 通过on_bar_live方法接收实时数据
    """

    # ============ 策略元信息 ============
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

        Args:
            params: 策略参数字典
        """
        self.params = params or {}
        self._apply_default_params()

        # ============ 回测状态 ============
        self.position = 0  # 当前持仓 (1=多, -1=空, 0=空仓)
        self.entry_price = 0
        self.entry_time = None
        self.record_high = 0  # 持仓最高价
        self.record_low = float('inf')  # 持仓最低价

        # ============ 实盘状态 ============
        self.is_live: bool = False
        self.strategy_id: str = f"{self.name}_{uuid.uuid4().hex[:8]}"

        # 实盘组件引用（由LiveEngine注入）
        self._event_engine = None
        self._order_manager = None
        self._position_manager = None
        self._risk_manager = None

        # 当前交易的合约
        self.current_symbol: str = ""
        self.current_exchange: str = ""

        # K线数据缓存（实盘模式）
        self._bar_cache: Dict[str, pd.DataFrame] = {}
        self._max_cache_size: int = 500

        # 信号回调
        self._signal_callback: Optional[Callable] = None

    def _apply_default_params(self):
        """应用默认参数"""
        for param in self.get_params():
            if param.name not in self.params:
                self.params[param.name] = param.default

    # ============ 抽象方法（必须实现）============

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

        Args:
            df: 原始OHLCV数据

        Returns:
            添加了指标列的DataFrame
        """
        pass

    @abstractmethod
    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """
        回测模式：每根K线调用

        Args:
            idx: 当前K线索引
            df: 包含指标的DataFrame
            capital: 当前可用资金

        Returns:
            交易信号或None
        """
        pass

    # ============ 可选方法 ============

    def on_trade(self, signal: Signal, is_entry: bool):
        """
        交易回调

        Args:
            signal: 交易信号
            is_entry: True=开仓, False=平仓
        """
        pass

    def on_order_filled(self, order_data: dict):
        """
        实盘模式：订单成交回调

        Args:
            order_data: 订单数据
        """
        pass

    def on_position_update(self, position_data: dict):
        """
        实盘模式：持仓更新回调

        Args:
            position_data: 持仓数据
        """
        pass

    def reset(self):
        """重置状态"""
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self.record_high = 0
        self.record_low = float('inf')
        self._bar_cache.clear()

    # ============ 实盘模式支持 ============

    def set_live_mode(self, event_engine, order_manager, position_manager, risk_manager=None):
        """
        设置实盘模式

        Args:
            event_engine: 事件引擎
            order_manager: 订单管理器
            position_manager: 持仓管理器
            risk_manager: 风控管理器（可选）
        """
        self.is_live = True
        self._event_engine = event_engine
        self._order_manager = order_manager
        self._position_manager = position_manager
        self._risk_manager = risk_manager

        logger.info(f"策略 {self.name} 进入实盘模式")

    def set_symbol(self, symbol: str, exchange: str = ""):
        """设置交易合约"""
        self.current_symbol = symbol
        self.current_exchange = exchange

    def set_signal_callback(self, callback: Callable):
        """设置信号回调函数"""
        self._signal_callback = callback

    def on_bar_live(self, bar_data: dict):
        """
        实盘模式：收到K线数据

        Args:
            bar_data: K线数据字典
                - symbol: 合约代码
                - datetime: 时间
                - open, high, low, close, volume
        """
        if not self.is_live:
            return

        symbol = bar_data.get('symbol', self.current_symbol)

        # 更新K线缓存
        self._update_bar_cache(symbol, bar_data)

        # 获取完整数据
        df = self._bar_cache.get(symbol)
        if df is None or len(df) < self.warmup_num:
            return

        # 计算指标
        df = self.calculate_indicators(df)

        # 生成信号
        idx = len(df) - 1
        capital = self._get_available_capital()

        signal = self.on_bar(idx, df, capital)

        if signal:
            self._process_live_signal(signal, symbol)

    def _update_bar_cache(self, symbol: str, bar_data: dict):
        """更新K线缓存"""
        if symbol not in self._bar_cache:
            self._bar_cache[symbol] = pd.DataFrame()

        df = self._bar_cache[symbol]

        new_row = pd.DataFrame([{
            'time': bar_data.get('datetime', datetime.now()),
            'open': bar_data.get('open', 0),
            'high': bar_data.get('high', 0),
            'low': bar_data.get('low', 0),
            'close': bar_data.get('close', 0),
            'volume': bar_data.get('volume', 0)
        }])

        self._bar_cache[symbol] = pd.concat([df, new_row], ignore_index=True)

        # 限制缓存大小
        if len(self._bar_cache[symbol]) > self._max_cache_size:
            self._bar_cache[symbol] = self._bar_cache[symbol].tail(self._max_cache_size).reset_index(drop=True)

    def _get_available_capital(self) -> float:
        """获取可用资金"""
        if self._order_manager and hasattr(self._order_manager, 'gateway'):
            account = self._order_manager.gateway.query_account()
            if account:
                return account.available
        return 100000.0  # 默认值

    def _process_live_signal(self, signal: Signal, symbol: str):
        """处理实盘信号"""
        logger.info(f"策略 {self.name} 产生信号: {signal.action} {symbol} @ {signal.price} [{signal.tag}]")

        # 调用信号回调
        if self._signal_callback:
            self._signal_callback(signal, symbol, self.strategy_id)

        # 如果有订单管理器，发送订单
        if self._order_manager:
            from models.base import Signal as ModelSignal, SignalAction

            # 转换信号格式
            action_map = {
                'buy': SignalAction.BUY,
                'sell': SignalAction.SELL,
                'close': SignalAction.CLOSE,
                'close_long': SignalAction.CLOSE_LONG,
                'close_short': SignalAction.CLOSE_SHORT
            }

            model_signal = ModelSignal(
                signal_id=f"{self.strategy_id}_{datetime.now().strftime('%H%M%S')}",
                strategy_name=self.name,
                symbol=symbol,
                action=action_map.get(signal.action, SignalAction.BUY),
                price=signal.price,
                volume=signal.volume,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                tag=signal.tag
            )

            # 发送订单
            self._order_manager.send_order_from_signal(
                model_signal,
                symbol,
                self.current_exchange
            )

    def get_position_for_symbol(self, symbol: str) -> int:
        """获取指定合约的持仓方向"""
        if self._position_manager:
            net = self._position_manager.get_net_position(symbol)
            if net > 0:
                return 1
            elif net < 0:
                return -1
        return self.position

    def load_history_bars(self, symbol: str, bars: pd.DataFrame):
        """
        加载历史K线（用于实盘初始化）

        Args:
            symbol: 合约代码
            bars: 历史K线DataFrame
        """
        self._bar_cache[symbol] = bars.tail(self._max_cache_size).reset_index(drop=True)
        logger.info(f"策略 {self.name} 加载 {symbol} 历史K线 {len(self._bar_cache[symbol])} 条")

    # ============ 工具方法 ============

    def get_param(self, name: str, default: Any = None) -> Any:
        """获取参数值"""
        return self.params.get(name, default)

    def set_param(self, name: str, value: Any):
        """设置参数值"""
        self.params[name] = value

    def get_info(self) -> Dict:
        """获取策略信息"""
        return {
            'strategy_id': self.strategy_id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'is_live': self.is_live,
            'current_symbol': self.current_symbol,
            'position': self.position,
            'params': self.params
        }


def get_strategy_info(strategy_class) -> Dict:
    """获取策略类信息"""
    return {
        'name': strategy_class.name,
        'display_name': strategy_class.display_name,
        'description': strategy_class.description,
        'version': strategy_class.version,
        'author': strategy_class.author,
        'warmup_num': strategy_class.warmup_num,
        'params': [p.__dict__ for p in strategy_class.get_params()]
    }


# ============ 策略注册表 ============

_strategy_registry: Dict[str, type] = {}


def register_strategy(strategy_class: type):
    """注册策略"""
    if hasattr(strategy_class, 'name'):
        _strategy_registry[strategy_class.name] = strategy_class
        logger.debug(f"注册策略: {strategy_class.name}")


def get_registered_strategies() -> Dict[str, type]:
    """获取已注册的策略"""
    return _strategy_registry.copy()


def get_strategy_class(name: str) -> Optional[type]:
    """根据名称获取策略类"""
    return _strategy_registry.get(name)


def create_strategy(name: str, params: Dict = None) -> Optional[BaseStrategy]:
    """创建策略实例"""
    cls = get_strategy_class(name)
    if cls:
        return cls(params)
    return None
