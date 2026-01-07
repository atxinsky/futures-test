# coding=utf-8
"""
实盘交易引擎
协调所有组件，提供统一的交易接口
"""

from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, time
from threading import Thread
import logging
import time as time_module

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import (
    BarData, TickData, Signal, Order, Trade, Position, Account,
    EventType, Event, Direction, Offset
)
from core.event_engine import EventEngine
from gateway.base_gateway import BaseGateway
from gateway.sim_gateway import SimGateway
from trading.order_manager import OrderManager
from trading.position_manager import PositionManager
from trading.risk_manager import RiskManager, RiskConfig
from trading.account_manager import AccountManager
from strategies.base import BaseStrategy, create_strategy, get_registered_strategies

logger = logging.getLogger(__name__)


class LiveEngine:
    """
    实盘交易引擎

    功能:
    1. 组件协调
    2. 策略管理
    3. 数据分发
    4. 风险监控
    5. 状态管理
    """

    def __init__(self, config: dict = None):
        """
        初始化实盘引擎

        Args:
            config: 配置字典
        """
        self.config = config or {}

        # 状态
        self.is_running: bool = False
        self.start_time: Optional[datetime] = None

        # ============ 核心组件 ============
        self.event_engine = EventEngine()
        self.gateway: Optional[BaseGateway] = None
        self.order_manager: Optional[OrderManager] = None
        self.position_manager: Optional[PositionManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.account_manager: Optional[AccountManager] = None

        # ============ 策略管理 ============
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_symbols: Dict[str, List[str]] = {}  # strategy_id -> symbols

        # ============ 数据缓存 ============
        self.last_bars: Dict[str, BarData] = {}
        self.last_ticks: Dict[str, TickData] = {}

        # ============ 品种配置 ============
        self.instrument_configs: Dict[str, dict] = {}

        # ============ 回调 ============
        self._on_signal_callback: Optional[Callable] = None
        self._on_order_callback: Optional[Callable] = None
        self._on_trade_callback: Optional[Callable] = None
        self._on_position_callback: Optional[Callable] = None
        self._on_log_callback: Optional[Callable] = None

        # 注册事件处理
        self._register_handlers()

    def _register_handlers(self):
        """注册事件处理器"""
        self.event_engine.register(EventType.BAR, self._on_bar)
        self.event_engine.register(EventType.TICK, self._on_tick)
        self.event_engine.register(EventType.SIGNAL, self._on_signal)
        self.event_engine.register(EventType.ORDER, self._on_order)
        self.event_engine.register(EventType.TRADE, self._on_trade)
        self.event_engine.register(EventType.POSITION, self._on_position)
        self.event_engine.register(EventType.ACCOUNT, self._on_account)
        self.event_engine.register(EventType.LOG, self._on_log)

    def init_gateway(self, gateway_type: str = "sim", gateway_config: dict = None):
        """
        初始化网关

        Args:
            gateway_type: 网关类型 (sim/ctp/...)
            gateway_config: 网关配置
        """
        gateway_config = gateway_config or {}

        if gateway_type == "sim":
            self.gateway = SimGateway(self.event_engine, gateway_config)
        # elif gateway_type == "ctp":
        #     self.gateway = CtpGateway(self.event_engine, gateway_config)
        else:
            raise ValueError(f"未知网关类型: {gateway_type}")

        # 初始化交易组件
        self.order_manager = OrderManager(self.event_engine, self.gateway)
        self.position_manager = PositionManager(self.event_engine)
        self.account_manager = AccountManager(self.event_engine)

        # 风控配置
        risk_config = RiskConfig(**self.config.get('risk', {}))
        self.risk_manager = RiskManager(self.event_engine, risk_config)
        self.risk_manager.set_position_manager(self.position_manager)

        # 设置品种配置
        for symbol, cfg in self.instrument_configs.items():
            self.position_manager.set_instrument_config(symbol, cfg)
            if hasattr(self.gateway, 'set_instrument_config'):
                self.gateway.set_instrument_config(symbol, cfg)

        logger.info(f"网关初始化完成: {gateway_type}")

    def set_instrument_config(self, symbol: str, config: dict):
        """
        设置品种配置

        Args:
            symbol: 品种代码（如 IF, RB, AU）
            config: 配置字典 {multiplier, margin_rate, tick_size, ...}
        """
        self.instrument_configs[symbol] = config

        if self.position_manager:
            self.position_manager.set_instrument_config(symbol, config)
        if self.gateway and hasattr(self.gateway, 'set_instrument_config'):
            self.gateway.set_instrument_config(symbol, config)

    def add_strategy(self, strategy: BaseStrategy, symbols: List[str]) -> str:
        """
        添加策略

        Args:
            strategy: 策略实例
            symbols: 交易品种列表

        Returns:
            策略ID
        """
        strategy_id = strategy.strategy_id

        # 设置实盘模式
        strategy.set_live_mode(
            self.event_engine,
            self.order_manager,
            self.position_manager,
            self.risk_manager
        )

        # 设置信号回调
        strategy.set_signal_callback(self._handle_strategy_signal)

        self.strategies[strategy_id] = strategy
        self.strategy_symbols[strategy_id] = symbols

        # 订阅行情
        if self.gateway and self.is_running:
            for symbol in symbols:
                self.gateway.subscribe(symbol)

        logger.info(f"添加策略: {strategy.name} ({strategy_id}), 交易品种: {symbols}")
        return strategy_id

    def remove_strategy(self, strategy_id: str):
        """移除策略"""
        if strategy_id in self.strategies:
            strategy = self.strategies.pop(strategy_id)
            self.strategy_symbols.pop(strategy_id, None)
            logger.info(f"移除策略: {strategy.name} ({strategy_id})")

    def get_strategy(self, strategy_id: str) -> Optional[BaseStrategy]:
        """获取策略"""
        return self.strategies.get(strategy_id)

    def get_all_strategies(self) -> List[dict]:
        """获取所有策略信息"""
        return [s.get_info() for s in self.strategies.values()]

    def _handle_strategy_signal(self, signal, symbol: str, strategy_id: str):
        """处理策略信号"""
        # 发送信号事件
        self.event_engine.emit(EventType.SIGNAL, {
            'signal': signal,
            'symbol': symbol,
            'strategy_id': strategy_id
        })

    def _on_bar(self, event: Event):
        """处理K线事件"""
        bar: BarData = event.data
        self.last_bars[bar.symbol] = bar

        # 分发给订阅该品种的策略
        for strategy_id, symbols in self.strategy_symbols.items():
            if bar.symbol in symbols:
                strategy = self.strategies.get(strategy_id)
                if strategy:
                    bar_dict = {
                        'symbol': bar.symbol,
                        'datetime': bar.datetime,
                        'open': bar.open,
                        'high': bar.high,
                        'low': bar.low,
                        'close': bar.close,
                        'volume': bar.volume
                    }
                    strategy.on_bar_live(bar_dict)

    def _on_tick(self, event: Event):
        """处理Tick事件"""
        tick: TickData = event.data
        self.last_ticks[tick.symbol] = tick

    def _on_signal(self, event: Event):
        """处理信号事件"""
        data = event.data
        signal = data.get('signal')
        symbol = data.get('symbol')
        strategy_id = data.get('strategy_id')

        logger.info(f"收到信号: {signal.action} {symbol} @ {signal.price} [{strategy_id}]")

        if self._on_signal_callback:
            self._on_signal_callback(signal, symbol, strategy_id)

    def _on_order(self, event: Event):
        """处理订单事件"""
        order: Order = event.data
        logger.debug(f"订单更新: {order.order_id} {order.status}")

        if self._on_order_callback:
            self._on_order_callback(order)

    def _on_trade(self, event: Event):
        """处理成交事件"""
        trade: Trade = event.data
        logger.info(f"成交: {trade.symbol} {trade.direction.value} {trade.volume}@{trade.price}")

        # 记录交易结果（用于风控统计）
        # 简化处理：这里需要更复杂的盈亏计算逻辑
        if self._on_trade_callback:
            self._on_trade_callback(trade)

    def _on_position(self, event: Event):
        """处理持仓事件"""
        position: Position = event.data

        if self._on_position_callback:
            self._on_position_callback(position)

    def _on_account(self, event: Event):
        """处理账户事件"""
        account: Account = event.data

        # 更新风控账户引用
        if self.risk_manager:
            self.risk_manager.set_account(account)

    def _on_log(self, event: Event):
        """处理日志事件"""
        log_data = event.data
        if self._on_log_callback:
            self._on_log_callback(log_data)

    def start(self, initial_capital: float = 100000.0):
        """
        启动引擎

        Args:
            initial_capital: 初始资金
        """
        if self.is_running:
            logger.warning("引擎已在运行中")
            return

        if not self.gateway:
            raise RuntimeError("请先初始化网关")

        # 初始化账户
        self.account_manager.initialize(initial_capital)

        # 启动事件引擎
        self.event_engine.start()

        # 连接网关
        self.gateway.connect()

        # 订阅所有策略的品种
        subscribed = set()
        for symbols in self.strategy_symbols.values():
            for symbol in symbols:
                if symbol not in subscribed:
                    self.gateway.subscribe(symbol)
                    subscribed.add(symbol)

        self.is_running = True
        self.start_time = datetime.now()

        logger.info(f"实盘引擎启动, 初始资金: {initial_capital:.2f}")

    def stop(self):
        """停止引擎"""
        if not self.is_running:
            return

        # 断开网关
        if self.gateway:
            self.gateway.disconnect()

        # 停止事件引擎
        self.event_engine.stop()

        self.is_running = False
        logger.info("实盘引擎停止")

    def feed_bar(self, bar: BarData):
        """
        喂入K线数据（用于模拟盘）

        Args:
            bar: K线数据
        """
        if self.gateway and hasattr(self.gateway, 'feed_bar'):
            self.gateway.feed_bar(bar)

    def feed_tick(self, tick: TickData):
        """
        喂入Tick数据（用于模拟盘）

        Args:
            tick: Tick数据
        """
        if self.gateway and hasattr(self.gateway, 'feed_tick'):
            self.gateway.feed_tick(tick)

    # ============ 查询接口 ============

    def get_account(self) -> Optional[Account]:
        """获取账户信息"""
        if self.account_manager:
            return self.account_manager.get_account()
        return None

    def get_positions(self) -> List[Position]:
        """获取所有持仓"""
        if self.position_manager:
            return self.position_manager.get_all_positions()
        return []

    def get_orders(self, symbol: str = None) -> List[Order]:
        """获取订单"""
        if self.order_manager:
            return self.order_manager.get_active_orders(symbol)
        return []

    def get_trades(self, symbol: str = None) -> List[Trade]:
        """获取成交"""
        if self.order_manager:
            return self.order_manager.get_trades(symbol)
        return []

    def get_risk_status(self) -> dict:
        """获取风险状态"""
        if self.risk_manager:
            return self.risk_manager.get_risk_report()
        return {}

    def get_performance(self) -> dict:
        """获取绩效指标"""
        if self.account_manager:
            return self.account_manager.get_summary()
        return {}

    def get_statistics(self) -> dict:
        """获取综合统计"""
        result = {
            'engine': {
                'is_running': self.is_running,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'strategy_count': len(self.strategies),
                'subscribed_symbols': list(set(
                    s for symbols in self.strategy_symbols.values() for s in symbols
                ))
            }
        }

        if self.account_manager:
            result['account'] = self.account_manager.get_summary()

        if self.position_manager:
            result['position'] = self.position_manager.get_statistics()

        if self.order_manager:
            result['order'] = self.order_manager.get_statistics()

        if self.risk_manager:
            result['risk'] = self.risk_manager.get_risk_report()

        return result

    # ============ 回调设置 ============

    def set_signal_callback(self, callback: Callable):
        """设置信号回调"""
        self._on_signal_callback = callback

    def set_order_callback(self, callback: Callable):
        """设置订单回调"""
        self._on_order_callback = callback

    def set_trade_callback(self, callback: Callable):
        """设置成交回调"""
        self._on_trade_callback = callback

    def set_position_callback(self, callback: Callable):
        """设置持仓回调"""
        self._on_position_callback = callback

    def set_log_callback(self, callback: Callable):
        """设置日志回调"""
        self._on_log_callback = callback


# ============ 便捷函数 ============

def create_live_engine(config: dict = None) -> LiveEngine:
    """创建实盘引擎"""
    return LiveEngine(config)


def quick_start_sim(
    strategy_name: str,
    symbols: List[str],
    params: dict = None,
    initial_capital: float = 100000.0,
    instrument_configs: dict = None
) -> LiveEngine:
    """
    快速启动模拟盘

    Args:
        strategy_name: 策略名称
        symbols: 交易品种
        params: 策略参数
        initial_capital: 初始资金
        instrument_configs: 品种配置

    Returns:
        LiveEngine实例
    """
    engine = LiveEngine()

    # 设置品种配置
    if instrument_configs:
        for symbol, cfg in instrument_configs.items():
            engine.set_instrument_config(symbol, cfg)

    # 初始化模拟网关
    engine.init_gateway("sim")

    # 创建并添加策略
    strategy = create_strategy(strategy_name, params)
    if strategy:
        engine.add_strategy(strategy, symbols)
    else:
        raise ValueError(f"未找到策略: {strategy_name}")

    # 启动
    engine.start(initial_capital)

    return engine
