# coding=utf-8
"""
交易网关基类
定义所有网关的统一接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Callable
from datetime import datetime
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import (
    TickData, BarData, OrderRequest, Order, Trade,
    Position, Account, EventType, Event
)
from core.event_engine import EventEngine

logger = logging.getLogger(__name__)


class BaseGateway(ABC):
    """
    交易网关基类

    所有具体网关（模拟盘、CTP、其他交易所）都需要继承此类
    """

    def __init__(self, event_engine: EventEngine, gateway_name: str):
        self.event_engine = event_engine
        self.gateway_name = gateway_name

        # 订阅的合约
        self.subscribed_symbols: List[str] = []

        # 状态
        self.connected: bool = False

    @abstractmethod
    def connect(self, config: dict) -> bool:
        """
        连接网关

        Args:
            config: 连接配置
                - 模拟盘: initial_capital, slippage等
                - CTP: address, broker_id, user_id, password等

        Returns:
            是否连接成功
        """
        pass

    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass

    @abstractmethod
    def subscribe(self, symbols: List[str]):
        """
        订阅行情

        Args:
            symbols: 合约代码列表，如 ['AU2506', 'RB2505']
        """
        pass

    @abstractmethod
    def send_order(self, request: OrderRequest) -> str:
        """
        发送订单

        Args:
            request: 订单请求

        Returns:
            订单ID
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        撤销订单

        Args:
            order_id: 订单ID

        Returns:
            是否成功
        """
        pass

    @abstractmethod
    def query_account(self) -> Optional[Account]:
        """查询账户"""
        pass

    @abstractmethod
    def query_positions(self) -> List[Position]:
        """查询持仓"""
        pass

    @abstractmethod
    def query_orders(self) -> List[Order]:
        """查询订单"""
        pass

    # ========== 事件发送方法 ==========

    def on_tick(self, tick: TickData):
        """收到Tick数据"""
        event = Event(type=EventType.TICK, data=tick)
        self.event_engine.put(event)

    def on_bar(self, bar: BarData):
        """收到K线数据"""
        event = Event(type=EventType.BAR, data=bar)
        self.event_engine.put(event)

    def on_order(self, order: Order):
        """订单状态更新"""
        event = Event(type=EventType.ORDER, data=order)
        self.event_engine.put(event)

    def on_trade(self, trade: Trade):
        """成交回报"""
        event = Event(type=EventType.TRADE, data=trade)
        self.event_engine.put(event)

    def on_position(self, position: Position):
        """持仓更新"""
        event = Event(type=EventType.POSITION, data=position)
        self.event_engine.put(event)

    def on_account(self, account: Account):
        """账户更新"""
        event = Event(type=EventType.ACCOUNT, data=account)
        self.event_engine.put(event)

    def on_log(self, msg: str, level: str = "info"):
        """日志事件"""
        event = Event(type=EventType.LOG, data={'msg': msg, 'level': level, 'gateway': self.gateway_name})
        self.event_engine.put(event)

    def on_error(self, msg: str, error_id: str = ""):
        """错误事件"""
        event = Event(type=EventType.ERROR, data={'msg': msg, 'error_id': error_id, 'gateway': self.gateway_name})
        self.event_engine.put(event)
