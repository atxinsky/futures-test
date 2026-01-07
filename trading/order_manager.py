# coding=utf-8
"""
订单管理器
负责订单的创建、发送、跟踪和状态管理
"""

from typing import Dict, List, Optional, Callable
from datetime import datetime
from collections import defaultdict
import logging
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import (
    Signal, SignalAction, OrderRequest, Order, Trade,
    Direction, Offset, OrderType, OrderStatus, EventType, Event
)
from core.event_engine import EventEngine
from gateway.base_gateway import BaseGateway

logger = logging.getLogger(__name__)


class OrderManager:
    """
    订单管理器

    功能:
    1. 信号转订单
    2. 订单发送
    3. 订单状态追踪
    4. 成交处理
    """

    def __init__(self, event_engine: EventEngine, gateway: BaseGateway):
        self.event_engine = event_engine
        self.gateway = gateway

        # 订单存储
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}

        # 信号 -> 订单映射
        self.signal_orders: Dict[str, List[str]] = defaultdict(list)

        # 成交记录
        self.trades: List[Trade] = []

        # 锁
        self._lock = threading.Lock()

        # 注册事件处理
        self.event_engine.register(EventType.ORDER, self._on_order_event)
        self.event_engine.register(EventType.TRADE, self._on_trade_event)

    def send_order_from_signal(self, signal: Signal, symbol: str, exchange: str = "") -> Optional[str]:
        """
        根据信号发送订单

        Args:
            signal: 交易信号
            symbol: 合约代码
            exchange: 交易所

        Returns:
            订单ID
        """
        # 转换信号为订单请求
        request = self._signal_to_request(signal, symbol, exchange)
        if not request:
            logger.warning(f"无法将信号转换为订单: {signal}")
            return None

        # 发送订单
        order_id = self.gateway.send_order(request)

        if order_id:
            with self._lock:
                self.signal_orders[signal.signal_id].append(order_id)

        return order_id

    def send_order(self, request: OrderRequest) -> Optional[str]:
        """直接发送订单请求"""
        return self.gateway.send_order(request)

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        return self.gateway.cancel_order(order_id)

    def cancel_all(self, symbol: str = None):
        """撤销所有订单（可选按品种过滤）"""
        orders_to_cancel = []

        with self._lock:
            for order_id, order in self.active_orders.items():
                if symbol is None or order.symbol == symbol:
                    orders_to_cancel.append(order_id)

        for order_id in orders_to_cancel:
            self.cancel_order(order_id)

    def _signal_to_request(self, signal: Signal, symbol: str, exchange: str) -> Optional[OrderRequest]:
        """信号转订单请求"""
        # 确定方向和开平
        if signal.action == SignalAction.BUY:
            direction = Direction.LONG
            offset = Offset.OPEN
        elif signal.action == SignalAction.SELL:
            direction = Direction.SHORT
            offset = Offset.OPEN
        elif signal.action == SignalAction.CLOSE:
            # 平仓需要知道当前持仓方向
            # 默认平多
            direction = Direction.SHORT
            offset = Offset.CLOSE
        elif signal.action == SignalAction.CLOSE_LONG:
            direction = Direction.SHORT
            offset = Offset.CLOSE
        elif signal.action == SignalAction.CLOSE_SHORT:
            direction = Direction.LONG
            offset = Offset.CLOSE
        else:
            return None

        return OrderRequest(
            symbol=symbol,
            exchange=exchange,
            direction=direction,
            offset=offset,
            order_type=OrderType.LIMIT,
            price=signal.price,
            volume=signal.volume,
            strategy_name=signal.strategy_name,
            signal_id=signal.signal_id
        )

    def _on_order_event(self, event: Event):
        """处理订单事件"""
        order: Order = event.data

        with self._lock:
            self.orders[order.order_id] = order

            if order.is_active:
                self.active_orders[order.order_id] = order
            elif order.order_id in self.active_orders:
                del self.active_orders[order.order_id]

    def _on_trade_event(self, event: Event):
        """处理成交事件"""
        trade: Trade = event.data

        with self._lock:
            self.trades.append(trade)

    def get_order(self, order_id: str) -> Optional[Order]:
        """获取订单"""
        return self.orders.get(order_id)

    def get_active_orders(self, symbol: str = None) -> List[Order]:
        """获取活动订单"""
        with self._lock:
            if symbol:
                return [o for o in self.active_orders.values() if o.symbol == symbol]
            return list(self.active_orders.values())

    def get_orders_by_signal(self, signal_id: str) -> List[Order]:
        """获取信号关联的订单"""
        order_ids = self.signal_orders.get(signal_id, [])
        return [self.orders[oid] for oid in order_ids if oid in self.orders]

    def get_trades(self, symbol: str = None, start_time: datetime = None) -> List[Trade]:
        """获取成交记录"""
        result = self.trades

        if symbol:
            result = [t for t in result if t.symbol == symbol]

        if start_time:
            result = [t for t in result if t.trade_time >= start_time]

        return result

    def get_daily_trades(self) -> List[Trade]:
        """获取当日成交"""
        today = datetime.now().date()
        return [t for t in self.trades if t.trade_time.date() == today]

    def get_statistics(self) -> dict:
        """获取订单统计"""
        total = len(self.orders)
        active = len(self.active_orders)
        filled = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        cancelled = len([o for o in self.orders.values() if o.status == OrderStatus.CANCELLED])
        rejected = len([o for o in self.orders.values() if o.status == OrderStatus.REJECTED])

        return {
            'total_orders': total,
            'active_orders': active,
            'filled_orders': filled,
            'cancelled_orders': cancelled,
            'rejected_orders': rejected,
            'total_trades': len(self.trades)
        }
