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


class StopOrder:
    """止损/止盈订单"""
    def __init__(self, stop_id: str, symbol: str, direction: Direction,
                 stop_price: float, volume: int, order_type: str = "stop_loss",
                 trailing: bool = False, trailing_percent: float = 0.0):
        self.stop_id = stop_id
        self.symbol = symbol
        self.direction = direction  # 持仓方向
        self.stop_price = stop_price
        self.volume = volume
        self.order_type = order_type  # stop_loss, take_profit, trailing_stop
        self.trailing = trailing
        self.trailing_percent = trailing_percent
        self.highest_price = 0.0  # 追踪止损用
        self.lowest_price = float('inf')
        self.triggered = False
        self.create_time = datetime.now()
        self.strategy_name = ""


class OrderManager:
    """
    订单管理器

    功能:
    1. 信号转订单
    2. 订单发送
    3. 订单状态追踪
    4. 成交处理
    5. 自动止损/止盈
    6. 追踪止损
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

        # 止损/止盈订单
        self.stop_orders: Dict[str, StopOrder] = {}
        self._stop_order_count = 0

        # 当前价格缓存
        self.last_prices: Dict[str, float] = {}

        # 锁
        self._lock = threading.Lock()

        # 注册事件处理
        self.event_engine.register(EventType.ORDER, self._on_order_event)
        self.event_engine.register(EventType.TRADE, self._on_trade_event)
        self.event_engine.register(EventType.TICK, self._on_tick_event)
        self.event_engine.register(EventType.BAR, self._on_bar_event)

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
            'total_trades': len(self.trades),
            'active_stop_orders': len([s for s in self.stop_orders.values() if not s.triggered])
        }

    # ========== 止损/止盈功能 ==========

    def _generate_stop_id(self) -> str:
        """生成止损单ID"""
        with self._lock:
            self._stop_order_count += 1
            return f"STOP_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._stop_order_count:06d}"

    def set_stop_loss(self, symbol: str, direction: Direction, stop_price: float,
                      volume: int, strategy_name: str = "") -> str:
        """
        设置止损单

        Args:
            symbol: 品种代码
            direction: 持仓方向 (止损时反向平仓)
            stop_price: 止损价格
            volume: 止损数量
            strategy_name: 策略名称

        Returns:
            止损单ID
        """
        stop_id = self._generate_stop_id()

        stop_order = StopOrder(
            stop_id=stop_id,
            symbol=symbol,
            direction=direction,
            stop_price=stop_price,
            volume=volume,
            order_type="stop_loss"
        )
        stop_order.strategy_name = strategy_name

        with self._lock:
            self.stop_orders[stop_id] = stop_order

        logger.info(f"设置止损: {stop_id} {symbol} {direction.value} @ {stop_price} x{volume}")
        return stop_id

    def set_take_profit(self, symbol: str, direction: Direction, take_price: float,
                        volume: int, strategy_name: str = "") -> str:
        """
        设置止盈单

        Args:
            symbol: 品种代码
            direction: 持仓方向
            take_price: 止盈价格
            volume: 止盈数量
            strategy_name: 策略名称

        Returns:
            止盈单ID
        """
        stop_id = self._generate_stop_id()

        stop_order = StopOrder(
            stop_id=stop_id,
            symbol=symbol,
            direction=direction,
            stop_price=take_price,
            volume=volume,
            order_type="take_profit"
        )
        stop_order.strategy_name = strategy_name

        with self._lock:
            self.stop_orders[stop_id] = stop_order

        logger.info(f"设置止盈: {stop_id} {symbol} {direction.value} @ {take_price} x{volume}")
        return stop_id

    def set_trailing_stop(self, symbol: str, direction: Direction,
                          trailing_percent: float, volume: int,
                          initial_price: float = 0, strategy_name: str = "") -> str:
        """
        设置追踪止损

        Args:
            symbol: 品种代码
            direction: 持仓方向
            trailing_percent: 追踪百分比 (如 0.03 表示3%)
            volume: 止损数量
            initial_price: 初始价格（用于计算初始止损位）
            strategy_name: 策略名称

        Returns:
            追踪止损ID
        """
        stop_id = self._generate_stop_id()

        stop_order = StopOrder(
            stop_id=stop_id,
            symbol=symbol,
            direction=direction,
            stop_price=0,  # 追踪止损价格动态计算
            volume=volume,
            order_type="trailing_stop",
            trailing=True,
            trailing_percent=trailing_percent
        )
        stop_order.strategy_name = strategy_name

        if initial_price > 0:
            stop_order.highest_price = initial_price
            stop_order.lowest_price = initial_price
            # 计算初始止损价
            if direction == Direction.LONG:
                stop_order.stop_price = initial_price * (1 - trailing_percent)
            else:
                stop_order.stop_price = initial_price * (1 + trailing_percent)

        with self._lock:
            self.stop_orders[stop_id] = stop_order

        logger.info(f"设置追踪止损: {stop_id} {symbol} {direction.value} trailing={trailing_percent*100:.1f}%")
        return stop_id

    def cancel_stop_order(self, stop_id: str) -> bool:
        """取消止损单"""
        with self._lock:
            if stop_id in self.stop_orders:
                del self.stop_orders[stop_id]
                logger.info(f"取消止损单: {stop_id}")
                return True
        return False

    def cancel_all_stop_orders(self, symbol: str = None):
        """取消所有止损单"""
        with self._lock:
            if symbol:
                to_cancel = [sid for sid, s in self.stop_orders.items() if s.symbol == symbol]
            else:
                to_cancel = list(self.stop_orders.keys())

            for stop_id in to_cancel:
                del self.stop_orders[stop_id]

            logger.info(f"取消止损单: {len(to_cancel)}个")

    def update_stop_price(self, stop_id: str, new_price: float):
        """更新止损价格"""
        with self._lock:
            if stop_id in self.stop_orders:
                self.stop_orders[stop_id].stop_price = new_price
                logger.info(f"更新止损价: {stop_id} -> {new_price}")

    def get_stop_orders(self, symbol: str = None) -> List[StopOrder]:
        """获取止损单列表"""
        with self._lock:
            if symbol:
                return [s for s in self.stop_orders.values() if s.symbol == symbol and not s.triggered]
            return [s for s in self.stop_orders.values() if not s.triggered]

    def _on_tick_event(self, event: Event):
        """处理Tick事件，检查止损"""
        tick = event.data
        self.last_prices[tick.symbol] = tick.last_price
        self._check_stop_orders(tick.symbol, tick.last_price)

    def _on_bar_event(self, event: Event):
        """处理K线事件，检查止损"""
        bar = event.data
        self.last_prices[bar.symbol] = bar.close
        self._check_stop_orders(bar.symbol, bar.close)

    def _check_stop_orders(self, symbol: str, current_price: float):
        """检查止损单是否触发"""
        to_trigger = []

        with self._lock:
            for stop_id, stop_order in self.stop_orders.items():
                if stop_order.symbol != symbol or stop_order.triggered:
                    continue

                # 更新追踪止损
                if stop_order.trailing:
                    self._update_trailing_stop(stop_order, current_price)

                # 检查是否触发
                triggered = False

                if stop_order.order_type == "stop_loss":
                    # 止损: 多头跌破止损价，空头涨破止损价
                    if stop_order.direction == Direction.LONG:
                        if current_price <= stop_order.stop_price:
                            triggered = True
                    else:
                        if current_price >= stop_order.stop_price:
                            triggered = True

                elif stop_order.order_type == "take_profit":
                    # 止盈: 多头涨到止盈价，空头跌到止盈价
                    if stop_order.direction == Direction.LONG:
                        if current_price >= stop_order.stop_price:
                            triggered = True
                    else:
                        if current_price <= stop_order.stop_price:
                            triggered = True

                elif stop_order.order_type == "trailing_stop":
                    # 追踪止损
                    if stop_order.direction == Direction.LONG:
                        if current_price <= stop_order.stop_price:
                            triggered = True
                    else:
                        if current_price >= stop_order.stop_price:
                            triggered = True

                if triggered:
                    to_trigger.append((stop_id, stop_order, current_price))

        # 执行触发的止损单
        for stop_id, stop_order, trigger_price in to_trigger:
            self._execute_stop_order(stop_id, stop_order, trigger_price)

    def _update_trailing_stop(self, stop_order: StopOrder, current_price: float):
        """更新追踪止损价格"""
        if stop_order.direction == Direction.LONG:
            # 多头: 追踪最高价
            if current_price > stop_order.highest_price:
                stop_order.highest_price = current_price
                stop_order.stop_price = current_price * (1 - stop_order.trailing_percent)
        else:
            # 空头: 追踪最低价
            if current_price < stop_order.lowest_price:
                stop_order.lowest_price = current_price
                stop_order.stop_price = current_price * (1 + stop_order.trailing_percent)

    def _execute_stop_order(self, stop_id: str, stop_order: StopOrder, trigger_price: float):
        """执行止损单"""
        stop_order.triggered = True

        # 平仓方向与持仓方向相反
        close_direction = Direction.SHORT if stop_order.direction == Direction.LONG else Direction.LONG

        request = OrderRequest(
            symbol=stop_order.symbol,
            exchange="",
            direction=close_direction,
            offset=Offset.CLOSE,
            order_type=OrderType.MARKET,
            price=trigger_price,
            volume=stop_order.volume,
            strategy_name=f"{stop_order.strategy_name}_{stop_order.order_type}",
            signal_id=stop_id
        )

        order_id = self.send_order(request)

        logger.info(f"止损触发: {stop_id} {stop_order.symbol} {stop_order.order_type} @ {trigger_price} -> {order_id}")

        # 从止损单列表移除
        with self._lock:
            if stop_id in self.stop_orders:
                del self.stop_orders[stop_id]

    # ========== 内存清理 ==========

    def cleanup_history(self, keep_days: int = 7):
        """
        清理历史数据，防止内存无限增长

        Args:
            keep_days: 保留最近N天的数据
        """
        from datetime import timedelta
        cutoff_time = datetime.now() - timedelta(days=keep_days)

        with self._lock:
            # 清理成交记录
            old_trades_count = len(self.trades)
            self.trades = [t for t in self.trades if t.trade_time >= cutoff_time]
            trades_cleaned = old_trades_count - len(self.trades)

            # 清理已完成订单
            old_orders_count = len(self.orders)
            orders_to_remove = []
            for order_id, order in self.orders.items():
                if order_id not in self.active_orders:
                    if order.order_time and order.order_time < cutoff_time:
                        orders_to_remove.append(order_id)

            for order_id in orders_to_remove:
                del self.orders[order_id]
            orders_cleaned = len(orders_to_remove)

            # 清理信号映射
            old_signals_count = len(self.signal_orders)
            signals_to_remove = []
            for signal_id, order_ids in self.signal_orders.items():
                # 如果关联的订单都已被清理，则移除信号映射
                if all(oid not in self.orders for oid in order_ids):
                    signals_to_remove.append(signal_id)

            for signal_id in signals_to_remove:
                del self.signal_orders[signal_id]
            signals_cleaned = len(signals_to_remove)

        if trades_cleaned > 0 or orders_cleaned > 0 or signals_cleaned > 0:
            logger.info(f"内存清理: 成交-{trades_cleaned}, 订单-{orders_cleaned}, 信号映射-{signals_cleaned}")

    def daily_cleanup(self):
        """日终清理（建议每日收盘后调用）"""
        self.cleanup_history(keep_days=7)

        # 清理已触发的止损单
        with self._lock:
            triggered = [sid for sid, s in self.stop_orders.items() if s.triggered]
            for stop_id in triggered:
                del self.stop_orders[stop_id]

            if triggered:
                logger.info(f"清理已触发止损单: {len(triggered)}个")

    def get_memory_usage(self) -> dict:
        """获取内存使用统计"""
        with self._lock:
            return {
                'orders': len(self.orders),
                'active_orders': len(self.active_orders),
                'trades': len(self.trades),
                'signal_orders': len(self.signal_orders),
                'stop_orders': len(self.stop_orders)
            }
