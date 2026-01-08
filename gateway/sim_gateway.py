# coding=utf-8
"""
模拟盘网关
用于策略验证，无需真实交易接口
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
import logging
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import (
    TickData, BarData, OrderRequest, Order, Trade, Position, Account,
    Direction, Offset, OrderStatus, OrderType, EventType, Event
)
from core.event_engine import EventEngine
from gateway.base_gateway import BaseGateway

logger = logging.getLogger(__name__)

# 延迟导入订单号生成器
_order_id_gen = None
def _get_order_id_generator():
    global _order_id_gen
    if _order_id_gen is None:
        try:
            from utils.order_id import get_order_id_generator
            _order_id_gen = get_order_id_generator()
        except ImportError:
            _order_id_gen = None
    return _order_id_gen


# 品种配置（简化版，实际应从config.py读取）
DEFAULT_INSTRUMENT_CONFIG = {
    'multiplier': 10,
    'margin_rate': 0.1,
    'commission_rate': 0.0001,
    'min_volume': 1,
    'price_tick': 1.0
}


class SimGateway(BaseGateway):
    """
    模拟盘网关

    功能:
    1. 模拟订单撮合
    2. 模拟持仓管理
    3. 模拟账户资金
    4. 支持滑点设置
    """

    def __init__(self, event_engine: EventEngine):
        super().__init__(event_engine, "SIM")

        # 配置
        self.initial_capital: float = 100000.0
        self.slippage: float = 0.0  # 滑点（价格单位）
        self.slippage_pct: float = 0.0  # 滑点（百分比）

        # 账户
        self.account = Account(account_id="SIM_ACCOUNT")

        # 持仓 {symbol: {direction: Position}}
        self.positions: Dict[str, Dict[Direction, Position]] = defaultdict(dict)

        # 订单
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}

        # 成交
        self.trades: List[Trade] = []

        # 当前价格
        self.last_prices: Dict[str, float] = {}

        # 品种配置
        self.instrument_configs: Dict[str, dict] = {}

        # 订单计数器
        self._order_count = 0
        self._trade_count = 0
        self._lock = threading.Lock()

    def connect(self, config: dict) -> bool:
        """连接模拟盘"""
        try:
            self.initial_capital = config.get('initial_capital', 100000.0)
            self.slippage = config.get('slippage', 0.0)
            self.slippage_pct = config.get('slippage_pct', 0.0)

            # 加载品种配置
            if 'instrument_configs' in config:
                self.instrument_configs = config['instrument_configs']

            # 初始化账户
            self.account.balance = self.initial_capital
            self.account.available = self.initial_capital
            self.account.pre_balance = self.initial_capital

            self.connected = True
            self.on_log("模拟盘已连接", "info")
            self.on_account(self.account)

            return True
        except Exception as e:
            self.on_error(f"模拟盘连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        self.connected = False
        self.on_log("模拟盘已断开", "info")

    def subscribe(self, symbols: List[str]):
        """订阅行情（模拟盘不需要真正订阅）"""
        self.subscribed_symbols.extend(symbols)
        self.on_log(f"订阅行情: {symbols}", "info")

    def set_instrument_config(self, symbol: str, config: dict):
        """设置品种配置"""
        self.instrument_configs[symbol] = config

    def get_instrument_config(self, symbol: str) -> dict:
        """获取品种配置"""
        # 从symbol提取品种代码
        product = ''.join([c for c in symbol if c.isalpha()]).upper()
        return self.instrument_configs.get(product, self.instrument_configs.get(symbol, DEFAULT_INSTRUMENT_CONFIG))

    def _generate_order_id(self, strategy_name: str = "", signal_id: str = "") -> str:
        """
        生成订单ID

        Args:
            strategy_name: 策略名称（用于生成有意义的订单号）
            signal_id: 关联的信号/交易ID

        Returns:
            订单号
        """
        gen = _get_order_id_generator()
        if gen and strategy_name:
            # 使用新的订单号生成器
            return gen.generate(strategy_name, signal_id)

        # 回退到原始方式
        with self._lock:
            self._order_count += 1
            return f"SIM_{datetime.now().strftime('%Y%m%d')}_{self._order_count:06d}"

    def _generate_trade_id(self) -> str:
        """生成成交ID"""
        with self._lock:
            self._trade_count += 1
            return f"TRD_{datetime.now().strftime('%Y%m%d')}_{self._trade_count:06d}"

    def send_order(self, request: OrderRequest) -> str:
        """发送订单"""
        if not self.connected:
            self.on_error("模拟盘未连接")
            return ""

        order_id = self._generate_order_id(request.strategy_name, request.signal_id)

        # 创建订单
        order = Order(
            order_id=order_id,
            symbol=request.symbol,
            exchange=request.exchange,
            direction=request.direction,
            offset=request.offset,
            order_type=request.order_type,
            price=request.price,
            volume=request.volume,
            status=OrderStatus.SUBMITTED,
            strategy_name=request.strategy_name,
            signal_id=request.signal_id,
            create_time=datetime.now()
        )

        self.orders[order_id] = order
        self.active_orders[order_id] = order

        self.on_order(order)
        self.on_log(f"订单已提交: {order_id} {request.direction.value} {request.offset.value} "
                   f"{request.symbol} {request.volume}@{request.price}", "info")

        # 立即尝试撮合（使用最新价格）
        if request.symbol in self.last_prices:
            self._try_match_order(order, self.last_prices[request.symbol])

        return order_id

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        if order_id not in self.active_orders:
            self.on_error(f"订单不存在或已完成: {order_id}")
            return False

        order = self.active_orders[order_id]
        order.status = OrderStatus.CANCELLED
        order.update_time = datetime.now()

        del self.active_orders[order_id]

        self.on_order(order)
        self.on_log(f"订单已撤销: {order_id}", "info")

        return True

    def on_bar(self, bar: BarData):
        """收到K线数据，尝试撮合"""
        self.last_prices[bar.symbol] = bar.close

        # 更新持仓盈亏
        self._update_position_pnl(bar.symbol, bar.close)

        # 尝试撮合挂单
        orders_to_match = [o for o in self.active_orders.values() if o.symbol == bar.symbol]
        for order in orders_to_match:
            self._try_match_order_with_bar(order, bar)

        # 发送K线事件
        super().on_bar(bar)

    def on_tick(self, tick: TickData):
        """收到Tick数据，尝试撮合"""
        self.last_prices[tick.symbol] = tick.last_price

        # 更新持仓盈亏
        self._update_position_pnl(tick.symbol, tick.last_price)

        # 尝试撮合挂单
        orders_to_match = [o for o in self.active_orders.values() if o.symbol == tick.symbol]
        for order in orders_to_match:
            self._try_match_order(order, tick.last_price)

        # 发送Tick事件
        super().on_tick(tick)

    def _try_match_order(self, order: Order, current_price: float):
        """尝试撮合订单（使用当前价格）"""
        if order.order_type == OrderType.MARKET:
            # 市价单立即成交
            fill_price = current_price
        else:
            # 限价单检查价格
            if order.direction == Direction.LONG:
                if current_price > order.price:
                    return  # 价格不满足
                fill_price = order.price
            else:
                if current_price < order.price:
                    return  # 价格不满足
                fill_price = order.price

        # 添加滑点
        fill_price = self._apply_slippage(fill_price, order.direction)

        # 执行成交
        self._execute_fill(order, fill_price, order.volume)

    def _try_match_order_with_bar(self, order: Order, bar: BarData):
        """使用K线数据撮合"""
        if order.order_type == OrderType.MARKET:
            # 市价单使用开盘价
            fill_price = bar.open
        else:
            # 限价单检查是否触及
            if order.direction == Direction.LONG:
                if bar.low <= order.price:
                    fill_price = order.price
                else:
                    return
            else:
                if bar.high >= order.price:
                    fill_price = order.price
                else:
                    return

        # 添加滑点
        fill_price = self._apply_slippage(fill_price, order.direction)

        # 执行成交
        self._execute_fill(order, fill_price, order.volume)

    def _apply_slippage(self, price: float, direction: Direction) -> float:
        """应用滑点"""
        slippage = self.slippage + price * self.slippage_pct

        if direction == Direction.LONG:
            return price + slippage
        else:
            return price - slippage

    def _execute_fill(self, order: Order, fill_price: float, fill_volume: int):
        """执行成交"""
        config = self.get_instrument_config(order.symbol)
        multiplier = config.get('multiplier', 10)
        margin_rate = config.get('margin_rate', 0.1)
        commission_rate = config.get('commission_rate', 0.0001)

        # 计算手续费
        commission = fill_price * multiplier * fill_volume * commission_rate

        # 创建成交记录
        trade = Trade(
            trade_id=self._generate_trade_id(),
            order_id=order.order_id,
            symbol=order.symbol,
            exchange=order.exchange,
            direction=order.direction,
            offset=order.offset,
            price=fill_price,
            volume=fill_volume,
            commission=commission,
            trade_time=datetime.now(),
            strategy_name=order.strategy_name
        )

        self.trades.append(trade)

        # 更新订单状态
        order.filled_volume += fill_volume
        order.avg_price = fill_price
        order.update_time = datetime.now()

        if order.filled_volume >= order.volume:
            order.status = OrderStatus.FILLED
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
        else:
            order.status = OrderStatus.PARTIAL

        # 更新持仓
        self._update_position(trade, config)

        # 更新账户
        self._update_account(trade, config)

        # 发送事件
        self.on_trade(trade)
        self.on_order(order)

        self.on_log(f"成交: {trade.trade_id} {order.direction.value} {order.offset.value} "
                   f"{order.symbol} {fill_volume}@{fill_price:.2f} 手续费:{commission:.2f}", "info")

    def _update_position(self, trade: Trade, config: dict):
        """更新持仓"""
        symbol = trade.symbol
        direction = trade.direction
        multiplier = config.get('multiplier', 10)
        margin_rate = config.get('margin_rate', 0.1)

        if trade.offset == Offset.OPEN:
            # 开仓
            key = direction
            if key not in self.positions[symbol]:
                self.positions[symbol][key] = Position(
                    symbol=symbol,
                    exchange=trade.exchange,
                    direction=direction,
                    volume=0,
                    avg_price=0,
                    strategy_name=trade.strategy_name
                )

            pos = self.positions[symbol][key]
            old_volume = pos.volume
            old_cost = pos.avg_price * old_volume

            new_cost = old_cost + trade.price * trade.volume
            pos.volume += trade.volume
            pos.avg_price = new_cost / pos.volume if pos.volume > 0 else 0

            pos.highest_price = max(pos.highest_price, trade.price)
            pos.lowest_price = min(pos.lowest_price, trade.price) if pos.lowest_price > 0 else trade.price
            pos.entry_time = datetime.now()

            # 计算保证金
            pos.margin = pos.avg_price * pos.volume * multiplier * margin_rate

            self.on_position(pos)

        else:
            # 平仓
            # 平仓方向与开仓相反
            close_direction = Direction.LONG if direction == Direction.SHORT else Direction.SHORT

            if close_direction in self.positions[symbol]:
                pos = self.positions[symbol][close_direction]

                # 计算盈亏
                if close_direction == Direction.LONG:
                    pnl = (trade.price - pos.avg_price) * trade.volume * multiplier
                else:
                    pnl = (pos.avg_price - trade.price) * trade.volume * multiplier

                pos.realized_pnl += pnl
                pos.volume -= trade.volume

                if pos.volume <= 0:
                    # 清空持仓
                    del self.positions[symbol][close_direction]
                else:
                    pos.margin = pos.avg_price * pos.volume * multiplier * margin_rate
                    self.on_position(pos)

    def _update_position_pnl(self, symbol: str, current_price: float):
        """更新持仓浮动盈亏"""
        if symbol not in self.positions:
            return

        for direction, pos in self.positions[symbol].items():
            if pos.volume <= 0:
                continue

            config = self.get_instrument_config(symbol)
            multiplier = config.get('multiplier', 10)

            if direction == Direction.LONG:
                pos.unrealized_pnl = (current_price - pos.avg_price) * pos.volume * multiplier
            else:
                pos.unrealized_pnl = (pos.avg_price - current_price) * pos.volume * multiplier

            pos.update_price(current_price)

    def _update_account(self, trade: Trade, config: dict):
        """更新账户"""
        multiplier = config.get('multiplier', 10)
        margin_rate = config.get('margin_rate', 0.1)

        # 扣除手续费
        self.account.commission += trade.commission
        self.account.available -= trade.commission

        if trade.offset == Offset.OPEN:
            # 开仓冻结保证金
            margin = trade.price * trade.volume * multiplier * margin_rate
            self.account.margin += margin
            self.account.available -= margin
        else:
            # 平仓释放保证金
            close_direction = Direction.LONG if trade.direction == Direction.SHORT else Direction.SHORT

            # 计算盈亏
            if trade.symbol in self.positions and close_direction in self.positions[trade.symbol]:
                pos = self.positions[trade.symbol][close_direction]
                if close_direction == Direction.LONG:
                    pnl = (trade.price - pos.avg_price) * trade.volume * multiplier
                else:
                    pnl = (pos.avg_price - trade.price) * trade.volume * multiplier

                self.account.realized_pnl += pnl
                self.account.available += pnl

            # 释放保证金
            margin = trade.price * trade.volume * multiplier * margin_rate
            self.account.margin -= margin
            self.account.available += margin

        # 计算总权益
        self._calculate_total_equity()
        self.account.update_time = datetime.now()

        self.on_account(self.account)

    def _calculate_total_equity(self):
        """计算总权益"""
        # 浮动盈亏
        unrealized_pnl = 0
        for symbol_positions in self.positions.values():
            for pos in symbol_positions.values():
                unrealized_pnl += pos.unrealized_pnl

        self.account.unrealized_pnl = unrealized_pnl
        self.account.balance = self.account.pre_balance + self.account.realized_pnl + unrealized_pnl - self.account.commission

        if self.account.balance > 0:
            self.account.risk_ratio = self.account.margin / self.account.balance
        else:
            self.account.risk_ratio = 1.0

    def query_account(self) -> Optional[Account]:
        """查询账户"""
        self._calculate_total_equity()
        return self.account

    def query_positions(self) -> List[Position]:
        """查询持仓"""
        result = []
        for symbol_positions in self.positions.values():
            for pos in symbol_positions.values():
                if pos.volume > 0:
                    result.append(pos)
        return result

    def query_orders(self) -> List[Order]:
        """查询订单"""
        return list(self.orders.values())

    def query_active_orders(self) -> List[Order]:
        """查询活动订单"""
        return list(self.active_orders.values())

    def get_position(self, symbol: str, direction: Direction) -> Optional[Position]:
        """获取指定持仓"""
        if symbol in self.positions and direction in self.positions[symbol]:
            return self.positions[symbol][direction]
        return None

    def reset(self):
        """重置模拟盘"""
        self.account = Account(account_id="SIM_ACCOUNT")
        self.account.balance = self.initial_capital
        self.account.available = self.initial_capital
        self.account.pre_balance = self.initial_capital

        self.positions.clear()
        self.orders.clear()
        self.active_orders.clear()
        self.trades.clear()
        self.last_prices.clear()

        self._order_count = 0
        self._trade_count = 0

        self.on_log("模拟盘已重置", "info")
