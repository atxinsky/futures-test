# coding=utf-8
"""
持仓管理器
负责持仓的跟踪、更新和查询
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import logging
import threading

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import (
    Position, Trade, BarData, TickData,
    Direction, Offset, EventType, Event
)
from core.event_engine import EventEngine

logger = logging.getLogger(__name__)


class PositionManager:
    """
    持仓管理器

    功能:
    1. 持仓跟踪
    2. 盈亏计算
    3. 止损追踪
    4. 持仓统计
    """

    def __init__(self, event_engine: EventEngine):
        self.event_engine = event_engine

        # 持仓存储 {symbol: {direction: Position}}
        self.positions: Dict[str, Dict[Direction, Position]] = defaultdict(dict)

        # 当前价格
        self.last_prices: Dict[str, float] = {}

        # 品种配置
        self.instrument_configs: Dict[str, dict] = {}

        # 锁
        self._lock = threading.Lock()

        # 注册事件处理
        self.event_engine.register(EventType.POSITION, self._on_position_event)
        self.event_engine.register(EventType.TRADE, self._on_trade_event)
        self.event_engine.register(EventType.BAR, self._on_bar_event)
        self.event_engine.register(EventType.TICK, self._on_tick_event)

    def set_instrument_config(self, symbol: str, config: dict):
        """设置品种配置"""
        self.instrument_configs[symbol] = config

    def get_instrument_config(self, symbol: str) -> dict:
        """获取品种配置"""
        product = ''.join([c for c in symbol if c.isalpha()]).upper()
        return self.instrument_configs.get(product, self.instrument_configs.get(symbol, {
            'multiplier': 10,
            'margin_rate': 0.1
        }))

    def _on_position_event(self, event: Event):
        """处理持仓事件"""
        position: Position = event.data

        with self._lock:
            if position.volume > 0:
                self.positions[position.symbol][position.direction] = position
            elif position.symbol in self.positions and position.direction in self.positions[position.symbol]:
                del self.positions[position.symbol][position.direction]

    def _on_trade_event(self, event: Event):
        """处理成交事件"""
        trade: Trade = event.data
        self._update_on_trade(trade)

    def _on_bar_event(self, event: Event):
        """处理K线事件"""
        bar: BarData = event.data
        self.last_prices[bar.symbol] = bar.close
        self._update_pnl(bar.symbol, bar.close)

    def _on_tick_event(self, event: Event):
        """处理Tick事件"""
        tick: TickData = event.data
        self.last_prices[tick.symbol] = tick.last_price
        self._update_pnl(tick.symbol, tick.last_price)

    def _update_on_trade(self, trade: Trade):
        """根据成交更新持仓"""
        with self._lock:
            config = self.get_instrument_config(trade.symbol)
            multiplier = config.get('multiplier', 10)
            margin_rate = config.get('margin_rate', 0.1)

            if trade.offset == Offset.OPEN:
                # 开仓
                direction = trade.direction
                if direction not in self.positions[trade.symbol]:
                    self.positions[trade.symbol][direction] = Position(
                        symbol=trade.symbol,
                        exchange=trade.exchange,
                        direction=direction,
                        volume=0,
                        avg_price=0,
                        strategy_name=trade.strategy_name
                    )

                pos = self.positions[trade.symbol][direction]
                old_volume = pos.volume
                old_cost = pos.avg_price * old_volume

                new_cost = old_cost + trade.price * trade.volume
                pos.volume += trade.volume
                pos.avg_price = new_cost / pos.volume if pos.volume > 0 else 0

                if pos.highest_price == 0:
                    pos.highest_price = trade.price
                if pos.lowest_price == 0:
                    pos.lowest_price = trade.price

                pos.highest_price = max(pos.highest_price, trade.price)
                pos.lowest_price = min(pos.lowest_price, trade.price)

                if pos.entry_time is None:
                    pos.entry_time = trade.trade_time

                pos.margin = pos.avg_price * pos.volume * multiplier * margin_rate

            else:
                # 平仓
                close_direction = Direction.LONG if trade.direction == Direction.SHORT else Direction.SHORT

                if close_direction in self.positions[trade.symbol]:
                    pos = self.positions[trade.symbol][close_direction]

                    # 计算盈亏
                    if close_direction == Direction.LONG:
                        pnl = (trade.price - pos.avg_price) * trade.volume * multiplier
                    else:
                        pnl = (pos.avg_price - trade.price) * trade.volume * multiplier

                    pos.realized_pnl += pnl
                    pos.volume -= trade.volume

                    if pos.volume <= 0:
                        del self.positions[trade.symbol][close_direction]
                    else:
                        pos.margin = pos.avg_price * pos.volume * multiplier * margin_rate

    def _update_pnl(self, symbol: str, current_price: float):
        """更新浮动盈亏"""
        if symbol not in self.positions:
            return

        with self._lock:
            config = self.get_instrument_config(symbol)
            multiplier = config.get('multiplier', 10)

            for direction, pos in self.positions[symbol].items():
                if pos.volume <= 0:
                    continue

                if direction == Direction.LONG:
                    pos.unrealized_pnl = (current_price - pos.avg_price) * pos.volume * multiplier
                else:
                    pos.unrealized_pnl = (pos.avg_price - current_price) * pos.volume * multiplier

                # 更新极值
                pos.highest_price = max(pos.highest_price, current_price)
                pos.lowest_price = min(pos.lowest_price, current_price)

    def get_position(self, symbol: str, direction: Direction) -> Optional[Position]:
        """获取指定持仓"""
        with self._lock:
            if symbol in self.positions and direction in self.positions[symbol]:
                return self.positions[symbol][direction]
        return None

    def get_long_position(self, symbol: str) -> Optional[Position]:
        """获取多头持仓"""
        return self.get_position(symbol, Direction.LONG)

    def get_short_position(self, symbol: str) -> Optional[Position]:
        """获取空头持仓"""
        return self.get_position(symbol, Direction.SHORT)

    def get_net_position(self, symbol: str) -> int:
        """获取净持仓"""
        long_pos = self.get_long_position(symbol)
        short_pos = self.get_short_position(symbol)

        long_vol = long_pos.volume if long_pos else 0
        short_vol = short_pos.volume if short_pos else 0

        return long_vol - short_vol

    def get_all_positions(self) -> List[Position]:
        """获取所有持仓"""
        result = []
        with self._lock:
            for symbol_positions in self.positions.values():
                for pos in symbol_positions.values():
                    if pos.volume > 0:
                        result.append(pos)
        return result

    def get_positions_by_strategy(self, strategy_name: str) -> List[Position]:
        """获取策略持仓"""
        return [p for p in self.get_all_positions() if p.strategy_name == strategy_name]

    def get_total_pnl(self) -> Tuple[float, float]:
        """获取总盈亏 (浮动盈亏, 已实现盈亏)"""
        unrealized = 0.0
        realized = 0.0

        with self._lock:
            for symbol_positions in self.positions.values():
                for pos in symbol_positions.values():
                    unrealized += pos.unrealized_pnl
                    realized += pos.realized_pnl

        return unrealized, realized

    def get_total_margin(self) -> float:
        """获取总保证金"""
        total = 0.0
        with self._lock:
            for symbol_positions in self.positions.values():
                for pos in symbol_positions.values():
                    total += pos.margin
        return total

    def check_stop_loss(self, symbol: str, direction: Direction, stop_price: float) -> bool:
        """
        检查是否触发止损

        Args:
            symbol: 合约
            direction: 持仓方向
            stop_price: 止损价格

        Returns:
            是否触发
        """
        current_price = self.last_prices.get(symbol, 0)
        if current_price == 0:
            return False

        if direction == Direction.LONG:
            return current_price <= stop_price
        else:
            return current_price >= stop_price

    def check_trailing_stop(self, symbol: str, direction: Direction, atr_multiplier: float, atr: float) -> bool:
        """
        检查移动止损

        Args:
            symbol: 合约
            direction: 持仓方向
            atr_multiplier: ATR倍数
            atr: ATR值

        Returns:
            是否触发
        """
        pos = self.get_position(symbol, direction)
        if not pos:
            return False

        current_price = self.last_prices.get(symbol, 0)
        if current_price == 0:
            return False

        if direction == Direction.LONG:
            stop_price = pos.highest_price - atr * atr_multiplier
            return current_price <= stop_price
        else:
            stop_price = pos.lowest_price + atr * atr_multiplier
            return current_price >= stop_price

    def get_statistics(self) -> dict:
        """获取持仓统计"""
        positions = self.get_all_positions()
        unrealized, realized = self.get_total_pnl()
        total_margin = self.get_total_margin()

        long_count = len([p for p in positions if p.direction == Direction.LONG])
        short_count = len([p for p in positions if p.direction == Direction.SHORT])

        symbols = set(p.symbol for p in positions)

        return {
            'total_positions': len(positions),
            'long_positions': long_count,
            'short_positions': short_count,
            'symbols': len(symbols),
            'unrealized_pnl': unrealized,
            'realized_pnl': realized,
            'total_margin': total_margin
        }
