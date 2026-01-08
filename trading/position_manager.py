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
from utils.state_persistence import get_state_persistence, StatePersistence

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

    def __init__(self, event_engine: EventEngine, enable_persistence: bool = True):
        """
        初始化持仓管理器

        Args:
            event_engine: 事件引擎
            enable_persistence: 是否启用持久化（实盘建议开启）
        """
        self.event_engine = event_engine

        # 持仓存储 {symbol: {direction: Position}}
        self.positions: Dict[str, Dict[Direction, Position]] = defaultdict(dict)

        # 当前价格
        self.last_prices: Dict[str, float] = {}

        # 品种配置
        self.instrument_configs: Dict[str, dict] = {}

        # 锁
        self._lock = threading.Lock()

        # 持久化支持
        self._enable_persistence = enable_persistence
        self._persistence: Optional[StatePersistence] = None
        if enable_persistence:
            self._persistence = get_state_persistence()

        # 注册事件处理
        self.event_engine.register(EventType.POSITION, self._on_position_event)
        self.event_engine.register(EventType.TRADE, self._on_trade_event)
        self.event_engine.register(EventType.BAR, self._on_bar_event)
        self.event_engine.register(EventType.TICK, self._on_tick_event)

    def set_instrument_config(self, symbol: str, config: dict):
        """设置品种配置（线程安全）"""
        with self._lock:
            self.instrument_configs[symbol] = config

    def get_instrument_config(self, symbol: str) -> dict:
        """获取品种配置（线程安全）"""
        with self._lock:
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
        with self._lock:
            self.last_prices[bar.symbol] = bar.close
        self._update_pnl(bar.symbol, bar.close)

    def _on_tick_event(self, event: Event):
        """处理Tick事件"""
        tick: TickData = event.data
        with self._lock:
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
                        # 持久化：删除已平仓的持仓记录
                        if self._persistence:
                            self._persistence.delete_position(
                                trade.symbol,
                                str(close_direction.value) if hasattr(close_direction, 'value') else str(close_direction)
                            )
                    else:
                        pos.margin = pos.avg_price * pos.volume * multiplier * margin_rate

            # 持久化：保存成交记录和持仓状态
            if self._persistence:
                # 保存成交
                self._persistence.save_trade({
                    'trade_id': trade.trade_id,
                    'order_id': trade.order_id,
                    'symbol': trade.symbol,
                    'exchange': str(trade.exchange) if trade.exchange else '',
                    'direction': str(trade.direction.value) if hasattr(trade.direction, 'value') else str(trade.direction),
                    'offset': str(trade.offset.value) if hasattr(trade.offset, 'value') else str(trade.offset),
                    'price': trade.price,
                    'volume': trade.volume,
                    'trade_time': trade.trade_time.isoformat() if trade.trade_time else '',
                    'strategy_name': trade.strategy_name
                })

                # 保存当前持仓状态
                self._save_position_to_db(trade.symbol)

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
        """获取净持仓（线程安全）"""
        with self._lock:
            long_vol = 0
            short_vol = 0

            if symbol in self.positions:
                if Direction.LONG in self.positions[symbol]:
                    long_vol = self.positions[symbol][Direction.LONG].volume
                if Direction.SHORT in self.positions[symbol]:
                    short_vol = self.positions[symbol][Direction.SHORT].volume

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
        检查是否触发止损（线程安全）

        Args:
            symbol: 合约
            direction: 持仓方向
            stop_price: 止损价格

        Returns:
            是否触发
        """
        with self._lock:
            current_price = self.last_prices.get(symbol, 0)

        if current_price == 0:
            return False

        if direction == Direction.LONG:
            return current_price <= stop_price
        else:
            return current_price >= stop_price

    def check_trailing_stop(self, symbol: str, direction: Direction, atr_multiplier: float, atr: float) -> bool:
        """
        检查移动止损（线程安全）

        Args:
            symbol: 合约
            direction: 持仓方向
            atr_multiplier: ATR倍数
            atr: ATR值

        Returns:
            是否触发
        """
        with self._lock:
            pos = None
            if symbol in self.positions and direction in self.positions[symbol]:
                pos = self.positions[symbol][direction]
            current_price = self.last_prices.get(symbol, 0)

        if not pos:
            return False

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

    # ============ 持久化方法 ============

    def _save_position_to_db(self, symbol: str):
        """保存指定品种的持仓到数据库"""
        if not self._persistence:
            return

        if symbol not in self.positions:
            return

        for direction, pos in self.positions[symbol].items():
            if pos.volume > 0:
                self._persistence.save_position({
                    'symbol': pos.symbol,
                    'exchange': str(pos.exchange) if pos.exchange else '',
                    'direction': str(direction.value) if hasattr(direction, 'value') else str(direction),
                    'volume': pos.volume,
                    'avg_price': pos.avg_price,
                    'margin': pos.margin,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl,
                    'highest_price': pos.highest_price,
                    'lowest_price': pos.lowest_price,
                    'entry_time': pos.entry_time.isoformat() if pos.entry_time else '',
                    'strategy_name': pos.strategy_name
                })

    def save_all_to_db(self):
        """保存所有持仓到数据库"""
        if not self._persistence:
            logger.warning("持久化未启用，无法保存")
            return

        positions_data = []
        with self._lock:
            for symbol, directions in self.positions.items():
                for direction, pos in directions.items():
                    if pos.volume > 0:
                        positions_data.append({
                            'symbol': pos.symbol,
                            'exchange': str(pos.exchange) if pos.exchange else '',
                            'direction': str(direction.value) if hasattr(direction, 'value') else str(direction),
                            'volume': pos.volume,
                            'avg_price': pos.avg_price,
                            'margin': pos.margin,
                            'unrealized_pnl': pos.unrealized_pnl,
                            'realized_pnl': pos.realized_pnl,
                            'highest_price': pos.highest_price,
                            'lowest_price': pos.lowest_price,
                            'entry_time': pos.entry_time.isoformat() if pos.entry_time else '',
                            'strategy_name': pos.strategy_name
                        })

        self._persistence.save_all_positions(positions_data)
        logger.info(f"保存 {len(positions_data)} 个持仓到数据库")

    def load_from_db(self) -> int:
        """
        从数据库加载持仓（程序启动时调用）

        Returns:
            加载的持仓数量
        """
        if not self._persistence:
            logger.warning("持久化未启用，无法加载")
            return 0

        positions_data = self._persistence.load_positions()

        with self._lock:
            for pos_data in positions_data:
                symbol = pos_data['symbol']
                direction_str = pos_data['direction']

                # 解析方向
                if direction_str in ('LONG', '1', 'Direction.LONG'):
                    direction = Direction.LONG
                elif direction_str in ('SHORT', '-1', 'Direction.SHORT'):
                    direction = Direction.SHORT
                else:
                    logger.warning(f"未知持仓方向: {direction_str}")
                    continue

                # 解析入场时间
                entry_time = None
                if pos_data.get('entry_time'):
                    try:
                        entry_time = datetime.fromisoformat(pos_data['entry_time'])
                    except:
                        pass

                # 创建持仓对象
                pos = Position(
                    symbol=symbol,
                    exchange=pos_data.get('exchange', ''),
                    direction=direction,
                    volume=pos_data['volume'],
                    avg_price=pos_data['avg_price'],
                    margin=pos_data.get('margin', 0),
                    unrealized_pnl=pos_data.get('unrealized_pnl', 0),
                    realized_pnl=pos_data.get('realized_pnl', 0),
                    highest_price=pos_data.get('highest_price', 0),
                    lowest_price=pos_data.get('lowest_price', 0),
                    entry_time=entry_time,
                    strategy_name=pos_data.get('strategy_name', '')
                )

                self.positions[symbol][direction] = pos

        logger.info(f"从数据库加载 {len(positions_data)} 个持仓")
        return len(positions_data)

    def clear_db(self):
        """清空数据库中的持仓记录"""
        if self._persistence:
            self._persistence.save_all_positions([])
            logger.info("已清空数据库持仓记录")
