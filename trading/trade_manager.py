# coding=utf-8
"""
交易管理器
管理StrategyTrade的完整生命周期（从开仓到平仓）
参考trader-master设计，支持分批建仓/平仓
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
    StrategyTrade, Trade, Direction, Offset, TradeStatus,
    EventType, Event
)

# 延迟导入EventEngine，避免循环依赖
EventEngine = None
def _get_event_engine():
    global EventEngine
    if EventEngine is None:
        try:
            from core.event_engine import EventEngine as EE
            EventEngine = EE
        except ImportError:
            pass
    return EventEngine

logger = logging.getLogger(__name__)


class TradeManager:
    """
    交易管理器

    职责:
    1. 创建和管理StrategyTrade生命周期
    2. 处理成交事件，更新交易状态
    3. 计算盈亏和保证金
    4. 支持分批建仓/平仓
    5. 提供交易查询接口

    使用模式:
        # 1. 创建交易
        trade = manager.create_trade(strategy, symbol, direction, shares)

        # 2. 关联开仓订单
        manager.add_open_order(trade.trade_id, order_id)

        # 3. 收到成交时自动更新
        # (通过事件引擎自动处理)

        # 4. 关联平仓订单
        manager.add_close_order(trade.trade_id, order_id)

        # 5. 查询活跃交易
        trades = manager.get_active_trades(symbol)
    """

    def __init__(self, event_engine=None):
        self.event_engine = event_engine

        # 交易存储
        self.trades: Dict[str, StrategyTrade] = {}           # trade_id -> StrategyTrade
        self.active_trades: Dict[str, StrategyTrade] = {}    # 活跃交易
        self.closed_trades: List[StrategyTrade] = []         # 已平仓交易

        # 索引
        self.trades_by_symbol: Dict[str, List[str]] = defaultdict(list)     # symbol -> [trade_id]
        self.trades_by_strategy: Dict[str, List[str]] = defaultdict(list)   # strategy -> [trade_id]
        self.trades_by_order: Dict[str, str] = {}            # order_id -> trade_id

        # ID生成
        self._trade_counter: Dict[str, int] = defaultdict(int)  # strategy -> counter

        # 合约乘数缓存
        self._multipliers: Dict[str, float] = {}

        # 锁
        self._lock = threading.Lock()

        # 注册事件处理
        if self.event_engine:
            self.event_engine.register(EventType.TRADE, self._on_trade_event)

    def set_multiplier(self, symbol: str, multiplier: float) -> None:
        """设置合约乘数"""
        self._multipliers[symbol] = multiplier

    def get_multiplier(self, symbol: str) -> float:
        """获取合约乘数"""
        return self._multipliers.get(symbol, 1.0)

    def _generate_trade_id(self, strategy_name: str) -> str:
        """
        生成交易ID

        格式: {策略名}_{日期}_{序号}
        示例: brother2v6_20260108_000001
        """
        with self._lock:
            self._trade_counter[strategy_name] += 1
            count = self._trade_counter[strategy_name]
            date_str = datetime.now().strftime('%Y%m%d')
            return f"{strategy_name}_{date_str}_{count:06d}"

    def create_trade(
        self,
        strategy_name: str,
        symbol: str,
        direction: Direction,
        shares: int,
        exchange: str = "",
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        signal_id: str = "",
        entry_tag: str = ""
    ) -> StrategyTrade:
        """
        创建新交易

        Args:
            strategy_name: 策略名称
            symbol: 合约代码
            direction: 方向
            shares: 计划手数
            exchange: 交易所
            stop_loss: 止损价
            take_profit: 止盈价
            signal_id: 信号ID
            entry_tag: 入场标签

        Returns:
            新创建的StrategyTrade
        """
        trade_id = self._generate_trade_id(strategy_name)

        trade = StrategyTrade(
            trade_id=trade_id,
            strategy_name=strategy_name,
            symbol=symbol,
            exchange=exchange,
            direction=direction,
            shares=shares,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
            signal_id=signal_id,
            entry_tag=entry_tag,
            status=TradeStatus.PENDING
        )

        with self._lock:
            self.trades[trade_id] = trade
            self.active_trades[trade_id] = trade
            self.trades_by_symbol[symbol].append(trade_id)
            self.trades_by_strategy[strategy_name].append(trade_id)

        logger.info(f"创建交易: {trade_id} {symbol} {direction.value} x{shares}")
        return trade

    def add_open_order(self, trade_id: str, order_id: str) -> bool:
        """
        关联开仓订单

        Args:
            trade_id: 交易ID
            order_id: 订单ID

        Returns:
            是否成功
        """
        with self._lock:
            trade = self.trades.get(trade_id)
            if not trade:
                logger.warning(f"交易不存在: {trade_id}")
                return False

            trade.open_order_ids.append(order_id)
            self.trades_by_order[order_id] = trade_id

        logger.debug(f"关联开仓订单: {trade_id} <- {order_id}")
        return True

    def add_close_order(self, trade_id: str, order_id: str) -> bool:
        """
        关联平仓订单

        Args:
            trade_id: 交易ID
            order_id: 订单ID

        Returns:
            是否成功
        """
        with self._lock:
            trade = self.trades.get(trade_id)
            if not trade:
                logger.warning(f"交易不存在: {trade_id}")
                return False

            trade.close_order_ids.append(order_id)
            self.trades_by_order[order_id] = trade_id

        logger.debug(f"关联平仓订单: {trade_id} <- {order_id}")
        return True

    def _on_trade_event(self, event: Event) -> None:
        """处理成交事件"""
        fill: Trade = event.data
        self.process_fill(fill)

    def process_fill(self, fill: Trade) -> Optional[StrategyTrade]:
        """
        处理成交记录

        根据订单ID找到关联的StrategyTrade，更新状态

        Args:
            fill: 成交记录

        Returns:
            更新后的StrategyTrade（如果有）
        """
        with self._lock:
            # 通过订单ID找交易
            trade_id = self.trades_by_order.get(fill.order_id)
            if not trade_id:
                # 尝试通过品种和策略匹配活跃交易
                trade = self._match_trade_for_fill(fill)
                if not trade:
                    logger.debug(f"成交未关联交易: {fill.trade_id}")
                    return None
                trade_id = trade.trade_id

            trade = self.trades.get(trade_id)
            if not trade:
                return None

            # 获取合约乘数
            multiplier = self.get_multiplier(fill.symbol)

            # 添加成交并更新状态
            trade.add_fill(fill, multiplier)

            # 如果交易已关闭，移到closed列表
            if trade.status == TradeStatus.CLOSED:
                if trade_id in self.active_trades:
                    del self.active_trades[trade_id]
                self.closed_trades.append(trade)
                logger.info(f"交易平仓: {trade_id} 盈亏={trade.total_pnl:.2f}")
            else:
                logger.debug(f"交易更新: {trade_id} 状态={trade.status.value} "
                           f"持仓={trade.holding_shares}手")

            return trade

    def _match_trade_for_fill(self, fill: Trade) -> Optional[StrategyTrade]:
        """
        为未关联的成交匹配交易

        匹配规则:
        1. 同一品种
        2. 同一策略
        3. 活跃状态
        4. 方向匹配
        """
        # 根据策略名和品种查找
        strategy = fill.strategy_name
        symbol = fill.symbol

        for trade_id in self.trades_by_strategy.get(strategy, []):
            trade = self.trades.get(trade_id)
            if not trade or not trade.is_active:
                continue
            if trade.symbol != symbol:
                continue

            # 方向匹配
            if fill.offset == Offset.OPEN:
                # 开仓成交需要方向一致
                if fill.direction == Direction.LONG and trade.direction == Direction.LONG:
                    return trade
                if fill.direction == Direction.SHORT and trade.direction == Direction.SHORT:
                    return trade
            else:
                # 平仓成交需要方向相反
                if fill.direction == Direction.SHORT and trade.direction == Direction.LONG:
                    return trade
                if fill.direction == Direction.LONG and trade.direction == Direction.SHORT:
                    return trade

        return None

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        批量更新价格

        Args:
            prices: {symbol: current_price}
        """
        with self._lock:
            for trade in self.active_trades.values():
                if trade.symbol in prices:
                    multiplier = self.get_multiplier(trade.symbol)
                    trade.update_price(prices[trade.symbol], multiplier)

    def update_price(self, symbol: str, price: float) -> None:
        """
        更新单个品种价格

        Args:
            symbol: 合约代码
            price: 当前价格
        """
        with self._lock:
            trade_ids = self.trades_by_symbol.get(symbol, [])
            multiplier = self.get_multiplier(symbol)

            for trade_id in trade_ids:
                trade = self.active_trades.get(trade_id)
                if trade:
                    trade.update_price(price, multiplier)

    def get_trade(self, trade_id: str) -> Optional[StrategyTrade]:
        """获取交易"""
        return self.trades.get(trade_id)

    def get_active_trades(self, symbol: str = None, strategy: str = None) -> List[StrategyTrade]:
        """
        获取活跃交易

        Args:
            symbol: 按品种过滤（可选）
            strategy: 按策略过滤（可选）

        Returns:
            活跃交易列表
        """
        with self._lock:
            trades = list(self.active_trades.values())

            if symbol:
                trades = [t for t in trades if t.symbol == symbol]
            if strategy:
                trades = [t for t in trades if t.strategy_name == strategy]

            return trades

    def get_holding_trade(self, symbol: str, strategy: str = None) -> Optional[StrategyTrade]:
        """
        获取当前持仓交易（HOLDING状态）

        Args:
            symbol: 合约代码
            strategy: 策略名称（可选）

        Returns:
            持仓交易（如果有）
        """
        trades = self.get_active_trades(symbol, strategy)
        for trade in trades:
            if trade.status == TradeStatus.HOLDING and trade.holding_shares > 0:
                return trade
        return None

    def get_closed_trades(
        self,
        symbol: str = None,
        strategy: str = None,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> List[StrategyTrade]:
        """
        获取已平仓交易

        Args:
            symbol: 按品种过滤
            strategy: 按策略过滤
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            已平仓交易列表
        """
        with self._lock:
            trades = self.closed_trades.copy()

        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        if strategy:
            trades = [t for t in trades if t.strategy_name == strategy]
        if start_time:
            trades = [t for t in trades if t.close_time and t.close_time >= start_time]
        if end_time:
            trades = [t for t in trades if t.close_time and t.close_time <= end_time]

        return trades

    def cancel_trade(self, trade_id: str, reason: str = "") -> bool:
        """
        取消交易

        Args:
            trade_id: 交易ID
            reason: 取消原因

        Returns:
            是否成功
        """
        with self._lock:
            trade = self.trades.get(trade_id)
            if not trade:
                return False

            if trade.status == TradeStatus.CLOSED:
                logger.warning(f"交易已平仓，无法取消: {trade_id}")
                return False

            if trade.holding_shares > 0:
                logger.warning(f"交易有持仓，无法取消: {trade_id} ({trade.holding_shares}手)")
                return False

            trade.status = TradeStatus.CANCELLED
            if trade_id in self.active_trades:
                del self.active_trades[trade_id]

        logger.info(f"交易取消: {trade_id} 原因={reason}")
        return True

    def get_statistics(self) -> dict:
        """获取统计信息"""
        with self._lock:
            total_trades = len(self.trades)
            active_trades = len(self.active_trades)
            closed_trades = len(self.closed_trades)

            # 盈亏统计
            total_pnl = sum(t.total_pnl for t in self.closed_trades)
            winning_trades = [t for t in self.closed_trades if t.total_pnl > 0]
            losing_trades = [t for t in self.closed_trades if t.total_pnl <= 0]

            win_rate = len(winning_trades) / closed_trades if closed_trades > 0 else 0

            # 活跃交易盈亏
            active_unrealized = sum(t.unrealized_pnl for t in self.active_trades.values())

            return {
                'total_trades': total_trades,
                'active_trades': active_trades,
                'closed_trades': closed_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'total_realized_pnl': total_pnl,
                'active_unrealized_pnl': active_unrealized,
                'avg_win': sum(t.total_pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0,
                'avg_loss': sum(t.total_pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0,
            }

    def get_pnl_summary(self, strategy: str = None) -> dict:
        """
        获取盈亏汇总

        Args:
            strategy: 按策略过滤

        Returns:
            盈亏汇总
        """
        with self._lock:
            trades = self.closed_trades
            if strategy:
                trades = [t for t in trades if t.strategy_name == strategy]

            if not trades:
                return {
                    'total_pnl': 0,
                    'realized_pnl': 0,
                    'commission': 0,
                    'trade_count': 0
                }

            return {
                'total_pnl': sum(t.total_pnl for t in trades),
                'realized_pnl': sum(t.realized_pnl for t in trades),
                'commission': sum(t.commission for t in trades),
                'trade_count': len(trades)
            }

    def cleanup_old_trades(self, keep_days: int = 30) -> int:
        """
        清理旧交易记录

        Args:
            keep_days: 保留天数

        Returns:
            清理数量
        """
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=keep_days)

        with self._lock:
            old_count = len(self.closed_trades)
            self.closed_trades = [
                t for t in self.closed_trades
                if t.close_time and t.close_time >= cutoff
            ]
            cleaned = old_count - len(self.closed_trades)

        if cleaned > 0:
            logger.info(f"清理旧交易: {cleaned}笔")

        return cleaned

    def export_trades(self, filepath: str = None) -> List[dict]:
        """
        导出交易记录

        Args:
            filepath: 导出文件路径（可选）

        Returns:
            交易记录列表
        """
        with self._lock:
            records = [t.to_dict() for t in self.closed_trades]

        if filepath:
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            logger.info(f"导出交易记录: {filepath} ({len(records)}笔)")

        return records
