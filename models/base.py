# coding=utf-8
"""
基础数据模型定义
定义交易系统中使用的所有数据结构
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any


class Direction(Enum):
    """方向"""
    LONG = "long"
    SHORT = "short"


class Offset(Enum):
    """开平"""
    OPEN = "open"
    CLOSE = "close"
    CLOSE_TODAY = "close_today"
    CLOSE_YESTERDAY = "close_yesterday"


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TradeStatus(Enum):
    """交易状态（完整生命周期）"""
    PENDING = "pending"          # 等待开仓
    OPENING = "opening"          # 开仓中（部分成交）
    HOLDING = "holding"          # 持仓中
    CLOSING = "closing"          # 平仓中（部分成交）
    CLOSED = "closed"            # 已平仓
    CANCELLED = "cancelled"      # 已取消


class OrderType(Enum):
    """订单类型"""
    LIMIT = "limit"
    MARKET = "market"
    STOP = "stop"
    FAK = "fak"
    FOK = "fok"


class SignalAction(Enum):
    """信号动作"""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    ROLL_CLOSE = "roll_close"    # 换月平旧
    ROLL_OPEN = "roll_open"      # 换月开新


@dataclass
class TickData:
    """Tick数据"""
    symbol: str
    exchange: str
    datetime: datetime

    last_price: float = 0.0
    volume: int = 0
    turnover: float = 0.0
    open_interest: float = 0.0

    bid_price_1: float = 0.0
    bid_volume_1: int = 0
    ask_price_1: float = 0.0
    ask_volume_1: int = 0

    high_price: float = 0.0
    low_price: float = 0.0
    open_price: float = 0.0
    pre_close: float = 0.0

    upper_limit: float = 0.0
    lower_limit: float = 0.0


@dataclass
class BarData:
    """K线数据"""
    symbol: str
    exchange: str
    datetime: datetime
    interval: str  # "1m", "5m", "15m", "60m", "1d"

    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume: float = 0.0
    turnover: float = 0.0
    open_interest: float = 0.0


@dataclass
class Signal:
    """交易信号"""
    signal_id: str = ""
    strategy_name: str = ""
    symbol: str = ""

    action: SignalAction = SignalAction.BUY
    price: float = 0.0
    volume: int = 1

    stop_loss: float = 0.0
    take_profit: float = 0.0
    tag: str = ""

    create_time: datetime = field(default_factory=datetime.now)
    trigger_time: Optional[datetime] = None
    processed: bool = False

    def to_dict(self) -> dict:
        return {
            'signal_id': self.signal_id,
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'action': self.action.value,
            'price': self.price,
            'volume': self.volume,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'tag': self.tag,
            'create_time': self.create_time.isoformat() if self.create_time else None,
            'processed': self.processed
        }


@dataclass
class OrderRequest:
    """订单请求"""
    symbol: str
    exchange: str
    direction: Direction
    offset: Offset
    order_type: OrderType
    price: float
    volume: int

    strategy_name: str = ""
    signal_id: str = ""

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'direction': self.direction.value,
            'offset': self.offset.value,
            'order_type': self.order_type.value,
            'price': self.price,
            'volume': self.volume,
            'strategy_name': self.strategy_name,
            'signal_id': self.signal_id
        }


@dataclass
class Order:
    """订单"""
    order_id: str
    symbol: str
    exchange: str

    direction: Direction = Direction.LONG
    offset: Offset = Offset.OPEN
    order_type: OrderType = OrderType.LIMIT

    price: float = 0.0
    volume: int = 0
    filled_volume: int = 0
    avg_price: float = 0.0

    status: OrderStatus = OrderStatus.PENDING

    strategy_name: str = ""
    signal_id: str = ""

    create_time: datetime = field(default_factory=datetime.now)
    update_time: datetime = field(default_factory=datetime.now)

    error_msg: str = ""

    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]

    @property
    def unfilled_volume(self) -> int:
        return self.volume - self.filled_volume

    def to_dict(self) -> dict:
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'exchange': self.exchange,
            'direction': self.direction.value,
            'offset': self.offset.value,
            'order_type': self.order_type.value,
            'price': self.price,
            'volume': self.volume,
            'filled_volume': self.filled_volume,
            'avg_price': self.avg_price,
            'status': self.status.value,
            'strategy_name': self.strategy_name,
            'create_time': self.create_time.isoformat() if self.create_time else None
        }


@dataclass
class Trade:
    """成交记录（单笔成交，也称Fill）"""
    trade_id: str
    order_id: str
    symbol: str
    exchange: str

    direction: Direction = Direction.LONG
    offset: Offset = Offset.OPEN

    price: float = 0.0
    volume: int = 0
    commission: float = 0.0

    trade_time: datetime = field(default_factory=datetime.now)

    strategy_name: str = ""

    def to_dict(self) -> dict:
        return {
            'trade_id': self.trade_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'offset': self.offset.value,
            'price': self.price,
            'volume': self.volume,
            'commission': self.commission,
            'trade_time': self.trade_time.isoformat() if self.trade_time else None
        }


# 别名，向后兼容
Fill = Trade


@dataclass
class StrategyTrade:
    """
    完整交易记录（从开仓到平仓的完整生命周期）

    参考trader-master设计，支持：
    - 分批建仓（多个开仓订单）
    - 分批平仓（多个平仓订单）
    - 加权均价计算
    - 盈亏追踪
    - 保证金冻结

    生命周期: PENDING -> OPENING -> HOLDING -> CLOSING -> CLOSED
    """
    trade_id: str                           # 交易ID（策略ID+自增序号）
    strategy_name: str                      # 策略名称
    symbol: str                             # 合约代码
    exchange: str = ""                      # 交易所

    direction: Direction = Direction.LONG   # 交易方向
    status: TradeStatus = TradeStatus.PENDING  # 交易状态

    # 计划与实际手数
    shares: int = 0                         # 计划总手数
    filled_shares: int = 0                  # 已开仓成交手数
    closed_shares: int = 0                  # 已平仓成交手数

    # 价格信息
    avg_entry_price: float = 0.0            # 加权入场均价
    avg_exit_price: float = 0.0             # 加权出场均价

    # 盈亏与成本
    unrealized_pnl: float = 0.0             # 未实现盈亏
    realized_pnl: float = 0.0               # 已实现盈亏
    commission: float = 0.0                 # 总手续费
    frozen_margin: float = 0.0              # 冻结保证金

    # 风控参数
    stop_loss_price: float = 0.0            # 止损价
    take_profit_price: float = 0.0          # 止盈价

    # 极值追踪（用于追踪止损）
    highest_price: float = 0.0              # 持仓期间最高价
    lowest_price: float = 0.0               # 持仓期间最低价

    # 时间信息
    create_time: datetime = field(default_factory=datetime.now)
    open_time: Optional[datetime] = None    # 首笔开仓时间
    close_time: Optional[datetime] = None   # 全部平仓时间

    # 关联订单（订单ID列表）
    open_order_ids: List[str] = field(default_factory=list)   # 开仓订单
    close_order_ids: List[str] = field(default_factory=list)  # 平仓订单

    # 成交明细（Fill列表）
    fills: List[Trade] = field(default_factory=list)

    # 信号来源
    signal_id: str = ""                     # 原始信号ID
    entry_tag: str = ""                     # 入场标签
    exit_tag: str = ""                      # 出场标签

    @property
    def holding_shares(self) -> int:
        """当前持仓手数"""
        return self.filled_shares - self.closed_shares

    @property
    def is_active(self) -> bool:
        """是否活跃（未完全平仓）"""
        return self.status not in [TradeStatus.CLOSED, TradeStatus.CANCELLED]

    @property
    def is_fully_filled(self) -> bool:
        """开仓是否完全成交"""
        return self.filled_shares >= self.shares

    @property
    def is_fully_closed(self) -> bool:
        """是否完全平仓"""
        return self.closed_shares >= self.filled_shares and self.filled_shares > 0

    @property
    def holding_duration(self) -> Optional[float]:
        """持仓时长（小时）"""
        if not self.open_time:
            return None
        end = self.close_time or datetime.now()
        return (end - self.open_time).total_seconds() / 3600

    def add_fill(self, fill: Trade, multiplier: float = 1.0) -> None:
        """
        添加成交记录并更新状态

        Args:
            fill: 成交记录
            multiplier: 合约乘数（用于盈亏计算）
        """
        self.fills.append(fill)
        self.commission += fill.commission

        if fill.offset == Offset.OPEN:
            # 开仓成交 - 更新加权入场均价
            total_value = self.avg_entry_price * self.filled_shares + fill.price * fill.volume
            self.filled_shares += fill.volume
            if self.filled_shares > 0:
                self.avg_entry_price = total_value / self.filled_shares

            # 记录首笔开仓时间
            if self.open_time is None:
                self.open_time = fill.trade_time
                self.highest_price = fill.price
                self.lowest_price = fill.price

            # 更新状态
            if self.filled_shares >= self.shares:
                self.status = TradeStatus.HOLDING
            else:
                self.status = TradeStatus.OPENING

        else:
            # 平仓成交 - 更新加权出场均价
            total_value = self.avg_exit_price * self.closed_shares + fill.price * fill.volume
            prev_closed = self.closed_shares
            self.closed_shares += fill.volume
            if self.closed_shares > 0:
                self.avg_exit_price = total_value / self.closed_shares

            # 计算本次平仓的已实现盈亏
            if self.direction == Direction.LONG:
                pnl = (fill.price - self.avg_entry_price) * fill.volume * multiplier
            else:
                pnl = (self.avg_entry_price - fill.price) * fill.volume * multiplier
            self.realized_pnl += pnl

            # 更新状态
            if self.closed_shares >= self.filled_shares:
                self.status = TradeStatus.CLOSED
                self.close_time = fill.trade_time
                self.unrealized_pnl = 0  # 全部平仓，未实现盈亏归零
            else:
                self.status = TradeStatus.CLOSING

    def update_price(self, current_price: float, multiplier: float = 1.0) -> None:
        """
        更新当前价格，计算未实现盈亏

        Args:
            current_price: 当前价格
            multiplier: 合约乘数
        """
        # 更新极值
        if current_price > self.highest_price:
            self.highest_price = current_price
        if current_price < self.lowest_price or self.lowest_price == 0:
            self.lowest_price = current_price

        # 计算未实现盈亏（只计算未平仓部分）
        holding = self.holding_shares
        if holding > 0:
            if self.direction == Direction.LONG:
                self.unrealized_pnl = (current_price - self.avg_entry_price) * holding * multiplier
            else:
                self.unrealized_pnl = (self.avg_entry_price - current_price) * holding * multiplier

    @property
    def total_pnl(self) -> float:
        """总盈亏（已实现+未实现-手续费）"""
        return self.realized_pnl + self.unrealized_pnl - self.commission

    @property
    def return_rate(self) -> float:
        """收益率"""
        if self.frozen_margin > 0:
            return self.total_pnl / self.frozen_margin
        return 0.0

    def to_dict(self) -> dict:
        return {
            'trade_id': self.trade_id,
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'status': self.status.value,
            'shares': self.shares,
            'filled_shares': self.filled_shares,
            'closed_shares': self.closed_shares,
            'holding_shares': self.holding_shares,
            'avg_entry_price': self.avg_entry_price,
            'avg_exit_price': self.avg_exit_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.total_pnl,
            'commission': self.commission,
            'frozen_margin': self.frozen_margin,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'create_time': self.create_time.isoformat() if self.create_time else None,
            'open_time': self.open_time.isoformat() if self.open_time else None,
            'close_time': self.close_time.isoformat() if self.close_time else None,
            'holding_duration': self.holding_duration,
            'return_rate': self.return_rate,
            'entry_tag': self.entry_tag,
            'exit_tag': self.exit_tag,
            'fills_count': len(self.fills)
        }


@dataclass
class Position:
    """持仓"""
    symbol: str
    exchange: str
    direction: Direction

    volume: int = 0
    frozen: int = 0
    avg_price: float = 0.0

    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin: float = 0.0

    # 止损追踪
    highest_price: float = 0.0
    lowest_price: float = 0.0
    entry_time: Optional[datetime] = None

    strategy_name: str = ""

    @property
    def available(self) -> int:
        return self.volume - self.frozen

    def update_price(self, price: float):
        """更新价格追踪"""
        if price > self.highest_price:
            self.highest_price = price
        if price < self.lowest_price or self.lowest_price == 0:
            self.lowest_price = price

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'exchange': self.exchange,
            'direction': self.direction.value,
            'volume': self.volume,
            'frozen': self.frozen,
            'avg_price': self.avg_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'margin': self.margin,
            'highest_price': self.highest_price,
            'lowest_price': self.lowest_price,
            'strategy_name': self.strategy_name
        }


@dataclass
class Account:
    """账户"""
    account_id: str = "default"

    balance: float = 0.0
    available: float = 0.0
    margin: float = 0.0

    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0

    pre_balance: float = 0.0
    deposit: float = 0.0
    withdraw: float = 0.0

    risk_ratio: float = 0.0  # 风险度 = margin / balance

    update_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            'account_id': self.account_id,
            'balance': self.balance,
            'available': self.available,
            'margin': self.margin,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'commission': self.commission,
            'risk_ratio': self.risk_ratio,
            'update_time': self.update_time.isoformat() if self.update_time else None
        }


@dataclass
class StrategyConfig:
    """策略配置"""
    name: str
    display_name: str = ""
    symbols: List[str] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    capital: float = 100000.0
    risk_per_trade: float = 0.02


@dataclass
class Performance:
    """绩效统计"""
    strategy_name: str
    date: datetime

    equity: float = 0.0
    daily_pnl: float = 0.0
    daily_return: float = 0.0
    cumulative_return: float = 0.0
    max_drawdown: float = 0.0

    total_trades: int = 0
    win_trades: int = 0
    lose_trades: int = 0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0


# 事件类型
class EventType(Enum):
    """事件类型"""
    TICK = "tick"
    BAR = "bar"
    SIGNAL = "signal"
    ORDER = "order"
    TRADE = "trade"
    POSITION = "position"
    ACCOUNT = "account"
    LOG = "log"
    TIMER = "timer"
    ERROR = "error"
    STRATEGY = "strategy"


@dataclass
class Event:
    """事件"""
    type: EventType
    data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
