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
    """成交记录"""
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
