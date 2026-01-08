# coding=utf-8
"""
回测专用数据模型
统一回测引擎使用的数据结构
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class BacktestTrade:
    """
    回测交易记录（完整的一买一卖）

    与 models.base.Trade 不同，这是一个完整的 round-trip 交易，
    包含入场和出场信息。
    """
    # 基本信息
    trade_id: int
    symbol: str
    direction: int  # 1=多, -1=空

    # 入场信息
    entry_time: datetime
    entry_price: float
    entry_tag: str = ""
    volume: int = 1

    # 出场信息
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_tag: Optional[str] = None

    # 盈亏统计
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0

    # 持仓统计
    holding_bars: int = 0
    holding_days: int = 0

    # 极值统计
    max_profit: float = 0.0
    max_loss: float = 0.0
    max_profit_pct: float = 0.0
    max_loss_pct: float = 0.0

    # 资金信息
    capital_before: float = 0.0
    capital_after: float = 0.0

    @property
    def is_closed(self) -> bool:
        """是否已平仓"""
        return self.exit_time is not None

    @property
    def is_winner(self) -> bool:
        """是否盈利"""
        return self.pnl > 0

    @property
    def direction_str(self) -> str:
        """方向字符串"""
        return "多" if self.direction == 1 else "空"

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'direction': self.direction_str,
            'entry_time': self.entry_time.strftime('%Y-%m-%d %H:%M') if self.entry_time else None,
            'entry_price': self.entry_price,
            'entry_tag': self.entry_tag,
            'exit_time': self.exit_time.strftime('%Y-%m-%d %H:%M') if self.exit_time else None,
            'exit_price': self.exit_price,
            'exit_tag': self.exit_tag,
            'volume': self.volume,
            'pnl': round(self.pnl, 2),
            'pnl_pct': round(self.pnl_pct * 100, 2),
            'commission': round(self.commission, 2),
            'holding_bars': self.holding_bars,
            'holding_days': self.holding_days,
        }


@dataclass
class BacktestResult:
    """
    回测结果

    包含完整的回测统计指标和交易记录
    """
    # 基本信息
    symbol: str
    strategy_name: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    period: str = "1h"  # 回测周期

    # 资金曲线
    final_capital: float = 0.0
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_returns: pd.Series = field(default_factory=pd.Series)

    # 收益指标
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    annual_return_pct: float = 0.0

    # 风险指标
    max_drawdown_pct: float = 0.0
    max_drawdown_val: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # 盈亏统计
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    avg_pnl: float = 0.0

    # 持仓统计
    avg_holding_bars: float = 0.0
    avg_holding_days: float = 0.0
    max_holding_days: int = 0

    # 费用统计
    total_commission: float = 0.0

    # 交易记录
    trades: List[BacktestTrade] = field(default_factory=list)

    # 分组统计
    monthly_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    yearly_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    exit_tag_stats: Dict[str, Any] = field(default_factory=dict)

    def calculate_statistics(self):
        """根据交易记录计算统计指标"""
        if not self.trades:
            return

        closed_trades = [t for t in self.trades if t.is_closed]
        if not closed_trades:
            return

        self.total_trades = len(closed_trades)

        # 盈亏统计
        pnls = [t.pnl for t in closed_trades]
        self.total_pnl = sum(pnls)
        self.avg_pnl = np.mean(pnls)

        winners = [t for t in closed_trades if t.is_winner]
        losers = [t for t in closed_trades if not t.is_winner]

        self.winning_trades = len(winners)
        self.losing_trades = len(losers)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        if winners:
            self.avg_win = np.mean([t.pnl for t in winners])
            self.max_win = max([t.pnl for t in winners])

        if losers:
            self.avg_loss = np.mean([t.pnl for t in losers])
            self.max_loss = min([t.pnl for t in losers])

        # 盈亏比
        total_profit = sum([t.pnl for t in winners]) if winners else 0
        total_loss = abs(sum([t.pnl for t in losers])) if losers else 0
        self.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # 持仓统计
        holding_days = [t.holding_days for t in closed_trades if t.holding_days > 0]
        holding_bars = [t.holding_bars for t in closed_trades if t.holding_bars > 0]

        if holding_days:
            self.avg_holding_days = np.mean(holding_days)
            self.max_holding_days = max(holding_days)

        if holding_bars:
            self.avg_holding_bars = np.mean(holding_bars)

        # 费用统计
        self.total_commission = sum([t.commission for t in closed_trades])

        # 收益率计算
        if self.initial_capital > 0:
            self.total_return_pct = self.total_pnl / self.initial_capital
            self.final_capital = self.initial_capital + self.total_pnl

            # 年化收益
            if self.start_date and self.end_date:
                days = (self.end_date - self.start_date).days
                if days > 0:
                    self.annual_return_pct = self.total_return_pct * (365 / days)

    def to_summary_dict(self) -> dict:
        """转换为摘要字典"""
        return {
            'symbol': self.symbol,
            'strategy': self.strategy_name,
            'period': f"{self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}",
            'initial_capital': f"{self.initial_capital:,.0f}",
            'final_capital': f"{self.final_capital:,.0f}",
            'total_pnl': f"{self.total_pnl:,.2f}",
            'total_return': f"{self.total_return_pct:.2%}",
            'annual_return': f"{self.annual_return_pct:.2%}",
            'max_drawdown': f"{self.max_drawdown_pct:.2%}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'total_trades': self.total_trades,
            'win_rate': f"{self.win_rate:.2%}",
            'profit_factor': f"{self.profit_factor:.2f}",
            'avg_holding_days': f"{self.avg_holding_days:.1f}",
        }


# 兼容性别名
Trade = BacktestTrade  # 供旧代码使用
TradeRecord = BacktestTrade  # 供旧代码使用
