# coding=utf-8
"""
ETF回测引擎
支持ETF策略回测、绩效计算、交易记录等
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ETFPosition:
    """ETF持仓信息"""
    code: str
    shares: int = 0
    avg_price: float = 0.0
    entry_date: str = ""
    entry_price: float = 0.0
    highest_price: float = 0.0
    stop_price: float = 0.0
    market_value: float = 0.0


@dataclass
class ETFTrade:
    """ETF交易记录"""
    date: str
    code: str
    direction: str  # "BUY" or "SELL"
    price: float
    shares: int
    amount: float
    commission: float
    reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class ETFBacktestResult:
    """ETF回测结果"""
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    annual_return: float
    benchmark_return: float
    excess_return: float
    max_drawdown: float
    max_drawdown_duration: int
    volatility: float
    downside_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_trades: int
    win_trades: int
    lose_trades: int
    win_rate: float
    profit_loss_ratio: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float
    avg_holding_days: float
    equity_curve: pd.DataFrame = None
    trades: List[ETFTrade] = None
    positions_history: pd.DataFrame = None


class ETFBacktestContext:
    """ETF回测上下文"""

    def __init__(self, engine: 'ETFBacktestEngine'):
        self._engine = engine

        # 策略参数
        self.base_position = 0.18
        self.atr_multiplier = 2.5
        self.max_loss = 0.07
        self.trail_start = 0.15
        self.trail_stop = 0.06
        self.max_hold = 120
        self.cooldown = 3
        self.adx_threshold = 20

        # 内部状态
        self.cooldown_dict = {}
        self.entry_prices = {}
        self.entry_dates = {}
        self.highest = {}
        self.stops = {}

        # 标的分类
        self.overseas = ['513100.SH', '159941.SZ', '518880.SH']
        self.high_vol = ['588000.SH', '516010.SH', '512480.SH']

    @property
    def current_date(self) -> str:
        return self._engine._current_date

    @property
    def current_dt(self) -> datetime:
        return datetime.strptime(self._engine._current_date, "%Y-%m-%d")

    @property
    def data(self) -> pd.DataFrame:
        return self._engine._current_data

    def get_account_positions(self) -> Dict[str, ETFPosition]:
        return self._engine._positions.copy()

    def get_cash(self) -> float:
        return self._engine._cash

    def get_total_value(self) -> float:
        return self._engine._get_total_value()

    def order_target_percent(self, code: str, target_percent: float):
        """按目标百分比下单"""
        self._engine._order_target_percent(code, target_percent)


class ETFBacktestEngine:
    """ETF回测引擎"""

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0001,
        slippage: float = 0.0001,
        benchmark: str = "510300.SH"
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.benchmark = benchmark

        self._cash = initial_capital
        self._positions: Dict[str, ETFPosition] = {}
        self._trades: List[ETFTrade] = []
        self._equity_history: List[dict] = []

        self._current_date = ""
        self._current_data = None
        self._all_data: Dict[str, pd.DataFrame] = {}
        self._benchmark_data: pd.DataFrame = None

        self._initialize_func = None
        self._handle_data_func = None
        self._context = None

    def set_strategy(self, initialize: Callable = None, handle_data: Callable = None):
        """设置策略函数"""
        self._initialize_func = initialize
        self._handle_data_func = handle_data

    def run(self, data: Dict[str, pd.DataFrame], start_date: str, end_date: str,
            benchmark_data: pd.DataFrame = None) -> ETFBacktestResult:
        """运行回测"""
        logger.info(f"开始ETF回测: {start_date} ~ {end_date}")

        self._reset()
        self._all_data = data
        self._benchmark_data = benchmark_data
        self._context = ETFBacktestContext(self)

        if self._initialize_func:
            self._initialize_func(self._context)

        # 获取所有交易日期
        all_dates = set()
        for code, df in data.items():
            if "date" in df.columns:
                all_dates.update(df["date"].tolist())

        trading_dates = sorted([d for d in all_dates if start_date <= d <= end_date])

        if not trading_dates:
            raise ValueError("没有可用的交易日期")

        logger.info(f"交易日期: {len(trading_dates)}天")

        for date in trading_dates:
            self._current_date = date
            self._current_data = self._get_daily_data(date)
            self._update_positions_market_value()

            if self._handle_data_func and len(self._current_data) > 0:
                self._handle_data_func(self._context, None)

            self._record_equity()

        result = self._calculate_result(start_date, end_date)
        logger.info(f"ETF回测完成: 总收益 {result.total_return*100:.2f}%")

        return result

    def _reset(self):
        self._cash = self.initial_capital
        self._positions = {}
        self._trades = []
        self._equity_history = []

    def _get_daily_data(self, date: str) -> pd.DataFrame:
        rows = []
        for code, df in self._all_data.items():
            day_df = df[df["date"] == date]
            if len(day_df) > 0:
                row = day_df.iloc[0].to_dict()
                row["instrument"] = code
                rows.append(row)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _update_positions_market_value(self):
        for code, pos in self._positions.items():
            if code in self._all_data:
                df = self._all_data[code]
                day_df = df[df["date"] == self._current_date]
                if len(day_df) > 0:
                    price = day_df.iloc[0]["close"]
                    pos.market_value = pos.shares * price
                    if price > pos.highest_price:
                        pos.highest_price = price

    def _get_total_value(self) -> float:
        total = self._cash
        for pos in self._positions.values():
            total += pos.market_value
        return total

    def _get_price(self, code: str) -> Optional[float]:
        if self._current_data is None or len(self._current_data) == 0:
            return None
        row = self._current_data[self._current_data["instrument"] == code]
        if len(row) > 0:
            return row.iloc[0]["close"]
        return None

    def _order_target_percent(self, code: str, target_percent: float):
        total_value = self._get_total_value()
        target_value = total_value * target_percent

        price = self._get_price(code)
        if price is None:
            return

        current_value = self._positions[code].market_value if code in self._positions else 0
        diff_value = target_value - current_value

        if abs(diff_value) < 100:
            return

        shares = int(diff_value / price / 100) * 100

        if shares != 0:
            self._order_shares(code, shares)

    def _order_shares(self, code: str, shares: int):
        price = self._get_price(code)
        if price is None:
            return

        exec_price = price * (1 + self.slippage) if shares > 0 else price * (1 - self.slippage)
        amount = abs(shares) * exec_price
        commission = max(amount * self.commission_rate, 0.1)

        if shares > 0:
            total_cost = amount + commission
            if total_cost > self._cash:
                available = self._cash - commission
                shares = int(available / exec_price / 100) * 100
                if shares <= 0:
                    return
                amount = shares * exec_price
                commission = max(amount * self.commission_rate, 0.1)
                total_cost = amount + commission

            self._cash -= total_cost

            if code not in self._positions:
                self._positions[code] = ETFPosition(
                    code=code,
                    shares=shares,
                    avg_price=exec_price,
                    entry_date=self._current_date,
                    entry_price=exec_price,
                    highest_price=exec_price,
                    market_value=shares * exec_price
                )
            else:
                pos = self._positions[code]
                total_shares = pos.shares + shares
                pos.avg_price = (pos.shares * pos.avg_price + shares * exec_price) / total_shares
                pos.shares = total_shares
                pos.market_value = total_shares * exec_price

            self._trades.append(ETFTrade(
                date=self._current_date, code=code, direction="BUY",
                price=exec_price, shares=shares, amount=amount, commission=commission
            ))

        elif shares < 0:
            shares = abs(shares)
            if code not in self._positions:
                return

            pos = self._positions[code]
            shares = min(shares, pos.shares)
            if shares <= 0:
                return

            amount = shares * exec_price
            commission = max(amount * self.commission_rate, 0.1)
            self._cash += amount - commission

            pnl = (exec_price - pos.avg_price) * shares - commission
            pnl_pct = (exec_price - pos.entry_price) / pos.entry_price if pos.entry_price > 0 else 0

            self._trades.append(ETFTrade(
                date=self._current_date, code=code, direction="SELL",
                price=exec_price, shares=shares, amount=amount, commission=commission,
                pnl=pnl, pnl_pct=pnl_pct
            ))

            pos.shares -= shares
            pos.market_value = pos.shares * exec_price

            if pos.shares <= 0:
                del self._positions[code]

    def _record_equity(self):
        self._equity_history.append({
            "date": self._current_date,
            "cash": self._cash,
            "market_value": sum(p.market_value for p in self._positions.values()),
            "total_value": self._get_total_value(),
            "positions_count": len(self._positions)
        })

    def _calculate_result(self, start_date: str, end_date: str) -> ETFBacktestResult:
        equity_df = pd.DataFrame(self._equity_history)
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df.set_index("date")
        equity_df["return"] = equity_df["total_value"].pct_change()
        equity_df["cumulative_return"] = equity_df["total_value"] / self.initial_capital - 1

        final_value = equity_df["total_value"].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        benchmark_return = 0
        if self._benchmark_data is not None and len(self._benchmark_data) > 0:
            bench_df = self._benchmark_data[
                (self._benchmark_data["date"] >= start_date) &
                (self._benchmark_data["date"] <= end_date)
            ]
            if len(bench_df) > 1:
                benchmark_return = bench_df.iloc[-1]["close"] / bench_df.iloc[0]["close"] - 1

        rolling_max = equity_df["total_value"].cummax()
        drawdown = (equity_df["total_value"] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        drawdown_duration = 0
        current_duration = 0
        for dd in drawdown:
            if dd < 0:
                current_duration += 1
                drawdown_duration = max(drawdown_duration, current_duration)
            else:
                current_duration = 0

        volatility = equity_df["return"].std() * np.sqrt(252)
        negative_returns = equity_df["return"][equity_df["return"] < 0]
        downside_volatility = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0

        risk_free_rate = 0.03
        excess_return_daily = equity_df["return"].mean() - risk_free_rate / 252
        sharpe_ratio = excess_return_daily / equity_df["return"].std() * np.sqrt(252) if equity_df["return"].std() > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

        sell_trades = [t for t in self._trades if t.direction == "SELL"]
        total_trades = len(sell_trades)
        win_trades = len([t for t in sell_trades if t.pnl > 0])
        lose_trades = len([t for t in sell_trades if t.pnl < 0])
        win_rate = win_trades / total_trades if total_trades > 0 else 0

        wins = [t.pnl for t in sell_trades if t.pnl > 0]
        losses = [abs(t.pnl) for t in sell_trades if t.pnl < 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        max_win = max(wins) if wins else 0
        max_loss = max(losses) if losses else 0

        avg_holding_days = 0
        if sell_trades:
            holding_days = []
            for trade in sell_trades:
                buy_trades = [t for t in self._trades if t.code == trade.code and t.direction == "BUY" and t.date <= trade.date]
                if buy_trades:
                    entry_date = datetime.strptime(buy_trades[-1].date, "%Y-%m-%d")
                    exit_date = datetime.strptime(trade.date, "%Y-%m-%d")
                    holding_days.append((exit_date - entry_date).days)
            avg_holding_days = np.mean(holding_days) if holding_days else 0

        return ETFBacktestResult(
            start_date=start_date, end_date=end_date,
            initial_capital=self.initial_capital, final_value=final_value,
            total_return=total_return, annual_return=annual_return,
            benchmark_return=benchmark_return, excess_return=total_return - benchmark_return,
            max_drawdown=max_drawdown, max_drawdown_duration=drawdown_duration,
            volatility=volatility, downside_volatility=downside_volatility,
            sharpe_ratio=sharpe_ratio, sortino_ratio=sortino_ratio, calmar_ratio=calmar_ratio,
            total_trades=total_trades, win_trades=win_trades, lose_trades=lose_trades,
            win_rate=win_rate, profit_loss_ratio=profit_loss_ratio,
            avg_win=avg_win, avg_loss=avg_loss, max_win=max_win, max_loss=max_loss,
            avg_holding_days=avg_holding_days,
            equity_curve=equity_df, trades=self._trades
        )
