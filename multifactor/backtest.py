# coding=utf-8
"""
多因子选股 - 回测引擎

支持:
- 每日换仓N只
- 等权重配置
- T+1执行
- 手续费和滑点
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class StockPosition:
    """持仓"""
    code: str
    shares: int
    avg_price: float
    entry_date: str
    market_value: float = 0


@dataclass
class StockTrade:
    """交易记录"""
    date: str
    code: str
    direction: str
    price: float
    shares: int
    amount: float
    commission: float
    pnl: float = 0


@dataclass
class BacktestResult:
    """回测结果"""
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    annual_return: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    total_trades: int
    win_rate: float
    turnover: float
    equity_curve: pd.DataFrame = None
    trades: List[StockTrade] = None


class MultifactorBacktest:
    """
    多因子选股回测引擎

    每日流程:
    1. 更新持仓市值
    2. 执行换仓 (T+1)
    3. 记录净值
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.001,  # 千分之一
        slippage: float = 0.001,
        hold_num: int = 10,  # 持仓数量
        rebalance_num: int = 1  # 每日换仓数量
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.hold_num = hold_num
        self.rebalance_num = rebalance_num

        self._cash = initial_capital
        self._positions: Dict[str, StockPosition] = {}
        self._trades: List[StockTrade] = []
        self._equity_history: List[dict] = []

        self._current_date = ""
        self._stock_data: Dict[str, pd.DataFrame] = {}
        self._pending_sells: List[str] = []
        self._pending_buys: List[str] = []

    def run(self, stock_data: Dict[str, pd.DataFrame],
            daily_selections: Dict[str, List[str]],
            start_date: str, end_date: str) -> BacktestResult:
        """
        运行回测

        Args:
            stock_data: {code: DataFrame} 股票数据
            daily_selections: {date: [code1, code2, ...]} 每日选股结果
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            BacktestResult
        """
        logger.info(f"开始多因子回测: {start_date} ~ {end_date}")

        self._reset()
        self._stock_data = stock_data

        # 获取交易日
        all_dates = set()
        for df in stock_data.values():
            all_dates.update(df["date"].tolist())
        trading_dates = sorted([d for d in all_dates if start_date <= d <= end_date])

        logger.info(f"交易日: {len(trading_dates)}天")

        prev_date = None
        for date in trading_dates:
            self._current_date = date

            # 1. 更新市值
            self._update_market_value()

            # 2. 执行前一日的换仓信号 (T+1)
            if self._pending_sells or self._pending_buys:
                self._execute_rebalance()

            # 3. 生成今日换仓信号
            if date in daily_selections:
                target_stocks = daily_selections[date][:self.hold_num]
                self._generate_rebalance_signal(target_stocks)

            # 4. 记录净值
            self._record_equity()

            prev_date = date

        return self._calculate_result(start_date, end_date)

    def _reset(self):
        self._cash = self.initial_capital
        self._positions = {}
        self._trades = []
        self._equity_history = []
        self._pending_sells = []
        self._pending_buys = []

    def _get_price(self, code: str, price_type: str = "close") -> Optional[float]:
        """获取价格"""
        if code not in self._stock_data:
            return None
        df = self._stock_data[code]
        day_df = df[df["date"] == self._current_date]
        if len(day_df) == 0:
            return None
        return day_df.iloc[0].get(price_type)

    def _update_market_value(self):
        """更新持仓市值"""
        for code, pos in self._positions.items():
            price = self._get_price(code)
            if price:
                pos.market_value = pos.shares * price

    def _get_total_value(self) -> float:
        return self._cash + sum(p.market_value for p in self._positions.values())

    def _generate_rebalance_signal(self, target_stocks: List[str]):
        """
        生成换仓信号

        策略: 卖出不在目标列表中的，买入新的目标股票
        每日最多换仓rebalance_num只
        """
        current_stocks = set(self._positions.keys())
        target_set = set(target_stocks)

        # 需要卖出的
        to_sell = list(current_stocks - target_set)
        # 需要买入的
        to_buy = [s for s in target_stocks if s not in current_stocks]

        # 限制每日换仓数量
        self._pending_sells = to_sell[:self.rebalance_num]
        self._pending_buys = to_buy[:self.rebalance_num]

    def _execute_rebalance(self):
        """执行换仓 (使用开盘价)"""
        # 先卖出
        for code in self._pending_sells:
            if code in self._positions:
                self._sell_stock(code)

        # 再买入
        if self._pending_buys:
            # 等权重配置
            per_stock_value = self._get_total_value() / self.hold_num

            for code in self._pending_buys:
                if code not in self._positions:
                    self._buy_stock(code, per_stock_value)

        self._pending_sells = []
        self._pending_buys = []

    def _buy_stock(self, code: str, target_value: float):
        """买入股票"""
        price = self._get_price(code, "open")
        if price is None or price <= 0:
            return

        exec_price = price * (1 + self.slippage)
        shares = int(target_value / exec_price / 100) * 100

        if shares <= 0:
            return

        amount = shares * exec_price
        commission = max(amount * self.commission_rate, 5)
        total_cost = amount + commission

        if total_cost > self._cash:
            shares = int((self._cash - commission) / exec_price / 100) * 100
            if shares <= 0:
                return
            amount = shares * exec_price
            commission = max(amount * self.commission_rate, 5)
            total_cost = amount + commission

        self._cash -= total_cost

        self._positions[code] = StockPosition(
            code=code,
            shares=shares,
            avg_price=exec_price,
            entry_date=self._current_date,
            market_value=shares * exec_price
        )

        self._trades.append(StockTrade(
            date=self._current_date, code=code, direction="BUY",
            price=exec_price, shares=shares, amount=amount, commission=commission
        ))

    def _sell_stock(self, code: str):
        """卖出股票"""
        if code not in self._positions:
            return

        pos = self._positions[code]
        price = self._get_price(code, "open")
        if price is None:
            price = pos.avg_price

        exec_price = price * (1 - self.slippage)
        amount = pos.shares * exec_price
        commission = max(amount * self.commission_rate, 5)

        pnl = (exec_price - pos.avg_price) * pos.shares - commission
        self._cash += amount - commission

        self._trades.append(StockTrade(
            date=self._current_date, code=code, direction="SELL",
            price=exec_price, shares=pos.shares, amount=amount,
            commission=commission, pnl=pnl
        ))

        del self._positions[code]

    def _record_equity(self):
        self._equity_history.append({
            "date": self._current_date,
            "cash": self._cash,
            "market_value": sum(p.market_value for p in self._positions.values()),
            "total_value": self._get_total_value(),
            "positions_count": len(self._positions)
        })

    def _calculate_result(self, start_date: str, end_date: str) -> BacktestResult:
        equity_df = pd.DataFrame(self._equity_history)
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        equity_df = equity_df.set_index("date")
        equity_df["return"] = equity_df["total_value"].pct_change()

        final_value = equity_df["total_value"].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        rolling_max = equity_df["total_value"].cummax()
        drawdown = (equity_df["total_value"] - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())

        volatility = equity_df["return"].std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.03) / volatility if volatility > 0 else 0
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

        # 交易统计
        sell_trades = [t for t in self._trades if t.direction == "SELL"]
        win_trades = len([t for t in sell_trades if t.pnl > 0])
        win_rate = win_trades / len(sell_trades) if sell_trades else 0

        # 换手率
        total_amount = sum(t.amount for t in self._trades)
        turnover = total_amount / self.initial_capital / years if years > 0 else 0

        return BacktestResult(
            start_date=start_date, end_date=end_date,
            initial_capital=self.initial_capital, final_value=final_value,
            total_return=total_return, annual_return=annual_return,
            max_drawdown=max_drawdown, sharpe_ratio=sharpe_ratio,
            calmar_ratio=calmar_ratio, total_trades=len(self._trades),
            win_rate=win_rate, turnover=turnover,
            equity_curve=equity_df, trades=self._trades
        )
