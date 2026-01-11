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
    2. 检查风控条件
    3. 执行换仓 (T+1)
    4. 记录净值

    风控功能:
    - 波动率择时: 高波动时减仓
    - 换仓阈值: 预测分差小时不换
    - 整体止损: 回撤超阈值清仓
    """

    def __init__(
        self,
        initial_capital: float = 1000000,
        commission_rate: float = 0.001,  # 千分之一
        slippage: float = 0.001,
        hold_num: int = 10,  # 持仓数量
        rebalance_num: int = 1,  # 每次换仓数量
        # === 换手率控制参数 ===
        rebalance_days: int = 1,  # 调仓周期（每N天调一次）
        position_sticky: float = 0.0,  # 持仓粘性（0-1，越高越不易换出）
        min_holding_days: int = 0,  # 最小持仓天数
        # === 风控参数 ===
        vol_timing: bool = False,  # 波动率择时
        vol_threshold: float = 0.30,  # 波动率阈值 (年化30%)
        vol_reduce_ratio: float = 0.5,  # 高波动时仓位比例
        rebalance_threshold: float = 0.0,  # 换仓阈值 (预测分差)
        drawdown_stop: bool = False,  # 整体止损
        max_drawdown_limit: float = 0.15,  # 最大回撤阈值
        cooldown_days: int = 10,  # 止损后冷却期
        # === 市场择时参数 ===
        market_timing: bool = False,  # 市场均线择时
        market_ma_days: int = 20  # 市场均线天数
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.hold_num = hold_num
        self.rebalance_num = rebalance_num

        # 换手率控制参数
        self.rebalance_days = rebalance_days
        self.position_sticky = position_sticky
        self.min_holding_days = min_holding_days

        # 风控参数
        self.vol_timing = vol_timing
        self.vol_threshold = vol_threshold
        self.vol_reduce_ratio = vol_reduce_ratio
        self.rebalance_threshold = rebalance_threshold
        self.drawdown_stop = drawdown_stop
        self.max_drawdown_limit = max_drawdown_limit
        self.cooldown_days = cooldown_days

        # 市场择时参数
        self.market_timing = market_timing
        self.market_ma_days = market_ma_days

        self._cash = initial_capital
        self._positions: Dict[str, StockPosition] = {}
        self._trades: List[StockTrade] = []
        self._equity_history: List[dict] = []

        self._current_date = ""
        self._stock_data: Dict[str, pd.DataFrame] = {}
        self._pending_sells: List[str] = []
        self._pending_buys: List[str] = []

        # 风控状态
        self._market_returns: List[float] = []  # 市场收益率序列
        self._peak_value: float = initial_capital  # 净值高点
        self._in_drawdown_stop: bool = False  # 是否在止损状态
        self._cooldown_counter: int = 0  # 冷却计数器
        self._current_vol: float = 0  # 当前波动率

        # 换手率控制状态
        self._day_counter: int = 0  # 调仓日计数器
        self._market_values: List[float] = []  # 市场净值序列（用于均线计算）
        self._market_above_ma: bool = True  # 市场是否在均线上方

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
            self._day_counter += 1

            # 1. 更新市值
            self._update_market_value()

            # 2. 更新市场收益率（用于波动率计算）
            self._update_market_return()

            # 3. 更新市场均线状态（用于市场择时）
            self._update_market_timing()

            # 4. 检查整体止损
            if self._check_drawdown_stop():
                # 止损状态：清仓并跳过交易
                if self._positions:
                    for code in list(self._positions.keys()):
                        self._sell_stock(code)
                self._pending_sells = []
                self._pending_buys = []
                self._record_equity()
                prev_date = date
                continue

            # 5. 检查市场择时（熊市减仓或空仓）
            if self.market_timing and not self._market_above_ma:
                # 市场在均线下方，减仓到一半或清仓
                if self._positions and len(self._positions) > self.hold_num // 2:
                    # 只保留一半持仓
                    keep_count = max(self.hold_num // 2, 1)
                    to_sell = list(self._positions.keys())[keep_count:]
                    for code in to_sell:
                        self._sell_stock(code)

            # 6. 执行前一日的换仓信号 (T+1)
            if self._pending_sells or self._pending_buys:
                self._execute_rebalance()

            # 7. 生成今日换仓信号（考虑调仓周期）
            is_rebalance_day = (self._day_counter % self.rebalance_days == 0)
            if date in daily_selections and is_rebalance_day:
                effective_hold_num = self._get_effective_hold_num()
                # 市场择时：熊市减仓
                if self.market_timing and not self._market_above_ma:
                    effective_hold_num = max(effective_hold_num // 2, 1)
                target_stocks = daily_selections[date][:effective_hold_num]
                self._generate_rebalance_signal(target_stocks)

            # 8. 记录净值
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
        # 风控状态重置
        self._market_returns = []
        self._peak_value = self.initial_capital
        self._in_drawdown_stop = False
        self._cooldown_counter = 0
        self._current_vol = 0
        # 换手率控制状态重置
        self._day_counter = 0
        self._market_values = []
        self._market_above_ma = True

    def _calculate_market_volatility(self) -> float:
        """计算市场20日波动率（年化）"""
        if len(self._market_returns) < 20:
            return 0.0
        recent_returns = self._market_returns[-20:]
        daily_vol = np.std(recent_returns)
        annual_vol = daily_vol * np.sqrt(252)
        return annual_vol

    def _update_market_timing(self):
        """更新市场均线状态（用于市场择时）"""
        current_value = self._get_total_value()
        self._market_values.append(current_value)

        if not self.market_timing:
            return

        # 计算MA
        if len(self._market_values) >= self.market_ma_days:
            ma = np.mean(self._market_values[-self.market_ma_days:])
            self._market_above_ma = current_value >= ma
        else:
            self._market_above_ma = True  # 数据不足，默认做多

    def _update_market_return(self):
        """更新市场收益率（用持仓股票平均收益）"""
        if not self._positions:
            return

        daily_returns = []
        for code in self._positions.keys():
            if code not in self._stock_data:
                continue
            df = self._stock_data[code]
            day_df = df[df["date"] == self._current_date]
            if len(day_df) == 0:
                continue
            row = day_df.iloc[0]
            if row["open"] > 0:
                ret = (row["close"] - row["open"]) / row["open"]
                daily_returns.append(ret)

        if daily_returns:
            avg_return = np.mean(daily_returns)
            self._market_returns.append(avg_return)

    def _check_drawdown_stop(self) -> bool:
        """检查是否触发整体止损"""
        if not self.drawdown_stop:
            return False

        current_value = self._get_total_value()

        # 更新高点
        if current_value > self._peak_value:
            self._peak_value = current_value

        # 计算回撤
        drawdown = (self._peak_value - current_value) / self._peak_value

        # 检查是否触发止损
        if drawdown >= self.max_drawdown_limit and not self._in_drawdown_stop:
            logger.warning(f"{self._current_date} 触发整体止损! 回撤: {drawdown*100:.1f}%")
            self._in_drawdown_stop = True
            self._cooldown_counter = self.cooldown_days
            return True

        # 冷却期处理
        if self._in_drawdown_stop:
            self._cooldown_counter -= 1
            if self._cooldown_counter <= 0:
                logger.info(f"{self._current_date} 冷却期结束，恢复交易")
                self._in_drawdown_stop = False
                self._peak_value = current_value  # 重置高点

        return self._in_drawdown_stop

    def _get_effective_hold_num(self) -> int:
        """获取有效持仓数量（考虑波动率择时）"""
        if not self.vol_timing:
            return self.hold_num

        self._current_vol = self._calculate_market_volatility()

        if self._current_vol > self.vol_threshold:
            reduced_num = int(self.hold_num * self.vol_reduce_ratio)
            return max(reduced_num, 1)

        return self.hold_num

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
        每次最多换仓rebalance_num只

        优化功能:
        - 持仓粘性: 已持有的股票不轻易换出
        - 最小持仓天数: 持仓不满N天不换
        - 波动率择时: 当需要减仓时，优先卖出排名靠后的持仓
        """
        current_stocks = set(self._positions.keys())
        target_set = set(target_stocks)
        effective_hold_num = len(target_stocks)

        # 需要卖出的（不在目标列表中的）
        to_sell_candidates = []
        for code in current_stocks:
            if code not in target_set:
                # 检查最小持仓天数
                if self.min_holding_days > 0:
                    pos = self._positions[code]
                    # 简化的天数计算（实际应该用交易日）
                    holding_days = len([d for d in self._equity_history
                                       if d.get("date", "") >= pos.entry_date])
                    if holding_days < self.min_holding_days:
                        continue  # 持仓时间不够，不卖

                # 检查持仓粘性：如果股票在target_stocks的后半部分，有一定概率保留
                if self.position_sticky > 0 and code in target_stocks:
                    rank = target_stocks.index(code)
                    if rank < len(target_stocks) * (1 - self.position_sticky):
                        continue  # 排名靠前，保留

                to_sell_candidates.append(code)

        to_sell = to_sell_candidates

        # 波动率择时减仓：如果当前持仓数 > 目标持仓数
        current_count = len(self._positions)
        if current_count > effective_hold_num:
            # 需要额外减仓的数量
            extra_to_sell = current_count - effective_hold_num
            # 从当前持仓中选择不在目标列表顶部的股票卖出
            for stock in reversed(list(self._positions.keys())):
                if stock not in to_sell and len(to_sell) < current_count - effective_hold_num + self.rebalance_num:
                    to_sell.append(stock)

        # 需要买入的
        to_buy = [s for s in target_stocks if s not in current_stocks]

        # 限制每次换仓数量
        self._pending_sells = to_sell[:self.rebalance_num + max(0, current_count - effective_hold_num)]
        self._pending_buys = to_buy[:self.rebalance_num] if current_count <= effective_hold_num else []

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
            "positions_count": len(self._positions),
            "volatility": self._current_vol,
            "in_stop": self._in_drawdown_stop
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
