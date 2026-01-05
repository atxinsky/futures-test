# coding=utf-8
"""
IntradayMomentum 股指日内动量策略

核心思路：
1. 用短期动量判断方向
2. 成交量确认
3. 顺势入场，逆势不做
4. 收盘前平仓

特点：不依赖开盘区间，随时可以入场
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, StrategyParam, Signal


class IntradayMomentum(BaseStrategy):
    """股指日内动量策略"""

    name = "intraday_momentum"
    display_name = "日内动量"
    description = "动量跟随，收盘前平仓"
    version = "1.0"
    author = "Claude"

    warmup_num = 20

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            # 动量参数
            StrategyParam("mom_len", "动量周期", 3, 2, 6, 1, "int"),
            StrategyParam("mom_thres", "动量阈值%", 0.3, 0.1, 0.5, 0.1, "float"),

            # 均线过滤
            StrategyParam("ma_len", "均线周期", 10, 5, 20, 5, "int"),

            # 成交量
            StrategyParam("vol_len", "量能周期", 5, 3, 10, 1, "int"),
            StrategyParam("vol_mult", "放量倍数", 1.2, 1.0, 2.0, 0.2, "float"),

            # 止损止盈
            StrategyParam("stop_pct", "止损%", 0.4, 0.2, 0.8, 0.1, "float"),
            StrategyParam("target_pct", "目标%", 0.8, 0.4, 1.5, 0.2, "float"),
            StrategyParam("trail_pct", "追踪%", 0.25, 0.1, 0.5, 0.05, "float"),

            # 时间控制
            StrategyParam("start_bar", "开始K线", 3, 2, 5, 1, "int"),
            StrategyParam("exit_bar", "平仓K线", 16, 14, 16, 1, "int"),
            StrategyParam("max_trades", "日最大交易", 2, 1, 3, 1, "int"),

            # 仓位
            StrategyParam("capital_rate", "资金比例", 0.5, 0.2, 0.8, 0.1, "float"),
            StrategyParam("risk_rate", "风险比例", 0.02, 0.01, 0.05, 0.01, "float"),
        ]

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.current_date = None
        self.trades_today = 0
        self.stop_price = 0
        self.target_price = 0
        self.highest = 0
        self.lowest = float('inf')

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        mom_len = self.params['mom_len']
        ma_len = self.params['ma_len']
        vol_len = self.params['vol_len']

        # 解析日期和时间
        if 'time' in df.columns:
            df['dt'] = pd.to_datetime(df['time'])
        elif 'datetime' in df.columns:
            df['dt'] = pd.to_datetime(df['datetime'])
        else:
            df['dt'] = pd.to_datetime(df.index)

        df['date'] = df['dt'].dt.date
        df['bar_of_day'] = df.groupby('date').cumcount() + 1

        # 动量
        df['mom'] = (df['close'] - df['close'].shift(mom_len)) / df['close'].shift(mom_len) * 100

        # 均线
        df['ma'] = df['close'].rolling(ma_len).mean()

        # 成交量
        df['vol_ma'] = df['volume'].rolling(vol_len).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']

        # 日内高低点
        df['day_high'] = df.groupby('date')['high'].cummax()
        df['day_low'] = df.groupby('date')['low'].cummin()

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        if idx < self.warmup_num:
            return None

        row = df.iloc[idx]

        close = row['close']
        high = row['high']
        low = row['low']
        mom = row['mom']
        ma = row['ma']
        vol_ratio = row['vol_ratio']
        bar_of_day = row['bar_of_day']
        current_date = row['date']

        mom_thres = self.params['mom_thres']
        vol_mult = self.params['vol_mult']
        stop_pct = self.params['stop_pct']
        target_pct = self.params['target_pct']
        trail_pct = self.params['trail_pct']
        start_bar = self.params['start_bar']
        exit_bar = self.params['exit_bar']
        max_trades = self.params['max_trades']

        if pd.isna(mom) or pd.isna(ma) or pd.isna(vol_ratio):
            return None

        # 新的一天
        if current_date != self.current_date:
            self.current_date = current_date
            self.trades_today = 0
            if self.position != 0:
                self._reset()

        # ========== 持仓管理 ==========
        if self.position != 0:
            # 收盘前强制平仓
            if bar_of_day >= exit_bar:
                self._reset()
                return Signal(action="close", price=close, tag="end_of_day")

            if self.position == 1:
                self.highest = max(self.highest, high)
                pnl_pct = (close - self.entry_price) / self.entry_price * 100

                # 止盈
                if close >= self.target_price:
                    self._reset()
                    return Signal(action="close", price=close, tag="take_profit")

                # 追踪止损 (盈利后启动)
                if pnl_pct > 0:
                    trail_stop = self.highest * (1 - trail_pct / 100)
                    self.stop_price = max(self.stop_price, trail_stop)

                # 止损
                if low <= self.stop_price:
                    self._reset()
                    return Signal(action="close", price=self.stop_price, tag="stop_loss")

            else:  # 空头
                self.lowest = min(self.lowest, low)
                pnl_pct = (self.entry_price - close) / self.entry_price * 100

                if close <= self.target_price:
                    self._reset()
                    return Signal(action="close", price=close, tag="take_profit")

                if pnl_pct > 0:
                    trail_stop = self.lowest * (1 + trail_pct / 100)
                    self.stop_price = min(self.stop_price, trail_stop)

                if high >= self.stop_price:
                    self._reset()
                    return Signal(action="close", price=self.stop_price, tag="stop_loss")

            return None

        # ========== 开仓条件 ==========
        # 时间限制
        if bar_of_day < start_bar or bar_of_day >= exit_bar - 2:
            return None

        # 今日交易次数限制
        if self.trades_today >= max_trades:
            return None

        # 成交量确认
        if vol_ratio < vol_mult:
            return None

        # 多头: 动量向上 + 价格在均线上方
        if mom > mom_thres and close > ma:
            stop = close * (1 - stop_pct / 100)
            target = close * (1 + target_pct / 100)
            self._enter(1, close, stop, target)
            return Signal(action="buy", price=close,
                         stop_loss=close - stop, tag="momentum_long")

        # 空头: 动量向下 + 价格在均线下方
        if mom < -mom_thres and close < ma:
            stop = close * (1 + stop_pct / 100)
            target = close * (1 - target_pct / 100)
            self._enter(-1, close, stop, target)
            return Signal(action="sell", price=close,
                         stop_loss=stop - close, tag="momentum_short")

        return None

    def _enter(self, direction: int, price: float, stop: float, target: float):
        self.position = direction
        self.entry_price = price
        self.stop_price = stop
        self.target_price = target
        self.trades_today += 1
        if direction == 1:
            self.highest = price
            self.lowest = float('inf')
        else:
            self.lowest = price
            self.highest = 0

    def _reset(self):
        self.position = 0
        self.highest = 0
        self.lowest = float('inf')

    def on_trade(self, signal: Signal, is_entry: bool):
        if not is_entry:
            self.position = 0

    def reset(self):
        super().reset()
        self.current_date = None
        self.trades_today = 0
        self.stop_price = 0
        self.target_price = 0
        self.highest = 0
        self.lowest = float('inf')
