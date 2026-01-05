# coding=utf-8
"""
Donchian Trend 唐奇安通道趋势策略

经典海龟交易法则的简化版:
1. 突破 N 日高点做多
2. 突破 N 日低点做空
3. ATR 追踪止损
4. 可选 ADX 趋势过滤

这是期货市场经过验证的趋势跟踪策略
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, StrategyParam, Signal


class DonchianTrend(BaseStrategy):
    """唐奇安通道趋势策略"""

    name = "donchian_trend"
    display_name = "唐奇安趋势"
    description = "经典突破策略，适合趋势行情"
    version = "1.0"
    author = "Claude"

    warmup_num = 60

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            # 通道参数
            StrategyParam("entry_len", "入场周期", 20, 10, 60, 5, "int", description="突破N日高低点入场"),
            StrategyParam("exit_len", "出场周期", 10, 5, 30, 5, "int", description="反向突破N日高低点出场"),

            # 过滤
            StrategyParam("use_adx", "使用ADX过滤", 1, 0, 1, 1, "int", description="1=启用ADX过滤"),
            StrategyParam("adx_len", "ADX周期", 14, 7, 30, 1, "int", description="ADX计算周期"),
            StrategyParam("adx_min", "ADX最小值", 20.0, 10.0, 35.0, 1.0, "float", description="ADX阈值"),

            # 止损
            StrategyParam("atr_len", "ATR周期", 20, 10, 30, 5, "int", description="ATR计算周期"),
            StrategyParam("stop_n", "止损ATR", 2.0, 1.0, 4.0, 0.5, "float", description="初始止损ATR倍数"),
            StrategyParam("trail_n", "追踪ATR", 1.5, 0.5, 3.0, 0.5, "float", description="追踪止损ATR倍数"),
            StrategyParam("trail_start", "追踪起始%", 3.0, 1.0, 8.0, 1.0, "float", description="盈利%后激活追踪"),
        ]

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.stop_price = 0
        self.highest_price = 0
        self.lowest_price = float('inf')
        self.trailing_active = False

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()

        entry_len = self.params['entry_len']
        exit_len = self.params['exit_len']
        atr_len = self.params['atr_len']
        adx_len = self.params['adx_len']

        # 唐奇安通道
        df['entry_high'] = df['high'].rolling(entry_len).max().shift(1)  # 入场上轨
        df['entry_low'] = df['low'].rolling(entry_len).min().shift(1)    # 入场下轨
        df['exit_high'] = df['high'].rolling(exit_len).max().shift(1)    # 出场上轨
        df['exit_low'] = df['low'].rolling(exit_len).min().shift(1)      # 出场下轨

        # ATR
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(atr_len).mean()

        # ADX
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_adx = tr.rolling(adx_len).mean()
        plus_di = 100 * (plus_dm.rolling(adx_len).mean() / atr_adx)
        minus_di = 100 * (minus_dm.rolling(adx_len).mean() / atr_adx)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        df['adx'] = dx.rolling(adx_len).mean()

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """每根K线的逻辑"""
        if idx < self.warmup_num:
            return None

        row = df.iloc[idx]
        close = row['close']
        high = row['high']
        low = row['low']
        atr = row['atr']
        adx = row['adx']

        entry_high = row['entry_high']
        entry_low = row['entry_low']
        exit_high = row['exit_high']
        exit_low = row['exit_low']

        use_adx = self.params['use_adx']
        adx_min = self.params['adx_min']
        stop_n = self.params['stop_n']
        trail_n = self.params['trail_n']
        trail_start = self.params['trail_start']

        if pd.isna(atr) or pd.isna(entry_high) or pd.isna(entry_low):
            return None

        # ========== 持仓管理 ==========
        if self.position != 0:
            if self.position == 1:  # 多头
                self.highest_price = max(self.highest_price, high)
                pnl_pct = (close - self.entry_price) / self.entry_price * 100

                # 激活追踪止损
                if pnl_pct >= trail_start and not self.trailing_active:
                    self.trailing_active = True

                # 更新追踪止损
                if self.trailing_active:
                    new_stop = self.highest_price - atr * trail_n
                    self.stop_price = max(self.stop_price, new_stop)

                # 止损触发
                if low <= self.stop_price:
                    tag = "trailing_stop" if self.trailing_active else "stop_loss"
                    self._reset_state()
                    return Signal(action="close", price=self.stop_price, tag=tag)

                # 反向突破出场 (跌破 exit_len 日低点)
                if low < exit_low:
                    self._reset_state()
                    return Signal(action="close", price=exit_low, tag="exit_break")

            else:  # 空头
                self.lowest_price = min(self.lowest_price, low)
                pnl_pct = (self.entry_price - close) / self.entry_price * 100

                if pnl_pct >= trail_start and not self.trailing_active:
                    self.trailing_active = True

                if self.trailing_active:
                    new_stop = self.lowest_price + atr * trail_n
                    self.stop_price = min(self.stop_price, new_stop)

                if high >= self.stop_price:
                    tag = "trailing_stop" if self.trailing_active else "stop_loss"
                    self._reset_state()
                    return Signal(action="close", price=self.stop_price, tag=tag)

                # 反向突破出场 (突破 exit_len 日高点)
                if high > exit_high:
                    self._reset_state()
                    return Signal(action="close", price=exit_high, tag="exit_break")

            return None

        # ========== 开仓逻辑 ==========
        # ADX 过滤
        if use_adx and (pd.isna(adx) or adx < adx_min):
            return None

        # 多头入场: 突破 entry_len 日高点
        if high > entry_high:
            stop = close - atr * stop_n
            self._setup_long(close, stop)
            return Signal(action="buy", price=close, stop_loss=stop, tag="breakout_long")

        # 空头入场: 突破 entry_len 日低点
        if low < entry_low:
            stop = close + atr * stop_n
            self._setup_short(close, stop)
            return Signal(action="sell", price=close, stop_loss=stop, tag="breakout_short")

        return None

    def _setup_long(self, price: float, stop: float):
        self.position = 1
        self.entry_price = price
        self.stop_price = stop
        self.highest_price = price
        self.lowest_price = float('inf')
        self.trailing_active = False

    def _setup_short(self, price: float, stop: float):
        self.position = -1
        self.entry_price = price
        self.stop_price = stop
        self.lowest_price = price
        self.highest_price = 0
        self.trailing_active = False

    def _reset_state(self):
        """平仓前重置状态"""
        self.position = 0
        self.trailing_active = False
        self.highest_price = 0
        self.lowest_price = float('inf')

    def on_trade(self, signal: Signal, is_entry: bool):
        if is_entry:
            self.position = 1 if signal.action == "buy" else -1
        else:
            self.position = 0
            self.trailing_active = False

    def reset(self):
        super().reset()
        self.stop_price = 0
        self.highest_price = 0
        self.lowest_price = float('inf')
        self.trailing_active = False
