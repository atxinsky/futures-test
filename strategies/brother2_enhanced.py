# coding=utf-8
"""
Brother2 增强版策略 - 针对商品期货优化

改进点：
1. 保留原版核心逻辑（SMA趋势 + 突破）
2. 增加ADX过滤弱趋势
3. 调整止损逻辑：使用更宽松的初始止损，避免频繁被打掉
4. 双向交易优化：根据趋势强度选择做多还是做空
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .base import BaseStrategy, StrategyParam, Signal


class Brother2EnhancedStrategy(BaseStrategy):
    """Brother2 增强版 - 商品期货优化"""

    name = "brother2_enhanced"
    display_name = "Brother2 增强版(期货优化)"
    description = """
    基于原版Brother2的增强策略，针对商品期货优化：

    **入场条件（做多）：**
    - 短期SMA > 长期SMA
    - ADX > 阈值（趋势强度确认）
    - 收盘价 >= 前一日N日收盘价最高点

    **入场条件（做空）：**
    - 短期SMA < 长期SMA
    - ADX > 阈值
    - 收盘价 <= 前一日N日收盘价最低点

    **出场机制：**
    - 趋势反转：短期SMA与长期SMA交叉
    - ATR移动止损
    """
    version = "2.0"
    author = "优化版"
    warmup_num = 100

    def __init__(self, params=None):
        super().__init__(params)
        self.record_high = 0
        self.record_low = float('inf')
        self.entry_atr = 0
        self.entry_idx = 0

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("short_n", "短期均线周期", 12, 8, 20, 1, "int"),
            StrategyParam("long_n", "长期均线周期", 50, 30, 80, 5, "int"),
            StrategyParam("break_n", "突破周期", 30, 15, 45, 5, "int"),
            StrategyParam("atr_n", "ATR周期", 20, 14, 30, 1, "int"),
            StrategyParam("adx_n", "ADX周期", 14, 10, 21, 1, "int"),
            StrategyParam("adx_thres", "ADX阈值", 20.0, 15.0, 30.0, 1.0, "float"),
            StrategyParam("stop_n", "止损ATR倍数", 3.5, 2.0, 5.0, 0.5, "float"),
            StrategyParam("capital_rate", "资金使用比例", 0.3, 0.1, 0.5, 0.1, "float"),
            StrategyParam("risk_rate", "单次风险比例", 0.02, 0.01, 0.05, 0.005, "float"),
        ]

    def reset(self):
        super().reset()
        self.record_high = 0
        self.record_low = float('inf')
        self.entry_atr = 0
        self.entry_idx = 0

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=p['atr_n']).mean()

        # 递归SMA
        short_n = p['short_n']
        long_n = p['long_n']

        short_trend = np.zeros(len(df))
        long_trend = np.zeros(len(df))

        short_trend[0] = df['close'].iloc[0]
        long_trend[0] = df['close'].iloc[0]

        for i in range(1, len(df)):
            short_trend[i] = (short_trend[i-1] * (short_n - 1) + df['close'].iloc[i]) / short_n
            long_trend[i] = (long_trend[i-1] * (long_n - 1) + df['close'].iloc[i]) / long_n

        df['short_trend'] = short_trend
        df['long_trend'] = long_trend

        # 突破线
        df['high_line'] = df['close'].rolling(window=p['break_n']).max()
        df['low_line'] = df['close'].rolling(window=p['break_n']).min()

        # ADX
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        atr_adx = tr.rolling(window=p['adx_n']).mean()
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=p['adx_n']).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=p['adx_n']).mean()

        plus_di = 100 * plus_dm_smooth / (atr_adx + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr_adx + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(window=p['adx_n']).mean()

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        p = self.params

        curr_close = df['close'].iloc[idx]
        curr_high = df['high'].iloc[idx]
        curr_low = df['low'].iloc[idx]
        short_trend = df['short_trend'].iloc[idx]
        long_trend = df['long_trend'].iloc[idx]
        high_line_prev = df['high_line'].iloc[idx-1] if idx > 0 else df['high_line'].iloc[idx]
        low_line_prev = df['low_line'].iloc[idx-1] if idx > 0 else df['low_line'].iloc[idx]
        atr = df['atr'].iloc[idx]
        adx = df['adx'].iloc[idx]

        if pd.isna(atr) or pd.isna(adx) or atr == 0:
            return None

        is_bullish = short_trend > long_trend
        is_bearish = short_trend < long_trend
        is_trend_strong = adx > p['adx_thres']

        # 多头持仓
        if self.position == 1:
            if curr_high > self.record_high:
                self.record_high = curr_high

            # 趋势反转出场
            if is_bearish:
                self.position = 0
                self._reset_position_state()
                return Signal("close", curr_close, tag="trend_reverse")

            # ATR止损
            stop_line = self.record_high - self.entry_atr * p['stop_n']
            if curr_close <= stop_line:
                self.position = 0
                self._reset_position_state()
                return Signal("close", curr_close, tag="long_stop")

        # 空头持仓
        elif self.position == -1:
            if curr_low < self.record_low:
                self.record_low = curr_low

            # 趋势反转出场
            if is_bullish:
                self.position = 0
                self._reset_position_state()
                return Signal("close", curr_close, tag="trend_reverse")

            # ATR止损
            stop_line = self.record_low + self.entry_atr * p['stop_n']
            if curr_close >= stop_line:
                self.position = 0
                self._reset_position_state()
                return Signal("close", curr_close, tag="short_stop")

        # 开仓信号
        if self.position == 0:
            buy_signal = is_bullish and is_trend_strong and curr_close >= high_line_prev
            sell_signal = is_bearish and is_trend_strong and curr_close <= low_line_prev

            if buy_signal:
                stop_dist = atr * p['stop_n']
                self.position = 1
                self.entry_price = curr_close
                self.record_high = curr_high
                self.record_low = float('inf')
                self.entry_atr = atr
                self.entry_idx = idx
                return Signal("buy", curr_close, 1, "long_breakout", stop_dist)

            elif sell_signal:
                stop_dist = atr * p['stop_n']
                self.position = -1
                self.entry_price = curr_close
                self.record_high = 0
                self.record_low = curr_low
                self.entry_atr = atr
                self.entry_idx = idx
                return Signal("sell", curr_close, 1, "short_breakout", stop_dist)

        return None

    def _reset_position_state(self):
        self.record_high = 0
        self.record_low = float('inf')
        self.entry_atr = 0
        self.entry_idx = 0
