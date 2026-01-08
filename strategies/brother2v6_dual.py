# coding=utf-8
"""
Brother2v6 双向版策略
完整的V6过滤逻辑（ADX+CHOP+成交量）+ 双向交易

这是BTC/ETH上表现良好的V6策略，移植到商品期货并支持双向交易
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .base import BaseStrategy, StrategyParam, Signal


class Brother2v6DualStrategy(BaseStrategy):
    """Brother2v6 双向版 - 趋势跟踪"""

    name = "brother2v6_dual"
    display_name = "Brother2v6 双向版"
    description = """
    V6策略移植到商品期货，支持多空双向：

    **入场条件（做多）：**
    - EMA短期 > EMA长期
    - ADX > 阈值
    - CHOP < 50（趋势市场）
    - 收盘价突破N日高点
    - 成交量 > N倍均量

    **入场条件（做空）：**
    - EMA短期 < EMA长期
    - ADX > 阈值
    - CHOP < 50
    - 收盘价跌破N日低点
    - 成交量 > N倍均量

    **出场：**
    - 趋势反转
    - ATR移动止损
    """
    version = "6.0-dual"
    author = "V6双向版"
    warmup_num = 100

    def __init__(self, params=None):
        super().__init__(params)
        self.record_high = 0
        self.record_low = float('inf')
        self.entry_atr = 0

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("sml_len", "短期EMA", 12, 8, 18, 1, "int"),
            StrategyParam("big_len", "长期EMA", 50, 35, 70, 5, "int"),
            StrategyParam("break_len", "突破周期", 30, 20, 45, 5, "int"),
            StrategyParam("atr_len", "ATR周期", 20, 14, 30, 1, "int"),
            StrategyParam("adx_len", "ADX周期", 14, 7, 21, 1, "int"),
            StrategyParam("adx_thres", "ADX阈值", 22.0, 18.0, 28.0, 1.0, "float"),
            StrategyParam("chop_len", "CHOP周期", 14, 10, 20, 1, "int"),
            StrategyParam("chop_thres", "CHOP阈值", 50.0, 45.0, 55.0, 1.0, "float"),
            StrategyParam("vol_len", "均量周期", 20, 15, 30, 1, "int"),
            StrategyParam("vol_multi", "放量倍数", 1.3, 1.1, 2.0, 0.1, "float"),
            StrategyParam("stop_n", "止损ATR倍数", 3.0, 2.0, 4.5, 0.5, "float"),
            StrategyParam("capital_rate", "资金使用比例", 0.3, 0.1, 0.5, 0.1, "float"),
            StrategyParam("risk_rate", "单次风险比例", 0.02, 0.01, 0.05, 0.005, "float"),
        ]

    def reset(self):
        super().reset()
        self.record_high = 0
        self.record_low = float('inf')
        self.entry_atr = 0

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        p = self.params

        # EMA
        df['ema_short'] = df['close'].ewm(span=p['sml_len'], adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=p['big_len'], adjust=False).mean()

        # 突破线（用High和Low）
        df['high_line'] = df['high'].rolling(window=p['break_len']).max()
        df['low_line'] = df['low'].rolling(window=p['break_len']).min()

        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=p['atr_len']).mean()

        # ADX
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        atr_adx = tr.rolling(window=p['adx_len']).mean()
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=p['adx_len']).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=p['adx_len']).mean()

        plus_di = 100 * plus_dm_smooth / (atr_adx + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr_adx + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(window=p['adx_len']).mean()

        # CHOP
        chop_len = p['chop_len']
        high_low_sum = tr.rolling(window=chop_len).sum()
        highest = df['high'].rolling(window=chop_len).max()
        lowest = df['low'].rolling(window=chop_len).min()
        range_hl = highest - lowest
        df['chop'] = 100 * np.log10(high_low_sum / (range_hl + 1e-10)) / np.log10(chop_len)

        # 成交量均线
        df['vol_ma'] = df['volume'].rolling(window=p['vol_len']).mean()

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        p = self.params

        curr_close = df['close'].iloc[idx]
        curr_high = df['high'].iloc[idx]
        curr_low = df['low'].iloc[idx]
        curr_vol = df['volume'].iloc[idx]
        ema_short = df['ema_short'].iloc[idx]
        ema_long = df['ema_long'].iloc[idx]
        high_line_prev = df['high_line'].iloc[idx-1] if idx > 0 else df['high_line'].iloc[idx]
        low_line_prev = df['low_line'].iloc[idx-1] if idx > 0 else df['low_line'].iloc[idx]
        atr = df['atr'].iloc[idx]
        adx = df['adx'].iloc[idx]
        chop = df['chop'].iloc[idx]
        vol_ma = df['vol_ma'].iloc[idx]

        if pd.isna(atr) or pd.isna(adx) or pd.isna(chop) or atr == 0:
            return None

        # 市场状态
        is_bullish = ema_short > ema_long
        is_bearish = ema_short < ema_long
        is_trend_strong = adx > p['adx_thres']
        is_trend_market = chop < p['chop_thres']
        has_vol_confirm = curr_vol > vol_ma * p['vol_multi']
        is_breakout_up = curr_close > high_line_prev
        is_breakout_down = curr_close < low_line_prev

        # 多头持仓
        if self.position == 1:
            if curr_high > self.record_high:
                self.record_high = curr_high

            # 趋势反转
            if is_bearish:
                self.position = 0
                self._reset_position_state()
                return Signal("close", curr_close, tag="trend_reverse")

            # ATR止损
            stop_line = self.record_high - self.entry_atr * p['stop_n']
            if curr_close < stop_line:
                self.position = 0
                self._reset_position_state()
                return Signal("close", curr_close, tag="long_stop")

        # 空头持仓
        elif self.position == -1:
            if curr_low < self.record_low:
                self.record_low = curr_low

            # 趋势反转
            if is_bullish:
                self.position = 0
                self._reset_position_state()
                return Signal("close", curr_close, tag="trend_reverse")

            # ATR止损
            stop_line = self.record_low + self.entry_atr * p['stop_n']
            if curr_close > stop_line:
                self.position = 0
                self._reset_position_state()
                return Signal("close", curr_close, tag="short_stop")

        # 开仓信号
        if self.position == 0:
            # 做多条件
            buy_signal = (is_bullish and is_trend_strong and is_trend_market
                         and is_breakout_up and has_vol_confirm)
            # 做空条件
            sell_signal = (is_bearish and is_trend_strong and is_trend_market
                          and is_breakout_down and has_vol_confirm)

            if buy_signal:
                stop_dist = atr * p['stop_n']
                self.position = 1
                self.entry_price = curr_close
                self.record_high = curr_high
                self.record_low = float('inf')
                self.entry_atr = atr
                return Signal("buy", curr_close, 1, "long_breakout", stop_dist)

            elif sell_signal:
                stop_dist = atr * p['stop_n']
                self.position = -1
                self.entry_price = curr_close
                self.record_high = 0
                self.record_low = curr_low
                self.entry_atr = atr
                return Signal("sell", curr_close, 1, "short_breakout", stop_dist)

        return None

    def _reset_position_state(self):
        self.record_high = 0
        self.record_low = float('inf')
        self.entry_atr = 0
