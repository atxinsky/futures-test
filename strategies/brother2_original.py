# coding=utf-8
"""
Brother2 原版策略 - 严格还原 brother2.py 的 calc_signal 逻辑

核心特点：
1. 使用递归SMA（不是EMA）计算趋势均线
2. 突破线使用收盘价的N日最高/最低（不是high/low）
3. 双向交易（多空都做）
4. ATR移动止损
5. 无ADX/CHOP/成交量等额外过滤

原版参数（来自数据库配置）：
- BreakPeriod: 突破周期
- AtrPeriod: ATR周期
- LongPeriod: 长周期均线
- ShortPeriod: 短周期均线
- StopLoss: 止损ATR倍数
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .base import BaseStrategy, StrategyParam, Signal


class Brother2OriginalStrategy(BaseStrategy):
    """Brother2 原版策略 - 严格还原"""

    name = "brother2_original"
    display_name = "Brother2 原版(多空双向)"
    description = """
    严格还原 brother2.py 原版策略逻辑：

    **入场条件（做多）：**
    - 短期SMA > 长期SMA（趋势向上）
    - 收盘价 >= 前一日N日收盘价最高点

    **入场条件（做空）：**
    - 短期SMA < 长期SMA（趋势向下）
    - 收盘价 <= 前一日N日收盘价最低点

    **出场机制：**
    - 多头止损：收盘价 <= 持仓后最高价 - ATR × 止损倍数
    - 空头止损：收盘价 >= 持仓后最低价 + ATR × 止损倍数

    **关键区别（vs V6）：**
    - 使用递归SMA，不是EMA
    - 突破线用收盘价，不是最高价
    - 无ADX/CHOP/成交量过滤
    - 支持做空
    """
    version = "1.0"
    author = "原版还原"
    warmup_num = 100

    def __init__(self, params=None):
        super().__init__(params)
        # 持仓状态
        self.record_high = 0  # 持仓期间最高价
        self.record_low = float('inf')  # 持仓期间最低价
        self.entry_atr = 0  # 入场时的ATR值
        self.entry_idx = 0  # 入场K线索引

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            # ========== 原版参数 ==========
            StrategyParam("short_n", "短期均线周期", 10, 5, 20, 1, "int",
                         description="短期SMA周期（原版ShortPeriod）"),
            StrategyParam("long_n", "长期均线周期", 30, 20, 60, 5, "int",
                         description="长期SMA周期（原版LongPeriod）"),
            StrategyParam("break_n", "突破周期", 20, 10, 40, 5, "int",
                         description="N日收盘价高低点突破周期（原版BreakPeriod）"),
            StrategyParam("atr_n", "ATR周期", 20, 10, 30, 1, "int",
                         description="ATR计算周期（原版AtrPeriod）"),
            StrategyParam("stop_n", "止损ATR倍数", 3.0, 1.5, 5.0, 0.5, "float",
                         description="止损距离=ATR×此倍数（原版StopLoss）"),

            # ========== 仓位管理 ==========
            StrategyParam("capital_rate", "资金使用比例", 0.3, 0.1, 0.5, 0.1, "float",
                         description="用于计算仓位的资金比例"),
            StrategyParam("risk_rate", "单次风险比例", 0.02, 0.01, 0.05, 0.005, "float",
                         description="每次交易最大风险占资金比例"),
        ]

    def reset(self):
        """重置状态"""
        super().reset()
        self.record_high = 0
        self.record_low = float('inf')
        self.entry_atr = 0
        self.entry_idx = 0

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        严格按照原版逻辑：使用递归SMA，突破线用收盘价
        """
        df = df.copy()
        p = self.params

        # ========== ATR（使用talib方式或手动计算） ==========
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=p['atr_n']).mean()

        # ========== 递归SMA（原版手动计算方式） ==========
        # 原版代码：
        # df.short_trend[idx] = (df.short_trend[idx - 1] * (short_n - 1) + df.close[idx]) / short_n
        # df.long_trend[idx] = (df.long_trend[idx - 1] * (long_n - 1) + df.close[idx]) / long_n

        short_n = p['short_n']
        long_n = p['long_n']

        short_trend = np.zeros(len(df))
        long_trend = np.zeros(len(df))

        # 初始化第一个值为收盘价
        short_trend[0] = df['close'].iloc[0]
        long_trend[0] = df['close'].iloc[0]

        # 递归计算SMA
        for i in range(1, len(df)):
            short_trend[i] = (short_trend[i-1] * (short_n - 1) + df['close'].iloc[i]) / short_n
            long_trend[i] = (long_trend[i-1] * (long_n - 1) + df['close'].iloc[i]) / long_n

        df['short_trend'] = short_trend
        df['long_trend'] = long_trend

        # ========== 突破线（原版用收盘价，不是high/low） ==========
        df['high_line'] = df['close'].rolling(window=p['break_n']).max()
        df['low_line'] = df['close'].rolling(window=p['break_n']).min()

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """每根K线处理"""
        p = self.params

        # 获取当前数据
        curr_close = df['close'].iloc[idx]
        curr_high = df['high'].iloc[idx]
        curr_low = df['low'].iloc[idx]
        short_trend = df['short_trend'].iloc[idx]
        long_trend = df['long_trend'].iloc[idx]
        high_line_prev = df['high_line'].iloc[idx-1] if idx > 0 else df['high_line'].iloc[idx]
        low_line_prev = df['low_line'].iloc[idx-1] if idx > 0 else df['low_line'].iloc[idx]
        atr = df['atr'].iloc[idx]

        if pd.isna(atr) or atr == 0:
            return None

        # ========== 趋势判断 ==========
        is_bullish = short_trend > long_trend  # 多头趋势
        is_bearish = short_trend < long_trend  # 空头趋势

        # ========== 持仓管理 ==========
        # 多头持仓
        if self.position == 1:
            # 更新持仓期间最高价（用于追踪止损）
            if curr_high > self.record_high:
                self.record_high = curr_high

            # 计算止损线：持仓后最高价 - ATR × stop_n
            # 原版用的是开仓时的ATR：df.atr[pos_idx - 1]
            stop_line = self.record_high - self.entry_atr * p['stop_n']

            # 多头止损
            if curr_close <= stop_line:
                self.position = 0
                old_entry = self.entry_price
                self.entry_price = old_entry  # 保留用于计算盈亏
                self._reset_position_state()
                return Signal("close", curr_close, tag="long_stop")

        # 空头持仓
        elif self.position == -1:
            # 更新持仓期间最低价（用于追踪止损）
            if curr_low < self.record_low:
                self.record_low = curr_low

            # 计算止损线：持仓后最低价 + ATR × stop_n
            stop_line = self.record_low + self.entry_atr * p['stop_n']

            # 空头止损
            if curr_close >= stop_line:
                self.position = 0
                old_entry = self.entry_price
                self.entry_price = old_entry
                self._reset_position_state()
                return Signal("close", curr_close, tag="short_stop")

        # ========== 开仓信号 ==========
        if self.position == 0:
            # 做多信号：趋势向上 + 突破前一日N日最高收盘价
            buy_signal = is_bullish and curr_close >= high_line_prev

            # 做空信号：趋势向下 + 跌破前一日N日最低收盘价
            sell_signal = is_bearish and curr_close <= low_line_prev

            if buy_signal:
                # 开多
                stop_dist = atr * p['stop_n']
                self.position = 1
                self.entry_price = curr_close
                self.record_high = curr_high
                self.record_low = float('inf')
                self.entry_atr = atr
                self.entry_idx = idx
                return Signal("buy", curr_close, 1, "long_breakout", stop_dist)

            elif sell_signal:
                # 开空
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
        """重置持仓相关状态"""
        self.record_high = 0
        self.record_low = float('inf')
        self.entry_atr = 0
        self.entry_idx = 0
