# coding=utf-8
"""
emanew V5 策略 - 分批止盈版

核心改进：分批止盈，解决"卖飞焦虑"问题

交易心理学背景：
- 问题：卖出后行情继续涨 → 懊恼 → 追高杀回 → 顶部震荡止损 → 心态资金双回撤
- 解决：分批止盈，第一次锁定利润消除焦虑，剩余仓位让利润奔跑

分批止盈逻辑：
- 第一次止盈（平仓50%）：盈利≥15% 且 从高点回撤≥6%
- 第二次止盈（平仓剩余）：从高点回撤≥12% 或 EMA死叉

其他止损机制：
- 信号K线止损：连续3天收盘低于金叉K线最低价
- 固定止损：亏损超过8%立即出场
- 保本止损：盈利超10%后跌回入场价
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .base import BaseStrategy, StrategyParam, Signal


class EmaNewV5Strategy(BaseStrategy):
    """emanew V5 策略 - 分批止盈"""

    name = "emanew_v5"
    display_name = "EmaNew V5 分批止盈"
    description = "EMA金叉+MACD动量确认，分批止盈解决卖飞焦虑"
    version = "5.0"
    author = "atxinsky"
    warmup_num = 50

    def __init__(self, params=None):
        super().__init__(params)
        # 持仓状态
        self.high_since = 0
        self.signal_low = 0
        self.days_below_signal = 0
        self.has_partial_exit = False
        self.high_after_partial = 0
        self.current_shares = 1.0

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("ema_fast", "快速EMA", 9, 5, 20, 1, "int", description="快速EMA周期"),
            StrategyParam("ema_slow", "慢速EMA", 21, 10, 50, 1, "int", description="慢速EMA周期"),
            StrategyParam("ma_len", "MA周期", 20, 10, 60, 1, "int", description="均线周期"),
            StrategyParam("macd_fast", "MACD快线", 12, 5, 20, 1, "int"),
            StrategyParam("macd_slow", "MACD慢线", 26, 15, 40, 1, "int"),
            StrategyParam("macd_smooth", "MACD信号", 9, 5, 15, 1, "int"),
            StrategyParam("stop_loss", "固定止损%", 0.08, 0.03, 0.15, 0.01, "float", description="最大亏损比例"),
            StrategyParam("break_even", "保本触发%", 0.10, 0.05, 0.20, 0.01, "float"),
            StrategyParam("partial_trigger", "首次止盈触发%", 0.15, 0.10, 0.30, 0.01, "float"),
            StrategyParam("partial_drawdown", "首次止盈回撤%", 0.06, 0.03, 0.10, 0.01, "float"),
            StrategyParam("partial_rate", "首次平仓比例", 0.50, 0.30, 0.70, 0.05, "float"),
            StrategyParam("full_drawdown", "完全止盈回撤%", 0.12, 0.08, 0.20, 0.01, "float"),
            StrategyParam("signal_low_days", "信号K止损天数", 3, 1, 5, 1, "int"),
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        p = self.params

        # EMA
        df['ema_fast'] = df['close'].ewm(span=p['ema_fast'], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=p['ema_slow'], adjust=False).mean()

        # MA
        df['ma'] = df['close'].rolling(window=p['ma_len']).mean()

        # MACD
        ema_fast = df['close'].ewm(span=p['macd_fast'], adjust=False).mean()
        ema_slow = df['close'].ewm(span=p['macd_slow'], adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=p['macd_smooth'], adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # EMA交叉信号
        df['ema_cross'] = 0
        for i in range(1, len(df)):
            if df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] <= df['ema_slow'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('ema_cross')] = 1  # 金叉
            elif df['ema_fast'].iloc[i] < df['ema_slow'].iloc[i] and df['ema_fast'].iloc[i-1] >= df['ema_slow'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('ema_cross')] = -1  # 死叉

        return df

    def reset(self):
        """重置状态"""
        super().reset()
        self.high_since = 0
        self.signal_low = 0
        self.days_below_signal = 0
        self.has_partial_exit = False
        self.high_after_partial = 0
        self.current_shares = 1.0

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """每根K线调用"""
        if idx < 1:
            return None

        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]
        price = row['close']
        p = self.params

        # 检查出场
        if self.position == 1 and self.current_shares > 0:
            # 更新最高价
            if price > self.high_since:
                self.high_since = price
            if self.has_partial_exit and price > self.high_after_partial:
                self.high_after_partial = price

            # 计算盈亏
            profit_rate = (price - self.entry_price) / self.entry_price
            max_profit_rate = (self.high_since - self.entry_price) / self.entry_price
            drawdown_from_high = (self.high_since - price) / self.high_since if self.high_since > 0 else 0

            # 信号K线止损检查
            if price < self.signal_low:
                self.days_below_signal += 1
            else:
                self.days_below_signal = 0

            # 信号K线止损
            if self.days_below_signal >= p['signal_low_days']:
                self.position = 0
                self.current_shares = 0
                return Signal("close", price, tag="signal_low_break")

            # 固定止损
            if profit_rate <= -p['stop_loss']:
                self.position = 0
                self.current_shares = 0
                return Signal("close", price, tag="stop_loss")

            # 分批止盈
            if not self.has_partial_exit:
                # 第一次止盈
                if max_profit_rate >= p['partial_trigger'] and drawdown_from_high >= p['partial_drawdown']:
                    self.has_partial_exit = True
                    self.high_after_partial = price
                    self.current_shares *= (1 - p['partial_rate'])
                    return Signal("close", price, tag="partial_stop")
            else:
                # 第二次止盈
                if self.high_after_partial > 0:
                    drawdown_after = (self.high_after_partial - price) / self.high_after_partial
                    if drawdown_after >= p['full_drawdown']:
                        self.position = 0
                        self.current_shares = 0
                        return Signal("close", price, tag="trail_stop_full")

            # 保本止损
            if max_profit_rate >= p['break_even'] and profit_rate <= 0:
                self.position = 0
                self.current_shares = 0
                return Signal("close", price, tag="break_even")

            # 死叉出场
            if row['ema_cross'] == -1 and price < row['ma']:
                self.position = 0
                self.current_shares = 0
                return Signal("close", price, tag="death_cross")

        # 检查入场
        if self.position == 0:
            # 入场条件：EMA金叉 + MACD柱>0且增强 + 收盘>MA
            if (row['ema_cross'] == 1 and
                row['macd_hist'] > 0 and
                (row['macd_hist'] > prev_row['macd_hist'] or prev_row['macd_hist'] < 0) and
                price > row['ma']):

                self.position = 1
                self.entry_price = price
                self.entry_time = row.get('time', idx)
                self.high_since = price
                self.signal_low = row['low']
                self.days_below_signal = 0
                self.has_partial_exit = False
                self.high_after_partial = 0
                self.current_shares = 1.0

                return Signal("buy", price, tag="ema_golden_cross")

        return None
