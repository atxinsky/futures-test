# coding=utf-8
"""
海龟交易策略
经典的唐奇安通道突破策略
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .base import BaseStrategy, StrategyParam, Signal


class TurtleStrategy(BaseStrategy):
    """海龟交易策略"""

    name = "turtle"
    display_name = "海龟交易"
    description = """
    经典海龟交易策略：
    - 价格突破N日最高点做多
    - 价格跌破M日最低点平仓
    - 使用ATR计算仓位和止损

    开多条件：收盘价 > N日最高价
    平仓条件：收盘价 < M日最低价 OR 止损触发
    """
    version = "1.0"
    author = "BanBot"
    warmup_num = 60

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("entry_len", "入场周期", 20, 10, 55, 5, "int",
                         description="入场突破周期（唐奇安通道）"),
            StrategyParam("exit_len", "出场周期", 10, 5, 20, 5, "int",
                         description="出场周期"),
            StrategyParam("atr_len", "ATR周期", 20, 10, 30, 5, "int",
                         description="ATR计算周期"),
            StrategyParam("stop_n", "止损ATR倍数", 2.0, 1.0, 4.0, 0.5, "float",
                         description="初始止损=入场价-ATR×倍数"),
            StrategyParam("capital_rate", "资金使用比例", 1.0, 0.1, 1.0, 0.1, "float",
                         description="用于计算仓位的资金比例"),
            StrategyParam("risk_rate", "单次风险比例", 0.02, 0.01, 0.10, 0.01, "float",
                         description="每次交易最大风险占资金比例"),
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        p = self.params

        # 唐奇安通道
        df['entry_high'] = df['high'].rolling(window=p['entry_len']).max()
        df['exit_low'] = df['low'].rolling(window=p['exit_len']).min()

        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=p['atr_len']).mean()

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """每根K线处理"""
        p = self.params

        if idx < 2:
            return None

        curr_close = df['close'].iloc[idx]
        entry_high_prev = df['entry_high'].iloc[idx-1]
        exit_low = df['exit_low'].iloc[idx]
        atr = df['atr'].iloc[idx]

        if pd.isna(entry_high_prev) or pd.isna(exit_low) or pd.isna(atr) or atr == 0:
            return None

        # ========== 持仓检查 ==========
        if self.position == 1:
            # 止损检查
            stop_price = self.entry_price - atr * p['stop_n']
            if curr_close < stop_price:
                self.position = 0
                return Signal("close", curr_close, tag="stop_loss")

            # 出场信号：跌破N日低点
            if curr_close < exit_low:
                self.position = 0
                return Signal("close", curr_close, tag="exit_low")

        # ========== 开仓信号 ==========
        if self.position == 0:
            # 突破N日高点
            if curr_close > entry_high_prev:
                # 计算仓位
                stake_amt = capital * p['capital_rate']
                risk_per_trade = stake_amt * p['risk_rate']
                stop_dist = atr * p['stop_n']

                if stop_dist <= 0:
                    stop_dist = curr_close * 0.02

                volume = max(1, int(risk_per_trade / stop_dist / 300))

                self.position = 1
                self.entry_price = curr_close

                return Signal("buy", curr_close, volume, "breakout_high", stop_dist)

        return None
