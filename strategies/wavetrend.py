# coding=utf-8
"""
WaveTrend 策略
基于经典WaveTrend指标的超买超卖交易策略

核心逻辑：
- 超卖区（WT1 < -53）金叉买入
- 超买区（WT1 > 53）死叉卖出/做空
- ATR移动止损
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from strategies.base import BaseStrategy, StrategyParam, Signal


class WaveTrendStrategy(BaseStrategy):
    """WaveTrend策略"""

    name = "wavetrend"
    display_name = "WaveTrend超买超卖"
    description = "超卖区金叉做多，超买区死叉做空，ATR止损"
    version = "1.0"
    author = "Eric"

    warmup_num = 50

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("n1", "通道长度", 10, 5, 20, 1, "int", description="ESA计算周期"),
            StrategyParam("n2", "平均长度", 21, 10, 30, 1, "int", description="TCI平滑周期"),
            StrategyParam("ob_level", "超买阈值", 53, 40, 70, 1, "int", description="超买区域阈值"),
            StrategyParam("os_level", "超卖阈值", -53, -70, -40, 1, "int", description="超卖区域阈值"),
            StrategyParam("atr_len", "ATR周期", 14, 7, 30, 1, "int", description="ATR计算周期"),
            StrategyParam("stop_atr", "止损ATR倍数", 2.0, 1.0, 4.0, 0.5, "float", description="止损距离=ATR*倍数"),
            StrategyParam("only_long", "只做多", True, param_type="bool", description="只做多头，不做空"),
            StrategyParam("capital_rate", "资金使用率", 1.0, 0.5, 1.0, 0.1, "float"),
            StrategyParam("risk_rate", "单笔风险", 0.02, 0.01, 0.05, 0.01, "float"),
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算WaveTrend和ATR指标"""
        df = df.copy()

        n1 = self.params['n1']
        n2 = self.params['n2']
        atr_len = self.params['atr_len']

        # HLC3
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3

        # ESA = EMA(HLC3, n1)
        df['esa'] = df['hlc3'].ewm(span=n1, adjust=False).mean()

        # D = EMA(|HLC3 - ESA|, n1)
        df['d'] = (df['hlc3'] - df['esa']).abs().ewm(span=n1, adjust=False).mean()

        # CI = (HLC3 - ESA) / (0.015 * D)
        df['ci'] = np.where(df['d'] > 0, (df['hlc3'] - df['esa']) / (0.015 * df['d']), 0)

        # WT1 = EMA(CI, n2)
        df['wt1'] = df['ci'].ewm(span=n2, adjust=False).mean()

        # WT2 = SMA(WT1, 4)
        df['wt2'] = df['wt1'].rolling(4).mean()

        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(window=atr_len).mean()

        # 信号预计算
        df['cross_up'] = (df['wt1'] > df['wt2']) & (df['wt1'].shift(1) <= df['wt2'].shift(1))
        df['cross_down'] = (df['wt1'] < df['wt2']) & (df['wt1'].shift(1) >= df['wt2'].shift(1))

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """每根K线处理"""

        curr_close = df['close'].iloc[idx]
        curr_high = df['high'].iloc[idx]
        curr_low = df['low'].iloc[idx]
        wt1 = df['wt1'].iloc[idx]
        wt2 = df['wt2'].iloc[idx]
        atr = df['atr'].iloc[idx]
        cross_up = df['cross_up'].iloc[idx]
        cross_down = df['cross_down'].iloc[idx]

        ob_level = self.params['ob_level']
        os_level = self.params['os_level']
        stop_atr = self.params['stop_atr']
        only_long = self.params['only_long']

        if pd.isna(wt1) or pd.isna(wt2) or pd.isna(atr) or atr == 0:
            return None

        # ========== 持仓检查 ==========
        if self.position == 1:  # 多头持仓
            # 更新最高价
            if curr_high > self.record_high:
                self.record_high = curr_high

            exit_tag = None

            # 移动止损
            stop_price = self.record_high - atr * stop_atr
            if curr_close < stop_price:
                exit_tag = "trailing_stop"

            # 超买死叉止盈
            if wt1 > ob_level and cross_down:
                exit_tag = "overbought_cross"

            # WT1回到零轴下方止损
            if wt1 < 0 and df['wt1'].iloc[idx-1] >= 0:
                exit_tag = "zero_cross_down"

            if exit_tag:
                self.position = 0
                return Signal(
                    action="close",
                    price=curr_close,
                    tag=exit_tag,
                    stop_loss=0
                )

        elif self.position == -1:  # 空头持仓
            # 更新最低价
            if curr_low < self.record_low:
                self.record_low = curr_low

            exit_tag = None

            # 移动止损（空头）
            stop_price = self.record_low + atr * stop_atr
            if curr_close > stop_price:
                exit_tag = "trailing_stop"

            # 超卖金叉止盈
            if wt1 < os_level and cross_up:
                exit_tag = "oversold_cross"

            # WT1回到零轴上方止损
            if wt1 > 0 and df['wt1'].iloc[idx-1] <= 0:
                exit_tag = "zero_cross_up"

            if exit_tag:
                self.position = 0
                return Signal(
                    action="close",
                    price=curr_close,
                    tag=exit_tag,
                    stop_loss=0
                )

        # ========== 开仓信号 ==========
        if self.position == 0:
            # 超卖区金叉 -> 做多
            if wt1 < os_level and cross_up:
                self.position = 1
                self.entry_price = curr_close
                self.record_high = curr_high
                stop_loss = curr_close - atr * stop_atr

                return Signal(
                    action="buy",
                    price=curr_close,
                    tag="oversold_buy",
                    stop_loss=stop_loss
                )

            # 超买区死叉 -> 做空（如果允许）
            if not only_long and wt1 > ob_level and cross_down:
                self.position = -1
                self.entry_price = curr_close
                self.record_low = curr_low
                stop_loss = curr_close + atr * stop_atr

                return Signal(
                    action="sell",
                    price=curr_close,
                    tag="overbought_sell",
                    stop_loss=stop_loss
                )

        return None
