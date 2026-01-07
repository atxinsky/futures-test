# coding=utf-8
"""
WaveTrend V2 策略
改进版：加入趋势过滤 + 优化止损逻辑

核心改进：
1. 加入EMA趋势过滤，只顺势交易
2. 更宽松的止损，减少被震出
3. 加入时间止损，避免无效持仓
4. 超买区主动止盈，不等死叉
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from strategies.base import BaseStrategy, StrategyParam, Signal


class WaveTrendV2Strategy(BaseStrategy):
    """WaveTrend V2策略"""

    name = "wavetrend_v2"
    display_name = "WaveTrend V2 (趋势过滤)"
    description = "超卖区金叉+趋势过滤做多，超买止盈，ATR止损"
    version = "2.0"
    author = "Eric"

    warmup_num = 60

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("n1", "通道长度", 10, 5, 20, 1, "int"),
            StrategyParam("n2", "平均长度", 21, 10, 30, 1, "int"),
            StrategyParam("ob_level", "超买阈值", 53, 40, 70, 1, "int"),
            StrategyParam("os_level", "超卖阈值", -53, -70, -40, 1, "int"),
            StrategyParam("atr_len", "ATR周期", 14, 7, 30, 1, "int"),
            StrategyParam("stop_atr", "止损ATR倍数", 3.0, 1.5, 5.0, 0.5, "float"),
            StrategyParam("ema_len", "趋势EMA周期", 50, 20, 100, 10, "int"),
            StrategyParam("use_trend_filter", "启用趋势过滤", True, param_type="bool"),
            StrategyParam("profit_target", "止盈WT阈值", 40, 20, 60, 5, "int", description="WT1超过此值开始移动止盈"),
            StrategyParam("max_hold_bars", "最大持仓K线数", 50, 20, 100, 10, "int"),
            StrategyParam("only_long", "只做多", True, param_type="bool"),
            StrategyParam("capital_rate", "资金使用率", 1.0, 0.5, 1.0, 0.1, "float"),
            StrategyParam("risk_rate", "单笔风险", 0.02, 0.01, 0.05, 0.01, "float"),
        ]

    def __init__(self, params=None):
        super().__init__(params)
        self.hold_bars = 0
        self.triggered_profit = False

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指标"""
        df = df.copy()

        n1 = self.params['n1']
        n2 = self.params['n2']
        atr_len = self.params['atr_len']
        ema_len = self.params['ema_len']

        # WaveTrend
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        df['esa'] = df['hlc3'].ewm(span=n1, adjust=False).mean()
        df['d'] = (df['hlc3'] - df['esa']).abs().ewm(span=n1, adjust=False).mean()
        df['ci'] = np.where(df['d'] > 0, (df['hlc3'] - df['esa']) / (0.015 * df['d']), 0)
        df['wt1'] = df['ci'].ewm(span=n2, adjust=False).mean()
        df['wt2'] = df['wt1'].rolling(4).mean()

        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(window=atr_len).mean()

        # 趋势EMA
        df['ema_trend'] = df['close'].ewm(span=ema_len, adjust=False).mean()

        # 信号
        df['cross_up'] = (df['wt1'] > df['wt2']) & (df['wt1'].shift(1) <= df['wt2'].shift(1))
        df['cross_down'] = (df['wt1'] < df['wt2']) & (df['wt1'].shift(1) >= df['wt2'].shift(1))

        return df

    def reset(self):
        super().reset()
        self.hold_bars = 0
        self.triggered_profit = False

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """每根K线处理"""

        curr_close = df['close'].iloc[idx]
        curr_high = df['high'].iloc[idx]
        curr_low = df['low'].iloc[idx]
        wt1 = df['wt1'].iloc[idx]
        wt2 = df['wt2'].iloc[idx]
        atr = df['atr'].iloc[idx]
        ema_trend = df['ema_trend'].iloc[idx]
        cross_up = df['cross_up'].iloc[idx]
        cross_down = df['cross_down'].iloc[idx]

        ob_level = self.params['ob_level']
        os_level = self.params['os_level']
        stop_atr = self.params['stop_atr']
        use_trend_filter = self.params['use_trend_filter']
        profit_target = self.params['profit_target']
        max_hold_bars = self.params['max_hold_bars']
        only_long = self.params['only_long']

        if pd.isna(wt1) or pd.isna(wt2) or pd.isna(atr) or atr == 0 or pd.isna(ema_trend):
            return None

        # ========== 多头持仓检查 ==========
        if self.position == 1:
            self.hold_bars += 1

            # 更新最高价
            if curr_high > self.record_high:
                self.record_high = curr_high

            exit_tag = None

            # 1. 基础止损（更宽松）
            stop_price = self.entry_price - atr * stop_atr
            if curr_close < stop_price:
                exit_tag = "stop_loss"

            # 2. 超买区止盈
            if wt1 > ob_level:
                self.triggered_profit = True

            # 触发止盈后，回落止盈
            if self.triggered_profit and wt1 < profit_target:
                exit_tag = "profit_take"

            # 3. 移动止盈（只在盈利后启用）
            if curr_close > self.entry_price * 1.02:  # 盈利2%以上
                trailing_stop = self.record_high - atr * 2.0
                if curr_close < trailing_stop:
                    exit_tag = "trailing_profit"

            # 4. 时间止损
            if self.hold_bars >= max_hold_bars:
                exit_tag = "time_stop"

            # 5. 趋势反转
            if use_trend_filter and curr_close < ema_trend and df['close'].iloc[idx-1] >= df['ema_trend'].iloc[idx-1]:
                if curr_close < self.entry_price:  # 只在亏损时退出
                    exit_tag = "trend_break"

            if exit_tag:
                self.position = 0
                self.hold_bars = 0
                self.triggered_profit = False
                return Signal(
                    action="close",
                    price=curr_close,
                    tag=exit_tag,
                    stop_loss=0
                )

        # ========== 空头持仓检查 ==========
        elif self.position == -1:
            self.hold_bars += 1

            if curr_low < self.record_low:
                self.record_low = curr_low

            exit_tag = None

            # 止损
            stop_price = self.entry_price + atr * stop_atr
            if curr_close > stop_price:
                exit_tag = "stop_loss"

            # 超卖止盈
            if wt1 < os_level:
                self.triggered_profit = True
            if self.triggered_profit and wt1 > -profit_target:
                exit_tag = "profit_take"

            # 移动止盈
            if curr_close < self.entry_price * 0.98:
                trailing_stop = self.record_low + atr * 2.0
                if curr_close > trailing_stop:
                    exit_tag = "trailing_profit"

            # 时间止损
            if self.hold_bars >= max_hold_bars:
                exit_tag = "time_stop"

            # 趋势反转
            if use_trend_filter and curr_close > ema_trend and df['close'].iloc[idx-1] <= df['ema_trend'].iloc[idx-1]:
                if curr_close > self.entry_price:
                    exit_tag = "trend_break"

            if exit_tag:
                self.position = 0
                self.hold_bars = 0
                self.triggered_profit = False
                return Signal(
                    action="close",
                    price=curr_close,
                    tag=exit_tag,
                    stop_loss=0
                )

        # ========== 开仓信号 ==========
        if self.position == 0:
            # 趋势过滤
            is_uptrend = curr_close > ema_trend if use_trend_filter else True
            is_downtrend = curr_close < ema_trend if use_trend_filter else True

            # 超卖区金叉 + 趋势向上 -> 做多
            if wt1 < os_level and cross_up and is_uptrend:
                self.position = 1
                self.entry_price = curr_close
                self.record_high = curr_high
                self.hold_bars = 0
                self.triggered_profit = False
                stop_loss = curr_close - atr * stop_atr

                return Signal(
                    action="buy",
                    price=curr_close,
                    tag="oversold_buy",
                    stop_loss=stop_loss
                )

            # 超买区死叉 + 趋势向下 -> 做空
            if not only_long and wt1 > ob_level and cross_down and is_downtrend:
                self.position = -1
                self.entry_price = curr_close
                self.record_low = curr_low
                self.hold_bars = 0
                self.triggered_profit = False
                stop_loss = curr_close + atr * stop_atr

                return Signal(
                    action="sell",
                    price=curr_close,
                    tag="overbought_sell",
                    stop_loss=stop_loss
                )

        return None
