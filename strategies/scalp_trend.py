# coding=utf-8
"""
ScalpTrend 15分钟趋势剥头皮策略

专为15分钟级别设计:
1. 快速动量突破 - 5根K线动量
2. 成交量确认 - 突破必须放量
3. 波动率过滤 - 只在波动活跃时交易
4. 快速止盈 - 不贪，达到目标就走
5. 严格止损 - 1.5ATR紧止损

特点: 高频交易，快进快出，适合震荡趋势行情
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, StrategyParam, Signal


class ScalpTrend(BaseStrategy):
    """15分钟剥头皮趋势策略"""

    name = "scalp_trend"
    display_name = "剥头皮趋势"
    description = "15分钟快速趋势策略，高频交易"
    version = "1.0"
    author = "Claude"

    warmup_num = 30

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            # 动量参数
            StrategyParam("mom_len", "动量周期", 5, 3, 10, 1, "int", description="动量计算周期"),
            StrategyParam("mom_thres", "动量阈值%", 0.3, 0.1, 1.0, 0.1, "float", description="突破动量阈值"),

            # 均线过滤
            StrategyParam("ma_len", "均线周期", 20, 10, 40, 5, "int", description="趋势过滤均线"),

            # 成交量
            StrategyParam("vol_len", "量能周期", 10, 5, 20, 5, "int", description="成交量均值周期"),
            StrategyParam("vol_mult", "放量倍数", 1.5, 1.0, 3.0, 0.1, "float", description="入场需放量"),

            # 波动率
            StrategyParam("atr_len", "ATR周期", 14, 7, 20, 1, "int", description="ATR计算周期"),
            StrategyParam("vol_filter", "波动过滤", 0.8, 0.5, 1.5, 0.1, "float", description="ATR需大于均值x倍"),

            # 止损止盈
            StrategyParam("stop_atr", "止损ATR", 1.5, 1.0, 3.0, 0.5, "float", description="紧止损"),
            StrategyParam("tp_atr", "止盈ATR", 2.0, 1.0, 4.0, 0.5, "float", description="止盈目标"),
            StrategyParam("trail_atr", "追踪ATR", 1.0, 0.5, 2.0, 0.5, "float", description="追踪止损"),
            StrategyParam("trail_start", "追踪起始%", 1.0, 0.5, 3.0, 0.5, "float", description="盈利后激活追踪"),

            # 时间过滤
            StrategyParam("max_bars", "最大持仓K线", 20, 10, 50, 5, "int", description="超时强平"),

            # 仓位
            StrategyParam("capital_rate", "资金比例", 0.3, 0.1, 0.8, 0.1, "float"),
            StrategyParam("risk_rate", "风险比例", 0.02, 0.01, 0.05, 0.01, "float"),
        ]

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.stop_price = 0
        self.tp_price = 0
        self.highest = 0
        self.lowest = float('inf')
        self.entry_bar = 0
        self.trailing = False

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        mom_len = self.params['mom_len']
        ma_len = self.params['ma_len']
        vol_len = self.params['vol_len']
        atr_len = self.params['atr_len']

        # 动量 = 收盘价变化率
        df['mom'] = (df['close'] - df['close'].shift(mom_len)) / df['close'].shift(mom_len) * 100

        # 均线
        df['ma'] = df['close'].rolling(ma_len).mean()

        # 成交量
        df['vol_ma'] = df['volume'].rolling(vol_len).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']

        # ATR
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(atr_len).mean()
        df['atr_ma'] = df['atr'].rolling(atr_len * 2).mean()

        # 趋势方向
        df['trend'] = np.where(df['close'] > df['ma'], 1, -1)

        # 前高前低 (用于突破判断)
        df['high_5'] = df['high'].rolling(5).max().shift(1)
        df['low_5'] = df['low'].rolling(5).min().shift(1)

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
        atr = row['atr']
        vol_ratio = row['vol_ratio']
        trend = row['trend']

        mom_thres = self.params['mom_thres']
        vol_mult = self.params['vol_mult']
        vol_filter = self.params['vol_filter']
        stop_atr = self.params['stop_atr']
        tp_atr = self.params['tp_atr']
        trail_atr = self.params['trail_atr']
        trail_start = self.params['trail_start']
        max_bars = self.params['max_bars']

        if pd.isna(atr) or pd.isna(mom) or pd.isna(ma):
            return None

        # ========== 持仓管理 ==========
        if self.position != 0:
            bars_held = idx - self.entry_bar

            if self.position == 1:
                self.highest = max(self.highest, high)
                pnl_pct = (close - self.entry_price) / self.entry_price * 100

                # 止盈
                if close >= self.tp_price:
                    self._reset()
                    return Signal(action="close", price=close, tag="take_profit")

                # 激活追踪
                if pnl_pct >= trail_start and not self.trailing:
                    self.trailing = True

                # 追踪止损
                if self.trailing:
                    new_stop = self.highest - atr * trail_atr
                    self.stop_price = max(self.stop_price, new_stop)

                # 止损
                if low <= self.stop_price:
                    tag = "trailing_stop" if self.trailing else "stop_loss"
                    self._reset()
                    return Signal(action="close", price=self.stop_price, tag=tag)

                # 超时
                if bars_held >= max_bars:
                    self._reset()
                    return Signal(action="close", price=close, tag="timeout")

            else:  # 空头
                self.lowest = min(self.lowest, low)
                pnl_pct = (self.entry_price - close) / self.entry_price * 100

                if close <= self.tp_price:
                    self._reset()
                    return Signal(action="close", price=close, tag="take_profit")

                if pnl_pct >= trail_start and not self.trailing:
                    self.trailing = True

                if self.trailing:
                    new_stop = self.lowest + atr * trail_atr
                    self.stop_price = min(self.stop_price, new_stop)

                if high >= self.stop_price:
                    tag = "trailing_stop" if self.trailing else "stop_loss"
                    self._reset()
                    return Signal(action="close", price=self.stop_price, tag=tag)

                if bars_held >= max_bars:
                    self._reset()
                    return Signal(action="close", price=close, tag="timeout")

            return None

        # ========== 开仓条件 ==========
        # 波动率过滤
        if pd.isna(row['atr_ma']) or atr < row['atr_ma'] * vol_filter:
            return None

        # 成交量确认
        if vol_ratio < vol_mult:
            return None

        # 多头: 动量突破 + 趋势向上 + 突破前高
        if mom > mom_thres and trend == 1 and high > row['high_5']:
            stop = close - atr * stop_atr
            tp = close + atr * tp_atr
            self._enter(1, close, stop, tp, idx)
            return Signal(action="buy", price=close, stop_loss=atr * stop_atr, tag="momentum_long")

        # 空头: 动量突破 + 趋势向下 + 突破前低
        if mom < -mom_thres and trend == -1 and low < row['low_5']:
            stop = close + atr * stop_atr
            tp = close - atr * tp_atr
            self._enter(-1, close, stop, tp, idx)
            return Signal(action="sell", price=close, stop_loss=atr * stop_atr, tag="momentum_short")

        return None

    def _enter(self, direction: int, price: float, stop: float, tp: float, bar: int):
        self.position = direction
        self.entry_price = price
        self.stop_price = stop
        self.tp_price = tp
        self.entry_bar = bar
        self.trailing = False
        if direction == 1:
            self.highest = price
            self.lowest = float('inf')
        else:
            self.lowest = price
            self.highest = 0

    def _reset(self):
        self.position = 0
        self.trailing = False
        self.highest = 0
        self.lowest = float('inf')

    def on_trade(self, signal: Signal, is_entry: bool):
        if not is_entry:
            self.position = 0

    def reset(self):
        super().reset()
        self.stop_price = 0
        self.tp_price = 0
        self.highest = 0
        self.lowest = float('inf')
        self.entry_bar = 0
        self.trailing = False
