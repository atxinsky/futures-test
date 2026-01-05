# coding=utf-8
"""
MeanRevert 均值回归策略

思路：价格偏离均线过多后必然回归
1. RSI超买超卖判断极端位置
2. 布林带突破作为反转信号
3. 快速止盈 - 回归到均线即可
4. 严格止损 - 错了就跑

适合: 震荡行情，15分钟级别
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, StrategyParam, Signal


class MeanRevert(BaseStrategy):
    """均值回归策略"""

    name = "mean_revert"
    display_name = "均值回归"
    description = "超买超卖反转，快进快出"
    version = "1.0"
    author = "Claude"

    warmup_num = 30

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            # RSI参数
            StrategyParam("rsi_len", "RSI周期", 7, 5, 14, 1, "int", description="RSI计算周期"),
            StrategyParam("rsi_ob", "超买阈值", 75, 65, 85, 5, "int", description="RSI超买"),
            StrategyParam("rsi_os", "超卖阈值", 25, 15, 35, 5, "int", description="RSI超卖"),

            # 布林带
            StrategyParam("bb_len", "布林周期", 20, 10, 30, 5, "int", description="布林带周期"),
            StrategyParam("bb_std", "布林倍数", 2.0, 1.5, 3.0, 0.5, "float", description="标准差倍数"),

            # 趋势过滤 (可选)
            StrategyParam("use_trend", "趋势过滤", 0, 0, 1, 1, "int", description="1=顺势交易"),
            StrategyParam("trend_len", "趋势周期", 50, 30, 100, 10, "int", description="趋势均线"),

            # 止损止盈
            StrategyParam("atr_len", "ATR周期", 14, 7, 20, 1, "int"),
            StrategyParam("stop_atr", "止损ATR", 1.5, 1.0, 3.0, 0.5, "float"),
            StrategyParam("tp_type", "止盈方式", 1, 0, 1, 1, "int", description="0=ATR, 1=回归均线"),
            StrategyParam("tp_atr", "止盈ATR", 1.5, 1.0, 3.0, 0.5, "float"),

            # 过滤
            StrategyParam("vol_len", "量能周期", 10, 5, 20, 5, "int"),
            StrategyParam("vol_mult", "放量确认", 1.2, 1.0, 2.0, 0.1, "float"),

            # 仓位
            StrategyParam("capital_rate", "资金比例", 0.3, 0.1, 0.8, 0.1, "float"),
            StrategyParam("risk_rate", "风险比例", 0.02, 0.01, 0.05, 0.01, "float"),
        ]

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.stop_price = 0
        self.tp_price = 0
        self.entry_ma = 0  # 入场时的均线位置

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        rsi_len = self.params['rsi_len']
        bb_len = self.params['bb_len']
        bb_std = self.params['bb_std']
        trend_len = self.params['trend_len']
        atr_len = self.params['atr_len']
        vol_len = self.params['vol_len']

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(rsi_len).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(rsi_len).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # 布林带
        df['bb_mid'] = df['close'].rolling(bb_len).mean()
        bb_std_val = df['close'].rolling(bb_len).std()
        df['bb_up'] = df['bb_mid'] + bb_std * bb_std_val
        df['bb_dn'] = df['bb_mid'] - bb_std * bb_std_val

        # 趋势均线
        df['trend_ma'] = df['close'].rolling(trend_len).mean()
        df['trend'] = np.where(df['close'] > df['trend_ma'], 1, -1)

        # ATR
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(atr_len).mean()

        # 成交量
        df['vol_ma'] = df['volume'].rolling(vol_len).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']

        # 信号辅助
        df['prev_rsi'] = df['rsi'].shift(1)
        df['prev_close'] = df['close'].shift(1)

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        if idx < self.warmup_num:
            return None

        row = df.iloc[idx]

        close = row['close']
        high = row['high']
        low = row['low']
        rsi = row['rsi']
        prev_rsi = row['prev_rsi']
        bb_mid = row['bb_mid']
        bb_up = row['bb_up']
        bb_dn = row['bb_dn']
        atr = row['atr']
        trend = row['trend']
        vol_ratio = row['vol_ratio']

        rsi_ob = self.params['rsi_ob']
        rsi_os = self.params['rsi_os']
        use_trend = self.params['use_trend']
        stop_atr = self.params['stop_atr']
        tp_type = self.params['tp_type']
        tp_atr = self.params['tp_atr']
        vol_mult = self.params['vol_mult']

        if pd.isna(atr) or pd.isna(rsi) or pd.isna(bb_mid):
            return None

        # ========== 持仓管理 ==========
        if self.position != 0:
            if self.position == 1:
                # 止盈：回归均线
                if tp_type == 1 and close >= self.entry_ma:
                    self._reset()
                    return Signal(action="close", price=close, tag="mean_revert")
                # 止盈：ATR目标
                if tp_type == 0 and close >= self.tp_price:
                    self._reset()
                    return Signal(action="close", price=close, tag="take_profit")
                # 止损
                if low <= self.stop_price:
                    self._reset()
                    return Signal(action="close", price=self.stop_price, tag="stop_loss")

            else:  # 空头
                if tp_type == 1 and close <= self.entry_ma:
                    self._reset()
                    return Signal(action="close", price=close, tag="mean_revert")
                if tp_type == 0 and close <= self.tp_price:
                    self._reset()
                    return Signal(action="close", price=close, tag="take_profit")
                if high >= self.stop_price:
                    self._reset()
                    return Signal(action="close", price=self.stop_price, tag="stop_loss")

            return None

        # ========== 开仓条件 ==========
        # 成交量确认
        if vol_ratio < vol_mult:
            return None

        # 多头反转: RSI从超卖区回升 + 触及布林下轨
        if prev_rsi < rsi_os and rsi > rsi_os:  # RSI从超卖回升
            if row['prev_close'] <= bb_dn:  # 前一根触及下轨
                # 趋势过滤
                if use_trend and trend != 1:
                    return None
                stop = close - atr * stop_atr
                tp = close + atr * tp_atr if tp_type == 0 else 0
                self._enter(1, close, stop, tp, bb_mid)
                return Signal(action="buy", price=close, stop_loss=atr * stop_atr, tag="oversold_long")

        # 空头反转: RSI从超买区回落 + 触及布林上轨
        if prev_rsi > rsi_ob and rsi < rsi_ob:  # RSI从超买回落
            if row['prev_close'] >= bb_up:  # 前一根触及上轨
                if use_trend and trend != -1:
                    return None
                stop = close + atr * stop_atr
                tp = close - atr * tp_atr if tp_type == 0 else 0
                self._enter(-1, close, stop, tp, bb_mid)
                return Signal(action="sell", price=close, stop_loss=atr * stop_atr, tag="overbought_short")

        return None

    def _enter(self, direction: int, price: float, stop: float, tp: float, ma: float):
        self.position = direction
        self.entry_price = price
        self.stop_price = stop
        self.tp_price = tp
        self.entry_ma = ma

    def _reset(self):
        self.position = 0

    def on_trade(self, signal: Signal, is_entry: bool):
        if not is_entry:
            self.position = 0

    def reset(self):
        super().reset()
        self.stop_price = 0
        self.tp_price = 0
        self.entry_ma = 0
