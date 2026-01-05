# coding=utf-8
"""
MomentumMean 动量均值回归策略 v2

简化版逻辑:
1. 趋势判断: 价格站上/跌破 EMA 均线
2. 入场: 趋势确立 + 价格回踩均线后反弹
3. 止损: ATR 动态止损
4. 止盈: 趋势反转或追踪止盈

适用于: 日线周期，趋势性品种
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, StrategyParam, Signal


class MomentumMean(BaseStrategy):
    """动量均值回归策略"""

    name = "momentum_mean"
    display_name = "动量均值回归"
    description = "顺势回调入场，趋势跟踪止盈"
    version = "2.0"
    author = "Claude"

    warmup_num = 60

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            # 均线参数
            StrategyParam("ma_len", "均线周期", 20, 10, 60, 5, "int", description="趋势均线周期"),
            StrategyParam("trend_len", "趋势确认", 40, 20, 80, 10, "int", description="长期趋势周期"),

            # 入场参数
            StrategyParam("touch_pct", "触线幅度%", 1.0, 0.3, 3.0, 0.1, "float", description="价格触及均线的距离"),
            StrategyParam("bounce_pct", "反弹确认%", 0.5, 0.2, 2.0, 0.1, "float", description="反弹幅度确认入场"),

            # 过滤
            StrategyParam("adx_len", "ADX周期", 14, 7, 30, 1, "int", description="ADX计算周期"),
            StrategyParam("adx_min", "ADX最小值", 18.0, 10.0, 30.0, 1.0, "float", description="趋势强度过滤"),

            # 止损止盈
            StrategyParam("atr_len", "ATR周期", 14, 7, 30, 1, "int", description="ATR计算周期"),
            StrategyParam("stop_n", "止损ATR", 2.5, 1.5, 4.0, 0.5, "float", description="初始止损倍数"),
            StrategyParam("trail_n", "追踪ATR", 2.0, 1.0, 3.0, 0.5, "float", description="追踪止损倍数"),
            StrategyParam("profit_start", "追踪起始%", 2.0, 1.0, 5.0, 0.5, "float", description="盈利多少激活追踪"),
        ]

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.stop_price = 0
        self.highest_price = 0
        self.lowest_price = float('inf')
        self.trailing_active = False
        self.touched_ma = False  # 是否触及过均线

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()

        ma_len = self.params['ma_len']
        trend_len = self.params['trend_len']
        atr_len = self.params['atr_len']
        adx_len = self.params['adx_len']

        # 均线
        df['ma'] = df['close'].rolling(ma_len).mean()
        df['ma_long'] = df['close'].rolling(trend_len).mean()

        # 趋势方向: 1=多头, -1=空头
        df['trend'] = np.where(df['ma'] > df['ma_long'], 1, -1)

        # 价格与均线的距离 (%)
        df['ma_dist'] = (df['close'] - df['ma']) / df['ma'] * 100

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

        # 昨日收盘与均线的关系
        df['prev_above_ma'] = df['close'].shift(1) > df['ma'].shift(1)
        df['prev_below_ma'] = df['close'].shift(1) < df['ma'].shift(1)

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """每根K线的逻辑"""
        if idx < self.warmup_num:
            return None

        row = df.iloc[idx]
        prev = df.iloc[idx - 1]

        close = row['close']
        ma = row['ma']
        atr = row['atr']
        trend = row['trend']
        prev_trend = prev['trend']
        adx = row['adx']
        ma_dist = row['ma_dist']

        touch_pct = self.params['touch_pct']
        bounce_pct = self.params['bounce_pct']
        adx_min = self.params['adx_min']
        stop_n = self.params['stop_n']
        trail_n = self.params['trail_n']
        profit_start = self.params['profit_start']

        if pd.isna(atr) or pd.isna(adx) or pd.isna(ma):
            return None

        # ========== 持仓管理 ==========
        if self.position != 0:
            if self.position == 1:  # 多头
                self.highest_price = max(self.highest_price, close)
                pnl_pct = (close - self.entry_price) / self.entry_price * 100

                # 激活追踪止损
                if pnl_pct >= profit_start and not self.trailing_active:
                    self.trailing_active = True

                # 更新追踪止损
                if self.trailing_active:
                    new_stop = self.highest_price - atr * trail_n
                    self.stop_price = max(self.stop_price, new_stop)

                # 止损/追踪止损触发
                if close <= self.stop_price:
                    tag = "trailing_stop" if self.trailing_active else "stop_loss"
                    return Signal(action="close", price=close, tag=tag)

                # 趋势反转平仓
                if trend == -1 and prev_trend == 1:
                    return Signal(action="close", price=close, tag="trend_reverse")

            else:  # 空头
                self.lowest_price = min(self.lowest_price, close)
                pnl_pct = (self.entry_price - close) / self.entry_price * 100

                if pnl_pct >= profit_start and not self.trailing_active:
                    self.trailing_active = True

                if self.trailing_active:
                    new_stop = self.lowest_price + atr * trail_n
                    self.stop_price = min(self.stop_price, new_stop)

                if close >= self.stop_price:
                    tag = "trailing_stop" if self.trailing_active else "stop_loss"
                    return Signal(action="close", price=close, tag=tag)

                if trend == 1 and prev_trend == -1:
                    return Signal(action="close", price=close, tag="trend_reverse")

            return None

        # ========== 开仓逻辑 ==========
        # ADX 过滤
        if adx < adx_min:
            return None

        # 多头入场：上升趋势 + 价格回踩均线后反弹
        if trend == 1 and prev_trend == 1:
            # 条件1: 价格曾触及均线附近 (当前或前一根)
            touched = abs(ma_dist) <= touch_pct or abs(prev['ma_dist']) <= touch_pct
            # 条件2: 从均线下方反弹 (或从均线附近向上)
            bouncing = close > prev['close'] and close > ma

            if touched and bouncing:
                stop = close - atr * stop_n
                self._setup_long(close, stop)
                return Signal(action="buy", price=close, stop_loss=stop, tag="ma_bounce_long")

        # 空头入场：下降趋势 + 价格反弹到均线后回落
        if trend == -1 and prev_trend == -1:
            touched = abs(ma_dist) <= touch_pct or abs(prev['ma_dist']) <= touch_pct
            falling = close < prev['close'] and close < ma

            if touched and falling:
                stop = close + atr * stop_n
                self._setup_short(close, stop)
                return Signal(action="sell", price=close, stop_loss=stop, tag="ma_bounce_short")

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
