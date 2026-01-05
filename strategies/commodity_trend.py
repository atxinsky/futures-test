# coding=utf-8
"""
CommodityTrend 商品期货趋势策略

专为商品期货设计:
1. 长周期趋势判断 (50日) - 商品趋势持续时间较长
2. 回调入场 - 等价格回调到均线附近
3. 宽止损 (3ATR) - 商品波动大，避免被洗出
4. 趋势反转出场 - 让利润奔跑

特点: 低频交易，高胜率，适合日线
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, StrategyParam, Signal


class CommodityTrend(BaseStrategy):
    """商品期货趋势策略"""

    name = "commodity_trend"
    display_name = "商品趋势"
    description = "商品期货专用趋势策略"
    version = "1.0"
    author = "Claude"

    warmup_num = 60

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            # 趋势参数
            StrategyParam("fast_ma", "快线周期", 20, 10, 40, 5, "int", description="快速均线"),
            StrategyParam("slow_ma", "慢线周期", 50, 30, 100, 10, "int", description="慢速均线"),

            # 入场
            StrategyParam("pullback_pct", "回调幅度%", 2.0, 0.5, 5.0, 0.5, "float", description="回调到均线距离"),
            StrategyParam("confirm_bars", "确认K线", 1, 1, 3, 1, "int", description="回调后确认上涨"),

            # 过滤
            StrategyParam("vol_ma", "量能周期", 20, 10, 30, 5, "int", description="成交量均线"),
            StrategyParam("vol_mult", "放量倍数", 1.0, 0.8, 2.0, 0.1, "float", description="入场需要放量"),

            # 止损
            StrategyParam("atr_len", "ATR周期", 20, 10, 30, 5, "int", description="ATR计算周期"),
            StrategyParam("stop_atr", "止损ATR", 3.0, 2.0, 5.0, 0.5, "float", description="宽止损"),
            StrategyParam("trail_atr", "追踪ATR", 2.5, 1.5, 4.0, 0.5, "float", description="追踪止损"),
            StrategyParam("trail_start", "追踪起始%", 5.0, 2.0, 10.0, 1.0, "float", description="盈利后激活追踪"),
        ]

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.stop_price = 0
        self.highest_price = 0
        self.lowest_price = float('inf')
        self.trailing_active = False
        self.pullback_detected = False
        self.pullback_bar = 0

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        fast = self.params['fast_ma']
        slow = self.params['slow_ma']
        atr_len = self.params['atr_len']
        vol_ma = self.params['vol_ma']

        # 均线
        df['ma_fast'] = df['close'].rolling(fast).mean()
        df['ma_slow'] = df['close'].rolling(slow).mean()

        # 趋势 1=多 -1=空
        df['trend'] = np.where(df['ma_fast'] > df['ma_slow'], 1, -1)

        # 价格与快线距离 %
        df['dist'] = (df['close'] - df['ma_fast']) / df['ma_fast'] * 100

        # ATR
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(atr_len).mean()

        # 成交量
        df['vol_ma'] = df['volume'].rolling(vol_ma).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']

        # 涨跌
        df['up'] = df['close'] > df['close'].shift(1)
        df['down'] = df['close'] < df['close'].shift(1)

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        if idx < self.warmup_num:
            return None

        row = df.iloc[idx]
        prev = df.iloc[idx - 1]

        close = row['close']
        high = row['high']
        low = row['low']
        atr = row['atr']
        trend = row['trend']
        prev_trend = prev['trend']
        dist = row['dist']
        vol_ratio = row['vol_ratio']
        ma_fast = row['ma_fast']

        pullback_pct = self.params['pullback_pct']
        vol_mult = self.params['vol_mult']
        stop_atr = self.params['stop_atr']
        trail_atr = self.params['trail_atr']
        trail_start = self.params['trail_start']
        confirm_bars = self.params['confirm_bars']

        if pd.isna(atr) or pd.isna(ma_fast):
            return None

        # ========== 持仓管理 ==========
        if self.position != 0:
            if self.position == 1:
                self.highest_price = max(self.highest_price, high)
                pnl_pct = (close - self.entry_price) / self.entry_price * 100

                # 追踪止损激活
                if pnl_pct >= trail_start and not self.trailing_active:
                    self.trailing_active = True

                if self.trailing_active:
                    new_stop = self.highest_price - atr * trail_atr
                    self.stop_price = max(self.stop_price, new_stop)

                # 止损
                if low <= self.stop_price:
                    tag = "trailing_stop" if self.trailing_active else "stop_loss"
                    self._reset_state()
                    return Signal(action="close", price=self.stop_price, tag=tag)

                # 趋势反转
                if trend == -1 and prev_trend == 1:
                    self._reset_state()
                    return Signal(action="close", price=close, tag="trend_reverse")

            else:  # 空头
                self.lowest_price = min(self.lowest_price, low)
                pnl_pct = (self.entry_price - close) / self.entry_price * 100

                if pnl_pct >= trail_start and not self.trailing_active:
                    self.trailing_active = True

                if self.trailing_active:
                    new_stop = self.lowest_price + atr * trail_atr
                    self.stop_price = min(self.stop_price, new_stop)

                if high >= self.stop_price:
                    tag = "trailing_stop" if self.trailing_active else "stop_loss"
                    self._reset_state()
                    return Signal(action="close", price=self.stop_price, tag=tag)

                if trend == 1 and prev_trend == -1:
                    self._reset_state()
                    return Signal(action="close", price=close, tag="trend_reverse")

            return None

        # ========== 开仓逻辑 ==========

        # 多头: 上升趋势 + 回调到均线 + 放量上涨
        if trend == 1 and prev_trend == 1:
            # 检测回调
            if dist <= pullback_pct and dist >= -pullback_pct:
                self.pullback_detected = True
                self.pullback_bar = idx

            # 回调后确认入场
            if self.pullback_detected and idx > self.pullback_bar:
                # 确认上涨 + 放量
                if row['up'] and vol_ratio >= vol_mult and close > ma_fast:
                    stop = close - atr * stop_atr
                    self._setup_long(close, stop)
                    self.pullback_detected = False
                    return Signal(action="buy", price=close, stop_loss=stop, tag="pullback_long")

        # 空头: 下降趋势 + 反弹到均线 + 放量下跌
        if trend == -1 and prev_trend == -1:
            if dist >= -pullback_pct and dist <= pullback_pct:
                self.pullback_detected = True
                self.pullback_bar = idx

            if self.pullback_detected and idx > self.pullback_bar:
                if row['down'] and vol_ratio >= vol_mult and close < ma_fast:
                    stop = close + atr * stop_atr
                    self._setup_short(close, stop)
                    self.pullback_detected = False
                    return Signal(action="sell", price=close, stop_loss=stop, tag="pullback_short")

        # 趋势变化时清除回调状态
        if trend != prev_trend:
            self.pullback_detected = False

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

    def _reset_state(self):
        self.position = 0
        self.trailing_active = False
        self.highest_price = 0
        self.lowest_price = float('inf')
        self.pullback_detected = False

    def on_trade(self, signal: Signal, is_entry: bool):
        if not is_entry:
            self.position = 0

    def reset(self):
        super().reset()
        self.stop_price = 0
        self.highest_price = 0
        self.lowest_price = float('inf')
        self.trailing_active = False
        self.pullback_detected = False
        self.pullback_bar = 0
