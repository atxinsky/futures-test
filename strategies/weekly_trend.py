# coding=utf-8
"""
WeeklyTrend 周线趋势策略

核心思路：
1. 周线定方向 - 只做周线趋势方向的交易
2. 日线找入场 - 等待日线回调到位再入场
3. 宽止损 - 给趋势足够空间发展
4. 只交易趋势性好的品种

适合: 日线级别，趋势性强的品种
目标: 年化10%+，回撤控制在15%以内
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, StrategyParam, Signal


class WeeklyTrend(BaseStrategy):
    """周线趋势策略"""

    name = "weekly_trend"
    display_name = "周线趋势"
    description = "周线定方向，日线找入场"
    version = "1.0"
    author = "Claude"

    warmup_num = 60  # 需要足够数据计算周线指标

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            # 周线趋势判断
            StrategyParam("week_ma", "周线均线", 10, 5, 20, 5, "int", description="周线趋势均线(对应日线50)"),

            # 日线入场
            StrategyParam("day_ma_fast", "日线快均", 10, 5, 20, 5, "int"),
            StrategyParam("day_ma_slow", "日线慢均", 30, 20, 50, 10, "int"),
            StrategyParam("pullback_pct", "回调幅度%", 3.0, 1.0, 5.0, 0.5, "float", description="回调到位才入场"),

            # 趋势强度过滤
            StrategyParam("adx_len", "ADX周期", 14, 7, 21, 7, "int"),
            StrategyParam("adx_thres", "ADX阈值", 20, 15, 30, 5, "float", description="趋势强度门槛"),

            # 止损止盈
            StrategyParam("atr_len", "ATR周期", 20, 10, 30, 5, "int"),
            StrategyParam("stop_atr", "止损ATR", 3.0, 2.0, 5.0, 0.5, "float", description="宽止损"),
            StrategyParam("trail_atr", "追踪ATR", 2.5, 1.5, 4.0, 0.5, "float"),
            StrategyParam("trail_start", "追踪起点%", 5.0, 3.0, 10.0, 1.0, "float"),

            # 时间管理
            StrategyParam("max_hold", "最大持仓天", 60, 30, 120, 15, "int"),
            StrategyParam("min_hold", "最小持仓天", 5, 3, 10, 1, "int"),

            # 仓位
            StrategyParam("capital_rate", "资金比例", 0.5, 0.2, 0.8, 0.1, "float"),
            StrategyParam("risk_rate", "风险比例", 0.03, 0.01, 0.05, 0.01, "float"),
        ]

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.stop_price = 0
        self.highest = 0
        self.lowest = float('inf')
        self.entry_bar = 0
        self.trailing = False

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        week_ma = self.params['week_ma']
        day_ma_fast = self.params['day_ma_fast']
        day_ma_slow = self.params['day_ma_slow']
        adx_len = self.params['adx_len']
        atr_len = self.params['atr_len']

        # 日线均线
        df['ma_fast'] = df['close'].rolling(day_ma_fast).mean()
        df['ma_slow'] = df['close'].rolling(day_ma_slow).mean()

        # 模拟周线 (5日为一周)
        df['week_close'] = df['close'].rolling(5).mean()
        df['week_ma'] = df['week_close'].rolling(week_ma).mean()
        df['week_trend'] = np.where(df['week_close'] > df['week_ma'], 1, -1)

        # 周线趋势持续确认 (连续3周同向)
        df['week_trend_confirm'] = df['week_trend'].rolling(15).apply(
            lambda x: 1 if (x == 1).all() else (-1 if (x == -1).all() else 0)
        )

        # ATR
        high, low, close = df['high'], df['low'], df['close']
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(atr_len).mean()

        # ADX 趋势强度
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr_smooth = tr.rolling(adx_len).mean()
        plus_di = 100 * plus_dm.rolling(adx_len).mean() / atr_smooth
        minus_di = 100 * minus_dm.rolling(adx_len).mean() / atr_smooth
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(adx_len).mean()

        # 回调幅度 (相对近期高低点)
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['pullback_from_high'] = (df['high_20'] - df['close']) / df['high_20'] * 100
        df['pullback_from_low'] = (df['close'] - df['low_20']) / df['low_20'] * 100

        # 成交量确认
        df['vol_ma'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        if idx < self.warmup_num:
            return None

        row = df.iloc[idx]

        close = row['close']
        high = row['high']
        low = row['low']
        atr = row['atr']
        adx = row['adx']
        week_trend = row['week_trend_confirm']
        ma_fast = row['ma_fast']
        ma_slow = row['ma_slow']
        pullback_high = row['pullback_from_high']
        pullback_low = row['pullback_from_low']

        adx_thres = self.params['adx_thres']
        pullback_pct = self.params['pullback_pct']
        stop_atr = self.params['stop_atr']
        trail_atr = self.params['trail_atr']
        trail_start = self.params['trail_start']
        max_hold = self.params['max_hold']
        min_hold = self.params['min_hold']

        if pd.isna(atr) or pd.isna(adx) or pd.isna(week_trend):
            return None

        # ========== 持仓管理 ==========
        if self.position != 0:
            bars_held = idx - self.entry_bar

            if self.position == 1:
                self.highest = max(self.highest, high)
                pnl_pct = (close - self.entry_price) / self.entry_price * 100

                # 激活追踪
                if pnl_pct >= trail_start and not self.trailing:
                    self.trailing = True
                    self.stop_price = max(self.stop_price, close - atr * trail_atr)

                # 更新追踪止损
                if self.trailing:
                    new_stop = self.highest - atr * trail_atr
                    self.stop_price = max(self.stop_price, new_stop)

                # 止损
                if low <= self.stop_price:
                    tag = "trailing_stop" if self.trailing else "stop_loss"
                    self._reset()
                    return Signal(action="close", price=self.stop_price, tag=tag)

                # 趋势反转退出 (周线转向)
                if week_trend == -1 and bars_held >= min_hold:
                    self._reset()
                    return Signal(action="close", price=close, tag="trend_reverse")

                # 超时退出
                if bars_held >= max_hold:
                    self._reset()
                    return Signal(action="close", price=close, tag="timeout")

            else:  # 空头
                self.lowest = min(self.lowest, low)
                pnl_pct = (self.entry_price - close) / self.entry_price * 100

                if pnl_pct >= trail_start and not self.trailing:
                    self.trailing = True
                    self.stop_price = min(self.stop_price, close + atr * trail_atr)

                if self.trailing:
                    new_stop = self.lowest + atr * trail_atr
                    self.stop_price = min(self.stop_price, new_stop)

                if high >= self.stop_price:
                    tag = "trailing_stop" if self.trailing else "stop_loss"
                    self._reset()
                    return Signal(action="close", price=self.stop_price, tag=tag)

                if week_trend == 1 and bars_held >= min_hold:
                    self._reset()
                    return Signal(action="close", price=close, tag="trend_reverse")

                if bars_held >= max_hold:
                    self._reset()
                    return Signal(action="close", price=close, tag="timeout")

            return None

        # ========== 开仓条件 ==========
        # 1. 趋势强度足够
        if adx < adx_thres:
            return None

        # 2. 周线趋势确认
        if week_trend == 0:
            return None

        # 3. 日线均线顺序正确
        # 4. 价格回调到位

        # 多头: 周线向上 + 日线回调 + 均线多头排列
        if week_trend == 1:
            if ma_fast > ma_slow and close > ma_slow:  # 均线多头
                if pullback_high >= pullback_pct:  # 有足够回调
                    if close > ma_fast:  # 回调后站上快线
                        stop = close - atr * stop_atr
                        self._enter(1, close, stop, idx)
                        return Signal(action="buy", price=close, stop_loss=atr * stop_atr, tag="trend_long")

        # 空头: 周线向下 + 日线反弹 + 均线空头排列
        if week_trend == -1:
            if ma_fast < ma_slow and close < ma_slow:  # 均线空头
                if pullback_low >= pullback_pct:  # 有足够反弹
                    if close < ma_fast:  # 反弹后跌破快线
                        stop = close + atr * stop_atr
                        self._enter(-1, close, stop, idx)
                        return Signal(action="sell", price=close, stop_loss=atr * stop_atr, tag="trend_short")

        return None

    def _enter(self, direction: int, price: float, stop: float, bar: int):
        self.position = direction
        self.entry_price = price
        self.stop_price = stop
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
        self.highest = 0
        self.lowest = float('inf')
        self.entry_bar = 0
        self.trailing = False
