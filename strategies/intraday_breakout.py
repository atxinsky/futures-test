# coding=utf-8
"""
IntradayBreakout 股指日内突破策略

核心思路：
1. 开盘30分钟确定当日区间 (ORB - Opening Range Breakout)
2. 突破区间后跟随入场
3. 收盘前强制平仓，不隔夜
4. 日内严格止损

适合: IF, IH, IC, IM 股指期货
周期: 15分钟
"""

from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from strategies.base import BaseStrategy, StrategyParam, Signal


class IntradayBreakout(BaseStrategy):
    """股指日内突破策略"""

    name = "intraday_breakout"
    display_name = "日内突破"
    description = "开盘区间突破，收盘前平仓"
    version = "1.0"
    author = "Claude"

    warmup_num = 10

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            # 开盘区间
            StrategyParam("orb_bars", "开盘K线数", 2, 1, 4, 1, "int",
                         description="开盘几根K线确定区间(15分钟*2=30分钟)"),
            StrategyParam("breakout_pct", "突破幅度%", 0.1, 0.05, 0.3, 0.05, "float",
                         description="超过区间多少才算突破"),

            # 过滤条件
            StrategyParam("min_range_pct", "最小区间%", 0.3, 0.1, 0.5, 0.1, "float",
                         description="开盘区间太小不做"),
            StrategyParam("max_range_pct", "最大区间%", 1.5, 1.0, 2.0, 0.5, "float",
                         description="开盘区间太大不做"),

            # 止损止盈
            StrategyParam("stop_pct", "止损%", 0.5, 0.3, 1.0, 0.1, "float"),
            StrategyParam("target_pct", "目标%", 1.0, 0.5, 2.0, 0.5, "float"),
            StrategyParam("trail_pct", "追踪止损%", 0.3, 0.2, 0.5, 0.1, "float"),
            StrategyParam("trail_start_pct", "追踪起点%", 0.5, 0.3, 1.0, 0.1, "float"),

            # 时间控制
            StrategyParam("entry_end_bar", "入场截止", 12, 8, 16, 2, "int",
                         description="第几根K线后不再入场"),
            StrategyParam("exit_bar", "平仓K线", 16, 14, 16, 1, "int",
                         description="第几根K线强制平仓(15:00前)"),

            # 仓位
            StrategyParam("capital_rate", "资金比例", 0.5, 0.2, 0.8, 0.1, "float"),
            StrategyParam("risk_rate", "风险比例", 0.02, 0.01, 0.05, 0.01, "float"),
        ]

    def __init__(self, params: Dict = None):
        super().__init__(params)
        self.day_high = 0
        self.day_low = float('inf')
        self.orb_high = 0
        self.orb_low = float('inf')
        self.orb_set = False
        self.bar_of_day = 0
        self.current_date = None
        self.stop_price = 0
        self.target_price = 0
        self.highest = 0
        self.lowest = float('inf')
        self.trailing = False
        self.traded_today = False

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 解析日期和时间
        if 'time' in df.columns:
            df['dt'] = pd.to_datetime(df['time'])
        elif 'datetime' in df.columns:
            df['dt'] = pd.to_datetime(df['datetime'])
        else:
            df['dt'] = pd.to_datetime(df.index)

        df['date'] = df['dt'].dt.date
        df['time_of_day'] = df['dt'].dt.time
        df['hour'] = df['dt'].dt.hour
        df['minute'] = df['dt'].dt.minute

        # 标记每天的K线序号 (从1开始)
        df['bar_of_day'] = df.groupby('date').cumcount() + 1

        # 判断是否为交易时间 (9:30-15:00)
        df['is_trading'] = ((df['hour'] == 9) & (df['minute'] >= 30)) | \
                           ((df['hour'] >= 10) & (df['hour'] < 15))

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        if idx < self.warmup_num:
            return None

        row = df.iloc[idx]
        prev_row = df.iloc[idx - 1] if idx > 0 else None

        close = row['close']
        high = row['high']
        low = row['low']
        current_date = row['date']
        bar_of_day = row['bar_of_day']

        orb_bars = self.params['orb_bars']
        breakout_pct = self.params['breakout_pct']
        min_range_pct = self.params['min_range_pct']
        max_range_pct = self.params['max_range_pct']
        stop_pct = self.params['stop_pct']
        target_pct = self.params['target_pct']
        trail_pct = self.params['trail_pct']
        trail_start_pct = self.params['trail_start_pct']
        entry_end_bar = self.params['entry_end_bar']
        exit_bar = self.params['exit_bar']

        # 新的一天，重置状态
        if current_date != self.current_date:
            self._new_day(current_date)

        # 更新当日高低点
        self.day_high = max(self.day_high, high)
        self.day_low = min(self.day_low, low)
        self.bar_of_day = bar_of_day

        # 开盘区间形成阶段
        if bar_of_day <= orb_bars:
            self.orb_high = max(self.orb_high, high)
            self.orb_low = min(self.orb_low, low)
            if bar_of_day == orb_bars:
                self.orb_set = True
            return None

        if not self.orb_set:
            return None

        # 计算区间幅度
        orb_range = self.orb_high - self.orb_low
        orb_range_pct = orb_range / self.orb_low * 100

        # ========== 持仓管理 ==========
        if self.position != 0:
            # 收盘前强制平仓
            if bar_of_day >= exit_bar:
                self._reset()
                return Signal(action="close", price=close, tag="end_of_day")

            if self.position == 1:
                self.highest = max(self.highest, high)
                pnl_pct = (close - self.entry_price) / self.entry_price * 100

                # 止盈
                if close >= self.target_price:
                    self._reset()
                    return Signal(action="close", price=close, tag="take_profit")

                # 激活追踪
                if pnl_pct >= trail_start_pct and not self.trailing:
                    self.trailing = True

                # 追踪止损
                if self.trailing:
                    new_stop = self.highest * (1 - trail_pct / 100)
                    self.stop_price = max(self.stop_price, new_stop)

                # 止损
                if low <= self.stop_price:
                    tag = "trailing_stop" if self.trailing else "stop_loss"
                    self._reset()
                    return Signal(action="close", price=self.stop_price, tag=tag)

            else:  # 空头
                self.lowest = min(self.lowest, low)
                pnl_pct = (self.entry_price - close) / self.entry_price * 100

                if close <= self.target_price:
                    self._reset()
                    return Signal(action="close", price=close, tag="take_profit")

                if pnl_pct >= trail_start_pct and not self.trailing:
                    self.trailing = True

                if self.trailing:
                    new_stop = self.lowest * (1 + trail_pct / 100)
                    self.stop_price = min(self.stop_price, new_stop)

                if high >= self.stop_price:
                    tag = "trailing_stop" if self.trailing else "stop_loss"
                    self._reset()
                    return Signal(action="close", price=self.stop_price, tag=tag)

            return None

        # ========== 开仓条件 ==========
        # 今日已交易过，不再开仓
        if self.traded_today:
            return None

        # 入场时间限制
        if bar_of_day > entry_end_bar:
            return None

        # 区间幅度检查
        if orb_range_pct < min_range_pct or orb_range_pct > max_range_pct:
            return None

        # 突破幅度
        breakout_dist = orb_range * breakout_pct

        # 多头突破
        if close > self.orb_high + breakout_dist:
            stop = close * (1 - stop_pct / 100)
            target = close * (1 + target_pct / 100)
            self._enter(1, close, stop, target)
            return Signal(action="buy", price=close,
                         stop_loss=close - stop, tag="orb_long")

        # 空头突破
        if close < self.orb_low - breakout_dist:
            stop = close * (1 + stop_pct / 100)
            target = close * (1 - target_pct / 100)
            self._enter(-1, close, stop, target)
            return Signal(action="sell", price=close,
                         stop_loss=stop - close, tag="orb_short")

        return None

    def _new_day(self, date):
        """新的一天重置"""
        self.current_date = date
        self.day_high = 0
        self.day_low = float('inf')
        self.orb_high = 0
        self.orb_low = float('inf')
        self.orb_set = False
        self.bar_of_day = 0
        self.traded_today = False
        # 如果有隔夜持仓（不应该发生），强制平仓
        if self.position != 0:
            self._reset()

    def _enter(self, direction: int, price: float, stop: float, target: float):
        self.position = direction
        self.entry_price = price
        self.stop_price = stop
        self.target_price = target
        self.trailing = False
        self.traded_today = True
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
        self.day_high = 0
        self.day_low = float('inf')
        self.orb_high = 0
        self.orb_low = float('inf')
        self.orb_set = False
        self.bar_of_day = 0
        self.current_date = None
        self.stop_price = 0
        self.target_price = 0
        self.highest = 0
        self.lowest = float('inf')
        self.trailing = False
        self.traded_today = False
