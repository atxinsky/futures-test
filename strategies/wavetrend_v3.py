# coding=utf-8
"""
WaveTrend V3 策略
多种止损方式对比测试

止损方法：
1. ATR止损 - 经典波动率止损
2. 结构止损 - 近N根K线最低点
3. WT信号止损 - WT死叉或回零轴
4. 百分比止损 - 固定比例
5. Chandelier止损 - 最高点回撤
6. 混合止损 - 结构+ATR结合
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from strategies.base import BaseStrategy, StrategyParam, Signal


class WaveTrendV3Strategy(BaseStrategy):
    """WaveTrend V3策略 - 多种止损"""

    name = "wavetrend_v3"
    display_name = "WaveTrend V3 (多种止损)"
    description = "测试不同止损方法的效果"
    version = "3.0"
    author = "Eric"

    warmup_num = 60

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            # WaveTrend参数
            StrategyParam("n1", "通道长度", 10, 5, 20, 1, "int"),
            StrategyParam("n2", "平均长度", 21, 10, 30, 1, "int"),
            StrategyParam("ob_level", "超买阈值", 53, 40, 70, 1, "int"),
            StrategyParam("os_level", "超卖阈值", -53, -70, -40, 1, "int"),

            # 止损方法选择
            StrategyParam("stop_method", "止损方法", "structure", param_type="select",
                         options=["atr", "structure", "wt_signal", "percent", "chandelier", "hybrid"],
                         description="atr=ATR倍数, structure=结构止损, wt_signal=WT信号, percent=固定百分比, chandelier=吊灯止损, hybrid=混合"),

            # ATR止损参数
            StrategyParam("atr_len", "ATR周期", 14, 7, 30, 1, "int"),
            StrategyParam("atr_mult", "ATR倍数", 3.0, 1.5, 5.0, 0.5, "float"),

            # 结构止损参数
            StrategyParam("structure_bars", "结构回看K线", 10, 5, 30, 5, "int"),

            # 百分比止损参数
            StrategyParam("stop_pct", "止损百分比", 3.0, 1.0, 10.0, 0.5, "float"),

            # Chandelier止损参数
            StrategyParam("chandelier_bars", "吊灯回看K线", 22, 10, 50, 5, "int"),
            StrategyParam("chandelier_mult", "吊灯ATR倍数", 3.0, 2.0, 5.0, 0.5, "float"),

            # 止盈参数
            StrategyParam("use_profit_target", "启用止盈", True, param_type="bool"),
            StrategyParam("profit_wt_level", "止盈WT阈值", 40, 20, 60, 10, "int"),

            # 其他
            StrategyParam("only_long", "只做多", True, param_type="bool"),
            StrategyParam("capital_rate", "资金使用率", 1.0, 0.5, 1.0, 0.1, "float"),
            StrategyParam("risk_rate", "单笔风险", 0.02, 0.01, 0.05, 0.01, "float"),
        ]

    def __init__(self, params=None):
        super().__init__(params)
        self.triggered_profit = False
        self.highest_since_entry = 0
        self.lowest_since_entry = float('inf')

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指标"""
        df = df.copy()

        n1 = self.params['n1']
        n2 = self.params['n2']
        atr_len = self.params['atr_len']
        structure_bars = self.params['structure_bars']
        chandelier_bars = self.params['chandelier_bars']

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

        # 结构止损位 (近N根K线最低点)
        df['structure_low'] = df['low'].rolling(window=structure_bars).min()
        df['structure_high'] = df['high'].rolling(window=structure_bars).max()

        # Chandelier止损位
        df['chandelier_high'] = df['high'].rolling(window=chandelier_bars).max()
        df['chandelier_low'] = df['low'].rolling(window=chandelier_bars).min()
        df['chandelier_atr'] = df['tr'].rolling(window=chandelier_bars).mean()

        # 信号
        df['cross_up'] = (df['wt1'] > df['wt2']) & (df['wt1'].shift(1) <= df['wt2'].shift(1))
        df['cross_down'] = (df['wt1'] < df['wt2']) & (df['wt1'].shift(1) >= df['wt2'].shift(1))
        df['wt_zero_cross_down'] = (df['wt1'] < 0) & (df['wt1'].shift(1) >= 0)
        df['wt_zero_cross_up'] = (df['wt1'] > 0) & (df['wt1'].shift(1) <= 0)

        return df

    def reset(self):
        super().reset()
        self.triggered_profit = False
        self.highest_since_entry = 0
        self.lowest_since_entry = float('inf')

    def _calculate_stop_price(self, idx: int, df: pd.DataFrame, direction: int) -> tuple:
        """
        计算止损价格
        返回: (止损价格, 止损类型描述)
        """
        stop_method = self.params['stop_method']
        curr_close = df['close'].iloc[idx]
        atr = df['atr'].iloc[idx]

        if direction == 1:  # 多头
            if stop_method == "atr":
                stop = self.entry_price - atr * self.params['atr_mult']
                return stop, "atr_stop"

            elif stop_method == "structure":
                # 结构止损：入场时的近N根最低点
                stop = df['structure_low'].iloc[idx] * 0.995  # 略低于结构位
                return stop, "structure_stop"

            elif stop_method == "percent":
                stop = self.entry_price * (1 - self.params['stop_pct'] / 100)
                return stop, "pct_stop"

            elif stop_method == "chandelier":
                # 吊灯止损：最高点 - ATR*倍数
                chandelier_atr = df['chandelier_atr'].iloc[idx]
                stop = self.highest_since_entry - chandelier_atr * self.params['chandelier_mult']
                return stop, "chandelier_stop"

            elif stop_method == "hybrid":
                # 混合：结构止损和ATR止损取更宽松的
                structure_stop = df['structure_low'].iloc[idx] * 0.995
                atr_stop = self.entry_price - atr * self.params['atr_mult']
                stop = min(structure_stop, atr_stop)  # 取更低的（更宽松）
                return stop, "hybrid_stop"

            elif stop_method == "wt_signal":
                # WT信号止损：不用固定价格，返回None
                return None, "wt_signal"

        else:  # 空头
            if stop_method == "atr":
                stop = self.entry_price + atr * self.params['atr_mult']
                return stop, "atr_stop"

            elif stop_method == "structure":
                stop = df['structure_high'].iloc[idx] * 1.005
                return stop, "structure_stop"

            elif stop_method == "percent":
                stop = self.entry_price * (1 + self.params['stop_pct'] / 100)
                return stop, "pct_stop"

            elif stop_method == "chandelier":
                chandelier_atr = df['chandelier_atr'].iloc[idx]
                stop = self.lowest_since_entry + chandelier_atr * self.params['chandelier_mult']
                return stop, "chandelier_stop"

            elif stop_method == "hybrid":
                structure_stop = df['structure_high'].iloc[idx] * 1.005
                atr_stop = self.entry_price + atr * self.params['atr_mult']
                stop = max(structure_stop, atr_stop)
                return stop, "hybrid_stop"

            elif stop_method == "wt_signal":
                return None, "wt_signal"

        return None, "unknown"

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
        wt_zero_cross_down = df['wt_zero_cross_down'].iloc[idx]
        wt_zero_cross_up = df['wt_zero_cross_up'].iloc[idx]

        ob_level = self.params['ob_level']
        os_level = self.params['os_level']
        stop_method = self.params['stop_method']
        use_profit_target = self.params['use_profit_target']
        profit_wt_level = self.params['profit_wt_level']
        only_long = self.params['only_long']

        if pd.isna(wt1) or pd.isna(wt2) or pd.isna(atr) or atr == 0:
            return None

        # ========== 多头持仓检查 ==========
        if self.position == 1:
            # 更新最高价
            if curr_high > self.highest_since_entry:
                self.highest_since_entry = curr_high

            exit_tag = None

            # 1. 止损检查
            if stop_method == "wt_signal":
                # WT信号止损：死叉或回零轴
                if cross_down:
                    exit_tag = "wt_death_cross"
                elif wt_zero_cross_down:
                    exit_tag = "wt_zero_cross"
            else:
                # 其他止损方法
                stop_price, stop_type = self._calculate_stop_price(idx, df, 1)
                if stop_price and curr_close < stop_price:
                    exit_tag = stop_type

            # 2. 止盈逻辑
            if use_profit_target and not exit_tag:
                if wt1 > ob_level:
                    self.triggered_profit = True

                if self.triggered_profit and wt1 < profit_wt_level:
                    exit_tag = "profit_take"

            if exit_tag:
                self.position = 0
                return Signal(action="close", price=curr_close, tag=exit_tag)

        # ========== 空头持仓检查 ==========
        elif self.position == -1:
            if curr_low < self.lowest_since_entry:
                self.lowest_since_entry = curr_low

            exit_tag = None

            if stop_method == "wt_signal":
                if cross_up:
                    exit_tag = "wt_golden_cross"
                elif wt_zero_cross_up:
                    exit_tag = "wt_zero_cross"
            else:
                stop_price, stop_type = self._calculate_stop_price(idx, df, -1)
                if stop_price and curr_close > stop_price:
                    exit_tag = stop_type

            if use_profit_target and not exit_tag:
                if wt1 < os_level:
                    self.triggered_profit = True
                if self.triggered_profit and wt1 > -profit_wt_level:
                    exit_tag = "profit_take"

            if exit_tag:
                self.position = 0
                return Signal(action="close", price=curr_close, tag=exit_tag)

        # ========== 开仓信号 ==========
        if self.position == 0:
            # 超卖区金叉 -> 做多
            if wt1 < os_level and cross_up:
                self.position = 1
                self.entry_price = curr_close
                self.highest_since_entry = curr_high
                self.triggered_profit = False

                # 计算止损用于仓位
                stop_price, _ = self._calculate_stop_price(idx, df, 1)
                if stop_price is None:
                    stop_price = curr_close - atr * 3  # 默认3倍ATR

                return Signal(
                    action="buy",
                    price=curr_close,
                    tag="oversold_buy",
                    stop_loss=stop_price
                )

            # 超买区死叉 -> 做空
            if not only_long and wt1 > ob_level and cross_down:
                self.position = -1
                self.entry_price = curr_close
                self.lowest_since_entry = curr_low
                self.triggered_profit = False

                stop_price, _ = self._calculate_stop_price(idx, df, -1)
                if stop_price is None:
                    stop_price = curr_close + atr * 3

                return Signal(
                    action="sell",
                    price=curr_close,
                    tag="overbought_sell",
                    stop_loss=stop_price
                )

        return None
