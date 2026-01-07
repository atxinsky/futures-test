# coding=utf-8
"""
WaveTrend Final 策略
基于回测优化的最终版本

各品种最优配置：
- AU(黄金): WT信号止损, 低回撤稳健
- RB(螺纹钢): 吊灯止损, 趋势跟踪
- M(豆粕): 百分比止损3%, 高收益
- CU(沪铜): ATR止损
- FG(玻璃): 吊灯止损
- I(铁矿石): ATR止损
- AG(白银): WT信号止损
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from strategies.base import BaseStrategy, StrategyParam, Signal


# 各品种最优参数配置
SYMBOL_CONFIGS = {
    'AU': {'stop_method': 'wt_signal', 'atr_mult': 3.0, 'stop_pct': 3.0, 'note': '黄金-WT信号止损'},
    'AG': {'stop_method': 'wt_signal', 'atr_mult': 3.0, 'stop_pct': 3.0, 'note': '白银-WT信号止损'},
    'RB': {'stop_method': 'chandelier', 'atr_mult': 3.0, 'chandelier_mult': 3.0, 'note': '螺纹钢-吊灯止损'},
    'M':  {'stop_method': 'percent', 'stop_pct': 3.0, 'note': '豆粕-百分比止损'},
    'CU': {'stop_method': 'atr', 'atr_mult': 3.5, 'note': '沪铜-ATR止损'},
    'FG': {'stop_method': 'chandelier', 'chandelier_mult': 3.0, 'note': '玻璃-吊灯止损'},
    'I':  {'stop_method': 'atr', 'atr_mult': 3.0, 'note': '铁矿石-ATR止损'},
    'NI': {'stop_method': 'atr', 'atr_mult': 3.0, 'note': '沪镍-ATR止损'},
    'TA': {'stop_method': 'chandelier', 'chandelier_mult': 3.0, 'note': 'PTA-吊灯止损'},
    'PP': {'stop_method': 'atr', 'atr_mult': 3.0, 'note': '聚丙烯-ATR止损'},
    'Y':  {'stop_method': 'percent', 'stop_pct': 3.0, 'note': '豆油-百分比止损'},
    # 默认配置
    'DEFAULT': {'stop_method': 'atr', 'atr_mult': 3.0, 'stop_pct': 3.0, 'chandelier_mult': 3.0},
}


class WaveTrendFinalStrategy(BaseStrategy):
    """WaveTrend最终版策略"""

    name = "wavetrend_final"
    display_name = "WaveTrend Final"
    description = "经过回测优化的WaveTrend策略，各品种自动选择最优止损"
    version = "1.0"
    author = "Eric"

    warmup_num = 60

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            # WaveTrend核心参数
            StrategyParam("n1", "通道长度", 10, 5, 20, 1, "int"),
            StrategyParam("n2", "平均长度", 21, 10, 30, 1, "int"),
            StrategyParam("ob_level", "超买阈值", 53, 40, 70, 1, "int"),
            StrategyParam("os_level", "超卖阈值", -53, -70, -40, 1, "int"),

            # 止损参数
            StrategyParam("atr_len", "ATR周期", 14, 7, 30, 1, "int"),
            StrategyParam("atr_mult", "ATR倍数", 3.0, 1.5, 5.0, 0.5, "float"),
            StrategyParam("stop_pct", "百分比止损", 3.0, 1.0, 10.0, 0.5, "float"),
            StrategyParam("chandelier_bars", "吊灯回看", 22, 10, 50, 5, "int"),
            StrategyParam("chandelier_mult", "吊灯倍数", 3.0, 2.0, 5.0, 0.5, "float"),

            # 止盈参数
            StrategyParam("use_profit_target", "启用WT止盈", True, param_type="bool"),
            StrategyParam("profit_wt_level", "止盈WT阈值", 40, 20, 60, 10, "int"),

            # 自动配置
            StrategyParam("auto_config", "自动选择最优配置", True, param_type="bool",
                         description="根据品种自动选择最优止损方法"),

            # 手动覆盖
            StrategyParam("stop_method", "止损方法(手动)", "auto", param_type="select",
                         options=["auto", "atr", "wt_signal", "percent", "chandelier"]),

            # 通用参数
            StrategyParam("only_long", "只做多", True, param_type="bool"),
            StrategyParam("capital_rate", "资金使用率", 1.0, 0.5, 1.0, 0.1, "float"),
            StrategyParam("risk_rate", "单笔风险", 0.02, 0.01, 0.05, 0.01, "float"),
        ]

    def __init__(self, params=None):
        super().__init__(params)
        self.triggered_profit = False
        self.highest_since_entry = 0
        self.lowest_since_entry = float('inf')
        self.current_symbol = None
        self.effective_stop_method = 'atr'

    def set_symbol(self, symbol: str):
        """设置当前交易品种，用于自动配置"""
        self.current_symbol = symbol.upper()
        if self.params.get('auto_config', True):
            config = SYMBOL_CONFIGS.get(self.current_symbol, SYMBOL_CONFIGS['DEFAULT'])
            self.effective_stop_method = config.get('stop_method', 'atr')
            # 可选：覆盖参数
            if 'atr_mult' in config:
                self.params['atr_mult'] = config['atr_mult']
            if 'stop_pct' in config:
                self.params['stop_pct'] = config['stop_pct']
            if 'chandelier_mult' in config:
                self.params['chandelier_mult'] = config['chandelier_mult']
        else:
            manual = self.params.get('stop_method', 'auto')
            self.effective_stop_method = 'atr' if manual == 'auto' else manual

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算指标"""
        df = df.copy()

        n1 = self.params['n1']
        n2 = self.params['n2']
        atr_len = self.params['atr_len']
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

        # Chandelier
        df['chandelier_high'] = df['high'].rolling(window=chandelier_bars).max()
        df['chandelier_atr'] = df['tr'].rolling(window=chandelier_bars).mean()

        # 信号
        df['cross_up'] = (df['wt1'] > df['wt2']) & (df['wt1'].shift(1) <= df['wt2'].shift(1))
        df['cross_down'] = (df['wt1'] < df['wt2']) & (df['wt1'].shift(1) >= df['wt2'].shift(1))
        df['wt_zero_cross_down'] = (df['wt1'] < 0) & (df['wt1'].shift(1) >= 0)

        return df

    def reset(self):
        super().reset()
        self.triggered_profit = False
        self.highest_since_entry = 0
        self.lowest_since_entry = float('inf')

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """每根K线处理"""

        curr_close = df['close'].iloc[idx]
        curr_high = df['high'].iloc[idx]
        wt1 = df['wt1'].iloc[idx]
        wt2 = df['wt2'].iloc[idx]
        atr = df['atr'].iloc[idx]
        cross_up = df['cross_up'].iloc[idx]
        cross_down = df['cross_down'].iloc[idx]
        wt_zero_cross_down = df['wt_zero_cross_down'].iloc[idx]

        ob_level = self.params['ob_level']
        os_level = self.params['os_level']
        use_profit_target = self.params['use_profit_target']
        profit_wt_level = self.params['profit_wt_level']
        only_long = self.params['only_long']

        if pd.isna(wt1) or pd.isna(wt2) or pd.isna(atr) or atr == 0:
            return None

        # ========== 多头持仓检查 ==========
        if self.position == 1:
            if curr_high > self.highest_since_entry:
                self.highest_since_entry = curr_high

            exit_tag = None

            # 根据止损方法判断
            if self.effective_stop_method == 'wt_signal':
                if cross_down:
                    exit_tag = "wt_death_cross"
                elif wt_zero_cross_down:
                    exit_tag = "wt_zero_cross"

            elif self.effective_stop_method == 'atr':
                stop = self.entry_price - atr * self.params['atr_mult']
                if curr_close < stop:
                    exit_tag = "atr_stop"

            elif self.effective_stop_method == 'percent':
                stop = self.entry_price * (1 - self.params['stop_pct'] / 100)
                if curr_close < stop:
                    exit_tag = "pct_stop"

            elif self.effective_stop_method == 'chandelier':
                chandelier_atr = df['chandelier_atr'].iloc[idx]
                if not pd.isna(chandelier_atr):
                    stop = self.highest_since_entry - chandelier_atr * self.params['chandelier_mult']
                    if curr_close < stop:
                        exit_tag = "chandelier_stop"

            # 止盈逻辑
            if use_profit_target and not exit_tag:
                if wt1 > ob_level:
                    self.triggered_profit = True
                if self.triggered_profit and wt1 < profit_wt_level:
                    exit_tag = "profit_take"

            if exit_tag:
                self.position = 0
                return Signal(action="close", price=curr_close, tag=exit_tag)

        # ========== 空头持仓 ==========
        elif self.position == -1:
            # 简化处理
            exit_tag = None
            if self.effective_stop_method == 'wt_signal':
                if cross_up:
                    exit_tag = "wt_golden_cross"
            else:
                stop = self.entry_price + atr * self.params['atr_mult']
                if curr_close > stop:
                    exit_tag = "stop_loss"

            if exit_tag:
                self.position = 0
                return Signal(action="close", price=curr_close, tag=exit_tag)

        # ========== 开仓信号 ==========
        if self.position == 0:
            if wt1 < os_level and cross_up:
                self.position = 1
                self.entry_price = curr_close
                self.highest_since_entry = curr_high
                self.triggered_profit = False

                stop_loss = curr_close - atr * self.params['atr_mult']

                return Signal(
                    action="buy",
                    price=curr_close,
                    tag="oversold_buy",
                    stop_loss=stop_loss
                )

            if not only_long and wt1 > ob_level and cross_down:
                self.position = -1
                self.entry_price = curr_close
                self.triggered_profit = False

                return Signal(
                    action="sell",
                    price=curr_close,
                    tag="overbought_sell",
                    stop_loss=curr_close + atr * self.params['atr_mult']
                )

        return None


def get_symbol_config(symbol: str) -> Dict:
    """获取品种配置"""
    return SYMBOL_CONFIGS.get(symbol.upper(), SYMBOL_CONFIGS['DEFAULT'])
