# coding=utf-8
"""
双均线策略
经典的均线金叉死叉策略
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .base import BaseStrategy, StrategyParam, Signal


class DualMAStrategy(BaseStrategy):
    """双均线交叉策略"""

    name = "dual_ma"
    display_name = "双均线交叉"
    description = """
    经典双均线交叉策略：
    - 短期均线上穿长期均线时做多
    - 短期均线下穿长期均线时平仓
    - 可选择SMA或EMA
    - 支持固定止损

    开多条件：短均线上穿长均线
    平仓条件：短均线下穿长均线 OR 止损触发
    """
    version = "1.0"
    author = "BanBot"
    warmup_num = 60

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("fast_len", "快线周期", 10, 5, 30, 1, "int",
                         description="短期均线周期"),
            StrategyParam("slow_len", "慢线周期", 30, 20, 100, 5, "int",
                         description="长期均线周期"),
            StrategyParam("ma_type", "均线类型", "EMA", options=["SMA", "EMA"],
                         param_type="select", description="均线计算方式"),
            StrategyParam("stop_pct", "止损比例%", 3.0, 0.5, 10.0, 0.5, "float",
                         description="固定止损百分比，0表示不止损"),
            StrategyParam("capital_rate", "资金使用比例", 1.0, 0.1, 1.0, 0.1, "float",
                         description="用于计算仓位的资金比例"),
            StrategyParam("risk_rate", "单次风险比例", 0.02, 0.01, 0.10, 0.01, "float",
                         description="每次交易最大风险占资金比例"),
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        p = self.params

        # 根据类型计算均线
        if p['ma_type'] == 'EMA':
            df['ma_fast'] = df['close'].ewm(span=p['fast_len'], adjust=False).mean()
            df['ma_slow'] = df['close'].ewm(span=p['slow_len'], adjust=False).mean()
        else:
            df['ma_fast'] = df['close'].rolling(window=p['fast_len']).mean()
            df['ma_slow'] = df['close'].rolling(window=p['slow_len']).mean()

        # ATR用于仓位计算
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """每根K线处理"""
        p = self.params

        if idx < 2:
            return None

        curr_close = df['close'].iloc[idx]
        ma_fast = df['ma_fast'].iloc[idx]
        ma_slow = df['ma_slow'].iloc[idx]
        ma_fast_prev = df['ma_fast'].iloc[idx-1]
        ma_slow_prev = df['ma_slow'].iloc[idx-1]
        atr = df['atr'].iloc[idx]

        if pd.isna(ma_fast) or pd.isna(ma_slow) or pd.isna(atr):
            return None

        # ========== 持仓检查 ==========
        if self.position == 1:
            # 止损检查
            if p['stop_pct'] > 0:
                stop_price = self.entry_price * (1 - p['stop_pct'] / 100)
                if curr_close < stop_price:
                    self.position = 0
                    return Signal("close", curr_close, tag="stop_loss")

            # 死叉平仓
            if ma_fast < ma_slow and ma_fast_prev >= ma_slow_prev:
                self.position = 0
                return Signal("close", curr_close, tag="ma_cross_down")

        # ========== 开仓信号 ==========
        if self.position == 0:
            # 金叉开仓
            is_cross_up = ma_fast > ma_slow and ma_fast_prev <= ma_slow_prev

            if is_cross_up:
                # 计算仓位
                stake_amt = capital * p['capital_rate']
                risk_per_trade = stake_amt * p['risk_rate']

                if p['stop_pct'] > 0:
                    stop_dist = curr_close * p['stop_pct'] / 100
                else:
                    stop_dist = atr * 2

                volume = max(1, int(risk_per_trade / stop_dist / 300))

                self.position = 1
                self.entry_price = curr_close

                return Signal("buy", curr_close, volume, "ma_cross_up", stop_dist)

        return None
