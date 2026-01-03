# coding=utf-8
"""
Brother2v5 策略
EMA趋势 + ADX过滤 + 突破入场 + ATR移动止损
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .base import BaseStrategy, StrategyParam, Signal


class Brother2v5Strategy(BaseStrategy):
    """Brother2v5 趋势突破策略"""

    name = "brother2v5"
    display_name = "Brother2v5 趋势突破"
    description = """
    经典趋势跟踪策略，结合多重过滤条件：
    - EMA均线判断趋势方向
    - ADX过滤确保趋势强度
    - 价格突破N日高点入场
    - ATR移动止损保护利润

    开多条件：EMA短>EMA长 AND ADX>阈值 AND 收盘>N日High
    平仓条件：EMA短<EMA长 OR 移动止损触发
    """
    version = "5.0"
    author = "BanBot"
    warmup_num = 100

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("sml_len", "短期EMA", 10, 5, 30, 1, "int",
                         description="短期EMA周期，用于判断短期趋势"),
            StrategyParam("big_len", "长期EMA", 40, 20, 100, 5, "int",
                         description="长期EMA周期，用于判断主趋势"),
            StrategyParam("break_len", "突破周期", 40, 10, 100, 5, "int",
                         description="N日High突破周期"),
            StrategyParam("atr_len", "ATR周期", 20, 10, 30, 1, "int",
                         description="ATR计算周期"),
            StrategyParam("adx_len", "ADX周期", 14, 7, 28, 1, "int",
                         description="ADX指标周期"),
            StrategyParam("adx_thres", "ADX阈值", 25.0, 15.0, 40.0, 1.0, "float",
                         description="ADX大于此值才开仓，过滤震荡市"),
            StrategyParam("stop_n", "止损ATR倍数", 4.0, 1.0, 8.0, 0.5, "float",
                         description="移动止损距离=ATR×此倍数"),
            StrategyParam("capital_rate", "资金使用比例", 1.0, 0.1, 1.0, 0.1, "float",
                         description="用于计算仓位的资金比例"),
            StrategyParam("risk_rate", "单次风险比例", 0.03, 0.01, 0.15, 0.01, "float",
                         description="每次交易最大风险占资金比例"),
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        p = self.params

        # EMA
        df['ema_short'] = df['close'].ewm(span=p['sml_len'], adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=p['big_len'], adjust=False).mean()

        # 突破线 (用High)
        df['high_line'] = df['high'].rolling(window=p['break_len']).max()

        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=p['atr_len']).mean()

        # ADX
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        atr_adx = tr.rolling(window=p['adx_len']).mean()
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=p['adx_len']).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=p['adx_len']).mean()

        plus_di = 100 * plus_dm_smooth / (atr_adx + 1e-10)
        minus_di = 100 * minus_dm_smooth / (atr_adx + 1e-10)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = dx.rolling(window=p['adx_len']).mean()

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """每根K线处理"""
        p = self.params

        curr_close = df['close'].iloc[idx]
        curr_high = df['high'].iloc[idx]
        ema_short = df['ema_short'].iloc[idx]
        ema_long = df['ema_long'].iloc[idx]
        high_line_prev = df['high_line'].iloc[idx-1]
        atr = df['atr'].iloc[idx]
        adx = df['adx'].iloc[idx]

        if pd.isna(atr) or pd.isna(adx) or atr == 0:
            return None

        # ========== 持仓检查 ==========
        if self.position == 1:
            # 更新记录最高价
            if curr_high > self.record_high:
                self.record_high = curr_high
            if self.record_high < self.entry_price:
                self.record_high = self.entry_price

            # 趋势反转
            if ema_short < ema_long:
                self.position = 0
                return Signal("close", curr_close, tag="trend_reverse")

            # 移动止损
            stop_line = self.record_high - (atr * p['stop_n'])
            if curr_close < stop_line:
                self.position = 0
                return Signal("close", curr_close, tag="trailing_stop")

        # ========== 开仓信号 ==========
        if self.position == 0:
            is_bullish = ema_short > ema_long
            is_trend_strong = adx > p['adx_thres']
            is_breakout = curr_close > high_line_prev

            if is_bullish and is_trend_strong and is_breakout:
                # 计算仓位
                stake_amt = capital * p['capital_rate']
                risk_per_trade = stake_amt * p['risk_rate']
                stop_dist = atr * p['stop_n']
                if stop_dist <= 0:
                    stop_dist = curr_close * 0.01

                volume = max(1, int(risk_per_trade / stop_dist / 300))  # 除以乘数估算

                self.position = 1
                self.entry_price = curr_close
                self.record_high = curr_high

                return Signal("buy", curr_close, volume, "long_breakout", stop_dist)

        return None
