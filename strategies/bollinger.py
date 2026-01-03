# coding=utf-8
"""
布林带突破策略
基于布林带通道的突破交易策略
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .base import BaseStrategy, StrategyParam, Signal


class BollingerStrategy(BaseStrategy):
    """布林带突破策略"""

    name = "bollinger"
    display_name = "布林带突破"
    description = """
    布林带突破策略：
    - 价格突破布林带上轨做多
    - 价格回落到中轨或下轨平仓
    - 可配置止损方式

    开多条件：收盘价突破上轨
    平仓条件：收盘价跌破中轨 OR 止损触发
    """
    version = "1.0"
    author = "BanBot"
    warmup_num = 30

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            StrategyParam("bb_len", "布林带周期", 20, 10, 50, 5, "int",
                         description="布林带计算周期"),
            StrategyParam("bb_std", "标准差倍数", 2.0, 1.0, 3.0, 0.5, "float",
                         description="布林带标准差倍数"),
            StrategyParam("exit_line", "平仓线", "middle", options=["middle", "lower"],
                         param_type="select", description="平仓触发线：中轨或下轨"),
            StrategyParam("use_atr_stop", "使用ATR止损", True, param_type="bool",
                         description="是否使用ATR动态止损"),
            StrategyParam("atr_mult", "ATR止损倍数", 2.0, 1.0, 5.0, 0.5, "float",
                         description="ATR止损距离倍数"),
            StrategyParam("capital_rate", "资金使用比例", 1.0, 0.1, 1.0, 0.1, "float",
                         description="用于计算仓位的资金比例"),
            StrategyParam("risk_rate", "单次风险比例", 0.02, 0.01, 0.10, 0.01, "float",
                         description="每次交易最大风险占资金比例"),
        ]

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        p = self.params

        # 布林带
        df['bb_middle'] = df['close'].rolling(window=p['bb_len']).mean()
        bb_std = df['close'].rolling(window=p['bb_len']).std()
        df['bb_upper'] = df['bb_middle'] + bb_std * p['bb_std']
        df['bb_lower'] = df['bb_middle'] - bb_std * p['bb_std']

        # ATR
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

        return df

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """每根K线处理"""
        p = self.params

        curr_close = df['close'].iloc[idx]
        curr_high = df['high'].iloc[idx]
        bb_upper = df['bb_upper'].iloc[idx]
        bb_middle = df['bb_middle'].iloc[idx]
        bb_lower = df['bb_lower'].iloc[idx]
        atr = df['atr'].iloc[idx]

        if pd.isna(bb_upper) or pd.isna(atr) or atr == 0:
            return None

        # ========== 持仓检查 ==========
        if self.position == 1:
            # 更新记录最高价
            if curr_high > self.record_high:
                self.record_high = curr_high

            # ATR止损检查
            if p['use_atr_stop']:
                stop_line = self.record_high - atr * p['atr_mult']
                if curr_close < stop_line:
                    self.position = 0
                    return Signal("close", curr_close, tag="atr_stop")

            # 平仓线检查
            exit_price = bb_middle if p['exit_line'] == 'middle' else bb_lower
            if curr_close < exit_price:
                self.position = 0
                tag = "below_middle" if p['exit_line'] == 'middle' else "below_lower"
                return Signal("close", curr_close, tag=tag)

        # ========== 开仓信号 ==========
        if self.position == 0:
            # 突破上轨
            if curr_close > bb_upper:
                # 计算仓位
                stake_amt = capital * p['capital_rate']
                risk_per_trade = stake_amt * p['risk_rate']

                if p['use_atr_stop']:
                    stop_dist = atr * p['atr_mult']
                else:
                    stop_dist = curr_close - bb_middle

                if stop_dist <= 0:
                    stop_dist = curr_close * 0.02

                volume = max(1, int(risk_per_trade / stop_dist / 300))

                self.position = 1
                self.entry_price = curr_close
                self.record_high = curr_high

                return Signal("buy", curr_close, volume, "breakout_upper", stop_dist)

        return None
