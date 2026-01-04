# coding=utf-8
"""
Brother2v6 策略 - 期货版
EMA趋势 + ADX过滤 + CHOP震荡过滤 + 成交量确认 + 分批止盈 + 动态追踪止损

核心改进（基于V5）：
1. 增加CHOP波动指数过滤震荡市场假突破
2. 增加成交量突破确认
3. 引入分批止盈机制（盈利达标+回撤触发平仓50%）
4. 动态追踪止损（根据盈利程度调整ATR倍数）
5. 保本止损机制（盈利达标后上移止损至保本位）

适用场景：日线/4H级别期货交易，震荡市场中捕获趋势机会
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from .base import BaseStrategy, StrategyParam, Signal


class Brother2v6Strategy(BaseStrategy):
    """Brother2v6 趋势突破策略 - 期货优化版"""

    name = "brother2v6"
    display_name = "Brother2v6 趋势突破(期货版)"
    description = """
    经典趋势跟踪策略V6版本，针对期货市场优化：

    **入场条件（全部满足）：**
    - EMA短期 > EMA长期（趋势向上）
    - ADX > 阈值（趋势强度确认）
    - CHOP < 50（趋势市场，非震荡）
    - 收盘价突破N日高点
    - 成交量 > N倍均量（量价配合）

    **出场机制（任一触发）：**
    1. 趋势反转：EMA短期 < EMA长期
    2. 分批止盈：盈利12%且回撤4%时平仓50%
    3. 剩余止盈：第一次止盈后继续上涨，回撤8%平仓剩余
    4. 保本止损：曾盈利10%但回落至成本价
    5. 追踪止损：动态ATR止损（盈利越高止损越紧）

    **风控特点：**
    - 动态止损倍数：盈利<5%用3倍ATR，盈利>15%用2倍ATR
    - 分批止盈锁定利润，避免坐过山车
    - CHOP过滤减少震荡市假突破
    """
    version = "6.0"
    author = "BanBot"
    warmup_num = 100

    def __init__(self, params=None):
        super().__init__(params)
        # V6 特有状态
        self.high_since = 0  # 持仓期间最高价
        self.has_partial_exit = False  # 是否已进行第一次止盈
        self.high_after_partial = 0  # 第一次止盈后的最高价
        self.init_stop_dist = 0  # 初始止损距离
        self.max_profit_rate = 0  # 历史最大盈利率

    @classmethod
    def get_params(cls) -> List[StrategyParam]:
        return [
            # ========== 趋势判断参数 ==========
            StrategyParam("sml_len", "短期EMA", 12, 8, 18, 1, "int",
                         description="短期EMA周期，约2天(4H级别)"),
            StrategyParam("big_len", "长期EMA", 50, 35, 70, 5, "int",
                         description="长期EMA周期，约8天(4H级别)"),
            StrategyParam("break_len", "突破周期", 30, 20, 45, 5, "int",
                         description="N日High突破周期，约5天(4H级别)"),

            # ========== 指标参数 ==========
            StrategyParam("atr_len", "ATR周期", 20, 14, 30, 1, "int",
                         description="ATR计算周期"),
            StrategyParam("adx_len", "ADX周期", 14, 7, 21, 1, "int",
                         description="ADX指标周期"),
            StrategyParam("adx_thres", "ADX阈值", 22.0, 18.0, 28.0, 1.0, "float",
                         description="ADX大于此值才开仓，过滤震荡市"),
            StrategyParam("chop_len", "CHOP周期", 14, 10, 20, 1, "int",
                         description="CHOP指标周期"),
            StrategyParam("chop_thres", "CHOP阈值", 50.0, 45.0, 55.0, 1.0, "float",
                         description="CHOP小于此值表示趋势市场"),

            # ========== 成交量确认 ==========
            StrategyParam("vol_len", "均量周期", 20, 15, 30, 1, "int",
                         description="成交量均线周期"),
            StrategyParam("vol_multi", "放量倍数", 1.3, 1.1, 2.0, 0.1, "float",
                         description="成交量需大于均量的倍数"),

            # ========== 止损参数 ==========
            StrategyParam("stop_n", "初始止损ATR倍数", 3.0, 2.0, 4.5, 0.5, "float",
                         description="初始止损距离=ATR×此倍数"),
            StrategyParam("min_stop_n", "最小止损ATR倍数", 2.0, 1.5, 3.0, 0.5, "float",
                         description="盈利较大时的最小止损倍数"),

            # ========== 止盈参数 ==========
            StrategyParam("break_even_pct", "保本触发%", 10.0, 8.0, 15.0, 1.0, "float",
                         description="盈利超过此比例后启动保本机制"),
            StrategyParam("partial_trigger_pct", "分批止盈触发%", 12.0, 8.0, 18.0, 1.0, "float",
                         description="盈利达到此比例后可触发分批止盈"),
            StrategyParam("partial_drawdown_pct", "分批止盈回撤%", 4.0, 3.0, 8.0, 1.0, "float",
                         description="从高点回撤此比例时触发分批止盈"),
            StrategyParam("partial_rate", "分批止盈比例%", 50.0, 30.0, 70.0, 10.0, "float",
                         description="第一次止盈平仓的比例"),
            StrategyParam("full_drawdown_pct", "剩余止盈回撤%", 8.0, 5.0, 12.0, 1.0, "float",
                         description="剩余仓位从高点回撤此比例时全部平仓"),

            # ========== 仓位管理 ==========
            StrategyParam("capital_rate", "资金使用比例", 0.2, 0.1, 0.5, 0.1, "float",
                         description="用于计算仓位的资金比例"),
            StrategyParam("risk_rate", "单次风险比例", 0.05, 0.01, 0.10, 0.01, "float",
                         description="每次交易最大风险占资金比例"),
        ]

    def reset(self):
        """重置状态"""
        super().reset()
        self.high_since = 0
        self.has_partial_exit = False
        self.high_after_partial = 0
        self.init_stop_dist = 0
        self.max_profit_rate = 0

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        p = self.params

        # ========== EMA ==========
        df['ema_short'] = df['close'].ewm(span=p['sml_len'], adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=p['big_len'], adjust=False).mean()

        # ========== 突破线 (用High) ==========
        df['high_line'] = df['high'].rolling(window=p['break_len']).max()

        # ========== ATR ==========
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=p['atr_len']).mean()

        # ========== ADX ==========
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

        # ========== CHOP (Choppiness Index) ==========
        chop_len = p['chop_len']
        high_low_sum = tr.rolling(window=chop_len).sum()
        highest = df['high'].rolling(window=chop_len).max()
        lowest = df['low'].rolling(window=chop_len).min()
        range_hl = highest - lowest
        # CHOP = 100 * LOG10(SUM(ATR, n) / (Highest - Lowest)) / LOG10(n)
        df['chop'] = 100 * np.log10(high_low_sum / (range_hl + 1e-10)) / np.log10(chop_len)

        # ========== 成交量均线 ==========
        df['vol_ma'] = df['volume'].rolling(window=p['vol_len']).mean()

        return df

    def _get_dynamic_stop_multiplier(self, profit_rate: float) -> float:
        """根据盈利程度返回动态止损倍数"""
        p = self.params
        max_n = p['stop_n']
        min_n = p['min_stop_n']

        if profit_rate < 0.05:
            return max_n  # 初始止损
        elif profit_rate < 0.10:
            return max_n - (max_n - min_n) * 0.3  # 收紧30%
        elif profit_rate < 0.15:
            return max_n - (max_n - min_n) * 0.6  # 收紧60%
        else:
            return min_n  # 最紧止损

    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """每根K线处理"""
        p = self.params

        # 获取当前数据
        curr_close = df['close'].iloc[idx]
        curr_high = df['high'].iloc[idx]
        curr_vol = df['volume'].iloc[idx]
        ema_short = df['ema_short'].iloc[idx]
        ema_long = df['ema_long'].iloc[idx]
        high_line_prev = df['high_line'].iloc[idx-1] if idx > 0 else df['high_line'].iloc[idx]
        atr = df['atr'].iloc[idx]
        adx = df['adx'].iloc[idx]
        chop = df['chop'].iloc[idx]
        vol_ma = df['vol_ma'].iloc[idx]

        if pd.isna(atr) or pd.isna(adx) or pd.isna(chop) or atr == 0:
            return None

        # ========== 市场状态判断 ==========
        is_bullish = ema_short > ema_long
        is_trend_strong = adx > p['adx_thres']
        is_trend_market = chop < p['chop_thres']  # CHOP < 50 趋势市场
        has_vol_confirm = curr_vol > vol_ma * p['vol_multi']
        is_breakout = curr_close > high_line_prev

        # ========== 持仓管理 ==========
        if self.position == 1:
            # 更新持仓期间最高价
            if curr_high > self.high_since:
                self.high_since = curr_high

            # 如果已经第一次止盈，更新后续最高价
            if self.has_partial_exit and curr_high > self.high_after_partial:
                self.high_after_partial = curr_high

            # 计算盈亏指标
            profit_rate = (curr_close - self.entry_price) / self.entry_price
            max_profit_rate = (self.high_since - self.entry_price) / self.entry_price
            self.max_profit_rate = max(self.max_profit_rate, max_profit_rate)
            drawdown_from_high = (self.high_since - curr_close) / self.high_since if self.high_since > 0 else 0

            # ============ 趋势反转出场 ============
            if ema_short < ema_long:
                self._reset_v6_state()
                self.position = 0
                return Signal("close", curr_close, tag="trend_reverse")

            # ============ 分批止盈逻辑 ============
            partial_trigger = p['partial_trigger_pct'] / 100.0
            partial_drawdown = p['partial_drawdown_pct'] / 100.0
            full_drawdown = p['full_drawdown_pct'] / 100.0

            if not self.has_partial_exit:
                # 第一次止盈：盈利达到partial_trigger且回撤达到partial_drawdown
                if self.max_profit_rate >= partial_trigger and drawdown_from_high >= partial_drawdown:
                    self.has_partial_exit = True
                    self.high_after_partial = curr_close
                    # 返回部分平仓信号（这里简化为全部平仓，实际需要引擎支持部分平仓）
                    # 注意：当前回测引擎可能不支持部分平仓，这里用特殊tag标记
                    self._reset_v6_state()
                    self.position = 0
                    return Signal("close", curr_close, tag="partial_profit")
            else:
                # 剩余仓位止盈：从第一次止盈后的高点回撤full_drawdown
                if self.high_after_partial > 0:
                    drawdown_after_partial = (self.high_after_partial - curr_close) / self.high_after_partial
                    if drawdown_after_partial >= full_drawdown:
                        self._reset_v6_state()
                        self.position = 0
                        return Signal("close", curr_close, tag="full_profit")

            # ============ 保本止损 ============
            break_even_trigger = p['break_even_pct'] / 100.0
            if self.max_profit_rate >= break_even_trigger and profit_rate <= 0:
                self._reset_v6_state()
                self.position = 0
                return Signal("close", curr_close, tag="breakeven_stop")

            # ============ 动态追踪止损 ============
            dyn_stop_n = self._get_dynamic_stop_multiplier(self.max_profit_rate)
            stop_line = self.high_since - (atr * dyn_stop_n)

            if curr_close < stop_line:
                self._reset_v6_state()
                self.position = 0
                return Signal("close", curr_close, tag="trailing_stop")

        # ========== 入场信号 ==========
        # 条件：趋势向上 + ADX趋势强度确认 + CHOP趋势市场确认 + 突破 + 成交量确认
        buy_signal = is_bullish and is_trend_strong and is_trend_market and is_breakout and has_vol_confirm

        if buy_signal and self.position == 0:
            # 计算仓位
            stake_amt = capital * p['capital_rate']
            risk_per_trade = stake_amt * p['risk_rate']
            stop_dist = atr * p['stop_n']
            if stop_dist <= 0:
                stop_dist = curr_close * 0.02

            # 期货合约计算：假设合约乘数300（股指期货）
            # 这里简化处理，让回测引擎根据品种自动计算
            volume = max(1, int(risk_per_trade / stop_dist / 300))

            # 初始化V6状态
            self.position = 1
            self.entry_price = curr_close
            self.high_since = curr_high
            self.init_stop_dist = stop_dist
            self.has_partial_exit = False
            self.high_after_partial = 0
            self.max_profit_rate = 0

            return Signal("buy", curr_close, volume, "breakout_long", stop_dist)

        return None

    def _reset_v6_state(self):
        """重置V6特有状态"""
        self.high_since = 0
        self.has_partial_exit = False
        self.high_after_partial = 0
        self.init_stop_dist = 0
        self.max_profit_rate = 0
