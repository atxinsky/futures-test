# coding=utf-8
"""
BigBrother V14 ETF策略 - 趋势轮动

标的池：
- 513100.SH 纳指ETF (海外科技)
- 513050.SH 中概互联 (中概股)
- 512480.SH 半导体ETF (国产替代)
- 515030.SH 新能车ETF (新能源)
- 518880.SH 黄金ETF (避险资产)
- 512890.SH 红利低波 (防守型)
- 588000.SH 科创50 (硬科技)
- 516010.SH 游戏动漫 (AI应用)
"""

from datetime import timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ETFBigBrotherV14:
    """BigBrother V14 ETF策略"""

    name = "BigBrother_V14"
    description = "ETF趋势轮动策略"

    # 默认ETF池
    DEFAULT_POOL = [
        "513100.SH", "513050.SH", "512480.SH", "515030.SH",
        "518880.SH", "512890.SH", "588000.SH", "516010.SH"
    ]

    # 海外/商品标的（不受A股大盘过滤）
    OVERSEAS = ["513100.SH", "159941.SZ", "518880.SH"]

    # 高波动标的（降低仓位）
    HIGH_VOL = ["588000.SH", "516010.SH", "512480.SH"]

    # A股基准
    BENCHMARK = "000300.SH"

    # 参数定义（用于UI）
    PARAMS = {
        "base_position": {"name": "基础仓位", "default": 0.18, "min": 0.05, "max": 0.30, "step": 0.01},
        "atr_multiplier": {"name": "ATR止损倍数", "default": 2.5, "min": 1.5, "max": 4.0, "step": 0.1},
        "max_loss": {"name": "硬止损比例", "default": 0.07, "min": 0.05, "max": 0.15, "step": 0.01},
        "trail_start": {"name": "追踪止盈触发", "default": 0.15, "min": 0.08, "max": 0.30, "step": 0.01},
        "trail_stop": {"name": "追踪回撤比例", "default": 0.06, "min": 0.03, "max": 0.15, "step": 0.01},
        "max_hold": {"name": "最长持仓天数", "default": 120, "min": 30, "max": 365, "step": 10},
        "cooldown": {"name": "冷却天数", "default": 3, "min": 1, "max": 10, "step": 1},
        "adx_threshold": {"name": "ADX阈值", "default": 20, "min": 15, "max": 30, "step": 1},
    }

    def __init__(self, pool=None, **kwargs):
        self.pool = pool or self.DEFAULT_POOL
        self.params = {k: v.get("default") for k, v in self.PARAMS.items()}
        self.params.update(kwargs)

    def initialize(self, context):
        """初始化策略"""
        context.base_position = self.params.get("base_position", 0.18)
        context.atr_multiplier = self.params.get("atr_multiplier", 2.5)
        context.max_loss = self.params.get("max_loss", 0.07)
        context.trail_start = self.params.get("trail_start", 0.15)
        context.trail_stop = self.params.get("trail_stop", 0.06)
        context.max_hold = self.params.get("max_hold", 120)
        context.cooldown = self.params.get("cooldown", 3)
        context.adx_threshold = self.params.get("adx_threshold", 20)

        context.cooldown_dict = {}
        context.entry_prices = {}
        context.entry_dates = {}
        context.highest = {}
        context.stops = {}
        context.overseas = self.OVERSEAS
        context.high_vol = self.HIGH_VOL

        logger.info(f"BigBrother V14 初始化, 标的池: {len(self.pool)}个")

    def handle_data(self, context, data):
        """每日处理"""
        current_date = context.current_date
        current_dt = context.current_dt

        today_df = context.data
        if today_df is None or len(today_df) == 0:
            return

        positions = context.get_account_positions()

        # A股大盘状态
        bench_df = today_df[today_df["instrument"] == self.BENCHMARK]
        a_market_ok = True

        if len(bench_df) > 0:
            row = bench_df.iloc[0]
            ma20 = row.get("ema_fast", None)
            ma60 = row.get("ema_slow", None)
            close = row["close"]

            if ma20 and ma60 and not pd.isna(ma20) and not pd.isna(ma60):
                if close < ma20 and ma20 < ma60:
                    a_market_ok = False

        # 遍历标的
        for _, row in today_df.iterrows():
            instrument = row["instrument"]

            if instrument == self.BENCHMARK:
                continue

            if instrument not in self.pool:
                continue

            price = row["close"]

            fields = ["ema_fast", "ema_slow", "adx", "atr", "high_20"]
            if any(pd.isna(row.get(f)) for f in fields):
                continue

            ma20 = row["ema_fast"]
            ma60 = row["ema_slow"]
            adx = row["adx"]
            atr = row["atr"]
            high20 = row["high_20"]

            golden = ma20 > ma60
            death = ma20 < ma60

            is_overseas = instrument in context.overseas
            is_high_vol = instrument in context.high_vol

            # 持仓处理
            if instrument in positions and positions[instrument].shares > 0:
                self._handle_position(
                    context, instrument, price, ma20, ma60, adx, atr,
                    golden, death, current_date, current_dt
                )
            # 开仓
            else:
                self._handle_entry(
                    context, instrument, price, ma20, ma60, adx, atr, high20,
                    golden, is_overseas, is_high_vol, a_market_ok, current_dt
                )

    def _handle_position(self, context, instrument, price, ma20, ma60, adx, atr,
                          golden, death, current_date, current_dt):
        """处理持仓"""
        entry = context.entry_prices.get(instrument, price)
        entry_dt = context.entry_dates.get(instrument, current_dt)
        high = context.highest.get(instrument, price)
        stop = context.stops.get(instrument, 0)

        if price > high:
            context.highest[instrument] = price
            high = price

        pnl = (price - entry) / entry
        days = (current_dt - entry_dt).days

        exit_flag = False
        reason = ""

        # 1. 硬止损
        if pnl <= -context.max_loss:
            exit_flag = True
            reason = f"止损{context.max_loss*100:.0f}%"
            context.cooldown_dict[instrument] = current_dt + timedelta(days=context.cooldown)

        # 2. ATR止损
        elif price <= stop:
            exit_flag = True
            reason = "ATR止损"
            context.cooldown_dict[instrument] = current_dt + timedelta(days=context.cooldown)

        # 3. 追踪止盈
        elif pnl >= context.trail_start:
            dd = (high - price) / high
            if dd >= context.trail_stop:
                exit_flag = True
                reason = f"止盈 | 最高:{(high-entry)/entry*100:.0f}%"

        # 4. 长期持仓
        elif days >= context.max_hold and pnl < 0.03:
            exit_flag = True
            reason = f"轮换{days}天"

        # 5. 死叉
        elif death:
            exit_flag = True
            reason = "死叉"

        if exit_flag:
            context.order_target_percent(instrument, 0)
            context.entry_prices.pop(instrument, None)
            context.entry_dates.pop(instrument, None)
            context.highest.pop(instrument, None)
            context.stops.pop(instrument, None)
            logger.info(f"[{current_date}] 卖出 {instrument} | {reason} | 盈亏:{pnl*100:+.1f}%")

    def _handle_entry(self, context, instrument, price, ma20, ma60, adx, atr, high20,
                       golden, is_overseas, is_high_vol, a_market_ok, current_dt):
        """处理开仓"""
        cd = context.cooldown_dict.get(instrument)
        if cd and current_dt < cd:
            return

        # 入场条件
        if not golden:
            return
        if adx < context.adx_threshold:
            return
        if price < high20 * 0.95:
            return
        if not is_overseas and not a_market_ok and adx < 30:
            return

        # 仓位
        vol = atr / price if price > 0 else 0.02
        adj = min(1.0, 0.02 / vol) if vol > 0 else 1.0
        pos = context.base_position * adj

        if is_high_vol:
            pos = pos * 0.7

        pos = max(0.10, min(0.25, pos))

        # 止损价
        atr_stop = price - context.atr_multiplier * atr
        hard_stop = price * (1 - context.max_loss)
        stop = max(atr_stop, hard_stop)

        context.order_target_percent(instrument, pos)
        context.entry_prices[instrument] = price
        context.entry_dates[instrument] = current_dt
        context.highest[instrument] = price
        context.stops[instrument] = stop

        tag = "[海外]" if is_overseas else ("[高波]" if is_high_vol else "")
        logger.info(f"[{context.current_date}] 买入 {instrument} {tag} | 价:{price:.3f} | ADX:{adx:.0f} | 仓位:{pos*100:.0f}%")


# 工厂函数
def create_etf_bigbrother_v14(**kwargs):
    """创建BigBrother V14策略实例"""
    return ETFBigBrotherV14(**kwargs)
