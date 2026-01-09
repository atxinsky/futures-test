# coding=utf-8
"""
BigBrother V17-V21 ETF策略 - Donchian Channel趋势突破

版本说明:
- V17经典版: 8只ETF，risk_per_trade=1%, max_pos=25%
- V19科技版: 9只ETF，risk_per_trade=1.2%, max_pos=22%
- V20均衡版: 5只核心ETF，risk_per_trade=1%, max_pos=30%
- V21防跳空版: 同V20，增加防高开/低开逻辑

核心逻辑:
- 买入: 收盘价 > 20日最高价
- 卖出: 收盘价 < 10日最低价
- 仓位: risk_per_trade / (ATR/价格)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class ETFBigBrotherV17:
    """BigBrother V17 经典版"""

    name = "BigBrother_V17"
    description = "ETF经典趋势突破 - Donchian(20/10)"

    # V17 ETF池
    DEFAULT_POOL = [
        "513100.SH",  # 纳指ETF
        "513050.SH",  # 中概互联
        "513330.SH",  # 恒生互联网
        "512480.SH",  # 半导体
        "515030.SH",  # 新能车
        "512690.SH",  # 酒ETF
        "518880.SH",  # 黄金ETF
        "512890.SH",  # 红利低波
    ]

    PARAMS = {
        "risk_per_trade": {"name": "单笔风险", "default": 0.01, "min": 0.005, "max": 0.03, "step": 0.002},
        "max_position": {"name": "最大仓位", "default": 0.25, "min": 0.10, "max": 0.40, "step": 0.05},
        "donchian_high_period": {"name": "突破周期", "default": 20, "min": 10, "max": 30, "step": 5},
        "donchian_low_period": {"name": "跌破周期", "default": 10, "min": 5, "max": 20, "step": 5},
        "atr_period": {"name": "ATR周期", "default": 14, "min": 7, "max": 21, "step": 7},
    }

    def __init__(self, pool=None, **kwargs):
        self.pool = pool or self.DEFAULT_POOL
        self.params = {k: v.get("default") for k, v in self.PARAMS.items()}
        self.params.update(kwargs)

    def initialize(self, context):
        """初始化策略"""
        context.risk_per_trade = self.params.get("risk_per_trade", 0.01)
        context.max_position = self.params.get("max_position", 0.25)
        context.donchian_high_period = self.params.get("donchian_high_period", 20)
        context.donchian_low_period = self.params.get("donchian_low_period", 10)
        context.atr_period = self.params.get("atr_period", 14)

        context.entry_prices = {}
        context.entry_dates = {}

        logger.info(f"{self.name} 初始化, 标的池: {len(self.pool)}个, "
                   f"风险: {context.risk_per_trade*100:.1f}%, 最大仓位: {context.max_position*100:.0f}%")

    def handle_data(self, context, data):
        """每日处理"""
        current_date = context.current_date
        current_dt = context.current_dt

        today_df = context.data
        if today_df is None or len(today_df) == 0:
            return

        positions = context.get_account_positions()

        for _, row in today_df.iterrows():
            instrument = row["instrument"]

            if instrument not in self.pool:
                continue

            price = row["close"]

            # 检查必需字段
            if pd.isna(row.get("donchian_high")) or pd.isna(row.get("donchian_low")) or pd.isna(row.get("atr")):
                continue

            donchian_high = row["donchian_high"]
            donchian_low = row["donchian_low"]
            atr = row["atr"]

            # 信号判断
            is_buy = price > donchian_high
            is_sell = price < donchian_low

            # 持仓处理
            if instrument in positions and positions[instrument].shares > 0:
                if is_sell:
                    entry = context.entry_prices.get(instrument, price)
                    pnl = (price - entry) / entry
                    context.order_target_percent(instrument, 0)
                    context.entry_prices.pop(instrument, None)
                    context.entry_dates.pop(instrument, None)
                    logger.info(f"[{current_date}] 卖出 {instrument} | 盈亏:{pnl*100:+.1f}%")

            # 开仓处理
            else:
                if is_buy:
                    vol_ratio = atr / price if price > 0 else 0
                    if vol_ratio > 0:
                        target_pct = min(context.max_position, context.risk_per_trade / vol_ratio)
                    else:
                        target_pct = 0.10

                    context.order_target_percent(instrument, target_pct)
                    context.entry_prices[instrument] = price
                    context.entry_dates[instrument] = current_dt
                    logger.info(f"[{current_date}] 买入 {instrument} | 价:{price:.3f} | 仓位:{target_pct*100:.0f}%")


class ETFBigBrotherV19(ETFBigBrotherV17):
    """BigBrother V19 科技版"""

    name = "BigBrother_V19"
    description = "ETF科技趋势突破 - 侧重科技股"

    # V19 ETF池 (增加科创50, 新能源汽车)
    DEFAULT_POOL = [
        "513100.SH",  # 纳指ETF
        "513050.SH",  # 中概互联
        "513330.SH",  # 恒生互联网
        "512480.SH",  # 半导体
        "515030.SH",  # 新能车
        "588000.SH",  # 科创50
        "516010.SH",  # 游戏动漫
        "518880.SH",  # 黄金ETF
        "512890.SH",  # 红利低波
    ]

    PARAMS = {
        "risk_per_trade": {"name": "单笔风险", "default": 0.012, "min": 0.005, "max": 0.03, "step": 0.002},
        "max_position": {"name": "最大仓位", "default": 0.22, "min": 0.10, "max": 0.40, "step": 0.05},
        "donchian_high_period": {"name": "突破周期", "default": 20, "min": 10, "max": 30, "step": 5},
        "donchian_low_period": {"name": "跌破周期", "default": 10, "min": 5, "max": 20, "step": 5},
        "atr_period": {"name": "ATR周期", "default": 14, "min": 7, "max": 21, "step": 7},
    }


class ETFBigBrotherV20(ETFBigBrotherV17):
    """BigBrother V20 均衡配置版"""

    name = "BigBrother_V20"
    description = "ETF均衡配置 - 踢除高波动窄基"

    # V20 精简ETF池 (只保留宽基+防守)
    DEFAULT_POOL = [
        "513100.SH",  # 纳指ETF (美股)
        "513050.SH",  # 中概互联 (中概)
        "588000.SH",  # 科创50 (硬科技)
        "518880.SH",  # 黄金ETF (避险)
        "512890.SH",  # 红利低波 (防守)
    ]

    PARAMS = {
        "risk_per_trade": {"name": "单笔风险", "default": 0.01, "min": 0.005, "max": 0.03, "step": 0.002},
        "max_position": {"name": "最大仓位", "default": 0.30, "min": 0.10, "max": 0.40, "step": 0.05},
        "donchian_high_period": {"name": "突破周期", "default": 20, "min": 10, "max": 30, "step": 5},
        "donchian_low_period": {"name": "跌破周期", "default": 10, "min": 5, "max": 20, "step": 5},
        "atr_period": {"name": "ATR周期", "default": 14, "min": 7, "max": 21, "step": 7},
    }


class ETFBigBrotherV21(ETFBigBrotherV20):
    """BigBrother V21 防跳空稳健版 - 2026-01-09优化参数"""

    name = "BigBrother_V21"
    description = "ETF防跳空版 - VWAP+防高开 (优化参数)"

    # V21 优化后的ETF池 (8个标的)
    DEFAULT_POOL = [
        "513100.SH",  # 纳指ETF
        "513050.SH",  # 中概互联
        "512480.SH",  # 半导体ETF
        "515030.SH",  # 新能车ETF
        "518880.SH",  # 黄金ETF
        "512890.SH",  # 红利低波
        "588000.SH",  # 科创50
        "516010.SH",  # 游戏动漫
    ]

    # 2026-01-09 Optuna优化后的参数 (V2 - 放宽仓位)
    PARAMS = {
        "risk_per_trade": {"name": "单笔风险", "default": 0.018, "min": 0.005, "max": 0.03, "step": 0.002},
        "max_position": {"name": "最大仓位", "default": 0.25, "min": 0.10, "max": 0.50, "step": 0.05},
        "donchian_high_period": {"name": "突破周期", "default": 27, "min": 10, "max": 40, "step": 1},
        "donchian_low_period": {"name": "跌破周期", "default": 13, "min": 5, "max": 25, "step": 1},
        "atr_period": {"name": "ATR周期", "default": 14, "min": 7, "max": 21, "step": 7},
        "gap_up_limit": {"name": "高开限制", "default": 0.025, "min": 0.01, "max": 0.05, "step": 0.005},
        "gap_down_limit": {"name": "低开限制", "default": 0.03, "min": 0.02, "max": 0.05, "step": 0.005},
    }

    def initialize(self, context):
        """初始化策略"""
        super().initialize(context)
        context.gap_up_limit = self.params.get("gap_up_limit", 0.02)
        context.gap_down_limit = self.params.get("gap_down_limit", 0.03)

    def handle_data(self, context, data):
        """每日处理 - 增加防跳空逻辑"""
        current_date = context.current_date
        current_dt = context.current_dt

        today_df = context.data
        if today_df is None or len(today_df) == 0:
            return

        positions = context.get_account_positions()

        for _, row in today_df.iterrows():
            instrument = row["instrument"]

            if instrument not in self.pool:
                continue

            price = row["close"]
            open_price = row.get("open", price)

            # 计算VWAP (如果有成交额和成交量)
            amount = row.get("amount", 0)
            volume = row.get("volume", 0)
            if volume > 0 and amount > 0:
                vwap = amount / volume
            else:
                vwap = price

            # 计算开盘涨跌幅
            prev_close = row.get("prev_close", price)
            if prev_close > 0:
                open_pct = (open_price / prev_close) - 1
            else:
                open_pct = 0

            # 检查必需字段
            if pd.isna(row.get("donchian_high")) or pd.isna(row.get("donchian_low")) or pd.isna(row.get("atr")):
                continue

            donchian_high = row["donchian_high"]
            donchian_low = row["donchian_low"]
            atr = row["atr"]

            # 信号判断
            is_buy = price > donchian_high
            is_sell = price < donchian_low

            # 持仓处理
            if instrument in positions and positions[instrument].shares > 0:
                if is_sell:
                    entry = context.entry_prices.get(instrument, price)
                    pnl = (vwap - entry) / entry  # 使用VWAP计算盈亏
                    context.order_target_percent(instrument, 0)
                    context.entry_prices.pop(instrument, None)
                    context.entry_dates.pop(instrument, None)
                    logger.info(f"[{current_date}] 离场 {instrument} | VWAP:{vwap:.3f} | 开盘跌幅:{open_pct*100:+.1f}% | 盈亏:{pnl*100:+.1f}%")

            # 开仓处理
            else:
                if is_buy:
                    # 防高开: 高开超过限制，放弃买入
                    if open_pct > context.gap_up_limit:
                        logger.info(f"[{current_date}] 放弃追高 {instrument} | 高开:{open_pct*100:+.1f}% > {context.gap_up_limit*100:.0f}%")
                        continue

                    vol_ratio = atr / vwap if vwap > 0 else 0
                    if vol_ratio > 0:
                        target_pct = min(context.max_position, context.risk_per_trade / vol_ratio)
                    else:
                        target_pct = 0.10

                    context.order_target_percent(instrument, target_pct)
                    context.entry_prices[instrument] = vwap  # 使用VWAP作为入场价
                    context.entry_dates[instrument] = current_dt
                    logger.info(f"[{current_date}] 稳健买入 {instrument} | VWAP:{vwap:.3f} | 仓位:{target_pct*100:.0f}%")


# 工厂函数
def create_etf_bigbrother_v17(**kwargs):
    return ETFBigBrotherV17(**kwargs)

def create_etf_bigbrother_v19(**kwargs):
    return ETFBigBrotherV19(**kwargs)

def create_etf_bigbrother_v20(**kwargs):
    return ETFBigBrotherV20(**kwargs)

def create_etf_bigbrother_v21(**kwargs):
    return ETFBigBrotherV21(**kwargs)


# 策略注册表
ETF_STRATEGIES = {
    "BigBrother_V17": ETFBigBrotherV17,
    "BigBrother_V19": ETFBigBrotherV19,
    "BigBrother_V20": ETFBigBrotherV20,
    "BigBrother_V21": ETFBigBrotherV21,
}
