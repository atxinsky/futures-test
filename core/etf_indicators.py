# coding=utf-8
"""
ETF技术指标计算模块
"""

import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均线"""
    return series.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均线"""
    return series.rolling(window=period).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均真实波幅"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均趋向指数"""
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    tr = atr(high, low, close, 1)
    atr_smooth = tr.ewm(span=period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_smooth)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    return dx.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指数"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """MACD指标"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def highest(series: pd.Series, period: int) -> pd.Series:
    """N周期最高值"""
    return series.rolling(window=period).max()


def lowest(series: pd.Series, period: int) -> pd.Series:
    """N周期最低值"""
    return series.rolling(window=period).min()


def donchian_channel(high: pd.Series, low: pd.Series, high_period: int = 20, low_period: int = 10) -> tuple:
    """
    Donchian Channel (唐奇安通道)

    Args:
        high: 最高价序列
        low: 最低价序列
        high_period: 上轨周期 (买入突破)
        low_period: 下轨周期 (卖出跌破)

    Returns:
        (donchian_high, donchian_low) - 昨日的通道值
    """
    donchian_high = highest(high, high_period).shift(1)  # 昨日N日最高
    donchian_low = lowest(low, low_period).shift(1)       # 昨日N日最低
    return donchian_high, donchian_low


def calculate_etf_indicators(df: pd.DataFrame, params: dict = None) -> pd.DataFrame:
    """
    计算所有ETF技术指标

    Args:
        df: 包含OHLCV数据的DataFrame
        params: 指标参数字典

    Returns:
        带有技术指标的DataFrame
    """
    if params is None:
        params = {
            "ema_fast": 20,
            "ema_slow": 60,
            "adx_period": 14,
            "atr_period": 14,
            "high_period": 20,
            "donchian_high_period": 20,
            "donchian_low_period": 10,
        }

    df = df.copy()

    # EMA
    df["ema_fast"] = ema(df["close"], params.get("ema_fast", 20))
    df["ema_slow"] = ema(df["close"], params.get("ema_slow", 60))

    # ADX
    df["adx"] = adx(df["high"], df["low"], df["close"], params.get("adx_period", 14))

    # ATR
    df["atr"] = atr(df["high"], df["low"], df["close"], params.get("atr_period", 14))

    # 高低点
    high_period = params.get("high_period", 20)
    df["high_20"] = highest(df["high"], high_period).shift(1)
    df["low_20"] = lowest(df["low"], high_period).shift(1)

    # Donchian Channel (V17-V21策略使用)
    donchian_high_period = params.get("donchian_high_period", 20)
    donchian_low_period = params.get("donchian_low_period", 10)
    df["donchian_high"], df["donchian_low"] = donchian_channel(
        df["high"], df["low"], donchian_high_period, donchian_low_period
    )

    # 昨收价 (V21防跳空策略使用)
    df["prev_close"] = df["close"].shift(1)

    # RSI
    df["rsi"] = rsi(df["close"], 14)

    # MACD
    macd_line, signal_line, hist = macd(df["close"])
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist

    # 金叉死叉信号
    df["golden_cross"] = (df["ema_fast"] > df["ema_slow"]).astype(int)
    df["death_cross"] = (df["ema_fast"] < df["ema_slow"]).astype(int)

    # 波动率
    df["volatility"] = df["atr"] / df["close"]

    return df
