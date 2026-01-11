# coding=utf-8
"""
多因子选股 - 因子计算模块

因子分类:
1. 量价因子 (Price-Volume Factors)
2. 动量因子 (Momentum Factors)
3. 波动因子 (Volatility Factors)
4. 技术因子 (Technical Factors)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

logger = logging.getLogger(__name__)

# CPU核心数
CPU_COUNT = multiprocessing.cpu_count()


def calculate_returns(df: pd.DataFrame, periods: list = [1, 5, 10, 20]) -> pd.DataFrame:
    """计算不同周期的收益率"""
    for p in periods:
        df[f"ret_{p}d"] = df["close"].pct_change(p)
    return df


def calculate_momentum_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    动量因子

    - mom_5d: 5日动量
    - mom_10d: 10日动量
    - mom_20d: 20日动量
    - mom_60d: 60日动量 (季度动量)
    """
    df["mom_5d"] = df["close"].pct_change(5)
    df["mom_10d"] = df["close"].pct_change(10)
    df["mom_20d"] = df["close"].pct_change(20)
    df["mom_60d"] = df["close"].pct_change(60)

    # 动量反转因子 (短期反转)
    df["reversal_5d"] = -df["close"].pct_change(5)

    return df


def calculate_volatility_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    波动率因子

    - vol_5d: 5日波动率
    - vol_20d: 20日波动率
    - vol_ratio: 短期/长期波动率比
    """
    df["ret_1d"] = df["close"].pct_change(1)

    df["vol_5d"] = df["ret_1d"].rolling(5).std() * np.sqrt(252)
    df["vol_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
    df["vol_60d"] = df["ret_1d"].rolling(60).std() * np.sqrt(252)

    # 波动率变化
    df["vol_ratio"] = df["vol_5d"] / (df["vol_20d"] + 1e-10)

    return df


def calculate_volume_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    成交量因子

    - vol_ma_ratio: 成交量/20日均量
    - amount_ma_ratio: 成交额/20日均额
    - turn_ma_ratio: 换手率/20日均换手
    """
    df["vol_ma5"] = df["volume"].rolling(5).mean()
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ma_ratio"] = df["volume"] / (df["vol_ma20"] + 1)

    df["amount_ma20"] = df["amount"].rolling(20).mean()
    df["amount_ma_ratio"] = df["amount"] / (df["amount_ma20"] + 1)

    if "turn" in df.columns:
        # 换手率因子（低换手溢价）
        df["turnover"] = df["turn"]  # 当日换手率
        df["turnover_20d"] = df["turn"].rolling(20).mean()  # 20日平均换手率
        df["turnover_60d"] = df["turn"].rolling(60).mean()  # 60日平均换手率
        df["turnover_ratio"] = df["turn"] / (df["turnover_20d"] + 0.01)  # 换手率变化
        df["turnover_std"] = df["turn"].rolling(20).std()  # 换手率波动

    return df


def calculate_price_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    价格因子

    - price_to_high: 距离20日高点
    - price_to_low: 距离20日低点
    - price_position: 价格位置 (0-1)
    """
    df["high_20d"] = df["high"].rolling(20).max()
    df["low_20d"] = df["low"].rolling(20).min()

    df["price_to_high"] = (df["close"] - df["high_20d"]) / df["high_20d"]
    df["price_to_low"] = (df["close"] - df["low_20d"]) / (df["low_20d"] + 0.01)

    # 价格位置 (0=最低, 1=最高)
    price_range = df["high_20d"] - df["low_20d"]
    df["price_position"] = (df["close"] - df["low_20d"]) / (price_range + 0.01)

    return df


def calculate_ma_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    均线因子

    - ma_bias_5: 5日乖离率
    - ma_bias_20: 20日乖离率
    - ma_cross: 均线多头排列
    """
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    # 乖离率
    df["ma_bias_5"] = (df["close"] - df["ma5"]) / df["ma5"]
    df["ma_bias_20"] = (df["close"] - df["ma20"]) / df["ma20"]
    df["ma_bias_60"] = (df["close"] - df["ma60"]) / df["ma60"]

    # 均线多头排列 (ma5 > ma10 > ma20 > ma60)
    df["ma_bull"] = ((df["ma5"] > df["ma10"]) &
                     (df["ma10"] > df["ma20"]) &
                     (df["ma20"] > df["ma60"])).astype(int)

    return df


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """RSI指标"""
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


def calculate_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD指标"""
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    return df


def calculate_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有因子"""
    if len(df) < 60:
        return df

    df = df.copy()

    # 确保基础列存在
    required_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"缺少列: {col}")
            return df

    # 计算各类因子
    df = calculate_momentum_factors(df)
    df = calculate_volatility_factors(df)
    df = calculate_volume_factors(df)
    df = calculate_price_factors(df)
    df = calculate_ma_factors(df)
    df = calculate_rsi(df)
    df = calculate_macd(df)

    return df


def get_factor_list() -> List[str]:
    """获取所有因子名称列表（P0版本：加入换手率因子）"""
    return [
        # 动量因子（保留中长期，去掉短期噪音）
        "mom_20d", "mom_60d",
        # 波动因子（精简）
        "vol_20d", "vol_60d",
        # 换手率因子（新增，A股有效）
        "turnover", "turnover_20d", "turnover_60d", "turnover_ratio",
        # 成交量因子
        "vol_ma_ratio", "amount_ma_ratio",
        # 价格因子
        "price_to_high", "price_to_low", "price_position",
        # 均线因子（精简）
        "ma_bias_20", "ma_bias_60",
        # 技术指标（精简）
        "rsi", "macd_hist"
    ]


def get_factor_list_full() -> List[str]:
    """获取完整因子列表（包含所有因子，用于对比测试）"""
    return [
        # 动量因子
        "mom_5d", "mom_10d", "mom_20d", "mom_60d", "reversal_5d",
        # 波动因子
        "vol_5d", "vol_20d", "vol_60d", "vol_ratio",
        # 换手率因子
        "turnover", "turnover_20d", "turnover_60d", "turnover_ratio", "turnover_std",
        # 成交量因子
        "vol_ma_ratio", "amount_ma_ratio",
        # 价格因子
        "price_to_high", "price_to_low", "price_position",
        # 均线因子
        "ma_bias_5", "ma_bias_20", "ma_bias_60", "ma_bull",
        # 技术指标
        "rsi", "macd", "macd_hist"
    ]


def prepare_factor_data(stock_data: Dict[str, pd.DataFrame],
                        factor_date: str,
                        already_computed: bool = True) -> pd.DataFrame:
    """
    准备某一天的因子截面数据

    Args:
        stock_data: {code: dataframe} 股票数据字典
        factor_date: 因子日期
        already_computed: 是否已经计算过因子（默认True，避免重复计算）

    Returns:
        DataFrame with columns: [code, factor1, factor2, ...]
    """
    rows = []
    factor_cols = get_factor_list()

    for code, df in stock_data.items():
        # 如果因子未计算过，才计算（默认假设已计算过）
        if not already_computed:
            df = calculate_all_factors(df)

        # 获取指定日期的因子值
        day_data = df[df["date"] == factor_date]
        if len(day_data) == 0:
            continue

        row = {"code": code, "date": factor_date}
        for col in factor_cols:
            if col in day_data.columns:
                row[col] = day_data[col].iloc[0]
            else:
                row[col] = np.nan

        # 添加未来收益作为标签
        idx = df[df["date"] == factor_date].index[0]
        if idx + 5 < len(df):
            future_close = df.iloc[idx + 5]["close"]
            current_close = df.iloc[idx]["close"]
            row["future_ret_5d"] = (future_close - current_close) / current_close
        else:
            row["future_ret_5d"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def normalize_factors(df: pd.DataFrame, factor_cols: List[str]) -> pd.DataFrame:
    """
    因子标准化 (截面Z-score)
    """
    df = df.copy()
    for col in factor_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            if std > 0:
                df[col] = (df[col] - mean) / std
            else:
                df[col] = 0
    return df


def _calculate_factors_single(args: Tuple[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    """计算单只股票的因子（用于并行）"""
    code, df = args
    try:
        df_factors = calculate_all_factors(df)
        return code, df_factors
    except Exception as e:
        logger.warning(f"计算{code}因子失败: {e}")
        return code, df


def calculate_factors_parallel(stock_data: Dict[str, pd.DataFrame],
                               n_jobs: int = None) -> Dict[str, pd.DataFrame]:
    """
    并行计算所有股票的因子

    Args:
        stock_data: {code: DataFrame} 股票数据字典
        n_jobs: 并行进程数，默认CPU核心数

    Returns:
        {code: DataFrame} 带因子的股票数据
    """
    if n_jobs is None:
        n_jobs = min(CPU_COUNT, len(stock_data))

    result = {}

    # 使用线程池（因为Pandas操作会释放GIL）
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(_calculate_factors_single, (code, df)): code
            for code, df in stock_data.items()
        }

        for future in as_completed(futures):
            try:
                code, df_factors = future.result()
                result[code] = df_factors
            except Exception as e:
                code = futures[future]
                logger.warning(f"计算{code}因子失败: {e}")
                result[code] = stock_data[code]

    return result


def _prepare_single_date(args: Tuple[str, pd.DataFrame, str, List[str]]) -> dict:
    """准备单只股票单日的因子数据（用于并行）"""
    code, df_factors, factor_date, factor_cols = args

    # 获取指定日期的因子值
    day_data = df_factors[df_factors["date"] == factor_date]
    if len(day_data) == 0:
        return None

    row = {"code": code, "date": factor_date}
    for col in factor_cols:
        if col in day_data.columns:
            row[col] = day_data[col].iloc[0]
        else:
            row[col] = np.nan

    # 添加未来收益作为标签
    idx = df_factors[df_factors["date"] == factor_date].index[0]
    if idx + 5 < len(df_factors):
        future_close = df_factors.iloc[idx + 5]["close"]
        current_close = df_factors.iloc[idx]["close"]
        row["future_ret_5d"] = (future_close - current_close) / current_close
    else:
        row["future_ret_5d"] = np.nan

    return row


def prepare_factor_data_parallel(stock_data: Dict[str, pd.DataFrame],
                                  factor_date: str,
                                  n_jobs: int = None) -> pd.DataFrame:
    """
    并行准备某一天的因子截面数据

    Args:
        stock_data: {code: dataframe} 股票数据字典（已计算因子）
        factor_date: 因子日期
        n_jobs: 并行线程数

    Returns:
        DataFrame with columns: [code, factor1, factor2, ...]
    """
    if n_jobs is None:
        n_jobs = min(CPU_COUNT, len(stock_data))

    factor_cols = get_factor_list()
    rows = []

    # 准备参数
    args_list = [
        (code, df, factor_date, factor_cols)
        for code, df in stock_data.items()
    ]

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(_prepare_single_date, args) for args in args_list]

        for future in as_completed(futures):
            try:
                row = future.result()
                if row is not None:
                    rows.append(row)
            except Exception as e:
                logger.warning(f"准备因子数据失败: {e}")

    return pd.DataFrame(rows)
