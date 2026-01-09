# coding=utf-8
"""
下载ETF历史数据

运行方式:
    pip install akshare
    python download_etf_data.py
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# 安装检查
try:
    import akshare as ak
except ImportError:
    print("请先安装akshare: pip install akshare")
    exit(1)


# 配置
OUTPUT_DIR = r"D:\期货\回测改造\data\etf"
START_DATE = "20200101"
END_DATE = datetime.now().strftime("%Y%m%d")

# 要下载的ETF
ETF_LIST = {
    # 代码: (市场, 名称)
    "513100": ("sh", "纳指ETF"),
    "513050": ("sh", "中概互联"),
    "512480": ("sh", "半导体ETF"),
    "515030": ("sh", "新能车ETF"),
    "518880": ("sh", "黄金ETF"),
    "512890": ("sh", "红利低波"),
    "588000": ("sh", "科创50"),
    "516010": ("sh", "游戏动漫"),
    "510300": ("sh", "沪深300ETF"),  # 基准
}

# 沪深300指数（用于大盘过滤）
INDEX_LIST = {
    "000300": ("sz", "沪深300"),
}


def calculate_indicators(df):
    """计算技术指标"""
    df = df.copy()

    # EMA
    df['ema_fast'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=60, adjust=False).mean()

    # ATR
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    # ADX
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']

    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    atr_smooth = df['atr'].replace(0, np.nan)
    df['plus_di'] = 100 * pd.Series(df['plus_dm']).ewm(span=14).mean() / atr_smooth
    df['minus_di'] = 100 * pd.Series(df['minus_dm']).ewm(span=14).mean() / atr_smooth

    di_sum = df['plus_di'] + df['minus_di']
    di_sum = di_sum.replace(0, np.nan)
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / di_sum
    df['adx'] = df['dx'].ewm(span=14).mean()

    # 20日高点
    df['high_20'] = df['high'].rolling(20).max().shift(1)

    return df


def download_etf(code, market, name):
    """下载单个ETF"""
    print(f"下载 {name} ({code})...", end=" ")

    try:
        # 使用东方财富数据源
        symbol = f"{code}"
        df = ak.fund_etf_hist_em(
            symbol=symbol,
            period="daily",
            start_date=START_DATE,
            end_date=END_DATE,
            adjust="qfq"  # 前复权
        )

        if df is None or len(df) == 0:
            print("无数据")
            return None

        # 重命名列
        df = df.rename(columns={
            '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '成交额': 'amount',
        })

        # 只保留需要的列
        cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]

        # 转换日期格式
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

        # 计算指标
        df = calculate_indicators(df)

        print(f"{len(df)}行")
        return df

    except Exception as e:
        print(f"失败: {e}")
        return None


def download_index(code, market, name):
    """下载指数"""
    print(f"下载 {name} ({code})...", end=" ")

    try:
        df = ak.stock_zh_index_daily(symbol=f"{market}{code}")

        if df is None or len(df) == 0:
            print("无数据")
            return None

        df = df.rename(columns={
            'date': 'date',
            'open': 'open',
            'close': 'close',
            'high': 'high',
            'low': 'low',
            'volume': 'volume',
        })

        # 过滤日期
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= START_DATE]
        df = df[df['date'] <= END_DATE]
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

        # 计算指标
        df = calculate_indicators(df)

        print(f"{len(df)}行")
        return df

    except Exception as e:
        print(f"失败: {e}")
        return None


def main():
    print("=" * 50)
    print("ETF数据下载工具")
    print("=" * 50)
    print(f"数据范围: {START_DATE} ~ {END_DATE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 50)

    # 创建目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    success = 0
    failed = 0

    # 下载ETF
    print("\n下载ETF数据:")
    print("-" * 30)
    for code, (market, name) in ETF_LIST.items():
        df = download_etf(code, market, name)
        if df is not None:
            # 保存
            suffix = ".SH" if market == "sh" else ".SZ"
            filepath = os.path.join(OUTPUT_DIR, f"{code}{suffix}.csv")
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            success += 1
        else:
            failed += 1

    # 下载指数
    print("\n下载指数数据:")
    print("-" * 30)
    for code, (market, name) in INDEX_LIST.items():
        df = download_index(code, market, name)
        if df is not None:
            suffix = ".SH" if market == "sh" else ".SZ"
            filepath = os.path.join(OUTPUT_DIR, f"{code}{suffix}.csv")
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            success += 1
        else:
            failed += 1

    print("\n" + "=" * 50)
    print(f"完成! 成功: {success}, 失败: {failed}")
    print(f"数据保存在: {OUTPUT_DIR}")
    print("=" * 50)

    # 检查数据
    print("\n数据预览:")
    print("-" * 30)
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(OUTPUT_DIR, f))
            print(f"{f}: {len(df)}行, {df['date'].min()} ~ {df['date'].max()}")


if __name__ == '__main__':
    main()
