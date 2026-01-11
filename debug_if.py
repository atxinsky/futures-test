# coding=utf-8
"""
调试IF回测问题
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

def main():
    print("=" * 60)
    print("调试IF回测问题")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 检查数据格式...")
    from utils.data_loader import load_futures_data

    df = load_futures_data('IF', '2022-01-01', '2025-12-31', '1d', auto_download=False)
    print(f"数据行数: {len(df)}")
    print(f"数据类型:\n{df.dtypes}")
    print(f"数据列: {df.columns.tolist()}")
    print(f"索引类型: {type(df.index)}")
    print(f"\n前5行:\n{df.head()}")
    print(f"\n数据描述:\n{df.describe()}")

    # 检查是否有volume和open_interest
    if 'volume' in df.columns:
        print(f"\n成交量范围: {df['volume'].min()} ~ {df['volume'].max()}")
    if 'open_interest' in df.columns:
        print(f"持仓量范围: {df['open_interest'].min()} ~ {df['open_interest'].max()}")

    # 2. 手动测试策略
    print("\n[2] 测试策略信号生成...")
    from strategies import get_strategy

    strategy_class = get_strategy('brother2v6')
    strategy = strategy_class()

    print(f"策略参数: {strategy.params}")

    # 计算信号
    signals = strategy.generate_signals(df)
    print(f"\n信号统计:")
    print(f"  信号数据类型: {type(signals)}")
    if isinstance(signals, pd.DataFrame):
        print(f"  信号列: {signals.columns.tolist()}")
        if 'signal' in signals.columns:
            print(f"  信号分布:\n{signals['signal'].value_counts()}")
        print(f"  前10行信号:\n{signals.head(10)}")
    elif isinstance(signals, pd.Series):
        print(f"  信号分布:\n{signals.value_counts()}")

    # 3. 测试回测引擎
    print("\n[3] 测试回测引擎...")
    from core.backtest_engine import BacktestEngine

    engine = BacktestEngine()
    result = engine.run(
        strategy=strategy,
        symbol='IF',
        data=df,
        initial_capital=1000000,
        check_limit_price=False
    )

    print(f"\n回测结果:")
    print(f"  总收益率: {result.total_return * 100:.2f}%" if result else "  无结果")
    print(f"  交易次数: {result.total_trades}" if result else "  无结果")
    print(f"  Sharpe: {result.sharpe_ratio:.4f}" if result else "  无结果")

    if result and hasattr(result, 'trades') and result.trades:
        print(f"\n前5笔交易:")
        for i, trade in enumerate(result.trades[:5]):
            print(f"  {i+1}. {trade}")

    # 4. 检查策略的各个过滤条件
    print("\n[4] 检查策略过滤条件...")

    # 计算指标
    import numpy as np

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values if 'volume' in df.columns else np.ones(len(df))

    # EMA
    def ema(arr, period):
        result = np.zeros_like(arr, dtype=float)
        alpha = 2.0 / (period + 1)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
        return result

    sml_len = 12
    big_len = 50
    break_len = 30
    adx_thres = 22
    chop_thres = 50
    vol_multi = 1.3
    vol_len = 20

    sml_ema = ema(close, sml_len)
    big_ema = ema(close, big_len)

    # 趋势条件
    trend_up = sml_ema > big_ema
    print(f"趋势向上天数: {trend_up.sum()} / {len(trend_up)}")

    # N日高点
    break_highs = pd.Series(high).rolling(break_len).max().values
    breakout = close > break_highs
    print(f"突破N日高点天数: {breakout.sum()} / {len(breakout)}")

    # 放量
    vol_ma = pd.Series(volume).rolling(vol_len).mean().values
    vol_ok = volume > vol_ma * vol_multi
    print(f"放量天数: {vol_ok.sum()} / {len(vol_ok)}")

    # 综合条件（简化版，不含ADX/CHOP）
    basic_signal = trend_up & breakout & vol_ok
    print(f"基本信号天数（趋势+突破+放量）: {basic_signal.sum()}")

    # 更宽松的条件
    trend_and_breakout = trend_up & breakout
    print(f"趋势+突破天数: {trend_and_breakout.sum()}")

    # 只看趋势
    print(f"\n最近30天趋势状态:")
    for i in range(-30, 0):
        if len(sml_ema) + i >= 0:
            status = "↑" if sml_ema[i] > big_ema[i] else "↓"
            print(f"  {df.index[i].strftime('%Y-%m-%d')}: close={close[i]:.2f}, sml_ema={sml_ema[i]:.2f}, big_ema={big_ema[i]:.2f} {status}")


if __name__ == '__main__':
    main()
