# coding=utf-8
"""
调试优化器的数据处理
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

def main():
    from utils.data_loader import load_futures_data
    from strategies import get_strategy
    from core.backtest_engine import BacktestEngine

    # 模拟优化器的数据加载
    train_start = '2022-01-01'
    train_end = '2024-12-31'
    val_start = '2025-01-01'
    val_end = '2025-12-31'

    # 优化器加载数据的方式
    df = load_futures_data('IF', train_start, val_end, '1d', auto_download=False)
    print(f"优化器加载数据: {len(df)} 行")
    print(f"数据范围: {df.index.min()} ~ {df.index.max()}")

    strategy_class = get_strategy('brother2v6')
    warmup = strategy_class.warmup_num
    print(f"策略warmup_num: {warmup}")

    # 测试训练集处理
    print("\n" + "=" * 60)
    print("训练集处理")
    print("=" * 60)

    start_ts = pd.Timestamp(train_start)
    end_ts = pd.Timestamp(train_end)

    start_idx = df.index.searchsorted(start_ts)
    warmup_start_idx = max(0, start_idx - warmup)

    print(f"train_start索引: {start_idx}")
    print(f"warmup_start索引: {warmup_start_idx}")

    train_df = df.iloc[warmup_start_idx:]
    train_df = train_df[train_df.index <= end_ts]

    print(f"训练集数据行数: {len(train_df)}")
    print(f"训练集范围: {train_df.index.min()} ~ {train_df.index.max()}")

    strategy = strategy_class()
    engine = BacktestEngine()
    result = engine.run(
        strategy=strategy,
        symbol='IF',
        data=train_df,
        initial_capital=1000000,
        check_limit_price=False
    )
    print(f"训练集交易次数: {result.total_trades}")
    if result.trades:
        for i, t in enumerate(result.trades):
            print(f"  {i+1}. {t.entry_time} -> {t.exit_time}")

    # 测试验证集处理
    print("\n" + "=" * 60)
    print("验证集处理")
    print("=" * 60)

    start_ts = pd.Timestamp(val_start)
    end_ts = pd.Timestamp(val_end)

    start_idx = df.index.searchsorted(start_ts)
    warmup_start_idx = max(0, start_idx - warmup)

    print(f"val_start索引: {start_idx}")
    print(f"warmup_start索引: {warmup_start_idx}")

    val_df = df.iloc[warmup_start_idx:]
    val_df = val_df[val_df.index <= end_ts]

    print(f"验证集数据行数: {len(val_df)}")
    print(f"验证集范围: {val_df.index.min()} ~ {val_df.index.max()}")

    strategy2 = strategy_class()
    result2 = engine.run(
        strategy=strategy2,
        symbol='IF',
        data=val_df,
        initial_capital=1000000,
        check_limit_price=False
    )
    print(f"验证集交易次数: {result2.total_trades}")
    if result2.trades:
        for i, t in enumerate(result2.trades):
            print(f"  {i+1}. {t.entry_time} -> {t.exit_time}")


if __name__ == '__main__':
    main()
