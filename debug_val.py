# coding=utf-8
"""
调试验证集回测
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

def main():
    from utils.data_loader import load_futures_data
    from strategies import get_strategy
    from core.backtest_engine import BacktestEngine

    df = load_futures_data('IF', '2022-01-01', '2025-12-31', '1d', auto_download=False)
    print(f"全部数据: {len(df)} 行")
    print(f"时间范围: {df.index.min()} ~ {df.index.max()}")

    if 'time' not in df.columns:
        df['time'] = df.index

    strategy_class = get_strategy('brother2v6')
    warmup = strategy_class.warmup_num  # 100
    print(f"策略warmup_num: {warmup}")

    # 模拟优化器的数据筛选 - 验证集
    val_start = '2025-01-01'
    val_end = '2025-12-31'

    val_start_ts = pd.Timestamp(val_start)
    val_end_ts = pd.Timestamp(val_end)

    # 找到开始位置
    start_idx = df.index.searchsorted(val_start_ts)
    print(f"\n验证集起始索引: {start_idx}")
    print(f"验证集起始日期: {df.index[start_idx]}")

    # 向前偏移warmup条
    warmup_start_idx = max(0, start_idx - warmup)
    print(f"预热起始索引: {warmup_start_idx}")
    print(f"预热起始日期: {df.index[warmup_start_idx]}")

    # 筛选数据
    val_df = df.iloc[warmup_start_idx:].copy()
    val_df = val_df[val_df.index <= val_end_ts]
    print(f"验证集（含预热）行数: {len(val_df)}")
    print(f"验证集时间范围: {val_df.index.min()} ~ {val_df.index.max()}")

    # 执行回测
    print("\n" + "=" * 60)
    print("验证集回测（含预热数据）")
    print("=" * 60)

    strategy = strategy_class()
    engine = BacktestEngine()
    result = engine.run(
        strategy=strategy,
        symbol='IF',
        data=val_df,
        initial_capital=1000000,
        check_limit_price=False
    )

    print(f"交易次数: {result.total_trades}")
    print(f"收益率: {result.total_return:.2%}")

    if result.trades:
        print(f"\n交易明细:")
        for i, t in enumerate(result.trades):
            print(f"  {i+1}. {t.entry_time} -> {t.exit_time}: pnl={t.pnl:.2f}")

    # 对比：只用2025年数据（无预热）
    print("\n" + "=" * 60)
    print("验证集回测（无预热数据）")
    print("=" * 60)

    val_df_no_warmup = df[(df.index >= val_start_ts) & (df.index <= val_end_ts)].copy()
    print(f"验证集（无预热）行数: {len(val_df_no_warmup)}")

    strategy2 = strategy_class()
    result2 = engine.run(
        strategy=strategy2,
        symbol='IF',
        data=val_df_no_warmup,
        initial_capital=1000000,
        check_limit_price=False
    )

    print(f"交易次数: {result2.total_trades}")
    if result2.trades:
        print(f"\n交易明细:")
        for i, t in enumerate(result2.trades):
            print(f"  {i+1}. {t.entry_time} -> {t.exit_time}: pnl={t.pnl:.2f}")

    # 期望的交易
    print("\n" + "=" * 60)
    print("期望的2025年交易（从全部数据回测得到）:")
    print("  1. 2025-02-21 -> 2025-03-28")
    print("  2. 2025-07-22 -> 2025-08-01")
    print("  3. 2025-08-18 -> 2025-10-14")


if __name__ == '__main__':
    main()
