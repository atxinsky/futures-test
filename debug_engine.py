# coding=utf-8
"""
调试回测引擎
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    from utils.data_loader import load_futures_data
    from strategies import get_strategy
    from core.backtest_engine import BacktestEngine

    # 加载全部数据
    df = load_futures_data('IF', '2022-01-01', '2025-12-31', '1d', auto_download=False)
    print(f"全部数据: {len(df)} 行")

    # 添加time列
    if 'time' not in df.columns:
        df['time'] = df.index

    # 测试1：使用全部数据
    print("\n" + "=" * 60)
    print("测试1：全部数据 (2022-2025)")
    print("=" * 60)

    strategy = get_strategy('brother2v6')()
    engine = BacktestEngine()
    result = engine.run(
        strategy=strategy,
        symbol='IF',
        data=df,
        initial_capital=1000000,
        check_limit_price=False
    )

    print(f"交易次数: {result.total_trades}")
    print(f"总收益率: {result.total_return:.2%}")
    print(f"Sharpe: {result.sharpe_ratio:.4f}")
    if result.trades:
        print(f"\n交易明细:")
        for i, t in enumerate(result.trades):
            print(f"  {i+1}. {t.entry_time} -> {t.exit_time}: {t.direction_str} pnl={t.pnl:.2f}")

    # 测试2：只用训练集数据
    print("\n" + "=" * 60)
    print("测试2：训练集 (2022-2024.12)")
    print("=" * 60)

    train_df = df[(df.index >= pd.Timestamp('2022-01-01')) & (df.index <= pd.Timestamp('2024-12-31'))].copy()
    print(f"训练集数据: {len(train_df)} 行")

    strategy2 = get_strategy('brother2v6')()
    result2 = engine.run(
        strategy=strategy2,
        symbol='IF',
        data=train_df,
        initial_capital=1000000,
        check_limit_price=False
    )

    print(f"交易次数: {result2.total_trades}")
    print(f"总收益率: {result2.total_return:.2%}")
    if result2.trades:
        print(f"\n交易明细:")
        for i, t in enumerate(result2.trades):
            print(f"  {i+1}. {t.entry_time} -> {t.exit_time}: {t.direction_str} pnl={t.pnl:.2f}")

    # 测试3：验证集
    print("\n" + "=" * 60)
    print("测试3：验证集 (2025)")
    print("=" * 60)

    val_df = df[(df.index >= pd.Timestamp('2025-01-01')) & (df.index <= pd.Timestamp('2025-12-31'))].copy()
    print(f"验证集数据: {len(val_df)} 行")

    strategy3 = get_strategy('brother2v6')()
    result3 = engine.run(
        strategy=strategy3,
        symbol='IF',
        data=val_df,
        initial_capital=1000000,
        check_limit_price=False
    )

    print(f"交易次数: {result3.total_trades}")
    print(f"总收益率: {result3.total_return:.2%}")
    if result3.trades:
        print(f"\n交易明细:")
        for i, t in enumerate(result3.trades):
            print(f"  {i+1}. {t.entry_time} -> {t.exit_time}: {t.direction_str} pnl={t.pnl:.2f}")

    # 问题分析
    print("\n" + "=" * 60)
    print("问题分析")
    print("=" * 60)
    print(f"全部数据交易次数: {result.total_trades}")
    print(f"训练集交易次数: {result2.total_trades}")
    print(f"验证集交易次数: {result3.total_trades}")
    print(f"训练集+验证集 = {result2.total_trades + result3.total_trades}")

    if result.total_trades != result2.total_trades + result3.total_trades:
        print("\n警告：交易次数不匹配！")
        print("可能原因：")
        print("1. 策略warmup_num=100，短数据集可能没有足够预热")
        print("2. 分割数据导致某些信号丢失")


if __name__ == '__main__':
    main()
