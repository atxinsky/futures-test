# coding=utf-8
"""
调试目标函数
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
    from optimization import ParamSpaceManager

    # 配置
    train_start = '2022-01-01'
    train_end = '2024-12-31'
    val_end = '2025-12-31'
    min_trades = 1
    max_drawdown = 0.5

    # 加载数据
    df = load_futures_data('IF', train_start, val_end, '1d', auto_download=False)
    print(f"数据: {len(df)} 行, {df.index.min()} ~ {df.index.max()}")

    strategy_class = get_strategy('brother2v6')
    warmup = strategy_class.warmup_num
    print(f"warmup_num: {warmup}")

    # 获取参数空间
    param_spaces = ParamSpaceManager.get_key_params('brother2v6')
    print(f"参数数量: {len(param_spaces)}")
    for name, space in param_spaces.items():
        print(f"  {name}: [{space.low}, {space.high}] default={space.default}")

    # 使用默认参数
    params = {name: space.default for name, space in param_spaces.items()}
    print(f"\n使用默认参数: {params}")

    # 模拟目标函数
    print("\n" + "=" * 60)
    print("模拟目标函数执行")
    print("=" * 60)

    # 训练集处理
    train_start_ts = pd.Timestamp(train_start)
    train_end_ts = pd.Timestamp(train_end)

    start_idx = df.index.searchsorted(train_start_ts)
    warmup_start_idx = max(0, start_idx - warmup)

    train_df = df.iloc[warmup_start_idx:]
    train_df = train_df[train_df.index <= train_end_ts]

    print(f"训练集: {len(train_df)} 行 (warmup_start_idx={warmup_start_idx})")

    if len(train_df) < warmup + 50:
        print("数据不足，跳过")
        return

    # 执行回测
    strategy = strategy_class(params=params)
    engine = BacktestEngine()
    result = engine.run(
        strategy=strategy,
        symbol='IF',
        data=train_df,
        initial_capital=1000000,
        check_limit_price=False
    )

    print(f"\n回测结果:")
    print(f"  total_trades: {result.total_trades}")
    print(f"  sharpe_ratio: {result.sharpe_ratio}")
    print(f"  total_return: {result.total_return}")
    print(f"  max_drawdown: {result.max_drawdown}")

    # 检查约束
    total_trades = result.total_trades
    max_dd = result.max_drawdown or 0
    avg_sharpe = result.sharpe_ratio or 0

    print(f"\n约束检查:")
    print(f"  total_trades ({total_trades}) < min_trades ({min_trades})? {total_trades < min_trades}")
    print(f"  max_dd ({max_dd:.4f}) > max_drawdown ({max_drawdown})? {max_dd > max_drawdown}")

    if total_trades < min_trades:
        print("  -> 返回 -999 (交易数不足)")
    elif max_dd > max_drawdown:
        print("  -> 返回 -999 (回撤过大)")
    else:
        print(f"  -> 返回 {avg_sharpe:.4f}")

    # 显示交易
    if result.trades:
        print(f"\n交易明细:")
        for i, t in enumerate(result.trades):
            print(f"  {i+1}. {t.entry_time} -> {t.exit_time}: pnl={t.pnl:.2f}")


if __name__ == '__main__':
    main()
