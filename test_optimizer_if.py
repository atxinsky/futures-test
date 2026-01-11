# coding=utf-8
"""
测试AI参数优化功能 - 沪深300股指期货(IF)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from datetime import datetime
import pandas as pd

def main():
    print("=" * 60)
    print("AI参数优化测试 - 沪深300股指期货(IF)")
    print("=" * 60)

    # 1. 检查数据
    print("\n[1] 检查IF数据...")
    from utils.data_loader import load_futures_data

    df = load_futures_data('IF', '2020-01-01', '2025-12-31', '1d', auto_download=False)

    if df is None or len(df) < 100:
        print("  IF数据不足，尝试自动下载...")
        df = load_futures_data('IF', '2020-01-01', '2025-12-31', '1d', auto_download=True)

    if df is None or len(df) < 100:
        print("  无法获取IF数据，请先运行数据下载")
        print("  跳过优化测试，检查数据库状态...")

        # 检查数据库
        import sqlite3
        db_path = os.path.join(os.path.dirname(__file__), "data", "futures_tq.db")
        if os.path.exists(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT symbol, period, COUNT(*) as cnt FROM kline_data GROUP BY symbol, period")
            rows = cursor.fetchall()
            print(f"\n  数据库中已有品种:")
            for row in rows:
                print(f"    {row[0]} ({row[1]}): {row[2]} 条")
            conn.close()
        else:
            print(f"  数据库不存在: {db_path}")
        return

    print(f"  IF数据: {len(df)} 条")
    print(f"  时间范围: {df.index.min()} ~ {df.index.max()}")

    # 2. 检查策略
    print("\n[2] 检查策略...")
    from strategies import get_strategy

    strategy_class = get_strategy('brother2v6')
    if not strategy_class:
        print("  策略 brother2v6 不存在!")
        return
    print(f"  策略 brother2v6 加载成功")

    # 3. 配置优化器
    print("\n[3] 配置优化器...")
    from optimization import OptunaOptimizer, OptimizationConfig, ParamSpaceManager

    config = OptimizationConfig(
        strategy_name='brother2v6',
        symbols=['IF'],
        train_start='2020-01-01',
        train_end='2024-06-30',
        val_start='2024-07-01',
        val_end='2025-12-31',
        n_trials=30,  # 减少试验次数加快测试
        objective='sharpe',
        initial_capital=1000000,  # 股指期货需要更多本金
        min_trades=5,
        max_drawdown=0.4
    )

    print(f"  训练集: {config.train_start} ~ {config.train_end}")
    print(f"  验证集: {config.val_start} ~ {config.val_end}")
    print(f"  试验次数: {config.n_trials}")
    print(f"  优化目标: {config.objective}")

    # 4. 获取关键参数空间（减少搜索维度）
    print("\n[4] 获取参数空间...")
    param_spaces = ParamSpaceManager.get_key_params('brother2v6')
    print(f"  关键参数数量: {len(param_spaces)}")
    for name, space in param_spaces.items():
        print(f"    {name}: [{space.low}, {space.high}] (默认: {space.default})")

    # 5. 运行优化
    print("\n[5] 开始优化...")
    print("-" * 60)

    def progress_callback(progress, message):
        print(f"  进度 {progress*100:.0f}%: {message}")

    optimizer = OptunaOptimizer(config)
    optimizer.set_progress_callback(progress_callback)

    try:
        result = optimizer.optimize(param_spaces)
    except Exception as e:
        print(f"  优化过程出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 6. 显示结果
    print("\n" + "=" * 60)
    print("优化结果")
    print("=" * 60)

    print(f"\n最优 {config.objective}: {result.best_value:.4f}")

    print("\n最优参数:")
    for name, value in result.best_params.items():
        default = param_spaces[name].default
        print(f"  {name}: {value} (默认: {default})")

    if result.train_metrics:
        print("\n训练集表现:")
        print(f"  Sharpe: {result.train_metrics.get('sharpe', 0):.4f}")
        print(f"  收益率: {result.train_metrics.get('return', 0)*100:.2f}%")
        print(f"  最大回撤: {result.train_metrics.get('drawdown', 0)*100:.2f}%")
        print(f"  交易次数: {result.train_metrics.get('trades', 0)}")

    if result.val_metrics:
        print("\n验证集表现:")
        print(f"  Sharpe: {result.val_metrics.get('sharpe', 0):.4f}")
        print(f"  收益率: {result.val_metrics.get('return', 0)*100:.2f}%")
        print(f"  最大回撤: {result.val_metrics.get('drawdown', 0)*100:.2f}%")
        print(f"  交易次数: {result.val_metrics.get('trades', 0)}")

    if result.param_importance:
        print("\n参数重要性排序:")
        sorted_importance = sorted(result.param_importance.items(), key=lambda x: x[1], reverse=True)
        for name, importance in sorted_importance:
            print(f"  {name}: {importance:.4f}")

    # 7. 过拟合检测
    if result.train_metrics and result.val_metrics:
        train_sharpe = result.train_metrics.get('sharpe', 0)
        val_sharpe = result.val_metrics.get('sharpe', 0)

        print("\n过拟合检测:")
        if train_sharpe > 0:
            degradation = (train_sharpe - val_sharpe) / train_sharpe * 100
            print(f"  训练集Sharpe: {train_sharpe:.4f}")
            print(f"  验证集Sharpe: {val_sharpe:.4f}")
            print(f"  性能衰减: {degradation:.1f}%")

            if degradation > 50:
                print("  ⚠️ 警告: 存在明显过拟合风险!")
            elif degradation > 30:
                print("  ⚠️ 注意: 可能存在一定程度过拟合")
            else:
                print("  ✓ 过拟合风险较低")

    print("\n测试完成!")


if __name__ == '__main__':
    main()
