# coding=utf-8
"""
测试修复后的优化器
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    print("=" * 60)
    print("测试修复后的优化器")
    print("=" * 60)

    from optimization import OptunaOptimizer, OptimizationConfig, ParamSpaceManager

    config = OptimizationConfig(
        strategy_name='brother2v6',
        symbols=['IF'],
        train_start='2022-01-01',
        train_end='2024-12-31',
        val_start='2025-01-01',
        val_end='2025-12-31',
        n_trials=20,
        objective='sharpe',
        initial_capital=1000000,
        min_trades=1,
        max_drawdown=0.5
    )

    print(f"训练集: {config.train_start} ~ {config.train_end}")
    print(f"验证集: {config.val_start} ~ {config.val_end}")

    param_spaces = ParamSpaceManager.get_key_params('brother2v6')

    optimizer = OptunaOptimizer(config)

    def progress_cb(p, m):
        if int(p * 100) % 20 == 0:
            print(f"[{p*100:.0f}%] {m}")

    optimizer.set_progress_callback(progress_cb)

    result = optimizer.optimize(param_spaces)

    print("\n" + "=" * 60)
    print("结果")
    print("=" * 60)
    print(f"最优Sharpe: {result.best_value:.4f}")

    if result.train_metrics:
        print(f"\n训练集: {result.train_metrics.get('trades', 0)}笔交易, "
              f"收益率={result.train_metrics.get('return', 0)*100:.2f}%")

    if result.val_metrics:
        print(f"验证集: {result.val_metrics.get('trades', 0)}笔交易, "
              f"收益率={result.val_metrics.get('return', 0)*100:.2f}%")

    print(f"\n期望: 训练集3笔, 验证集3笔")


if __name__ == '__main__':
    main()
