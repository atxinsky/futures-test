# coding=utf-8
"""
简化测试AI参数优化功能 - IF
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    print("=" * 60)
    print("AI参数优化测试 - 沪深300股指期货(IF)")
    print("=" * 60)

    # 1. 配置优化器
    from optimization import OptunaOptimizer, OptimizationConfig, ParamSpaceManager

    config = OptimizationConfig(
        strategy_name='brother2v6',
        symbols=['IF'],
        train_start='2022-01-01',  # IF数据从2022开始
        train_end='2024-06-30',
        val_start='2024-07-01',
        val_end='2025-12-31',
        n_trials=30,
        objective='sharpe',
        initial_capital=1000000,
        min_trades=3,  # 降低最小交易次数要求
        max_drawdown=0.5
    )

    print(f"训练集: {config.train_start} ~ {config.train_end}")
    print(f"验证集: {config.val_start} ~ {config.val_end}")
    print(f"试验次数: {config.n_trials}")

    # 2. 获取关键参数空间
    param_spaces = ParamSpaceManager.get_key_params('brother2v6')
    print(f"关键参数: {list(param_spaces.keys())}")

    # 3. 运行优化
    print("\n开始优化...")

    def progress_callback(progress, message):
        if int(progress * 100) % 10 == 0:
            print(f"  {progress*100:.0f}%: {message}")

    optimizer = OptunaOptimizer(config)
    optimizer.set_progress_callback(progress_callback)

    try:
        result = optimizer.optimize(param_spaces)
    except Exception as e:
        print(f"优化出错: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. 显示结果
    print("\n" + "=" * 60)
    print("优化结果")
    print("=" * 60)

    print(f"最优 {config.objective}: {result.best_value:.4f}")

    print("\n最优参数:")
    for name, value in result.best_params.items():
        print(f"  {name}: {value}")

    if result.train_metrics:
        print("\n训练集:")
        print(f"  Sharpe: {result.train_metrics.get('sharpe', 0):.4f}")
        print(f"  收益率: {result.train_metrics.get('return', 0)*100:.2f}%")
        print(f"  交易次数: {result.train_metrics.get('trades', 0)}")

    if result.val_metrics:
        print("\n验证集:")
        print(f"  Sharpe: {result.val_metrics.get('sharpe', 0):.4f}")
        print(f"  收益率: {result.val_metrics.get('return', 0)*100:.2f}%")
        print(f"  交易次数: {result.val_metrics.get('trades', 0)}")

    if result.param_importance:
        print("\n参数重要性:")
        for name, imp in sorted(result.param_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {name}: {imp:.4f}")

    print("\n测试完成!")


if __name__ == '__main__':
    main()
