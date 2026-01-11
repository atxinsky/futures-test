# coding=utf-8
"""
测试60m级别BigBrother策略优化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimization import OptunaOptimizer, OptimizationConfig, ParamSpace
from strategies import get_strategy, list_strategies

def main():
    print("=" * 60)
    print("60m级别 Brother2v6 策略参数优化测试")
    print("=" * 60)

    # 1. 检查策略是否存在
    print("\n1. 检查策略注册...")
    strategies = list_strategies()
    print(f"   已注册策略: {len(strategies)} 个")

    strategy_class = get_strategy("brother2v6")
    if strategy_class:
        print(f"   [OK] brother2v6 found: {strategy_class.display_name}")
    else:
        print("   [FAIL] brother2v6 not found!")
        return

    # 2. 检查数据
    print("\n2. 检查60m数据...")
    from utils.data_loader import load_futures_data

    test_symbols = ['RB', 'I', 'MA']
    available_symbols = []

    for symbol in test_symbols:
        df = load_futures_data(symbol, '2023-01-01', '2024-12-31', period='60m', auto_download=False)
        if df is not None and len(df) > 100:
            print(f"   [OK] {symbol}: {len(df)} bars of 60m data")
            available_symbols.append(symbol)
        else:
            print(f"   [FAIL] {symbol}: insufficient data")

    if not available_symbols:
        print("\n   No available data! Please download 60m data first.")
        return

    # 3. 配置优化参数
    print("\n3. 配置优化...")

    config = OptimizationConfig(
        strategy_name='brother2v6',
        symbols=available_symbols[:2],  # 先用2个品种测试
        train_start='2023-01-01',
        train_end='2024-06-30',
        val_start='2024-07-01',
        val_end='2024-12-31',
        n_trials=20,  # 快速测试用20次
        objective='sharpe',
        min_trades=5,
        max_drawdown=0.30,
        initial_capital=100000,
    )
    # 设置timeframe
    config.timeframe = '60m'

    # 4. 定义参数搜索空间（BigBrother核心参数）
    param_spaces = {
        'sml_len': ParamSpace('sml_len', 8, 16, 1, 'int'),      # 短期EMA
        'big_len': ParamSpace('big_len', 40, 60, 5, 'int'),     # 长期EMA
        'break_len': ParamSpace('break_len', 20, 40, 5, 'int'), # 突破周期
        'adx_thres': ParamSpace('adx_thres', 18.0, 28.0, 2.0, 'float'),  # ADX阈值
        'stop_n': ParamSpace('stop_n', 2.0, 4.0, 0.5, 'float'),  # 止损ATR倍数
    }

    print(f"   策略: {config.strategy_name}")
    print(f"   品种: {', '.join(config.symbols)}")
    print(f"   周期: {config.timeframe}")
    print(f"   训练集: {config.train_start} ~ {config.train_end}")
    print(f"   验证集: {config.val_start} ~ {config.val_end}")
    print(f"   优化次数: {config.n_trials}")
    print(f"   优化目标: {config.objective}")

    # 5. 执行优化
    print("\n4. 开始优化...")
    print("-" * 40)

    def progress_callback(progress, message):
        bar_len = 30
        filled = int(bar_len * progress)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f"\r   [{bar}] {progress*100:.0f}% - {message}", end='', flush=True)

    optimizer = OptunaOptimizer(config)
    optimizer.set_progress_callback(progress_callback)

    result = optimizer.optimize(param_spaces)

    print("\n")
    print("-" * 40)

    # 6. 输出结果
    print("\n5. 优化结果:")
    print(f"   最优{config.objective}: {result.best_value:.4f}")
    print(f"\n   最优参数:")
    for name, value in result.best_params.items():
        print(f"      {name}: {value}")

    if result.param_importance:
        print(f"\n   参数重要性:")
        for name, importance in sorted(result.param_importance.items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"      {name}: {importance:.2%}")

    if result.train_metrics:
        print(f"\n   训练集指标:")
        for k, v in result.train_metrics.items():
            print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")

    if result.val_metrics:
        print(f"\n   验证集指标:")
        for k, v in result.val_metrics.items():
            print(f"      {k}: {v:.4f}" if isinstance(v, float) else f"      {k}: {v}")

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
