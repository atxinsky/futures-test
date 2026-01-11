# coding=utf-8
"""
端到端优化测试 - 验证整合后的优化流程
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime

def run_e2e_test():
    """运行端到端测试"""
    print("=" * 60)
    print("E2E Optimization Test (5 trials)")
    print("=" * 60)

    from optimization import OptunaOptimizer, OptimizationConfig, ParamSpaceManager
    from utils.data_loader import load_futures_data
    from strategies.brother2v6 import Brother2v6Strategy
    from core.backtest_engine import BacktestEngine
    import optuna

    # 1. 配置
    config = OptimizationConfig(
        strategy_name="brother2v6",
        symbols=["RB"],
        train_start="2023-01-01",
        train_end="2023-12-31",
        val_start="2024-01-01",
        val_end="2024-06-30",
        timeframe="1h",
        n_trials=5,
        objective="sharpe",
        initial_capital=100000,
        min_trades=1
    )
    print(f"Config: {config.strategy_name}, symbols={config.symbols}, n_trials={config.n_trials}")

    # 2. 加载数据
    print("\nLoading data...")
    df = load_futures_data("RB", config.train_start, config.val_end, auto_download=True)
    if df is None or len(df) == 0:
        print("[SKIP] No data available")
        return None
    print(f"Data loaded: {len(df)} rows, {df.index[0]} ~ {df.index[-1]}")

    # 3. 获取参数空间
    key_params = ParamSpaceManager.get_key_params("brother2v6")
    print(f"Key params: {list(key_params.keys())}")

    # 4. 定义目标函数
    def objective(trial):
        params = {}
        for name, space in key_params.items():
            if space.param_type == 'int':
                params[name] = trial.suggest_int(name, int(space.low), int(space.high))
            else:
                params[name] = trial.suggest_float(name, space.low, space.high)

        # 训练集回测
        train_df = df[(df.index >= config.train_start) & (df.index <= config.train_end)]
        if len(train_df) < 50:
            return -999

        strategy = Brother2v6Strategy(params=params)
        engine = BacktestEngine()
        result = engine.run(
            strategy=strategy,
            symbol="RB",
            data=train_df,
            initial_capital=config.initial_capital,
            check_limit_price=False
        )

        if result is None or result.total_trades == 0:
            return -999

        return result.sharpe_ratio or 0

    # 5. 运行优化
    print("\nRunning optimization...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=config.n_trials, show_progress_bar=False)

    # 6. 结果
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best value (Sharpe): {study.best_value:.4f}")
    print(f"Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 7. 验证集测试
    print("\nValidation test...")
    val_df = df[(df.index >= config.val_start) & (df.index <= config.val_end)]
    if len(val_df) > 0:
        strategy = Brother2v6Strategy(params=study.best_params)
        engine = BacktestEngine()
        val_result = engine.run(
            strategy=strategy,
            symbol="RB",
            data=val_df,
            initial_capital=config.initial_capital,
            check_limit_price=False
        )
        if val_result:
            print(f"Validation Sharpe: {val_result.sharpe_ratio:.4f}")
            print(f"Validation Return: {val_result.total_return*100:.2f}%")

            # 过拟合检测
            if study.best_value > 0:
                decay = (study.best_value - (val_result.sharpe_ratio or 0)) / study.best_value * 100
                print(f"Decay: {decay:.1f}%")
                if decay > 40:
                    print("  -> HIGH overfitting risk")
                elif decay > 20:
                    print("  -> Moderate overfitting risk")
                else:
                    print("  -> Parameters look robust")

    print("\n" + "=" * 60)
    print("E2E Test PASSED")
    print("=" * 60)

    return study.best_params


if __name__ == "__main__":
    run_e2e_test()
