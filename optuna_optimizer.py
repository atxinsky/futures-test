# coding=utf-8
"""
Optuna参数优化器
自动搜索策略最优参数，并分析参数稳定性

使用方法:
    python optuna_optimizer.py
"""

import optuna
from optuna.visualization import plot_param_importances, plot_optimization_history
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import os

# 导入你的回测系统
from core.backtest_engine import BacktestEngine
from strategies.wavetrend_final import WaveTrendFinalStrategy

# 可以换成其他策略
# from strategies.brother2v6 import Brother2V6Strategy

logging.basicConfig(level=logging.WARNING)  # 减少回测日志


# ============ 配置区 ============

# 优化目标品种
SYMBOL = 'RB'
PERIOD = '1d'

# 回测区间（训练集）
TRAIN_START = datetime(2019, 1, 1)
TRAIN_END = datetime(2023, 12, 31)

# 验证集（检验过拟合）
VAL_START = datetime(2024, 1, 1)
VAL_END = datetime(2025, 12, 31)

# 初始资金
INITIAL_CAPITAL = 100000.0

# 优化轮数
N_TRIALS = 100  # 100轮通常够用，可以调到200

# 优化目标
OPTIMIZE_TARGET = 'sharpe'  # 可选: 'sharpe', 'calmar', 'return', 'sortino'


# ============ 参数空间定义 ============

def define_param_space(trial: optuna.Trial) -> dict:
    """
    定义参数搜索空间

    这里定义的范围要根据你的策略逻辑合理设置
    """
    params = {
        # WaveTrend核心参数
        'n1': trial.suggest_int('n1', 5, 20),          # 通道长度
        'n2': trial.suggest_int('n2', 10, 30),         # 平均长度
        'ob_level': trial.suggest_int('ob_level', 40, 70),   # 超买阈值
        'os_level': trial.suggest_int('os_level', -70, -40), # 超卖阈值

        # 止损参数
        'atr_len': trial.suggest_int('atr_len', 7, 30),
        'atr_mult': trial.suggest_float('atr_mult', 1.5, 5.0, step=0.5),

        # 止盈参数
        'use_profit_target': trial.suggest_categorical('use_profit_target', [True, False]),
        'profit_wt_level': trial.suggest_int('profit_wt_level', 20, 60, step=10),

        # 固定参数（不优化）
        'auto_config': False,  # 关闭自动配置，使用我们优化的参数
        'stop_method': 'atr',  # 固定止损方法，避免搜索空间过大
        'only_long': True,
        'capital_rate': 1.0,
        'risk_rate': 0.02,
    }
    return params


# ============ 目标函数 ============

def objective(trial: optuna.Trial) -> float:
    """
    Optuna优化的目标函数
    返回值越大越好
    """
    # 获取参数
    params = define_param_space(trial)

    # 创建策略实例
    strategy = WaveTrendFinalStrategy(params=params)

    # 运行回测
    engine = BacktestEngine()
    result = engine.run(
        strategy=strategy,
        symbol=SYMBOL,
        period=PERIOD,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        initial_capital=INITIAL_CAPITAL,
        check_limit_price=False  # 优化时关闭涨跌停检查，加速
    )

    # 提取指标
    sharpe = result.sharpe_ratio or 0
    calmar = result.calmar_ratio or 0
    total_return = result.total_return or 0
    max_drawdown = result.max_drawdown or 0.5
    trade_count = result.total_trades or 0

    # 惩罚交易次数太少的参数组合（统计不显著）
    if trade_count < 20:
        return -999

    # 惩罚回撤过大的组合
    if max_drawdown > 0.4:
        return -999

    # 根据优化目标返回
    if OPTIMIZE_TARGET == 'sharpe':
        return sharpe
    elif OPTIMIZE_TARGET == 'calmar':
        return calmar
    elif OPTIMIZE_TARGET == 'return':
        return total_return
    elif OPTIMIZE_TARGET == 'sortino':
        return result.sortino_ratio or 0
    else:
        return sharpe


# ============ 主流程 ============

def run_optimization():
    """运行优化"""
    print("=" * 60)
    print(f"Optuna参数优化 - {SYMBOL} {PERIOD}")
    print(f"训练集: {TRAIN_START.date()} ~ {TRAIN_END.date()}")
    print(f"验证集: {VAL_START.date()} ~ {VAL_END.date()}")
    print(f"优化目标: {OPTIMIZE_TARGET}")
    print(f"优化轮数: {N_TRIALS}")
    print("=" * 60)

    # 创建Study（使用TPE采样器，比随机搜索更智能）
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name=f'{SYMBOL}_wavetrend_optimize'
    )

    # 开始优化
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        n_jobs=1  # 单线程，避免数据冲突
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("优化完成！")
    print("=" * 60)

    best_params = study.best_params
    best_value = study.best_value

    print(f"\n最优参数 ({OPTIMIZE_TARGET}={best_value:.3f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # ============ 验证集测试 ============
    print("\n" + "-" * 40)
    print("验证集测试（检验过拟合）")
    print("-" * 40)

    # 构建完整参数
    full_params = {
        **best_params,
        'auto_config': False,
        'stop_method': 'atr',
        'only_long': True,
        'capital_rate': 1.0,
        'risk_rate': 0.02,
    }

    strategy = WaveTrendFinalStrategy(params=full_params)
    engine = BacktestEngine()

    # 训练集结果
    train_result = engine.run(
        strategy=strategy,
        symbol=SYMBOL,
        period=PERIOD,
        start_date=TRAIN_START,
        end_date=TRAIN_END,
        initial_capital=INITIAL_CAPITAL
    )

    # 重置策略
    strategy.reset()

    # 验证集结果
    val_result = engine.run(
        strategy=strategy,
        symbol=SYMBOL,
        period=PERIOD,
        start_date=VAL_START,
        end_date=VAL_END,
        initial_capital=INITIAL_CAPITAL
    )

    print(f"\n{'指标':<15} {'训练集':<12} {'验证集':<12} {'衰减':<10}")
    print("-" * 50)

    train_sharpe = train_result.sharpe_ratio or 0
    val_sharpe = val_result.sharpe_ratio or 0
    decay = (train_sharpe - val_sharpe) / train_sharpe * 100 if train_sharpe > 0 else 0

    print(f"{'Sharpe':<15} {train_sharpe:<12.3f} {val_sharpe:<12.3f} {decay:>8.1f}%")
    print(f"{'年化收益':<15} {(train_result.annual_return or 0)*100:<11.1f}% {(val_result.annual_return or 0)*100:<11.1f}%")
    print(f"{'最大回撤':<15} {(train_result.max_drawdown or 0)*100:<11.1f}% {(val_result.max_drawdown or 0)*100:<11.1f}%")
    print(f"{'交易次数':<15} {train_result.total_trades or 0:<12} {val_result.total_trades or 0:<12}")

    # 过拟合判断
    print("\n" + "-" * 40)
    if decay > 40:
        print("⚠️  警告：验证集衰减 > 40%，可能存在过拟合！")
        print("   建议：减少参数数量，或使用更保守的参数范围")
    elif decay > 20:
        print("⚡ 注意：验证集衰减 20-40%，轻度过拟合")
        print("   建议：参数可用，但要持续监控实盘表现")
    else:
        print("✅ 通过：验证集衰减 < 20%，参数较为鲁棒")

    # ============ 参数重要性分析 ============
    print("\n" + "-" * 40)
    print("参数重要性分析")
    print("-" * 40)

    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in sorted(importances.items(), key=lambda x: -x[1]):
            bar = "█" * int(importance * 30)
            print(f"  {param:<20} {importance:.3f} {bar}")
    except Exception as e:
        print(f"  无法计算参数重要性: {e}")

    # ============ 保存结果 ============
    output_dir = 'optuna_results'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'{output_dir}/{SYMBOL}_{timestamp}.json'

    result_data = {
        'symbol': SYMBOL,
        'period': PERIOD,
        'train_range': [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
        'val_range': [VAL_START.isoformat(), VAL_END.isoformat()],
        'best_params': best_params,
        'train_sharpe': train_sharpe,
        'val_sharpe': val_sharpe,
        'decay_pct': decay,
        'n_trials': N_TRIALS,
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {result_file}")

    return study, best_params


# ============ 参数稳定性分析 ============

def analyze_param_stability(study: optuna.Study, param_name: str, best_value, n_neighbors: int = 10):
    """
    分析单个参数的稳定性

    在最优值附近微调，看结果变化多大
    """
    print(f"\n分析参数 {param_name} 的稳定性...")

    # 获取所有试验中该参数的值和对应的目标值
    trials_df = study.trials_dataframe()

    if f'params_{param_name}' not in trials_df.columns:
        print(f"  参数 {param_name} 未在搜索空间中")
        return

    param_values = trials_df[f'params_{param_name}']
    objectives = trials_df['value']

    # 找到最优值附近的试验
    best_idx = objectives.idxmax()

    # 计算参数变化±20%时，目标值的变化
    param_range = param_values.max() - param_values.min()
    tolerance = param_range * 0.2

    nearby_mask = abs(param_values - best_value) < tolerance
    nearby_objectives = objectives[nearby_mask]

    if len(nearby_objectives) > 1:
        stability = nearby_objectives.std() / abs(nearby_objectives.mean()) * 100
        print(f"  参数 ±20% 范围内，目标值变化: {stability:.1f}%")

        if stability < 10:
            print(f"  ✅ 稳定：参数不敏感")
        elif stability < 20:
            print(f"  ⚡ 一般：参数有一定影响")
        else:
            print(f"  ⚠️  敏感：参数影响大，需谨慎")
    else:
        print(f"  数据不足，无法分析稳定性")


if __name__ == '__main__':
    study, best_params = run_optimization()

    # 对关键参数做稳定性分析
    print("\n" + "=" * 60)
    print("参数稳定性分析")
    print("=" * 60)

    for param, value in best_params.items():
        if isinstance(value, (int, float)):
            analyze_param_stability(study, param, value)
