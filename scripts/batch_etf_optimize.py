# coding=utf-8
"""
ETF策略批量优化脚本
自动对BigBrother V14/V17/V21策略进行参数优化
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optuna
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 禁用Optuna的详细日志
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 配置
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2025-01-08"
N_TRIALS = 30  # 每个策略优化30轮
INITIAL_CAPITAL = 1000000

# BigBrother默认池
ETF_POOL = [
    "513100.SH",  # 纳指ETF
    "513050.SH",  # 中概互联
    "512480.SH",  # 半导体ETF
    "515030.SH",  # 新能车ETF
    "518880.SH",  # 黄金ETF
    "512890.SH",  # 红利低波
    "588000.SH",  # 科创50
    "516010.SH",  # 游戏动漫
]


def load_data():
    """加载ETF数据"""
    from core.etf_data_service import get_etf_data_service

    logger.info("加载ETF数据...")
    ds = get_etf_data_service()

    data = {}
    for code in ETF_POOL:
        df = ds.get_data_with_indicators(code, TRAIN_START, VAL_END)
        if len(df) > 0:
            data[code] = df
            logger.info(f"  {code}: {len(df)}行")
        else:
            logger.warning(f"  {code}: 无数据")

    return data


def optimize_v14(data):
    """优化BigBrother V14策略"""
    from core.etf_backtest_engine import ETFBacktestEngine
    from strategies.etf_bigbrother_v14 import ETFBigBrotherV14

    logger.info("\n" + "="*50)
    logger.info("优化 BigBrother V14 (EMA+ADX)")
    logger.info("="*50)

    def objective(trial):
        params = {
            'base_position': trial.suggest_float('base_position', 0.12, 0.28, step=0.02),
            'atr_multiplier': trial.suggest_float('atr_multiplier', 2.0, 3.5, step=0.25),
            'max_loss': trial.suggest_float('max_loss', 0.05, 0.10, step=0.01),
            'trail_start': trial.suggest_float('trail_start', 0.10, 0.20, step=0.02),
            'trail_stop': trial.suggest_float('trail_stop', 0.04, 0.08, step=0.01),
            'adx_threshold': trial.suggest_int('adx_threshold', 15, 25),
        }

        try:
            strategy = ETFBigBrotherV14(pool=ETF_POOL, **params)
            engine = ETFBacktestEngine(initial_capital=INITIAL_CAPITAL, commission_rate=0.0001)
            engine.set_strategy(strategy.initialize, strategy.handle_data)
            result = engine.run(data=data, start_date=TRAIN_START, end_date=TRAIN_END)

            if result.total_trades < 10:
                return -999
            if result.max_drawdown > 0.35:
                return -999

            return result.sharpe_ratio or 0
        except Exception as e:
            logger.warning(f"Trial失败: {e}")
            return -999

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_sharpe = study.best_value

    # 验证集测试
    train_result, val_result = validate_v14(data, best_params)

    return {
        'strategy': 'BigBrother V14 (EMA+ADX)',
        'best_params': best_params,
        'train_sharpe': train_result['sharpe'],
        'train_return': train_result['return'],
        'train_drawdown': train_result['drawdown'],
        'train_trades': train_result['trades'],
        'val_sharpe': val_result['sharpe'],
        'val_return': val_result['return'],
        'val_drawdown': val_result['drawdown'],
        'val_trades': val_result['trades'],
    }


def validate_v14(data, params):
    """验证V14参数"""
    from core.etf_backtest_engine import ETFBacktestEngine
    from strategies.etf_bigbrother_v14 import ETFBigBrotherV14

    results = {}
    for name, start, end in [("train", TRAIN_START, TRAIN_END), ("val", VAL_START, VAL_END)]:
        strategy = ETFBigBrotherV14(pool=ETF_POOL, **params)
        engine = ETFBacktestEngine(initial_capital=INITIAL_CAPITAL, commission_rate=0.0001)
        engine.set_strategy(strategy.initialize, strategy.handle_data)
        result = engine.run(data=data, start_date=start, end_date=end)

        results[name] = {
            'sharpe': result.sharpe_ratio,
            'return': result.total_return,
            'drawdown': result.max_drawdown,
            'trades': result.total_trades,
            'win_rate': result.win_rate
        }

    return results['train'], results['val']


def optimize_v17(data):
    """优化BigBrother V17策略 (Donchian)"""
    from core.etf_backtest_engine import ETFBacktestEngine
    from strategies.etf_bigbrother_v17_v21 import ETFBigBrotherV17

    logger.info("\n" + "="*50)
    logger.info("优化 BigBrother V17 (Donchian经典)")
    logger.info("="*50)

    def objective(trial):
        params = {
            'risk_per_trade': trial.suggest_float('risk_per_trade', 0.008, 0.018, step=0.002),
            'max_position': trial.suggest_float('max_position', 0.20, 0.35, step=0.05),
            'donchian_high_period': trial.suggest_int('donchian_high_period', 15, 30),
            'donchian_low_period': trial.suggest_int('donchian_low_period', 8, 18),
        }

        try:
            strategy = ETFBigBrotherV17(pool=ETF_POOL, **params)
            engine = ETFBacktestEngine(initial_capital=INITIAL_CAPITAL, commission_rate=0.0001)
            engine.set_strategy(strategy.initialize, strategy.handle_data)
            result = engine.run(data=data, start_date=TRAIN_START, end_date=TRAIN_END)

            if result.total_trades < 10:
                return -999
            if result.max_drawdown > 0.35:
                return -999

            return result.sharpe_ratio or 0
        except Exception as e:
            logger.warning(f"Trial失败: {e}")
            return -999

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params

    # 验证集测试
    train_result, val_result = validate_v17(data, best_params)

    return {
        'strategy': 'BigBrother V17 (Donchian经典)',
        'best_params': best_params,
        'train_sharpe': train_result['sharpe'],
        'train_return': train_result['return'],
        'train_drawdown': train_result['drawdown'],
        'train_trades': train_result['trades'],
        'val_sharpe': val_result['sharpe'],
        'val_return': val_result['return'],
        'val_drawdown': val_result['drawdown'],
        'val_trades': val_result['trades'],
    }


def validate_v17(data, params):
    """验证V17参数"""
    from core.etf_backtest_engine import ETFBacktestEngine
    from strategies.etf_bigbrother_v17_v21 import ETFBigBrotherV17

    results = {}
    for name, start, end in [("train", TRAIN_START, TRAIN_END), ("val", VAL_START, VAL_END)]:
        strategy = ETFBigBrotherV17(pool=ETF_POOL, **params)
        engine = ETFBacktestEngine(initial_capital=INITIAL_CAPITAL, commission_rate=0.0001)
        engine.set_strategy(strategy.initialize, strategy.handle_data)
        result = engine.run(data=data, start_date=start, end_date=end)

        results[name] = {
            'sharpe': result.sharpe_ratio,
            'return': result.total_return,
            'drawdown': result.max_drawdown,
            'trades': result.total_trades,
            'win_rate': result.win_rate
        }

    return results['train'], results['val']


def optimize_v21(data):
    """优化BigBrother V21策略 (防跳空)"""
    from core.etf_backtest_engine import ETFBacktestEngine
    from strategies.etf_bigbrother_v17_v21 import ETFBigBrotherV21

    logger.info("\n" + "="*50)
    logger.info("优化 BigBrother V21 (防跳空)")
    logger.info("="*50)

    def objective(trial):
        params = {
            'risk_per_trade': trial.suggest_float('risk_per_trade', 0.008, 0.018, step=0.002),
            'max_position': trial.suggest_float('max_position', 0.20, 0.35, step=0.05),
            'donchian_high_period': trial.suggest_int('donchian_high_period', 15, 30),
            'donchian_low_period': trial.suggest_int('donchian_low_period', 8, 18),
            'gap_up_limit': trial.suggest_float('gap_up_limit', 0.015, 0.035, step=0.005),
        }

        try:
            strategy = ETFBigBrotherV21(pool=ETF_POOL, **params)
            engine = ETFBacktestEngine(initial_capital=INITIAL_CAPITAL, commission_rate=0.0001)
            engine.set_strategy(strategy.initialize, strategy.handle_data)
            result = engine.run(data=data, start_date=TRAIN_START, end_date=TRAIN_END)

            if result.total_trades < 10:
                return -999
            if result.max_drawdown > 0.35:
                return -999

            return result.sharpe_ratio or 0
        except Exception as e:
            logger.warning(f"Trial失败: {e}")
            return -999

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best_params = study.best_params

    # 验证集测试
    train_result, val_result = validate_v21(data, best_params)

    return {
        'strategy': 'BigBrother V21 (防跳空)',
        'best_params': best_params,
        'train_sharpe': train_result['sharpe'],
        'train_return': train_result['return'],
        'train_drawdown': train_result['drawdown'],
        'train_trades': train_result['trades'],
        'val_sharpe': val_result['sharpe'],
        'val_return': val_result['return'],
        'val_drawdown': val_result['drawdown'],
        'val_trades': val_result['trades'],
    }


def validate_v21(data, params):
    """验证V21参数"""
    from core.etf_backtest_engine import ETFBacktestEngine
    from strategies.etf_bigbrother_v17_v21 import ETFBigBrotherV21

    results = {}
    for name, start, end in [("train", TRAIN_START, TRAIN_END), ("val", VAL_START, VAL_END)]:
        strategy = ETFBigBrotherV21(pool=ETF_POOL, **params)
        engine = ETFBacktestEngine(initial_capital=INITIAL_CAPITAL, commission_rate=0.0001)
        engine.set_strategy(strategy.initialize, strategy.handle_data)
        result = engine.run(data=data, start_date=start, end_date=end)

        results[name] = {
            'sharpe': result.sharpe_ratio,
            'return': result.total_return,
            'drawdown': result.max_drawdown,
            'trades': result.total_trades,
            'win_rate': result.win_rate
        }

    return results['train'], results['val']


def generate_report(results):
    """生成Markdown报告"""
    report = f"""# ETF BigBrother策略优化报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 测试配置

| 配置项 | 值 |
|--------|-----|
| 训练集 | {TRAIN_START} ~ {TRAIN_END} |
| 验证集 | {VAL_START} ~ {VAL_END} |
| 初始资金 | ¥{INITIAL_CAPITAL:,} |
| 优化轮数 | {N_TRIALS}轮/策略 |
| ETF池 | {len(ETF_POOL)}个标的 |

### ETF标的池
"""
    for code in ETF_POOL:
        report += f"- {code}\n"

    report += "\n---\n\n## 优化结果汇总\n\n"

    # 汇总表
    report += "| 策略 | 训练Sharpe | 训练收益 | 验证Sharpe | 验证收益 | 衰减 | 评级 |\n"
    report += "|------|-----------|---------|-----------|---------|------|------|\n"

    for r in results:
        decay = (r['train_sharpe'] - r['val_sharpe']) / r['train_sharpe'] * 100 if r['train_sharpe'] > 0 else 0

        if decay < 20 and r['val_sharpe'] > 0.5:
            rating = "⭐⭐⭐ 优秀"
        elif decay < 40 and r['val_sharpe'] > 0:
            rating = "⭐⭐ 良好"
        elif r['val_sharpe'] > 0:
            rating = "⭐ 一般"
        else:
            rating = "❌ 不推荐"

        report += f"| {r['strategy']} | {r['train_sharpe']:.3f} | {r['train_return']*100:.1f}% | {r['val_sharpe']:.3f} | {r['val_return']*100:.1f}% | {decay:.1f}% | {rating} |\n"

    # 详细结果
    report += "\n---\n\n## 详细参数\n\n"

    for r in results:
        report += f"### {r['strategy']}\n\n"
        report += "**最优参数:**\n```python\n"
        for k, v in r['best_params'].items():
            if isinstance(v, float):
                report += f"{k} = {v:.4f}\n"
            else:
                report += f"{k} = {v}\n"
        report += "```\n\n"

        report += "**训练集表现:**\n"
        report += f"- Sharpe: {r['train_sharpe']:.3f}\n"
        report += f"- 收益率: {r['train_return']*100:.2f}%\n"
        report += f"- 最大回撤: {r['train_drawdown']*100:.2f}%\n"
        report += f"- 交易次数: {r['train_trades']}\n\n"

        report += "**验证集表现:**\n"
        report += f"- Sharpe: {r['val_sharpe']:.3f}\n"
        report += f"- 收益率: {r['val_return']*100:.2f}%\n"
        report += f"- 最大回撤: {r['val_drawdown']*100:.2f}%\n"
        report += f"- 交易次数: {r['val_trades']}\n\n"

    # 结论
    report += "---\n\n## 结论与建议\n\n"

    # 找最佳策略
    best = max(results, key=lambda x: x['val_sharpe'])
    report += f"**推荐策略**: {best['strategy']}\n\n"
    report += f"该策略在验证集上取得了最高的Sharpe比率({best['val_sharpe']:.3f})，"
    report += f"验证集收益{best['val_return']*100:.1f}%，最大回撤{best['val_drawdown']*100:.1f}%。\n\n"

    report += "**风险提示**: 历史表现不代表未来收益，建议先用模拟盘验证后再实盘。\n"

    return report


def main():
    """主函数"""
    print("\n" + "="*60)
    print("ETF BigBrother策略批量优化")
    print("="*60 + "\n")

    # 加载数据
    data = load_data()
    if not data:
        print("无法加载数据，退出")
        return

    print(f"\n成功加载 {len(data)} 个ETF数据\n")

    # 运行优化
    results = []

    # V14
    try:
        r = optimize_v14(data)
        results.append(r)
        print(f"\nV14完成: 训练Sharpe={r['train_sharpe']:.3f}, 验证Sharpe={r['val_sharpe']:.3f}")
    except Exception as e:
        print(f"V14优化失败: {e}")

    # V17
    try:
        r = optimize_v17(data)
        results.append(r)
        print(f"\nV17完成: 训练Sharpe={r['train_sharpe']:.3f}, 验证Sharpe={r['val_sharpe']:.3f}")
    except Exception as e:
        print(f"V17优化失败: {e}")

    # V21
    try:
        r = optimize_v21(data)
        results.append(r)
        print(f"\nV21完成: 训练Sharpe={r['train_sharpe']:.3f}, 验证Sharpe={r['val_sharpe']:.3f}")
    except Exception as e:
        print(f"V21优化失败: {e}")

    if not results:
        print("所有优化都失败了")
        return

    # 生成报告
    report = generate_report(results)

    # 保存报告
    report_path = os.path.join(os.path.dirname(__file__), "etf_optimize_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print("\n" + "="*60)
    print("优化完成！")
    print("="*60)
    print(f"\n报告已保存: {report_path}")

    # 打印报告
    print("\n" + report)

    # 返回结果供外部使用
    return results, report


if __name__ == "__main__":
    main()
