# coding=utf-8
"""
BigBrother V14 ETF策略 - 本地Optuna参数优化

运行方式:
    cd D:\期货\回测改造
    python optuna_etf_v14.py
"""

import optuna
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.etf_backtest_engine import ETFBacktestEngine
from strategies.etf_bigbrother_v14 import ETFBigBrotherV14

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============ 配置区 ============

# ETF数据路径（需要你有本地数据）
DATA_DIR = r"D:\期货\回测改造\data\etf"

# 回测区间
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-12-31"
VAL_START = "2024-01-01"
VAL_END = "2025-12-31"

# 优化轮数
N_TRIALS = 100

# 标的池
ETF_POOL = [
    "513100.SH",  # 纳指ETF
    "513050.SH",  # 中概互联
    "512480.SH",  # 半导体
    "515030.SH",  # 新能车
    "518880.SH",  # 黄金ETF
    "512890.SH",  # 红利低波
    "588000.SH",  # 科创50
    "516010.SH",  # 游戏动漫
]

BENCHMARK = "000300.SH"


# ============ 数据加载 ============

def load_etf_data():
    """加载ETF数据"""
    data = {}

    # 方法1: 从本地CSV加载
    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            if filename.endswith('.csv'):
                code = filename.replace('.csv', '')
                filepath = os.path.join(DATA_DIR, filename)
                df = pd.read_csv(filepath)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                data[code] = df
                print(f"加载: {code} - {len(df)}行")

    # 方法2: 使用AKShare获取数据（如果本地没有）
    if not data:
        print("本地数据不存在，尝试从AKShare获取...")
        try:
            import akshare as ak

            for code in ETF_POOL + [BENCHMARK]:
                # 转换代码格式
                ak_code = code.split('.')[0]
                try:
                    if code.endswith('.SH'):
                        df = ak.fund_etf_hist_sina(symbol=f"sh{ak_code}")
                    else:
                        df = ak.fund_etf_hist_sina(symbol=f"sz{ak_code}")

                    df = df.rename(columns={
                        '日期': 'date', '开盘': 'open', '最高': 'high',
                        '最低': 'low', '收盘': 'close', '成交量': 'volume'
                    })
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

                    # 计算指标
                    df = calculate_indicators(df)
                    data[code] = df
                    print(f"下载: {code} - {len(df)}行")
                except Exception as e:
                    print(f"下载 {code} 失败: {e}")

        except ImportError:
            print("请安装akshare: pip install akshare")
            return None

    return data


def calculate_indicators(df):
    """计算策略需要的指标"""
    df = df.copy()

    # EMA
    df['ema_fast'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=60, adjust=False).mean()

    # ATR
    df['tr'] = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    df['atr'] = df['tr'].rolling(14).mean()

    # ADX (简化版)
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']

    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    df['plus_di'] = 100 * df['plus_dm'].ewm(span=14).mean() / df['atr']
    df['minus_di'] = 100 * df['minus_dm'].ewm(span=14).mean() / df['atr']

    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 0.0001)
    df['adx'] = df['dx'].ewm(span=14).mean()

    # 20日高点
    df['high_20'] = df['high'].rolling(20).max().shift(1)

    return df


# ============ 目标函数 ============

def objective(trial: optuna.Trial) -> float:
    """Optuna目标函数"""

    # 定义参数搜索空间
    params = {
        'base_position': trial.suggest_float('base_position', 0.12, 0.25, step=0.02),
        'atr_multiplier': trial.suggest_float('atr_multiplier', 2.0, 3.5, step=0.25),
        'max_loss': trial.suggest_float('max_loss', 0.05, 0.10, step=0.01),
        'trail_start': trial.suggest_float('trail_start', 0.10, 0.20, step=0.02),
        'trail_stop': trial.suggest_float('trail_stop', 0.04, 0.08, step=0.01),
        'max_hold': trial.suggest_int('max_hold', 60, 180, step=30),
        'cooldown': trial.suggest_int('cooldown', 2, 5),
        'adx_threshold': trial.suggest_int('adx_threshold', 15, 25, step=2),
    }

    # 运行回测
    result = run_backtest(params, TRAIN_START, TRAIN_END)

    if result is None:
        return -999

    # 惩罚交易次数太少
    if result.total_trades < 15:
        return -999

    # 惩罚回撤过大
    if result.max_drawdown > 0.35:
        return -999

    return result.sharpe_ratio or 0


def run_backtest(params, start_date, end_date):
    """运行单次回测"""

    global _cached_data
    if '_cached_data' not in globals() or _cached_data is None:
        _cached_data = load_etf_data()

    if _cached_data is None:
        return None

    # 创建策略
    strategy = ETFBigBrotherV14(pool=ETF_POOL, **params)

    # 创建回测引擎
    engine = ETFBacktestEngine(
        initial_capital=1000000,
        commission_rate=0.0001,
        slippage=0.0001
    )

    # 设置策略
    engine.set_strategy(
        initialize=strategy.initialize,
        handle_data=strategy.handle_data
    )

    try:
        result = engine.run(
            data=_cached_data,
            start_date=start_date,
            end_date=end_date
        )
        return result
    except Exception as e:
        logger.warning(f"回测失败: {e}")
        return None


# ============ 主流程 ============

def main():
    print("=" * 60)
    print("BigBrother V14 ETF策略 - 本地Optuna参数优化")
    print("=" * 60)
    print(f"训练集: {TRAIN_START} ~ {TRAIN_END}")
    print(f"验证集: {VAL_START} ~ {VAL_END}")
    print(f"优化轮数: {N_TRIALS}")
    print("=" * 60)

    # 预加载数据
    print("\n加载数据...")
    global _cached_data
    _cached_data = load_etf_data()

    if _cached_data is None or len(_cached_data) == 0:
        print("错误: 无法加载数据")
        print(f"请将ETF数据放在: {DATA_DIR}")
        print("或安装akshare: pip install akshare")
        return

    print(f"加载了 {len(_cached_data)} 个标的的数据")

    # 创建Study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name='bigbrother_v14_etf'
    )

    # 开始优化
    print("\n开始优化...")
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        n_jobs=1
    )

    # 输出结果
    print("\n" + "=" * 60)
    print("优化完成！")
    print("=" * 60)

    best_params = study.best_params
    print(f"\n最优参数 (训练集Sharpe={study.best_value:.3f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # 验证集测试
    print("\n" + "-" * 40)
    print("验证集测试")
    print("-" * 40)

    train_result = run_backtest(best_params, TRAIN_START, TRAIN_END)
    val_result = run_backtest(best_params, VAL_START, VAL_END)

    if train_result and val_result:
        train_sharpe = train_result.sharpe_ratio or 0
        val_sharpe = val_result.sharpe_ratio or 0
        decay = (train_sharpe - val_sharpe) / train_sharpe * 100 if train_sharpe > 0 else 0

        print(f"\n{'指标':<15} {'训练集':<12} {'验证集':<12} {'衰减':<10}")
        print("-" * 50)
        print(f"{'Sharpe':<15} {train_sharpe:<12.3f} {val_sharpe:<12.3f} {decay:>8.1f}%")
        print(f"{'年化收益':<15} {train_result.annual_return*100:<11.1f}% {val_result.annual_return*100:<11.1f}%")
        print(f"{'最大回撤':<15} {train_result.max_drawdown*100:<11.1f}% {val_result.max_drawdown*100:<11.1f}%")
        print(f"{'交易次数':<15} {train_result.total_trades:<12} {val_result.total_trades:<12}")
        print(f"{'胜率':<15} {train_result.win_rate*100:<11.1f}% {val_result.win_rate*100:<11.1f}%")

        # 过拟合判断
        print("\n" + "-" * 40)
        if decay > 40:
            print("⚠️  警告：验证集衰减 > 40%，存在过拟合！")
        elif decay > 20:
            print("⚡ 注意：验证集衰减 20-40%，轻度过拟合")
        else:
            print("✅ 通过：验证集衰减 < 20%，参数较为鲁棒")

    # 参数重要性
    print("\n" + "-" * 40)
    print("参数重要性分析")
    print("-" * 40)

    try:
        importances = optuna.importance.get_param_importances(study)
        for param, importance in sorted(importances.items(), key=lambda x: -x[1]):
            bar = "█" * int(importance * 30)
            print(f"  {param:<20} {importance:.3f} {bar}")
    except Exception as e:
        print(f"  无法计算: {e}")

    # 保存结果
    output_dir = 'optuna_results'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'{output_dir}/etf_v14_{timestamp}.json'

    result_data = {
        'best_params': best_params,
        'train_sharpe': train_sharpe if train_result else 0,
        'val_sharpe': val_sharpe if val_result else 0,
        'decay_pct': decay if train_result and val_result else 0,
        'n_trials': N_TRIALS,
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {result_file}")

    # 生成可复制的参数代码
    print("\n" + "=" * 60)
    print("复制以下代码到策略中使用最优参数:")
    print("=" * 60)
    print(f"""
# BigBrother V14 最优参数 ({timestamp})
strategy = ETFBigBrotherV14(
    pool=ETF_POOL,
    base_position={best_params.get('base_position', 0.18)},
    atr_multiplier={best_params.get('atr_multiplier', 2.5)},
    max_loss={best_params.get('max_loss', 0.07)},
    trail_start={best_params.get('trail_start', 0.15)},
    trail_stop={best_params.get('trail_stop', 0.06)},
    max_hold={best_params.get('max_hold', 120)},
    cooldown={best_params.get('cooldown', 3)},
    adx_threshold={best_params.get('adx_threshold', 20)},
)
""")

    return study, best_params


if __name__ == '__main__':
    main()
