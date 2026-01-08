# coding=utf-8
"""
全品种日线回测脚本 - Brother2原版策略
多空双向交易
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import datetime
from data_manager import load_data_auto, FUTURES_SYMBOLS
from engine import run_backtest_with_strategy
from strategies.brother2_original import Brother2OriginalStrategy
from config import INSTRUMENTS

# 回测配置
INITIAL_CAPITAL = 1000000  # 初始资金100万
MIN_DATA_DAYS = 500  # 最少需要500天数据

# 要回测的品种分类
SYMBOL_CATEGORIES = {
    "贵金属": ["AU", "AG"],
    "有色金属": ["CU", "AL", "ZN", "NI"],
    "黑色系": ["RB", "HC", "I", "J", "JM"],
    "能源": ["SC", "FU", "BU"],
    "化工": ["L", "V", "PP", "TA", "MA", "SA", "FG", "EG"],
    "油脂油料": ["M", "Y", "P", "OI", "RM"],
    "农产品": ["C", "CF", "SR", "AP"],
}

def run_single_backtest(symbol: str, strategy_params: dict = None) -> dict:
    """运行单个品种回测"""
    # 检查品种是否在配置中
    if symbol not in INSTRUMENTS:
        return {"symbol": symbol, "error": "品种不在配置中"}

    # 加载数据
    df = load_data_auto(symbol, period="1d")
    if df is None or len(df) < MIN_DATA_DAYS:
        return {"symbol": symbol, "error": f"数据不足，需要{MIN_DATA_DAYS}天，实际{len(df) if df is not None else 0}天"}

    # 默认参数（可调整）
    params = {
        "short_n": 10,      # 短期均线
        "long_n": 30,       # 长期均线
        "break_n": 20,      # 突破周期
        "atr_n": 20,        # ATR周期
        "stop_n": 3.0,      # 止损倍数
        "capital_rate": 0.3,
        "risk_rate": 0.02,
    }
    if strategy_params:
        params.update(strategy_params)

    # 创建策略
    strategy = Brother2OriginalStrategy(params)

    try:
        # 运行回测
        result = run_backtest_with_strategy(
            df=df,
            symbol=symbol,
            strategy=strategy,
            initial_capital=INITIAL_CAPITAL
        )

        # 统计多空交易
        long_trades = [t for t in result.trades if t.direction == 1]
        short_trades = [t for t in result.trades if t.direction == -1]

        long_pnl = sum(t.pnl for t in long_trades)
        short_pnl = sum(t.pnl for t in short_trades)

        long_wins = len([t for t in long_trades if t.pnl > 0])
        short_wins = len([t for t in short_trades if t.pnl > 0])

        return {
            "symbol": symbol,
            "name": FUTURES_SYMBOLS.get(symbol, ("未知", "", ""))[0],
            "data_days": len(df),
            "total_return": result.total_return_pct,
            "annual_return": result.annual_return_pct,
            "max_drawdown": result.max_drawdown_pct,
            "sharpe": result.sharpe_ratio,
            "win_rate": result.win_rate,
            "profit_factor": result.profit_factor,
            "total_trades": len(result.trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "long_win_rate": long_wins / len(long_trades) * 100 if long_trades else 0,
            "short_win_rate": short_wins / len(short_trades) * 100 if short_trades else 0,
            "error": None
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def run_all_backtests():
    """运行所有品种回测"""
    results = []

    # 收集所有品种
    all_symbols = []
    for category, symbols in SYMBOL_CATEGORIES.items():
        all_symbols.extend(symbols)

    print(f"=" * 80)
    print(f"Brother2 原版策略 - 全品种日线回测（多空双向）")
    print(f"初始资金: {INITIAL_CAPITAL:,}")
    print(f"品种数量: {len(all_symbols)}")
    print(f"=" * 80)

    for i, symbol in enumerate(all_symbols, 1):
        print(f"[{i}/{len(all_symbols)}] 回测 {symbol}...", end=" ")
        result = run_single_backtest(symbol)

        if result.get("error"):
            print(f"失败: {result['error']}")
        else:
            print(f"收益: {result['total_return']:.1f}%, 回撤: {result['max_drawdown']:.1f}%, "
                  f"交易: {result['total_trades']}笔 (多{result['long_trades']}/空{result['short_trades']})")

        results.append(result)

    return results


def print_summary(results: list):
    """打印汇总结果"""
    # 过滤有效结果
    valid_results = [r for r in results if r.get("error") is None]

    if not valid_results:
        print("没有有效的回测结果")
        return

    # 转为DataFrame
    df = pd.DataFrame(valid_results)

    print(f"\n{'=' * 80}")
    print("回测汇总")
    print(f"{'=' * 80}")

    # 按收益率排序
    df_sorted = df.sort_values("total_return", ascending=False)

    print(f"\n【收益排名 TOP 10】")
    print("-" * 80)
    print(f"{'品种':6} {'名称':10} {'总收益%':>10} {'年化%':>8} {'回撤%':>8} {'夏普':>6} {'胜率%':>6} {'交易':>5} {'多':>4} {'空':>4}")
    print("-" * 80)

    for _, row in df_sorted.head(10).iterrows():
        print(f"{row['symbol']:6} {row['name']:10} {row['total_return']:>10.1f} {row['annual_return']:>8.1f} "
              f"{row['max_drawdown']:>8.1f} {row['sharpe']:>6.2f} {row['win_rate']:>6.1f} "
              f"{row['total_trades']:>5} {row['long_trades']:>4} {row['short_trades']:>4}")

    print(f"\n【收益排名 BOTTOM 10】")
    print("-" * 80)
    for _, row in df_sorted.tail(10).iterrows():
        print(f"{row['symbol']:6} {row['name']:10} {row['total_return']:>10.1f} {row['annual_return']:>8.1f} "
              f"{row['max_drawdown']:>8.1f} {row['sharpe']:>6.2f} {row['win_rate']:>6.1f} "
              f"{row['total_trades']:>5} {row['long_trades']:>4} {row['short_trades']:>4}")

    # 整体统计
    print(f"\n{'=' * 80}")
    print("整体统计")
    print(f"{'=' * 80}")

    profitable = len(df[df['total_return'] > 0])
    print(f"盈利品种: {profitable}/{len(df)} ({profitable/len(df)*100:.1f}%)")
    print(f"平均收益: {df['total_return'].mean():.1f}%")
    print(f"平均年化: {df['annual_return'].mean():.1f}%")
    print(f"平均回撤: {df['max_drawdown'].mean():.1f}%")
    print(f"平均夏普: {df['sharpe'].mean():.2f}")
    print(f"平均胜率: {df['win_rate'].mean():.1f}%")

    # 多空统计
    total_long_pnl = df['long_pnl'].sum()
    total_short_pnl = df['short_pnl'].sum()
    total_long_trades = df['long_trades'].sum()
    total_short_trades = df['short_trades'].sum()

    print(f"\n【多空统计】")
    print(f"多头总盈亏: {total_long_pnl:,.0f} ({total_long_trades}笔)")
    print(f"空头总盈亏: {total_short_pnl:,.0f} ({total_short_trades}笔)")
    print(f"多头平均胜率: {df['long_win_rate'].mean():.1f}%")
    print(f"空头平均胜率: {df['short_win_rate'].mean():.1f}%")

    # 保存结果
    output_file = "backtest_results_original.csv"
    df_sorted.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    results = run_all_backtests()
    print_summary(results)
