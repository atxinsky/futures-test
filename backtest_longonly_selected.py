# coding=utf-8
"""
精选品种组合回测 - 只做多
选择V6双向版表现好的品种，只做多
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
from data_manager import load_data_auto, FUTURES_SYMBOLS
from engine import run_backtest_with_strategy, BacktestResult
from strategies.brother2v6_dual import Brother2v6DualStrategy
from config import INSTRUMENTS

INITIAL_CAPITAL = 1000000
MIN_DATA_DAYS = 500

# 精选品种（根据V6双向版回测结果筛选）
SELECTED_SYMBOLS = {
    "贵金属": ["AU", "AG"],           # 表现最好
    "有色金属": ["CU"],                # 铜还不错
    "黑色系": ["RB", "J", "JM", "FG"], # 黑色系部分品种
    "化工": ["PP", "TA", "CF"],        # 化工部分
    "农产品": ["SR", "C"],             # 白糖、玉米
}


class LongOnlyStrategy(Brother2v6DualStrategy):
    """只做多版本"""
    name = "brother2v6_longonly"
    display_name = "Brother2v6 只做多"

    def on_bar(self, idx, df, capital):
        """只返回做多信号"""
        signal = super().on_bar(idx, df, capital)

        # 过滤掉做空信号
        if signal and signal.action == "sell":
            self.position = 0  # 重置持仓状态
            self._reset_position_state()
            return None

        return signal


def run_single_backtest(symbol: str) -> dict:
    if symbol not in INSTRUMENTS:
        return {"symbol": symbol, "error": "品种不在配置中"}

    df = load_data_auto(symbol, period="1d")
    if df is None or len(df) < MIN_DATA_DAYS:
        return {"symbol": symbol, "error": f"数据不足"}

    params = {
        "sml_len": 12,
        "big_len": 50,
        "break_len": 30,
        "atr_len": 20,
        "adx_len": 14,
        "adx_thres": 22.0,
        "chop_len": 14,
        "chop_thres": 50.0,
        "vol_len": 20,
        "vol_multi": 1.3,
        "stop_n": 3.0,
        "capital_rate": 0.3,
        "risk_rate": 0.02,
    }

    strategy = LongOnlyStrategy(params)

    try:
        result = run_backtest_with_strategy(
            df=df,
            symbol=symbol,
            strategy=strategy,
            initial_capital=INITIAL_CAPITAL
        )

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
            "total_pnl": result.total_pnl,
            "error": None
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def run_portfolio_backtest():
    """运行组合回测"""
    results = []
    all_symbols = []
    for category, symbols in SELECTED_SYMBOLS.items():
        all_symbols.extend(symbols)

    print(f"=" * 80)
    print(f"精选品种组合回测 - 只做多")
    print(f"参数: EMA(12/50), Break(30), ADX>22, CHOP<50, Vol>1.3x, Stop=3×ATR")
    print(f"初始资金: {INITIAL_CAPITAL:,}")
    print(f"精选品种: {len(all_symbols)}个")
    print(f"品种列表: {', '.join(all_symbols)}")
    print(f"=" * 80)

    for i, symbol in enumerate(all_symbols, 1):
        print(f"[{i}/{len(all_symbols)}] 回测 {symbol}...", end=" ")
        result = run_single_backtest(symbol)

        if result.get("error"):
            print(f"失败: {result['error']}")
        else:
            print(f"收益: {result['total_return']:.1f}%, 回撤: {result['max_drawdown']:.1f}%, "
                  f"交易: {result['total_trades']}笔, 胜率: {result['win_rate']:.1f}%")

        results.append(result)

    return results


def print_summary(results: list):
    valid_results = [r for r in results if r.get("error") is None]
    if not valid_results:
        print("没有有效的回测结果")
        return

    df = pd.DataFrame(valid_results)
    df_sorted = df.sort_values("total_return", ascending=False)

    print(f"\n{'=' * 80}")
    print("精选品种回测汇总（只做多）")
    print(f"{'=' * 80}")

    print(f"\n【各品种表现】")
    print("-" * 80)
    print(f"{'品种':6} {'名称':10} {'总收益%':>10} {'年化%':>8} {'回撤%':>8} {'夏普':>6} {'胜率%':>6} {'交易':>5} {'盈亏':>12}")
    print("-" * 80)

    for _, row in df_sorted.iterrows():
        print(f"{row['symbol']:6} {row['name']:10} {row['total_return']:>10.1f} {row['annual_return']:>8.1f} "
              f"{row['max_drawdown']:>8.1f} {row['sharpe']:>6.2f} {row['win_rate']:>6.1f} "
              f"{row['total_trades']:>5} {row['total_pnl']:>12,.0f}")

    # 组合统计
    print(f"\n{'=' * 80}")
    print("组合整体统计")
    print(f"{'=' * 80}")

    profitable = len(df[df['total_return'] > 0])
    total_pnl = df['total_pnl'].sum()
    total_trades = df['total_trades'].sum()

    # 假设等权配置，计算组合收益
    avg_return = df['total_return'].mean()
    avg_annual = df['annual_return'].mean()

    print(f"盈利品种: {profitable}/{len(df)} ({profitable/len(df)*100:.1f}%)")
    print(f"总交易次数: {total_trades}")
    print(f"总盈亏: {total_pnl:,.0f}")
    print(f"\n等权组合指标:")
    print(f"  平均收益率: {avg_return:.1f}%")
    print(f"  平均年化: {avg_annual:.1f}%")
    print(f"  平均回撤: {df['max_drawdown'].mean():.1f}%")
    print(f"  平均夏普: {df['sharpe'].mean():.2f}")
    print(f"  平均胜率: {df['win_rate'].mean():.1f}%")

    # 分类统计
    print(f"\n【分类统计】")
    for category, symbols in SELECTED_SYMBOLS.items():
        cat_df = df[df['symbol'].isin(symbols)]
        if len(cat_df) > 0:
            cat_return = cat_df['total_return'].mean()
            cat_pnl = cat_df['total_pnl'].sum()
            print(f"  {category}: 平均收益 {cat_return:.1f}%, 总盈亏 {cat_pnl:,.0f}")

    output_file = "backtest_results_longonly_selected.csv"
    df_sorted.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    results = run_portfolio_backtest()
    print_summary(results)
