# coding=utf-8
"""
WaveTrend 分钟级回测
使用天勤4年数据 (2022-2025)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# 添加路径
sys.path.insert(0, 'D:/期货/回测改造')

from strategies.wavetrend_final import WaveTrendFinalStrategy, SYMBOL_CONFIGS


# 天勤数据库
TQ_DB = 'D:/期货/回测改造/data/futures_tq.db'


def load_tianqin_data(symbol: str, period: str) -> pd.DataFrame:
    """加载天勤数据"""
    conn = sqlite3.connect(TQ_DB)
    query = """
        SELECT datetime as time, open, high, low, close, volume, open_interest
        FROM kline_data
        WHERE symbol = ? AND period = ?
        ORDER BY datetime
    """
    df = pd.read_sql_query(query, conn, params=[symbol, period])
    conn.close()

    if len(df) > 0:
        df['time'] = pd.to_datetime(df['time'])
    return df


def run_backtest(df: pd.DataFrame, symbol: str, strategy) -> dict:
    """运行回测"""
    if len(df) < strategy.warmup_num + 10:
        return None

    # 计算指标
    df = strategy.calculate_indicators(df)

    # 初始资金
    initial_capital = 100000
    capital = initial_capital

    trades = []
    equity_curve = []

    # 遍历K线
    for idx in range(strategy.warmup_num, len(df)):
        signal = strategy.on_bar(idx, df, capital)

        if signal:
            if signal.action == 'buy':
                trades.append({
                    'type': 'buy',
                    'time': df['time'].iloc[idx],
                    'price': signal.price,
                    'tag': signal.tag
                })
            elif signal.action == 'close' and len(trades) > 0 and trades[-1]['type'] == 'buy':
                entry = trades[-1]
                exit_price = signal.price
                pnl_pct = (exit_price - entry['price']) / entry['price'] * 100
                trades[-1]['exit_time'] = df['time'].iloc[idx]
                trades[-1]['exit_price'] = exit_price
                trades[-1]['exit_tag'] = signal.tag
                trades[-1]['pnl_pct'] = pnl_pct
                capital *= (1 + pnl_pct / 100)

        equity_curve.append(capital)

    # 统计
    completed_trades = [t for t in trades if 'pnl_pct' in t]
    if len(completed_trades) == 0:
        return None

    win_trades = [t for t in completed_trades if t['pnl_pct'] > 0]
    lose_trades = [t for t in completed_trades if t['pnl_pct'] <= 0]

    win_rate = len(win_trades) / len(completed_trades) * 100 if completed_trades else 0
    avg_win = np.mean([t['pnl_pct'] for t in win_trades]) if win_trades else 0
    avg_loss = abs(np.mean([t['pnl_pct'] for t in lose_trades])) if lose_trades else 0
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

    total_return = (capital - initial_capital) / initial_capital * 100

    # 计算最大回撤
    peak = initial_capital
    max_drawdown = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        if dd > max_drawdown:
            max_drawdown = dd

    return {
        'symbol': symbol,
        'trades': len(completed_trades),
        'win_rate': win_rate,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'stop_method': strategy.effective_stop_method
    }


def main():
    symbols = ['AU', 'AG', 'RB', 'M', 'CU', 'FG', 'I', 'NI', 'TA', 'PP', 'Y', 'MA']
    periods = ['60m', '15m']

    results = {}

    for period in periods:
        print(f"\n{'='*60}")
        print(f"回测周期: {period}")
        print('='*60)

        period_results = []

        for symbol in symbols:
            # 加载数据
            df = load_tianqin_data(symbol, period)
            if len(df) < 100:
                print(f"{symbol}: 数据不足")
                continue

            # 创建策略
            strategy = WaveTrendFinalStrategy()
            strategy.set_symbol(symbol)
            strategy.reset()

            # 运行回测
            result = run_backtest(df, symbol, strategy)

            if result:
                period_results.append(result)
                print(f"{symbol:4} | 交易:{result['trades']:3} | 胜率:{result['win_rate']:5.1f}% | "
                      f"收益:{result['total_return']:+7.1f}% | 回撤:{result['max_drawdown']:5.1f}% | "
                      f"盈亏比:{result['profit_factor']:4.2f} | 止损:{result['stop_method']}")
            else:
                print(f"{symbol}: 无交易")

        results[period] = period_results

        # 统计
        if period_results:
            profitable = [r for r in period_results if r['total_return'] > 0]
            avg_return = np.mean([r['total_return'] for r in period_results])
            print(f"\n{period} 统计:")
            print(f"  盈利品种: {len(profitable)}/{len(period_results)}")
            print(f"  平均收益: {avg_return:+.1f}%")

    return results


if __name__ == '__main__':
    results = main()
