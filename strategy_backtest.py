# coding=utf-8
"""
策略对比回测：macdema (V3) vs emanew (V5)
- V3: 一次性止盈
- V5: 分批止盈（50% + 50%）
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

# ==================== 合约参数 ====================
IF_MULTIPLIER = 300      # 合约乘数
IF_MARGIN_RATE = 0.12    # 保证金比例
IF_COMMISSION = 0.000023 # 手续费率
INITIAL_CAPITAL = 1000000  # 初始资金100万

# ==================== 策略参数 ====================
# 技术指标参数
EMA_FAST = 9
EMA_SLOW = 21
MA_LEN = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SMOOTH = 9

# V3 止损参数
V3_STOP_LOSS = 0.08         # 固定止损 8%
V3_BREAK_EVEN = 0.10        # 保本触发 10%
V3_TRAIL_TRIGGER = 0.18     # 追踪止损触发 18%
V3_TRAIL_DRAWDOWN = 0.10    # 追踪止损回撤 10%
V3_SIGNAL_LOW_DAYS = 3      # 信号K线止损天数

# V5 止损参数
V5_STOP_LOSS = 0.08
V5_BREAK_EVEN = 0.10
V5_SIGNAL_LOW_DAYS = 3
V5_PARTIAL_TRIGGER = 0.15   # 第一次止盈触发 15%
V5_PARTIAL_DRAWDOWN = 0.06  # 第一次止盈回撤 6%
V5_PARTIAL_RATE = 0.50      # 第一次平仓比例 50%
V5_FULL_DRAWDOWN = 0.12     # 第二次止盈回撤 12%


def calculate_ema(series, period):
    """计算EMA"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series, period):
    """计算SMA"""
    return series.rolling(window=period).mean()


def calculate_macd(close, fast=12, slow=26, smooth=9):
    """计算MACD"""
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    macd = ema_fast - ema_slow
    signal = calculate_ema(macd, smooth)
    histogram = macd - signal
    return macd, signal, histogram


def detect_cross(fast, slow):
    """检测金叉死叉: 1=金叉, -1=死叉, 0=无"""
    cross = np.zeros(len(fast))
    for i in range(1, len(fast)):
        if fast.iloc[i] > slow.iloc[i] and fast.iloc[i-1] <= slow.iloc[i-1]:
            cross[i] = 1  # 金叉
        elif fast.iloc[i] < slow.iloc[i] and fast.iloc[i-1] >= slow.iloc[i-1]:
            cross[i] = -1  # 死叉
    return cross


def backtest_v3(df, symbol_name):
    """
    回测 macdema V3 策略 - 一次性止盈
    """
    df = df.copy()

    # 计算指标
    df['ema_fast'] = calculate_ema(df['close'], EMA_FAST)
    df['ema_slow'] = calculate_ema(df['close'], EMA_SLOW)
    df['ma'] = calculate_sma(df['close'], MA_LEN)
    df['macd'], df['signal'], df['hist'] = calculate_macd(df['close'], MACD_FAST, MACD_SLOW, MACD_SMOOTH)
    df['ema_cross'] = detect_cross(df['ema_fast'], df['ema_slow'])

    # 交易记录
    trades = []
    position = 0
    entry_price = 0
    entry_time = None
    high_since = 0
    signal_low = 0
    days_below_signal = 0
    shares = 1

    capital = INITIAL_CAPITAL
    equity_curve = []

    warmup = max(EMA_SLOW, MA_LEN, MACD_SLOW) + 10

    for i in range(warmup, len(df)):
        price = df['close'].iloc[i]
        current_time = df['time'].iloc[i]
        ema_cross = df['ema_cross'].iloc[i]
        hist_val = df['hist'].iloc[i]
        hist_prev = df['hist'].iloc[i-1] if i > 0 else 0
        ma_val = df['ma'].iloc[i]
        low_val = df['low'].iloc[i]

        exit_tag = None

        # 持仓期间的止损检查
        if position > 0:
            if price > high_since:
                high_since = price

            profit_rate = (price - entry_price) / entry_price
            drawdown_from_high = (high_since - price) / high_since if high_since > 0 else 0
            max_profit_rate = (high_since - entry_price) / entry_price

            # 信号K线止损
            if price < signal_low:
                days_below_signal += 1
            else:
                days_below_signal = 0

            if days_below_signal >= V3_SIGNAL_LOW_DAYS:
                exit_tag = "signal_low_break"
            elif profit_rate <= -V3_STOP_LOSS:
                exit_tag = "stop_loss"
            elif max_profit_rate >= V3_TRAIL_TRIGGER and drawdown_from_high >= V3_TRAIL_DRAWDOWN:
                exit_tag = "trail_stop"
            elif max_profit_rate >= V3_BREAK_EVEN and profit_rate <= 0:
                exit_tag = "break_even"
            elif ema_cross == -1 and price < ma_val:
                exit_tag = "death_cross"

            if exit_tag:
                # 平仓
                pnl = (price - entry_price) * shares * IF_MULTIPLIER
                commission = price * shares * IF_MULTIPLIER * IF_COMMISSION * 2
                net_pnl = pnl - commission
                capital += net_pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'shares': shares,
                    'pnl': net_pnl,
                    'pnl_pct': (price - entry_price) / entry_price * 100,
                    'exit_tag': exit_tag,
                    'max_profit_pct': max_profit_rate * 100
                })
                position = 0

        # 入场信号
        buy_signal = (ema_cross == 1 and
                     hist_val > 0 and
                     (hist_val > hist_prev or hist_prev < 0) and
                     price > ma_val)

        if buy_signal and position == 0:
            position = 1
            entry_price = price
            entry_time = current_time
            high_since = price
            signal_low = low_val
            days_below_signal = 0

        # 记录权益
        unrealized = 0
        if position > 0:
            unrealized = (price - entry_price) * shares * IF_MULTIPLIER
        equity_curve.append({'time': current_time, 'equity': capital + unrealized})

    return trades, equity_curve, capital


def backtest_v5(df, symbol_name):
    """
    回测 emanew V5 策略 - 分批止盈
    """
    df = df.copy()

    # 计算指标
    df['ema_fast'] = calculate_ema(df['close'], EMA_FAST)
    df['ema_slow'] = calculate_ema(df['close'], EMA_SLOW)
    df['ma'] = calculate_sma(df['close'], MA_LEN)
    df['macd'], df['signal'], df['hist'] = calculate_macd(df['close'], MACD_FAST, MACD_SLOW, MACD_SMOOTH)
    df['ema_cross'] = detect_cross(df['ema_fast'], df['ema_slow'])

    # 交易记录
    trades = []
    position = 0  # 当前持仓比例 (0, 0.5, 1)
    entry_price = 0
    entry_time = None
    high_since = 0
    signal_low = 0
    days_below_signal = 0
    has_partial_exit = False
    high_after_partial = 0
    initial_shares = 1
    current_shares = 0

    capital = INITIAL_CAPITAL
    equity_curve = []

    warmup = max(EMA_SLOW, MA_LEN, MACD_SLOW) + 10

    for i in range(warmup, len(df)):
        price = df['close'].iloc[i]
        current_time = df['time'].iloc[i]
        ema_cross = df['ema_cross'].iloc[i]
        hist_val = df['hist'].iloc[i]
        hist_prev = df['hist'].iloc[i-1] if i > 0 else 0
        ma_val = df['ma'].iloc[i]
        low_val = df['low'].iloc[i]

        exit_tag = None
        exit_shares = 0

        # 持仓期间的止损检查
        if position > 0:
            if price > high_since:
                high_since = price
            if has_partial_exit and price > high_after_partial:
                high_after_partial = price

            profit_rate = (price - entry_price) / entry_price
            drawdown_from_high = (high_since - price) / high_since if high_since > 0 else 0
            max_profit_rate = (high_since - entry_price) / entry_price

            # 信号K线止损
            if price < signal_low:
                days_below_signal += 1
            else:
                days_below_signal = 0

            # 检查止损条件
            if days_below_signal >= V5_SIGNAL_LOW_DAYS:
                exit_tag = "signal_low_break"
                exit_shares = current_shares  # 全部平仓
            elif profit_rate <= -V5_STOP_LOSS:
                exit_tag = "stop_loss"
                exit_shares = current_shares
            elif not has_partial_exit:
                # 检查第一次止盈
                if max_profit_rate >= V5_PARTIAL_TRIGGER and drawdown_from_high >= V5_PARTIAL_DRAWDOWN:
                    exit_tag = "partial_stop"
                    exit_shares = current_shares * V5_PARTIAL_RATE  # 平仓50%
                    has_partial_exit = True
                    high_after_partial = price
            else:
                # 已经第一次止盈，检查第二次
                drawdown_after_partial = (high_after_partial - price) / high_after_partial if high_after_partial > 0 else 0
                if drawdown_after_partial >= V5_FULL_DRAWDOWN:
                    exit_tag = "trail_stop_full"
                    exit_shares = current_shares

            # 保本止损
            if exit_tag is None and max_profit_rate >= V5_BREAK_EVEN and profit_rate <= 0:
                exit_tag = "break_even"
                exit_shares = current_shares

            # 死叉出场
            if exit_tag is None and ema_cross == -1 and price < ma_val:
                exit_tag = "death_cross"
                exit_shares = current_shares

            if exit_tag and exit_shares > 0:
                # 平仓
                pnl = (price - entry_price) * exit_shares * IF_MULTIPLIER
                commission = price * exit_shares * IF_MULTIPLIER * IF_COMMISSION * 2
                net_pnl = pnl - commission
                capital += net_pnl

                trades.append({
                    'entry_time': entry_time,
                    'exit_time': current_time,
                    'entry_price': entry_price,
                    'exit_price': price,
                    'shares': exit_shares,
                    'pnl': net_pnl,
                    'pnl_pct': (price - entry_price) / entry_price * 100,
                    'exit_tag': exit_tag,
                    'max_profit_pct': max_profit_rate * 100
                })

                current_shares -= exit_shares
                if current_shares <= 0.01:  # 基本清仓
                    position = 0
                    current_shares = 0
                    has_partial_exit = False

        # 入场信号
        buy_signal = (ema_cross == 1 and
                     hist_val > 0 and
                     (hist_val > hist_prev or hist_prev < 0) and
                     price > ma_val)

        if buy_signal and position == 0:
            position = 1
            entry_price = price
            entry_time = current_time
            high_since = price
            signal_low = low_val
            days_below_signal = 0
            has_partial_exit = False
            high_after_partial = 0
            current_shares = initial_shares

        # 记录权益
        unrealized = 0
        if current_shares > 0:
            unrealized = (price - entry_price) * current_shares * IF_MULTIPLIER
        equity_curve.append({'time': current_time, 'equity': capital + unrealized})

    return trades, equity_curve, capital


def calculate_metrics(trades, equity_curve, initial_capital):
    """计算回测指标"""
    if not trades:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
        }

    df_trades = pd.DataFrame(trades)
    df_equity = pd.DataFrame(equity_curve)

    # 基础统计
    total_trades = len(df_trades)
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] <= 0]
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0

    total_pnl = df_trades['pnl'].sum()
    total_return = (total_pnl / initial_capital) * 100

    # 最大回撤
    df_equity['peak'] = df_equity['equity'].cummax()
    df_equity['drawdown'] = (df_equity['peak'] - df_equity['equity']) / df_equity['peak']
    max_drawdown = df_equity['drawdown'].max() * 100

    # 盈亏比
    avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 1
    profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

    # 夏普比率 (简化计算)
    if len(df_equity) > 1:
        returns = df_equity['equity'].pct_change().dropna()
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
    else:
        sharpe_ratio = 0

    # 出场原因统计
    exit_tags = df_trades['exit_tag'].value_counts().to_dict()

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'exit_tags': exit_tags
    }


def generate_report(v3_metrics, v5_metrics, v3_trades, v5_trades, symbol):
    """生成对比报告"""
    report = f"""
# 策略回测对比报告
## {symbol} 股指期货

**回测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 一、策略概述

| 策略 | 核心特点 |
|------|----------|
| **macdema (V3)** | 一次性止盈，盈利18%后回撤10%全部出场 |
| **emanew (V5)** | 分批止盈，盈利15%后回撤6%先出50%，再回撤12%出剩余 |

---

## 二、回测结果对比

| 指标 | macdema (V3) | emanew (V5) | 差异 |
|------|-------------|-------------|------|
| **总交易次数** | {v3_metrics['total_trades']} | {v5_metrics['total_trades']} | {v5_metrics['total_trades'] - v3_metrics['total_trades']:+d} |
| **胜率** | {v3_metrics['win_rate']:.1f}% | {v5_metrics['win_rate']:.1f}% | {v5_metrics['win_rate'] - v3_metrics['win_rate']:+.1f}% |
| **总收益** | {v3_metrics['total_pnl']:,.0f} 元 | {v5_metrics['total_pnl']:,.0f} 元 | {v5_metrics['total_pnl'] - v3_metrics['total_pnl']:+,.0f} |
| **收益率** | {v3_metrics['total_return']:.2f}% | {v5_metrics['total_return']:.2f}% | {v5_metrics['total_return'] - v3_metrics['total_return']:+.2f}% |
| **最大回撤** | {v3_metrics['max_drawdown']:.2f}% | {v5_metrics['max_drawdown']:.2f}% | {v5_metrics['max_drawdown'] - v3_metrics['max_drawdown']:+.2f}% |
| **夏普比率** | {v3_metrics['sharpe_ratio']:.2f} | {v5_metrics['sharpe_ratio']:.2f} | {v5_metrics['sharpe_ratio'] - v3_metrics['sharpe_ratio']:+.2f} |
| **盈亏比** | {v3_metrics['profit_factor']:.2f} | {v5_metrics['profit_factor']:.2f} | {v5_metrics['profit_factor'] - v3_metrics['profit_factor']:+.2f} |
| **平均盈利** | {v3_metrics['avg_win']:,.0f} 元 | {v5_metrics['avg_win']:,.0f} 元 | - |
| **平均亏损** | {v3_metrics['avg_loss']:,.0f} 元 | {v5_metrics['avg_loss']:,.0f} 元 | - |

---

## 三、出场原因统计

### macdema (V3)
| 出场原因 | 次数 |
|----------|------|
"""
    for tag, count in v3_metrics.get('exit_tags', {}).items():
        report += f"| {tag} | {count} |\n"

    report += f"""
### emanew (V5)
| 出场原因 | 次数 |
|----------|------|
"""
    for tag, count in v5_metrics.get('exit_tags', {}).items():
        report += f"| {tag} | {count} |\n"

    report += f"""
---

## 四、结论

"""
    # 自动生成结论
    if v3_metrics['total_return'] > v5_metrics['total_return']:
        report += f"- **收益率**: V3 ({v3_metrics['total_return']:.2f}%) 优于 V5 ({v5_metrics['total_return']:.2f}%)，一次性止盈在趋势行情中收益更高\n"
    else:
        report += f"- **收益率**: V5 ({v5_metrics['total_return']:.2f}%) 优于 V3 ({v3_metrics['total_return']:.2f}%)，分批止盈锁定了更多利润\n"

    if v3_metrics['max_drawdown'] < v5_metrics['max_drawdown']:
        report += f"- **风险控制**: V3 最大回撤 ({v3_metrics['max_drawdown']:.2f}%) 更小\n"
    else:
        report += f"- **风险控制**: V5 最大回撤 ({v5_metrics['max_drawdown']:.2f}%) 更小，分批止盈降低了风险\n"

    if v3_metrics['win_rate'] > v5_metrics['win_rate']:
        report += f"- **胜率**: V3 ({v3_metrics['win_rate']:.1f}%) 胜率更高\n"
    else:
        report += f"- **胜率**: V5 ({v5_metrics['win_rate']:.1f}%) 胜率更高，分批出场提高了盈利概率\n"

    report += f"""
---

## 五、建议

1. **震荡市场**: 优先使用 emanew (V5)，分批止盈更能锁定利润
2. **趋势市场**: 优先使用 macdema (V3)，能吃到更多趋势利润
3. **风险偏好低**: 选择 V5，心理压力更小
4. **追求最大收益**: 选择 V3，但需要承受更大的盈利回吐

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    return report


def main():
    """主函数"""
    # 读取数据
    data_path = r"D:\期货\股指期货\CFFEX_DLY_IF1!, 1D_6ce25.csv"
    print(f"读取数据: {data_path}")

    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]

    # 确保时间格式正确
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)

    print(f"数据范围: {df['time'].min()} ~ {df['time'].max()}")
    print(f"数据条数: {len(df)}")

    # 运行V3回测
    print("\n运行 macdema (V3) 回测...")
    v3_trades, v3_equity, v3_capital = backtest_v3(df, "IF")
    v3_metrics = calculate_metrics(v3_trades, v3_equity, INITIAL_CAPITAL)
    print(f"  交易次数: {v3_metrics['total_trades']}")
    print(f"  总收益: {v3_metrics['total_pnl']:,.0f} 元")
    print(f"  收益率: {v3_metrics['total_return']:.2f}%")

    # 运行V5回测
    print("\n运行 emanew (V5) 回测...")
    v5_trades, v5_equity, v5_capital = backtest_v5(df, "IF")
    v5_metrics = calculate_metrics(v5_trades, v5_equity, INITIAL_CAPITAL)
    print(f"  交易次数: {v5_metrics['total_trades']}")
    print(f"  总收益: {v5_metrics['total_pnl']:,.0f} 元")
    print(f"  收益率: {v5_metrics['total_return']:.2f}%")

    # 生成报告
    print("\n生成对比报告...")
    report = generate_report(v3_metrics, v5_metrics, v3_trades, v5_trades, "IF (沪深300股指期货)")

    # 保存报告到桌面
    desktop_path = r"C:\Users\atxin\Desktop"
    report_path = os.path.join(desktop_path, "策略回测对比报告.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"报告已保存: {report_path}")

    # 保存交易明细
    if v3_trades:
        v3_df = pd.DataFrame(v3_trades)
        v3_df.to_csv(os.path.join(desktop_path, "V3_macdema_trades.csv"), index=False, encoding='utf-8-sig')
    if v5_trades:
        v5_df = pd.DataFrame(v5_trades)
        v5_df.to_csv(os.path.join(desktop_path, "V5_emanew_trades.csv"), index=False, encoding='utf-8-sig')

    print("\n回测完成!")
    return v3_metrics, v5_metrics


if __name__ == "__main__":
    main()
