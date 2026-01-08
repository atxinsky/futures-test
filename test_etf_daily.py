# coding=utf-8
"""10万资金 500ETF 日线策略模拟"""

import sys
sys.path.insert(0, r'D:\期货\回测改造')

import pandas as pd
import numpy as np
from data_manager import load_from_database

# 用IC日线数据模拟500ETF
df = load_from_database('IC', '2018-01-01', '2025-12-31')

print('='*60)
print('10万资金 500ETF 日线趋势策略')
print('='*60)

# 计算指标
df['ma10'] = df['close'].rolling(10).mean()
df['ma20'] = df['close'].rolling(20).mean()
df['ma60'] = df['close'].rolling(60).mean()

# ATR
tr = pd.concat([
    df['high'] - df['low'],
    (df['high'] - df['close'].shift()).abs(),
    (df['low'] - df['close'].shift()).abs()
], axis=1).max(axis=1)
df['atr'] = tr.rolling(20).mean()
df['atr_pct'] = df['atr'] / df['close'] * 100

# 趋势判断
df['trend'] = np.where(df['ma10'] > df['ma20'], 1, -1)
df['big_trend'] = np.where(df['close'] > df['ma60'], 1, -1)

# 模拟交易
initial_capital = 100000
capital = initial_capital
position = 0
entry_price = 0
trades = []
equity = [initial_capital]

for i in range(60, len(df)):
    row = df.iloc[i]
    prev = df.iloc[i-1]

    close = row['close']
    ma10 = row['ma10']
    ma20 = row['ma20']
    ma60 = row['ma60']
    atr_pct = row['atr_pct']
    trend = row['trend']
    big_trend = row['big_trend']

    # ETF价格 (假设 = IC/1000)
    etf_price = close / 1000

    if pd.isna(atr_pct):
        equity.append(equity[-1])
        continue

    # 持仓管理
    if position > 0:
        current_value = position * etf_price
        pnl_pct = (etf_price - entry_price) / entry_price * 100

        # 出场条件
        should_exit = False
        exit_tag = ''

        # 1. 趋势反转 (MA10下穿MA20)
        if prev['trend'] == 1 and trend == -1:
            should_exit = True
            exit_tag = 'trend_reverse'

        # 2. 跌破MA60 (大趋势反转)
        if close < ma60 and prev['close'] >= prev['ma60']:
            should_exit = True
            exit_tag = 'break_ma60'

        # 3. 止损 -5%
        if pnl_pct < -5:
            should_exit = True
            exit_tag = 'stop_loss'

        # 4. 止盈 +15%
        if pnl_pct > 15:
            should_exit = True
            exit_tag = 'take_profit'

        if should_exit:
            sell_value = position * etf_price
            commission = sell_value * 0.0003
            capital = sell_value - commission
            trades.append({
                'pnl_pct': pnl_pct,
                'pnl': capital - (position * entry_price),
                'tag': exit_tag
            })
            position = 0

    # 开仓条件 (只做多)
    if position == 0 and capital > 1000:
        # MA10上穿MA20 + 价格在MA60上方
        if prev['trend'] == -1 and trend == 1 and big_trend == 1:
            buy_value = capital * 0.95
            commission = buy_value * 0.0003
            position = int((buy_value - commission) / etf_price)
            entry_price = etf_price

    # 记录权益
    if position > 0:
        equity.append(position * etf_price)
    else:
        equity.append(capital)

# 最终结算
if position > 0:
    final_value = position * (df.iloc[-1]['close'] / 1000)
else:
    final_value = capital

# 计算结果
equity_arr = np.array(equity)
peak = np.maximum.accumulate(equity_arr)
drawdown = (peak - equity_arr) / peak * 100
max_dd = drawdown.max()

years = len(df) / 250
total_return = (final_value / initial_capital - 1) * 100
annual_return = total_return / years

trades_df = pd.DataFrame(trades)
if len(trades_df) > 0:
    win_rate = (trades_df['pnl_pct'] > 0).sum() / len(trades_df) * 100
else:
    win_rate = 0

print(f'''
【回测结果】
回测周期: 2018-2025 ({years:.0f}年)
初始资金: {initial_capital:,} 元
最终资金: {final_value:,.0f} 元
总收益: {final_value - initial_capital:+,.0f} 元 ({total_return:+.1f}%)
年化收益: {annual_return:.1f}%
最大回撤: {max_dd:.1f}%
交易次数: {len(trades_df)} 笔
胜率: {win_rate:.1f}%
''')

if len(trades_df) > 0:
    print('【出场原因统计】')
    exit_stats = trades_df.groupby('tag').agg({
        'pnl_pct': ['count', 'sum', 'mean']
    })
    exit_stats.columns = ['次数', '累计%', '平均%']
    print(exit_stats.to_string())

# 和指数本身对比
index_return = (df.iloc[-1]['close'] / df.iloc[60]['close'] - 1) * 100
print(f'''
【对比基准】
中证500指数涨幅: {index_return:.1f}%
策略超额收益: {total_return - index_return:+.1f}%
''')
