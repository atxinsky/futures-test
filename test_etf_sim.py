# coding=utf-8
"""模拟10万资金跑500ETF"""

import sys
sys.path.insert(0, r'D:\期货\回测改造')

import pandas as pd
import numpy as np
from data_manager import load_from_tianqin

# 用IC 5分钟数据模拟500ETF走势
df = load_from_tianqin('IC', '5m', '2022-01-01', '2025-12-31')

print('='*60)
print('10万资金 500ETF模拟 (基于IC走势)')
print('='*60)

# 策略参数 (简化版scalp_trend)
initial_capital = 100000  # 10万
commission_rate = 0.0003  # ETF手续费万3

# 计算指标
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date
df['ma20'] = df['close'].rolling(20).mean()
df['mom5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
df['vol_ma'] = df['volume'].rolling(10).mean()
df['vol_ratio'] = df['volume'] / df['vol_ma']

# ATR
tr = pd.concat([
    df['high'] - df['low'],
    (df['high'] - df['close'].shift()).abs(),
    (df['low'] - df['close'].shift()).abs()
], axis=1).max(axis=1)
df['atr'] = tr.rolling(14).mean()

# 模拟交易 - ETF版本
# 简化：每天最多1次交易，T+1所以当天不能卖
capital = initial_capital
position = 0  # 持有股数
entry_price = 0
entry_date = None
trades = []

for i in range(30, len(df)):
    row = df.iloc[i]
    close = row['close']
    mom = row['mom5']
    ma = row['ma20']
    atr = row['atr']
    vol_ratio = row['vol_ratio']
    current_date = row['date']

    if pd.isna(mom) or pd.isna(ma) or pd.isna(atr):
        continue

    # ETF价格模拟 (假设ETF价格 = IC点位 / 1000，约5-7元)
    etf_price = close / 1000

    # 持仓管理
    if position > 0:
        # T+1: 只能在入场次日后卖出
        if current_date > entry_date:
            pnl_pct = (etf_price - entry_price) / entry_price * 100

            # 止盈: 1.5%
            if pnl_pct >= 1.5:
                sell_value = position * etf_price
                commission = sell_value * commission_rate
                capital = sell_value - commission
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': etf_price,
                    'pnl_pct': pnl_pct,
                    'pnl': capital - (position * entry_price),
                    'tag': 'take_profit'
                })
                position = 0
                continue

            # 止损: -1%
            if pnl_pct <= -1.0:
                sell_value = position * etf_price
                commission = sell_value * commission_rate
                capital = sell_value - commission
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': etf_price,
                    'pnl_pct': pnl_pct,
                    'pnl': capital - (position * entry_price),
                    'tag': 'stop_loss'
                })
                position = 0
                continue

    # 开仓条件 (只做多，ETF做空需要融券)
    if position == 0 and capital > 1000:
        if mom > 0.3 and close > ma and vol_ratio > 1.5:
            # 满仓买入
            buy_value = capital * 0.95  # 留5%现金
            commission = buy_value * commission_rate
            position = int((buy_value - commission) / etf_price)
            entry_price = etf_price
            entry_date = current_date

# 最终结算
if position > 0:
    final_value = position * (df.iloc[-1]['close'] / 1000)
else:
    final_value = capital

trades_df = pd.DataFrame(trades)

print(f'''
初始资金: {initial_capital:,} 元
最终资金: {final_value:,.0f} 元
总收益: {final_value - initial_capital:+,.0f} 元 ({(final_value/initial_capital-1)*100:+.1f}%)
''')

if len(trades_df) > 0:
    years = (df['time'].iloc[-1] - df['time'].iloc[0]).days / 365
    total_return = (final_value / initial_capital - 1) * 100
    annual_return = total_return / years

    win_trades = trades_df[trades_df['pnl_pct'] > 0]
    win_rate = len(win_trades) / len(trades_df) * 100

    print(f'''回测年数: {years:.1f} 年
年化收益: {annual_return:.1f}%
交易次数: {len(trades_df)} 笔
胜率: {win_rate:.1f}%
平均盈利: {trades_df[trades_df['pnl_pct']>0]['pnl_pct'].mean():.2f}%
平均亏损: {trades_df[trades_df['pnl_pct']<0]['pnl_pct'].mean():.2f}%
''')

    # 年度统计
    trades_df['year'] = pd.to_datetime(trades_df['exit_date']).dt.year
    yearly = trades_df.groupby('year').agg({
        'pnl_pct': ['count', 'sum', lambda x: (x>0).sum()]
    })
    yearly.columns = ['交易数', '累计收益%', '盈利笔数']

    print('年度统计:')
    print(yearly.to_string())
else:
    print('无交易记录')
