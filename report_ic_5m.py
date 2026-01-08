# coding=utf-8
"""IC中证500 5分钟策略详细报告"""

import sys
sys.path.insert(0, r'D:\期货\回测改造')

import pandas as pd
import numpy as np
from data_manager import load_from_tianqin
from engine import run_backtest_with_strategy
from strategies.scalp_trend import ScalpTrend

print('='*70)
print('IC中证500 5分钟 ScalpTrend策略 详细回测报告')
print('='*70)

# 加载数据
df = load_from_tianqin('IC', '5m', '2022-01-01', '2025-12-31')
print(f'\n数据范围: {df["time"].iloc[0]} ~ {df["time"].iloc[-1]}')
print(f'K线数量: {len(df)}')

# 运行回测
params = {
    'stop_atr': 1.0,
    'tp_atr': 1.5,
    'capital_rate': 0.4,
    'risk_rate': 0.02
}
strategy = ScalpTrend(params)
result = run_backtest_with_strategy(
    df=df,
    symbol='IC',
    strategy=strategy,
    initial_capital=1000000
)

# ==================== 核心指标 ====================
print('\n' + '='*70)
print('一、核心指标')
print('='*70)
print(f'''
初始资金:        1,000,000 元
最终资金:        {result.final_capital:,.0f} 元
总收益:          {result.total_pnl:,.0f} 元 ({result.total_return_pct:.1f}%)
年化收益:        {result.annual_return_pct:.1f}%
最大回撤:        {result.max_drawdown_pct:.1f}% ({result.max_drawdown_val:,.0f} 元)

交易次数:        {len(result.trades)} 笔
胜率:            {result.win_rate:.1f}%
盈亏比:          {result.profit_factor:.2f}
平均盈利:        {result.avg_win:,.0f} 元
平均亏损:        {result.avg_loss:,.0f} 元
最大单笔盈利:    {result.max_win:,.0f} 元
最大单笔亏损:    {result.max_loss:,.0f} 元

夏普比率:        {result.sharpe_ratio:.2f}
索提诺比率:      {result.sortino_ratio:.2f}
卡尔玛比率:      {result.calmar_ratio:.2f}
平均持仓天数:    {result.avg_holding_days:.1f} 天
总手续费:        {result.total_commission:,.0f} 元
''')

# ==================== 年度统计 ====================
print('='*70)
print('二、年度统计')
print('='*70)

trades_df = pd.DataFrame([{
    'entry_time': t.entry_time,
    'exit_time': t.exit_time,
    'direction': '多' if t.direction == 1 else '空',
    'entry_price': t.entry_price,
    'exit_price': t.exit_price,
    'volume': t.volume,
    'pnl': t.pnl,
    'pnl_pct': t.pnl_pct,
    'exit_tag': t.exit_tag,
    'holding_days': t.holding_days
} for t in result.trades])

trades_df['year'] = pd.to_datetime(trades_df['entry_time']).dt.year

yearly = trades_df.groupby('year').agg({
    'pnl': ['count', 'sum', lambda x: (x > 0).sum()],
}).round(0)
yearly.columns = ['交易数', '盈亏', '盈利笔数']
yearly['胜率'] = (yearly['盈利笔数'] / yearly['交易数'] * 100).round(1)
yearly['盈亏'] = yearly['盈亏'].apply(lambda x: f'{x:,.0f}')

print(yearly.to_string())

# ==================== 月度统计 ====================
print('\n' + '='*70)
print('三、月度盈亏分布')
print('='*70)

trades_df['month'] = pd.to_datetime(trades_df['entry_time']).dt.to_period('M')
monthly = trades_df.groupby('month').agg({
    'pnl': ['count', 'sum']
})
monthly.columns = ['交易数', '盈亏']

# 按年分组显示
for year in sorted(trades_df['year'].unique()):
    print(f'\n{year}年:')
    year_data = monthly[monthly.index.year == year]
    for idx, row in year_data.iterrows():
        bar = '█' * int(abs(row['盈亏']) / 5000) if row['盈亏'] != 0 else ''
        sign = '+' if row['盈亏'] > 0 else ''
        print(f"  {idx.month:2d}月: {row['交易数']:3.0f}笔 {sign}{row['盈亏']:>10,.0f} {bar}")

# ==================== 出场原因统计 ====================
print('\n' + '='*70)
print('四、出场原因统计')
print('='*70)

exit_stats = trades_df.groupby('exit_tag').agg({
    'pnl': ['count', 'sum', 'mean']
}).round(0)
exit_stats.columns = ['次数', '总盈亏', '平均盈亏']
exit_stats['占比'] = (exit_stats['次数'] / len(trades_df) * 100).round(1)
exit_stats = exit_stats.sort_values('次数', ascending=False)

print(exit_stats.to_string())

# ==================== 多空统计 ====================
print('\n' + '='*70)
print('五、多空方向统计')
print('='*70)

direction_stats = trades_df.groupby('direction').agg({
    'pnl': ['count', 'sum', lambda x: (x > 0).sum(), 'mean']
})
direction_stats.columns = ['交易数', '总盈亏', '盈利笔数', '平均盈亏']
direction_stats['胜率'] = (direction_stats['盈利笔数'] / direction_stats['交易数'] * 100).round(1)

print(direction_stats.to_string())

# ==================== 最大连续盈亏 ====================
print('\n' + '='*70)
print('六、连续盈亏统计')
print('='*70)

# 计算连续盈亏
trades_df['win'] = trades_df['pnl'] > 0
trades_df['streak_id'] = (trades_df['win'] != trades_df['win'].shift()).cumsum()
streaks = trades_df.groupby(['streak_id', 'win']).size().reset_index(name='count')

win_streaks = streaks[streaks['win'] == True]['count']
loss_streaks = streaks[streaks['win'] == False]['count']

print(f'最大连续盈利: {win_streaks.max()} 笔')
print(f'最大连续亏损: {loss_streaks.max()} 笔')
print(f'平均连续盈利: {win_streaks.mean():.1f} 笔')
print(f'平均连续亏损: {loss_streaks.mean():.1f} 笔')

# ==================== 持仓时间分布 ====================
print('\n' + '='*70)
print('七、持仓时间分布')
print('='*70)

# 计算持仓K线数
trades_df['entry_dt'] = pd.to_datetime(trades_df['entry_time'])
trades_df['exit_dt'] = pd.to_datetime(trades_df['exit_time'])
trades_df['hold_minutes'] = (trades_df['exit_dt'] - trades_df['entry_dt']).dt.total_seconds() / 60

hold_bins = [0, 30, 60, 120, 240, 480, float('inf')]
hold_labels = ['<30分钟', '30-60分钟', '1-2小时', '2-4小时', '4-8小时', '>8小时']
trades_df['hold_range'] = pd.cut(trades_df['hold_minutes'], bins=hold_bins, labels=hold_labels)

hold_dist = trades_df.groupby('hold_range').agg({
    'pnl': ['count', 'sum', 'mean']
})
hold_dist.columns = ['次数', '总盈亏', '平均盈亏']
print(hold_dist.to_string())

# ==================== 最近20笔交易 ====================
print('\n' + '='*70)
print('八、最近20笔交易明细')
print('='*70)

recent = trades_df.tail(20)[['entry_time', 'direction', 'entry_price', 'exit_price', 'pnl', 'exit_tag']]
recent['entry_time'] = pd.to_datetime(recent['entry_time']).dt.strftime('%m-%d %H:%M')
recent['pnl'] = recent['pnl'].apply(lambda x: f'{x:+,.0f}')
print(recent.to_string(index=False))

# ==================== 策略参数 ====================
print('\n' + '='*70)
print('九、策略参数')
print('='*70)
print(f'''
策略名称:        ScalpTrend (剥头皮趋势)
交易周期:        5分钟
品种:            IC (中证500股指期货)

动量参数:
  - mom_len:     5 (动量计算周期)
  - mom_thres:   0.3% (动量阈值)

均线过滤:
  - ma_len:      20 (趋势过滤均线)

成交量:
  - vol_len:     10 (量能周期)
  - vol_mult:    1.5 (放量倍数)

止损止盈:
  - stop_atr:    1.0 (紧止损)
  - tp_atr:      1.5 (快止盈)
  - trail_atr:   1.0 (追踪止损)
  - trail_start: 1.0% (追踪起点)

仓位管理:
  - capital_rate: 0.4 (40%资金使用)
  - risk_rate:    0.02 (2%单笔风险)
''')

print('='*70)
print('报告生成完成')
print('='*70)
