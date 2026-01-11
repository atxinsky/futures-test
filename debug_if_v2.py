# coding=utf-8
"""
调试IF回测问题 - 详细版
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    print("=" * 60)
    print("调试IF回测问题 - 详细版")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1] 加载数据...")
    from utils.data_loader import load_futures_data

    df = load_futures_data('IF', '2022-01-01', '2025-12-31', '1d', auto_download=False)
    print(f"  数据行数: {len(df)}")
    print(f"  时间范围: {df.index.min()} ~ {df.index.max()}")

    # 2. 加载策略并计算指标
    print("\n[2] 计算策略指标...")
    from strategies import get_strategy

    strategy_class = get_strategy('brother2v6')
    strategy = strategy_class()
    p = strategy.params

    print(f"  策略参数:")
    for k, v in p.items():
        print(f"    {k}: {v}")

    df = strategy.calculate_indicators(df)

    # 3. 检查每个条件
    print("\n[3] 检查入场条件（最近500条数据）...")

    # 只取最近的数据
    check_df = df.tail(500).copy()

    # 条件1: 趋势向上 (ema_short > ema_long)
    check_df['cond_trend'] = check_df['ema_short'] > check_df['ema_long']
    cond1_count = check_df['cond_trend'].sum()
    print(f"\n  条件1 - EMA趋势向上: {cond1_count} 天 ({cond1_count/len(check_df)*100:.1f}%)")

    # 条件2: ADX趋势强度 (adx > adx_thres)
    check_df['cond_adx'] = check_df['adx'] > p['adx_thres']
    cond2_count = check_df['cond_adx'].sum()
    print(f"  条件2 - ADX > {p['adx_thres']}: {cond2_count} 天 ({cond2_count/len(check_df)*100:.1f}%)")

    # 条件3: CHOP趋势市场 (chop < chop_thres)
    check_df['cond_chop'] = check_df['chop'] < p['chop_thres']
    cond3_count = check_df['cond_chop'].sum()
    print(f"  条件3 - CHOP < {p['chop_thres']}: {cond3_count} 天 ({cond3_count/len(check_df)*100:.1f}%)")

    # 条件4: 突破N日高点 (close > 前一日的high_line)
    check_df['high_line_prev'] = check_df['high_line'].shift(1)
    check_df['cond_breakout'] = check_df['close'] > check_df['high_line_prev']
    cond4_count = check_df['cond_breakout'].sum()
    print(f"  条件4 - 突破{p['break_len']}日高点: {cond4_count} 天 ({cond4_count/len(check_df)*100:.1f}%)")

    # 条件5: 成交量确认 (volume > vol_ma * vol_multi)
    check_df['cond_vol'] = check_df['volume'] > check_df['vol_ma'] * p['vol_multi']
    cond5_count = check_df['cond_vol'].sum()
    print(f"  条件5 - 放量 > {p['vol_multi']}倍: {cond5_count} 天 ({cond5_count/len(check_df)*100:.1f}%)")

    # 组合条件
    check_df['cond_1_2'] = check_df['cond_trend'] & check_df['cond_adx']
    check_df['cond_1_2_3'] = check_df['cond_1_2'] & check_df['cond_chop']
    check_df['cond_1_2_3_4'] = check_df['cond_1_2_3'] & check_df['cond_breakout']
    check_df['cond_all'] = check_df['cond_1_2_3_4'] & check_df['cond_vol']

    print(f"\n  条件1+2（趋势+ADX）: {check_df['cond_1_2'].sum()} 天")
    print(f"  条件1+2+3（+CHOP）: {check_df['cond_1_2_3'].sum()} 天")
    print(f"  条件1+2+3+4（+突破）: {check_df['cond_1_2_3_4'].sum()} 天")
    print(f"  全部5个条件: {check_df['cond_all'].sum()} 天")

    # 4. 找出满足前4个条件但没成交量的日子
    cond_without_vol = check_df['cond_1_2_3_4'] & ~check_df['cond_vol']
    print(f"\n  满足前4个条件但成交量不足: {cond_without_vol.sum()} 天")

    # 如果有满足前4个条件的，展示详情
    if check_df['cond_1_2_3_4'].sum() > 0:
        print("\n  满足前4个条件的日期（成交量检查）:")
        for idx in check_df[check_df['cond_1_2_3_4']].index[:10]:
            row = check_df.loc[idx]
            vol_ratio = row['volume'] / row['vol_ma'] if row['vol_ma'] > 0 else 0
            vol_ok = "✓" if row['cond_vol'] else "✗"
            print(f"    {idx.strftime('%Y-%m-%d')}: vol={row['volume']:.0f}, vol_ma={row['vol_ma']:.0f}, ratio={vol_ratio:.2f}x {vol_ok}")

    # 5. 展示指标统计
    print("\n[4] 指标统计（最近500天）:")
    print(f"  ADX 范围: {check_df['adx'].min():.1f} ~ {check_df['adx'].max():.1f}, 均值: {check_df['adx'].mean():.1f}")
    print(f"  CHOP 范围: {check_df['chop'].min():.1f} ~ {check_df['chop'].max():.1f}, 均值: {check_df['chop'].mean():.1f}")
    print(f"  成交量放量比例: {(check_df['volume']/check_df['vol_ma']).mean():.2f}x")

    # 6. 建议调参
    print("\n[5] 参数优化建议:")

    # 找到能产生信号的ADX阈值
    for adx_th in [15, 18, 20, 22, 25]:
        cond = check_df['cond_trend'] & (check_df['adx'] > adx_th) & check_df['cond_chop'] & check_df['cond_breakout'] & check_df['cond_vol']
        print(f"  ADX阈值={adx_th}: {cond.sum()} 个信号")

    # 找到能产生信号的CHOP阈值
    for chop_th in [45, 50, 55, 60]:
        cond = check_df['cond_trend'] & check_df['cond_adx'] & (check_df['chop'] < chop_th) & check_df['cond_breakout'] & check_df['cond_vol']
        print(f"  CHOP阈值={chop_th}: {cond.sum()} 个信号")

    # 找到能产生信号的放量倍数
    for vol_m in [1.0, 1.1, 1.2, 1.3, 1.5]:
        cond = check_df['cond_trend'] & check_df['cond_adx'] & check_df['cond_chop'] & check_df['cond_breakout'] & (check_df['volume'] > check_df['vol_ma'] * vol_m)
        print(f"  放量倍数={vol_m}: {cond.sum()} 个信号")

    # 7. 放宽条件测试
    print("\n[6] 放宽条件后的信号数:")
    # 只用趋势+突破
    basic = check_df['cond_trend'] & check_df['cond_breakout']
    print(f"  只用 趋势+突破: {basic.sum()} 个信号")

    # 趋势+突破+ADX（降低阈值）
    with_adx = check_df['cond_trend'] & (check_df['adx'] > 18) & check_df['cond_breakout']
    print(f"  趋势+突破+ADX(>18): {with_adx.sum()} 个信号")

    # 去掉成交量要求
    no_vol = check_df['cond_trend'] & check_df['cond_adx'] & check_df['cond_chop'] & check_df['cond_breakout']
    print(f"  去掉成交量要求: {no_vol.sum()} 个信号")

    # 去掉CHOP要求
    no_chop = check_df['cond_trend'] & check_df['cond_adx'] & check_df['cond_breakout'] & check_df['cond_vol']
    print(f"  去掉CHOP要求: {no_chop.sum()} 个信号")

    # 8. 检查实际回测
    print("\n[7] 手动模拟回测...")
    # 重新设置策略
    strategy = strategy_class()

    # 添加time列如果没有
    if 'time' not in df.columns:
        df['time'] = df.index

    df = strategy.calculate_indicators(df)

    signals = []
    for idx in range(strategy.warmup_num, len(df)):
        signal = strategy.on_bar(idx, df, 1000000)
        if signal:
            signals.append({
                'date': df.iloc[idx]['time'] if 'time' in df.columns else df.index[idx],
                'action': signal.action,
                'price': signal.price,
                'tag': signal.tag
            })

    print(f"  产生信号数: {len(signals)}")
    if signals:
        for s in signals[:10]:
            print(f"    {s['date']}: {s['action']} @ {s['price']:.2f} [{s['tag']}]")


if __name__ == '__main__':
    main()
