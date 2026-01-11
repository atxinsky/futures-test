# coding=utf-8
"""
详细调试回测过程
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

def main():
    print("=" * 60)
    print("详细调试回测过程")
    print("=" * 60)

    # 1. 加载数据
    from utils.data_loader import load_futures_data
    df = load_futures_data('IF', '2022-01-01', '2025-12-31', '1d', auto_download=False)
    print(f"数据行数: {len(df)}")

    # 2. 加载策略
    from strategies import get_strategy
    strategy_class = get_strategy('brother2v6')
    strategy = strategy_class()

    # 确保有time列
    if 'time' not in df.columns:
        df['time'] = df.index

    # 3. 计算指标
    df = strategy.calculate_indicators(df)

    # 4. 手动逐K线检查信号
    print("\n" + "=" * 60)
    print("逐K线检查信号")
    print("=" * 60)

    signals = []
    strategy.reset()

    for idx in range(strategy.warmup_num, len(df)):
        signal = strategy.on_bar(idx, df, 1000000)
        if signal:
            date = df.iloc[idx]['time']
            signals.append({
                'idx': idx,
                'date': date,
                'action': signal.action,
                'price': signal.price,
                'tag': signal.tag,
                'position_before': strategy.position
            })
            print(f"[{idx}] {date}: {signal.action} @ {signal.price:.2f} [{signal.tag}] pos={strategy.position}")

    print(f"\n总信号数: {len(signals)}")

    # 5. 分析信号
    if signals:
        buy_signals = [s for s in signals if s['action'] == 'buy']
        close_signals = [s for s in signals if s['action'] == 'close']
        print(f"买入信号: {len(buy_signals)}")
        print(f"平仓信号: {len(close_signals)}")

    # 6. 检查问题：策略状态管理
    print("\n" + "=" * 60)
    print("检查策略状态管理问题")
    print("=" * 60)

    # 重新运行，打印每次信号后的状态
    strategy2 = strategy_class()
    strategy2.reset()

    signal_count = 0
    for idx in range(strategy2.warmup_num, len(df)):
        pos_before = strategy2.position
        signal = strategy2.on_bar(idx, df, 1000000)

        if signal:
            signal_count += 1
            date = df.iloc[idx]['time']
            print(f"\n信号 #{signal_count}:")
            print(f"  日期: {date}")
            print(f"  动作: {signal.action}")
            print(f"  标签: {signal.tag}")
            print(f"  策略position变化: {pos_before} -> {strategy2.position}")
            print(f"  策略entry_price: {strategy2.entry_price}")

    # 7. 检查on_bar中的position处理
    print("\n" + "=" * 60)
    print("检查on_bar逻辑")
    print("=" * 60)

    # 读取策略代码关键部分
    print("问题可能在于：策略内部在on_bar中同时修改了position")
    print("当发出buy信号时，策略已经把position设为1")
    print("然后回测引擎再次设置position=1")
    print("但平仓时，策略先把position设为0，回测引擎检查发现已经是0")


if __name__ == '__main__':
    main()
