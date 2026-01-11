# coding=utf-8
"""
调试数据筛选问题
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

def main():
    from utils.data_loader import load_futures_data

    df = load_futures_data('IF', '2022-01-01', '2025-12-31', '1d', auto_download=False)
    print(f"原始数据: {len(df)} 行")
    print(f"索引类型: {type(df.index)}")
    print(f"索引范围: {df.index.min()} ~ {df.index.max()}")

    # 模拟优化器的数据筛选
    train_start = '2022-01-01'
    train_end = '2024-12-31'

    print(f"\n筛选条件: {train_start} ~ {train_end}")

    # 方式1：字符串直接比较
    try:
        train_df1 = df[(df.index >= train_start) & (df.index <= train_end)]
        print(f"方式1（字符串）: {len(train_df1)} 行")
    except Exception as e:
        print(f"方式1 出错: {e}")

    # 方式2：转换为datetime
    from datetime import datetime
    train_start_dt = datetime.strptime(train_start, '%Y-%m-%d')
    train_end_dt = datetime.strptime(train_end, '%Y-%m-%d')

    train_df2 = df[(df.index >= train_start_dt) & (df.index <= train_end_dt)]
    print(f"方式2（datetime）: {len(train_df2)} 行")

    # 方式3：使用pd.Timestamp
    train_df3 = df[(df.index >= pd.Timestamp(train_start)) & (df.index <= pd.Timestamp(train_end))]
    print(f"方式3（Timestamp）: {len(train_df3)} 行")

    # 检查训练集和验证集
    print("\n" + "=" * 40)
    print("分析优化器使用的数据范围")
    print("=" * 40)

    # 测试用例1: train 2022-2024, val 2025
    train_start1, train_end1 = '2022-01-01', '2024-12-31'
    val_start1, val_end1 = '2025-01-01', '2025-12-31'

    train1 = df[(df.index >= pd.Timestamp(train_start1)) & (df.index <= pd.Timestamp(train_end1))]
    val1 = df[(df.index >= pd.Timestamp(val_start1)) & (df.index <= pd.Timestamp(val_end1))]
    print(f"\n配置1: train={train_start1}~{train_end1}, val={val_start1}~{val_end1}")
    print(f"  训练集: {len(train1)} 行 ({train1.index.min()} ~ {train1.index.max()})")
    print(f"  验证集: {len(val1)} 行 ({val1.index.min()} ~ {val1.index.max()})")

    # 测试用例2: train 2022-2024.6, val 2024.7-2025
    train_start2, train_end2 = '2022-01-01', '2024-06-30'
    val_start2, val_end2 = '2024-07-01', '2025-12-31'

    train2 = df[(df.index >= pd.Timestamp(train_start2)) & (df.index <= pd.Timestamp(train_end2))]
    val2 = df[(df.index >= pd.Timestamp(val_start2)) & (df.index <= pd.Timestamp(val_end2))]
    print(f"\n配置2: train={train_start2}~{train_end2}, val={val_start2}~{val_end2}")
    print(f"  训练集: {len(train2)} 行 ({train2.index.min()} ~ {train2.index.max()})")
    print(f"  验证集: {len(val2)} 行 ({val2.index.min()} ~ {val2.index.max()})")


if __name__ == '__main__':
    main()
