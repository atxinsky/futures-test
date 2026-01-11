# coding=utf-8
"""
调试ADX计算问题
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("调试ADX计算问题")
    print("=" * 60)

    # 1. 加载数据
    from utils.data_loader import load_futures_data
    df = load_futures_data('IF', '2022-01-01', '2025-12-31', '1d', auto_download=False)
    print(f"数据行数: {len(df)}")
    print(f"数据列: {df.columns.tolist()}")
    print(f"索引类型: {type(df.index)}")
    print(f"\n前3行:")
    print(df.head(3))

    # 2. 检查数据类型
    print("\n数据类型:")
    print(df.dtypes)

    # 3. 手动计算ADX各步骤
    print("\n手动计算ADX...")

    adx_len = 14

    # TR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    print(f"TR 前10个: {tr.head(10).tolist()}")
    print(f"TR NaN数: {tr.isna().sum()}")

    # DM
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    print(f"up_move 前10个: {up_move.head(10).tolist()}")
    print(f"down_move 前10个: {down_move.head(10).tolist()}")

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    print(f"plus_dm 前10个: {plus_dm[:10].tolist()}")
    print(f"minus_dm 前10个: {minus_dm[:10].tolist()}")

    # Smoothed values
    atr_adx = tr.rolling(window=adx_len).mean()
    print(f"atr_adx 前20个: {atr_adx.head(20).tolist()}")
    print(f"atr_adx NaN数: {atr_adx.isna().sum()}")

    plus_dm_smooth = pd.Series(plus_dm).rolling(window=adx_len).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=adx_len).mean()
    print(f"plus_dm_smooth 前20个: {plus_dm_smooth.head(20).tolist()}")

    # DI
    plus_di = 100 * plus_dm_smooth / (atr_adx + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_adx + 1e-10)
    print(f"plus_di 前20个: {plus_di.head(20).tolist()}")
    print(f"minus_di 前20个: {minus_di.head(20).tolist()}")

    # DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    print(f"dx 前30个: {dx.head(30).tolist()}")
    print(f"dx NaN数: {dx.isna().sum()}")

    # ADX
    adx = dx.rolling(window=adx_len).mean()
    print(f"adx 前40个: {adx.head(40).tolist()}")
    print(f"adx NaN数: {adx.isna().sum()}")
    print(f"adx 非NaN数: {(~adx.isna()).sum()}")

    # 检查非NaN的ADX值
    adx_valid = adx.dropna()
    if len(adx_valid) > 0:
        print(f"\n有效ADX值:")
        print(f"  范围: {adx_valid.min():.2f} ~ {adx_valid.max():.2f}")
        print(f"  均值: {adx_valid.mean():.2f}")
        print(f"  前10个非NaN: {adx_valid.head(10).tolist()}")
    else:
        print("\n没有有效的ADX值!")

    # 4. 检查数据索引问题
    print("\n检查索引问题...")
    print(f"plus_dm_smooth 索引: {plus_dm_smooth.index[:5].tolist() if hasattr(plus_dm_smooth, 'index') else 'No index'}")
    print(f"atr_adx 索引: {atr_adx.index[:5].tolist() if hasattr(atr_adx, 'index') else 'No index'}")

    # 5. 使用对齐的方式重新计算
    print("\n使用对齐方式重新计算...")
    df_calc = df.copy()
    df_calc['tr'] = tr
    df_calc['plus_dm'] = plus_dm
    df_calc['minus_dm'] = minus_dm
    df_calc['atr_adx'] = df_calc['tr'].rolling(window=adx_len).mean()
    df_calc['plus_dm_smooth'] = df_calc['plus_dm'].rolling(window=adx_len).mean()
    df_calc['minus_dm_smooth'] = df_calc['minus_dm'].rolling(window=adx_len).mean()
    df_calc['plus_di'] = 100 * df_calc['plus_dm_smooth'] / (df_calc['atr_adx'] + 1e-10)
    df_calc['minus_di'] = 100 * df_calc['minus_dm_smooth'] / (df_calc['atr_adx'] + 1e-10)
    df_calc['dx'] = 100 * abs(df_calc['plus_di'] - df_calc['minus_di']) / (df_calc['plus_di'] + df_calc['minus_di'] + 1e-10)
    df_calc['adx'] = df_calc['dx'].rolling(window=adx_len).mean()

    print(f"对齐后ADX:")
    print(f"  NaN数: {df_calc['adx'].isna().sum()}")
    print(f"  非NaN数: {(~df_calc['adx'].isna()).sum()}")

    adx_valid2 = df_calc['adx'].dropna()
    if len(adx_valid2) > 0:
        print(f"  范围: {adx_valid2.min():.2f} ~ {adx_valid2.max():.2f}")
        print(f"  均值: {adx_valid2.mean():.2f}")
        print(f"  > 22 的天数: {(adx_valid2 > 22).sum()}")


if __name__ == '__main__':
    main()
