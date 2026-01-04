# coding=utf-8
"""
天勤量化数据下载器
支持下载期货历史K线数据到SQLite数据库
"""

import os
import sqlite3
import pandas as pd
from datetime import date, datetime, timedelta
from typing import List, Dict, Optional
from tqsdk import TqApi, TqAuth
from tqsdk.tools import DataDownloader
import time

# 天勤账号配置
TQ_USER = "tretra"
TQ_PASSWORD = "LOVE*101512"

# 数据库路径
DB_PATH = os.path.join(os.path.dirname(__file__), "data", "futures_tq.db")

# 期货品种映射 (本地代码 -> 天勤代码)
SYMBOL_MAPPING = {
    # 中金所 CFFEX
    "IF": "KQ.m@CFFEX.IF",  # 沪深300
    "IH": "KQ.m@CFFEX.IH",  # 上证50
    "IC": "KQ.m@CFFEX.IC",  # 中证500
    "IM": "KQ.m@CFFEX.IM",  # 中证1000

    # 上期所 SHFE
    "AU": "KQ.m@SHFE.au",   # 黄金
    "AG": "KQ.m@SHFE.ag",   # 白银
    "CU": "KQ.m@SHFE.cu",   # 铜
    "AL": "KQ.m@SHFE.al",   # 铝
    "ZN": "KQ.m@SHFE.zn",   # 锌
    "NI": "KQ.m@SHFE.ni",   # 镍
    "RB": "KQ.m@SHFE.rb",   # 螺纹钢
    "HC": "KQ.m@SHFE.hc",   # 热轧卷板
    "RU": "KQ.m@SHFE.ru",   # 橡胶
    "FU": "KQ.m@SHFE.fu",   # 燃油
    "BU": "KQ.m@SHFE.bu",   # 沥青
    "SS": "KQ.m@SHFE.ss",   # 不锈钢
    "SP": "KQ.m@SHFE.sp",   # 纸浆

    # 上期能源 INE
    "SC": "KQ.m@INE.sc",    # 原油
    "NR": "KQ.m@INE.nr",    # 20号胶
    "LU": "KQ.m@INE.lu",    # 低硫燃油
    "BC": "KQ.m@INE.bc",    # 国际铜

    # 大商所 DCE
    "M": "KQ.m@DCE.m",      # 豆粕
    "Y": "KQ.m@DCE.y",      # 豆油
    "A": "KQ.m@DCE.a",      # 豆一
    "C": "KQ.m@DCE.c",      # 玉米
    "CS": "KQ.m@DCE.cs",    # 淀粉
    "P": "KQ.m@DCE.p",      # 棕榈油
    "I": "KQ.m@DCE.i",      # 铁矿石
    "J": "KQ.m@DCE.j",      # 焦炭
    "JM": "KQ.m@DCE.jm",    # 焦煤
    "L": "KQ.m@DCE.l",      # 塑料
    "PP": "KQ.m@DCE.pp",    # 聚丙烯
    "V": "KQ.m@DCE.v",      # PVC
    "EG": "KQ.m@DCE.eg",    # 乙二醇
    "EB": "KQ.m@DCE.eb",    # 苯乙烯
    "PG": "KQ.m@DCE.pg",    # LPG
    "RR": "KQ.m@DCE.rr",    # 粳米
    "JD": "KQ.m@DCE.jd",    # 鸡蛋
    "LH": "KQ.m@DCE.lh",    # 生猪

    # 郑商所 CZCE
    "CF": "KQ.m@CZCE.CF",   # 棉花
    "SR": "KQ.m@CZCE.SR",   # 白糖
    "TA": "KQ.m@CZCE.TA",   # PTA
    "MA": "KQ.m@CZCE.MA",   # 甲醇
    "OI": "KQ.m@CZCE.OI",   # 菜油
    "RM": "KQ.m@CZCE.RM",   # 菜粕
    "FG": "KQ.m@CZCE.FG",   # 玻璃
    "SA": "KQ.m@CZCE.SA",   # 纯碱
    "AP": "KQ.m@CZCE.AP",   # 苹果
    "CJ": "KQ.m@CZCE.CJ",   # 红枣
    "PF": "KQ.m@CZCE.PF",   # 短纤
    "ZC": "KQ.m@CZCE.ZC",   # 动力煤
    "SF": "KQ.m@CZCE.SF",   # 硅铁
    "SM": "KQ.m@CZCE.SM",   # 锰硅
    "UR": "KQ.m@CZCE.UR",   # 尿素
    "PK": "KQ.m@CZCE.PK",   # 花生
}

# 周期映射 (秒)
PERIOD_MAPPING = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "60m": 3600,
    "1d": 86400,
}

# 周期显示名称
PERIOD_NAMES = {
    "1m": "1分钟",
    "5m": "5分钟",
    "15m": "15分钟",
    "30m": "30分钟",
    "60m": "60分钟",
    "1d": "日线",
}


def get_db_connection():
    """获取数据库连接"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    return conn


def init_database():
    """初始化数据库表"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # 创建K线数据表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kline_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            period TEXT NOT NULL,
            datetime TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL DEFAULT 0,
            open_interest REAL DEFAULT 0,
            UNIQUE(symbol, period, datetime)
        )
    """)

    # 创建索引
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_kline_symbol_period
        ON kline_data(symbol, period)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_kline_datetime
        ON kline_data(datetime)
    """)

    conn.commit()
    conn.close()


def download_symbol_data(
    api: TqApi,
    symbol: str,
    period: str,
    start_date: date,
    end_date: date,
    temp_dir: str = None
) -> Optional[pd.DataFrame]:
    """
    下载单个品种的K线数据

    Args:
        api: 天勤API实例
        symbol: 本地品种代码 (如 IF, AU)
        period: 周期 (1m, 5m, 15m, 30m, 60m, 1d)
        start_date: 开始日期
        end_date: 结束日期
        temp_dir: 临时文件目录

    Returns:
        DataFrame或None
    """
    if symbol not in SYMBOL_MAPPING:
        print(f"  X 未知品种: {symbol}")
        return None

    tq_symbol = SYMBOL_MAPPING[symbol]
    dur_sec = PERIOD_MAPPING.get(period, 300)

    if temp_dir is None:
        temp_dir = os.path.join(os.path.dirname(__file__), "data", "temp")
    os.makedirs(temp_dir, exist_ok=True)

    csv_file = os.path.join(temp_dir, f"{symbol}_{period}.csv")

    try:
        # 创建下载任务
        download_task = DataDownloader(
            api,
            symbol_list=[tq_symbol],
            dur_sec=dur_sec,
            start_dt=start_date,
            end_dt=end_date,
            csv_file_name=csv_file
        )

        # 等待下载完成
        last_progress = 0
        while not download_task.is_finished():
            api.wait_update()
            progress = download_task.get_progress()
            if progress - last_progress > 0.1:  # 每10%打印一次
                print(f"  下载进度: {progress:.0%}", end='\r')
                last_progress = progress

        # 读取CSV
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)

            # 标准化列名
            df = df.rename(columns={
                'datetime': 'datetime',
                f'{tq_symbol}.open': 'open',
                f'{tq_symbol}.high': 'high',
                f'{tq_symbol}.low': 'low',
                f'{tq_symbol}.close': 'close',
                f'{tq_symbol}.volume': 'volume',
                f'{tq_symbol}.open_oi': 'open_interest',
                f'{tq_symbol}.close_oi': 'close_interest',
            })

            # 只保留需要的列
            cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            if 'open_interest' in df.columns:
                cols.append('open_interest')
            elif 'close_interest' in df.columns:
                df['open_interest'] = df['close_interest']
                cols.append('open_interest')
            else:
                df['open_interest'] = 0
                cols.append('open_interest')

            df = df[cols]

            # 转换时间格式
            df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')

            # 删除临时文件
            os.remove(csv_file)

            return df

    except Exception as e:
        print(f"  X 下载失败: {e}")
        return None

    return None


def save_to_database(symbol: str, period: str, df: pd.DataFrame):
    """保存数据到数据库"""
    if df is None or len(df) == 0:
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()

    # 添加品种和周期列
    df['symbol'] = symbol
    df['period'] = period

    # 插入数据 (忽略重复)
    inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO kline_data
                (symbol, period, datetime, open, high, low, close, volume, open_interest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row['symbol'], row['period'], row['datetime'],
                row['open'], row['high'], row['low'], row['close'],
                row['volume'], row.get('open_interest', 0)
            ))
            inserted += 1
        except Exception as e:
            pass

    conn.commit()
    conn.close()

    return inserted


def download_all_data(
    symbols: List[str] = None,
    periods: List[str] = None,
    start_date: date = None,
    end_date: date = None,
    callback = None
):
    """
    批量下载数据

    Args:
        symbols: 品种列表，默认所有
        periods: 周期列表，默认 ['5m', '15m', '30m', '60m', '1d']
        start_date: 开始日期，默认3年前
        end_date: 结束日期，默认今天
        callback: 进度回调函数 callback(symbol, period, progress, total)
    """
    if symbols is None:
        symbols = list(SYMBOL_MAPPING.keys())

    if periods is None:
        periods = ['5m', '15m', '30m', '60m', '1d']

    if start_date is None:
        start_date = date.today() - timedelta(days=365*3)

    if end_date is None:
        end_date = date.today()

    # 初始化数据库
    init_database()

    # 连接天勤
    print("连接天勤量化...")
    api = TqApi(auth=TqAuth(TQ_USER, TQ_PASSWORD))

    total = len(symbols) * len(periods)
    current = 0
    results = []

    try:
        for symbol in symbols:
            for period in periods:
                current += 1
                period_name = PERIOD_NAMES.get(period, period)
                print(f"[{current}/{total}] {symbol} {period_name}...", end=' ')

                if callback:
                    callback(symbol, period, current, total)

                # 下载数据
                df = download_symbol_data(api, symbol, period, start_date, end_date)

                if df is not None and len(df) > 0:
                    # 保存到数据库
                    count = save_to_database(symbol, period, df)
                    print(f"OK {count}条")
                    results.append({
                        'symbol': symbol,
                        'period': period,
                        'count': count,
                        'status': 'success'
                    })
                else:
                    print("X 无数据")
                    results.append({
                        'symbol': symbol,
                        'period': period,
                        'count': 0,
                        'status': 'failed'
                    })

                # 避免请求过快
                time.sleep(0.5)

    finally:
        api.close()

    return results


def get_data_from_db(
    symbol: str,
    period: str,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    从数据库读取数据

    Args:
        symbol: 品种代码
        period: 周期
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)

    Returns:
        DataFrame
    """
    conn = get_db_connection()

    query = """
        SELECT datetime, open, high, low, close, volume, open_interest
        FROM kline_data
        WHERE symbol = ? AND period = ?
    """
    params = [symbol, period]

    if start_date:
        query += " AND datetime >= ?"
        params.append(start_date)

    if end_date:
        query += " AND datetime <= ?"
        params.append(end_date + " 23:59:59")

    query += " ORDER BY datetime"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    return df


def get_available_data() -> Dict:
    """获取数据库中已有数据的统计"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            SELECT symbol, period, COUNT(*) as count,
                   MIN(datetime) as start_date,
                   MAX(datetime) as end_date
            FROM kline_data
            GROUP BY symbol, period
            ORDER BY symbol, period
        """)

        rows = cursor.fetchall()

        data = {}
        for row in rows:
            symbol, period, count, start_date, end_date = row
            if symbol not in data:
                data[symbol] = {}
            data[symbol][period] = {
                'count': count,
                'start_date': start_date,
                'end_date': end_date
            }

        return data

    except:
        return {}

    finally:
        conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='天勤期货数据下载器')
    parser.add_argument('--symbols', nargs='+', help='品种列表')
    parser.add_argument('--periods', nargs='+', default=['5m', '15m', '30m', '60m', '1d'], help='周期列表')
    parser.add_argument('--start', help='开始日期 YYYYMMDD')
    parser.add_argument('--end', help='结束日期 YYYYMMDD')
    parser.add_argument('--list', action='store_true', help='列出已有数据')

    args = parser.parse_args()

    if args.list:
        print("=== 数据库中已有数据 ===")
        data = get_available_data()
        for symbol, periods in data.items():
            print(f"\n{symbol}:")
            for period, info in periods.items():
                print(f"  {period}: {info['count']}条 ({info['start_date'][:10]} ~ {info['end_date'][:10]})")
    else:
        symbols = args.symbols
        periods = args.periods
        start_date = datetime.strptime(args.start, '%Y%m%d').date() if args.start else None
        end_date = datetime.strptime(args.end, '%Y%m%d').date() if args.end else None

        print("=== 天勤期货数据下载 ===")
        print(f"品种: {symbols if symbols else '全部'}")
        print(f"周期: {periods}")
        print(f"时间: {start_date or '3年前'} ~ {end_date or '今天'}")
        print()

        results = download_all_data(symbols, periods, start_date, end_date)

        print("\n=== 下载完成 ===")
        success = sum(1 for r in results if r['status'] == 'success')
        total_count = sum(r['count'] for r in results)
        print(f"成功: {success}/{len(results)}")
        print(f"总数据量: {total_count}条")
