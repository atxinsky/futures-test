# coding=utf-8
"""
数据管理模块
支持从多个数据源下载期货数据，存储到SQLite数据库
"""

import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
import time

# 数据库路径 (Docker环境使用/app/data, 本地环境使用当前目录)
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(__file__), "data"))
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "futures_data.db")

# 期货品种配置 (代码, 名称, 交易所, 上市日期)
FUTURES_SYMBOLS = {
    # 股指期货 - 中金所
    "IF": ("沪深300股指", "CFFEX", "2010-04-16"),
    "IH": ("上证50股指", "CFFEX", "2015-04-16"),
    "IC": ("中证500股指", "CFFEX", "2015-04-16"),
    "IM": ("中证1000股指", "CFFEX", "2022-07-22"),

    # 国债期货
    "T": ("10年期国债", "CFFEX", "2015-03-20"),
    "TF": ("5年期国债", "CFFEX", "2013-09-06"),
    "TS": ("2年期国债", "CFFEX", "2018-08-17"),
    "TL": ("30年期国债", "CFFEX", "2023-04-21"),

    # 贵金属 - 上期所
    "AU": ("黄金", "SHFE", "2008-01-09"),
    "AG": ("白银", "SHFE", "2012-05-10"),

    # 有色金属 - 上期所
    "CU": ("沪铜", "SHFE", "1995-01-01"),
    "AL": ("沪铝", "SHFE", "1995-01-01"),
    "ZN": ("沪锌", "SHFE", "2007-03-26"),
    "PB": ("沪铅", "SHFE", "2011-03-24"),
    "NI": ("沪镍", "SHFE", "2015-03-27"),
    "SN": ("沪锡", "SHFE", "2015-03-27"),
    "SS": ("不锈钢", "SHFE", "2019-09-25"),
    "AO": ("氧化铝", "SHFE", "2023-06-19"),
    "BC": ("国际铜", "INE", "2020-11-19"),

    # 黑色系
    "RB": ("螺纹钢", "SHFE", "2009-03-27"),
    "HC": ("热卷", "SHFE", "2014-03-21"),
    "WR": ("线材", "SHFE", "2009-03-27"),
    "I": ("铁矿石", "DCE", "2013-10-18"),
    "J": ("焦炭", "DCE", "2011-04-15"),
    "JM": ("焦煤", "DCE", "2013-03-22"),
    "SF": ("硅铁", "CZCE", "2014-08-08"),
    "SM": ("锰硅", "CZCE", "2014-08-08"),

    # 能源
    "SC": ("原油", "INE", "2018-03-26"),
    "FU": ("燃油", "SHFE", "2004-08-25"),
    "LU": ("低硫燃油", "INE", "2020-06-22"),
    "BU": ("沥青", "SHFE", "2013-10-09"),

    # 化工
    "L": ("塑料LLDPE", "DCE", "2007-07-31"),
    "V": ("PVC", "DCE", "2009-05-25"),
    "PP": ("聚丙烯", "DCE", "2014-02-28"),
    "EG": ("乙二醇", "DCE", "2018-12-10"),
    "EB": ("苯乙烯", "DCE", "2019-09-26"),
    "PG": ("LPG", "DCE", "2020-03-30"),
    "TA": ("PTA", "CZCE", "2006-12-18"),
    "MA": ("甲醇", "CZCE", "2011-10-28"),
    "SA": ("纯碱", "CZCE", "2019-12-06"),
    "FG": ("玻璃", "CZCE", "2012-12-03"),
    "UR": ("尿素", "CZCE", "2019-08-09"),
    "PF": ("短纤", "CZCE", "2020-10-12"),
    "PX": ("对二甲苯", "CZCE", "2023-09-15"),
    "SH": ("烧碱", "CZCE", "2023-09-15"),

    # 橡胶
    "RU": ("天然橡胶", "SHFE", "1999-01-01"),
    "NR": ("20号胶", "INE", "2019-08-12"),
    "SP": ("纸浆", "SHFE", "2018-11-27"),
    "BR": ("丁二烯橡胶", "SHFE", "2023-07-28"),

    # 油脂油料
    "M": ("豆粕", "DCE", "2000-07-17"),
    "Y": ("豆油", "DCE", "2006-01-09"),
    "P": ("棕榈油", "DCE", "2007-10-29"),
    "A": ("豆一", "DCE", "2002-03-15"),
    "B": ("豆二", "DCE", "2004-12-22"),
    "OI": ("菜油", "CZCE", "2007-06-08"),
    "RM": ("菜粕", "CZCE", "2012-12-28"),

    # 农产品
    "C": ("玉米", "DCE", "2004-09-22"),
    "CS": ("玉米淀粉", "DCE", "2014-12-19"),
    "JD": ("鸡蛋", "DCE", "2013-11-08"),
    "RR": ("粳米", "DCE", "2019-08-16"),
    "CF": ("棉花", "CZCE", "2004-06-01"),
    "SR": ("白糖", "CZCE", "2006-01-06"),
    "AP": ("苹果", "CZCE", "2017-12-22"),
    "CJ": ("红枣", "CZCE", "2019-04-30"),
    "PK": ("花生", "CZCE", "2021-02-01"),
    "CY": ("棉纱", "CZCE", "2017-08-18"),

    # 生猪
    "LH": ("生猪", "DCE", "2021-01-08"),

    # 新能源 - 广期所
    "SI": ("工业硅", "GFEX", "2022-12-22"),
    "LC": ("碳酸锂", "GFEX", "2023-07-21"),

    # 航运
    "EC": ("集运指数", "INE", "2023-08-18"),
}


def init_database():
    """初始化数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 创建期货日线数据表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS futures_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            time DATE NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            open_interest REAL,
            UNIQUE(symbol, time)
        )
    ''')

    # 创建索引
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_time ON futures_daily(symbol, time)')

    # 创建期货分钟数据表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS futures_minute (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,
            period TEXT NOT NULL,
            time DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            open_interest REAL,
            UNIQUE(symbol, period, time)
        )
    ''')

    # 创建分钟数据索引
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_minute_symbol_period_time ON futures_minute(symbol, period, time)')

    # 创建数据更新记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_update_log (
            symbol TEXT PRIMARY KEY,
            last_update DATETIME,
            start_date DATE,
            end_date DATE,
            record_count INTEGER
        )
    ''')

    # 创建分钟数据更新记录表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS minute_update_log (
            symbol TEXT NOT NULL,
            period TEXT NOT NULL,
            last_update DATETIME,
            start_time DATETIME,
            end_time DATETIME,
            record_count INTEGER,
            PRIMARY KEY(symbol, period)
        )
    ''')

    conn.commit()
    conn.close()


def get_data_status() -> pd.DataFrame:
    """获取所有品种的数据状态"""
    init_database()
    conn = sqlite3.connect(DB_PATH)

    # 查询每个品种的数据情况
    query = '''
        SELECT
            symbol,
            MIN(time) as start_date,
            MAX(time) as end_date,
            COUNT(*) as record_count
        FROM futures_daily
        GROUP BY symbol
    '''
    df = pd.read_sql(query, conn)
    conn.close()

    # 合并品种信息
    result = []
    for symbol, (name, exchange, list_date) in FUTURES_SYMBOLS.items():
        row = {'symbol': symbol, 'name': name, 'exchange': exchange, 'list_date': list_date}
        if len(df) > 0 and symbol in df['symbol'].values:
            data_row = df[df['symbol'] == symbol].iloc[0]
            row['start_date'] = data_row['start_date']
            row['end_date'] = data_row['end_date']
            row['record_count'] = data_row['record_count']
        else:
            row['start_date'] = None
            row['end_date'] = None
            row['record_count'] = 0
        result.append(row)

    return pd.DataFrame(result)


def save_to_database(symbol: str, exchange: str, df: pd.DataFrame) -> int:
    """保存数据到数据库"""
    if df is None or len(df) == 0:
        return 0

    init_database()
    conn = sqlite3.connect(DB_PATH)

    # 准备数据
    df = df.copy()
    df['symbol'] = symbol
    df['exchange'] = exchange

    # 确保列名正确
    cols = ['symbol', 'exchange', 'time', 'open', 'high', 'low', 'close', 'volume']
    if 'open_interest' in df.columns:
        cols.append('open_interest')
    else:
        df['open_interest'] = 0
        cols.append('open_interest')

    df = df[cols]
    df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d')

    # 插入数据 (忽略重复)
    inserted = 0
    for _, row in df.iterrows():
        try:
            conn.execute('''
                INSERT OR REPLACE INTO futures_daily
                (symbol, exchange, time, open, high, low, close, volume, open_interest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', tuple(row))
            inserted += 1
        except Exception:
            pass

    conn.commit()

    # 更新日志
    conn.execute('''
        INSERT OR REPLACE INTO data_update_log (symbol, last_update, start_date, end_date, record_count)
        VALUES (?, ?, ?, ?, ?)
    ''', (symbol, datetime.now().isoformat(), df['time'].min(), df['time'].max(), len(df)))

    conn.commit()
    conn.close()

    return inserted


def load_from_database(symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """从数据库加载数据"""
    init_database()
    conn = sqlite3.connect(DB_PATH)

    query = 'SELECT time, open, high, low, close, volume, open_interest FROM futures_daily WHERE symbol = ?'
    params = [symbol]

    if start_date:
        query += ' AND time >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND time <= ?'
        params.append(end_date)

    query += ' ORDER BY time'

    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if len(df) > 0:
        df['time'] = pd.to_datetime(df['time'])

    return df


def download_from_akshare(symbol: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """使用akshare下载日线数据"""
    try:
        import akshare as ak

        # 使用 futures_zh_daily_sina 接口，符号格式为 AU0 (主力连续)
        sina_symbol = f"{symbol.upper()}0"
        df = ak.futures_zh_daily_sina(symbol=sina_symbol)

        if df is None or len(df) == 0:
            return None

        # 标准化列名 (futures_zh_daily_sina 返回的列名)
        df = df.rename(columns={
            'date': 'time',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'hold': 'open_interest'
        })

        # 确保必要的列存在
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"缺少列: {col}")
                return None

        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)

        # 过滤日期范围
        if start_date:
            df = df[df['time'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['time'] <= pd.to_datetime(end_date)]

        return df

    except Exception as e:
        print(f"akshare下载失败 {symbol}: {e}")
        return None


# 分钟周期映射
MINUTE_PERIODS = {
    "5分钟": "5",
    "15分钟": "15",
    "30分钟": "30",
    "60分钟": "60",
}


def download_minute_from_akshare(symbol: str, period: str = "60") -> Optional[pd.DataFrame]:
    """
    使用akshare下载分钟数据

    参数:
        symbol: 品种代码如 AU, RB
        period: 周期 '5', '15', '30', '60'

    返回:
        DataFrame 包含 time, open, high, low, close, volume, open_interest
    """
    try:
        import akshare as ak

        # 使用 futures_zh_minute_sina 接口
        sina_symbol = f"{symbol.upper()}0"
        df = ak.futures_zh_minute_sina(symbol=sina_symbol, period=period)

        if df is None or len(df) == 0:
            return None

        # 标准化列名
        df = df.rename(columns={
            'datetime': 'time',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'hold': 'open_interest'
        })

        # 确保必要的列存在
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"分钟数据缺少列: {col}")
                return None

        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)

        return df

    except Exception as e:
        print(f"akshare分钟数据下载失败 {symbol} {period}分钟: {e}")
        return None


def save_minute_to_database(symbol: str, exchange: str, period: str, df: pd.DataFrame) -> int:
    """保存分钟数据到数据库"""
    if df is None or len(df) == 0:
        return 0

    init_database()
    conn = sqlite3.connect(DB_PATH)

    # 准备数据
    df = df.copy()
    df['symbol'] = symbol
    df['exchange'] = exchange
    df['period'] = period

    # 确保列名正确
    cols = ['symbol', 'exchange', 'period', 'time', 'open', 'high', 'low', 'close', 'volume']
    if 'open_interest' in df.columns:
        cols.append('open_interest')
    else:
        df['open_interest'] = 0
        cols.append('open_interest')

    df = df[cols]
    df['time'] = pd.to_datetime(df['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # 插入数据 (忽略重复)
    inserted = 0
    for _, row in df.iterrows():
        try:
            conn.execute('''
                INSERT OR REPLACE INTO futures_minute
                (symbol, exchange, period, time, open, high, low, close, volume, open_interest)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', tuple(row))
            inserted += 1
        except Exception:
            pass

    conn.commit()

    # 更新日志
    conn.execute('''
        INSERT OR REPLACE INTO minute_update_log (symbol, period, last_update, start_time, end_time, record_count)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (symbol, period, datetime.now().isoformat(), df['time'].min(), df['time'].max(), len(df)))

    conn.commit()
    conn.close()

    return inserted


def load_minute_from_database(symbol: str, period: str, start_time: str = None, end_time: str = None) -> pd.DataFrame:
    """从数据库加载分钟数据"""
    init_database()
    conn = sqlite3.connect(DB_PATH)

    query = 'SELECT time, open, high, low, close, volume, open_interest FROM futures_minute WHERE symbol = ? AND period = ?'
    params = [symbol, period]

    if start_time:
        query += ' AND time >= ?'
        params.append(start_time)
    if end_time:
        query += ' AND time <= ?'
        params.append(end_time)

    query += ' ORDER BY time'

    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if len(df) > 0:
        df['time'] = pd.to_datetime(df['time'])

    return df


def download_minute_symbol(
    symbol: str,
    period: str = "60",
    save_to_db: bool = True
) -> Tuple[bool, str, int]:
    """
    下载单个品种的分钟数据

    参数:
        symbol: 品种代码
        period: 周期 '5', '15', '30', '60'
        save_to_db: 是否保存到数据库

    返回: (成功与否, 消息, 数据条数)
    """
    if symbol not in FUTURES_SYMBOLS:
        return False, f"未知品种: {symbol}", 0

    name, exchange, list_date = FUTURES_SYMBOLS[symbol]

    # 下载数据
    df = download_minute_from_akshare(symbol, period)

    if df is None or len(df) == 0:
        return False, f"{symbol} ({name}) {period}分钟 无数据", 0

    # 保存到数据库
    if save_to_db:
        count = save_minute_to_database(symbol, exchange, period, df)
        return True, f"{symbol} ({name}) {period}分钟 下载成功", count
    else:
        return True, f"{symbol} ({name}) {period}分钟 下载成功", len(df)


def get_minute_data_status() -> pd.DataFrame:
    """获取所有品种的分钟数据状态"""
    init_database()
    conn = sqlite3.connect(DB_PATH)

    # 查询每个品种每个周期的数据情况
    query = '''
        SELECT
            symbol,
            period,
            MIN(time) as start_time,
            MAX(time) as end_time,
            COUNT(*) as record_count
        FROM futures_minute
        GROUP BY symbol, period
    '''
    df = pd.read_sql(query, conn)
    conn.close()

    return df


def download_from_tushare(symbol: str, start_date: str = None, end_date: str = None, token: str = None) -> Optional[pd.DataFrame]:
    """使用tushare下载数据 (需要token)"""
    if not token:
        return None

    try:
        import tushare as ts
        ts.set_token(token)
        pro = ts.pro_api()

        # tushare期货主力连续代码格式
        ts_symbol = f"{symbol}.CFX" if symbol in ['IF', 'IH', 'IC', 'IM', 'T', 'TF', 'TS', 'TL'] else f"{symbol}.ZCE"

        df = pro.fut_daily(
            ts_code=ts_symbol,
            start_date=start_date.replace('-', '') if start_date else None,
            end_date=end_date.replace('-', '') if end_date else None
        )

        if df is None or len(df) == 0:
            return None

        df = df.rename(columns={
            'trade_date': 'time',
            'vol': 'volume',
            'oi': 'open_interest'
        })

        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)

        return df[['time', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]

    except Exception as e:
        print(f"tushare下载失败 {symbol}: {e}")
        return None


def download_symbol(
    symbol: str,
    start_date: str = None,
    end_date: str = None,
    source: str = "akshare",
    tushare_token: str = None,
    save_to_db: bool = True
) -> Tuple[bool, str, int]:
    """
    下载单个品种数据

    返回: (成功与否, 消息, 数据条数)
    """
    if symbol not in FUTURES_SYMBOLS:
        return False, f"未知品种: {symbol}", 0

    name, exchange, list_date = FUTURES_SYMBOLS[symbol]

    # 设置默认日期范围
    if not start_date:
        start_date = list_date
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # 下载数据
    df = None
    if source == "akshare":
        df = download_from_akshare(symbol, start_date, end_date)
    elif source == "tushare" and tushare_token:
        df = download_from_tushare(symbol, start_date, end_date, tushare_token)

    if df is None or len(df) == 0:
        return False, f"{symbol} ({name}) 无数据", 0

    # 保存到数据库
    if save_to_db:
        count = save_to_database(symbol, exchange, df)
        return True, f"{symbol} ({name}) 下载成功", count
    else:
        return True, f"{symbol} ({name}) 下载成功", len(df)


def download_batch(
    symbols: List[str],
    start_date: str = None,
    end_date: str = None,
    source: str = "akshare",
    tushare_token: str = None,
    progress_callback=None
) -> Dict[str, Tuple[bool, str, int]]:
    """
    批量下载数据

    progress_callback: 进度回调函数 (current, total, symbol, message)
    """
    results = {}
    total = len(symbols)

    for i, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback(i, total, symbol, "下载中...")

        success, msg, count = download_symbol(
            symbol, start_date, end_date, source, tushare_token
        )
        results[symbol] = (success, msg, count)

        # 避免请求过快
        time.sleep(0.5)

    if progress_callback:
        progress_callback(total, total, "", "完成")

    return results


def export_to_csv(symbol: str, output_dir: str, start_date: str = None, end_date: str = None) -> str:
    """导出数据到CSV文件"""
    df = load_from_database(symbol, start_date, end_date)
    if len(df) == 0:
        return None

    if symbol in FUTURES_SYMBOLS:
        name, exchange, _ = FUTURES_SYMBOLS[symbol]
    else:
        name, exchange = symbol, "UNKNOWN"

    filename = f"{exchange}_{symbol}_主力连续.csv"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')

    return filepath


def get_symbol_list_by_category() -> Dict[str, List[Tuple[str, str]]]:
    """按类别获取品种列表"""
    categories = {
        "股指期货": ["IF", "IH", "IC", "IM"],
        "国债期货": ["T", "TF", "TS", "TL"],
        "贵金属": ["AU", "AG"],
        "有色金属": ["CU", "AL", "ZN", "PB", "NI", "SN", "SS", "AO", "BC"],
        "黑色系": ["RB", "HC", "WR", "I", "J", "JM", "SF", "SM"],
        "能源": ["SC", "FU", "LU", "BU"],
        "化工": ["L", "V", "PP", "EG", "EB", "PG", "TA", "MA", "SA", "FG", "UR", "PF", "PX", "SH"],
        "橡胶木材": ["RU", "NR", "SP", "BR"],
        "油脂油料": ["M", "Y", "P", "A", "B", "OI", "RM"],
        "农产品": ["C", "CS", "JD", "RR", "CF", "SR", "AP", "CJ", "PK", "CY"],
        "生猪": ["LH"],
        "新能源": ["SI", "LC"],
        "航运": ["EC"],
    }

    result = {}
    for cat, symbols in categories.items():
        result[cat] = [(s, FUTURES_SYMBOLS[s][0]) for s in symbols if s in FUTURES_SYMBOLS]

    return result


# ============== 天勤数据支持 ==============

# 天勤数据库路径
TQ_DB_PATH = os.path.join(DATA_DIR, "futures_tq.db")

# 周期映射 (天勤 -> 本地)
TQ_PERIOD_MAP = {
    "5m": "5",
    "15m": "15",
    "30m": "30",
    "60m": "60",
    "1d": "日线",
}


def load_from_tianqin(
    symbol: str,
    period: str = "1d",
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    从天勤数据库加载数据

    Args:
        symbol: 品种代码 (IF, AU, RB等)
        period: 周期 ('5m', '15m', '30m', '60m', '1d')
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)

    Returns:
        DataFrame with columns: time, open, high, low, close, volume, open_interest
    """
    if not os.path.exists(TQ_DB_PATH):
        return pd.DataFrame()

    conn = sqlite3.connect(TQ_DB_PATH)

    query = """
        SELECT datetime as time, open, high, low, close, volume, open_interest
        FROM kline_data
        WHERE symbol = ? AND period = ?
    """
    params = [symbol.upper(), period]

    if start_date:
        query += " AND datetime >= ?"
        params.append(start_date)

    if end_date:
        query += " AND datetime <= ?"
        params.append(end_date + " 23:59:59")

    query += " ORDER BY datetime"

    try:
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if len(df) > 0:
            df['time'] = pd.to_datetime(df['time'])

        return df
    except Exception as e:
        conn.close()
        return pd.DataFrame()


def get_tianqin_data_status() -> pd.DataFrame:
    """获取天勤数据库中的数据状态"""
    if not os.path.exists(TQ_DB_PATH):
        return pd.DataFrame()

    conn = sqlite3.connect(TQ_DB_PATH)

    try:
        query = """
            SELECT
                symbol,
                period,
                COUNT(*) as record_count,
                MIN(datetime) as start_date,
                MAX(datetime) as end_date
            FROM kline_data
            GROUP BY symbol, period
            ORDER BY symbol, period
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except:
        conn.close()
        return pd.DataFrame()


def load_data_auto(
    symbol: str,
    period: str = "日线",
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    自动选择最佳数据源加载数据

    优先级: 天勤 > 本地数据库 > akshare下载

    Args:
        symbol: 品种代码
        period: 周期 ('5分钟', '15分钟', '30分钟', '60分钟', '日线' 或 '5m', '15m', '30m', '60m', '1d')
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame
    """
    # 标准化周期格式
    period_map = {
        "5分钟": "5m", "15分钟": "15m", "30分钟": "30m", "60分钟": "60m", "日线": "1d",
        "5m": "5m", "15m": "15m", "30m": "30m", "60m": "60m", "1d": "1d",
        "5": "5m", "15": "15m", "30": "30m", "60": "60m",
    }
    tq_period = period_map.get(period, "1d")

    # 1. 尝试从天勤数据库加载
    df = load_from_tianqin(symbol, tq_period, start_date, end_date)
    if len(df) > 0:
        return df

    # 2. 尝试从本地数据库加载
    if tq_period == "1d":
        df = load_from_database(symbol, start_date, end_date)
        if len(df) > 0:
            return df
    else:
        # 分钟数据
        local_period = {"5m": "5", "15m": "15", "30m": "30", "60m": "60"}.get(tq_period)
        if local_period:
            df = load_minute_from_database(symbol, local_period, start_date, end_date)
            if len(df) > 0:
                return df

    # 3. 如果没有本地数据，尝试下载
    if tq_period == "1d":
        download_df = download_from_akshare(symbol, start_date, end_date)
        if download_df is not None and len(download_df) > 0:
            # 保存到数据库
            if symbol in FUTURES_SYMBOLS:
                _, exchange, _ = FUTURES_SYMBOLS[symbol]
                save_to_database(symbol, exchange, download_df)
            return download_df
    else:
        # 分钟数据从akshare下载
        local_period = {"5m": "5", "15m": "15", "30m": "30", "60m": "60"}.get(tq_period)
        if local_period:
            download_df = download_minute_from_akshare(symbol, local_period)
            if download_df is not None and len(download_df) > 0:
                if symbol in FUTURES_SYMBOLS:
                    _, exchange, _ = FUTURES_SYMBOLS[symbol]
                    save_minute_to_database(symbol, exchange, local_period, download_df)
                return download_df

    return pd.DataFrame()


# 初始化数据库
init_database()
