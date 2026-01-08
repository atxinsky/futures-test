# coding=utf-8
"""
ETF数据服务
使用与期货系统相同的数据库，支持AKShare获取ETF数据
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import time
import os

logger = logging.getLogger(__name__)

# 使用与期货系统相同的数据目录
DATA_DIR = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "futures_data.db")  # 与期货共用数据库

# ETF池配置
ETF_POOLS = {
    "海外指数": {
        "513100.SH": "纳指ETF",
        "513500.SH": "标普500ETF",
        "159920.SZ": "恒生ETF",
        "513030.SH": "德国DAX",
    },
    "A股宽基": {
        "510300.SH": "沪深300ETF",
        "510050.SH": "上证50ETF",
        "510500.SH": "中证500ETF",
        "159915.SZ": "创业板ETF",
        "588000.SH": "科创50ETF",
    },
    "行业主题": {
        "512480.SH": "半导体ETF",
        "515030.SH": "新能车ETF",
        "512010.SH": "医药ETF",
        "159928.SZ": "消费ETF",
        "512880.SH": "证券ETF",
        "512660.SH": "军工ETF",
        "516010.SH": "游戏ETF",
    },
    "商品": {
        "518880.SH": "黄金ETF",
        "161226.SZ": "白银基金",
        "159985.SZ": "豆粕ETF",
    },
    "债券": {
        "511260.SH": "十年国债",
        "511010.SH": "国债ETF",
    },
    "防守型": {
        "512890.SH": "红利低波",
        "513050.SH": "中概互联",
    }
}

# 所有ETF代码
ALL_ETFS = {}
for category, etfs in ETF_POOLS.items():
    ALL_ETFS.update(etfs)

# BigBrother V14 默认池
BIGBROTHER_POOL = [
    "513100.SH",  # 纳指ETF
    "513050.SH",  # 中概互联
    "512480.SH",  # 半导体ETF
    "515030.SH",  # 新能车ETF
    "518880.SH",  # 黄金ETF
    "512890.SH",  # 红利低波
    "588000.SH",  # 科创50
    "516010.SH",  # 游戏动漫
]


class ETFDataService:
    """
    ETF数据服务

    功能：
    1. 从AKShare获取ETF日线数据
    2. 与期货系统共用SQLite数据库
    3. 数据缓存和增量更新
    4. 技术指标计算
    """

    def __init__(self, db_path: str = None):
        """
        初始化数据服务

        Args:
            db_path: 数据库路径，默认与期货系统共用
        """
        self.db_path = db_path or DB_PATH
        self._init_database()
        self._cache = {}

    def _init_database(self):
        """初始化ETF数据表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ETF日线数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_daily (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                amount REAL,
                UNIQUE(code, date)
            )
        """)

        # 创建索引
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_etf_daily_code_date
            ON etf_daily(code, date)
        """)

        # ETF基本信息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_info (
                code TEXT PRIMARY KEY,
                name TEXT,
                category TEXT,
                last_update TEXT
            )
        """)

        # 数据更新记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS etf_update_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                update_time TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                rows_added INTEGER
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"ETF数据表初始化完成: {self.db_path}")

    def fetch_from_akshare(self, code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        从AKShare获取ETF日线数据

        Args:
            code: ETF代码 (如 '513100.SH')
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, amount
        """
        try:
            import akshare as ak
        except ImportError:
            logger.error("请安装akshare: pip install akshare")
            raise ImportError("akshare未安装，请运行: pip install akshare")

        # 转换代码格式
        ak_code = code.split(".")[0]

        try:
            df = ak.fund_etf_hist_em(
                symbol=ak_code,
                period="daily",
                start_date=start_date.replace("-", "") if start_date else "20190101",
                end_date=end_date.replace("-", "") if end_date else datetime.now().strftime("%Y%m%d"),
                adjust="qfq"
            )

            if df is None or len(df) == 0:
                logger.warning(f"AKShare返回空数据: {code}")
                return pd.DataFrame()

            # 标准化列名
            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount"
            })

            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            columns = ["date", "open", "high", "low", "close", "volume", "amount"]
            df = df[[col for col in columns if col in df.columns]]

            logger.info(f"从AKShare获取数据成功: {code}, {len(df)}条记录")
            return df

        except Exception as e:
            logger.error(f"从AKShare获取数据失败: {code}, 错误: {e}")
            return pd.DataFrame()

    def save_to_db(self, code: str, df: pd.DataFrame) -> int:
        """保存数据到数据库"""
        if df is None or len(df) == 0:
            return 0

        conn = sqlite3.connect(self.db_path)

        df = df.copy()
        df["code"] = code

        rows_before = pd.read_sql("SELECT COUNT(*) as cnt FROM etf_daily WHERE code=?",
                                   conn, params=[code]).iloc[0]["cnt"]

        df.to_sql("etf_daily", conn, if_exists="append", index=False,
                  method=lambda table, conn, keys, data_iter:
                  conn.executemany(
                      f"INSERT OR IGNORE INTO {table.name} ({','.join(keys)}) VALUES ({','.join(['?']*len(keys))})",
                      list(data_iter)
                  ))

        rows_after = pd.read_sql("SELECT COUNT(*) as cnt FROM etf_daily WHERE code=?",
                                  conn, params=[code]).iloc[0]["cnt"]

        rows_added = rows_after - rows_before

        # 记录更新日志
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO etf_update_log (code, update_time, start_date, end_date, rows_added)
            VALUES (?, ?, ?, ?, ?)
        """, (code, datetime.now().isoformat(),
              df["date"].min(), df["date"].max(), rows_added))

        conn.commit()
        conn.close()

        if code in self._cache:
            del self._cache[code]

        logger.info(f"保存ETF数据: {code}, 新增{rows_added}条")
        return rows_added

    def get_data(self, code: str, start_date: str = None, end_date: str = None,
                 use_cache: bool = True) -> pd.DataFrame:
        """获取ETF数据"""
        cache_key = f"{code}_{start_date}_{end_date}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        conn = sqlite3.connect(self.db_path)

        query = "SELECT date, open, high, low, close, volume, amount FROM etf_daily WHERE code=?"
        params = [code]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        df = pd.read_sql(query, conn, params=params)
        conn.close()

        if len(df) == 0:
            logger.info(f"本地无数据，从网络获取: {code}")
            df = self.fetch_from_akshare(code, start_date, end_date)
            if len(df) > 0:
                self.save_to_db(code, df)

        if use_cache and len(df) > 0:
            self._cache[cache_key] = df.copy()

        return df

    def update_data(self, code: str, force: bool = False) -> int:
        """更新ETF数据（增量更新）"""
        conn = sqlite3.connect(self.db_path)

        if force:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM etf_daily WHERE code=?", [code])
            conn.commit()
            start_date = "2019-01-01"
        else:
            result = pd.read_sql(
                "SELECT MAX(date) as last_date FROM etf_daily WHERE code=?",
                conn, params=[code]
            )
            last_date = result.iloc[0]["last_date"]

            if last_date:
                start_date = (datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            else:
                start_date = "2019-01-01"

        conn.close()

        end_date = datetime.now().strftime("%Y-%m-%d")
        df = self.fetch_from_akshare(code, start_date, end_date)

        if len(df) > 0:
            return self.save_to_db(code, df)
        return 0

    def update_all(self, codes: List[str] = None, force: bool = False) -> Dict[str, int]:
        """批量更新ETF数据"""
        if codes is None:
            codes = list(ALL_ETFS.keys())

        results = {}
        for i, code in enumerate(codes):
            logger.info(f"更新ETF数据 [{i+1}/{len(codes)}]: {code}")
            try:
                rows = self.update_data(code, force)
                results[code] = rows
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"更新失败: {code}, 错误: {e}")
                results[code] = -1

        return results

    def get_data_with_indicators(self, code: str, start_date: str = None,
                                  end_date: str = None) -> pd.DataFrame:
        """获取带技术指标的数据"""
        from core.etf_indicators import calculate_etf_indicators

        actual_start = start_date
        if start_date:
            dt = datetime.strptime(start_date, "%Y-%m-%d")
            extended_start = (dt - timedelta(days=120)).strftime("%Y-%m-%d")
        else:
            extended_start = None

        df = self.get_data(code, extended_start, end_date)

        if len(df) == 0:
            return df

        df = calculate_etf_indicators(df)

        if actual_start:
            df = df[df["date"] >= actual_start]

        return df.reset_index(drop=True)

    def get_multiple_data(self, codes: List[str], start_date: str = None,
                          end_date: str = None, with_indicators: bool = True) -> Dict[str, pd.DataFrame]:
        """获取多个ETF的数据"""
        results = {}
        for code in codes:
            if with_indicators:
                df = self.get_data_with_indicators(code, start_date, end_date)
            else:
                df = self.get_data(code, start_date, end_date)
            results[code] = df
        return results

    def get_available_codes(self) -> List[str]:
        """获取数据库中有数据的ETF代码列表"""
        conn = sqlite3.connect(self.db_path)
        result = pd.read_sql("SELECT DISTINCT code FROM etf_daily", conn)
        conn.close()
        return result["code"].tolist()

    def get_data_info(self) -> pd.DataFrame:
        """获取数据库中所有ETF的数据统计信息"""
        conn = sqlite3.connect(self.db_path)
        result = pd.read_sql("""
            SELECT
                code,
                COUNT(*) as rows,
                MIN(date) as start_date,
                MAX(date) as end_date
            FROM etf_daily
            GROUP BY code
            ORDER BY code
        """, conn)
        conn.close()
        return result

    def clear_cache(self):
        """清除内存缓存"""
        self._cache.clear()


# 单例
_etf_service_instance = None

def get_etf_data_service(db_path: str = None) -> ETFDataService:
    """获取ETF数据服务单例"""
    global _etf_service_instance
    if _etf_service_instance is None:
        _etf_service_instance = ETFDataService(db_path)
    return _etf_service_instance
