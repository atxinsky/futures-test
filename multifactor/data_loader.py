# coding=utf-8
"""
多因子选股 - 数据加载模块
使用AKShare获取A股数据
"""

import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta
import sqlite3
import os
import time
import logging

logger = logging.getLogger(__name__)

# 数据库路径
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DB_PATH = os.path.join(DATA_DIR, "stock_data.db")


class StockDataLoader:
    """A股数据加载器"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
        self._cache = {}

    def _init_db(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 股票日线数据表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_daily (
                code TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                amount REAL,
                turn REAL,
                pct_chg REAL,
                PRIMARY KEY (code, date)
            )
        """)

        # 股票基本信息表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_info (
                code TEXT PRIMARY KEY,
                name TEXT,
                industry TEXT,
                list_date TEXT,
                total_mv REAL,
                circ_mv REAL,
                pe REAL,
                pb REAL
            )
        """)

        conn.commit()
        conn.close()
        logger.info(f"数据库初始化完成: {self.db_path}")

    def get_stock_list(self, market: str = "all") -> pd.DataFrame:
        """
        获取股票列表

        Args:
            market: "sh" / "sz" / "all"
        """
        try:
            df = ak.stock_zh_a_spot_em()
            df = df.rename(columns={
                "代码": "code",
                "名称": "name",
                "最新价": "close",
                "总市值": "total_mv",
                "流通市值": "circ_mv",
                "市盈率-动态": "pe",
                "市净率": "pb",
                "换手率": "turnover"
            })

            # 过滤ST和退市股
            df = df[~df["name"].str.contains("ST|退", na=False)]

            # 过滤市场
            if market == "sh":
                df = df[df["code"].str.startswith("6")]
            elif market == "sz":
                df = df[df["code"].str.startswith(("0", "3"))]

            # 过滤科创板和北交所（流动性差）
            df = df[~df["code"].str.startswith(("68", "4", "8"))]

            logger.info(f"获取股票列表: {len(df)}只")
            return df[["code", "name", "close", "total_mv", "circ_mv", "pe", "pb", "turnover"]]
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()

    def fetch_stock_daily(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从AKShare获取单只股票日线数据
        """
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq"  # 前复权
            )

            if df is None or len(df) == 0:
                return pd.DataFrame()

            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turn",
                "涨跌幅": "pct_chg"
            })

            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df["code"] = code

            return df[["code", "date", "open", "high", "low", "close", "volume", "amount", "turn", "pct_chg"]]
        except Exception as e:
            logger.warning(f"获取{code}数据失败: {e}")
            return pd.DataFrame()

    def save_to_db(self, df: pd.DataFrame) -> int:
        """保存数据到数据库"""
        if df is None or len(df) == 0:
            return 0

        conn = sqlite3.connect(self.db_path)

        # 使用INSERT OR REPLACE
        df.to_sql("stock_daily", conn, if_exists="append", index=False,
                  method=lambda table, conn, keys, data_iter:
                  conn.executemany(
                      f"INSERT OR REPLACE INTO {table.name} ({','.join(keys)}) VALUES ({','.join(['?']*len(keys))})",
                      list(data_iter)
                  ))

        conn.commit()
        conn.close()
        return len(df)

    def get_stock_data(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取股票数据（优先从数据库读取）"""
        conn = sqlite3.connect(self.db_path)

        df = pd.read_sql("""
            SELECT * FROM stock_daily
            WHERE code = ? AND date >= ? AND date <= ?
            ORDER BY date
        """, conn, params=[code, start_date, end_date])

        conn.close()
        return df

    def update_stock_data(self, codes: list, start_date: str, end_date: str,
                          show_progress: bool = True) -> dict:
        """批量更新股票数据"""
        results = {}
        total = len(codes)

        for i, code in enumerate(codes):
            if show_progress and (i + 1) % 50 == 0:
                print(f"  进度: {i+1}/{total}")

            try:
                df = self.fetch_stock_daily(code, start_date, end_date)
                if len(df) > 0:
                    rows = self.save_to_db(df)
                    results[code] = rows
                else:
                    results[code] = 0

                time.sleep(0.1)  # 避免请求过快
            except Exception as e:
                logger.warning(f"更新{code}失败: {e}")
                results[code] = -1

        return results

    def get_all_stock_data(self, codes: list, start_date: str, end_date: str) -> dict:
        """获取多只股票的数据"""
        data = {}
        for code in codes:
            df = self.get_stock_data(code, start_date, end_date)
            if len(df) > 0:
                data[code] = df
        return data

    def get_index_data(self, index_code: str = "000300", start_date: str = None,
                       end_date: str = None) -> pd.DataFrame:
        """获取指数数据（沪深300等）"""
        try:
            df = ak.stock_zh_index_daily(symbol=f"sh{index_code}")
            df = df.rename(columns={
                "date": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume"
            })
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

            if start_date:
                df = df[df["date"] >= start_date]
            if end_date:
                df = df[df["date"] <= end_date]

            return df
        except Exception as e:
            logger.error(f"获取指数数据失败: {e}")
            return pd.DataFrame()


def get_hs300_components() -> list:
    """获取沪深300成分股"""
    try:
        df = ak.index_stock_cons(symbol="000300")
        codes = df["品种代码"].tolist()
        logger.info(f"获取沪深300成分股: {len(codes)}只")
        return codes
    except Exception as e:
        logger.error(f"获取沪深300成分股失败: {e}")
        return []


def get_zz500_components() -> list:
    """获取中证500成分股"""
    try:
        df = ak.index_stock_cons(symbol="000905")
        codes = df["品种代码"].tolist()
        logger.info(f"获取中证500成分股: {len(codes)}只")
        return codes
    except Exception as e:
        logger.error(f"获取中证500成分股失败: {e}")
        return []


def get_zz1000_components() -> list:
    """获取中证1000成分股"""
    try:
        df = ak.index_stock_cons(symbol="000852")
        codes = df["品种代码"].tolist()
        logger.info(f"获取中证1000成分股: {len(codes)}只")
        return codes
    except Exception as e:
        logger.error(f"获取中证1000成分股失败: {e}")
        return []


def get_index_components(index_name: str = "zz1000") -> list:
    """
    获取指数成分股

    Args:
        index_name: "hs300" / "zz500" / "zz1000"
    """
    if index_name == "hs300":
        return get_hs300_components()
    elif index_name == "zz500":
        return get_zz500_components()
    elif index_name == "zz1000":
        return get_zz1000_components()
    else:
        logger.warning(f"未知指数: {index_name}")
        return []
