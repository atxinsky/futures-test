# coding=utf-8
"""
统一数据服务层
整合多个数据源，提供统一的数据访问接口
"""

import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from threading import Lock
from collections import OrderedDict
from dataclasses import dataclass, field
import time

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    data: pd.DataFrame
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    size_bytes: int = 0

    def update_access(self):
        """更新访问时间"""
        self.last_access = time.time()
        self.access_count += 1


class LRUDataCache:
    """
    LRU数据缓存

    特性：
    1. LRU淘汰策略
    2. 可配置的最大容量和条目数
    3. 过期时间支持
    4. 缓存统计
    """

    def __init__(
        self,
        max_entries: int = 100,
        max_size_mb: float = 500,
        expire_seconds: int = 3600
    ):
        """
        初始化缓存

        Args:
            max_entries: 最大缓存条目数
            max_size_mb: 最大缓存大小(MB)
            expire_seconds: 过期时间(秒)，0表示不过期
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = Lock()
        self.max_entries = max_entries
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.expire_seconds = expire_seconds

        # 统计
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[pd.DataFrame]:
        """获取缓存数据"""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # 检查过期
            if self.expire_seconds > 0:
                if time.time() - entry.created_at > self.expire_seconds:
                    del self._cache[key]
                    self._misses += 1
                    return None

            # 更新访问信息并移到末尾(LRU)
            entry.update_access()
            self._cache.move_to_end(key)
            self._hits += 1

            return entry.data.copy()

    def put(self, key: str, data: pd.DataFrame):
        """存入缓存"""
        if data.empty:
            return

        size_bytes = data.memory_usage(deep=True).sum()

        with self._lock:
            # 如果key已存在，先删除
            if key in self._cache:
                del self._cache[key]

            # 检查容量，淘汰旧数据
            while len(self._cache) >= self.max_entries:
                self._evict_oldest()

            # 检查内存大小
            current_size = sum(e.size_bytes for e in self._cache.values())
            while current_size + size_bytes > self.max_size_bytes and self._cache:
                self._evict_oldest()
                current_size = sum(e.size_bytes for e in self._cache.values())

            # 存入新数据
            self._cache[key] = CacheEntry(
                data=data.copy(),
                size_bytes=size_bytes
            )

    def _evict_oldest(self):
        """淘汰最老的条目"""
        if self._cache:
            self._cache.popitem(last=False)
            self._evictions += 1

    def clear(self, prefix: str = None):
        """清理缓存"""
        with self._lock:
            if prefix:
                keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
                for k in keys_to_remove:
                    del self._cache[k]
            else:
                self._cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0

            current_size = sum(e.size_bytes for e in self._cache.values())

            return {
                'entries': len(self._cache),
                'size_mb': current_size / 1024 / 1024,
                'max_entries': self.max_entries,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'evictions': self._evictions
            }

    def get_keys(self) -> List[str]:
        """获取所有缓存键"""
        with self._lock:
            return list(self._cache.keys())


class DataSource(Enum):
    """数据源类型"""
    AUTO = "auto"           # 自动选择
    LOCAL_DB = "local_db"   # 本地SQLite数据库
    TIANQIN = "tianqin"     # 天勤API
    AKSHARE = "akshare"     # AKShare
    TUSHARE = "tushare"     # Tushare


class Period(Enum):
    """K线周期"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    M60 = "60m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @classmethod
    def from_string(cls, s: str) -> 'Period':
        """从字符串转换"""
        mapping = {
            '1m': cls.M1, '1分钟': cls.M1,
            '5m': cls.M5, '5分钟': cls.M5, '5': cls.M5,
            '15m': cls.M15, '15分钟': cls.M15, '15': cls.M15,
            '30m': cls.M30, '30分钟': cls.M30, '30': cls.M30,
            '60m': cls.M60, '60分钟': cls.M60, '60': cls.M60,
            '1h': cls.H1, '小时': cls.H1,
            '4h': cls.H4, '4小时': cls.H4,
            '1d': cls.D1, 'd': cls.D1, 'D': cls.D1, '日线': cls.D1, '日': cls.D1,
        }
        return mapping.get(s, cls.D1)

    def to_seconds(self) -> int:
        """转换为秒数"""
        seconds_map = {
            Period.M1: 60,
            Period.M5: 300,
            Period.M15: 900,
            Period.M30: 1800,
            Period.M60: 3600,
            Period.H1: 3600,
            Period.H4: 14400,
            Period.D1: 86400,
        }
        return seconds_map.get(self, 86400)

    def to_akshare_period(self) -> str:
        """转换为akshare周期格式"""
        ak_map = {
            Period.M5: '5',
            Period.M15: '15',
            Period.M30: '30',
            Period.M60: '60',
            Period.H1: '60',
        }
        return ak_map.get(self, '60')


# ============== 统一品种配置 ==============
# 合并自 data_manager.py 和 utils/data_loader.py

FUTURES_CONFIG: Dict[str, Dict[str, Any]] = {
    # 股指期货 - 中金所
    "IF": {"name": "沪深300股指", "exchange": "CFFEX", "list_date": "2010-04-16", "multiplier": 300, "margin_rate": 0.12},
    "IH": {"name": "上证50股指", "exchange": "CFFEX", "list_date": "2015-04-16", "multiplier": 300, "margin_rate": 0.12},
    "IC": {"name": "中证500股指", "exchange": "CFFEX", "list_date": "2015-04-16", "multiplier": 200, "margin_rate": 0.12},
    "IM": {"name": "中证1000股指", "exchange": "CFFEX", "list_date": "2022-07-22", "multiplier": 200, "margin_rate": 0.12},

    # 国债期货
    "T": {"name": "10年期国债", "exchange": "CFFEX", "list_date": "2015-03-20", "multiplier": 10000, "margin_rate": 0.03},
    "TF": {"name": "5年期国债", "exchange": "CFFEX", "list_date": "2013-09-06", "multiplier": 10000, "margin_rate": 0.02},
    "TS": {"name": "2年期国债", "exchange": "CFFEX", "list_date": "2018-08-17", "multiplier": 20000, "margin_rate": 0.01},
    "TL": {"name": "30年期国债", "exchange": "CFFEX", "list_date": "2023-04-21", "multiplier": 10000, "margin_rate": 0.035},

    # 贵金属 - 上期所
    "AU": {"name": "黄金", "exchange": "SHFE", "list_date": "2008-01-09", "multiplier": 1000, "margin_rate": 0.08},
    "AG": {"name": "白银", "exchange": "SHFE", "list_date": "2012-05-10", "multiplier": 15, "margin_rate": 0.09},

    # 有色金属 - 上期所
    "CU": {"name": "沪铜", "exchange": "SHFE", "list_date": "1995-01-01", "multiplier": 5, "margin_rate": 0.10},
    "AL": {"name": "沪铝", "exchange": "SHFE", "list_date": "1995-01-01", "multiplier": 5, "margin_rate": 0.10},
    "ZN": {"name": "沪锌", "exchange": "SHFE", "list_date": "2007-03-26", "multiplier": 5, "margin_rate": 0.10},
    "PB": {"name": "沪铅", "exchange": "SHFE", "list_date": "2011-03-24", "multiplier": 5, "margin_rate": 0.10},
    "NI": {"name": "沪镍", "exchange": "SHFE", "list_date": "2015-03-27", "multiplier": 1, "margin_rate": 0.12},
    "SN": {"name": "沪锡", "exchange": "SHFE", "list_date": "2015-03-27", "multiplier": 1, "margin_rate": 0.12},
    "SS": {"name": "不锈钢", "exchange": "SHFE", "list_date": "2019-09-25", "multiplier": 5, "margin_rate": 0.10},
    "AO": {"name": "氧化铝", "exchange": "SHFE", "list_date": "2023-06-19", "multiplier": 20, "margin_rate": 0.10},
    "BC": {"name": "国际铜", "exchange": "INE", "list_date": "2020-11-19", "multiplier": 5, "margin_rate": 0.10},

    # 黑色系
    "RB": {"name": "螺纹钢", "exchange": "SHFE", "list_date": "2009-03-27", "multiplier": 10, "margin_rate": 0.10},
    "HC": {"name": "热卷", "exchange": "SHFE", "list_date": "2014-03-21", "multiplier": 10, "margin_rate": 0.10},
    "WR": {"name": "线材", "exchange": "SHFE", "list_date": "2009-03-27", "multiplier": 10, "margin_rate": 0.10},
    "I": {"name": "铁矿石", "exchange": "DCE", "list_date": "2013-10-18", "multiplier": 100, "margin_rate": 0.12},
    "J": {"name": "焦炭", "exchange": "DCE", "list_date": "2011-04-15", "multiplier": 100, "margin_rate": 0.12},
    "JM": {"name": "焦煤", "exchange": "DCE", "list_date": "2013-03-22", "multiplier": 60, "margin_rate": 0.12},
    "SF": {"name": "硅铁", "exchange": "CZCE", "list_date": "2014-08-08", "multiplier": 5, "margin_rate": 0.10},
    "SM": {"name": "锰硅", "exchange": "CZCE", "list_date": "2014-08-08", "multiplier": 5, "margin_rate": 0.10},

    # 能源
    "SC": {"name": "原油", "exchange": "INE", "list_date": "2018-03-26", "multiplier": 1000, "margin_rate": 0.10},
    "FU": {"name": "燃油", "exchange": "SHFE", "list_date": "2004-08-25", "multiplier": 10, "margin_rate": 0.10},
    "LU": {"name": "低硫燃油", "exchange": "INE", "list_date": "2020-06-22", "multiplier": 10, "margin_rate": 0.10},
    "BU": {"name": "沥青", "exchange": "SHFE", "list_date": "2013-10-09", "multiplier": 10, "margin_rate": 0.10},

    # 化工
    "L": {"name": "塑料LLDPE", "exchange": "DCE", "list_date": "2007-07-31", "multiplier": 5, "margin_rate": 0.10},
    "V": {"name": "PVC", "exchange": "DCE", "list_date": "2009-05-25", "multiplier": 5, "margin_rate": 0.10},
    "PP": {"name": "聚丙烯", "exchange": "DCE", "list_date": "2014-02-28", "multiplier": 5, "margin_rate": 0.10},
    "EG": {"name": "乙二醇", "exchange": "DCE", "list_date": "2018-12-10", "multiplier": 10, "margin_rate": 0.10},
    "EB": {"name": "苯乙烯", "exchange": "DCE", "list_date": "2019-09-26", "multiplier": 5, "margin_rate": 0.10},
    "PG": {"name": "LPG", "exchange": "DCE", "list_date": "2020-03-30", "multiplier": 20, "margin_rate": 0.10},
    "TA": {"name": "PTA", "exchange": "CZCE", "list_date": "2006-12-18", "multiplier": 5, "margin_rate": 0.10},
    "MA": {"name": "甲醇", "exchange": "CZCE", "list_date": "2011-10-28", "multiplier": 10, "margin_rate": 0.10},
    "SA": {"name": "纯碱", "exchange": "CZCE", "list_date": "2019-12-06", "multiplier": 20, "margin_rate": 0.10},
    "FG": {"name": "玻璃", "exchange": "CZCE", "list_date": "2012-12-03", "multiplier": 20, "margin_rate": 0.10},
    "UR": {"name": "尿素", "exchange": "CZCE", "list_date": "2019-08-09", "multiplier": 20, "margin_rate": 0.10},
    "PF": {"name": "短纤", "exchange": "CZCE", "list_date": "2020-10-12", "multiplier": 5, "margin_rate": 0.10},
    "PX": {"name": "对二甲苯", "exchange": "CZCE", "list_date": "2023-09-15", "multiplier": 5, "margin_rate": 0.10},
    "SH": {"name": "烧碱", "exchange": "CZCE", "list_date": "2023-09-15", "multiplier": 30, "margin_rate": 0.10},

    # 橡胶
    "RU": {"name": "天然橡胶", "exchange": "SHFE", "list_date": "1999-01-01", "multiplier": 10, "margin_rate": 0.10},
    "NR": {"name": "20号胶", "exchange": "INE", "list_date": "2019-08-12", "multiplier": 10, "margin_rate": 0.10},
    "SP": {"name": "纸浆", "exchange": "SHFE", "list_date": "2018-11-27", "multiplier": 10, "margin_rate": 0.10},
    "BR": {"name": "丁二烯橡胶", "exchange": "SHFE", "list_date": "2023-07-28", "multiplier": 5, "margin_rate": 0.10},

    # 油脂油料
    "M": {"name": "豆粕", "exchange": "DCE", "list_date": "2000-07-17", "multiplier": 10, "margin_rate": 0.08},
    "Y": {"name": "豆油", "exchange": "DCE", "list_date": "2006-01-09", "multiplier": 10, "margin_rate": 0.08},
    "P": {"name": "棕榈油", "exchange": "DCE", "list_date": "2007-10-29", "multiplier": 10, "margin_rate": 0.08},
    "A": {"name": "豆一", "exchange": "DCE", "list_date": "2002-03-15", "multiplier": 10, "margin_rate": 0.08},
    "B": {"name": "豆二", "exchange": "DCE", "list_date": "2004-12-22", "multiplier": 10, "margin_rate": 0.08},
    "OI": {"name": "菜油", "exchange": "CZCE", "list_date": "2007-06-08", "multiplier": 10, "margin_rate": 0.08},
    "RM": {"name": "菜粕", "exchange": "CZCE", "list_date": "2012-12-28", "multiplier": 10, "margin_rate": 0.08},

    # 农产品
    "C": {"name": "玉米", "exchange": "DCE", "list_date": "2004-09-22", "multiplier": 10, "margin_rate": 0.08},
    "CS": {"name": "玉米淀粉", "exchange": "DCE", "list_date": "2014-12-19", "multiplier": 10, "margin_rate": 0.08},
    "JD": {"name": "鸡蛋", "exchange": "DCE", "list_date": "2013-11-08", "multiplier": 10, "margin_rate": 0.08},
    "RR": {"name": "粳米", "exchange": "DCE", "list_date": "2019-08-16", "multiplier": 10, "margin_rate": 0.08},
    "CF": {"name": "棉花", "exchange": "CZCE", "list_date": "2004-06-01", "multiplier": 5, "margin_rate": 0.08},
    "SR": {"name": "白糖", "exchange": "CZCE", "list_date": "2006-01-06", "multiplier": 10, "margin_rate": 0.08},
    "AP": {"name": "苹果", "exchange": "CZCE", "list_date": "2017-12-22", "multiplier": 10, "margin_rate": 0.08},
    "CJ": {"name": "红枣", "exchange": "CZCE", "list_date": "2019-04-30", "multiplier": 5, "margin_rate": 0.08},
    "PK": {"name": "花生", "exchange": "CZCE", "list_date": "2021-02-01", "multiplier": 5, "margin_rate": 0.08},
    "CY": {"name": "棉纱", "exchange": "CZCE", "list_date": "2017-08-18", "multiplier": 5, "margin_rate": 0.08},

    # 生猪
    "LH": {"name": "生猪", "exchange": "DCE", "list_date": "2021-01-08", "multiplier": 16, "margin_rate": 0.10},

    # 新能源 - 广期所
    "SI": {"name": "工业硅", "exchange": "GFEX", "list_date": "2022-12-22", "multiplier": 5, "margin_rate": 0.10},
    "LC": {"name": "碳酸锂", "exchange": "GFEX", "list_date": "2023-07-21", "multiplier": 1, "margin_rate": 0.10},

    # 航运
    "EC": {"name": "集运指数", "exchange": "INE", "list_date": "2023-08-18", "multiplier": 50, "margin_rate": 0.10},
}


def get_futures_config(symbol: str) -> Optional[Dict[str, Any]]:
    """获取品种配置"""
    # 提取品种代码（去掉合约月份）
    product = ''.join([c for c in symbol if c.isalpha()]).upper()
    return FUTURES_CONFIG.get(product)


def get_tq_symbol(symbol: str, main_contract: bool = True) -> str:
    """
    转换为天勤合约代码

    Args:
        symbol: 品种代码 (如 RB, RB2505)
        main_contract: 是否使用主力合约

    Returns:
        天勤格式合约代码 (如 SHFE.rb2505, KQ.m@SHFE.rb)
    """
    product = ''.join([c for c in symbol if c.isalpha()]).upper()
    month = ''.join([c for c in symbol if c.isdigit()])

    config = FUTURES_CONFIG.get(product)
    if not config:
        logger.warning(f"未知品种: {symbol}")
        return symbol

    exchange = config['exchange']

    if month:
        # 具体合约
        return f"{exchange}.{product.lower()}{month}"
    elif main_contract:
        # 主力合约
        return f"KQ.m@{exchange}.{product.lower()}"
    else:
        return f"{exchange}.{product.lower()}"


def get_sina_symbol(symbol: str) -> str:
    """转换为新浪格式（主力连续）"""
    product = ''.join([c for c in symbol if c.isalpha()]).upper()
    return f"{product}0"


class DataService:
    """
    统一数据服务

    整合多个数据源，提供统一接口：
    1. 本地SQLite数据库 (日线/分钟)
    2. 天勤API (实时/历史)
    3. AKShare (历史数据下载)
    """

    def __init__(
        self,
        data_dir: str = None,
        cache_max_entries: int = 100,
        cache_max_size_mb: float = 500,
        cache_expire_seconds: int = 3600
    ):
        """
        初始化

        Args:
            data_dir: 数据目录，默认为项目data目录
            cache_max_entries: 最大缓存条目数
            cache_max_size_mb: 最大缓存大小(MB)
            cache_expire_seconds: 缓存过期时间(秒)，0表示不过期
        """
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data'
            )

        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

        # 使用LRU缓存替代简单字典
        self._cache = LRUDataCache(
            max_entries=cache_max_entries,
            max_size_mb=cache_max_size_mb,
            expire_seconds=cache_expire_seconds
        )

        # 延迟加载的数据源
        self._tq_api = None
        self._db_initialized = False

    def load_bars(
        self,
        symbol: str,
        period: str = "1d",
        start_date: str = None,
        end_date: str = None,
        source: DataSource = DataSource.AUTO,
        limit: int = None
    ) -> pd.DataFrame:
        """
        加载K线数据（统一入口）

        Args:
            symbol: 品种代码 (RB, AU, IF等)
            period: K线周期 ('1m', '5m', '15m', '30m', '60m', '1h', '4h', '1d')
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            source: 数据源
            limit: 数量限制

        Returns:
            DataFrame with columns: time, open, high, low, close, volume, open_interest
        """
        period_enum = Period.from_string(period)
        product = ''.join([c for c in symbol if c.isalpha()]).upper()

        # 检查缓存
        cache_key = f"{product}_{period_enum.value}_{start_date}_{end_date}"
        cached_df = self._cache.get(cache_key)
        if cached_df is not None:
            if limit:
                return cached_df.tail(limit).copy()
            return cached_df

        df = pd.DataFrame()

        if source == DataSource.AUTO:
            # 自动选择数据源
            # 1. 尝试天勤
            df = self._load_from_tianqin(product, period_enum, start_date, end_date, limit)

            # 2. 尝试本地数据库
            if df.empty:
                df = self._load_from_local_db(product, period_enum, start_date, end_date)

            # 3. 尝试AKShare下载
            if df.empty:
                df = self._download_from_akshare(product, period_enum, start_date, end_date)

        elif source == DataSource.TIANQIN:
            df = self._load_from_tianqin(product, period_enum, start_date, end_date, limit)
        elif source == DataSource.LOCAL_DB:
            df = self._load_from_local_db(product, period_enum, start_date, end_date)
        elif source == DataSource.AKSHARE:
            df = self._download_from_akshare(product, period_enum, start_date, end_date)

        if not df.empty:
            # 缓存
            self._cache.put(cache_key, df)

            if limit:
                return df.tail(limit).copy()

        return df

    def _load_from_tianqin(
        self,
        symbol: str,
        period: Period,
        start_date: str = None,
        end_date: str = None,
        limit: int = None
    ) -> pd.DataFrame:
        """从天勤加载数据"""
        try:
            from tqsdk import TqApi, TqAuth

            if self._tq_api is None:
                # 尝试创建API连接
                try:
                    self._tq_api = TqApi(auth=TqAuth("", ""))
                except:
                    return pd.DataFrame()

            tq_symbol = get_tq_symbol(symbol, main_contract=True)
            duration = period.to_seconds()

            # 计算需要的K线数量
            if limit is None:
                if start_date and end_date:
                    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    days = (end_dt - start_dt).days + 1
                    if duration >= 86400:
                        limit = days
                    else:
                        limit = days * 86400 // duration
                else:
                    limit = 2000

            klines = self._tq_api.get_kline_serial(tq_symbol, duration, limit)
            self._tq_api.wait_update()

            if klines is not None and len(klines) > 0:
                df = pd.DataFrame({
                    'time': pd.to_datetime(klines['datetime'], unit='ns'),
                    'open': klines['open'],
                    'high': klines['high'],
                    'low': klines['low'],
                    'close': klines['close'],
                    'volume': klines['volume'],
                    'open_interest': klines.get('open_oi', 0)
                })

                df = df.dropna()

                # 过滤日期范围
                if start_date:
                    df = df[df['time'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['time'] <= pd.to_datetime(end_date + ' 23:59:59')]

                return df.reset_index(drop=True)

        except ImportError:
            logger.debug("TqSdk未安装")
        except Exception as e:
            logger.debug(f"天勤加载失败: {e}")

        return pd.DataFrame()

    def _load_from_local_db(
        self,
        symbol: str,
        period: Period,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """从本地SQLite数据库加载"""
        try:
            import sqlite3

            db_path = os.path.join(self.data_dir, "futures_data.db")
            if not os.path.exists(db_path):
                return pd.DataFrame()

            conn = sqlite3.connect(db_path)

            if period == Period.D1:
                # 日线数据
                query = '''
                    SELECT time, open, high, low, close, volume, open_interest
                    FROM futures_daily WHERE symbol = ?
                '''
                params = [symbol.upper()]
            else:
                # 分钟数据
                ak_period = period.to_akshare_period()
                query = '''
                    SELECT time, open, high, low, close, volume, open_interest
                    FROM futures_minute WHERE symbol = ? AND period = ?
                '''
                params = [symbol.upper(), ak_period]

            if start_date:
                query += ' AND time >= ?'
                params.append(start_date)
            if end_date:
                query += ' AND time <= ?'
                params.append(end_date + ' 23:59:59' if period != Period.D1 else end_date)

            query += ' ORDER BY time'

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            if not df.empty:
                df['time'] = pd.to_datetime(df['time'])

            return df

        except Exception as e:
            logger.debug(f"本地数据库加载失败: {e}")
            return pd.DataFrame()

    def _download_from_akshare(
        self,
        symbol: str,
        period: Period,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """从AKShare下载数据"""
        try:
            import akshare as ak

            sina_symbol = get_sina_symbol(symbol)

            if period == Period.D1:
                # 日线数据
                df = ak.futures_zh_daily_sina(symbol=sina_symbol)

                if df is not None and len(df) > 0:
                    df = df.rename(columns={
                        'date': 'time',
                        'hold': 'open_interest'
                    })
            else:
                # 分钟数据
                ak_period = period.to_akshare_period()
                df = ak.futures_zh_minute_sina(symbol=sina_symbol, period=ak_period)

                if df is not None and len(df) > 0:
                    df = df.rename(columns={
                        'datetime': 'time',
                        'hold': 'open_interest'
                    })

            if df is None or df.empty:
                return pd.DataFrame()

            df['time'] = pd.to_datetime(df['time'])
            df = df.sort_values('time').reset_index(drop=True)

            # 过滤日期范围
            if start_date:
                df = df[df['time'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['time'] <= pd.to_datetime(end_date + ' 23:59:59')]

            # 保存到数据库
            self._save_to_local_db(symbol, period, df)

            return df

        except ImportError:
            logger.debug("AKShare未安装")
        except Exception as e:
            logger.debug(f"AKShare下载失败: {e}")

        return pd.DataFrame()

    def _save_to_local_db(self, symbol: str, period: Period, df: pd.DataFrame):
        """保存数据到本地数据库"""
        if df.empty:
            return

        try:
            import sqlite3

            db_path = os.path.join(self.data_dir, "futures_data.db")
            conn = sqlite3.connect(db_path)

            config = get_futures_config(symbol)
            exchange = config['exchange'] if config else 'UNKNOWN'

            df_save = df.copy()
            df_save['symbol'] = symbol.upper()
            df_save['exchange'] = exchange

            if period == Period.D1:
                # 日线数据
                df_save['time'] = pd.to_datetime(df_save['time']).dt.strftime('%Y-%m-%d')

                for _, row in df_save.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO futures_daily
                        (symbol, exchange, time, open, high, low, close, volume, open_interest)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (row['symbol'], row['exchange'], row['time'],
                          row['open'], row['high'], row['low'], row['close'],
                          row['volume'], row.get('open_interest', 0)))
            else:
                # 分钟数据
                df_save['period'] = period.to_akshare_period()
                df_save['time'] = pd.to_datetime(df_save['time']).dt.strftime('%Y-%m-%d %H:%M:%S')

                for _, row in df_save.iterrows():
                    conn.execute('''
                        INSERT OR REPLACE INTO futures_minute
                        (symbol, exchange, period, time, open, high, low, close, volume, open_interest)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (row['symbol'], row['exchange'], row['period'], row['time'],
                          row['open'], row['high'], row['low'], row['close'],
                          row['volume'], row.get('open_interest', 0)))

            conn.commit()
            conn.close()
            logger.debug(f"数据已保存: {symbol} {period.value} {len(df)}条")

        except Exception as e:
            logger.debug(f"保存数据失败: {e}")

    def get_available_symbols(self) -> List[str]:
        """获取所有可用品种"""
        return sorted(list(FUTURES_CONFIG.keys()))

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取品种信息"""
        return get_futures_config(symbol)

    def get_symbols_by_category(self) -> Dict[str, List[Tuple[str, str]]]:
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
            result[cat] = [
                (s, FUTURES_CONFIG[s]['name'])
                for s in symbols if s in FUTURES_CONFIG
            ]

        return result

    def clear_cache(self, symbol: str = None):
        """清理缓存"""
        if symbol:
            self._cache.clear(prefix=symbol.upper())
        else:
            self._cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self._cache.get_stats()

    def close(self):
        """关闭连接"""
        if self._tq_api:
            try:
                self._tq_api.close()
            except:
                pass
            self._tq_api = None


# 全局单例
_data_service: Optional[DataService] = None


def get_data_service(data_dir: str = None) -> DataService:
    """获取数据服务单例"""
    global _data_service
    if _data_service is None:
        _data_service = DataService(data_dir)
    return _data_service


# 便捷函数
def load_bars(
    symbol: str,
    period: str = "1d",
    start_date: str = None,
    end_date: str = None,
    source: str = "auto"
) -> pd.DataFrame:
    """
    加载K线数据（便捷函数）

    Args:
        symbol: 品种代码
        period: K线周期
        start_date: 开始日期
        end_date: 结束日期
        source: 数据源 ('auto', 'local_db', 'tianqin', 'akshare')

    Returns:
        DataFrame
    """
    source_map = {
        'auto': DataSource.AUTO,
        'local_db': DataSource.LOCAL_DB,
        'tianqin': DataSource.TIANQIN,
        'akshare': DataSource.AKSHARE,
    }
    ds = source_map.get(source, DataSource.AUTO)
    return get_data_service().load_bars(symbol, period, start_date, end_date, ds)
