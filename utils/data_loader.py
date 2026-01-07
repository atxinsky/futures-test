# coding=utf-8
"""
数据加载器
支持从TianQin数据库和本地文件加载数据
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)


class DataLoader:
    """
    数据加载器

    支持:
    1. TianQin数据库
    2. 本地CSV文件
    3. 内存缓存
    """

    # 品种代码映射
    SYMBOL_MAP = {
        'RB': 'SHFE.rb',
        'I': 'DCE.i',
        'AU': 'SHFE.au',
        'CU': 'SHFE.cu',
        'AL': 'SHFE.al',
        'NI': 'SHFE.ni',
        'TA': 'CZCE.TA',
        'MA': 'CZCE.MA',
        'PP': 'DCE.pp',
        'M': 'DCE.m',
        'Y': 'DCE.y',
        'P': 'DCE.p',
        'JM': 'DCE.jm',
        'J': 'DCE.j',
        'AG': 'SHFE.ag',
        'ZN': 'SHFE.zn',
        'PB': 'SHFE.pb',
        'HC': 'SHFE.hc',
        'BU': 'SHFE.bu',
        'FU': 'SHFE.fu',
        'SR': 'CZCE.SR',
        'CF': 'CZCE.CF',
        'OI': 'CZCE.OI',
        'RM': 'CZCE.RM',
        'FG': 'CZCE.FG',
        'SA': 'CZCE.SA',
        'IF': 'CFFEX.IF',
        'IC': 'CFFEX.IC',
        'IH': 'CFFEX.IH',
    }

    # 周期映射
    PERIOD_MAP = {
        '1m': 60,
        '5m': 300,
        '15m': 900,
        '30m': 1800,
        '60m': 3600,
        '1h': 3600,
        '4h': 14400,
        '1d': 86400,
        'D': 86400,
    }

    def __init__(self, data_dir: str = None):
        """
        初始化数据加载器

        Args:
            data_dir: 数据目录
        """
        self.data_dir = data_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data'
        )
        self._cache: Dict[str, pd.DataFrame] = {}
        self._tq_api = None

    def _get_tq_api(self):
        """获取TianQin API"""
        if self._tq_api is None:
            try:
                from tqsdk import TqApi, TqAuth
                # 使用免费行情
                self._tq_api = TqApi(auth=TqAuth("", ""))
            except ImportError:
                logger.warning("TianQin SDK未安装，仅支持本地数据")
            except Exception as e:
                logger.warning(f"TianQin连接失败: {e}")
        return self._tq_api

    def load_bars(
        self,
        symbol: str,
        period: str = '1d',
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = None
    ) -> pd.DataFrame:
        """
        加载K线数据

        Args:
            symbol: 品种代码（如 RB, I）
            period: 周期（1m, 5m, 15m, 30m, 60m, 1d）
            start_date: 开始时间
            end_date: 结束时间
            limit: 数量限制

        Returns:
            DataFrame with columns: time, open, high, low, close, volume, open_interest
        """
        cache_key = f"{symbol}_{period}"

        # 尝试从缓存获取
        if cache_key in self._cache:
            df = self._cache[cache_key]
            return self._filter_date_range(df, start_date, end_date, limit)

        # 尝试从本地文件加载
        df = self._load_from_file(symbol, period)

        if df is None or df.empty:
            # 尝试从TianQin加载
            df = self._load_from_tianqin(symbol, period, start_date, end_date, limit)

        if df is not None and not df.empty:
            # 缓存数据
            self._cache[cache_key] = df
            return self._filter_date_range(df, start_date, end_date, limit)

        logger.warning(f"无法加载数据: {symbol} {period}")
        return pd.DataFrame()

    def _load_from_file(self, symbol: str, period: str) -> Optional[pd.DataFrame]:
        """从本地文件加载"""
        # 尝试多种文件格式
        file_patterns = [
            f"{symbol}_{period}.csv",
            f"{symbol}_{period}.parquet",
            f"{symbol.lower()}_{period}.csv",
            f"{symbol}/{period}.csv",
        ]

        for pattern in file_patterns:
            file_path = os.path.join(self.data_dir, pattern)
            if os.path.exists(file_path):
                try:
                    if file_path.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                    else:
                        df = pd.read_csv(file_path)

                    df = self._standardize_columns(df)
                    logger.info(f"从文件加载数据: {file_path}, {len(df)} 条")
                    return df
                except Exception as e:
                    logger.error(f"加载文件失败: {file_path} - {e}")

        return None

    def _load_from_tianqin(
        self,
        symbol: str,
        period: str,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = None
    ) -> Optional[pd.DataFrame]:
        """从TianQin加载数据"""
        api = self._get_tq_api()
        if api is None:
            return None

        try:
            from tqsdk import tafunc

            # 获取合约代码
            tq_symbol = self._get_tq_symbol(symbol)
            if not tq_symbol:
                logger.warning(f"未知品种: {symbol}")
                return None

            # 获取周期秒数
            duration = self.PERIOD_MAP.get(period, 86400)

            # 计算数量
            if limit is None:
                if start_date and end_date:
                    days = (end_date - start_date).days
                    if duration >= 86400:
                        limit = days + 1
                    else:
                        limit = days * 86400 // duration
                else:
                    limit = 2000  # 默认获取2000根K线

            # 获取K线
            klines = api.get_kline_serial(tq_symbol, duration, limit)

            if klines is not None and len(klines) > 0:
                df = klines.copy()
                df = self._standardize_columns(df)
                return df

        except Exception as e:
            logger.error(f"从TianQin加载数据失败: {symbol} - {e}")

        return None

    def _get_tq_symbol(self, symbol: str) -> Optional[str]:
        """获取TianQin合约代码"""
        # 如果已经是完整代码
        if '.' in symbol:
            return symbol

        # 查找映射
        upper_symbol = symbol.upper()
        if upper_symbol in self.SYMBOL_MAP:
            # 获取主力合约
            exchange_product = self.SYMBOL_MAP[upper_symbol]
            # 主力合约用 @ 表示
            return f"KQ.m@{exchange_product}"

        return None

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        # 列名映射
        column_map = {
            'datetime': 'time',
            'date': 'time',
            'timestamp': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'vol': 'volume',
            'oi': 'open_interest',
            'OpenInterest': 'open_interest',
        }

        df = df.rename(columns=column_map)

        # 确保有time列
        if 'time' not in df.columns:
            if df.index.name in ['time', 'datetime', 'date']:
                df = df.reset_index()
            elif isinstance(df.index, pd.DatetimeIndex):
                df['time'] = df.index
                df = df.reset_index(drop=True)

        # 转换时间列
        if 'time' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['time']):
                df['time'] = pd.to_datetime(df['time'])

        # 确保必要的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"缺少列: {col}")

        return df

    def _filter_date_range(
        self,
        df: pd.DataFrame,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = None
    ) -> pd.DataFrame:
        """过滤日期范围"""
        if df.empty:
            return df

        result = df.copy()

        if 'time' in result.columns:
            if start_date:
                result = result[result['time'] >= start_date]
            if end_date:
                result = result[result['time'] <= end_date]

        if limit:
            result = result.tail(limit)

        return result.reset_index(drop=True)

    def get_available_symbols(self) -> List[str]:
        """获取可用品种列表"""
        symbols = set()

        # 从文件系统检测
        if os.path.exists(self.data_dir):
            for f in os.listdir(self.data_dir):
                if f.endswith(('.csv', '.parquet')):
                    symbol = f.split('_')[0].upper()
                    symbols.add(symbol)

        # 添加预定义品种
        symbols.update(self.SYMBOL_MAP.keys())

        return sorted(list(symbols))

    def get_main_contract(self, product: str, date: datetime = None) -> str:
        """
        获取主力合约代码

        Args:
            product: 品种代码（如 RB, I）
            date: 日期

        Returns:
            主力合约代码（如 RB2505）
        """
        if date is None:
            date = datetime.now()

        # 简化实现：根据月份推算主力合约
        # 实际应该查询数据库或API
        month = date.month
        year = date.year % 100

        # 主力合约通常在1, 5, 9月（部分品种不同）
        main_months = [1, 5, 9]  # 默认主力月份

        # 找下一个主力月份
        for m in main_months:
            if m > month:
                return f"{product}{year:02d}{m:02d}"

        # 跨年
        return f"{product}{(year + 1) % 100:02d}{main_months[0]:02d}"

    def clear_cache(self, symbol: str = None):
        """清理缓存"""
        if symbol:
            keys_to_remove = [k for k in self._cache if k.startswith(symbol)]
            for k in keys_to_remove:
                del self._cache[k]
        else:
            self._cache.clear()

    def close(self):
        """关闭连接"""
        if self._tq_api:
            try:
                self._tq_api.close()
            except:
                pass
            self._tq_api = None


# 全局实例
_data_loader = None


def get_data_loader(data_dir: str = None) -> DataLoader:
    """获取数据加载器单例"""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader(data_dir)
    return _data_loader
