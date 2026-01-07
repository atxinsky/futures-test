# coding=utf-8
"""
天勤实时数据流
提供实时K线数据的订阅和推送
"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Callable, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class TqDataFeed:
    """
    天勤实时数据流

    功能:
    1. 订阅多品种实时K线
    2. 自动生成周期K线
    3. 历史数据加载
    4. 数据回调分发
    """

    def __init__(self, tq_user: str = "", tq_password: str = ""):
        """
        初始化

        Args:
            tq_user: 天勤用户名
            tq_password: 天勤密码
        """
        self.tq_user = tq_user
        self.tq_password = tq_password

        self.api = None
        self.connected = False

        # 订阅管理
        self.quotes: Dict[str, any] = {}
        self.klines: Dict[str, Dict[str, any]] = {}

        # 回调
        self._on_bar_callback: Optional[Callable] = None
        self._on_tick_callback: Optional[Callable] = None

        # 线程控制
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # 品种交易所映射
        self.exchange_map = {
            'IF': 'CFFEX', 'IH': 'CFFEX', 'IC': 'CFFEX', 'IM': 'CFFEX',
            'T': 'CFFEX', 'TF': 'CFFEX', 'TS': 'CFFEX', 'TL': 'CFFEX',
            'AU': 'SHFE', 'AG': 'SHFE', 'CU': 'SHFE', 'AL': 'SHFE',
            'ZN': 'SHFE', 'PB': 'SHFE', 'NI': 'SHFE', 'SN': 'SHFE',
            'SS': 'SHFE', 'AO': 'SHFE', 'RB': 'SHFE', 'HC': 'SHFE',
            'WR': 'SHFE', 'FU': 'SHFE', 'BU': 'SHFE', 'RU': 'SHFE',
            'SP': 'SHFE',
            'I': 'DCE', 'J': 'DCE', 'JM': 'DCE', 'L': 'DCE', 'V': 'DCE',
            'PP': 'DCE', 'EG': 'DCE', 'EB': 'DCE', 'PG': 'DCE', 'M': 'DCE',
            'Y': 'DCE', 'P': 'DCE', 'A': 'DCE', 'B': 'DCE', 'C': 'DCE',
            'CS': 'DCE', 'JD': 'DCE', 'LH': 'DCE',
            'SF': 'CZCE', 'SM': 'CZCE', 'TA': 'CZCE', 'MA': 'CZCE',
            'SA': 'CZCE', 'FG': 'CZCE', 'UR': 'CZCE', 'PF': 'CZCE',
            'OI': 'CZCE', 'RM': 'CZCE', 'CF': 'CZCE', 'SR': 'CZCE',
            'AP': 'CZCE', 'CJ': 'CZCE', 'PK': 'CZCE',
            'SC': 'INE', 'LU': 'INE', 'NR': 'INE', 'EC': 'INE',
            'SI': 'GFEX', 'LC': 'GFEX',
        }

    def connect(self) -> bool:
        """连接天勤"""
        try:
            from tqsdk import TqApi, TqAuth

            auth = TqAuth(self.tq_user, self.tq_password) if self.tq_user else None
            self.api = TqApi(auth=auth)
            self.connected = True

            logger.info("天勤数据流连接成功")
            return True

        except Exception as e:
            logger.error(f"天勤连接失败: {e}")
            return False

    def disconnect(self):
        """断开连接"""
        self._running = False

        if self._thread:
            self._thread.join(timeout=5)

        if self.api:
            try:
                self.api.close()
            except:
                pass
            self.api = None

        self.connected = False
        logger.info("天勤数据流已断开")

    def _to_tq_symbol(self, symbol: str) -> str:
        """转换为天勤合约代码"""
        product = ''.join([c for c in symbol if c.isalpha()]).upper()
        month = ''.join([c for c in symbol if c.isdigit()])

        exchange = self.exchange_map.get(product, 'SHFE')

        if month:
            return f"{exchange}.{product.lower()}{month}"
        else:
            return f"{exchange}.{product.lower()}@MAIN"

    def subscribe(self, symbols: List[str], intervals: List[str] = None):
        """
        订阅行情

        Args:
            symbols: 品种列表，如 ['RB2505', 'AU2506'] 或 ['RB', 'AU'] (主力)
            intervals: K线周期列表，如 ['1m', '5m', '1h']
        """
        if not self.api:
            logger.error("API未连接")
            return

        intervals = intervals or ['1m', '5m', '15m', '1h', '4h', '1d']

        interval_seconds = {
            '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
            '1h': 3600, '4h': 14400, '1d': 86400
        }

        for symbol in symbols:
            try:
                tq_symbol = self._to_tq_symbol(symbol)

                # 订阅Quote
                quote = self.api.get_quote(tq_symbol)
                self.quotes[symbol] = quote

                # 订阅K线
                self.klines[symbol] = {}
                for interval in intervals:
                    if interval in interval_seconds:
                        kline = self.api.get_kline_serial(
                            tq_symbol,
                            interval_seconds[interval],
                            200
                        )
                        self.klines[symbol][interval] = kline

                logger.info(f"订阅: {symbol} -> {tq_symbol}, 周期: {intervals}")

            except Exception as e:
                logger.error(f"订阅失败 {symbol}: {e}")

    def start(self):
        """启动数据流"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._data_loop, daemon=True)
        self._thread.start()
        logger.info("数据流已启动")

    def stop(self):
        """停止数据流"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("数据流已停止")

    def _data_loop(self):
        """数据更新循环"""
        while self._running and self.api:
            try:
                self.api.wait_update()

                # 检查K线更新
                for symbol, intervals in self.klines.items():
                    for interval, kline in intervals.items():
                        if self.api.is_changing(kline):
                            self._on_kline_update(symbol, interval, kline)

                # 检查行情更新
                for symbol, quote in self.quotes.items():
                    if self.api.is_changing(quote):
                        self._on_quote_update(symbol, quote)

            except Exception as e:
                if self._running:
                    logger.error(f"数据循环错误: {e}")
                    time.sleep(1)

    def _on_kline_update(self, symbol: str, interval: str, kline):
        """K线更新回调"""
        if self._on_bar_callback and len(kline) > 0:
            try:
                last_idx = len(kline) - 1
                bar = {
                    'symbol': symbol,
                    'interval': interval,
                    'datetime': datetime.fromtimestamp(kline.iloc[last_idx]['datetime'] / 1e9),
                    'open': float(kline.iloc[last_idx]['open']),
                    'high': float(kline.iloc[last_idx]['high']),
                    'low': float(kline.iloc[last_idx]['low']),
                    'close': float(kline.iloc[last_idx]['close']),
                    'volume': float(kline.iloc[last_idx]['volume'])
                }
                self._on_bar_callback(bar)
            except Exception as e:
                logger.debug(f"K线回调错误: {e}")

    def _on_quote_update(self, symbol: str, quote):
        """行情更新回调"""
        if self._on_tick_callback:
            try:
                tick = {
                    'symbol': symbol,
                    'datetime': datetime.fromtimestamp(quote.datetime / 1e9) if quote.datetime else datetime.now(),
                    'last_price': quote.last_price,
                    'bid_price': quote.bid_price1,
                    'ask_price': quote.ask_price1,
                    'bid_volume': quote.bid_volume1,
                    'ask_volume': quote.ask_volume1,
                    'volume': quote.volume,
                    'high': quote.highest,
                    'low': quote.lowest,
                    'open': quote.open
                }
                self._on_tick_callback(tick)
            except Exception as e:
                logger.debug(f"行情回调错误: {e}")

    def set_on_bar_callback(self, callback: Callable):
        """设置K线回调"""
        self._on_bar_callback = callback

    def set_on_tick_callback(self, callback: Callable):
        """设置行情回调"""
        self._on_tick_callback = callback

    def get_history(self, symbol: str, interval: str = '1h',
                    start: datetime = None, end: datetime = None,
                    count: int = 500) -> pd.DataFrame:
        """
        获取历史K线数据

        Args:
            symbol: 品种代码
            interval: K线周期
            start: 开始时间
            end: 结束时间
            count: K线数量

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
        """
        if not self.api:
            return pd.DataFrame()

        try:
            tq_symbol = self._to_tq_symbol(symbol)

            interval_seconds = {
                '1m': 60, '5m': 300, '15m': 900, '30m': 1800,
                '1h': 3600, '4h': 14400, '1d': 86400
            }

            if interval not in interval_seconds:
                return pd.DataFrame()

            kline = self.api.get_kline_serial(
                tq_symbol,
                interval_seconds[interval],
                count
            )

            self.api.wait_update()

            # 转换为DataFrame
            df = pd.DataFrame({
                'datetime': pd.to_datetime(kline['datetime'], unit='ns'),
                'open': kline['open'],
                'high': kline['high'],
                'low': kline['low'],
                'close': kline['close'],
                'volume': kline['volume']
            })

            # 过滤时间范围
            if start:
                df = df[df['datetime'] >= start]
            if end:
                df = df[df['datetime'] <= end]

            return df.dropna().reset_index(drop=True)

        except Exception as e:
            logger.error(f"获取历史数据失败: {e}")
            return pd.DataFrame()

    def get_quote(self, symbol: str) -> Optional[dict]:
        """获取最新行情"""
        if symbol in self.quotes:
            quote = self.quotes[symbol]
            return {
                'symbol': symbol,
                'last_price': quote.last_price,
                'bid_price': quote.bid_price1,
                'ask_price': quote.ask_price1,
                'high': quote.highest,
                'low': quote.lowest,
                'volume': quote.volume
            }
        return None


def create_datafeed(tq_user: str = "", tq_password: str = "") -> TqDataFeed:
    """创建数据流实例"""
    return TqDataFeed(tq_user, tq_password)
