# coding=utf-8
"""
涨跌停价格管理模块
处理中国期货市场特有的涨跌停板限制
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime, date
from threading import Lock

logger = logging.getLogger(__name__)


# 各交易所涨跌停比例配置
# 注意：实际涨跌停比例可能因品种、交易状态而异
LIMIT_RATIOS = {
    # 中金所 - 股指期货
    'CFFEX': {
        'default': 0.10,  # 10%
        'IF': 0.10, 'IH': 0.10, 'IC': 0.10, 'IM': 0.10,
        'T': 0.012, 'TF': 0.012, 'TS': 0.005, 'TL': 0.02,  # 国债期货特殊
    },
    # 上期所
    'SHFE': {
        'default': 0.07,  # 大部分7%
        'AU': 0.08, 'AG': 0.09,  # 贵金属
        'CU': 0.07, 'AL': 0.07, 'ZN': 0.08, 'PB': 0.07,  # 有色
        'NI': 0.10, 'SN': 0.08,
        'RB': 0.08, 'HC': 0.08, 'WR': 0.07,  # 黑色
        'RU': 0.08, 'SP': 0.07, 'NR': 0.08,  # 橡胶
        'FU': 0.08, 'BU': 0.08,  # 能源
    },
    # 大商所
    'DCE': {
        'default': 0.08,
        'I': 0.10, 'J': 0.10, 'JM': 0.10,  # 黑色系波动大
        'M': 0.07, 'Y': 0.07, 'P': 0.08,  # 油脂
        'C': 0.06, 'CS': 0.06, 'JD': 0.08,  # 农产品
        'L': 0.07, 'V': 0.07, 'PP': 0.07, 'EG': 0.08, 'EB': 0.08,  # 化工
        'LH': 0.08,  # 生猪
    },
    # 郑商所
    'CZCE': {
        'default': 0.07,
        'TA': 0.07, 'MA': 0.08, 'SA': 0.08, 'FG': 0.08,  # 化工
        'CF': 0.07, 'SR': 0.07, 'AP': 0.08, 'CJ': 0.08,  # 农产品
        'OI': 0.07, 'RM': 0.07,  # 油脂
        'SF': 0.08, 'SM': 0.08,  # 铁合金
        'UR': 0.07, 'PF': 0.07,
    },
    # 上海国际能源交易中心
    'INE': {
        'default': 0.10,
        'SC': 0.10, 'LU': 0.10, 'NR': 0.08, 'BC': 0.08,
        'EC': 0.10,
    },
    # 广期所
    'GFEX': {
        'default': 0.10,
        'SI': 0.10, 'LC': 0.12,  # 碳酸锂波动大
    }
}


@dataclass
class LimitPriceInfo:
    """涨跌停价格信息"""
    symbol: str
    trade_date: date
    settlement_price: float  # 结算价（上一交易日）
    limit_up: float          # 涨停价
    limit_down: float        # 跌停价
    limit_ratio: float       # 涨跌停比例
    price_tick: float        # 最小变动价位
    update_time: datetime = None

    def __post_init__(self):
        if self.update_time is None:
            self.update_time = datetime.now()


class LimitPriceManager:
    """
    涨跌停价格管理器

    功能:
    1. 根据结算价计算涨跌停价格
    2. 缓存当日涨跌停价格
    3. 提供价格合法性检查
    """

    def __init__(self):
        self._cache: Dict[str, LimitPriceInfo] = {}
        self._lock = Lock()

    def get_limit_ratio(self, symbol: str, exchange: str = None) -> float:
        """
        获取品种涨跌停比例

        Args:
            symbol: 品种代码（如RB、AU）
            exchange: 交易所代码

        Returns:
            涨跌停比例
        """
        # 提取纯品种代码
        product = ''.join([c for c in symbol if c.isalpha()]).upper()

        # 如果未指定交易所，尝试从配置获取
        if not exchange:
            try:
                from config import get_instrument
                inst = get_instrument(product)
                if inst:
                    exchange = inst.get('exchange', '')
            except:
                pass

        if exchange and exchange in LIMIT_RATIOS:
            exchange_limits = LIMIT_RATIOS[exchange]
            return exchange_limits.get(product, exchange_limits.get('default', 0.10))

        # 默认10%
        return 0.10

    def get_price_tick(self, symbol: str) -> float:
        """获取最小变动价位"""
        product = ''.join([c for c in symbol if c.isalpha()]).upper()
        try:
            from config import get_instrument
            inst = get_instrument(product)
            if inst:
                return inst.get('price_tick', 1.0)
        except:
            pass
        return 1.0

    def calculate_limit_prices(
        self,
        symbol: str,
        settlement_price: float,
        exchange: str = None
    ) -> Tuple[float, float]:
        """
        计算涨跌停价格

        Args:
            symbol: 品种代码
            settlement_price: 结算价
            exchange: 交易所

        Returns:
            (涨停价, 跌停价)
        """
        if settlement_price <= 0:
            return 0, 0

        ratio = self.get_limit_ratio(symbol, exchange)
        price_tick = self.get_price_tick(symbol)

        # 计算涨跌停价（需要按最小变动价位取整）
        limit_up_raw = settlement_price * (1 + ratio)
        limit_down_raw = settlement_price * (1 - ratio)

        # 涨停价向下取整，跌停价向上取整（对交易者不利的方向）
        if price_tick > 0:
            limit_up = int(limit_up_raw / price_tick) * price_tick
            limit_down = (int(limit_down_raw / price_tick) + 1) * price_tick
            # 确保跌停价不低于原始计算值太多
            if limit_down > limit_down_raw + price_tick:
                limit_down = int(limit_down_raw / price_tick) * price_tick
        else:
            limit_up = limit_up_raw
            limit_down = limit_down_raw

        return limit_up, limit_down

    def update_limit_price(
        self,
        symbol: str,
        settlement_price: float,
        trade_date: date = None,
        exchange: str = None
    ) -> LimitPriceInfo:
        """
        更新品种涨跌停价格

        Args:
            symbol: 品种代码
            settlement_price: 结算价
            trade_date: 交易日期
            exchange: 交易所

        Returns:
            LimitPriceInfo
        """
        if trade_date is None:
            trade_date = date.today()

        limit_up, limit_down = self.calculate_limit_prices(
            symbol, settlement_price, exchange
        )

        info = LimitPriceInfo(
            symbol=symbol,
            trade_date=trade_date,
            settlement_price=settlement_price,
            limit_up=limit_up,
            limit_down=limit_down,
            limit_ratio=self.get_limit_ratio(symbol, exchange),
            price_tick=self.get_price_tick(symbol)
        )

        with self._lock:
            self._cache[symbol.upper()] = info

        logger.debug(f"更新涨跌停: {symbol} 结算价={settlement_price:.2f} "
                    f"涨停={limit_up:.2f} 跌停={limit_down:.2f}")

        return info

    def get_limit_price(self, symbol: str) -> Optional[LimitPriceInfo]:
        """获取缓存的涨跌停价格"""
        with self._lock:
            return self._cache.get(symbol.upper())

    def check_price_valid(
        self,
        symbol: str,
        price: float,
        settlement_price: float = None
    ) -> Tuple[bool, str]:
        """
        检查价格是否在涨跌停范围内

        Args:
            symbol: 品种代码
            price: 待检查价格
            settlement_price: 结算价（如果未缓存则需要提供）

        Returns:
            (是否有效, 原因)
        """
        if price <= 0:
            return False, "价格必须大于0"

        # 尝试从缓存获取
        info = self.get_limit_price(symbol)

        # 如果没有缓存但提供了结算价，临时计算
        if info is None and settlement_price and settlement_price > 0:
            limit_up, limit_down = self.calculate_limit_prices(symbol, settlement_price)
            if price > limit_up:
                return False, f"价格{price:.2f}超过涨停价{limit_up:.2f}"
            if price < limit_down:
                return False, f"价格{price:.2f}低于跌停价{limit_down:.2f}"
            return True, ""

        # 如果有缓存信息
        if info:
            if price > info.limit_up:
                return False, f"价格{price:.2f}超过涨停价{info.limit_up:.2f}"
            if price < info.limit_down:
                return False, f"价格{price:.2f}低于跌停价{info.limit_down:.2f}"
            return True, ""

        # 无法验证时默认通过（避免误拦截）
        return True, ""

    def is_limit_up(self, symbol: str, price: float, tolerance: float = 0.001) -> bool:
        """检查是否涨停"""
        info = self.get_limit_price(symbol)
        if info and info.limit_up > 0:
            return abs(price - info.limit_up) / info.limit_up < tolerance
        return False

    def is_limit_down(self, symbol: str, price: float, tolerance: float = 0.001) -> bool:
        """检查是否跌停"""
        info = self.get_limit_price(symbol)
        if info and info.limit_down > 0:
            return abs(price - info.limit_down) / info.limit_down < tolerance
        return False

    def clear_cache(self, symbol: str = None):
        """清除缓存"""
        with self._lock:
            if symbol:
                self._cache.pop(symbol.upper(), None)
            else:
                self._cache.clear()


# 全局单例
_limit_price_manager: Optional[LimitPriceManager] = None


def get_limit_price_manager() -> LimitPriceManager:
    """获取涨跌停管理器单例"""
    global _limit_price_manager
    if _limit_price_manager is None:
        _limit_price_manager = LimitPriceManager()
    return _limit_price_manager


def check_limit_price(
    symbol: str,
    price: float,
    settlement_price: float = None
) -> Tuple[bool, str]:
    """
    便捷函数：检查价格是否在涨跌停范围内

    Args:
        symbol: 品种代码
        price: 待检查价格
        settlement_price: 结算价

    Returns:
        (是否有效, 原因)
    """
    return get_limit_price_manager().check_price_valid(symbol, price, settlement_price)
