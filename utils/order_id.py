# coding=utf-8
"""
订单号生成器

订单号格式设计：
  {策略简码}_{日期}_{序号}_{信号ID后4位}

示例：
  B6_0108_000001_0000  (brother2v6策略，1月8日，第1单，无关联信号)
  B6_0108_000002_1234  (brother2v6策略，1月8日，第2单，信号ID末4位1234)

特点：
  1. 可读性强：一眼看出策略、日期、顺序
  2. 可追溯：通过信号ID关联到StrategyTrade
  3. 唯一性：策略+日期+序号保证唯一
  4. 兼容CTP：总长度<=13位（去掉下划线后）
"""

from datetime import datetime
from typing import Dict, Optional
import threading


# 策略名称到简码的映射
STRATEGY_CODES = {
    'brother2v6': 'B6',
    'brother2v5': 'B5',
    'brother2_dual': 'BD',
    'brother2_enhanced': 'BE',
    'macd_pullback': 'MP',
    'macdema_v3': 'M3',
    'emanew_v5': 'E5',
    'donchian_trend': 'DT',
    'momentum_mean': 'MM',
    'commodity_trend': 'CT',
    'scalp_trend': 'ST',
    'intraday_breakout': 'IB',
    'intraday_momentum': 'IM',
    'mean_revert': 'MR',
    'bollinger': 'BB',
    'dual_ma': 'DM',
    'turtle': 'TT',
    'weekly_trend': 'WT',
    # 默认/未知策略
    'default': 'XX',
    'manual': 'MN',  # 手动下单
    'stop_loss': 'SL',  # 止损单
    'take_profit': 'TP',  # 止盈单
}


class OrderIdGenerator:
    """
    订单号生成器

    线程安全，支持多策略并发生成订单号

    Usage:
        generator = OrderIdGenerator()

        # 生成订单号
        order_id = generator.generate("brother2v6")
        # -> "B6_0108_000001_0000"

        # 带信号ID
        order_id = generator.generate("brother2v6", signal_id="trade_20260108_001234")
        # -> "B6_0108_000002_1234"

        # 解析订单号
        info = generator.parse("B6_0108_000001_1234")
        # -> {'strategy_code': 'B6', 'date': '0108', 'seq': 1, 'signal_suffix': '1234'}
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._counters: Dict[str, int] = {}  # {strategy_date: counter}
        self._counter_lock = threading.Lock()
        self._initialized = True

    def _get_strategy_code(self, strategy_name: str) -> str:
        """获取策略简码"""
        # 尝试精确匹配
        if strategy_name in STRATEGY_CODES:
            return STRATEGY_CODES[strategy_name]

        # 尝试模糊匹配
        name_lower = strategy_name.lower()
        for full_name, code in STRATEGY_CODES.items():
            if full_name in name_lower or name_lower in full_name:
                return code

        # 使用策略名前两个字符大写
        return strategy_name[:2].upper() if len(strategy_name) >= 2 else 'XX'

    def _get_signal_suffix(self, signal_id: Optional[str]) -> str:
        """提取信号ID后4位"""
        if not signal_id:
            return '0000'

        # 尝试提取数字
        digits = ''.join(c for c in signal_id if c.isdigit())
        if len(digits) >= 4:
            return digits[-4:]
        elif digits:
            return digits.zfill(4)
        else:
            return '0000'

    def generate(
        self,
        strategy_name: str = 'default',
        signal_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        生成订单号

        Args:
            strategy_name: 策略名称
            signal_id: 关联的信号/交易ID（可选）
            timestamp: 时间戳（默认当前时间）

        Returns:
            订单号，格式: {策略简码}_{MMDD}_{6位序号}_{4位信号后缀}
        """
        now = timestamp or datetime.now()
        date_str = now.strftime('%m%d')
        strategy_code = self._get_strategy_code(strategy_name)

        # 生成计数器key
        counter_key = f"{strategy_code}_{date_str}"

        # 原子递增
        with self._counter_lock:
            if counter_key not in self._counters:
                self._counters[counter_key] = 0
            self._counters[counter_key] += 1
            seq = self._counters[counter_key]

        # 提取信号后缀
        signal_suffix = self._get_signal_suffix(signal_id)

        # 组装订单号
        order_id = f"{strategy_code}_{date_str}_{seq:06d}_{signal_suffix}"

        return order_id

    def generate_compact(
        self,
        strategy_name: str = 'default',
        signal_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        生成紧凑格式订单号（无下划线，兼容CTP 13位限制）

        Returns:
            订单号，格式: {2位策略}{4位日期}{6位序号} = 12位
        """
        now = timestamp or datetime.now()
        date_str = now.strftime('%m%d')
        strategy_code = self._get_strategy_code(strategy_name)

        counter_key = f"{strategy_code}_{date_str}"

        with self._counter_lock:
            if counter_key not in self._counters:
                self._counters[counter_key] = 0
            self._counters[counter_key] += 1
            seq = self._counters[counter_key]

        # 紧凑格式：B601080000001
        return f"{strategy_code}{date_str}{seq:06d}"

    def parse(self, order_id: str) -> Optional[Dict]:
        """
        解析订单号

        Args:
            order_id: 订单号

        Returns:
            解析结果字典，解析失败返回None
        """
        try:
            if '_' in order_id:
                # 完整格式: B6_0108_000001_1234
                parts = order_id.split('_')
                if len(parts) >= 3:
                    return {
                        'strategy_code': parts[0],
                        'date': parts[1],
                        'seq': int(parts[2]),
                        'signal_suffix': parts[3] if len(parts) > 3 else '0000',
                        'format': 'full'
                    }
            else:
                # 紧凑格式: B60108000001
                if len(order_id) >= 12:
                    return {
                        'strategy_code': order_id[:2],
                        'date': order_id[2:6],
                        'seq': int(order_id[6:12]),
                        'signal_suffix': order_id[12:] if len(order_id) > 12 else '',
                        'format': 'compact'
                    }
        except (ValueError, IndexError):
            pass

        return None

    def reset_daily(self):
        """重置每日计数器（建议在每日开盘前调用）"""
        with self._counter_lock:
            self._counters.clear()

    def get_statistics(self) -> Dict[str, int]:
        """获取各策略今日订单数统计"""
        with self._counter_lock:
            return dict(self._counters)


# 全局单例
_generator = None

def get_order_id_generator() -> OrderIdGenerator:
    """获取全局订单号生成器"""
    global _generator
    if _generator is None:
        _generator = OrderIdGenerator()
    return _generator


def generate_order_id(
    strategy_name: str = 'default',
    signal_id: Optional[str] = None,
    compact: bool = False
) -> str:
    """
    便捷函数：生成订单号

    Args:
        strategy_name: 策略名称
        signal_id: 关联信号ID
        compact: 是否使用紧凑格式

    Returns:
        订单号
    """
    gen = get_order_id_generator()
    if compact:
        return gen.generate_compact(strategy_name, signal_id)
    return gen.generate(strategy_name, signal_id)
