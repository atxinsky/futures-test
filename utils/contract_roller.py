# coding=utf-8
"""
合约换月管理模块

自动检测主力合约变更，生成换月信号

换月流程：
1. 检测当前持仓合约是否仍为主力
2. 如果主力发生变化，生成 ROLL_CLOSE 平掉旧合约
3. 生成 ROLL_OPEN 在新主力合约开仓
4. 保持仓位方向和数量不变

使用场景：
- 趋势策略持有头寸接近交割月
- 主力合约切换（成交量/持仓量转移）
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, Optional, List, Callable
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)


class RollReason(Enum):
    """换月原因"""
    MAIN_CONTRACT_CHANGE = "main_change"   # 主力合约变更
    NEAR_EXPIRY = "near_expiry"            # 临近交割
    VOLUME_SHIFT = "volume_shift"          # 成交量转移
    MANUAL = "manual"                      # 手动触发


@dataclass
class RollSignal:
    """换月信号"""
    product: str                    # 品种代码（如RB）
    old_contract: str               # 旧合约（如RB2501）
    new_contract: str               # 新合约（如RB2505）
    direction: str                  # 持仓方向（long/short）
    volume: int                     # 手数
    reason: RollReason              # 换月原因
    signal_time: datetime = field(default_factory=datetime.now)
    old_price: float = 0.0          # 旧合约当前价
    new_price: float = 0.0          # 新合约当前价
    basis: float = 0.0              # 基差（新-旧）
    priority: int = 1               # 优先级（1最高）
    processed: bool = False         # 是否已处理


@dataclass
class ContractInfo:
    """合约信息"""
    product: str                    # 品种代码
    contract: str                   # 合约代码（如RB2505）
    expire_month: int               # 到期月份（如2505）
    exchange: str = ""              # 交易所
    volume: int = 0                 # 今日成交量
    open_interest: float = 0.0      # 持仓量
    last_price: float = 0.0         # 最新价
    update_time: datetime = None


@dataclass
class RollConfig:
    """换月配置"""
    # 提前换月天数（距交割月）
    days_before_expiry: int = 5

    # 主力判断标准
    use_volume: bool = True         # 使用成交量判断
    use_open_interest: bool = True  # 使用持仓量判断
    min_volume_ratio: float = 0.3   # 新合约成交量至少达到旧的30%
    min_oi_ratio: float = 0.5       # 新合约持仓量至少达到旧的50%

    # 换月执行
    auto_roll: bool = True          # 自动执行换月
    roll_at_close: bool = True      # 收盘前执行（vs 开盘执行）
    max_spread_ratio: float = 0.02  # 最大价差比例（基差/价格）

    # 品种特殊配置
    main_months: Dict[str, List[int]] = field(default_factory=lambda: {
        # 默认主力月份：1,5,9月
        'default': [1, 5, 9],
        # 特殊品种
        'IF': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # 股指每月
        'IC': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'IH': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'IM': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'AU': [2, 4, 6, 8, 10, 12],  # 黄金双月
        'AG': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # 白银每月
    })


class ContractRoller:
    """
    合约换月管理器

    功能：
    1. 跟踪各品种主力合约
    2. 检测主力变更
    3. 生成换月信号
    4. 管理换月执行

    Usage:
        roller = ContractRoller()

        # 更新合约信息
        roller.update_contract_info('RB2501', volume=50000, oi=100000, price=3800)
        roller.update_contract_info('RB2505', volume=80000, oi=150000, price=3850)

        # 检查是否需要换月
        if roller.should_roll('RB', 'RB2501'):
            signals = roller.generate_roll_signals('RB', 'RB2501', 'long', 10)
    """

    def __init__(self, config: RollConfig = None):
        self.config = config or RollConfig()

        # 合约信息缓存 {contract: ContractInfo}
        self._contracts: Dict[str, ContractInfo] = {}

        # 主力合约 {product: contract}
        self._main_contracts: Dict[str, str] = {}

        # 上一个主力 {product: contract}
        self._last_main: Dict[str, str] = {}

        # 换月信号队列
        self._roll_signals: List[RollSignal] = []

        # 已处理的换月记录
        self._roll_history: List[RollSignal] = []

        self._lock = Lock()

        # 回调
        self._on_roll_signal: Optional[Callable] = None

    def set_roll_callback(self, callback: Callable):
        """设置换月信号回调"""
        self._on_roll_signal = callback

    def parse_contract(self, contract: str) -> tuple:
        """
        解析合约代码

        Args:
            contract: 合约代码（如 RB2505, rb2505, AU2506）

        Returns:
            (product, expire_month): ('RB', 2505)
        """
        contract = contract.upper()

        # 提取字母部分（品种）和数字部分（月份）
        match = re.match(r'([A-Z]+)(\d+)', contract)
        if match:
            product = match.group(1)
            month_str = match.group(2)

            # 处理2位或4位月份
            if len(month_str) == 4:
                expire_month = int(month_str)
            elif len(month_str) == 3:
                # 如 501 -> 2501
                expire_month = 2000 + int(month_str)
            else:
                # 如 05 -> 2505（假设当前年份）
                year = datetime.now().year % 100
                expire_month = year * 100 + int(month_str)

            return product, expire_month

        return contract, 0

    def update_contract_info(
        self,
        contract: str,
        volume: int = 0,
        open_interest: float = 0,
        price: float = 0,
        exchange: str = ""
    ):
        """
        更新合约信息

        Args:
            contract: 合约代码
            volume: 成交量
            open_interest: 持仓量
            price: 最新价
            exchange: 交易所
        """
        product, expire_month = self.parse_contract(contract)

        with self._lock:
            if contract not in self._contracts:
                self._contracts[contract] = ContractInfo(
                    product=product,
                    contract=contract,
                    expire_month=expire_month,
                    exchange=exchange
                )

            info = self._contracts[contract]
            info.volume = volume
            info.open_interest = open_interest
            info.last_price = price
            info.update_time = datetime.now()
            if exchange:
                info.exchange = exchange

    def get_main_contract(self, product: str) -> Optional[str]:
        """获取品种当前主力合约"""
        with self._lock:
            return self._main_contracts.get(product.upper())

    def set_main_contract(self, product: str, contract: str):
        """设置主力合约"""
        product = product.upper()
        contract = contract.upper()

        with self._lock:
            old_main = self._main_contracts.get(product)
            if old_main and old_main != contract:
                self._last_main[product] = old_main
                logger.info(f"[换月] {product} 主力变更: {old_main} -> {contract}")

            self._main_contracts[product] = contract

    def detect_main_contract(self, product: str) -> Optional[str]:
        """
        根据成交量/持仓量检测主力合约

        Args:
            product: 品种代码

        Returns:
            主力合约代码
        """
        product = product.upper()

        with self._lock:
            # 获取该品种所有合约
            contracts = [
                info for info in self._contracts.values()
                if info.product == product and info.expire_month > 0
            ]

        if not contracts:
            return None

        # 过滤有效合约（未到期）
        today = datetime.now()
        current_ym = today.year % 100 * 100 + today.month

        valid_contracts = [
            c for c in contracts
            if c.expire_month >= current_ym
        ]

        if not valid_contracts:
            return None

        # 按成交量+持仓量排序
        def score(c):
            vol_score = c.volume if self.config.use_volume else 0
            oi_score = c.open_interest if self.config.use_open_interest else 0
            return vol_score + oi_score * 0.5

        valid_contracts.sort(key=score, reverse=True)

        return valid_contracts[0].contract

    def should_roll(
        self,
        product: str,
        current_contract: str,
        check_expiry: bool = True
    ) -> tuple:
        """
        检查是否需要换月

        Args:
            product: 品种代码
            current_contract: 当前持仓合约
            check_expiry: 是否检查到期日

        Returns:
            (should_roll, new_contract, reason)
        """
        product = product.upper()
        current_contract = current_contract.upper()

        # 获取主力合约
        main_contract = self.get_main_contract(product)

        if not main_contract:
            # 尝试检测主力
            main_contract = self.detect_main_contract(product)
            if main_contract:
                self.set_main_contract(product, main_contract)

        if not main_contract:
            return False, None, None

        # 当前合约就是主力，无需换月
        if current_contract == main_contract:
            return False, None, None

        # 解析合约月份
        _, current_month = self.parse_contract(current_contract)
        _, main_month = self.parse_contract(main_contract)

        # 当前合约比主力更老（月份更小），需要换月
        if current_month < main_month:
            return True, main_contract, RollReason.MAIN_CONTRACT_CHANGE

        # 检查是否临近到期
        if check_expiry:
            today = datetime.now()
            expire_year = 2000 + current_month // 100
            expire_month = current_month % 100

            # 交割日通常在到期月份的第三个周五（或15号左右）
            # 简化处理：到期月份的第10天开始考虑换月
            from datetime import timedelta
            try:
                expire_date = date(expire_year, expire_month, 10)
                days_to_expiry = (expire_date - today.date()).days

                if days_to_expiry <= self.config.days_before_expiry:
                    return True, main_contract, RollReason.NEAR_EXPIRY
            except ValueError:
                pass

        return False, None, None

    def generate_roll_signals(
        self,
        product: str,
        old_contract: str,
        direction: str,
        volume: int,
        reason: RollReason = RollReason.MAIN_CONTRACT_CHANGE
    ) -> List[RollSignal]:
        """
        生成换月信号

        Args:
            product: 品种代码
            old_contract: 旧合约
            direction: 持仓方向 ('long'/'short')
            volume: 手数
            reason: 换月原因

        Returns:
            换月信号列表 [ROLL_CLOSE, ROLL_OPEN]
        """
        new_contract = self.get_main_contract(product)
        if not new_contract:
            logger.warning(f"[换月] {product} 无法获取新主力合约")
            return []

        # 获取价格信息
        old_info = self._contracts.get(old_contract.upper())
        new_info = self._contracts.get(new_contract.upper())

        old_price = old_info.last_price if old_info else 0
        new_price = new_info.last_price if new_info else 0
        basis = new_price - old_price

        # 检查基差是否过大
        if old_price > 0 and abs(basis) / old_price > self.config.max_spread_ratio:
            logger.warning(f"[换月] {product} 基差过大: {basis:.2f} ({basis/old_price*100:.1f}%)")

        signal = RollSignal(
            product=product.upper(),
            old_contract=old_contract.upper(),
            new_contract=new_contract.upper(),
            direction=direction.lower(),
            volume=volume,
            reason=reason,
            old_price=old_price,
            new_price=new_price,
            basis=basis
        )

        with self._lock:
            self._roll_signals.append(signal)

        logger.info(
            f"[换月信号] {product} {direction} {volume}手 "
            f"{old_contract} -> {new_contract} 基差:{basis:.2f}"
        )

        # 触发回调
        if self._on_roll_signal:
            try:
                self._on_roll_signal(signal)
            except Exception as e:
                logger.error(f"换月回调错误: {e}")

        return [signal]

    def get_pending_signals(self) -> List[RollSignal]:
        """获取待处理的换月信号"""
        with self._lock:
            return [s for s in self._roll_signals if not s.processed]

    def mark_signal_processed(self, signal: RollSignal):
        """标记信号已处理"""
        with self._lock:
            signal.processed = True
            self._roll_history.append(signal)

    def get_next_main_month(self, product: str, current_month: int) -> int:
        """
        获取下一个主力月份

        Args:
            product: 品种代码
            current_month: 当前月份（如2501）

        Returns:
            下一个主力月份（如2505）
        """
        product = product.upper()
        main_months = self.config.main_months.get(
            product,
            self.config.main_months['default']
        )

        year = current_month // 100
        month = current_month % 100

        # 找下一个主力月份
        for m in main_months:
            if m > month:
                return year * 100 + m

        # 跨年
        return (year + 1) * 100 + main_months[0]

    def get_statistics(self) -> dict:
        """获取统计信息"""
        with self._lock:
            pending = len([s for s in self._roll_signals if not s.processed])
            processed = len(self._roll_history)

            by_reason = {}
            for s in self._roll_history:
                key = s.reason.value
                by_reason[key] = by_reason.get(key, 0) + 1

            return {
                'pending_signals': pending,
                'processed_signals': processed,
                'main_contracts': dict(self._main_contracts),
                'by_reason': by_reason
            }

    def clear(self):
        """清除所有数据"""
        with self._lock:
            self._contracts.clear()
            self._main_contracts.clear()
            self._last_main.clear()
            self._roll_signals.clear()


# 全局单例
_contract_roller: Optional[ContractRoller] = None


def get_contract_roller(config: RollConfig = None) -> ContractRoller:
    """获取合约换月管理器单例"""
    global _contract_roller
    if _contract_roller is None:
        _contract_roller = ContractRoller(config)
    return _contract_roller
