# coding=utf-8
"""
订单智能重报模块

处理因涨跌停等原因被拒绝的订单，自动调价重报

场景：
1. 买入价格超过涨停价 -> 调整为涨停价重报
2. 卖出价格低于跌停价 -> 调整为跌停价重报
3. 极端行情下无法成交 -> 记录并通知
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Callable, List
from threading import Lock
from enum import Enum

logger = logging.getLogger(__name__)


class RejectReason(Enum):
    """订单拒绝原因"""
    PRICE_OVER_LIMIT_UP = "price_over_limit_up"      # 价格超过涨停
    PRICE_UNDER_LIMIT_DOWN = "price_under_limit_down"  # 价格低于跌停
    LIMIT_UP_NO_LIQUIDITY = "limit_up_no_liquidity"   # 涨停无买盘（卖出时）
    LIMIT_DOWN_NO_LIQUIDITY = "limit_down_no_liquidity"  # 跌停无卖盘（买入时）
    INSUFFICIENT_MARGIN = "insufficient_margin"       # 保证金不足
    POSITION_LIMIT = "position_limit"                 # 持仓限制
    OTHER = "other"                                   # 其他原因


@dataclass
class RetryRecord:
    """重试记录"""
    original_order_id: str
    retry_order_id: str
    retry_count: int
    original_price: float
    adjusted_price: float
    reason: RejectReason
    retry_time: datetime = field(default_factory=datetime.now)
    success: bool = False


@dataclass
class OrderRetryConfig:
    """重试配置"""
    max_retries: int = 3                    # 最大重试次数
    retry_on_limit_price: bool = True       # 涨跌停时是否重试
    use_limit_price: bool = True            # 使用涨跌停价格重报
    price_offset_ticks: int = 0             # 价格偏移tick数（0=精确涨跌停价）
    notify_on_final_fail: bool = True       # 最终失败时是否通知


class OrderRetryHandler:
    """
    订单智能重报处理器

    功能：
    1. 解析订单拒绝原因
    2. 判断是否可以通过调价重报
    3. 计算调整后的价格
    4. 触发重报并跟踪结果

    Usage:
        handler = OrderRetryHandler(config)

        # 当订单被拒绝时调用
        if handler.should_retry(order, reject_msg):
            new_price = handler.get_adjusted_price(order, limit_up, limit_down)
            new_order_id = gateway.send_order(adjusted_request)
            handler.record_retry(order.order_id, new_order_id, new_price, reason)
    """

    def __init__(self, config: OrderRetryConfig = None):
        self.config = config or OrderRetryConfig()
        self._retry_counts: Dict[str, int] = {}  # order_id -> retry_count
        self._retry_records: List[RetryRecord] = []
        self._lock = Lock()

        # 回调函数
        self._on_retry_callback: Optional[Callable] = None
        self._on_final_fail_callback: Optional[Callable] = None

    def set_callbacks(
        self,
        on_retry: Callable = None,
        on_final_fail: Callable = None
    ):
        """设置回调函数"""
        self._on_retry_callback = on_retry
        self._on_final_fail_callback = on_final_fail

    def parse_reject_reason(self, reject_msg: str) -> RejectReason:
        """
        解析订单拒绝原因

        Args:
            reject_msg: 拒绝消息（来自交易所或TqSdk）

        Returns:
            RejectReason
        """
        msg_lower = reject_msg.lower() if reject_msg else ""

        # 涨跌停相关
        if any(kw in msg_lower for kw in ['涨停', 'limit up', 'upper limit', '超过涨']):
            return RejectReason.PRICE_OVER_LIMIT_UP
        if any(kw in msg_lower for kw in ['跌停', 'limit down', 'lower limit', '低于跌']):
            return RejectReason.PRICE_UNDER_LIMIT_DOWN

        # 流动性相关
        if '无买盘' in msg_lower or 'no bid' in msg_lower:
            return RejectReason.LIMIT_UP_NO_LIQUIDITY
        if '无卖盘' in msg_lower or 'no ask' in msg_lower:
            return RejectReason.LIMIT_DOWN_NO_LIQUIDITY

        # 资金相关
        if any(kw in msg_lower for kw in ['保证金', 'margin', '资金不足']):
            return RejectReason.INSUFFICIENT_MARGIN

        # 持仓相关
        if any(kw in msg_lower for kw in ['持仓', 'position', '超限']):
            return RejectReason.POSITION_LIMIT

        return RejectReason.OTHER

    def can_retry_by_adjusting_price(self, reason: RejectReason) -> bool:
        """
        判断是否可以通过调价解决

        Args:
            reason: 拒绝原因

        Returns:
            是否可以调价重报
        """
        # 只有价格超限可以通过调价解决
        return reason in (
            RejectReason.PRICE_OVER_LIMIT_UP,
            RejectReason.PRICE_UNDER_LIMIT_DOWN
        )

    def get_retry_count(self, order_id: str) -> int:
        """获取订单重试次数"""
        with self._lock:
            return self._retry_counts.get(order_id, 0)

    def should_retry(
        self,
        order_id: str,
        reject_msg: str
    ) -> tuple:
        """
        判断是否应该重试

        Args:
            order_id: 订单ID
            reject_msg: 拒绝消息

        Returns:
            (should_retry, reason, message)
        """
        reason = self.parse_reject_reason(reject_msg)
        retry_count = self.get_retry_count(order_id)

        # 检查重试次数
        if retry_count >= self.config.max_retries:
            return False, reason, f"已达最大重试次数 {self.config.max_retries}"

        # 检查是否可以通过调价解决
        if not self.can_retry_by_adjusting_price(reason):
            return False, reason, f"原因 {reason.value} 无法通过调价解决"

        # 检查配置是否允许
        if not self.config.retry_on_limit_price:
            return False, reason, "配置禁止涨跌停重试"

        return True, reason, f"第 {retry_count + 1} 次重试"

    def calculate_adjusted_price(
        self,
        original_price: float,
        direction: str,
        limit_up: float,
        limit_down: float,
        price_tick: float = 1.0,
        reason: RejectReason = None
    ) -> float:
        """
        计算调整后的价格

        Args:
            original_price: 原始价格
            direction: 方向 ('long'/'short')
            limit_up: 涨停价
            limit_down: 跌停价
            price_tick: 最小变动价位
            reason: 拒绝原因

        Returns:
            调整后的价格
        """
        offset = self.config.price_offset_ticks * price_tick

        if reason == RejectReason.PRICE_OVER_LIMIT_UP:
            # 买入价超涨停 -> 使用涨停价
            adjusted = limit_up - offset
        elif reason == RejectReason.PRICE_UNDER_LIMIT_DOWN:
            # 卖出价低于跌停 -> 使用跌停价
            adjusted = limit_down + offset
        else:
            # 根据方向调整
            if direction.lower() in ('long', 'buy'):
                # 买入：使用涨停价（最高可接受价格）
                adjusted = limit_up - offset
            else:
                # 卖出：使用跌停价（最低可接受价格）
                adjusted = limit_down + offset

        # 确保价格在合法范围内
        adjusted = max(limit_down, min(limit_up, adjusted))

        # 按价格tick取整
        if price_tick > 0:
            adjusted = round(adjusted / price_tick) * price_tick

        return adjusted

    def record_retry(
        self,
        original_order_id: str,
        new_order_id: str,
        original_price: float,
        adjusted_price: float,
        reason: RejectReason
    ) -> RetryRecord:
        """
        记录重试

        Args:
            original_order_id: 原订单ID
            new_order_id: 新订单ID
            original_price: 原价格
            adjusted_price: 调整后价格
            reason: 拒绝原因

        Returns:
            RetryRecord
        """
        with self._lock:
            # 更新重试计数
            count = self._retry_counts.get(original_order_id, 0) + 1
            self._retry_counts[original_order_id] = count

            # 新订单继承重试计数
            self._retry_counts[new_order_id] = count

            record = RetryRecord(
                original_order_id=original_order_id,
                retry_order_id=new_order_id,
                retry_count=count,
                original_price=original_price,
                adjusted_price=adjusted_price,
                reason=reason
            )
            self._retry_records.append(record)

        logger.info(
            f"[订单重报] {original_order_id} -> {new_order_id} "
            f"价格 {original_price:.2f} -> {adjusted_price:.2f} "
            f"原因: {reason.value} 第{count}次重试"
        )

        # 触发回调
        if self._on_retry_callback:
            try:
                self._on_retry_callback(record)
            except Exception as e:
                logger.error(f"重试回调错误: {e}")

        return record

    def record_success(self, order_id: str):
        """记录重试成功"""
        with self._lock:
            for record in reversed(self._retry_records):
                if record.retry_order_id == order_id:
                    record.success = True
                    logger.info(f"[订单重报成功] {order_id}")
                    break

    def record_final_fail(self, order_id: str, reason: str = ""):
        """
        记录最终失败

        Args:
            order_id: 订单ID
            reason: 失败原因
        """
        logger.warning(f"[订单重报最终失败] {order_id} {reason}")

        if self.config.notify_on_final_fail and self._on_final_fail_callback:
            try:
                self._on_final_fail_callback(order_id, reason)
            except Exception as e:
                logger.error(f"最终失败回调错误: {e}")

    def get_retry_records(self, order_id: str = None) -> List[RetryRecord]:
        """获取重试记录"""
        with self._lock:
            if order_id:
                return [r for r in self._retry_records
                        if r.original_order_id == order_id or r.retry_order_id == order_id]
            return list(self._retry_records)

    def get_statistics(self) -> dict:
        """获取统计信息"""
        with self._lock:
            total = len(self._retry_records)
            success = len([r for r in self._retry_records if r.success])

            by_reason = {}
            for r in self._retry_records:
                key = r.reason.value
                if key not in by_reason:
                    by_reason[key] = {'total': 0, 'success': 0}
                by_reason[key]['total'] += 1
                if r.success:
                    by_reason[key]['success'] += 1

            return {
                'total_retries': total,
                'successful': success,
                'failed': total - success,
                'success_rate': success / total if total > 0 else 0,
                'by_reason': by_reason
            }

    def clear(self):
        """清除所有记录"""
        with self._lock:
            self._retry_counts.clear()
            self._retry_records.clear()


# 全局单例
_retry_handler: Optional[OrderRetryHandler] = None


def get_order_retry_handler(config: OrderRetryConfig = None) -> OrderRetryHandler:
    """获取订单重试处理器单例"""
    global _retry_handler
    if _retry_handler is None:
        _retry_handler = OrderRetryHandler(config)
    return _retry_handler
