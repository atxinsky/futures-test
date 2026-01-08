# coding=utf-8
"""
订单智能重报模块测试
"""

import sys
import os

# 直接导入模块，避免触发utils/__init__.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 直接导入order_retry模块
import importlib.util
spec = importlib.util.spec_from_file_location(
    "order_retry",
    os.path.join(project_root, "utils", "order_retry.py")
)
order_retry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(order_retry)

OrderRetryHandler = order_retry.OrderRetryHandler
OrderRetryConfig = order_retry.OrderRetryConfig
RejectReason = order_retry.RejectReason


def test_parse_reject_reason():
    """测试拒绝原因解析"""
    handler = OrderRetryHandler()

    # 涨停相关
    assert handler.parse_reject_reason("价格超过涨停价") == RejectReason.PRICE_OVER_LIMIT_UP
    assert handler.parse_reject_reason("Price over upper limit") == RejectReason.PRICE_OVER_LIMIT_UP

    # 跌停相关
    assert handler.parse_reject_reason("价格低于跌停价") == RejectReason.PRICE_UNDER_LIMIT_DOWN
    assert handler.parse_reject_reason("below lower limit") == RejectReason.PRICE_UNDER_LIMIT_DOWN

    # 保证金不足
    assert handler.parse_reject_reason("保证金不足") == RejectReason.INSUFFICIENT_MARGIN

    # 其他
    assert handler.parse_reject_reason("未知错误") == RejectReason.OTHER

    print("test_parse_reject_reason PASSED")


def test_can_retry():
    """测试是否可重试判断"""
    handler = OrderRetryHandler()

    # 涨跌停可以通过调价解决
    assert handler.can_retry_by_adjusting_price(RejectReason.PRICE_OVER_LIMIT_UP) == True
    assert handler.can_retry_by_adjusting_price(RejectReason.PRICE_UNDER_LIMIT_DOWN) == True

    # 保证金不足不能通过调价解决
    assert handler.can_retry_by_adjusting_price(RejectReason.INSUFFICIENT_MARGIN) == False
    assert handler.can_retry_by_adjusting_price(RejectReason.OTHER) == False

    print("test_can_retry PASSED")


def test_calculate_adjusted_price():
    """测试价格调整计算"""
    handler = OrderRetryHandler()

    limit_up = 5000.0
    limit_down = 4500.0
    price_tick = 1.0

    # 买入价超涨停 -> 调整为涨停价
    adjusted = handler.calculate_adjusted_price(
        original_price=5100,
        direction='long',
        limit_up=limit_up,
        limit_down=limit_down,
        price_tick=price_tick,
        reason=RejectReason.PRICE_OVER_LIMIT_UP
    )
    assert adjusted == limit_up, f"预期 {limit_up}，实际 {adjusted}"

    # 卖出价低于跌停 -> 调整为跌停价
    adjusted = handler.calculate_adjusted_price(
        original_price=4400,
        direction='short',
        limit_up=limit_up,
        limit_down=limit_down,
        price_tick=price_tick,
        reason=RejectReason.PRICE_UNDER_LIMIT_DOWN
    )
    assert adjusted == limit_down, f"预期 {limit_down}，实际 {adjusted}"

    print("test_calculate_adjusted_price PASSED")


def test_should_retry():
    """测试是否应重试判断"""
    config = OrderRetryConfig(max_retries=3)
    handler = OrderRetryHandler(config)

    order_id = "TEST_001"

    # 第一次重试 - 应该重试
    should, reason, msg = handler.should_retry(order_id, "价格超过涨停价")
    assert should == True
    assert reason == RejectReason.PRICE_OVER_LIMIT_UP

    # 模拟3次重试后
    handler._retry_counts[order_id] = 3
    should, reason, msg = handler.should_retry(order_id, "价格超过涨停价")
    assert should == False  # 达到最大重试次数

    # 保证金不足不重试
    should, reason, msg = handler.should_retry("TEST_002", "保证金不足")
    assert should == False

    print("test_should_retry PASSED")


def test_record_retry():
    """测试重试记录"""
    handler = OrderRetryHandler()

    # 记录重试
    record = handler.record_retry(
        original_order_id="ORDER_001",
        new_order_id="ORDER_002",
        original_price=5100,
        adjusted_price=5000,
        reason=RejectReason.PRICE_OVER_LIMIT_UP
    )

    assert record.retry_count == 1
    assert record.original_price == 5100
    assert record.adjusted_price == 5000
    assert record.success == False

    # 新订单继承重试计数
    assert handler.get_retry_count("ORDER_002") == 1

    # 记录成功
    handler.record_success("ORDER_002")
    records = handler.get_retry_records("ORDER_002")
    assert len(records) == 1
    assert records[0].success == True

    print("test_record_retry PASSED")


def test_statistics():
    """测试统计功能"""
    handler = OrderRetryHandler()

    # 模拟一些重试
    handler.record_retry("O1", "O2", 100, 99, RejectReason.PRICE_OVER_LIMIT_UP)
    handler.record_retry("O3", "O4", 100, 101, RejectReason.PRICE_UNDER_LIMIT_DOWN)
    handler.record_success("O2")

    stats = handler.get_statistics()
    assert stats['total_retries'] == 2
    assert stats['successful'] == 1
    assert stats['failed'] == 1
    assert stats['success_rate'] == 0.5

    print("test_statistics PASSED")


if __name__ == "__main__":
    test_parse_reject_reason()
    test_can_retry()
    test_calculate_adjusted_price()
    test_should_retry()
    test_record_retry()
    test_statistics()

    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
