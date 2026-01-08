# coding=utf-8
"""
审计日志系统测试
"""

import pytest
import os
from datetime import datetime


class TestAuditEventType:
    """审计事件类型测试"""

    def test_event_types_exist(self):
        """测试事件类型存在"""
        from utils.audit_logger import AuditEventType

        # 交易相关
        assert hasattr(AuditEventType, 'ORDER_SUBMITTED')
        assert hasattr(AuditEventType, 'ORDER_FILLED')
        assert hasattr(AuditEventType, 'TRADE_EXECUTED')

        # 风控相关
        assert hasattr(AuditEventType, 'RISK_CHECK_PASSED')
        assert hasattr(AuditEventType, 'RISK_CHECK_FAILED')

        # 策略相关
        assert hasattr(AuditEventType, 'STRATEGY_STARTED')
        assert hasattr(AuditEventType, 'STRATEGY_STOPPED')

    def test_event_type_values(self):
        """测试事件类型值"""
        from utils.audit_logger import AuditEventType

        assert AuditEventType.ORDER_SUBMITTED.value == "order_submitted"
        assert AuditEventType.RISK_CHECK_PASSED.value == "risk_check_passed"


class TestAuditLogger:
    """审计日志记录器测试"""

    def test_init(self, temp_dir):
        """测试初始化"""
        from utils.audit_logger import AuditLogger

        logger = AuditLogger(log_dir=temp_dir, max_memory_entries=100)
        assert logger.log_dir == temp_dir
        assert os.path.exists(temp_dir)

    def test_log_basic(self, temp_dir):
        """测试基本日志记录"""
        from utils.audit_logger import AuditLogger, AuditEventType

        logger = AuditLogger(log_dir=temp_dir)
        logger.log(AuditEventType.ENGINE_STARTED, "引擎启动")

        recent = logger.get_recent(10)
        assert len(recent) == 1
        assert recent[0]['event_type'] == 'engine_started'
        assert recent[0]['message'] == '引擎启动'

    def test_log_with_data(self, temp_dir):
        """测试带数据的日志"""
        from utils.audit_logger import AuditLogger, AuditEventType

        logger = AuditLogger(log_dir=temp_dir)
        logger.log(
            AuditEventType.ORDER_SUBMITTED,
            "订单提交",
            data={'order_id': 'ORD001', 'symbol': 'RB2505'}
        )

        recent = logger.get_recent(10)
        assert recent[0]['data']['order_id'] == 'ORD001'
        assert recent[0]['data']['symbol'] == 'RB2505'

    def test_log_order(self, temp_dir):
        """测试订单日志"""
        from utils.audit_logger import AuditLogger, AuditEventType

        logger = AuditLogger(log_dir=temp_dir)
        logger.log_order(
            AuditEventType.ORDER_SUBMITTED,
            order_id='ORD001',
            symbol='RB2505',
            direction='BUY',
            volume=2,
            price=3500.0
        )

        recent = logger.get_recent(10)
        assert recent[0]['data']['order_id'] == 'ORD001'
        assert recent[0]['data']['volume'] == 2

    def test_log_trade(self, temp_dir):
        """测试成交日志"""
        from utils.audit_logger import AuditLogger

        logger = AuditLogger(log_dir=temp_dir)
        logger.log_trade(
            trade_id='TRD001',
            order_id='ORD001',
            symbol='RB2505',
            direction='BUY',
            volume=2,
            price=3500.0,
            pnl=150.0
        )

        recent = logger.get_recent(10)
        assert recent[0]['data']['pnl'] == 150.0

    def test_log_risk(self, temp_dir):
        """测试风控日志"""
        from utils.audit_logger import AuditLogger, AuditEventType

        logger = AuditLogger(log_dir=temp_dir)
        logger.log_risk(
            AuditEventType.RISK_CHECK_FAILED,
            reason="超过日亏损限制",
            symbol='RB2505'
        )

        recent = logger.get_recent(10)
        assert '亏损' in recent[0]['data']['reason']
        assert recent[0]['level'] == 'WARNING'

    def test_log_strategy(self, temp_dir):
        """测试策略日志"""
        from utils.audit_logger import AuditLogger, AuditEventType

        logger = AuditLogger(log_dir=temp_dir)
        logger.log_strategy(
            AuditEventType.STRATEGY_STARTED,
            strategy_id='STR001',
            strategy_name='TestStrategy'
        )

        recent = logger.get_recent(10)
        assert recent[0]['data']['strategy_id'] == 'STR001'

    def test_get_recent_with_filter(self, temp_dir):
        """测试按类型过滤"""
        from utils.audit_logger import AuditLogger, AuditEventType

        logger = AuditLogger(log_dir=temp_dir)

        # 添加不同类型的日志
        logger.log(AuditEventType.ENGINE_STARTED, "引擎启动")
        logger.log(AuditEventType.ORDER_SUBMITTED, "订单提交")
        logger.log(AuditEventType.ENGINE_STOPPED, "引擎停止")

        # 过滤引擎事件
        recent = logger.get_recent(10, event_type=AuditEventType.ENGINE_STARTED)
        assert len(recent) == 1
        assert recent[0]['event_type'] == 'engine_started'

    def test_get_statistics(self, temp_dir):
        """测试统计功能"""
        from utils.audit_logger import AuditLogger, AuditEventType

        logger = AuditLogger(log_dir=temp_dir)

        # 添加多条日志
        logger.log(AuditEventType.ORDER_SUBMITTED, "订单1")
        logger.log(AuditEventType.ORDER_SUBMITTED, "订单2")
        logger.log(AuditEventType.ORDER_FILLED, "成交1")

        stats = logger.get_statistics()
        assert stats.get('order_submitted', 0) == 2
        assert stats.get('order_filled', 0) == 1

    def test_file_persistence(self, temp_dir):
        """测试文件持久化"""
        from utils.audit_logger import AuditLogger, AuditEventType

        logger = AuditLogger(log_dir=temp_dir)
        logger.log(AuditEventType.ENGINE_STARTED, "测试持久化")

        # 检查文件是否存在
        today = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(temp_dir, f"audit_{today}.log")
        assert os.path.exists(log_file)

        # 读取文件验证内容
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            assert 'engine_started' in content
            assert '测试持久化' in content

    def test_max_memory_entries(self, temp_dir):
        """测试内存缓冲限制"""
        from utils.audit_logger import AuditLogger, AuditEventType

        logger = AuditLogger(log_dir=temp_dir, max_memory_entries=5)

        # 添加超过限制的日志
        for i in range(10):
            logger.log(AuditEventType.ORDER_SUBMITTED, f"订单{i}")

        # 内存中只保留最后5条
        recent = logger.get_recent(100)
        assert len(recent) == 5


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_get_audit_logger_singleton(self, temp_dir):
        """测试单例获取"""
        from utils.audit_logger import get_audit_logger, _audit_logger

        # 重置单例
        import utils.audit_logger
        utils.audit_logger._audit_logger = None

        logger1 = get_audit_logger(temp_dir)
        logger2 = get_audit_logger()

        assert logger1 is logger2
