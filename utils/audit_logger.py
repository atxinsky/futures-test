# coding=utf-8
"""
统一审计日志系统
记录所有关键操作、风控事件、交易记录
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from threading import Lock
from collections import deque

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """审计事件类型"""
    # 交易相关
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"
    TRADE_EXECUTED = "trade_executed"

    # 风控相关
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    RISK_ALERT = "risk_alert"
    RISK_FORCE_CLOSE = "risk_force_close"

    # 策略相关
    STRATEGY_STARTED = "strategy_started"
    STRATEGY_PAUSED = "strategy_paused"
    STRATEGY_RESUMED = "strategy_resumed"
    STRATEGY_STOPPED = "strategy_stopped"
    STRATEGY_ERROR = "strategy_error"
    SIGNAL_GENERATED = "signal_generated"

    # 系统相关
    ENGINE_STARTED = "engine_started"
    ENGINE_STOPPED = "engine_stopped"
    GATEWAY_CONNECTED = "gateway_connected"
    GATEWAY_DISCONNECTED = "gateway_disconnected"
    SYSTEM_ERROR = "system_error"


class AuditLogger:
    """
    审计日志记录器

    功能:
    1. 统一的审计日志记录
    2. 文件持久化
    3. 内存环形缓冲（最近N条）
    4. 按类型过滤查询
    5. 线程安全
    """

    def __init__(self, log_dir: str = None, max_memory_entries: int = 1000):
        """
        初始化审计日志记录器

        Args:
            log_dir: 日志目录
            max_memory_entries: 内存中保留的最大条目数
        """
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data', 'audit'
            )

        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 内存缓冲
        self._buffer: deque = deque(maxlen=max_memory_entries)
        self._lock = Lock()

        # 当前日志文件
        self._current_date: str = ""
        self._log_file: Optional[str] = None

    def _get_log_file(self) -> str:
        """获取当前日志文件路径"""
        today = datetime.now().strftime('%Y%m%d')
        if today != self._current_date:
            self._current_date = today
            self._log_file = os.path.join(self.log_dir, f"audit_{today}.log")
        return self._log_file

    def log(self,
            event_type: AuditEventType,
            message: str,
            data: Dict[str, Any] = None,
            level: str = "INFO"):
        """
        记录审计日志

        Args:
            event_type: 事件类型
            message: 消息描述
            data: 附加数据
            level: 日志级别 (INFO, WARNING, ERROR)
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type.value,
            'level': level,
            'message': message,
            'data': data or {}
        }

        with self._lock:
            # 添加到内存缓冲
            self._buffer.append(entry)

            # 写入文件
            try:
                log_file = self._get_log_file()
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            except Exception as e:
                logger.error(f"审计日志写入失败: {e}")

        # 同时输出到标准日志
        log_msg = f"[AUDIT] {event_type.value}: {message}"
        if level == "ERROR":
            logger.error(log_msg)
        elif level == "WARNING":
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def log_order(self, event_type: AuditEventType, order_id: str,
                  symbol: str, direction: str, volume: int, price: float,
                  **kwargs):
        """记录订单事件"""
        data = {
            'order_id': order_id,
            'symbol': symbol,
            'direction': direction,
            'volume': volume,
            'price': price,
            **kwargs
        }
        level = "WARNING" if event_type == AuditEventType.ORDER_REJECTED else "INFO"
        self.log(event_type, f"{symbol} {direction} {volume}@{price}", data, level)

    def log_trade(self, trade_id: str, order_id: str, symbol: str,
                  direction: str, volume: int, price: float, pnl: float = 0, **kwargs):
        """记录成交事件"""
        data = {
            'trade_id': trade_id,
            'order_id': order_id,
            'symbol': symbol,
            'direction': direction,
            'volume': volume,
            'price': price,
            'pnl': pnl,
            **kwargs
        }
        self.log(AuditEventType.TRADE_EXECUTED,
                 f"{symbol} {direction} {volume}@{price} PnL:{pnl:.2f}", data)

    def log_risk(self, event_type: AuditEventType, reason: str,
                 symbol: str = None, **kwargs):
        """记录风控事件"""
        data = {'reason': reason, **kwargs}
        if symbol:
            data['symbol'] = symbol
        level = "ERROR" if event_type == AuditEventType.RISK_FORCE_CLOSE else "WARNING"
        self.log(event_type, reason, data, level)

    def log_strategy(self, event_type: AuditEventType, strategy_id: str,
                     strategy_name: str, **kwargs):
        """记录策略事件"""
        data = {
            'strategy_id': strategy_id,
            'strategy_name': strategy_name,
            **kwargs
        }
        level = "ERROR" if event_type == AuditEventType.STRATEGY_ERROR else "INFO"
        self.log(event_type, f"{strategy_name} ({strategy_id})", data, level)

    def log_system(self, event_type: AuditEventType, message: str, **kwargs):
        """记录系统事件"""
        level = "ERROR" if event_type == AuditEventType.SYSTEM_ERROR else "INFO"
        self.log(event_type, message, kwargs, level)

    def get_recent(self, count: int = 100,
                   event_type: AuditEventType = None) -> List[Dict]:
        """
        获取最近的审计日志

        Args:
            count: 数量
            event_type: 可选，按类型过滤

        Returns:
            日志条目列表
        """
        with self._lock:
            entries = list(self._buffer)

        if event_type:
            entries = [e for e in entries if e['event_type'] == event_type.value]

        return entries[-count:]

    def get_by_date(self, date: str) -> List[Dict]:
        """
        获取指定日期的审计日志

        Args:
            date: 日期字符串 (YYYYMMDD)

        Returns:
            日志条目列表
        """
        log_file = os.path.join(self.log_dir, f"audit_{date}.log")
        if not os.path.exists(log_file):
            return []

        entries = []
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(json.loads(line))
        except Exception as e:
            logger.error(f"读取审计日志失败: {e}")

        return entries

    def get_statistics(self, date: str = None) -> Dict[str, int]:
        """
        获取审计日志统计

        Args:
            date: 可选，指定日期

        Returns:
            各类型事件计数
        """
        if date:
            entries = self.get_by_date(date)
        else:
            with self._lock:
                entries = list(self._buffer)

        stats = {}
        for entry in entries:
            event_type = entry.get('event_type', 'unknown')
            stats[event_type] = stats.get(event_type, 0) + 1

        return stats


# 全局单例
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(log_dir: str = None) -> AuditLogger:
    """获取审计日志记录器单例"""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_dir)
    return _audit_logger


# 便捷函数
def audit_order(event_type: AuditEventType, order_id: str, symbol: str,
                direction: str, volume: int, price: float, **kwargs):
    """记录订单审计日志"""
    get_audit_logger().log_order(event_type, order_id, symbol, direction, volume, price, **kwargs)


def audit_trade(trade_id: str, order_id: str, symbol: str,
                direction: str, volume: int, price: float, pnl: float = 0, **kwargs):
    """记录成交审计日志"""
    get_audit_logger().log_trade(trade_id, order_id, symbol, direction, volume, price, pnl, **kwargs)


def audit_risk(event_type: AuditEventType, reason: str, symbol: str = None, **kwargs):
    """记录风控审计日志"""
    get_audit_logger().log_risk(event_type, reason, symbol, **kwargs)


def audit_strategy(event_type: AuditEventType, strategy_id: str, strategy_name: str, **kwargs):
    """记录策略审计日志"""
    get_audit_logger().log_strategy(event_type, strategy_id, strategy_name, **kwargs)


def audit_system(event_type: AuditEventType, message: str, **kwargs):
    """记录系统审计日志"""
    get_audit_logger().log_system(event_type, message, **kwargs)
