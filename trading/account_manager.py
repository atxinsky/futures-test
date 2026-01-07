# coding=utf-8
"""
账户管理器
负责账户资金的跟踪和管理
"""

from typing import Dict, List, Optional
from datetime import datetime, date
from dataclasses import dataclass, field
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import Account, EventType, Event
from core.event_engine import EventEngine

logger = logging.getLogger(__name__)


@dataclass
class DailyRecord:
    """日度记录"""
    date: date
    opening_balance: float = 0.0
    closing_balance: float = 0.0
    deposit: float = 0.0
    withdraw: float = 0.0
    realized_pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0


@dataclass
class PerformanceMetrics:
    """绩效指标"""
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    trading_days: int = 0


class AccountManager:
    """
    账户管理器

    功能:
    1. 账户状态跟踪
    2. 资金历史记录
    3. 绩效统计
    4. 入金出金管理
    """

    def __init__(self, event_engine: EventEngine):
        self.event_engine = event_engine

        # 当前账户
        self.account: Optional[Account] = None

        # 历史记录
        self.daily_records: Dict[date, DailyRecord] = {}
        self.equity_history: List[tuple] = []  # [(datetime, equity)]

        # 初始资金
        self.initial_capital: float = 0.0

        # 峰值追踪
        self._peak_equity: float = 0.0
        self._max_drawdown: float = 0.0
        self._max_drawdown_pct: float = 0.0

        # 注册事件
        self.event_engine.register(EventType.ACCOUNT, self._on_account)

    def initialize(self, initial_capital: float):
        """初始化账户"""
        self.initial_capital = initial_capital
        self._peak_equity = initial_capital

        self.account = Account(
            account_id="MAIN",
            balance=initial_capital,
            available=initial_capital,
            pre_balance=initial_capital
        )

        logger.info(f"账户初始化: 初始资金 {initial_capital:.2f}")

    def _on_account(self, event: Event):
        """处理账户事件"""
        account: Account = event.data
        self.account = account

        # 记录权益历史
        self.equity_history.append((datetime.now(), account.balance))

        # 更新峰值和回撤
        if account.balance > self._peak_equity:
            self._peak_equity = account.balance

        if self._peak_equity > 0:
            drawdown = self._peak_equity - account.balance
            drawdown_pct = drawdown / self._peak_equity

            if drawdown > self._max_drawdown:
                self._max_drawdown = drawdown
            if drawdown_pct > self._max_drawdown_pct:
                self._max_drawdown_pct = drawdown_pct

        # 更新日度记录
        self._update_daily_record()

    def _update_daily_record(self):
        """更新日度记录"""
        if not self.account:
            return

        today = datetime.now().date()

        if today not in self.daily_records:
            # 新的一天
            self.daily_records[today] = DailyRecord(
                date=today,
                opening_balance=self.account.pre_balance
            )

        record = self.daily_records[today]
        record.closing_balance = self.account.balance
        record.realized_pnl = self.account.realized_pnl
        record.commission = self.account.commission
        record.net_pnl = record.closing_balance - record.opening_balance
        record.return_pct = record.net_pnl / record.opening_balance if record.opening_balance > 0 else 0

    def deposit(self, amount: float):
        """入金"""
        if self.account:
            self.account.deposit += amount
            self.account.balance += amount
            self.account.available += amount

            today = datetime.now().date()
            if today in self.daily_records:
                self.daily_records[today].deposit += amount

            logger.info(f"入金: {amount:.2f}, 当前余额: {self.account.balance:.2f}")

    def withdraw(self, amount: float) -> bool:
        """出金"""
        if not self.account:
            return False

        if amount > self.account.available:
            logger.warning(f"出金失败: 可用资金不足 ({self.account.available:.2f} < {amount:.2f})")
            return False

        self.account.withdraw += amount
        self.account.balance -= amount
        self.account.available -= amount

        today = datetime.now().date()
        if today in self.daily_records:
            self.daily_records[today].withdraw += amount

        logger.info(f"出金: {amount:.2f}, 当前余额: {self.account.balance:.2f}")
        return True

    def get_account(self) -> Optional[Account]:
        """获取当前账户"""
        return self.account

    def get_balance(self) -> float:
        """获取余额"""
        return self.account.balance if self.account else 0.0

    def get_available(self) -> float:
        """获取可用资金"""
        return self.account.available if self.account else 0.0

    def get_equity_curve(self) -> List[tuple]:
        """获取权益曲线"""
        return self.equity_history

    def get_daily_records(self) -> List[DailyRecord]:
        """获取日度记录"""
        return sorted(self.daily_records.values(), key=lambda x: x.date)

    def get_performance_metrics(self) -> PerformanceMetrics:
        """计算绩效指标"""
        metrics = PerformanceMetrics()

        if not self.account or self.initial_capital <= 0:
            return metrics

        # 总收益
        metrics.total_return = (self.account.balance - self.initial_capital) / self.initial_capital

        # 交易天数
        metrics.trading_days = len(self.daily_records)

        # 年化收益（假设250个交易日）
        if metrics.trading_days > 0:
            metrics.annual_return = metrics.total_return * (250 / metrics.trading_days)

        # 最大回撤
        metrics.max_drawdown = self._max_drawdown
        metrics.max_drawdown_pct = self._max_drawdown_pct

        # Calmar Ratio
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.annual_return / metrics.max_drawdown_pct

        # Sharpe Ratio (简化计算)
        if len(self.daily_records) > 1:
            returns = [r.return_pct for r in self.daily_records.values()]
            if returns:
                import numpy as np
                avg_return = np.mean(returns)
                std_return = np.std(returns)
                if std_return > 0:
                    # 假设无风险利率3%
                    rf_daily = 0.03 / 250
                    metrics.sharpe_ratio = (avg_return - rf_daily) / std_return * (250 ** 0.5)

                    # Sortino Ratio (仅下行波动)
                    negative_returns = [r for r in returns if r < 0]
                    if negative_returns:
                        downside_std = np.std(negative_returns)
                        if downside_std > 0:
                            metrics.sortino_ratio = (avg_return - rf_daily) / downside_std * (250 ** 0.5)

        return metrics

    def get_summary(self) -> dict:
        """获取账户摘要"""
        if not self.account:
            return {}

        metrics = self.get_performance_metrics()

        return {
            'account_id': self.account.account_id,
            'initial_capital': self.initial_capital,
            'current_balance': self.account.balance,
            'available': self.account.available,
            'margin': self.account.margin,
            'unrealized_pnl': self.account.unrealized_pnl,
            'realized_pnl': self.account.realized_pnl,
            'commission': self.account.commission,
            'risk_ratio': self.account.risk_ratio,

            'total_return': f"{metrics.total_return:.2%}",
            'annual_return': f"{metrics.annual_return:.2%}",
            'max_drawdown': f"{metrics.max_drawdown_pct:.2%}",
            'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
            'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
            'calmar_ratio': f"{metrics.calmar_ratio:.2f}",
            'trading_days': metrics.trading_days,

            'update_time': self.account.update_time.isoformat() if self.account.update_time else None
        }

    def reset(self):
        """重置账户"""
        self.daily_records.clear()
        self.equity_history.clear()
        self._peak_equity = self.initial_capital
        self._max_drawdown = 0.0
        self._max_drawdown_pct = 0.0

        if self.account:
            self.account.balance = self.initial_capital
            self.account.available = self.initial_capital
            self.account.margin = 0.0
            self.account.unrealized_pnl = 0.0
            self.account.realized_pnl = 0.0
            self.account.commission = 0.0
            self.account.pre_balance = self.initial_capital

        logger.info("账户已重置")
