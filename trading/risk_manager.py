# coding=utf-8
"""
风险管理器
负责交易前风控检查和运行时风险监控
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from collections import defaultdict
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import (
    OrderRequest, Order, Trade, Position, Account,
    Direction, Offset, EventType, Event
)
from core.event_engine import EventEngine
from trading.position_manager import PositionManager

logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """风控配置"""
    # 持仓限制
    max_position_per_symbol: int = 10           # 单品种最大持仓
    max_position_total: int = 50                # 总最大持仓
    max_order_per_symbol: int = 5               # 单品种最大挂单

    # 资金风控
    max_margin_ratio: float = 0.8               # 最大保证金占用比例
    max_order_value_ratio: float = 0.3          # 单笔订单最大资金占用比例
    min_available: float = 10000                # 最小可用资金

    # 亏损控制
    max_daily_loss_ratio: float = 0.05          # 日最大亏损比例
    max_drawdown_ratio: float = 0.15            # 最大回撤比例
    max_consecutive_losses: int = 5             # 最大连续亏损次数

    # 止损设置
    default_stop_loss_ratio: float = 0.03       # 默认止损比例
    stop_loss_atr_mult: float = 3.0             # ATR止损倍数

    # 开关
    enabled: bool = True                        # 是否启用风控
    allow_open_when_risk: bool = False          # 风险时是否允许开仓
    force_close_on_max_loss: bool = True        # 达到最大亏损时是否强平


@dataclass
class RiskStatus:
    """风险状态"""
    is_safe: bool = True
    risk_level: str = "low"  # low, medium, high, critical
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # 统计数据
    margin_ratio: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_ratio: float = 0.0
    drawdown_ratio: float = 0.0
    consecutive_losses: int = 0

    update_time: datetime = field(default_factory=datetime.now)


class RiskManager:
    """
    风险管理器

    功能:
    1. 订单前风控检查
    2. 持仓风险监控
    3. 账户风险监控
    4. 止损管理
    """

    def __init__(self, event_engine: EventEngine, config: RiskConfig = None):
        self.event_engine = event_engine
        self.config = config or RiskConfig()

        # 状态
        self.status = RiskStatus()

        # 账户和持仓引用
        self.account: Optional[Account] = None
        self.position_manager: Optional[PositionManager] = None

        # 日内统计
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._daily_wins: int = 0
        self._daily_losses: int = 0
        self._consecutive_losses: int = 0
        self._last_trade_date: date = None

        # 历史峰值
        self._peak_equity: float = 0.0
        self._initial_equity: float = 0.0

        # 注册事件
        self.event_engine.register(EventType.TRADE, self._on_trade)
        self.event_engine.register(EventType.ACCOUNT, self._on_account)

    def set_account(self, account: Account):
        """设置账户"""
        self.account = account
        if self._initial_equity == 0:
            self._initial_equity = account.balance
            self._peak_equity = account.balance

    def set_position_manager(self, pm: PositionManager):
        """设置持仓管理器"""
        self.position_manager = pm

    def check_order(self, request: OrderRequest) -> Tuple[bool, str]:
        """
        订单风控检查

        Args:
            request: 订单请求

        Returns:
            (是否通过, 原因)
        """
        if not self.config.enabled:
            return True, ""

        # 1. 检查账户
        if not self.account:
            return False, "账户未初始化"

        # 2. 检查资金
        if self.account.available < self.config.min_available:
            return False, f"可用资金不足: {self.account.available:.2f} < {self.config.min_available}"

        # 3. 检查保证金占用
        if self.account.risk_ratio >= self.config.max_margin_ratio:
            if request.offset == Offset.OPEN:
                return False, f"保证金占用过高: {self.account.risk_ratio:.2%} >= {self.config.max_margin_ratio:.2%}"

        # 4. 检查持仓限制（开仓时）
        if request.offset == Offset.OPEN and self.position_manager:
            # 单品种持仓限制
            current_pos = abs(self.position_manager.get_net_position(request.symbol))
            if current_pos + request.volume > self.config.max_position_per_symbol:
                return False, f"超过单品种持仓限制: {current_pos} + {request.volume} > {self.config.max_position_per_symbol}"

            # 总持仓限制
            all_positions = self.position_manager.get_all_positions()
            total_volume = sum(p.volume for p in all_positions)
            if total_volume + request.volume > self.config.max_position_total:
                return False, f"超过总持仓限制: {total_volume} + {request.volume} > {self.config.max_position_total}"

        # 5. 检查日亏损限制
        if self._daily_pnl < 0:
            daily_loss_ratio = abs(self._daily_pnl) / self._initial_equity if self._initial_equity > 0 else 0
            if daily_loss_ratio >= self.config.max_daily_loss_ratio:
                if request.offset == Offset.OPEN:
                    return False, f"日亏损超限: {daily_loss_ratio:.2%} >= {self.config.max_daily_loss_ratio:.2%}"

        # 6. 检查连续亏损
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            if request.offset == Offset.OPEN and not self.config.allow_open_when_risk:
                return False, f"连续亏损次数超限: {self._consecutive_losses} >= {self.config.max_consecutive_losses}"

        # 7. 检查回撤
        if self._peak_equity > 0 and self.account.balance < self._peak_equity:
            drawdown = (self._peak_equity - self.account.balance) / self._peak_equity
            if drawdown >= self.config.max_drawdown_ratio:
                if request.offset == Offset.OPEN:
                    return False, f"回撤超限: {drawdown:.2%} >= {self.config.max_drawdown_ratio:.2%}"

        return True, ""

    def _on_trade(self, event: Event):
        """处理成交事件"""
        trade: Trade = event.data

        # 更新日期
        today = datetime.now().date()
        if self._last_trade_date != today:
            self._reset_daily_stats()
            self._last_trade_date = today

        self._daily_trades += 1

        # 统计盈亏（仅平仓有意义）
        if trade.offset != Offset.OPEN:
            # 需要从position_manager获取盈亏
            # 这里简化处理，假设成交后会更新account
            pass

    def _on_account(self, event: Event):
        """处理账户事件"""
        account: Account = event.data
        self.account = account

        # 更新峰值
        if account.balance > self._peak_equity:
            self._peak_equity = account.balance

        # 更新日盈亏
        self._daily_pnl = account.realized_pnl + account.unrealized_pnl

        # 更新风险状态
        self._update_status()

    def _reset_daily_stats(self):
        """重置日内统计"""
        self._daily_pnl = 0.0
        self._daily_trades = 0
        self._daily_wins = 0
        self._daily_losses = 0

    def _update_status(self):
        """更新风险状态"""
        self.status.warnings.clear()
        self.status.errors.clear()
        self.status.is_safe = True
        self.status.risk_level = "low"

        if not self.account:
            return

        # 计算指标
        self.status.margin_ratio = self.account.risk_ratio
        self.status.daily_pnl = self._daily_pnl
        self.status.daily_pnl_ratio = self._daily_pnl / self._initial_equity if self._initial_equity > 0 else 0
        self.status.consecutive_losses = self._consecutive_losses

        if self._peak_equity > 0:
            self.status.drawdown_ratio = (self._peak_equity - self.account.balance) / self._peak_equity
        else:
            self.status.drawdown_ratio = 0

        # 检查风险等级
        # 保证金占用
        if self.status.margin_ratio >= self.config.max_margin_ratio:
            self.status.errors.append(f"保证金占用过高: {self.status.margin_ratio:.2%}")
            self.status.risk_level = "critical"
            self.status.is_safe = False
        elif self.status.margin_ratio >= self.config.max_margin_ratio * 0.8:
            self.status.warnings.append(f"保证金占用较高: {self.status.margin_ratio:.2%}")
            self.status.risk_level = "high"

        # 日亏损
        if self.status.daily_pnl_ratio <= -self.config.max_daily_loss_ratio:
            self.status.errors.append(f"日亏损超限: {self.status.daily_pnl_ratio:.2%}")
            self.status.risk_level = "critical"
            self.status.is_safe = False
        elif self.status.daily_pnl_ratio <= -self.config.max_daily_loss_ratio * 0.8:
            self.status.warnings.append(f"日亏损较高: {self.status.daily_pnl_ratio:.2%}")
            if self.status.risk_level == "low":
                self.status.risk_level = "medium"

        # 回撤
        if self.status.drawdown_ratio >= self.config.max_drawdown_ratio:
            self.status.errors.append(f"回撤超限: {self.status.drawdown_ratio:.2%}")
            self.status.risk_level = "critical"
            self.status.is_safe = False
        elif self.status.drawdown_ratio >= self.config.max_drawdown_ratio * 0.7:
            self.status.warnings.append(f"回撤较大: {self.status.drawdown_ratio:.2%}")
            if self.status.risk_level == "low":
                self.status.risk_level = "medium"

        # 连续亏损
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            self.status.warnings.append(f"连续亏损: {self._consecutive_losses}次")
            if self.status.risk_level == "low":
                self.status.risk_level = "high"

        self.status.update_time = datetime.now()

    def get_status(self) -> RiskStatus:
        """获取风险状态"""
        self._update_status()
        return self.status

    def get_risk_report(self) -> dict:
        """获取风险报告"""
        status = self.get_status()

        return {
            'is_safe': status.is_safe,
            'risk_level': status.risk_level,
            'warnings': status.warnings,
            'errors': status.errors,
            'metrics': {
                'margin_ratio': f"{status.margin_ratio:.2%}",
                'daily_pnl': f"{status.daily_pnl:.2f}",
                'daily_pnl_ratio': f"{status.daily_pnl_ratio:.2%}",
                'drawdown_ratio': f"{status.drawdown_ratio:.2%}",
                'consecutive_losses': status.consecutive_losses
            },
            'limits': {
                'max_margin_ratio': f"{self.config.max_margin_ratio:.2%}",
                'max_daily_loss_ratio': f"{self.config.max_daily_loss_ratio:.2%}",
                'max_drawdown_ratio': f"{self.config.max_drawdown_ratio:.2%}",
                'max_consecutive_losses': self.config.max_consecutive_losses
            },
            'update_time': status.update_time.isoformat()
        }

    def record_trade_result(self, is_win: bool):
        """记录交易结果"""
        if is_win:
            self._daily_wins += 1
            self._consecutive_losses = 0
        else:
            self._daily_losses += 1
            self._consecutive_losses += 1

    def should_force_close(self) -> bool:
        """是否应该强制平仓"""
        if not self.config.force_close_on_max_loss:
            return False

        status = self.get_status()
        return status.risk_level == "critical"

    def calculate_position_size(self,
                                 capital: float,
                                 price: float,
                                 stop_loss: float,
                                 multiplier: float = 10,
                                 risk_per_trade: float = 0.02) -> int:
        """
        计算仓位大小

        Args:
            capital: 可用资金
            price: 入场价格
            stop_loss: 止损价格
            multiplier: 合约乘数
            risk_per_trade: 单笔风险比例

        Returns:
            建议手数
        """
        if stop_loss <= 0 or price <= 0:
            return 1

        # 计算止损距离
        stop_distance = abs(price - stop_loss)
        if stop_distance == 0:
            return 1

        # 计算风险金额
        risk_amount = capital * risk_per_trade

        # 计算手数
        volume = risk_amount / (stop_distance * multiplier)

        # 向下取整
        volume = int(volume)

        # 最小1手
        return max(1, volume)
