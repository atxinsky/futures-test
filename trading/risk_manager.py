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
import threading
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import (
    OrderRequest, Order, Trade, Position, Account,
    Direction, Offset, EventType, Event
)
from core.event_engine import EventEngine
from trading.position_manager import PositionManager
from utils.limit_price import get_limit_price_manager, LimitPriceManager

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

        # 线程锁 - 保证风控检查原子性
        self._lock = threading.Lock()

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

        # 审计日志文件路径
        self._audit_log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data', 'risk_audit.log'
        )

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

    def _estimate_margin(self, symbol: str, price: float, volume: int) -> float:
        """
        预估新订单所需保证金

        Args:
            symbol: 合约代码
            price: 价格
            volume: 手数

        Returns:
            预估保证金金额
        """
        try:
            from config import get_instrument
            inst = get_instrument(symbol)
            if inst:
                multiplier = inst.get('multiplier', 10)
                margin_rate = inst.get('margin_rate', 0.1)
                return price * multiplier * volume * margin_rate
        except Exception as e:
            logger.warning(f"预估保证金失败: {e}")
        # 默认按10%保证金率估算
        return price * volume * 10 * 0.1

    def check_order(self, request: OrderRequest) -> Tuple[bool, str]:
        """
        订单风控检查（线程安全）

        Args:
            request: 订单请求

        Returns:
            (是否通过, 原因)
        """
        if not self.config.enabled:
            return True, ""

        # 使用锁保证原子性检查
        with self._lock:
            # 1. 检查账户
            if not self.account:
                self._audit_log("ORDER_REJECTED", request, "账户未初始化")
                return False, "账户未初始化"

            # 2. 检查资金
            if self.account.available < self.config.min_available:
                reason = f"可用资金不足: {self.account.available:.2f} < {self.config.min_available}"
                self._audit_log("ORDER_REJECTED", request, reason)
                return False, reason

            # 3. 检查保证金占用（含新订单预估）
            if request.offset == Offset.OPEN:
                # 预估新订单保证金
                estimated_margin = self._estimate_margin(
                    request.symbol,
                    request.price if request.price > 0 else 0,
                    request.volume
                )
                # 计算开仓后的预估保证金比例
                new_margin = self.account.margin + estimated_margin
                new_margin_ratio = new_margin / self.account.balance if self.account.balance > 0 else 1.0

                if new_margin_ratio >= self.config.max_margin_ratio:
                    reason = f"保证金占用将超限: {new_margin_ratio:.2%} >= {self.config.max_margin_ratio:.2%}"
                    self._audit_log("ORDER_REJECTED", request, reason)
                    return False, reason

                # 检查可用资金是否够开仓
                if estimated_margin > self.account.available:
                    reason = f"可用资金不足开仓: 需要 {estimated_margin:.2f}, 可用 {self.account.available:.2f}"
                    self._audit_log("ORDER_REJECTED", request, reason)
                    return False, reason

            # 4. 检查涨跌停限制
            if request.price > 0:  # 有报价时检查
                limit_mgr = get_limit_price_manager()
                is_valid, limit_reason = limit_mgr.check_price_valid(
                    request.symbol,
                    request.price
                )
                if not is_valid:
                    self._audit_log("ORDER_REJECTED", request, f"涨跌停限制: {limit_reason}")
                    return False, f"涨跌停限制: {limit_reason}"

                # 警告接近涨跌停
                info = limit_mgr.get_limit_price(request.symbol)
                if info:
                    # 接近涨停（距离小于1%）
                    if info.limit_up > 0 and request.price > info.limit_up * 0.99:
                        logger.warning(f"[风控] 订单价格接近涨停: {request.symbol} {request.price:.2f} (涨停: {info.limit_up:.2f})")
                    # 接近跌停（距离小于1%）
                    if info.limit_down > 0 and request.price < info.limit_down * 1.01:
                        logger.warning(f"[风控] 订单价格接近跌停: {request.symbol} {request.price:.2f} (跌停: {info.limit_down:.2f})")

            # 5. 检查持仓限制（开仓时）
            if request.offset == Offset.OPEN and self.position_manager:
                # 单品种持仓限制
                current_pos = abs(self.position_manager.get_net_position(request.symbol))
                if current_pos + request.volume > self.config.max_position_per_symbol:
                    reason = f"超过单品种持仓限制: {current_pos} + {request.volume} > {self.config.max_position_per_symbol}"
                    self._audit_log("ORDER_REJECTED", request, reason)
                    return False, reason

                # 总持仓限制
                all_positions = self.position_manager.get_all_positions()
                total_volume = sum(p.volume for p in all_positions)
                if total_volume + request.volume > self.config.max_position_total:
                    reason = f"超过总持仓限制: {total_volume} + {request.volume} > {self.config.max_position_total}"
                    self._audit_log("ORDER_REJECTED", request, reason)
                    return False, reason

            # 5. 检查日亏损限制
            if self._daily_pnl < 0:
                daily_loss_ratio = abs(self._daily_pnl) / self._initial_equity if self._initial_equity > 0 else 0
                if daily_loss_ratio >= self.config.max_daily_loss_ratio:
                    if request.offset == Offset.OPEN:
                        reason = f"日亏损超限: {daily_loss_ratio:.2%} >= {self.config.max_daily_loss_ratio:.2%}"
                        self._audit_log("ORDER_REJECTED", request, reason)
                        return False, reason

            # 6. 检查连续亏损
            if self._consecutive_losses >= self.config.max_consecutive_losses:
                if request.offset == Offset.OPEN and not self.config.allow_open_when_risk:
                    reason = f"连续亏损次数超限: {self._consecutive_losses} >= {self.config.max_consecutive_losses}"
                    self._audit_log("ORDER_REJECTED", request, reason)
                    return False, reason

            # 7. 检查回撤
            if self._peak_equity > 0 and self.account.balance < self._peak_equity:
                drawdown = (self._peak_equity - self.account.balance) / self._peak_equity
                if drawdown >= self.config.max_drawdown_ratio:
                    if request.offset == Offset.OPEN:
                        reason = f"回撤超限: {drawdown:.2%} >= {self.config.max_drawdown_ratio:.2%}"
                        self._audit_log("ORDER_REJECTED", request, reason)
                        return False, reason

            # 通过所有检查
            self._audit_log("ORDER_PASSED", request, "风控检查通过")
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
            # 从 trade 对象获取实现盈亏
            realized_pnl = getattr(trade, 'realized_pnl', 0.0)

            if realized_pnl >= 0:
                self._daily_wins += 1
                self._consecutive_losses = 0
                logger.info(f"[风控] 盈利交易: {trade.symbol} +{realized_pnl:.2f}")
            else:
                self._daily_losses += 1
                self._consecutive_losses += 1
                logger.warning(f"[风控] 亏损交易: {trade.symbol} {realized_pnl:.2f}, 连续亏损: {self._consecutive_losses}")

            # 记录审计日志
            self._audit_log("TRADE_CLOSED", trade, f"盈亏: {realized_pnl:.2f}")

            # 检查是否需要强平
            if self._consecutive_losses >= self.config.max_consecutive_losses:
                self._audit_log("RISK_ALERT", None, f"连续亏损达到{self._consecutive_losses}次，触发风控警告")

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

    def _audit_log(self, event_type: str, data: any, message: str = ""):
        """
        风控审计日志

        Args:
            event_type: 事件类型 (ORDER_REJECTED, ORDER_PASSED, TRADE_CLOSED, RISK_ALERT)
            data: 相关数据对象
            message: 附加消息
        """
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'message': message,
                'risk_level': self.status.risk_level,
                'consecutive_losses': self._consecutive_losses,
                'daily_pnl': self._daily_pnl
            }

            # 添加数据详情
            if data:
                if hasattr(data, 'symbol'):
                    log_entry['symbol'] = data.symbol
                if hasattr(data, 'volume'):
                    log_entry['volume'] = data.volume
                if hasattr(data, 'price'):
                    log_entry['price'] = data.price
                if hasattr(data, 'direction'):
                    log_entry['direction'] = str(data.direction)

            # 写入日志文件
            os.makedirs(os.path.dirname(self._audit_log_path), exist_ok=True)
            with open(self._audit_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

            # 同时输出到logger
            if event_type in ('ORDER_REJECTED', 'RISK_ALERT'):
                logger.warning(f"[RISK_AUDIT] {event_type}: {message}")
            else:
                logger.info(f"[RISK_AUDIT] {event_type}: {message}")

        except Exception as e:
            logger.error(f"审计日志写入失败: {e}")

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
