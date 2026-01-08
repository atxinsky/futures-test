# coding=utf-8
"""
对账服务模块
自动同步本地持仓与柜台持仓，防止状态不一致
"""

import logging
import threading
import schedule
from datetime import datetime, time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """对账结果"""
    success: bool
    reconcile_time: datetime
    local_positions: int = 0
    broker_positions: int = 0
    matched: int = 0
    mismatched: int = 0
    missing_local: int = 0   # 柜台有但本地没有
    missing_broker: int = 0  # 本地有但柜台没有
    details: List[dict] = field(default_factory=list)
    error_message: str = ""


class ReconciliationService:
    """
    对账服务

    功能:
    1. 定时自动对账（盘前 08:50, 20:50）
    2. 手动触发对账
    3. 发现差异时自动修正或告警
    4. 记录对账日志
    """

    # 对账时间点（盘前）
    RECONCILE_TIMES = [
        time(8, 50),   # 日盘开盘前
        time(20, 50),  # 夜盘开盘前
    ]

    def __init__(self, position_manager, gateway=None):
        """
        初始化对账服务

        Args:
            position_manager: 持仓管理器
            gateway: 交易网关（用于查询柜台持仓）
        """
        self.position_manager = position_manager
        self.gateway = gateway

        self._running = False
        self._scheduler_thread = None
        self._lock = threading.Lock()

        # 对账结果历史
        self._history: List[ReconciliationResult] = []
        self._max_history = 100

        # 回调函数
        self._on_mismatch_callback: Optional[Callable] = None
        self._on_complete_callback: Optional[Callable] = None

        # 自动修正模式
        self.auto_fix = True  # 发现差异时是否自动以柜台为准

    def set_gateway(self, gateway):
        """设置交易网关"""
        self.gateway = gateway

    def set_on_mismatch(self, callback: Callable):
        """设置差异回调"""
        self._on_mismatch_callback = callback

    def set_on_complete(self, callback: Callable):
        """设置完成回调"""
        self._on_complete_callback = callback

    def start(self):
        """启动定时对账服务"""
        if self._running:
            logger.warning("对账服务已在运行")
            return

        self._running = True

        # 设置定时任务
        for t in self.RECONCILE_TIMES:
            schedule.every().day.at(t.strftime("%H:%M")).do(self.reconcile)
            logger.info(f"设置对账时间: {t.strftime('%H:%M')}")

        # 启动调度线程
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            daemon=True,
            name="ReconciliationScheduler"
        )
        self._scheduler_thread.start()

        logger.info("对账服务已启动")

    def stop(self):
        """停止对账服务"""
        self._running = False
        schedule.clear()
        logger.info("对账服务已停止")

    def _run_scheduler(self):
        """运行调度器"""
        while self._running:
            schedule.run_pending()
            threading.Event().wait(30)  # 每30秒检查一次

    def reconcile(self) -> ReconciliationResult:
        """
        执行对账

        Returns:
            对账结果
        """
        logger.info("开始执行对账...")

        result = ReconciliationResult(
            success=False,
            reconcile_time=datetime.now()
        )

        try:
            # 1. 获取本地持仓
            local_positions = self._get_local_positions()
            result.local_positions = len(local_positions)

            # 2. 查询柜台持仓
            if not self.gateway:
                result.error_message = "网关未设置，无法查询柜台持仓"
                logger.error(result.error_message)
                return result

            broker_positions = self._query_broker_positions()
            if broker_positions is None:
                result.error_message = "查询柜台持仓失败"
                logger.error(result.error_message)
                return result

            result.broker_positions = len(broker_positions)

            # 3. 对比差异
            differences = self._compare_positions(local_positions, broker_positions)

            result.matched = differences['matched']
            result.mismatched = differences['mismatched']
            result.missing_local = differences['missing_local']
            result.missing_broker = differences['missing_broker']
            result.details = differences['details']

            # 4. 处理差异
            if result.mismatched > 0 or result.missing_local > 0 or result.missing_broker > 0:
                logger.warning(f"对账发现差异: 不匹配={result.mismatched}, "
                              f"本地缺失={result.missing_local}, 柜台缺失={result.missing_broker}")

                # 回调通知
                if self._on_mismatch_callback:
                    self._on_mismatch_callback(result)

                # 自动修正
                if self.auto_fix:
                    self._fix_positions(broker_positions)
                    logger.info("已自动修正本地持仓为柜台状态")

            result.success = True
            logger.info(f"对账完成: 本地={result.local_positions}, 柜台={result.broker_positions}, "
                       f"匹配={result.matched}")

        except Exception as e:
            result.error_message = str(e)
            logger.error(f"对账异常: {e}")

        # 记录历史
        with self._lock:
            self._history.append(result)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # 完成回调
        if self._on_complete_callback:
            self._on_complete_callback(result)

        return result

    def _get_local_positions(self) -> Dict[str, dict]:
        """
        获取本地持仓

        Returns:
            {symbol_direction: position_data}
        """
        positions = {}
        for pos in self.position_manager.get_all_positions():
            key = f"{pos.symbol}_{pos.direction.name if hasattr(pos.direction, 'name') else pos.direction}"
            positions[key] = {
                'symbol': pos.symbol,
                'direction': pos.direction,
                'volume': pos.volume,
                'avg_price': pos.avg_price
            }
        return positions

    def _query_broker_positions(self) -> Optional[Dict[str, dict]]:
        """
        查询柜台持仓

        Returns:
            {symbol_direction: position_data} 或 None
        """
        try:
            # 调用网关查询持仓
            broker_list = self.gateway.query_position()
            if broker_list is None:
                return None

            positions = {}
            for pos in broker_list:
                # 统一处理Position对象或字典
                if hasattr(pos, 'symbol'):
                    symbol = pos.symbol
                    direction = pos.direction
                    volume = pos.volume
                    avg_price = pos.avg_price
                else:
                    symbol = pos.get('symbol')
                    direction = pos.get('direction')
                    volume = pos.get('volume', 0)
                    avg_price = pos.get('avg_price', 0)

                if volume > 0:
                    dir_name = direction.name if hasattr(direction, 'name') else str(direction)
                    key = f"{symbol}_{dir_name}"
                    positions[key] = {
                        'symbol': symbol,
                        'direction': direction,
                        'volume': volume,
                        'avg_price': avg_price
                    }

            return positions

        except Exception as e:
            logger.error(f"查询柜台持仓失败: {e}")
            return None

    def _compare_positions(
        self,
        local: Dict[str, dict],
        broker: Dict[str, dict]
    ) -> dict:
        """
        对比持仓差异

        Returns:
            差异统计
        """
        result = {
            'matched': 0,
            'mismatched': 0,
            'missing_local': 0,
            'missing_broker': 0,
            'details': []
        }

        all_keys = set(local.keys()) | set(broker.keys())

        for key in all_keys:
            local_pos = local.get(key)
            broker_pos = broker.get(key)

            if local_pos and broker_pos:
                # 两边都有，检查数量是否一致
                if local_pos['volume'] == broker_pos['volume']:
                    result['matched'] += 1
                else:
                    result['mismatched'] += 1
                    result['details'].append({
                        'key': key,
                        'type': 'mismatch',
                        'local_volume': local_pos['volume'],
                        'broker_volume': broker_pos['volume'],
                        'local_price': local_pos['avg_price'],
                        'broker_price': broker_pos['avg_price']
                    })

            elif broker_pos and not local_pos:
                # 柜台有但本地没有
                result['missing_local'] += 1
                result['details'].append({
                    'key': key,
                    'type': 'missing_local',
                    'broker_volume': broker_pos['volume'],
                    'broker_price': broker_pos['avg_price']
                })

            elif local_pos and not broker_pos:
                # 本地有但柜台没有（可能是私下平仓或强平）
                result['missing_broker'] += 1
                result['details'].append({
                    'key': key,
                    'type': 'missing_broker',
                    'local_volume': local_pos['volume'],
                    'local_price': local_pos['avg_price']
                })

        return result

    def _fix_positions(self, broker_positions: Dict[str, dict]):
        """
        以柜台持仓为准修正本地持仓

        Args:
            broker_positions: 柜台持仓
        """
        from models.base import Position, Direction

        # 清空本地持仓
        with self.position_manager._lock:
            self.position_manager.positions.clear()

            # 按柜台数据重建
            for key, pos_data in broker_positions.items():
                symbol = pos_data['symbol']
                direction = pos_data['direction']

                # 确保direction是Direction枚举
                if not isinstance(direction, Direction):
                    if str(direction).upper() in ('LONG', '1'):
                        direction = Direction.LONG
                    else:
                        direction = Direction.SHORT

                pos = Position(
                    symbol=symbol,
                    direction=direction,
                    volume=pos_data['volume'],
                    avg_price=pos_data['avg_price']
                )

                self.position_manager.positions[symbol][direction] = pos

        # 保存到数据库
        self.position_manager.save_all_to_db()

        logger.info(f"已修正本地持仓，共 {len(broker_positions)} 个")

    def get_history(self, limit: int = 10) -> List[ReconciliationResult]:
        """获取对账历史"""
        with self._lock:
            return list(self._history[-limit:])

    def get_last_result(self) -> Optional[ReconciliationResult]:
        """获取最近一次对账结果"""
        with self._lock:
            if self._history:
                return self._history[-1]
            return None


# 全局单例
_reconciliation_service: Optional[ReconciliationService] = None


def get_reconciliation_service(
    position_manager=None,
    gateway=None
) -> ReconciliationService:
    """获取对账服务单例"""
    global _reconciliation_service
    if _reconciliation_service is None:
        if position_manager is None:
            raise ValueError("首次获取需要提供 position_manager")
        _reconciliation_service = ReconciliationService(position_manager, gateway)
    return _reconciliation_service
