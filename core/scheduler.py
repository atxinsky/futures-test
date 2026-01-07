# coding=utf-8
"""
定时调度器
支持基于cron表达式和固定间隔的任务调度
"""

from typing import Dict, List, Callable, Optional, Any
from datetime import datetime, time, timedelta
from threading import Thread, Event
from dataclasses import dataclass, field
import logging
import time as time_module
import re

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """调度任务"""
    task_id: str
    name: str
    handler: Callable
    cron: str = ""           # cron表达式
    interval: int = 0        # 间隔秒数（与cron二选一）
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    error_count: int = 0
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)


class CronParser:
    """
    简化的Cron解析器
    支持格式: 分 时 日 月 周
    示例:
        "0 9 * * 1-5"  每个工作日9:00
        "*/5 * * * *"  每5分钟
        "30 14 * * *"  每天14:30
    """

    @staticmethod
    def parse(cron_expr: str) -> dict:
        """解析cron表达式"""
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            raise ValueError(f"无效的cron表达式: {cron_expr}")

        return {
            'minute': CronParser._parse_field(parts[0], 0, 59),
            'hour': CronParser._parse_field(parts[1], 0, 23),
            'day': CronParser._parse_field(parts[2], 1, 31),
            'month': CronParser._parse_field(parts[3], 1, 12),
            'weekday': CronParser._parse_field(parts[4], 0, 6)  # 0=周一, 6=周日
        }

    @staticmethod
    def _parse_field(field: str, min_val: int, max_val: int) -> List[int]:
        """解析单个字段"""
        if field == '*':
            return list(range(min_val, max_val + 1))

        result = set()

        for part in field.split(','):
            # 处理步长 */n 或 start-end/n
            if '/' in part:
                range_part, step = part.split('/')
                step = int(step)
            else:
                range_part = part
                step = 1

            # 处理范围
            if range_part == '*':
                values = range(min_val, max_val + 1, step)
            elif '-' in range_part:
                start, end = map(int, range_part.split('-'))
                values = range(start, end + 1, step)
            else:
                values = [int(range_part)]

            result.update(values)

        return sorted(result)

    @staticmethod
    def get_next_run(cron_expr: str, from_time: datetime = None) -> datetime:
        """计算下次运行时间"""
        if from_time is None:
            from_time = datetime.now()

        parsed = CronParser.parse(cron_expr)

        # 从下一分钟开始检查
        current = from_time.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # 最多检查一年
        max_iterations = 525600  # 一年的分钟数

        for _ in range(max_iterations):
            if (current.minute in parsed['minute'] and
                current.hour in parsed['hour'] and
                current.day in parsed['day'] and
                current.month in parsed['month'] and
                current.weekday() in parsed['weekday']):
                return current

            current += timedelta(minutes=1)

        raise ValueError(f"无法计算下次运行时间: {cron_expr}")


class Scheduler:
    """
    定时调度器

    功能:
    1. Cron表达式调度
    2. 固定间隔调度
    3. 交易时段检查
    4. 任务管理
    """

    # 期货交易时段（简化版）
    TRADING_SESSIONS = {
        'day': [
            (time(9, 0), time(10, 15)),
            (time(10, 30), time(11, 30)),
            (time(13, 30), time(15, 0)),
        ],
        'night': [
            (time(21, 0), time(23, 59)),
            (time(0, 0), time(2, 30)),  # 夜盘跨日
        ]
    }

    def __init__(self):
        self.tasks: Dict[str, ScheduledTask] = {}
        self._running = False
        self._thread: Optional[Thread] = None
        self._stop_event = Event()

        # 任务计数器
        self._task_counter = 0

    def add_task(
        self,
        name: str,
        handler: Callable,
        cron: str = "",
        interval: int = 0,
        args: tuple = (),
        kwargs: dict = None
    ) -> str:
        """
        添加调度任务

        Args:
            name: 任务名称
            handler: 处理函数
            cron: cron表达式（与interval二选一）
            interval: 间隔秒数
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            任务ID
        """
        if not cron and not interval:
            raise ValueError("必须指定cron或interval")

        self._task_counter += 1
        task_id = f"task_{self._task_counter}"

        task = ScheduledTask(
            task_id=task_id,
            name=name,
            handler=handler,
            cron=cron,
            interval=interval,
            args=args,
            kwargs=kwargs or {}
        )

        # 计算下次运行时间
        if cron:
            task.next_run = CronParser.get_next_run(cron)
        else:
            task.next_run = datetime.now() + timedelta(seconds=interval)

        self.tasks[task_id] = task
        logger.info(f"添加调度任务: {name} ({task_id}), 下次运行: {task.next_run}")

        return task_id

    def remove_task(self, task_id: str):
        """移除任务"""
        if task_id in self.tasks:
            task = self.tasks.pop(task_id)
            logger.info(f"移除调度任务: {task.name} ({task_id})")

    def enable_task(self, task_id: str):
        """启用任务"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = True

    def disable_task(self, task_id: str):
        """禁用任务"""
        if task_id in self.tasks:
            self.tasks[task_id].enabled = False

    def get_task(self, task_id: str) -> Optional[ScheduledTask]:
        """获取任务"""
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[dict]:
        """获取所有任务信息"""
        return [{
            'task_id': t.task_id,
            'name': t.name,
            'cron': t.cron,
            'interval': t.interval,
            'enabled': t.enabled,
            'last_run': t.last_run.isoformat() if t.last_run else None,
            'next_run': t.next_run.isoformat() if t.next_run else None,
            'run_count': t.run_count,
            'error_count': t.error_count
        } for t in self.tasks.values()]

    def start(self):
        """启动调度器"""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = Thread(target=self._run_loop, daemon=True)
        self._thread.start()

        logger.info("调度器启动")

    def stop(self):
        """停止调度器"""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._thread:
            self._thread.join(timeout=5)

        logger.info("调度器停止")

    def _run_loop(self):
        """调度循环"""
        while self._running and not self._stop_event.is_set():
            now = datetime.now()

            for task in list(self.tasks.values()):
                if not task.enabled:
                    continue

                if task.next_run and now >= task.next_run:
                    self._execute_task(task)

                    # 更新下次运行时间
                    if task.cron:
                        task.next_run = CronParser.get_next_run(task.cron, now)
                    else:
                        task.next_run = now + timedelta(seconds=task.interval)

            # 每秒检查一次
            self._stop_event.wait(1)

    def _execute_task(self, task: ScheduledTask):
        """执行任务"""
        try:
            logger.debug(f"执行任务: {task.name}")
            task.handler(*task.args, **task.kwargs)
            task.last_run = datetime.now()
            task.run_count += 1
        except Exception as e:
            task.error_count += 1
            logger.error(f"任务执行失败: {task.name} - {e}")

    @staticmethod
    def is_trading_time(dt: datetime = None, include_night: bool = True) -> bool:
        """
        检查是否在交易时段

        Args:
            dt: 检查时间，默认当前时间
            include_night: 是否包含夜盘

        Returns:
            是否在交易时段
        """
        if dt is None:
            dt = datetime.now()

        t = dt.time()
        weekday = dt.weekday()

        # 周末不交易
        if weekday >= 5:  # 5=周六, 6=周日
            return False

        # 检查日盘
        for start, end in Scheduler.TRADING_SESSIONS['day']:
            if start <= t <= end:
                return True

        # 检查夜盘
        if include_night:
            for start, end in Scheduler.TRADING_SESSIONS['night']:
                if start <= t or t <= end:
                    # 周五夜盘检查
                    if weekday == 4 and t >= time(21, 0):
                        return True
                    # 其他工作日夜盘
                    elif weekday < 4:
                        return True

        return False

    @staticmethod
    def get_next_trading_time(dt: datetime = None) -> datetime:
        """获取下一个交易时段开始时间"""
        if dt is None:
            dt = datetime.now()

        # 简化实现：返回下一个9:00
        next_dt = dt.replace(hour=9, minute=0, second=0, microsecond=0)

        if dt.time() >= time(9, 0):
            next_dt += timedelta(days=1)

        # 跳过周末
        while next_dt.weekday() >= 5:
            next_dt += timedelta(days=1)

        return next_dt


# ============ 预定义任务 ============

class TradingScheduler(Scheduler):
    """
    交易专用调度器

    预定义常用交易任务
    """

    def __init__(self, engine=None):
        super().__init__()
        self.engine = engine

    def add_market_open_task(self, handler: Callable, session: str = "day"):
        """添加开盘任务"""
        if session == "day":
            cron = "0 9 * * 1-5"  # 工作日9:00
        else:
            cron = "0 21 * * 1-5"  # 工作日21:00（夜盘）

        return self.add_task(f"market_open_{session}", handler, cron=cron)

    def add_market_close_task(self, handler: Callable, session: str = "day"):
        """添加收盘任务"""
        if session == "day":
            cron = "0 15 * * 1-5"  # 工作日15:00
        else:
            cron = "30 2 * * 2-6"  # 周二到周六凌晨2:30

        return self.add_task(f"market_close_{session}", handler, cron=cron)

    def add_bar_task(self, handler: Callable, interval_minutes: int = 1):
        """添加K线周期任务"""
        interval = interval_minutes * 60
        return self.add_task(f"bar_{interval_minutes}m", handler, interval=interval)

    def add_risk_check_task(self, handler: Callable, interval_seconds: int = 30):
        """添加风控检查任务"""
        return self.add_task("risk_check", handler, interval=interval_seconds)

    def add_daily_report_task(self, handler: Callable):
        """添加日报任务"""
        cron = "30 15 * * 1-5"  # 工作日15:30
        return self.add_task("daily_report", handler, cron=cron)

    def add_position_sync_task(self, handler: Callable, interval_seconds: int = 60):
        """添加持仓同步任务"""
        return self.add_task("position_sync", handler, interval=interval_seconds)
