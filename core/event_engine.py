# coding=utf-8
"""
事件驱动引擎
核心组件，负责事件的分发和处理
"""

import asyncio
import threading
from queue import Queue, Empty
from collections import defaultdict
from typing import Callable, Dict, List, Any
from datetime import datetime
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import Event, EventType

logger = logging.getLogger(__name__)


class EventEngine:
    """
    事件驱动引擎

    功能:
    1. 事件队列管理
    2. 事件处理器注册
    3. 事件分发
    4. 定时器支持
    """

    def __init__(self):
        self._queue: Queue = Queue()
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._general_handlers: List[Callable] = []

        self._active: bool = False
        self._thread: threading.Thread = None
        self._timer_thread: threading.Thread = None

        self._timer_handlers: Dict[int, List[Callable]] = defaultdict(list)
        self._timer_interval: int = 1  # 定时器间隔（秒）

    def register(self, event_type: EventType, handler: Callable):
        """
        注册事件处理器

        Args:
            event_type: 事件类型
            handler: 处理函数，接收Event参数
        """
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)
            logger.debug(f"注册事件处理器: {event_type.value} -> {handler.__name__}")

    def unregister(self, event_type: EventType, handler: Callable):
        """注销事件处理器"""
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def register_general(self, handler: Callable):
        """注册通用处理器（接收所有事件）"""
        if handler not in self._general_handlers:
            self._general_handlers.append(handler)

    def register_timer(self, interval: int, handler: Callable):
        """
        注册定时器处理器

        Args:
            interval: 间隔秒数
            handler: 处理函数
        """
        if handler not in self._timer_handlers[interval]:
            self._timer_handlers[interval].append(handler)

    def put(self, event: Event):
        """发送事件到队列"""
        self._queue.put(event)

    def emit(self, event_type: EventType, data: Any = None):
        """便捷方法：发送事件"""
        event = Event(type=event_type, data=data)
        self.put(event)

    def start(self):
        """启动事件引擎"""
        if self._active:
            return

        self._active = True

        # 启动事件处理线程
        self._thread = threading.Thread(target=self._run, name="EventEngine")
        self._thread.daemon = True
        self._thread.start()

        # 启动定时器线程
        if self._timer_handlers:
            self._timer_thread = threading.Thread(target=self._run_timer, name="EventTimer")
            self._timer_thread.daemon = True
            self._timer_thread.start()

        logger.info("事件引擎已启动")

    def stop(self):
        """停止事件引擎"""
        self._active = False

        if self._thread:
            self._thread.join(timeout=2)
        if self._timer_thread:
            self._timer_thread.join(timeout=2)

        logger.info("事件引擎已停止")

    def _run(self):
        """事件处理循环"""
        while self._active:
            try:
                event = self._queue.get(timeout=1)
                self._process(event)
            except Empty:
                continue
            except Exception as e:
                logger.error(f"事件处理异常: {e}", exc_info=True)

    def _process(self, event: Event):
        """处理单个事件"""
        # 调用特定类型处理器
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"处理器 {handler.__name__} 异常: {e}", exc_info=True)

        # 调用通用处理器
        for handler in self._general_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"通用处理器 {handler.__name__} 异常: {e}", exc_info=True)

    def _run_timer(self):
        """定时器循环"""
        counter = 0
        while self._active:
            try:
                asyncio.run(asyncio.sleep(self._timer_interval))
                counter += 1

                for interval, handlers in self._timer_handlers.items():
                    if counter % interval == 0:
                        for handler in handlers:
                            try:
                                event = Event(type=EventType.TIMER, data={'interval': interval})
                                handler(event)
                            except Exception as e:
                                logger.error(f"定时器处理器异常: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"定时器异常: {e}", exc_info=True)

    @property
    def is_active(self) -> bool:
        return self._active


class AsyncEventEngine:
    """
    异步事件驱动引擎
    用于需要asyncio的场景
    """

    def __init__(self):
        self._queue: asyncio.Queue = None
        self._handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._active: bool = False
        self._loop: asyncio.AbstractEventLoop = None

    async def start(self):
        """启动异步事件引擎"""
        self._queue = asyncio.Queue()
        self._active = True
        self._loop = asyncio.get_event_loop()

        asyncio.create_task(self._run())
        logger.info("异步事件引擎已启动")

    async def stop(self):
        """停止"""
        self._active = False

    def register(self, event_type: EventType, handler: Callable):
        """注册处理器"""
        if handler not in self._handlers[event_type]:
            self._handlers[event_type].append(handler)

    async def put(self, event: Event):
        """发送事件"""
        await self._queue.put(event)

    async def emit(self, event_type: EventType, data: Any = None):
        """发送事件"""
        event = Event(type=event_type, data=data)
        await self.put(event)

    async def _run(self):
        """事件处理循环"""
        while self._active:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1)
                await self._process(event)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"异步事件处理异常: {e}", exc_info=True)

    async def _process(self, event: Event):
        """处理事件"""
        if event.type in self._handlers:
            for handler in self._handlers[event.type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    logger.error(f"处理器异常: {e}", exc_info=True)
