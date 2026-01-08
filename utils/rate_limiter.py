# coding=utf-8
"""
流量控制模块
使用令牌桶算法实现CTP接口的流控，防止请求被拒
"""

import time
import logging
import threading
from typing import Optional, Callable
from dataclasses import dataclass
from functools import wraps
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class RateLimiterConfig:
    """流控配置"""
    # 令牌桶容量（最大突发请求数）
    bucket_size: int = 10

    # 令牌生成速率（每秒生成的令牌数）
    tokens_per_second: float = 2.0

    # 请求超时时间（秒）
    timeout: float = 5.0

    # 是否阻塞等待令牌
    blocking: bool = True

    # 最大等待队列长度
    max_queue_size: int = 100


class TokenBucketRateLimiter:
    """
    令牌桶限流器

    功能:
    1. 控制请求速率
    2. 允许突发流量（令牌累积）
    3. 超时处理
    4. 统计信息
    """

    def __init__(self, config: RateLimiterConfig = None):
        """
        初始化限流器

        Args:
            config: 限流配置
        """
        self.config = config or RateLimiterConfig()

        # 令牌桶状态
        self._tokens = float(self.config.bucket_size)
        self._last_update = time.time()
        self._lock = threading.Lock()

        # 统计
        self._total_requests = 0
        self._accepted_requests = 0
        self._rejected_requests = 0
        self._total_wait_time = 0.0

    def _refill_tokens(self):
        """补充令牌"""
        now = time.time()
        elapsed = now - self._last_update
        new_tokens = elapsed * self.config.tokens_per_second

        self._tokens = min(
            self.config.bucket_size,
            self._tokens + new_tokens
        )
        self._last_update = now

    def acquire(self, tokens: int = 1, timeout: float = None) -> bool:
        """
        获取令牌

        Args:
            tokens: 需要的令牌数量
            timeout: 超时时间（秒），None 使用默认配置

        Returns:
            是否成功获取
        """
        if timeout is None:
            timeout = self.config.timeout

        start_time = time.time()
        deadline = start_time + timeout

        with self._lock:
            self._total_requests += 1

            while True:
                self._refill_tokens()

                if self._tokens >= tokens:
                    # 有足够令牌，扣除并返回
                    self._tokens -= tokens
                    wait_time = time.time() - start_time
                    self._total_wait_time += wait_time
                    self._accepted_requests += 1

                    if wait_time > 0.1:
                        logger.debug(f"流控等待 {wait_time:.2f}s 后获取令牌")

                    return True

                if not self.config.blocking:
                    # 非阻塞模式，直接拒绝
                    self._rejected_requests += 1
                    return False

                # 计算需要等待的时间
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self.config.tokens_per_second

                # 检查是否超时
                if time.time() + wait_time > deadline:
                    self._rejected_requests += 1
                    logger.warning(f"流控超时: 需要等待 {wait_time:.2f}s，超过限制")
                    return False

                # 释放锁并等待
                self._lock.release()
                try:
                    time.sleep(min(wait_time, 0.1))  # 最多等待100ms然后重试
                finally:
                    self._lock.acquire()

    def try_acquire(self, tokens: int = 1) -> bool:
        """
        尝试获取令牌（非阻塞）

        Args:
            tokens: 需要的令牌数量

        Returns:
            是否成功获取
        """
        with self._lock:
            self._refill_tokens()

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._accepted_requests += 1
                return True

            self._rejected_requests += 1
            return False

    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            self._refill_tokens()
            return {
                'available_tokens': self._tokens,
                'total_requests': self._total_requests,
                'accepted_requests': self._accepted_requests,
                'rejected_requests': self._rejected_requests,
                'rejection_rate': (self._rejected_requests / self._total_requests
                                   if self._total_requests > 0 else 0),
                'avg_wait_time': (self._total_wait_time / self._accepted_requests
                                  if self._accepted_requests > 0 else 0)
            }

    def reset_stats(self):
        """重置统计"""
        with self._lock:
            self._total_requests = 0
            self._accepted_requests = 0
            self._rejected_requests = 0
            self._total_wait_time = 0.0


class GatewayRateLimiter:
    """
    网关流控器

    针对CTP接口特点的流控：
    - 查询类接口：每秒2次
    - 下单类接口：每秒5次
    - 总请求：每秒10次
    """

    def __init__(self):
        # 查询限流器（较严格）
        self.query_limiter = TokenBucketRateLimiter(RateLimiterConfig(
            bucket_size=5,
            tokens_per_second=2.0,
            timeout=10.0
        ))

        # 下单限流器（相对宽松）
        self.order_limiter = TokenBucketRateLimiter(RateLimiterConfig(
            bucket_size=10,
            tokens_per_second=5.0,
            timeout=5.0
        ))

        # 总体限流器
        self.global_limiter = TokenBucketRateLimiter(RateLimiterConfig(
            bucket_size=20,
            tokens_per_second=10.0,
            timeout=10.0
        ))

    def acquire_for_query(self) -> bool:
        """获取查询令牌"""
        # 先获取全局令牌，再获取查询令牌
        if not self.global_limiter.try_acquire():
            return self.global_limiter.acquire(timeout=5.0)

        return self.query_limiter.acquire()

    def acquire_for_order(self) -> bool:
        """获取下单令牌"""
        if not self.global_limiter.try_acquire():
            return self.global_limiter.acquire(timeout=3.0)

        return self.order_limiter.acquire()

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'query': self.query_limiter.get_stats(),
            'order': self.order_limiter.get_stats(),
            'global': self.global_limiter.get_stats()
        }


def rate_limited(limiter_attr: str = 'rate_limiter', limit_type: str = 'query'):
    """
    流控装饰器

    Args:
        limiter_attr: 限流器属性名
        limit_type: 限流类型 ('query' 或 'order')

    Usage:
        class Gateway:
            def __init__(self):
                self.rate_limiter = GatewayRateLimiter()

            @rate_limited('rate_limiter', 'query')
            def query_position(self):
                ...
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            limiter = getattr(self, limiter_attr, None)

            if limiter is None:
                # 没有限流器，直接执行
                return func(self, *args, **kwargs)

            # 根据类型获取令牌
            if isinstance(limiter, GatewayRateLimiter):
                if limit_type == 'order':
                    acquired = limiter.acquire_for_order()
                else:
                    acquired = limiter.acquire_for_query()
            else:
                acquired = limiter.acquire()

            if not acquired:
                logger.warning(f"流控拒绝请求: {func.__name__}")
                raise RateLimitExceeded(f"请求被流控拒绝: {func.__name__}")

            return func(self, *args, **kwargs)

        return wrapper
    return decorator


class RateLimitExceeded(Exception):
    """流控超限异常"""
    pass


# 预定义的限流器配置
RATE_LIMITER_CONFIGS = {
    # CTP标准配置
    'ctp_standard': RateLimiterConfig(
        bucket_size=10,
        tokens_per_second=2.0,
        timeout=10.0
    ),

    # 保守配置（用于不稳定网络）
    'conservative': RateLimiterConfig(
        bucket_size=5,
        tokens_per_second=1.0,
        timeout=15.0
    ),

    # 宽松配置（用于模拟盘测试）
    'relaxed': RateLimiterConfig(
        bucket_size=50,
        tokens_per_second=20.0,
        timeout=5.0
    )
}


def get_rate_limiter(config_name: str = 'ctp_standard') -> TokenBucketRateLimiter:
    """获取预配置的限流器"""
    config = RATE_LIMITER_CONFIGS.get(config_name, RATE_LIMITER_CONFIGS['ctp_standard'])
    return TokenBucketRateLimiter(config)
