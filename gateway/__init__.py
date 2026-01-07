# coding=utf-8
"""
Gateway模块
包含基础网关类和各种网关实现
"""

from gateway.base_gateway import BaseGateway
from gateway.sim_gateway import SimGateway

# TqGateway需要tqsdk，按需导入
try:
    from gateway.tq_gateway import TqGateway
    __all__ = [
        'BaseGateway',
        'SimGateway',
        'TqGateway',
    ]
except ImportError:
    __all__ = [
        'BaseGateway',
        'SimGateway',
    ]
