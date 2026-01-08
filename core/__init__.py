# coding=utf-8
"""
Core模块
包含事件引擎、回测引擎、实盘引擎、调度器、数据服务
"""

from core.event_engine import EventEngine
from core.backtest_engine import BacktestEngine, BacktestResult, TradeRecord, generate_report
from core.live_engine import LiveEngine, create_live_engine, quick_start_sim
from core.scheduler import Scheduler, TradingScheduler, CronParser
from core.data_service import (
    DataService, DataSource, Period,
    get_data_service, load_bars,
    FUTURES_CONFIG, get_futures_config, get_tq_symbol
)

__all__ = [
    'EventEngine',
    'BacktestEngine',
    'BacktestResult',
    'TradeRecord',
    'generate_report',
    'LiveEngine',
    'create_live_engine',
    'quick_start_sim',
    'Scheduler',
    'TradingScheduler',
    'CronParser',
    # 数据服务
    'DataService',
    'DataSource',
    'Period',
    'get_data_service',
    'load_bars',
    'FUTURES_CONFIG',
    'get_futures_config',
    'get_tq_symbol',
]
