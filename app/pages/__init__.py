# coding=utf-8
"""
页面模块
将main.py中的各个页面拆分为独立模块
"""

from app.pages.dashboard import render_dashboard_page
from app.pages.backtest import render_backtest_page
from app.pages.risk_center import render_risk_center_page
from app.pages.data_management import render_data_management_page
from app.pages.settings import render_settings_page

__all__ = [
    'render_dashboard_page',
    'render_backtest_page',
    'render_risk_center_page',
    'render_data_management_page',
    'render_settings_page',
]
