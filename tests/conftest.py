# coding=utf-8
"""
Pytest配置和共享fixtures
"""

import os
import sys
import pytest
import tempfile
import shutil

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def temp_dir():
    """创建临时目录"""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp, ignore_errors=True)


@pytest.fixture
def temp_config_file(temp_dir):
    """创建临时配置文件路径"""
    return os.path.join(temp_dir, "tq_config.json")


@pytest.fixture
def sample_config():
    """示例配置数据"""
    return {
        'tq_user': 'test_user',
        'tq_password': 'test_password',
        'sim_mode': True,
        'broker_id': '',
        'td_account': '',
        'td_password': '',
        'default_symbols': ['RB', 'AU', 'IF'],
        'initial_capital': 100000,
        'risk_config': {
            'max_position_per_symbol': 10,
            'max_daily_loss': 0.05,
            'max_drawdown': 0.15
        }
    }


@pytest.fixture
def sample_trade_data():
    """示例交易数据"""
    from datetime import datetime
    return {
        'trade_id': 1,
        'symbol': 'RB2505',
        'direction': 1,  # 多
        'entry_time': datetime(2025, 1, 1, 9, 30),
        'entry_price': 3500.0,
        'exit_time': datetime(2025, 1, 2, 14, 30),
        'exit_price': 3550.0,
        'volume': 1,
        'pnl': 500.0,
    }
