# coding=utf-8
"""
系统配置
包含路径、风控、Web、日志等系统级配置
"""

import os

# ============ 路径配置 ============
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
LOG_DIR = os.path.join(ROOT_DIR, "logs")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")
BACKUP_DIR = os.path.join(ROOT_DIR, "backup")

# 确保目录存在
for dir_path in [DATA_DIR, LOG_DIR, CACHE_DIR, BACKUP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============ 交易配置 ============
TRADING_CONFIG = {
    # 初始资金
    'initial_capital': 100000.0,

    # 默认交易数量
    'default_volume': 1,

    # 最大持仓数
    'max_positions': 10,

    # 单笔最大风险
    'max_risk_per_trade': 0.02,  # 2%
}

# ============ 风控配置 ============
RISK_CONFIG = {
    # 持仓限制
    'max_position_per_symbol': 10,
    'max_position_total': 50,
    'max_order_per_symbol': 5,

    # 资金风控
    'max_margin_ratio': 0.8,
    'max_order_value_ratio': 0.3,
    'min_available': 10000,

    # 亏损控制
    'max_daily_loss_ratio': 0.05,
    'max_drawdown_ratio': 0.15,
    'max_consecutive_losses': 5,

    # 止损设置
    'default_stop_loss_ratio': 0.03,
    'stop_loss_atr_mult': 3.0,

    # 开关
    'enabled': True,
    'allow_open_when_risk': False,
    'force_close_on_max_loss': True,
}

# ============ Web配置 ============
WEB_CONFIG = {
    'host': '0.0.0.0',
    'port': 8504,  # 8501-8503已被Docker占用
    'debug': False,
    'theme': 'dark',
    'refresh_interval': 3,  # 秒
}

# ============ 日志配置 ============
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(LOG_DIR, 'trading.log'),
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5,
}

# ============ 数据源配置 ============
DATA_CONFIG = {
    # TianQin配置
    'tianqin': {
        'auth_user': '',
        'auth_password': '',
    },

    # 本地数据目录
    'local_data_dir': DATA_DIR,

    # 缓存设置
    'cache_enabled': True,
    'cache_dir': CACHE_DIR,
    'cache_max_size': 1000,  # MB
}

# ============ 模拟盘配置 ============
SIM_CONFIG = {
    # 滑点
    'slippage_ticks': 1,

    # 成交比例
    'fill_ratio': 1.0,

    # 模拟延迟
    'latency_ms': 0,
}

# ============ 回测配置 ============
BACKTEST_CONFIG = {
    # 默认参数
    'default_capital': 100000,
    'default_volume': 1,

    # 手续费
    'use_actual_commission': True,

    # 滑点
    'slippage_ticks': 1,

    # 输出
    'save_trades': True,
    'save_equity_curve': True,
}

# ============ 通知配置 ============
NOTIFICATION_CONFIG = {
    'enabled': False,
    'email': {
        'smtp_server': '',
        'smtp_port': 587,
        'username': '',
        'password': '',
        'to_addresses': [],
    },
    'webhook': {
        'url': '',
    },
}


def setup_logging():
    """配置日志"""
    import logging
    from logging.handlers import RotatingFileHandler

    # 创建日志目录
    os.makedirs(LOG_DIR, exist_ok=True)

    # 配置根日志
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, LOG_CONFIG['level']))

    # 文件处理器
    file_handler = RotatingFileHandler(
        LOG_CONFIG['file'],
        maxBytes=LOG_CONFIG['max_bytes'],
        backupCount=LOG_CONFIG['backup_count'],
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(LOG_CONFIG['format']))
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_CONFIG['format']))
    logger.addHandler(console_handler)

    return logger
