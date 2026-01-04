# coding=utf-8
"""
YAML配置文件管理器
支持banbot风格的配置文件
"""

import os
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# 默认配置模板
DEFAULT_CONFIG = {
    "name": "backtest",
    "env": "backtest",
    "leverage": 1,
    "market_type": "futures",
    "contract_type": "index",
    "initial_capital": 1000000,
    "time_start": "20200101",
    "time_end": "20251231",
    "run_policy": {
        "name": "brother2v6",
        "timeframes": "日线",
        "params": {}
    },
    "pairs": ["IF"],
    "exchange": {
        "name": "cffex",
        "fees": {
            "maker": 0.000023,
            "taker": 0.000023
        }
    }
}

# 策略默认参数
STRATEGY_DEFAULTS = {
    "brother2v6": {
        "sml_len": 12,
        "big_len": 50,
        "break_len": 30,
        "atr_len": 20,
        "adx_len": 14,
        "adx_thres": 22.0,
        "chop_len": 14,
        "chop_thres": 50.0,
        "vol_len": 20,
        "vol_multi": 1.3,
        "stop_n": 3.0,
        "min_stop_n": 2.0,
        "break_even_pct": 10.0,
        "partial_trigger_pct": 12.0,
        "partial_drawdown_pct": 4.0,
        "partial_rate": 50.0,
        "full_drawdown_pct": 8.0,
        "capital_rate": 0.2,
        "risk_rate": 0.05
    },
    "brother2v5": {
        "sml_len": 10,
        "big_len": 40,
        "break_len": 40,
        "atr_len": 20,
        "adx_len": 14,
        "adx_thres": 25.0,
        "stop_n": 4.0,
        "capital_rate": 1.0,
        "risk_rate": 0.03
    },
    "dual_ma": {
        "fast_period": 10,
        "slow_period": 30,
        "stop_loss_pct": 3.0
    },
    "macdema_v3": {
        "ema_len": 20,
        "macd_fast": 12,
        "macd_slow": 26,
        "macd_signal": 9,
        "atr_len": 14,
        "stop_n": 2.5
    }
}

# 交易所费率配置
EXCHANGE_FEES = {
    "cffex": {  # 中金所
        "IF": {"maker": 0.000023, "taker": 0.000023, "multiplier": 300},
        "IH": {"maker": 0.000023, "taker": 0.000023, "multiplier": 300},
        "IC": {"maker": 0.000023, "taker": 0.000023, "multiplier": 200},
        "IM": {"maker": 0.000023, "taker": 0.000023, "multiplier": 200},
    },
    "shfe": {  # 上期所
        "AU": {"maker": 0.0001, "taker": 0.0001, "multiplier": 1000},
        "AG": {"maker": 0.00005, "taker": 0.00005, "multiplier": 15},
        "CU": {"maker": 0.00005, "taker": 0.00005, "multiplier": 5},
        "AL": {"maker": 0.00003, "taker": 0.00003, "multiplier": 5},
        "RB": {"maker": 0.0001, "taker": 0.0001, "multiplier": 10},
    },
    "dce": {  # 大商所
        "M": {"maker": 0.00015, "taker": 0.00015, "multiplier": 10},
        "Y": {"maker": 0.00025, "taker": 0.00025, "multiplier": 10},
        "P": {"maker": 0.00025, "taker": 0.00025, "multiplier": 10},
        "I": {"maker": 0.0001, "taker": 0.0001, "multiplier": 100},
    },
    "czce": {  # 郑商所
        "TA": {"maker": 0.00003, "taker": 0.00003, "multiplier": 5},
        "MA": {"maker": 0.00002, "taker": 0.00002, "multiplier": 10},
        "SR": {"maker": 0.00003, "taker": 0.00003, "multiplier": 10},
    }
}


def get_config_dir() -> str:
    """获取配置文件目录"""
    config_dir = os.path.join(os.path.dirname(__file__), "configs")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    return config_dir


def list_configs() -> List[str]:
    """列出所有配置文件"""
    config_dir = get_config_dir()
    configs = []
    for f in os.listdir(config_dir):
        if f.endswith('.yml') or f.endswith('.yaml'):
            configs.append(f)
    return sorted(configs)


def load_config(filename: str) -> Dict:
    """加载配置文件"""
    config_dir = get_config_dir()
    filepath = os.path.join(config_dir, filename)

    if not os.path.exists(filepath):
        return DEFAULT_CONFIG.copy()

    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def save_config(filename: str, config: Dict) -> str:
    """保存配置文件"""
    config_dir = get_config_dir()
    if not filename.endswith('.yml') and not filename.endswith('.yaml'):
        filename += '.yml'

    filepath = os.path.join(config_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    return filepath


def delete_config(filename: str) -> bool:
    """删除配置文件"""
    config_dir = get_config_dir()
    filepath = os.path.join(config_dir, filename)

    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False


def config_to_yaml(config: Dict) -> str:
    """配置转YAML字符串"""
    return yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)


def yaml_to_config(yaml_str: str) -> Dict:
    """YAML字符串转配置"""
    return yaml.safe_load(yaml_str)


def create_default_config(strategy_name: str, symbol: str = "IF") -> Dict:
    """创建默认配置"""
    config = DEFAULT_CONFIG.copy()
    config["run_policy"] = {
        "name": strategy_name,
        "timeframes": "日线",
        "params": STRATEGY_DEFAULTS.get(strategy_name, {}).copy()
    }
    config["pairs"] = [symbol]

    # 设置交易所费率
    for exchange, symbols in EXCHANGE_FEES.items():
        if symbol in symbols:
            config["exchange"]["name"] = exchange
            config["exchange"]["fees"] = {
                "maker": symbols[symbol]["maker"],
                "taker": symbols[symbol]["taker"]
            }
            break

    return config


def get_strategy_param_groups(strategy_name: str) -> Dict[str, List[str]]:
    """获取策略参数分组"""
    groups = {
        "brother2v6": {
            "趋势参数": ["sml_len", "big_len", "break_len"],
            "指标参数": ["atr_len", "adx_len", "adx_thres", "chop_len", "chop_thres"],
            "成交量": ["vol_len", "vol_multi"],
            "止损参数": ["stop_n", "min_stop_n"],
            "止盈参数": ["break_even_pct", "partial_trigger_pct", "partial_drawdown_pct", "partial_rate", "full_drawdown_pct"],
            "仓位管理": ["capital_rate", "risk_rate"]
        },
        "brother2v5": {
            "趋势参数": ["sml_len", "big_len", "break_len"],
            "指标参数": ["atr_len", "adx_len", "adx_thres"],
            "止损参数": ["stop_n"],
            "仓位管理": ["capital_rate", "risk_rate"]
        }
    }
    return groups.get(strategy_name, {"参数": list(STRATEGY_DEFAULTS.get(strategy_name, {}).keys())})
