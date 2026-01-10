# coding=utf-8
"""
优化结果配置应用器
负责将优化结果保存为YAML配置文件
"""

import logging
from typing import Dict, Optional
from datetime import datetime

# 路径已在 __init__.py 中统一设置

from .base import OptimizationResult

logger = logging.getLogger(__name__)


class ConfigApplier:
    """配置应用器"""

    @staticmethod
    def save_optimized_config(result: OptimizationResult,
                              symbol: Optional[str] = None,
                              auto_name: bool = True) -> str:
        """
        保存优化结果为YAML配置文件

        Args:
            result: 优化结果
            symbol: 品种代码（None表示多品种）
            auto_name: 是否自动生成文件名

        Returns:
            保存的文件路径
        """
        from config_manager import save_config, create_default_config

        # 1. 生成文件名
        if auto_name:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if symbol:
                filename = f"{result.strategy_name}_{symbol}_opt_{timestamp}.yml"
            else:
                symbols_str = "_".join(result.config.symbols[:3]) if result.config else "multi"
                filename = f"{result.strategy_name}_{symbols_str}_opt_{timestamp}.yml"
        else:
            if symbol:
                filename = f"{result.strategy_name}_{symbol}_optimized.yml"
            else:
                filename = f"{result.strategy_name}_optimized.yml"

        # 2. 创建基础配置
        target_symbol = symbol or (result.config.symbols[0] if result.config and result.config.symbols else "IF")
        config = create_default_config(result.strategy_name, target_symbol)

        # 3. 更新参数
        config["run_policy"]["params"] = result.best_params.copy()

        # 4. 如果是多品种，更新品种列表
        if not symbol and result.config and result.config.symbols:
            config["pairs"] = result.config.symbols

        # 5. 添加优化元信息
        config["optimization_info"] = {
            "best_value": round(result.best_value, 4),
            "objective": result.config.objective if result.config else "sharpe",
            "train_sharpe": round(result.train_metrics.get("sharpe", 0), 4),
            "train_return": round(result.train_metrics.get("return", 0), 4),
            "val_sharpe": round(result.val_metrics.get("sharpe", 0), 4),
            "val_return": round(result.val_metrics.get("return", 0), 4),
            "train_range": f"{result.config.train_start}~{result.config.train_end}" if result.config else "",
            "val_range": f"{result.config.val_start}~{result.config.val_end}" if result.config else "",
            "optimized_at": result.created_at.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": symbol or "multi",
            "n_trials": result.config.n_trials if result.config else 50
        }

        # 6. 保存
        filepath = save_config(filename, config)
        logger.info(f"优化配置已保存: {filepath}")

        return filepath

    @staticmethod
    def save_all_symbols(results: Dict[str, OptimizationResult]) -> Dict[str, str]:
        """
        批量保存每品种优化结果

        Args:
            results: {symbol: OptimizationResult}

        Returns:
            {symbol: filepath}
        """
        saved_files = {}

        for symbol, result in results.items():
            try:
                filepath = ConfigApplier.save_optimized_config(result, symbol, auto_name=True)
                saved_files[symbol] = filepath
            except Exception as e:
                logger.error(f"保存 {symbol} 配置失败: {e}")

        return saved_files

    @staticmethod
    def apply_to_session_state(result: OptimizationResult,
                               symbol: Optional[str] = None) -> Dict:
        """
        应用到Streamlit session_state（用于回测页面）

        Returns:
            应用信息字典
        """
        return {
            'strategy_name': result.strategy_name,
            'symbol': symbol,
            'params': result.best_params.copy(),
            'best_value': result.best_value,
            'train_metrics': result.train_metrics.copy(),
            'val_metrics': result.val_metrics.copy(),
            'optimized_at': result.created_at.strftime("%Y-%m-%d %H:%M:%S")
        }

    @staticmethod
    def load_optimized_params(strategy_name: str, symbol: str = None) -> Optional[Dict]:
        """
        从配置文件加载优化过的参数

        Args:
            strategy_name: 策略名称
            symbol: 品种代码

        Returns:
            参数字典，如果未找到返回None
        """
        from config_manager import list_configs, load_config

        # 搜索匹配的配置文件
        configs = list_configs()
        pattern = f"{strategy_name}_{symbol}_opt" if symbol else f"{strategy_name}_"

        matching_configs = [c for c in configs if c.startswith(pattern) and 'opt' in c]

        if not matching_configs:
            return None

        # 获取最新的配置文件
        latest_config = sorted(matching_configs)[-1]
        config = load_config(latest_config)

        if config and 'run_policy' in config and 'params' in config['run_policy']:
            return {
                'params': config['run_policy']['params'],
                'config_file': latest_config,
                'optimization_info': config.get('optimization_info', {})
            }

        return None

    @staticmethod
    def get_optimization_history(strategy_name: str = None) -> list:
        """
        获取优化历史记录

        Args:
            strategy_name: 策略名称筛选（可选）

        Returns:
            优化记录列表
        """
        from config_manager import list_configs, load_config

        configs = list_configs()
        history = []

        for config_name in configs:
            if '_opt_' not in config_name:
                continue

            if strategy_name and not config_name.startswith(strategy_name):
                continue

            try:
                config = load_config(config_name)
                if config and 'optimization_info' in config:
                    info = config['optimization_info']
                    history.append({
                        'config_file': config_name,
                        'strategy': config.get('run_policy', {}).get('name', ''),
                        'symbol': info.get('symbol', ''),
                        'best_value': info.get('best_value', 0),
                        'objective': info.get('objective', 'sharpe'),
                        'train_sharpe': info.get('train_sharpe', 0),
                        'val_sharpe': info.get('val_sharpe', 0),
                        'optimized_at': info.get('optimized_at', ''),
                        'n_trials': info.get('n_trials', 0),
                    })
            except Exception as e:
                logger.warning(f"加载配置 {config_name} 失败: {e}")

        # 按时间倒序排列
        history.sort(key=lambda x: x.get('optimized_at', ''), reverse=True)
        return history
