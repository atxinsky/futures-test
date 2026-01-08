# coding=utf-8
"""
TqSdk配置管理模块
统一管理天勤SDK相关配置，整合凭证安全存储
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from threading import Lock

logger = logging.getLogger(__name__)

# 默认配置文件路径
DEFAULT_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tq_config.json"
)


@dataclass
class RiskConfig:
    """风控配置"""
    max_position_per_symbol: int = 10      # 单品种最大持仓
    max_daily_loss: float = 0.05           # 最大日亏损比例
    max_drawdown: float = 0.15             # 最大回撤比例
    max_order_volume: int = 100            # 单笔最大委托量
    position_warning_threshold: float = 0.8  # 持仓预警阈值

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'RiskConfig':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TqConfig:
    """TqSdk完整配置"""
    # 天勤账号
    tq_user: str = ""
    tq_password: str = ""

    # 模拟/实盘模式
    sim_mode: bool = True

    # 期货公司账号（实盘用）
    broker_id: str = ""
    td_account: str = ""
    td_password: str = ""

    # 交易配置
    default_symbols: List[str] = field(default_factory=lambda: ['RB', 'AU', 'IF'])
    initial_capital: float = 100000

    # 风控配置
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    def to_dict(self) -> dict:
        """转换为字典（用于JSON序列化）"""
        data = asdict(self)
        data['risk_config'] = self.risk_config.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'TqConfig':
        """从字典创建（用于JSON反序列化）"""
        risk_data = data.pop('risk_config', {})
        config = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if risk_data:
            config.risk_config = RiskConfig.from_dict(risk_data)
        return config

    @property
    def sensitive_keys(self) -> List[str]:
        """敏感字段列表"""
        return ['tq_password', 'td_password']

    @property
    def credential_keys(self) -> List[str]:
        """需要凭证管理的字段"""
        return ['tq_user', 'tq_password', 'broker_id', 'td_account', 'td_password']


class TqConfigManager:
    """
    TqSdk配置管理器

    功能：
    1. 统一的配置加载/保存接口
    2. 敏感信息安全存储（支持环境变量、keyring、配置文件）
    3. 配置验证
    4. 线程安全
    """

    _instance: Optional['TqConfigManager'] = None
    _lock = Lock()

    def __new__(cls, config_file: str = None):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_file: str = None):
        if self._initialized:
            return

        self.config_file = config_file or DEFAULT_CONFIG_FILE
        self._config: Optional[TqConfig] = None
        self._credential_manager = None
        self._has_credential_manager = False

        # 尝试加载凭证管理器
        try:
            from utils.credentials import get_credential_manager
            self._credential_manager = get_credential_manager(self.config_file)
            self._has_credential_manager = True
        except ImportError:
            logger.debug("凭证管理器不可用，将使用配置文件存储")

        self._initialized = True

    def load(self) -> TqConfig:
        """
        加载配置

        优先级：
        1. 环境变量（敏感信息）
        2. Keyring（敏感信息）
        3. 配置文件
        """
        with self._lock:
            config_data = self._get_default_config_dict()

            # 从配置文件加载非敏感配置
            if os.path.exists(self.config_file):
                try:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        file_config = json.load(f)
                        # 更新非敏感配置
                        for key in ['sim_mode', 'default_symbols', 'initial_capital', 'risk_config']:
                            if key in file_config:
                                config_data[key] = file_config[key]
                        # 也加载凭证相关的非密码字段
                        for key in ['tq_user', 'broker_id', 'td_account']:
                            if key in file_config and file_config[key]:
                                config_data[key] = file_config[key]
                except Exception as e:
                    logger.warning(f"加载配置文件失败: {e}")

            # 使用凭证管理器获取敏感信息
            if self._has_credential_manager:
                for key in TqConfig().credential_keys:
                    try:
                        value = self._credential_manager.get_credential(key)
                        if value:
                            config_data[key] = value
                    except Exception as e:
                        logger.debug(f"获取凭证 {key} 失败: {e}")
            else:
                # 回退到配置文件读取密码
                if os.path.exists(self.config_file):
                    try:
                        with open(self.config_file, 'r', encoding='utf-8') as f:
                            file_config = json.load(f)
                            for key in ['tq_password', 'td_password']:
                                if key in file_config:
                                    config_data[key] = file_config[key]
                    except:
                        pass

            self._config = TqConfig.from_dict(config_data)
            return self._config

    def save(self, config: TqConfig):
        """
        保存配置

        敏感信息优先存储到keyring，非敏感信息存储到配置文件
        """
        with self._lock:
            config_dict = config.to_dict()

            # 分离敏感和非敏感配置
            sensitive_keys = config.sensitive_keys
            non_sensitive_config = {k: v for k, v in config_dict.items() if k not in sensitive_keys}

            # 保存非敏感配置到文件
            try:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(non_sensitive_config, f, indent=2, ensure_ascii=False)
            except Exception as e:
                logger.error(f"保存配置文件失败: {e}")
                raise

            # 使用凭证管理器存储敏感信息
            if self._has_credential_manager:
                for key in sensitive_keys:
                    if key in config_dict and config_dict[key]:
                        try:
                            self._credential_manager.set_credential(key, config_dict[key])
                        except Exception as e:
                            logger.warning(f"存储凭证 {key} 失败: {e}")
            else:
                # 警告用户并写入文件（包含密码）
                logger.warning(
                    "[安全警告] 密码将以明文存储。建议安装 keyring: pip install keyring"
                )
                try:
                    with open(self.config_file, 'w', encoding='utf-8') as f:
                        json.dump(config_dict, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"保存配置文件失败: {e}")
                    raise

            self._config = config

    def get(self) -> TqConfig:
        """获取当前配置（如未加载则先加载）"""
        if self._config is None:
            return self.load()
        return self._config

    def update(self, **kwargs):
        """更新部分配置"""
        config = self.get()
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        self.save(config)

    def validate(self, config: TqConfig = None) -> tuple[bool, List[str]]:
        """
        验证配置

        Returns:
            (是否有效, 错误信息列表)
        """
        config = config or self.get()
        errors = []

        # 验证天勤账号
        if not config.tq_user:
            errors.append("天勤用户名不能为空")
        if not config.tq_password:
            errors.append("天勤密码不能为空")

        # 实盘模式下验证期货账号
        if not config.sim_mode:
            if not config.broker_id:
                errors.append("实盘模式需要配置期货公司代码")
            if not config.td_account:
                errors.append("实盘模式需要配置交易账号")
            if not config.td_password:
                errors.append("实盘模式需要配置交易密码")

        # 验证交易配置
        if config.initial_capital <= 0:
            errors.append("初始资金必须大于0")

        # 验证风控配置
        if config.risk_config.max_position_per_symbol <= 0:
            errors.append("单品种最大持仓必须大于0")
        if not 0 < config.risk_config.max_daily_loss <= 1:
            errors.append("最大日亏损比例应在0-1之间")
        if not 0 < config.risk_config.max_drawdown <= 1:
            errors.append("最大回撤比例应在0-1之间")

        return len(errors) == 0, errors

    def test_connection(self, tq_user: str = None, tq_password: str = None) -> tuple[bool, str]:
        """
        测试天勤连接

        Args:
            tq_user: 用户名（可选，默认使用配置中的值）
            tq_password: 密码（可选，默认使用配置中的值）

        Returns:
            (成功与否, 消息)
        """
        config = self.get()
        user = tq_user or config.tq_user
        password = tq_password or config.tq_password

        if not user or not password:
            return False, "请先配置天勤账号和密码"

        try:
            from tqsdk import TqApi, TqAuth, TqSim

            api = TqApi(
                account=TqSim(),
                auth=TqAuth(user, password)
            )
            api.close()
            return True, "连接成功"

        except ImportError:
            return False, "TqSdk未安装，请执行: pip install tqsdk"
        except Exception as e:
            error_msg = str(e)
            if "认证" in error_msg or "auth" in error_msg.lower():
                return False, "认证失败，请检查用户名和密码"
            return False, f"连接失败: {error_msg}"

    def _get_default_config_dict(self) -> dict:
        """获取默认配置字典"""
        return TqConfig().to_dict()

    @property
    def has_credential_manager(self) -> bool:
        """是否有凭证管理器"""
        return self._has_credential_manager


# 全局单例
_config_manager: Optional[TqConfigManager] = None


def get_tq_config_manager(config_file: str = None) -> TqConfigManager:
    """获取TqSdk配置管理器单例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = TqConfigManager(config_file)
    return _config_manager


# ============== 便捷函数（兼容旧代码） ==============

def load_tq_config() -> dict:
    """
    加载TqSdk配置（兼容旧接口）

    Returns:
        配置字典
    """
    return get_tq_config_manager().get().to_dict()


def save_tq_config(config: dict):
    """
    保存TqSdk配置（兼容旧接口）

    Args:
        config: 配置字典
    """
    tq_config = TqConfig.from_dict(config)
    get_tq_config_manager().save(tq_config)


def load_tq_config_for_settings() -> dict:
    """加载TqSdk配置（用于系统设置，兼容main.py）"""
    return load_tq_config()


def save_tq_config_for_settings(config: dict):
    """保存TqSdk配置（用于系统设置，兼容main.py）"""
    save_tq_config(config)


def test_tq_connection(tq_user: str = None, tq_password: str = None) -> tuple[bool, str]:
    """测试天勤连接（兼容旧接口）"""
    return get_tq_config_manager().test_connection(tq_user, tq_password)
