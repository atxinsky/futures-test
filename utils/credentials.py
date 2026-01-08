# coding=utf-8
"""
凭证安全管理器
支持从环境变量、keyring或配置文件读取敏感信息
"""

import os
import json
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

# 尝试导入 keyring
try:
    import keyring
    HAS_KEYRING = True
except ImportError:
    HAS_KEYRING = False
    logger.debug("keyring未安装，将使用环境变量或配置文件存储凭证")


class CredentialManager:
    """
    凭证管理器

    优先级（从高到低）：
    1. 环境变量
    2. keyring (如果可用)
    3. 配置文件 (不推荐用于密码)
    """

    SERVICE_NAME = "futures_quant"  # keyring服务名称

    # 环境变量映射
    ENV_MAPPING = {
        'tq_user': 'TQ_USER',
        'tq_password': 'TQ_PASSWORD',
        'broker_id': 'BROKER_ID',
        'td_account': 'TD_ACCOUNT',
        'td_password': 'TD_PASSWORD',
    }

    def __init__(self, config_file: str = None):
        """
        初始化凭证管理器

        Args:
            config_file: 配置文件路径（作为回退）
        """
        self.config_file = config_file
        self._config_cache: Optional[Dict] = None

    def get_credential(self, key: str, default: str = "") -> str:
        """
        获取凭证

        Args:
            key: 凭证键名（如 'tq_password'）
            default: 默认值

        Returns:
            凭证值
        """
        # 1. 尝试环境变量
        env_key = self.ENV_MAPPING.get(key, key.upper())
        value = os.environ.get(env_key)
        if value:
            logger.debug(f"从环境变量获取凭证: {key}")
            return value

        # 2. 尝试 keyring
        if HAS_KEYRING:
            try:
                value = keyring.get_password(self.SERVICE_NAME, key)
                if value:
                    logger.debug(f"从keyring获取凭证: {key}")
                    return value
            except Exception as e:
                logger.debug(f"keyring读取失败: {e}")

        # 3. 回退到配置文件
        if self.config_file and os.path.exists(self.config_file):
            if self._config_cache is None:
                try:
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        self._config_cache = json.load(f)
                except Exception as e:
                    logger.warning(f"配置文件读取失败: {e}")
                    self._config_cache = {}

            value = self._config_cache.get(key)
            if value:
                # 警告用户密码存储不安全
                if 'password' in key.lower():
                    logger.warning(
                        f"[安全警告] 从配置文件读取密码 '{key}'。"
                        f"建议使用环境变量 {env_key} 或 keyring 存储敏感信息。"
                    )
                return value

        return default

    def set_credential(self, key: str, value: str, use_keyring: bool = True) -> bool:
        """
        设置凭证

        Args:
            key: 凭证键名
            value: 凭证值
            use_keyring: 是否使用keyring存储

        Returns:
            是否成功
        """
        if use_keyring and HAS_KEYRING:
            try:
                keyring.set_password(self.SERVICE_NAME, key, value)
                logger.info(f"凭证已安全存储到keyring: {key}")
                return True
            except Exception as e:
                logger.warning(f"keyring存储失败: {e}")

        # 回退到配置文件（不推荐存储密码）
        if self.config_file:
            try:
                config = {}
                if os.path.exists(self.config_file):
                    with open(self.config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)

                config[key] = value
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)

                self._config_cache = config

                if 'password' in key.lower():
                    logger.warning(
                        f"[安全警告] 密码 '{key}' 已存储到配置文件（明文）。"
                        f"建议安装 keyring 库: pip install keyring"
                    )
                return True
            except Exception as e:
                logger.error(f"配置文件写入失败: {e}")

        return False

    def delete_credential(self, key: str) -> bool:
        """
        删除凭证

        Args:
            key: 凭证键名

        Returns:
            是否成功
        """
        success = False

        # 从 keyring 删除
        if HAS_KEYRING:
            try:
                keyring.delete_password(self.SERVICE_NAME, key)
                success = True
            except Exception:
                pass

        # 从配置文件删除
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                if key in config:
                    del config[key]
                    with open(self.config_file, 'w', encoding='utf-8') as f:
                        json.dump(config, f, indent=2, ensure_ascii=False)
                    self._config_cache = config
                    success = True
            except Exception as e:
                logger.error(f"配置文件更新失败: {e}")

        return success

    def get_tq_config(self) -> Dict:
        """
        获取天勤配置（安全方式）

        Returns:
            配置字典
        """
        return {
            'tq_user': self.get_credential('tq_user'),
            'tq_password': self.get_credential('tq_password'),
            'broker_id': self.get_credential('broker_id'),
            'td_account': self.get_credential('td_account'),
            'td_password': self.get_credential('td_password'),
            'sim_mode': self._get_config_value('sim_mode', True),
            'initial_capital': self._get_config_value('initial_capital', 1000000),
        }

    def _get_config_value(self, key: str, default):
        """获取非敏感配置值"""
        if self._config_cache is None and self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self._config_cache = json.load(f)
            except:
                self._config_cache = {}

        if self._config_cache:
            return self._config_cache.get(key, default)
        return default

    @staticmethod
    def check_security() -> Dict[str, bool]:
        """
        检查安全配置状态

        Returns:
            安全检查结果
        """
        checks = {
            'keyring_available': HAS_KEYRING,
            'env_tq_user_set': bool(os.environ.get('TQ_USER')),
            'env_tq_password_set': bool(os.environ.get('TQ_PASSWORD')),
        }

        # 检查配置文件权限（仅Unix）
        if os.name != 'nt':
            import stat
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'tq_config.json'
            )
            if os.path.exists(config_path):
                mode = os.stat(config_path).st_mode
                checks['config_file_secure'] = not bool(mode & stat.S_IROTH)  # 其他人不可读
            else:
                checks['config_file_secure'] = True

        return checks


# 全局单例
_credential_manager: Optional[CredentialManager] = None


def get_credential_manager(config_file: str = None) -> CredentialManager:
    """
    获取凭证管理器单例

    Args:
        config_file: 配置文件路径

    Returns:
        CredentialManager实例
    """
    global _credential_manager
    if _credential_manager is None:
        if config_file is None:
            config_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'tq_config.json'
            )
        _credential_manager = CredentialManager(config_file)
    return _credential_manager


def get_tq_credentials() -> Dict:
    """
    便捷函数：获取天勤凭证

    Returns:
        {tq_user, tq_password, ...}
    """
    return get_credential_manager().get_tq_config()
