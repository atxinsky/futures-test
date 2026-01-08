# coding=utf-8
"""
TqSdk配置管理模块测试
"""

import pytest
import json
import os


class TestTqConfig:
    """TqConfig数据类测试"""

    def test_default_values(self):
        """测试默认值"""
        from utils.tq_config import TqConfig

        config = TqConfig()
        assert config.tq_user == ""
        assert config.tq_password == ""
        assert config.sim_mode == True
        assert config.initial_capital == 100000
        assert 'RB' in config.default_symbols

    def test_to_dict(self):
        """测试转换为字典"""
        from utils.tq_config import TqConfig

        config = TqConfig(tq_user="test", sim_mode=False)
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d['tq_user'] == "test"
        assert d['sim_mode'] == False
        assert 'risk_config' in d

    def test_from_dict(self):
        """测试从字典创建"""
        from utils.tq_config import TqConfig

        data = {
            'tq_user': 'user1',
            'tq_password': 'pass1',
            'sim_mode': False,
            'initial_capital': 200000,
        }

        config = TqConfig.from_dict(data)
        assert config.tq_user == 'user1'
        assert config.sim_mode == False
        assert config.initial_capital == 200000


class TestRiskConfig:
    """RiskConfig数据类测试"""

    def test_default_values(self):
        """测试默认值"""
        from utils.tq_config import RiskConfig

        risk = RiskConfig()
        assert risk.max_position_per_symbol == 10
        assert risk.max_daily_loss == 0.05
        assert risk.max_drawdown == 0.15

    def test_to_dict(self):
        """测试转换为字典"""
        from utils.tq_config import RiskConfig

        risk = RiskConfig(max_position_per_symbol=20)
        d = risk.to_dict()

        assert d['max_position_per_symbol'] == 20


class TestTqConfigManager:
    """TqConfigManager测试"""

    def test_load_default(self, temp_config_file):
        """测试加载默认配置"""
        from utils.tq_config import TqConfigManager

        # 重置单例
        TqConfigManager._instance = None

        manager = TqConfigManager(temp_config_file)
        config = manager.load()

        assert config.sim_mode == True
        assert config.initial_capital == 100000

    def test_save_and_load(self, temp_config_file, sample_config):
        """测试保存和加载"""
        from utils.tq_config import TqConfigManager, TqConfig

        # 重置单例
        TqConfigManager._instance = None

        manager = TqConfigManager(temp_config_file)

        # 创建配置
        config = TqConfig.from_dict(sample_config)
        manager.save(config)

        # 验证文件存在
        assert os.path.exists(temp_config_file)

        # 重新加载
        loaded = manager.load()
        assert loaded.tq_user == sample_config['tq_user']
        assert loaded.sim_mode == sample_config['sim_mode']

    def test_validate_empty_user(self, temp_config_file):
        """测试验证空用户名"""
        from utils.tq_config import TqConfigManager, TqConfig

        TqConfigManager._instance = None
        manager = TqConfigManager(temp_config_file)

        config = TqConfig()  # 空用户名
        is_valid, errors = manager.validate(config)

        assert not is_valid
        assert any('用户名' in e for e in errors)

    def test_validate_live_mode_requires_broker(self, temp_config_file):
        """测试实盘模式需要期货账号"""
        from utils.tq_config import TqConfigManager, TqConfig

        TqConfigManager._instance = None
        manager = TqConfigManager(temp_config_file)

        config = TqConfig(
            tq_user='user',
            tq_password='pass',
            sim_mode=False,  # 实盘模式
            broker_id='',    # 空
        )
        is_valid, errors = manager.validate(config)

        assert not is_valid
        assert any('期货公司' in e for e in errors)


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_load_tq_config(self, temp_config_file, sample_config):
        """测试load_tq_config"""
        from utils.tq_config import TqConfigManager, TqConfig, load_tq_config

        TqConfigManager._instance = None
        manager = TqConfigManager(temp_config_file)

        config = TqConfig.from_dict(sample_config)
        manager.save(config)

        # 使用便捷函数
        loaded = load_tq_config()
        assert isinstance(loaded, dict)
        assert loaded['tq_user'] == sample_config['tq_user']

    def test_save_tq_config(self, temp_config_file):
        """测试save_tq_config"""
        from utils.tq_config import TqConfigManager, save_tq_config, load_tq_config

        TqConfigManager._instance = None
        TqConfigManager(temp_config_file)

        new_config = {
            'tq_user': 'new_user',
            'tq_password': 'new_pass',
            'sim_mode': True,
            'default_symbols': ['AU'],
            'initial_capital': 50000,
            'risk_config': {}
        }

        save_tq_config(new_config)
        loaded = load_tq_config()

        assert loaded['tq_user'] == 'new_user'
        assert loaded['initial_capital'] == 50000
