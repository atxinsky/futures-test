# coding=utf-8
"""
统一数据服务测试
"""

import pytest
import os


class TestPeriod:
    """Period枚举测试"""

    def test_from_string(self):
        """测试字符串转换"""
        from core.data_service import Period

        assert Period.from_string('1h') == Period.H1
        assert Period.from_string('60m') == Period.M60
        assert Period.from_string('日线') == Period.D1
        assert Period.from_string('5分钟') == Period.M5

    def test_to_seconds(self):
        """测试转换为秒"""
        from core.data_service import Period

        assert Period.M1.to_seconds() == 60
        assert Period.H1.to_seconds() == 3600
        assert Period.D1.to_seconds() == 86400


class TestFuturesConfig:
    """期货配置测试"""

    def test_futures_config_exists(self):
        """测试配置存在"""
        from core.data_service import FUTURES_CONFIG

        assert 'RB' in FUTURES_CONFIG
        assert 'AU' in FUTURES_CONFIG
        assert 'IF' in FUTURES_CONFIG

    def test_futures_config_structure(self):
        """测试配置结构"""
        from core.data_service import FUTURES_CONFIG

        rb_config = FUTURES_CONFIG['RB']
        assert 'name' in rb_config
        assert 'exchange' in rb_config
        assert 'multiplier' in rb_config
        assert 'margin_rate' in rb_config

    def test_get_futures_config(self):
        """测试获取配置"""
        from core.data_service import get_futures_config

        config = get_futures_config('RB')
        assert config is not None
        assert config['name'] == '螺纹钢'
        assert config['exchange'] == 'SHFE'

        # 测试带合约月份
        config2 = get_futures_config('RB2505')
        assert config2 is not None
        assert config2['name'] == '螺纹钢'

    def test_get_futures_config_not_found(self):
        """测试未找到配置"""
        from core.data_service import get_futures_config

        config = get_futures_config('UNKNOWN')
        assert config is None


class TestSymbolConversion:
    """品种代码转换测试"""

    def test_get_tq_symbol_main(self):
        """测试主力合约代码"""
        from core.data_service import get_tq_symbol

        tq_symbol = get_tq_symbol('RB')
        assert 'SHFE' in tq_symbol
        assert 'rb' in tq_symbol.lower()

    def test_get_tq_symbol_specific(self):
        """测试具体合约代码"""
        from core.data_service import get_tq_symbol

        tq_symbol = get_tq_symbol('RB2505', main_contract=False)
        assert 'SHFE' in tq_symbol
        assert '2505' in tq_symbol

    def test_get_sina_symbol(self):
        """测试新浪代码"""
        from core.data_service import get_sina_symbol

        sina = get_sina_symbol('RB')
        assert sina == 'RB0'


class TestDataService:
    """DataService类测试"""

    def test_init(self, temp_dir):
        """测试初始化"""
        from core.data_service import DataService

        service = DataService(data_dir=temp_dir)
        assert service.data_dir == temp_dir
        assert os.path.exists(temp_dir)

    def test_get_available_symbols(self, temp_dir):
        """测试获取可用品种"""
        from core.data_service import DataService

        service = DataService(data_dir=temp_dir)
        symbols = service.get_available_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert 'RB' in symbols
        assert 'AU' in symbols

    def test_get_symbol_info(self, temp_dir):
        """测试获取品种信息"""
        from core.data_service import DataService

        service = DataService(data_dir=temp_dir)
        info = service.get_symbol_info('RB')

        assert info is not None
        assert info['name'] == '螺纹钢'

    def test_get_symbols_by_category(self, temp_dir):
        """测试按类别获取品种"""
        from core.data_service import DataService

        service = DataService(data_dir=temp_dir)
        categories = service.get_symbols_by_category()

        assert isinstance(categories, dict)
        assert '黑色系' in categories
        assert '贵金属' in categories

        # 验证黑色系包含螺纹钢
        black_metals = categories['黑色系']
        symbols = [s[0] for s in black_metals]
        assert 'RB' in symbols

    def test_clear_cache(self, temp_dir):
        """测试清理缓存"""
        from core.data_service import DataService

        service = DataService(data_dir=temp_dir)

        # 手动添加缓存
        service._cache['test_key'] = 'test_value'
        assert 'test_key' in service._cache

        # 清理缓存
        service.clear_cache()
        assert len(service._cache) == 0


class TestDataServiceSingleton:
    """数据服务单例测试"""

    def test_get_data_service(self, temp_dir):
        """测试获取单例"""
        from core.data_service import get_data_service, _data_service

        # 重置单例
        import core.data_service
        core.data_service._data_service = None

        service1 = get_data_service(temp_dir)
        service2 = get_data_service()

        assert service1 is service2


class TestLoadBarsConvenience:
    """load_bars便捷函数测试"""

    def test_load_bars_function_exists(self):
        """测试函数存在"""
        from core.data_service import load_bars

        assert callable(load_bars)
