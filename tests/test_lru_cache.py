# coding=utf-8
"""
LRU缓存测试
"""

import pytest
import pandas as pd
import numpy as np
import time


class TestLRUDataCache:
    """LRU数据缓存测试"""

    def test_init(self):
        """测试初始化"""
        from core.data_service import LRUDataCache

        cache = LRUDataCache(max_entries=10, max_size_mb=100, expire_seconds=60)
        assert cache.max_entries == 10
        assert cache.max_size_bytes == 100 * 1024 * 1024
        assert cache.expire_seconds == 60

    def test_put_and_get(self):
        """测试存取数据"""
        from core.data_service import LRUDataCache

        cache = LRUDataCache(max_entries=10)

        df = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100),
            'close': np.random.randn(100)
        })

        cache.put('test_key', df)
        result = cache.get('test_key')

        assert result is not None
        assert len(result) == 100
        assert 'close' in result.columns

    def test_get_nonexistent(self):
        """测试获取不存在的键"""
        from core.data_service import LRUDataCache

        cache = LRUDataCache()
        result = cache.get('nonexistent')

        assert result is None

    def test_lru_eviction(self):
        """测试LRU淘汰策略"""
        from core.data_service import LRUDataCache

        cache = LRUDataCache(max_entries=3)

        # 添加3个条目
        for i in range(3):
            df = pd.DataFrame({'value': [i]})
            cache.put(f'key_{i}', df)

        # 访问key_0使其变为最近使用
        cache.get('key_0')

        # 添加第4个条目，应该淘汰key_1
        df = pd.DataFrame({'value': [3]})
        cache.put('key_3', df)

        # key_1应该被淘汰
        assert cache.get('key_1') is None
        # key_0应该还在
        assert cache.get('key_0') is not None
        # key_3应该在
        assert cache.get('key_3') is not None

    def test_expiration(self):
        """测试过期机制"""
        from core.data_service import LRUDataCache

        cache = LRUDataCache(expire_seconds=1)  # 1秒过期

        df = pd.DataFrame({'value': [1]})
        cache.put('expire_test', df)

        # 立即获取应该成功
        assert cache.get('expire_test') is not None

        # 等待过期
        time.sleep(1.1)

        # 过期后获取应该返回None
        assert cache.get('expire_test') is None

    def test_no_expiration(self):
        """测试不过期模式"""
        from core.data_service import LRUDataCache

        cache = LRUDataCache(expire_seconds=0)  # 0表示不过期

        df = pd.DataFrame({'value': [1]})
        cache.put('no_expire', df)

        # 应该一直能获取
        assert cache.get('no_expire') is not None
        time.sleep(0.1)
        assert cache.get('no_expire') is not None

    def test_clear(self):
        """测试清理缓存"""
        from core.data_service import LRUDataCache

        cache = LRUDataCache()

        for i in range(5):
            df = pd.DataFrame({'value': [i]})
            cache.put(f'key_{i}', df)

        # 清理所有
        cache.clear()

        for i in range(5):
            assert cache.get(f'key_{i}') is None

    def test_clear_with_prefix(self):
        """测试按前缀清理"""
        from core.data_service import LRUDataCache

        cache = LRUDataCache()

        cache.put('RB_1d_2024', pd.DataFrame({'v': [1]}))
        cache.put('RB_5m_2024', pd.DataFrame({'v': [2]}))
        cache.put('AU_1d_2024', pd.DataFrame({'v': [3]}))

        # 只清理RB开头的
        cache.clear(prefix='RB')

        assert cache.get('RB_1d_2024') is None
        assert cache.get('RB_5m_2024') is None
        assert cache.get('AU_1d_2024') is not None

    def test_stats(self):
        """测试统计功能"""
        from core.data_service import LRUDataCache

        cache = LRUDataCache(max_entries=10, max_size_mb=100)

        # 初始统计
        stats = cache.get_stats()
        assert stats['entries'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0

        # 添加数据
        df = pd.DataFrame({'value': np.random.randn(100)})
        cache.put('test', df)

        # 命中
        cache.get('test')
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 0

        # 未命中
        cache.get('nonexistent')
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1

    def test_get_keys(self):
        """测试获取所有键"""
        from core.data_service import LRUDataCache

        cache = LRUDataCache()

        cache.put('key1', pd.DataFrame({'v': [1]}))
        cache.put('key2', pd.DataFrame({'v': [2]}))

        keys = cache.get_keys()
        assert 'key1' in keys
        assert 'key2' in keys

    def test_data_copy(self):
        """测试数据副本（防止原始数据被修改）"""
        from core.data_service import LRUDataCache

        cache = LRUDataCache()

        df = pd.DataFrame({'value': [1, 2, 3]})
        cache.put('test', df)

        # 获取并修改
        result = cache.get('test')
        result['value'] = [10, 20, 30]

        # 再次获取应该是原始值
        result2 = cache.get('test')
        assert list(result2['value']) == [1, 2, 3]

    def test_empty_dataframe_not_cached(self):
        """测试空DataFrame不会被缓存"""
        from core.data_service import LRUDataCache

        cache = LRUDataCache()

        cache.put('empty', pd.DataFrame())

        assert cache.get('empty') is None
