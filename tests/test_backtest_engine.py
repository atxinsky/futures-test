# coding=utf-8
"""
回测引擎集成测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class MockStrategy:
    """模拟策略，用于测试"""

    name = "mock_strategy"
    warmup_num = 10

    def __init__(self, params=None):
        self.params = params or {}
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self._signal_idx = 0
        self._signals = []

    def set_signals(self, signals):
        """设置预定义信号序列"""
        self._signals = signals
        self._signal_idx = 0

    def reset(self):
        """重置策略状态"""
        self.position = 0
        self.entry_price = 0
        self.entry_time = None
        self._signal_idx = 0

    def calculate_indicators(self, df):
        """计算指标（直接返回）"""
        return df

    def on_bar(self, idx, df, capital):
        """返回预定义信号"""
        from strategies.base import Signal

        if self._signal_idx < len(self._signals):
            signal_def = self._signals[self._signal_idx]
            if signal_def.get('idx') == idx:
                self._signal_idx += 1
                return Signal(
                    action=signal_def['action'],
                    price=df.iloc[idx]['close'],
                    tag=signal_def.get('tag', '')
                )
        return None


class MockDataLoader:
    """模拟数据加载器"""

    def __init__(self, data=None):
        self._data = data

    def set_data(self, data):
        self._data = data

    def load_bars(self, symbol, period, start_date=None, end_date=None):
        """返回模拟数据"""
        if self._data is not None:
            return self._data

        # 生成默认测试数据
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        df = pd.DataFrame({
            'time': dates,
            'open': np.random.uniform(3000, 3200, 100),
            'high': np.random.uniform(3100, 3300, 100),
            'low': np.random.uniform(2900, 3100, 100),
            'close': np.random.uniform(3000, 3200, 100),
            'volume': np.random.randint(1000, 10000, 100)
        })
        return df


class TestBacktestEngine:
    """回测引擎测试"""

    def test_init(self):
        """测试初始化"""
        from core.backtest_engine import BacktestEngine

        engine = BacktestEngine()
        assert engine is not None
        assert engine.data_loader is not None
        assert 'RB' in engine.instrument_configs

    def test_set_instrument_config(self):
        """测试设置品种配置"""
        from core.backtest_engine import BacktestEngine

        engine = BacktestEngine()
        engine.set_instrument_config('TEST', {
            'multiplier': 100,
            'margin_rate': 0.15,
            'commission': 5
        })

        assert 'TEST' in engine.instrument_configs
        assert engine.instrument_configs['TEST']['multiplier'] == 100

    def test_run_with_mock_data(self):
        """测试使用模拟数据运行回测"""
        from core.backtest_engine import BacktestEngine

        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'time': dates,
            'open': 3000 + np.arange(50) * 10,
            'high': 3050 + np.arange(50) * 10,
            'low': 2950 + np.arange(50) * 10,
            'close': 3000 + np.arange(50) * 10,
            'volume': [5000] * 50
        })

        # 创建模拟数据加载器
        loader = MockDataLoader(df)
        engine = BacktestEngine(data_loader=loader)

        # 创建策略并设置信号
        strategy = MockStrategy()
        strategy.set_signals([
            {'idx': 15, 'action': 'buy', 'tag': '做多'},
            {'idx': 30, 'action': 'close', 'tag': '平仓'},
        ])

        # 运行回测
        result = engine.run(
            strategy=strategy,
            symbol='RB',
            period='1d',
            initial_capital=100000,
            volume=1
        )

        assert result is not None
        assert result.strategy_name == 'mock_strategy'
        assert result.symbol == 'RB'
        assert result.initial_capital == 100000

    def test_run_generates_trades(self):
        """测试回测生成交易记录"""
        from core.backtest_engine import BacktestEngine

        # 创建固定价格数据
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        prices = [3000.0] * 15 + [3100.0] * 20 + [3050.0] * 15
        df = pd.DataFrame({
            'time': dates,
            'open': prices,
            'high': [p + 50 for p in prices],
            'low': [p - 50 for p in prices],
            'close': prices,
            'volume': [5000] * 50
        })

        loader = MockDataLoader(df)
        engine = BacktestEngine(data_loader=loader)

        strategy = MockStrategy()
        strategy.set_signals([
            {'idx': 15, 'action': 'buy'},
            {'idx': 35, 'action': 'close'},
        ])

        result = engine.run(
            strategy=strategy,
            symbol='RB',
            period='1d',
            initial_capital=100000,
            volume=1
        )

        assert result.total_trades >= 1
        assert len(result.trades) >= 1

    def test_run_empty_data(self):
        """测试空数据处理"""
        from core.backtest_engine import BacktestEngine

        loader = MockDataLoader(pd.DataFrame())
        engine = BacktestEngine(data_loader=loader)

        strategy = MockStrategy()
        result = engine.run(
            strategy=strategy,
            symbol='RB',
            period='1d',
            initial_capital=100000
        )

        # 应该返回默认结果而不是报错
        assert result is not None
        assert result.total_trades == 0


class TestBacktestResult:
    """回测结果测试"""

    def test_result_attributes(self):
        """测试结果属性"""
        from models.backtest_models import BacktestResult

        result = BacktestResult(
            strategy_name='test',
            symbol='RB',
            period='1d',
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_capital=100000
        )

        assert result.strategy_name == 'test'
        assert result.initial_capital == 100000
        assert result.total_trades == 0
        assert result.win_rate == 0

    def test_result_with_trades(self):
        """测试含交易的结果"""
        from models.backtest_models import BacktestResult, BacktestTrade

        trades = [
            BacktestTrade(
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                symbol='RB',
                direction=1,
                entry_price=3000,
                exit_price=3100,
                volume=1,
                pnl=1000,
                pnl_pct=0.033,
                holding_bars=10
            ),
            BacktestTrade(
                entry_time=datetime.now(),
                exit_time=datetime.now(),
                symbol='RB',
                direction=1,
                entry_price=3100,
                exit_price=3050,
                volume=1,
                pnl=-500,
                pnl_pct=-0.016,
                holding_bars=5
            )
        ]

        result = BacktestResult(
            strategy_name='test',
            symbol='RB',
            period='1d',
            start_date=datetime.now(),
            end_date=datetime.now(),
            initial_capital=100000,
            trades=trades
        )

        assert len(result.trades) == 2


class TestGenerateReport:
    """报告生成测试"""

    def test_generate_report(self):
        """测试生成报告"""
        from core.backtest_engine import generate_report
        from models.backtest_models import BacktestResult

        result = BacktestResult(
            strategy_name='TestStrategy',
            symbol='RB',
            period='1d',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            initial_capital=100000,
            final_capital=120000,
            total_return=0.2,
            annual_return=0.2,
            max_drawdown_pct=0.1,
            sharpe_ratio=1.5,
            total_trades=50,
            win_rate=0.6
        )

        report = generate_report(result)

        assert '回测报告' in report
        assert 'TestStrategy' in report
        assert 'RB' in report
        assert '100,000' in report


class TestMultiSymbolBacktest:
    """多品种回测测试"""

    def test_run_multi_symbol(self):
        """测试多品种回测"""
        from core.backtest_engine import BacktestEngine

        # 使用空数据加载器（避免实际加载数据）
        loader = MockDataLoader()
        engine = BacktestEngine(data_loader=loader)

        strategy = MockStrategy()

        results = engine.run_multi_symbol(
            strategy=strategy,
            symbols=['RB', 'AU'],
            period='1d',
            initial_capital=100000
        )

        assert isinstance(results, dict)
        assert 'RB' in results
        assert 'AU' in results
