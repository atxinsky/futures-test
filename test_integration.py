# coding=utf-8
"""
整合测试脚本 - 测试OptunaOptimizer整合到param_optimizer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import traceback

results = []

def test(name):
    """测试装饰器"""
    def decorator(func):
        def wrapper():
            try:
                result = func()
                results.append({"name": name, "status": "PASS", "detail": result or "OK"})
                print(f"[PASS] {name}")
                return True
            except Exception as e:
                results.append({"name": name, "status": "FAIL", "detail": str(e)})
                print(f"[FAIL] {name}: {e}")
                traceback.print_exc()
                return False
        return wrapper
    return decorator


@test("导入optimization模块")
def test_import_optimization():
    from optimization import (
        OptunaOptimizer, OptimizationConfig, ParamSpaceManager,
        ConfigApplier, OptimizationResult, ParamSpace
    )
    return "所有类导入成功"


@test("导入param_optimizer页面")
def test_import_page():
    from app.pages.param_optimizer import render_param_optimizer_page
    return "页面导入成功"


@test("ParamSpaceManager.get_supported_strategies")
def test_param_space_strategies():
    from optimization import ParamSpaceManager
    strategies = ParamSpaceManager.get_supported_strategies()
    assert len(strategies) > 0, "没有支持的策略"
    assert "brother2v6" in strategies, "brother2v6不在支持列表"
    return f"支持 {len(strategies)} 个策略: {strategies}"


@test("ParamSpaceManager.get_all_params(brother2v6)")
def test_param_space_all():
    from optimization import ParamSpaceManager
    params = ParamSpaceManager.get_all_params("brother2v6")
    assert len(params) > 0, "没有参数"
    param_names = list(params.keys())
    return f"{len(params)}个参数: {param_names[:5]}..."


@test("ParamSpaceManager.get_key_params(brother2v6)")
def test_param_space_key():
    from optimization import ParamSpaceManager
    params = ParamSpaceManager.get_key_params("brother2v6")
    assert len(params) > 0, "没有关键参数"
    return f"{len(params)}个关键参数"


@test("ParamSpaceManager.get_param_groups(brother2v6)")
def test_param_groups():
    from optimization import ParamSpaceManager
    groups = ParamSpaceManager.get_param_groups("brother2v6")
    assert len(groups) > 0, "没有参数分组"
    return f"{len(groups)}个分组: {list(groups.keys())}"


@test("OptimizationConfig创建")
def test_config_create():
    from optimization import OptimizationConfig
    config = OptimizationConfig(
        strategy_name="brother2v6",
        symbols=["RB", "I"],
        train_start="2020-01-01",
        train_end="2023-12-31",
        val_start="2024-01-01",
        val_end="2024-12-31",
        timeframe="1h",
        n_trials=10,
        objective="sharpe"
    )
    assert config.strategy_name == "brother2v6"
    assert config.n_trials == 10
    return f"配置创建成功: {config.strategy_name}, {config.symbols}"


@test("OptunaOptimizer初始化")
def test_optimizer_init():
    from optimization import OptunaOptimizer, OptimizationConfig
    config = OptimizationConfig(
        strategy_name="brother2v6",
        symbols=["RB"],
        train_start="2020-01-01",
        train_end="2023-12-31",
        val_start="2024-01-01",
        val_end="2024-12-31"
    )
    optimizer = OptunaOptimizer(config)
    assert optimizer is not None
    return "优化器初始化成功"


@test("加载期货数据(RB)")
def test_load_data():
    from utils.data_loader import load_futures_data
    df = load_futures_data("RB", "2024-01-01", "2024-03-31", auto_download=False)
    if df is None or len(df) == 0:
        return "无本地数据(跳过)"
    return f"加载 {len(df)} 行数据"


@test("Brother2v6策略类导入")
def test_strategy_import():
    from strategies.brother2v6 import Brother2v6Strategy
    strategy = Brother2v6Strategy(params={
        'sml_len': 12,
        'big_len': 50,
        'break_len': 30,
        'adx_thres': 22,
        'chop_thres': 50
    })
    assert strategy is not None
    return "策略类创建成功"


@test("回测引擎导入")
def test_backtest_engine():
    from core.backtest_engine import BacktestEngine
    engine = BacktestEngine()
    assert engine is not None
    return "回测引擎创建成功"


@test("ConfigApplier静态方法测试")
def test_config_applier():
    from optimization import ConfigApplier
    # ConfigApplier是静态方法类，测试方法存在
    assert hasattr(ConfigApplier, 'save_optimized_config')
    assert hasattr(ConfigApplier, 'apply_to_session_state')
    assert hasattr(ConfigApplier, 'load_optimized_params')
    return "ConfigApplier static methods OK"


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("OptunaOptimizer Integration Test")
    print("=" * 60)
    print()

    # 执行所有测试
    test_import_optimization()
    test_import_page()
    test_param_space_strategies()
    test_param_space_all()
    test_param_space_key()
    test_param_groups()
    test_config_create()
    test_optimizer_init()
    test_load_data()
    test_strategy_import()
    test_backtest_engine()
    test_config_applier()

    # 统计结果
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")

    print()
    print("=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_all_tests()
