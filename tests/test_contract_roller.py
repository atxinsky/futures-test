# coding=utf-8
"""
合约换月模块测试
"""

import sys
import os

# 直接导入模块，避免触发utils/__init__.py
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import importlib.util
spec = importlib.util.spec_from_file_location(
    "contract_roller",
    os.path.join(project_root, "utils", "contract_roller.py")
)
contract_roller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract_roller)

ContractRoller = contract_roller.ContractRoller
RollConfig = contract_roller.RollConfig
RollReason = contract_roller.RollReason


def test_parse_contract():
    """测试合约代码解析"""
    roller = ContractRoller()

    # 4位月份
    product, month = roller.parse_contract('RB2505')
    assert product == 'RB'
    assert month == 2505

    # 小写
    product, month = roller.parse_contract('au2506')
    assert product == 'AU'
    assert month == 2506

    # 3位月份
    product, month = roller.parse_contract('rb501')
    assert product == 'RB'
    assert month == 2501

    print("test_parse_contract PASSED")


def test_update_contract_info():
    """测试合约信息更新"""
    roller = ContractRoller()

    roller.update_contract_info('RB2501', volume=50000, open_interest=100000, price=3800)
    roller.update_contract_info('RB2505', volume=80000, open_interest=150000, price=3850)

    assert 'RB2501' in roller._contracts
    assert 'RB2505' in roller._contracts
    assert roller._contracts['RB2501'].volume == 50000
    assert roller._contracts['RB2505'].last_price == 3850

    print("test_update_contract_info PASSED")


def test_detect_main_contract():
    """测试主力合约检测"""
    roller = ContractRoller()

    # 添加合约信息（使用2026年合约，因为当前是2026年1月）
    roller.update_contract_info('RB2601', volume=50000, open_interest=100000, price=3800)
    roller.update_contract_info('RB2605', volume=80000, open_interest=150000, price=3850)
    roller.update_contract_info('RB2609', volume=20000, open_interest=50000, price=3900)

    # 检测主力（成交量+持仓量最大的）
    main = roller.detect_main_contract('RB')
    assert main == 'RB2605', f"预期 RB2605，实际 {main}"

    print("test_detect_main_contract PASSED")


def test_should_roll():
    """测试换月判断"""
    roller = ContractRoller()

    # 设置合约信息（使用2026年合约）
    roller.update_contract_info('RB2601', volume=50000, open_interest=100000, price=3800)
    roller.update_contract_info('RB2605', volume=80000, open_interest=150000, price=3850)

    # 设置主力合约
    roller.set_main_contract('RB', 'RB2605')

    # 持有旧合约，应该换月
    should, new_contract, reason = roller.should_roll('RB', 'RB2601', check_expiry=False)
    assert should == True
    assert new_contract == 'RB2605'
    assert reason == RollReason.MAIN_CONTRACT_CHANGE

    # 持有主力合约，不需要换月
    should, new_contract, reason = roller.should_roll('RB', 'RB2605', check_expiry=False)
    assert should == False

    print("test_should_roll PASSED")


def test_generate_roll_signals():
    """测试换月信号生成"""
    roller = ContractRoller()

    # 设置合约信息（使用2026年合约）
    roller.update_contract_info('RB2601', volume=50000, open_interest=100000, price=3800)
    roller.update_contract_info('RB2605', volume=80000, open_interest=150000, price=3850)
    roller.set_main_contract('RB', 'RB2605')

    # 生成换月信号
    signals = roller.generate_roll_signals('RB', 'RB2601', 'long', 10)

    assert len(signals) == 1
    signal = signals[0]
    assert signal.product == 'RB'
    assert signal.old_contract == 'RB2601'
    assert signal.new_contract == 'RB2605'
    assert signal.direction == 'long'
    assert signal.volume == 10
    assert signal.basis == 50  # 3850 - 3800

    print("test_generate_roll_signals PASSED")


def test_get_next_main_month():
    """测试下一主力月份"""
    roller = ContractRoller()

    # 默认品种（1,5,9月）
    next_month = roller.get_next_main_month('RB', 2601)
    assert next_month == 2605

    next_month = roller.get_next_main_month('RB', 2609)
    assert next_month == 2701  # 跨年

    # 股指（每月）
    next_month = roller.get_next_main_month('IF', 2603)
    assert next_month == 2604

    print("test_get_next_main_month PASSED")


def test_roll_callback():
    """测试换月回调"""
    roller = ContractRoller()

    callback_received = []

    def on_roll(signal):
        callback_received.append(signal)

    roller.set_roll_callback(on_roll)

    # 设置合约并生成信号（使用2026年合约）
    roller.update_contract_info('RB2601', volume=50000, open_interest=100000, price=3800)
    roller.update_contract_info('RB2605', volume=80000, open_interest=150000, price=3850)
    roller.set_main_contract('RB', 'RB2605')

    roller.generate_roll_signals('RB', 'RB2601', 'short', 5)

    assert len(callback_received) == 1
    assert callback_received[0].direction == 'short'

    print("test_roll_callback PASSED")


def test_statistics():
    """测试统计功能"""
    roller = ContractRoller()

    # 使用2026年合约
    roller.update_contract_info('RB2601', volume=50000, open_interest=100000, price=3800)
    roller.update_contract_info('RB2605', volume=80000, open_interest=150000, price=3850)
    roller.set_main_contract('RB', 'RB2605')

    # 生成信号
    signals = roller.generate_roll_signals('RB', 'RB2601', 'long', 10)

    # 标记处理
    roller.mark_signal_processed(signals[0])

    stats = roller.get_statistics()
    assert stats['pending_signals'] == 0
    assert stats['processed_signals'] == 1
    assert stats['main_contracts']['RB'] == 'RB2605'

    print("test_statistics PASSED")


if __name__ == "__main__":
    test_parse_contract()
    test_update_contract_info()
    test_detect_main_contract()
    test_should_roll()
    test_generate_roll_signals()
    test_get_next_main_month()
    test_roll_callback()
    test_statistics()

    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)
