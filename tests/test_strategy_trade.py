# coding=utf-8
"""
测试StrategyTrade模型和TradeManager

测试场景：
1. 单次开仓 + 单次平仓
2. 分批开仓（2笔成交）+ 单次平仓
3. 单次开仓 + 分批平仓（2笔成交）
4. 分批开仓 + 分批平仓
5. 盈亏计算验证
"""

import sys
import os

# 直接添加项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from datetime import datetime

# 直接导入模型，避免循环依赖
from models.base import (
    StrategyTrade, Trade, Direction, Offset, TradeStatus
)

# 直接导入trade_manager，不通过trading包
sys.path.insert(0, os.path.join(project_root, 'trading'))
from trade_manager import TradeManager


def test_single_open_close():
    """测试1: 单次开仓 + 单次平仓"""
    print("\n" + "="*60)
    print("测试1: 单次开仓 + 单次平仓")
    print("="*60)

    manager = TradeManager()
    manager.set_multiplier("RB2505", 10)  # 螺纹钢乘数10

    # 创建交易：做多螺纹钢2手
    trade = manager.create_trade(
        strategy_name="brother2v6",
        symbol="RB2505",
        direction=Direction.LONG,
        shares=2,
        stop_loss=3200,
        take_profit=3500,
        entry_tag="突破入场"
    )

    print(f"创建交易: {trade.trade_id}")
    print(f"  状态: {trade.status.value}")
    print(f"  计划手数: {trade.shares}")

    # 模拟开仓成交
    fill1 = Trade(
        trade_id="fill_001",
        order_id="order_001",
        symbol="RB2505",
        exchange="SHFE",
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=3300,
        volume=2,
        commission=10,
        strategy_name="brother2v6"
    )
    manager.add_open_order(trade.trade_id, "order_001")
    manager.process_fill(fill1)

    print(f"\n开仓成交后:")
    print(f"  状态: {trade.status.value}")
    print(f"  持仓: {trade.holding_shares}手")
    print(f"  入场均价: {trade.avg_entry_price}")

    # 更新价格
    manager.update_price("RB2505", 3400)
    print(f"\n价格更新到3400后:")
    print(f"  未实现盈亏: {trade.unrealized_pnl}")

    # 模拟平仓成交
    fill2 = Trade(
        trade_id="fill_002",
        order_id="order_002",
        symbol="RB2505",
        exchange="SHFE",
        direction=Direction.SHORT,
        offset=Offset.CLOSE,
        price=3400,
        volume=2,
        commission=10,
        strategy_name="brother2v6"
    )
    manager.add_close_order(trade.trade_id, "order_002")
    manager.process_fill(fill2)

    print(f"\n平仓成交后:")
    print(f"  状态: {trade.status.value}")
    print(f"  持仓: {trade.holding_shares}手")
    print(f"  出场均价: {trade.avg_exit_price}")
    print(f"  已实现盈亏: {trade.realized_pnl}")
    print(f"  手续费: {trade.commission}")
    print(f"  净盈亏: {trade.total_pnl}")

    # 验证
    expected_pnl = (3400 - 3300) * 2 * 10 - 20  # 价差 * 手数 * 乘数 - 手续费
    assert trade.status == TradeStatus.CLOSED, "状态应为CLOSED"
    assert trade.total_pnl == expected_pnl, f"盈亏计算错误: {trade.total_pnl} != {expected_pnl}"
    print(f"\n验证通过! 预期盈亏={expected_pnl}, 实际盈亏={trade.total_pnl}")


def test_partial_open():
    """测试2: 分批开仓（2笔成交）+ 单次平仓"""
    print("\n" + "="*60)
    print("测试2: 分批开仓（2笔成交）+ 单次平仓")
    print("="*60)

    manager = TradeManager()
    manager.set_multiplier("AU2506", 1000)  # 黄金乘数1000

    # 创建交易：做多黄金5手
    trade = manager.create_trade(
        strategy_name="brother2v6",
        symbol="AU2506",
        direction=Direction.LONG,
        shares=5,
        entry_tag="趋势入场"
    )

    # 第一笔开仓成交：3手 @ 510
    fill1 = Trade(
        trade_id="fill_101",
        order_id="order_101",
        symbol="AU2506",
        exchange="SHFE",
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=510,
        volume=3,
        commission=30,
        strategy_name="brother2v6"
    )
    manager.add_open_order(trade.trade_id, "order_101")
    manager.process_fill(fill1)

    print(f"\n第一笔开仓成交 (3手@510):")
    print(f"  状态: {trade.status.value}")
    print(f"  已成交: {trade.filled_shares}手")
    print(f"  入场均价: {trade.avg_entry_price}")

    # 第二笔开仓成交：2手 @ 512
    fill2 = Trade(
        trade_id="fill_102",
        order_id="order_101",  # 同一订单
        symbol="AU2506",
        exchange="SHFE",
        direction=Direction.LONG,
        offset=Offset.OPEN,
        price=512,
        volume=2,
        commission=20,
        strategy_name="brother2v6"
    )
    manager.process_fill(fill2)

    print(f"\n第二笔开仓成交 (2手@512):")
    print(f"  状态: {trade.status.value}")
    print(f"  已成交: {trade.filled_shares}手")
    print(f"  加权入场均价: {trade.avg_entry_price:.2f}")

    # 验证加权均价: (510*3 + 512*2) / 5 = 510.8
    expected_avg = (510 * 3 + 512 * 2) / 5
    assert abs(trade.avg_entry_price - expected_avg) < 0.01, f"加权均价错误: {trade.avg_entry_price} != {expected_avg}"

    # 平仓成交：5手 @ 520
    fill3 = Trade(
        trade_id="fill_103",
        order_id="order_102",
        symbol="AU2506",
        exchange="SHFE",
        direction=Direction.SHORT,
        offset=Offset.CLOSE,
        price=520,
        volume=5,
        commission=50,
        strategy_name="brother2v6"
    )
    manager.add_close_order(trade.trade_id, "order_102")
    manager.process_fill(fill3)

    print(f"\n全部平仓成交 (5手@520):")
    print(f"  状态: {trade.status.value}")
    print(f"  出场均价: {trade.avg_exit_price}")
    print(f"  已实现盈亏: {trade.realized_pnl}")
    print(f"  手续费: {trade.commission}")
    print(f"  净盈亏: {trade.total_pnl}")

    # 验证盈亏: (520 - 510.8) * 5 * 1000 - 100
    expected_pnl = (520 - expected_avg) * 5 * 1000 - 100
    assert abs(trade.total_pnl - expected_pnl) < 0.01, f"盈亏计算错误: {trade.total_pnl} != {expected_pnl}"
    print(f"\n验证通过! 预期盈亏={expected_pnl:.2f}, 实际盈亏={trade.total_pnl:.2f}")


def test_partial_close():
    """测试3: 单次开仓 + 分批平仓（2笔成交）"""
    print("\n" + "="*60)
    print("测试3: 单次开仓 + 分批平仓（2笔成交）")
    print("="*60)

    manager = TradeManager()
    manager.set_multiplier("IF2501", 300)  # 股指期货乘数300

    # 创建交易：做空IF 4手
    trade = manager.create_trade(
        strategy_name="brother2v6",
        symbol="IF2501",
        direction=Direction.SHORT,
        shares=4,
        entry_tag="做空信号"
    )

    # 开仓成交：4手 @ 4000
    fill1 = Trade(
        trade_id="fill_201",
        order_id="order_201",
        symbol="IF2501",
        exchange="CFFEX",
        direction=Direction.SHORT,
        offset=Offset.OPEN,
        price=4000,
        volume=4,
        commission=120,
        strategy_name="brother2v6"
    )
    manager.add_open_order(trade.trade_id, "order_201")
    manager.process_fill(fill1)

    print(f"\n开仓成交 (做空4手@4000):")
    print(f"  状态: {trade.status.value}")
    print(f"  持仓: {trade.holding_shares}手")

    # 第一笔平仓：2手 @ 3900 (止盈)
    fill2 = Trade(
        trade_id="fill_202",
        order_id="order_202",
        symbol="IF2501",
        exchange="CFFEX",
        direction=Direction.LONG,
        offset=Offset.CLOSE,
        price=3900,
        volume=2,
        commission=60,
        strategy_name="brother2v6"
    )
    manager.add_close_order(trade.trade_id, "order_202")
    manager.process_fill(fill2)

    print(f"\n第一笔平仓 (2手@3900):")
    print(f"  状态: {trade.status.value}")
    print(f"  剩余持仓: {trade.holding_shares}手")
    print(f"  已实现盈亏: {trade.realized_pnl}")

    # 验证部分平仓盈亏: (4000 - 3900) * 2 * 300 = 60000
    expected_partial_pnl = (4000 - 3900) * 2 * 300
    assert abs(trade.realized_pnl - expected_partial_pnl) < 0.01

    # 第二笔平仓：2手 @ 3950 (剩余止盈)
    fill3 = Trade(
        trade_id="fill_203",
        order_id="order_203",
        symbol="IF2501",
        exchange="CFFEX",
        direction=Direction.LONG,
        offset=Offset.CLOSE,
        price=3950,
        volume=2,
        commission=60,
        strategy_name="brother2v6"
    )
    manager.add_close_order(trade.trade_id, "order_203")
    manager.process_fill(fill3)

    print(f"\n第二笔平仓 (2手@3950):")
    print(f"  状态: {trade.status.value}")
    print(f"  剩余持仓: {trade.holding_shares}手")
    print(f"  出场均价: {trade.avg_exit_price}")
    print(f"  已实现盈亏: {trade.realized_pnl}")
    print(f"  手续费: {trade.commission}")
    print(f"  净盈亏: {trade.total_pnl}")

    # 验证总盈亏
    # 做空盈亏: (入场价 - 出场价) * 手数 * 乘数
    # = (4000 - 3900) * 2 * 300 + (4000 - 3950) * 2 * 300 = 60000 + 30000 = 90000
    # 净盈亏 = 90000 - 240 (手续费) = 89760
    expected_pnl = 90000 - 240
    assert abs(trade.total_pnl - expected_pnl) < 0.01, f"盈亏计算错误: {trade.total_pnl} != {expected_pnl}"
    print(f"\n验证通过! 预期净盈亏={expected_pnl}, 实际净盈亏={trade.total_pnl}")


def test_manager_statistics():
    """测试4: TradeManager统计功能"""
    print("\n" + "="*60)
    print("测试4: TradeManager统计功能")
    print("="*60)

    manager = TradeManager()
    manager.set_multiplier("RB2505", 10)

    # 创建3笔交易：2盈1亏
    for i, (entry, exit_price) in enumerate([(3300, 3400), (3350, 3450), (3400, 3350)]):
        trade = manager.create_trade(
            strategy_name="test_strategy",
            symbol="RB2505",
            direction=Direction.LONG,
            shares=1,
            entry_tag=f"test_{i}"
        )

        # 开仓
        fill_open = Trade(
            trade_id=f"fill_open_{i}",
            order_id=f"order_open_{i}",
            symbol="RB2505",
            exchange="SHFE",
            direction=Direction.LONG,
            offset=Offset.OPEN,
            price=entry,
            volume=1,
            commission=5,
            strategy_name="test_strategy"
        )
        manager.add_open_order(trade.trade_id, f"order_open_{i}")
        manager.process_fill(fill_open)

        # 平仓
        fill_close = Trade(
            trade_id=f"fill_close_{i}",
            order_id=f"order_close_{i}",
            symbol="RB2505",
            exchange="SHFE",
            direction=Direction.SHORT,
            offset=Offset.CLOSE,
            price=exit_price,
            volume=1,
            commission=5,
            strategy_name="test_strategy"
        )
        manager.add_close_order(trade.trade_id, f"order_close_{i}")
        manager.process_fill(fill_close)

    stats = manager.get_statistics()
    print(f"\n统计结果:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 验证
    assert stats['closed_trades'] == 3, "应有3笔已平仓交易"
    assert stats['winning_trades'] == 2, "应有2笔盈利"
    assert stats['losing_trades'] == 1, "应有1笔亏损"
    assert abs(stats['win_rate'] - 2/3) < 0.01, "胜率应为66.7%"

    print("\n验证通过! 统计功能正常")


def test_trade_lifecycle():
    """测试5: 交易完整生命周期"""
    print("\n" + "="*60)
    print("测试5: 交易完整生命周期状态变化")
    print("="*60)

    manager = TradeManager()
    manager.set_multiplier("CU2503", 5)

    trade = manager.create_trade(
        strategy_name="test",
        symbol="CU2503",
        direction=Direction.LONG,
        shares=3,
        entry_tag="test"
    )

    states = []

    # 初始状态
    states.append(trade.status.value)
    print(f"1. 创建后: {trade.status.value}")

    # 部分开仓
    fill1 = Trade(
        trade_id="f1", order_id="o1", symbol="CU2503", exchange="SHFE",
        direction=Direction.LONG, offset=Offset.OPEN, price=70000, volume=1,
        commission=10, strategy_name="test"
    )
    manager.add_open_order(trade.trade_id, "o1")
    manager.process_fill(fill1)
    states.append(trade.status.value)
    print(f"2. 部分开仓后 (1/3): {trade.status.value}")

    # 完成开仓
    fill2 = Trade(
        trade_id="f2", order_id="o1", symbol="CU2503", exchange="SHFE",
        direction=Direction.LONG, offset=Offset.OPEN, price=70100, volume=2,
        commission=20, strategy_name="test"
    )
    manager.process_fill(fill2)
    states.append(trade.status.value)
    print(f"3. 完成开仓后 (3/3): {trade.status.value}")

    # 部分平仓
    fill3 = Trade(
        trade_id="f3", order_id="o2", symbol="CU2503", exchange="SHFE",
        direction=Direction.SHORT, offset=Offset.CLOSE, price=71000, volume=2,
        commission=20, strategy_name="test"
    )
    manager.add_close_order(trade.trade_id, "o2")
    manager.process_fill(fill3)
    states.append(trade.status.value)
    print(f"4. 部分平仓后 (平2/持3): {trade.status.value}")

    # 完成平仓
    fill4 = Trade(
        trade_id="f4", order_id="o3", symbol="CU2503", exchange="SHFE",
        direction=Direction.SHORT, offset=Offset.CLOSE, price=71200, volume=1,
        commission=10, strategy_name="test"
    )
    manager.add_close_order(trade.trade_id, "o3")
    manager.process_fill(fill4)
    states.append(trade.status.value)
    print(f"5. 完成平仓后 (平3/持3): {trade.status.value}")

    # 验证状态变化
    expected_states = ['pending', 'opening', 'holding', 'closing', 'closed']
    assert states == expected_states, f"状态变化错误: {states} != {expected_states}"

    print(f"\n验证通过! 生命周期状态变化: {' -> '.join(states)}")


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("StrategyTrade模型测试")
    print("="*60)

    try:
        test_single_open_close()
        test_partial_open()
        test_partial_close()
        test_manager_statistics()
        test_trade_lifecycle()

        print("\n" + "="*60)
        print("所有测试通过!")
        print("="*60)
    except AssertionError as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n测试异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
