# coding=utf-8
"""
模拟交易Web界面模块
界面风格与回测系统一致，提供策略配置和模拟交易功能
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List
import logging
import json
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import INSTRUMENTS, get_instrument
from strategies import get_all_strategies, get_strategy

logger = logging.getLogger(__name__)

# 配置文件路径
CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tq_config.json")


def load_tq_config() -> dict:
    """加载TqSdk配置"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        'tq_user': '',
        'tq_password': '',
        'sim_mode': True,
        'default_symbols': ['RB', 'AU', 'IF'],
        'initial_capital': 100000
    }


def save_tq_config(config: dict):
    """保存TqSdk配置"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_sim_engine():
    """获取模拟交易引擎"""
    if 'sim_engine' not in st.session_state:
        st.session_state.sim_engine = None
    return st.session_state.sim_engine


def render_sim_trading_page():
    """渲染模拟交易页面"""
    st.title("模拟交易")

    # 检查TqSdk
    try:
        import tqsdk
        tqsdk_installed = True
    except ImportError:
        tqsdk_installed = False

    if not tqsdk_installed:
        st.error("TqSdk未安装，请执行: `pip install tqsdk`")
        return

    # 选项卡（连接设置已移到系统设置）
    tab1, tab2, tab3 = st.tabs(["交易配置", "持仓监控", "订单记录"])

    with tab1:
        render_trading_config()

    with tab2:
        render_sim_positions()

    with tab3:
        render_sim_orders()


def render_trading_config():
    """渲染交易配置 - 回测风格"""
    engine = get_sim_engine()
    is_running = engine is not None and engine.is_running if engine else False

    # 运行状态栏
    if is_running:
        render_running_status()
        st.markdown("---")

    # ========== 三列布局：基础设置 | 策略参数 | 合约信息 ==========
    strategies = get_all_strategies()
    strategy_names = list(strategies.keys())
    strategy_display = {k: v.display_name for k, v in strategies.items()}

    # 默认选择brother2v6
    default_idx = strategy_names.index('brother2v6') if 'brother2v6' in strategy_names else 0

    col_settings, col_params, col_info = st.columns([1, 1.5, 0.8])

    # ========== 左列：基础设置 ==========
    with col_settings:
        st.subheader("交易设置")

        # 策略选择
        selected_strategy_name = st.selectbox(
            "选择策略",
            options=strategy_names,
            index=default_idx,
            format_func=lambda x: f"{strategy_display.get(x, x)}",
            disabled=is_running
        )
        strategy_class = strategies[selected_strategy_name]

        # 品种选择
        symbol = st.selectbox(
            "选择品种",
            options=list(INSTRUMENTS.keys()),
            format_func=lambda x: f"{x} - {INSTRUMENTS.get(x, {}).get('name', x)}",
            disabled=is_running
        )

        # 时间周期
        timeframe_options = ["日线", "60分钟", "30分钟", "15分钟", "5分钟"]
        time_period = st.selectbox(
            "K线周期",
            options=timeframe_options,
            disabled=is_running
        )

        # 资金设置
        config = load_tq_config()
        initial_capital = st.number_input(
            "初始资金",
            min_value=10000,
            max_value=10000000,
            value=int(config.get('initial_capital', 100000)),
            step=10000,
            disabled=is_running
        )

    # ========== 中列：策略参数 ==========
    with col_params:
        st.subheader(f"{strategy_class.display_name} 参数")
        params = render_strategy_params(strategy_class, is_running)

    # ========== 右列：合约信息 ==========
    with col_info:
        st.subheader("合约规格")

        inst = get_instrument(symbol)
        if inst:
            st.metric("品种", f"{inst['name']}")
            st.metric("合约乘数", f"{inst['multiplier']}")
            st.metric("最小变动", f"{inst['price_tick']}")
            st.metric("保证金率", f"{inst['margin_rate']*100:.0f}%")
            if inst.get('commission_fixed', 0) > 0:
                st.metric("手续费", f"{inst['commission_fixed']}元/手")
            else:
                st.metric("手续费率", f"{inst.get('commission_rate', 0)*10000:.2f}%%")
            st.metric("交易所", inst.get('exchange', '-'))

    # ========== 启动/停止按钮（在三列之后）==========
    st.markdown("---")

    if not is_running:
        if st.button("启动模拟交易", type="primary", use_container_width=True):
            start_sim_trading(selected_strategy_name, symbol, time_period, initial_capital, params)
    else:
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("停止交易", use_container_width=True):
                stop_sim_trading()
        with col2:
            if st.button("刷新状态", use_container_width=True):
                st.rerun()


def render_strategy_params(strategy_class, disabled: bool = False) -> dict:
    """渲染策略参数"""
    params = {}
    param_defs = strategy_class.get_params()

    # 按类型分组参数
    grouped_params = {
        '均线/周期参数': [],
        '风控参数': [],
        '仓位参数': [],
        '其他参数': []
    }

    for p in param_defs:
        if any(k in p.name for k in ['len', 'period', 'ma', 'ema', 'sma', 'fast', 'slow', 'bb', 'macd', 'chop', 'vol']):
            grouped_params['均线/周期参数'].append(p)
        elif any(k in p.name for k in ['stop', 'atr', 'risk', 'adx', 'drawdown', 'trigger', 'break', 'partial', 'full']):
            grouped_params['风控参数'].append(p)
        elif any(k in p.name for k in ['capital', 'risk_rate', 'position']):
            grouped_params['仓位参数'].append(p)
        else:
            grouped_params['其他参数'].append(p)

    # 渲染各组参数
    for group_name, group_params in grouped_params.items():
        if not group_params:
            continue

        with st.expander(group_name, expanded=True):
            cols = st.columns(2)
            for i, p in enumerate(group_params):
                with cols[i % 2]:
                    if p.param_type == 'int':
                        params[p.name] = st.number_input(
                            p.label,
                            min_value=int(p.min_val) if p.min_val else 1,
                            max_value=int(p.max_val) if p.max_val else 100,
                            value=int(p.default),
                            step=int(p.step) if p.step else 1,
                            help=p.description,
                            key=f"sim_param_{p.name}",
                            disabled=disabled
                        )
                    elif p.param_type == 'float':
                        params[p.name] = st.number_input(
                            p.label,
                            min_value=float(p.min_val) if p.min_val else 0.0,
                            max_value=float(p.max_val) if p.max_val else 100.0,
                            value=float(p.default),
                            step=float(p.step) if p.step else 0.01,
                            format="%.2f",
                            help=p.description,
                            key=f"sim_param_{p.name}",
                            disabled=disabled
                        )
                    elif p.param_type == 'bool':
                        params[p.name] = st.checkbox(
                            p.label,
                            value=bool(p.default),
                            help=p.description,
                            key=f"sim_param_{p.name}",
                            disabled=disabled
                        )

    return params


def render_running_status():
    """渲染运行状态"""
    engine = get_sim_engine()
    if not engine:
        return

    st.success("模拟交易运行中")

    # 账户概览
    account = engine.get_account()
    positions = engine.get_positions()

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if account:
            st.metric("账户权益", f"¥{account.balance:,.0f}")
        else:
            st.metric("账户权益", "-")

    with col2:
        if account:
            st.metric("可用资金", f"¥{account.available:,.0f}")
        else:
            st.metric("可用资金", "-")

    with col3:
        if account:
            pnl = account.balance - account.balance  # 需要记录初始资金
            st.metric("浮动盈亏", f"¥{pnl:+,.0f}")
        else:
            st.metric("浮动盈亏", "-")

    with col4:
        st.metric("持仓数量", f"{len(positions)}")

    with col5:
        orders = engine.get_orders() if engine else []
        st.metric("活动订单", f"{len(orders)}")

    # 最近信号
    st.markdown("---")
    st.write("**最近信号**")

    # 从session获取信号历史
    signals = st.session_state.get('sim_signals', [])
    if signals:
        for sig in signals[-5:]:
            st.write(f"  {sig['time']} | {sig['symbol']} | {sig['action']} @ {sig['price']}")
    else:
        st.caption("暂无信号")


def render_sim_positions():
    """渲染持仓监控"""
    st.subheader("持仓监控")

    engine = get_sim_engine()
    is_running = engine is not None and engine.is_running if engine else False

    if not is_running:
        st.info("请先启动模拟交易")
        return

    positions = engine.get_positions()

    if not positions:
        st.info("当前无持仓")
        return

    # 持仓汇总
    total_margin = sum(p.margin for p in positions)
    total_pnl = sum(p.unrealized_pnl for p in positions)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("持仓品种", f"{len(positions)}")
    with col2:
        st.metric("占用保证金", f"¥{total_margin:,.0f}")
    with col3:
        st.metric("浮动盈亏", f"¥{total_pnl:+,.0f}")

    st.markdown("---")

    # 持仓明细
    positions_data = []
    for pos in positions:
        inst = get_instrument(pos.symbol)
        inst_name = inst['name'] if inst else pos.symbol

        positions_data.append({
            '品种': f"{pos.symbol} - {inst_name}",
            '方向': '多' if pos.direction.value == 'long' else '空',
            '数量': pos.volume,
            '开仓价': f"{pos.avg_price:.2f}",
            '现价': f"{pos.last_price:.2f}" if pos.last_price else '-',
            '浮盈': f"¥{pos.unrealized_pnl:+,.0f}",
            '保证金': f"¥{pos.margin:,.0f}"
        })

    df = pd.DataFrame(positions_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # 操作按钮
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("平多头", use_container_width=True):
            close_positions('long')

    with col2:
        if st.button("平空头", use_container_width=True):
            close_positions('short')

    with col3:
        if st.button("全部平仓", type="primary", use_container_width=True):
            close_all_positions()


def render_sim_orders():
    """渲染订单记录"""
    st.subheader("订单记录")

    engine = get_sim_engine()
    is_running = engine is not None and engine.is_running if engine else False

    if not is_running:
        st.info("请先启动模拟交易")
        return

    # 活动订单
    st.write("**活动订单**")
    orders = engine.get_orders()

    if orders:
        orders_data = []
        for order in orders:
            orders_data.append({
                '订单号': order.order_id[:12] + '...',
                '品种': order.symbol,
                '方向': '买' if order.direction.value == 'long' else '卖',
                '开平': order.offset.value,
                '委托价': f"{order.price:.2f}",
                '委托量': order.volume,
                '已成': order.traded,
                '状态': order.status.value,
                '时间': order.create_time.strftime('%H:%M:%S') if order.create_time else '-'
            })

        df = pd.DataFrame(orders_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if st.button("撤销全部"):
            cancel_all_orders()
    else:
        st.info("无活动订单")

    st.markdown("---")

    # 今日成交
    st.write("**今日成交**")
    trades = engine.get_trades()

    if trades:
        trades_data = []
        for trade in trades:
            trades_data.append({
                '成交号': trade.trade_id[:12] + '...' if trade.trade_id else '-',
                '品种': trade.symbol,
                '方向': '买' if trade.direction.value == 'long' else '卖',
                '开平': trade.offset.value,
                '成交价': f"{trade.price:.2f}",
                '成交量': trade.volume,
                '手续费': f"¥{trade.commission:.2f}" if trade.commission else '-',
                '时间': trade.trade_time.strftime('%H:%M:%S') if trade.trade_time else '-'
            })

        df = pd.DataFrame(trades_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # 成交统计
        total_commission = sum(t.commission for t in trades if t.commission)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("成交笔数", f"{len(trades)}")
        with col2:
            st.metric("手续费合计", f"¥{total_commission:.2f}")
    else:
        st.info("今日无成交")


def start_sim_trading(strategy_name: str, symbol: str, timeframe: str, capital: float, params: dict):
    """启动模拟交易"""
    try:
        from core.live_engine import LiveEngine
        from strategies.base import create_strategy

        config = load_tq_config()

        if not config.get('tq_user') or not config.get('tq_password'):
            st.error("请先在「系统设置」中配置天勤账号")
            return

        # 创建引擎
        engine = LiveEngine()

        # 设置品种配置
        inst = get_instrument(symbol)
        if inst:
            engine.set_instrument_config(symbol, inst)

        # 初始化TqSdk网关（模拟盘）
        gateway_config = {
            'tq_user': config['tq_user'],
            'tq_password': config['tq_password'],
            'sim_mode': True
        }
        engine.init_gateway("tq_sim", gateway_config)

        # 创建策略
        strategy = create_strategy(strategy_name, params)
        if strategy:
            engine.add_strategy(strategy, [symbol])

        # 设置信号回调
        def on_signal(signal, sym, sid):
            if 'sim_signals' not in st.session_state:
                st.session_state.sim_signals = []
            st.session_state.sim_signals.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'symbol': sym,
                'action': signal.action,
                'price': signal.price
            })

        engine.set_signal_callback(on_signal)

        # 启动
        engine.start(capital)

        # 保存到session
        st.session_state.sim_engine = engine
        st.session_state.sim_signals = []

        # 更新配置
        config['initial_capital'] = capital
        save_tq_config(config)

        st.success("模拟交易已启动!")
        st.rerun()

    except Exception as e:
        st.error(f"启动失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def stop_sim_trading():
    """停止模拟交易"""
    engine = get_sim_engine()
    if engine:
        try:
            engine.stop()
            st.session_state.sim_engine = None
            st.success("模拟交易已停止")
            st.rerun()
        except Exception as e:
            st.error(f"停止失败: {e}")


def close_positions(direction: str):
    """平仓指定方向"""
    engine = get_sim_engine()
    if not engine:
        return

    try:
        from models.base import Direction, Offset

        positions = engine.get_positions()
        target_dir = Direction.LONG if direction == 'long' else Direction.SHORT

        for pos in positions:
            if pos.direction == target_dir and pos.volume > 0:
                close_dir = Direction.SHORT if direction == 'long' else Direction.LONG
                engine.order_manager.send_order(
                    symbol=pos.symbol,
                    direction=close_dir,
                    offset=Offset.CLOSE,
                    volume=pos.volume,
                    price=0.0
                )

        st.success(f"{'多头' if direction == 'long' else '空头'}平仓订单已提交")

    except Exception as e:
        st.error(f"操作失败: {e}")


def close_all_positions():
    """全部平仓"""
    close_positions('long')
    close_positions('short')


def cancel_all_orders():
    """撤销全部订单"""
    engine = get_sim_engine()
    if not engine:
        return

    try:
        orders = engine.get_orders()
        for order in orders:
            engine.order_manager.cancel_order(order.order_id)
        st.success(f"已撤销 {len(orders)} 个订单")
    except Exception as e:
        st.error(f"撤单失败: {e}")
