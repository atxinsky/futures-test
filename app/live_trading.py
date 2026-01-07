# coding=utf-8
"""
实盘交易Web界面模块
提供TqSdk实盘交易的Web界面
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
        'broker_id': '',
        'td_account': '',
        'td_password': '',
        'default_symbols': ['RB', 'AU', 'IF'],
        'risk_config': {
            'max_position_per_symbol': 10,
            'max_daily_loss': 0.05,
            'max_drawdown': 0.15
        }
    }


def save_tq_config(config: dict):
    """保存TqSdk配置"""
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def get_live_engine():
    """获取或创建实盘引擎"""
    if 'live_engine' not in st.session_state:
        st.session_state.live_engine = None
    return st.session_state.live_engine


def render_live_trading_page():
    """渲染实盘交易页面"""
    st.title("实盘交易")

    # 检查TqSdk是否安装
    try:
        import tqsdk
        tqsdk_installed = True
        tqsdk_version = tqsdk.__version__
    except ImportError:
        tqsdk_installed = False
        tqsdk_version = None

    if not tqsdk_installed:
        st.error("TqSdk未安装，请执行: pip install tqsdk")
        st.code("pip install tqsdk")
        return

    # 实盘警告
    st.warning("实盘交易涉及真实资金，请谨慎操作！")

    # 选项卡 - 与模拟交易保持一致
    tab1, tab2, tab3, tab4 = st.tabs([
        "交易配置", "持仓监控", "订单记录", "连接设置"
    ])

    with tab1:
        render_trading_panel()

    with tab2:
        render_position_management()

    with tab3:
        render_order_records()

    with tab4:
        render_connection_settings(tqsdk_version)


def render_trading_panel():
    """渲染交易面板"""
    st.subheader("交易面板")

    engine = get_live_engine()
    is_running = engine is not None and engine.is_running if engine else False

    # 连接状态
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if is_running:
            st.success("已连接")
        else:
            st.warning("未连接")

    with col2:
        if not is_running:
            if st.button("启动交易", type="primary"):
                start_live_trading()
        else:
            if st.button("停止交易", type="secondary"):
                stop_live_trading()

    with col3:
        config = load_tq_config()
        mode = "模拟盘" if config.get('sim_mode', True) else "实盘"
        st.info(f"当前模式: {mode}")

    st.markdown("---")

    if not is_running:
        st.info("请先启动交易引擎")

        # 显示快速启动配置
        st.subheader("快速启动")

        config = load_tq_config()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**基础配置**")
            tq_user = st.text_input("天勤账号", value=config.get('tq_user', ''), key="quick_tq_user")
            tq_password = st.text_input("天勤密码", type="password", key="quick_tq_password")
            sim_mode = st.checkbox("使用模拟盘", value=config.get('sim_mode', True))

        with col2:
            st.write("**交易设置**")
            symbols = st.multiselect(
                "交易品种",
                options=list(INSTRUMENTS.keys()),
                default=config.get('default_symbols', ['RB', 'AU', 'IF']),
                format_func=lambda x: f"{x} - {INSTRUMENTS.get(x, {}).get('name', x)}"
            )
            initial_capital = st.number_input("初始资金", value=100000, min_value=10000, step=10000)

        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("启动", type="primary", use_container_width=True):
                # 保存配置
                config['tq_user'] = tq_user
                config['tq_password'] = tq_password
                config['sim_mode'] = sim_mode
                config['default_symbols'] = symbols
                save_tq_config(config)

                # 启动
                start_live_trading(symbols, initial_capital)

        return

    # 已连接时显示交易面板
    render_quick_trade_panel()


def render_quick_trade_panel():
    """渲染快速交易面板"""
    st.subheader("快速下单")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        symbol = st.selectbox(
            "品种",
            options=list(INSTRUMENTS.keys()),
            format_func=lambda x: f"{x} - {INSTRUMENTS.get(x, {}).get('name', x)}",
            key="trade_symbol"
        )

    with col2:
        direction = st.selectbox("方向", options=["买入", "卖出"], key="trade_direction")

    with col3:
        offset = st.selectbox("开平", options=["开仓", "平仓", "平今"], key="trade_offset")

    with col4:
        volume = st.number_input("数量", min_value=1, value=1, step=1, key="trade_volume")

    with col5:
        price_type = st.selectbox("价格类型", options=["市价", "限价"], key="trade_price_type")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if price_type == "限价":
            price = st.number_input("委托价格", min_value=0.0, value=0.0, step=1.0, key="trade_price")
        else:
            st.write("市价委托")
            price = 0.0

    with col2:
        if st.button("下单", type="primary", use_container_width=True):
            submit_order(symbol, direction, offset, volume, price, price_type)

    with col3:
        if st.button("撤销全部", use_container_width=True):
            cancel_all_orders()

    st.markdown("---")

    # 显示账户信息
    render_account_summary()


def render_account_summary():
    """渲染账户摘要"""
    st.subheader("账户概览")

    engine = get_live_engine()
    if not engine:
        return

    account = engine.get_account()

    if account:
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        with col1:
            st.metric("账户权益", f"¥{account.balance:,.2f}")
        with col2:
            st.metric("可用资金", f"¥{account.available:,.2f}")
        with col3:
            pnl = account.balance - account.balance  # 简化
            st.metric("今日盈亏", f"¥{pnl:+,.2f}")
        with col4:
            margin_ratio = (account.margin / account.balance * 100) if account.balance > 0 else 0
            st.metric("保证金占用", f"{margin_ratio:.1f}%")
        with col5:
            position_count = len(engine.get_positions())
            st.metric("持仓品种", f"{position_count}")
        with col6:
            order_count = len(engine.get_orders())
            st.metric("活动订单", f"{order_count}")
    else:
        st.warning("无法获取账户信息")


def render_realtime_quotes():
    """渲染实时行情"""
    st.subheader("实时行情")

    engine = get_live_engine()
    is_running = engine is not None and engine.is_running if engine else False

    if not is_running:
        st.info("请先启动交易引擎查看实时行情")
        return

    # 行情表格
    quotes_data = []

    config = load_tq_config()
    symbols = config.get('default_symbols', ['RB', 'AU', 'IF'])

    for symbol in symbols:
        inst = get_instrument(symbol)
        if inst:
            # 从引擎获取行情（如果有）
            bar = engine.last_bars.get(symbol)
            tick = engine.last_ticks.get(symbol)

            if tick:
                quotes_data.append({
                    '品种': f"{symbol} - {inst['name']}",
                    '最新价': f"{tick.last_price:.2f}",
                    '涨跌': '-',
                    '涨跌%': '-',
                    '买一': f"{tick.bid_price:.2f}",
                    '卖一': f"{tick.ask_price:.2f}",
                    '成交量': f"{tick.volume:,}",
                    '更新时间': tick.datetime.strftime('%H:%M:%S') if tick.datetime else '-'
                })
            elif bar:
                quotes_data.append({
                    '品种': f"{symbol} - {inst['name']}",
                    '最新价': f"{bar.close:.2f}",
                    '涨跌': '-',
                    '涨跌%': '-',
                    '买一': '-',
                    '卖一': '-',
                    '成交量': f"{bar.volume:,}",
                    '更新时间': bar.datetime.strftime('%H:%M:%S') if bar.datetime else '-'
                })
            else:
                quotes_data.append({
                    '品种': f"{symbol} - {inst['name']}",
                    '最新价': '-',
                    '涨跌': '-',
                    '涨跌%': '-',
                    '买一': '-',
                    '卖一': '-',
                    '成交量': '-',
                    '更新时间': '-'
                })

    if quotes_data:
        df_quotes = pd.DataFrame(quotes_data)
        st.dataframe(df_quotes, use_container_width=True, hide_index=True)
    else:
        st.warning("暂无行情数据")

    # 自动刷新
    if st.button("刷新行情"):
        st.rerun()


def render_position_management():
    """渲染持仓管理"""
    st.subheader("持仓管理")

    engine = get_live_engine()
    is_running = engine is not None and engine.is_running if engine else False

    if not is_running:
        st.info("请先启动交易引擎查看持仓")
        return

    positions = engine.get_positions()

    if not positions:
        st.info("当前无持仓")
        return

    # 持仓表格
    positions_data = []

    for pos in positions:
        inst = get_instrument(pos.symbol)
        inst_name = inst['name'] if inst else pos.symbol

        positions_data.append({
            '品种': f"{pos.symbol} - {inst_name}",
            '方向': '多' if pos.direction.value == 'long' else '空',
            '数量': pos.volume,
            '可用': pos.available,
            '开仓价': f"{pos.avg_price:.2f}",
            '现价': f"{pos.last_price:.2f}" if pos.last_price else '-',
            '浮盈': f"¥{pos.unrealized_pnl:+,.2f}",
            '浮盈%': f"{pos.unrealized_pnl_pct:+.2f}%" if hasattr(pos, 'unrealized_pnl_pct') else '-',
            '保证金': f"¥{pos.margin:,.2f}"
        })

    df_positions = pd.DataFrame(positions_data)
    st.dataframe(df_positions, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 批量操作
    st.write("**批量操作**")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("平多头", use_container_width=True):
            close_all_long()

    with col2:
        if st.button("平空头", use_container_width=True):
            close_all_short()

    with col3:
        if st.button("全部平仓", type="primary", use_container_width=True):
            close_all_positions()

    with col4:
        if st.button("刷新", use_container_width=True):
            st.rerun()


def render_order_records():
    """渲染订单记录"""
    st.subheader("订单记录")

    engine = get_live_engine()
    is_running = engine is not None and engine.is_running if engine else False

    if not is_running:
        st.info("请先启动交易引擎查看订单")
        return

    # 活动订单
    st.write("**活动订单**")

    orders = engine.get_orders()

    if orders:
        orders_data = []
        for order in orders:
            orders_data.append({
                '订单号': order.order_id[:8] + '...',
                '品种': order.symbol,
                '方向': '买' if order.direction.value == 'long' else '卖',
                '开平': order.offset.value,
                '委托价': f"{order.price:.2f}",
                '委托量': order.volume,
                '已成': order.traded,
                '状态': order.status.value,
                '时间': order.create_time.strftime('%H:%M:%S') if order.create_time else '-'
            })

        df_orders = pd.DataFrame(orders_data)
        st.dataframe(df_orders, use_container_width=True, hide_index=True)

        if st.button("撤销全部订单"):
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
                '成交号': trade.trade_id[:8] + '...' if trade.trade_id else '-',
                '品种': trade.symbol,
                '方向': '买' if trade.direction.value == 'long' else '卖',
                '开平': trade.offset.value,
                '成交价': f"{trade.price:.2f}",
                '成交量': trade.volume,
                '手续费': f"¥{trade.commission:.2f}" if trade.commission else '-',
                '时间': trade.trade_time.strftime('%H:%M:%S') if trade.trade_time else '-'
            })

        df_trades = pd.DataFrame(trades_data)
        st.dataframe(df_trades, use_container_width=True, hide_index=True)
    else:
        st.info("今日无成交")


def render_connection_settings(tqsdk_version: str = None):
    """渲染连接设置"""
    st.subheader("连接设置")

    if tqsdk_version:
        st.success(f"TqSdk 已安装 (版本: {tqsdk_version})")

    config = load_tq_config()

    # 天勤账号设置
    st.write("**天勤账号**")

    col1, col2 = st.columns(2)

    with col1:
        tq_user = st.text_input("天勤用户名", value=config.get('tq_user', ''))

    with col2:
        tq_password = st.text_input("天勤密码", type="password", value=config.get('tq_password', ''))

    st.markdown("---")

    # 交易模式
    st.write("**交易模式**")

    sim_mode = st.radio(
        "选择模式",
        options=["模拟盘 (TqSim)", "实盘 (需要期货账号)"],
        index=0 if config.get('sim_mode', True) else 1,
        horizontal=True
    )

    sim_mode_bool = sim_mode == "模拟盘 (TqSim)"

    # 实盘配置
    if not sim_mode_bool:
        st.write("**期货账号配置**")

        col1, col2 = st.columns(2)

        with col1:
            broker_id = st.text_input("期货公司代码", value=config.get('broker_id', ''))
            td_account = st.text_input("交易账号", value=config.get('td_account', ''))

        with col2:
            td_password = st.text_input("交易密码", type="password", value=config.get('td_password', ''))

        st.info("实盘交易需要开通期货账户，并获取期货公司的交易前置地址")

    st.markdown("---")

    # 默认交易品种
    st.write("**默认交易品种**")

    default_symbols = st.multiselect(
        "选择品种",
        options=list(INSTRUMENTS.keys()),
        default=config.get('default_symbols', ['RB', 'AU', 'IF']),
        format_func=lambda x: f"{x} - {INSTRUMENTS.get(x, {}).get('name', x)}"
    )

    st.markdown("---")

    # 风控设置
    st.write("**风控设置**")

    risk_config = config.get('risk_config', {})

    col1, col2, col3 = st.columns(3)

    with col1:
        max_pos = st.number_input(
            "单品种最大持仓",
            min_value=1, max_value=100,
            value=risk_config.get('max_position_per_symbol', 10)
        )

    with col2:
        max_daily_loss = st.slider(
            "日最大亏损%",
            min_value=1, max_value=20,
            value=int(risk_config.get('max_daily_loss', 0.05) * 100)
        )

    with col3:
        max_drawdown = st.slider(
            "最大回撤%",
            min_value=5, max_value=50,
            value=int(risk_config.get('max_drawdown', 0.15) * 100)
        )

    st.markdown("---")

    # 保存按钮
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("保存配置", type="primary", use_container_width=True):
            new_config = {
                'tq_user': tq_user,
                'tq_password': tq_password,
                'sim_mode': sim_mode_bool,
                'broker_id': config.get('broker_id', '') if sim_mode_bool else broker_id,
                'td_account': config.get('td_account', '') if sim_mode_bool else td_account,
                'td_password': config.get('td_password', '') if sim_mode_bool else td_password,
                'default_symbols': default_symbols,
                'risk_config': {
                    'max_position_per_symbol': max_pos,
                    'max_daily_loss': max_daily_loss / 100,
                    'max_drawdown': max_drawdown / 100
                }
            }
            save_tq_config(new_config)
            st.success("配置已保存!")

    # 测试连接
    st.markdown("---")

    st.write("**连接测试**")

    if st.button("测试天勤连接"):
        test_tq_connection(tq_user, tq_password)


def start_live_trading(symbols: List[str] = None, initial_capital: float = 100000):
    """启动实盘交易"""
    try:
        from core.live_engine import LiveEngine
        from strategies.base import get_registered_strategies, create_strategy

        config = load_tq_config()

        if not config.get('tq_user') or not config.get('tq_password'):
            st.error("请先配置天勤账号")
            return

        # 创建引擎
        engine = LiveEngine()

        # 设置品种配置
        symbols = symbols or config.get('default_symbols', ['RB', 'AU', 'IF'])

        for symbol in symbols:
            inst = get_instrument(symbol)
            if inst:
                engine.set_instrument_config(symbol, inst)

        # 初始化网关
        gateway_type = "tq_sim" if config.get('sim_mode', True) else "tq_live"
        gateway_config = {
            'tq_user': config['tq_user'],
            'tq_password': config['tq_password'],
            'sim_mode': config.get('sim_mode', True),
            'broker_id': config.get('broker_id', ''),
            'td_account': config.get('td_account', ''),
            'td_password': config.get('td_password', '')
        }

        engine.init_gateway(gateway_type, gateway_config)

        # 启动引擎
        engine.start(initial_capital)

        # 保存到session
        st.session_state.live_engine = engine

        st.success("交易引擎启动成功!")
        st.rerun()

    except Exception as e:
        st.error(f"启动失败: {e}")
        import traceback
        st.code(traceback.format_exc())


def stop_live_trading():
    """停止实盘交易"""
    engine = get_live_engine()
    if engine:
        try:
            engine.stop()
            st.session_state.live_engine = None
            st.success("交易引擎已停止")
            st.rerun()
        except Exception as e:
            st.error(f"停止失败: {e}")


def submit_order(symbol: str, direction: str, offset: str, volume: int, price: float, price_type: str):
    """提交订单"""
    engine = get_live_engine()
    if not engine or not engine.is_running:
        st.error("交易引擎未启动")
        return

    try:
        from models.base import Direction, Offset

        # 转换方向
        dir_enum = Direction.LONG if direction == "买入" else Direction.SHORT

        # 转换开平
        offset_map = {"开仓": Offset.OPEN, "平仓": Offset.CLOSE, "平今": Offset.CLOSE_TODAY}
        offset_enum = offset_map.get(offset, Offset.OPEN)

        # 下单
        if engine.order_manager:
            order_id = engine.order_manager.send_order(
                symbol=symbol,
                direction=dir_enum,
                offset=offset_enum,
                volume=volume,
                price=price if price_type == "限价" else 0.0
            )

            if order_id:
                st.success(f"订单已提交: {order_id[:16]}...")
            else:
                st.error("订单提交失败")
        else:
            st.error("订单管理器未初始化")

    except Exception as e:
        st.error(f"下单失败: {e}")


def cancel_all_orders():
    """撤销全部订单"""
    engine = get_live_engine()
    if not engine or not engine.is_running:
        st.error("交易引擎未启动")
        return

    try:
        orders = engine.get_orders()
        canceled = 0

        for order in orders:
            if engine.order_manager:
                engine.order_manager.cancel_order(order.order_id)
                canceled += 1

        st.success(f"已撤销 {canceled} 个订单")

    except Exception as e:
        st.error(f"撤单失败: {e}")


def close_all_positions():
    """平掉所有持仓"""
    engine = get_live_engine()
    if not engine or not engine.is_running:
        st.error("交易引擎未启动")
        return

    try:
        positions = engine.get_positions()
        closed = 0

        for pos in positions:
            if pos.volume > 0 and engine.order_manager:
                from models.base import Direction, Offset

                # 反向平仓
                close_direction = Direction.SHORT if pos.direction == Direction.LONG else Direction.LONG

                engine.order_manager.send_order(
                    symbol=pos.symbol,
                    direction=close_direction,
                    offset=Offset.CLOSE,
                    volume=pos.volume,
                    price=0.0  # 市价
                )
                closed += 1

        st.success(f"已提交 {closed} 个平仓订单")

    except Exception as e:
        st.error(f"平仓失败: {e}")


def close_all_long():
    """平掉所有多头"""
    engine = get_live_engine()
    if not engine or not engine.is_running:
        return

    try:
        from models.base import Direction, Offset

        positions = engine.get_positions()
        for pos in positions:
            if pos.direction == Direction.LONG and pos.volume > 0:
                engine.order_manager.send_order(
                    symbol=pos.symbol,
                    direction=Direction.SHORT,
                    offset=Offset.CLOSE,
                    volume=pos.volume,
                    price=0.0
                )

        st.success("多头平仓订单已提交")

    except Exception as e:
        st.error(f"操作失败: {e}")


def close_all_short():
    """平掉所有空头"""
    engine = get_live_engine()
    if not engine or not engine.is_running:
        return

    try:
        from models.base import Direction, Offset

        positions = engine.get_positions()
        for pos in positions:
            if pos.direction == Direction.SHORT and pos.volume > 0:
                engine.order_manager.send_order(
                    symbol=pos.symbol,
                    direction=Direction.LONG,
                    offset=Offset.CLOSE,
                    volume=pos.volume,
                    price=0.0
                )

        st.success("空头平仓订单已提交")

    except Exception as e:
        st.error(f"操作失败: {e}")


def test_tq_connection(tq_user: str, tq_password: str):
    """测试天勤连接"""
    if not tq_user or not tq_password:
        st.error("请输入天勤账号和密码")
        return

    try:
        from tqsdk import TqApi, TqAuth

        with st.spinner("正在连接天勤..."):
            auth = TqAuth(tq_user, tq_password)
            api = TqApi(auth=auth)

            # 获取行情测试
            quote = api.get_quote("SHFE.rb2505")
            api.wait_update()

            api.close()

        st.success(f"连接成功! 测试行情: RB2505 最新价 {quote.last_price}")

    except Exception as e:
        st.error(f"连接失败: {e}")
