# coding=utf-8
"""
专业交易系统 Web界面
主入口文件
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 页面配置
st.set_page_config(
    page_title="期货量化交易系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    /* 主题色 */
    :root {
        --primary-color: #1f77b4;
        --success-color: #2ecc71;
        --danger-color: #e74c3c;
        --warning-color: #f39c12;
        --bg-dark: #1e1e1e;
        --bg-card: #2d2d2d;
        --text-primary: #ffffff;
        --text-secondary: #b0b0b0;
    }

    /* 隐藏默认菜单 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* 卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #1e1e1e 100%);
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #3d3d3d;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #ffffff;
    }

    .metric-label {
        font-size: 14px;
        color: #b0b0b0;
        margin-bottom: 5px;
    }

    .metric-change-positive {
        color: #2ecc71;
        font-size: 14px;
    }

    .metric-change-negative {
        color: #e74c3c;
        font-size: 14px;
    }

    /* 状态指示器 */
    .status-running {
        color: #2ecc71;
        font-weight: bold;
    }

    .status-stopped {
        color: #e74c3c;
        font-weight: bold;
    }

    /* 表格样式优化 */
    .dataframe {
        font-size: 13px !important;
    }

    /* 侧边栏 */
    .css-1d391kg {
        background-color: #1e1e1e;
    }

    /* 标题 */
    h1 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    h2, h3 {
        color: #e0e0e0 !important;
    }

    /* 按钮 */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 8px 16px;
    }

    .stButton > button:hover {
        background-color: #1668a3;
    }

    /* 盈亏颜色 */
    .profit {
        color: #2ecc71 !important;
    }

    .loss {
        color: #e74c3c !important;
    }
</style>
""", unsafe_allow_html=True)


def render_metric_card(label: str, value: str, change: str = None, change_type: str = "neutral"):
    """渲染指标卡片"""
    change_class = "metric-change-positive" if change_type == "positive" else "metric-change-negative"
    change_html = f'<div class="{change_class}">{change}</div>' if change else ""

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)


def main():
    """主函数"""
    # 侧边栏
    with st.sidebar:
        st.title("📈 期货量化系统")
        st.markdown("---")

        # 导航
        page = st.radio(
            "功能模块",
            ["仪表盘", "策略管理", "持仓监控", "订单管理", "风控中心", "回测系统", "系统设置"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        # 系统状态
        st.markdown("### 系统状态")

        # 这里应该从实际引擎获取状态
        engine_running = st.session_state.get('engine_running', False)

        if engine_running:
            st.markdown('<span class="status-running">● 运行中</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-stopped">● 已停止</span>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("启动", use_container_width=True, disabled=engine_running):
                st.session_state['engine_running'] = True
                st.rerun()
        with col2:
            if st.button("停止", use_container_width=True, disabled=not engine_running):
                st.session_state['engine_running'] = False
                st.rerun()

        st.markdown("---")
        st.caption(f"更新时间: {datetime.now().strftime('%H:%M:%S')}")

    # 主内容区
    if page == "仪表盘":
        render_dashboard()
    elif page == "策略管理":
        render_strategy_management()
    elif page == "持仓监控":
        render_position_monitor()
    elif page == "订单管理":
        render_order_management()
    elif page == "风控中心":
        render_risk_center()
    elif page == "回测系统":
        render_backtest()
    elif page == "系统设置":
        render_settings()


def render_dashboard():
    """渲染仪表盘"""
    st.title("交易仪表盘")

    # 顶部指标
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("账户权益", "¥125,680", "+5,680 (4.73%)")
    with col2:
        st.metric("可用资金", "¥98,450", "78.3%")
    with col3:
        st.metric("今日盈亏", "¥2,350", "+1.90%")
    with col4:
        st.metric("持仓数量", "3", "")
    with col5:
        st.metric("活动订单", "2", "")

    st.markdown("---")

    # 权益曲线和持仓分布
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("权益曲线")

        # 模拟数据
        dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
        equity = 100000 + np.cumsum(np.random.randn(30) * 1000)

        chart_data = pd.DataFrame({
            '日期': dates,
            '权益': equity
        }).set_index('日期')

        st.line_chart(chart_data, height=300)

    with col2:
        st.subheader("持仓分布")

        # 持仓分布饼图数据
        position_data = pd.DataFrame({
            '品种': ['螺纹钢', '铁矿石', '黄金'],
            '占比': [40, 35, 25]
        })

        st.bar_chart(position_data.set_index('品种'), height=300)

    st.markdown("---")

    # 持仓列表和最新交易
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("当前持仓")

        positions_df = pd.DataFrame({
            '合约': ['RB2505', 'I2505', 'AU2506'],
            '方向': ['多', '多', '空'],
            '数量': [5, 3, 2],
            '开仓价': [3580, 820, 580],
            '现价': [3620, 815, 575],
            '浮盈': ['+2,000', '-150', '+100'],
            '盈亏%': ['+1.12%', '-0.61%', '+0.86%']
        })

        st.dataframe(positions_df, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("最新成交")

        trades_df = pd.DataFrame({
            '时间': ['14:35:20', '14:20:15', '11:30:00', '10:45:30'],
            '合约': ['RB2505', 'I2505', 'AU2506', 'RB2505'],
            '方向': ['买', '买', '卖', '买'],
            '价格': [3580, 820, 580, 3570],
            '数量': [2, 3, 2, 3]
        })

        st.dataframe(trades_df, hide_index=True, use_container_width=True)


def render_strategy_management():
    """渲染策略管理页面"""
    st.title("策略管理")

    # 策略列表
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("已加载策略")

    with col2:
        if st.button("+ 添加策略", use_container_width=True):
            st.session_state['show_add_strategy'] = True

    # 策略表格
    strategies_df = pd.DataFrame({
        '策略名称': ['WaveTrend趋势', 'MACD动量', 'EMA突破'],
        '状态': ['运行中', '运行中', '已停止'],
        '交易品种': ['RB, I, AU', 'CU, AL', 'RB'],
        '今日盈亏': ['+2,350', '+850', '0'],
        '累计盈亏': ['+25,680', '+8,450', '+3,200'],
        '胜率': ['58.3%', '52.1%', '61.5%'],
        '最大回撤': ['8.5%', '12.3%', '6.8%']
    })

    # 显示策略列表
    for idx, row in strategies_df.iterrows():
        with st.expander(f"📊 {row['策略名称']} - {row['状态']}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("今日盈亏", row['今日盈亏'])
            with col2:
                st.metric("累计盈亏", row['累计盈亏'])
            with col3:
                st.metric("胜率", row['胜率'])
            with col4:
                st.metric("最大回撤", row['最大回撤'])

            st.write(f"**交易品种**: {row['交易品种']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.button("编辑参数", key=f"edit_{idx}")
            with col2:
                if row['状态'] == '运行中':
                    st.button("暂停", key=f"pause_{idx}")
                else:
                    st.button("启动", key=f"start_{idx}")
            with col3:
                st.button("移除", key=f"remove_{idx}")

    # 添加策略弹窗
    if st.session_state.get('show_add_strategy', False):
        st.markdown("---")
        st.subheader("添加新策略")

        col1, col2 = st.columns(2)

        with col1:
            strategy_type = st.selectbox(
                "选择策略",
                ["WaveTrend趋势策略", "MACD动量策略", "EMA突破策略", "自定义策略"]
            )

            symbols = st.multiselect(
                "交易品种",
                ["RB", "I", "AU", "CU", "AL", "NI", "TA", "MA", "PP"]
            )

        with col2:
            st.write("**策略参数**")
            param1 = st.number_input("参数1", value=10)
            param2 = st.number_input("参数2", value=20)
            param3 = st.number_input("参数3", value=50)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("确认添加", use_container_width=True):
                st.success("策略添加成功!")
                st.session_state['show_add_strategy'] = False
                st.rerun()
        with col2:
            if st.button("取消", use_container_width=True):
                st.session_state['show_add_strategy'] = False
                st.rerun()


def render_position_monitor():
    """渲染持仓监控页面"""
    st.title("持仓监控")

    # 汇总信息
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总持仓市值", "¥358,000")
    with col2:
        st.metric("浮动盈亏", "¥1,950", "+0.55%")
    with col3:
        st.metric("已用保证金", "¥27,230")
    with col4:
        st.metric("保证金占用", "21.7%")

    st.markdown("---")

    # 持仓明细
    st.subheader("持仓明细")

    positions_df = pd.DataFrame({
        '合约': ['RB2505', 'I2505', 'AU2506', 'CU2505'],
        '方向': ['多', '多', '空', '多'],
        '数量': [5, 3, 2, 1],
        '开仓价': [3580.0, 820.0, 580.0, 75200.0],
        '现价': [3620.0, 815.0, 575.0, 75500.0],
        '浮盈(元)': [2000.0, -150.0, 100.0, 300.0],
        '浮盈%': [1.12, -0.61, 0.86, 0.40],
        '保证金': [17900.0, 2460.0, 1160.0, 7520.0],
        '持仓时间': ['2小时', '1天', '3小时', '2天'],
        '策略': ['WaveTrend', 'WaveTrend', 'MACD', 'EMA']
    })

    # 格式化显示
    def color_pnl(val):
        if isinstance(val, (int, float)):
            color = '#2ecc71' if val > 0 else '#e74c3c' if val < 0 else '#ffffff'
            return f'color: {color}'
        return ''

    styled_df = positions_df.style.applymap(
        color_pnl, subset=['浮盈(元)', '浮盈%']
    )

    st.dataframe(styled_df, hide_index=True, use_container_width=True, height=300)

    st.markdown("---")

    # 持仓操作
    st.subheader("快捷操作")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("一键平多", use_container_width=True):
            st.warning("确认平掉所有多头持仓?")
    with col2:
        if st.button("一键平空", use_container_width=True):
            st.warning("确认平掉所有空头持仓?")
    with col3:
        if st.button("全部平仓", use_container_width=True, type="primary"):
            st.error("确认平掉所有持仓?")
    with col4:
        if st.button("刷新数据", use_container_width=True):
            st.rerun()


def render_order_management():
    """渲染订单管理页面"""
    st.title("订单管理")

    # 选项卡
    tab1, tab2, tab3 = st.tabs(["活动订单", "今日成交", "历史订单"])

    with tab1:
        st.subheader("活动订单")

        active_orders = pd.DataFrame({
            '订单号': ['ORD001', 'ORD002'],
            '时间': ['14:35:20', '14:20:15'],
            '合约': ['RB2505', 'I2505'],
            '方向': ['买', '卖'],
            '开平': ['开', '平'],
            '报价': [3575, 825],
            '数量': [2, 1],
            '已成': [0, 0],
            '状态': ['等待成交', '等待成交'],
            '策略': ['WaveTrend', 'MACD']
        })

        st.dataframe(active_orders, hide_index=True, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("撤销选中", use_container_width=True):
                st.info("请先选择要撤销的订单")
        with col2:
            if st.button("全部撤单", use_container_width=True):
                st.warning("确认撤销所有挂单?")

    with tab2:
        st.subheader("今日成交")

        trades_df = pd.DataFrame({
            '成交号': ['TRD001', 'TRD002', 'TRD003', 'TRD004'],
            '时间': ['14:35:20', '14:20:15', '11:30:00', '10:45:30'],
            '合约': ['RB2505', 'I2505', 'AU2506', 'RB2505'],
            '方向': ['买', '买', '卖', '买'],
            '开平': ['开', '开', '开', '开'],
            '价格': [3580, 820, 580, 3570],
            '数量': [2, 3, 2, 3],
            '手续费': [8.5, 6.0, 12.0, 8.5],
            '策略': ['WaveTrend', 'WaveTrend', 'MACD', 'WaveTrend']
        })

        st.dataframe(trades_df, hide_index=True, use_container_width=True)

        # 成交统计
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("成交笔数", "4")
        with col2:
            st.metric("成交手数", "10")
        with col3:
            st.metric("手续费合计", "¥35.0")
        with col4:
            st.metric("已实现盈亏", "¥2,350")

    with tab3:
        st.subheader("历史订单查询")

        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input("开始日期", datetime.now() - timedelta(days=7))
        with col2:
            end_date = st.date_input("结束日期", datetime.now())
        with col3:
            symbol_filter = st.selectbox("品种", ["全部", "RB", "I", "AU", "CU"])

        if st.button("查询"):
            st.info("查询历史订单...")


def render_risk_center():
    """渲染风控中心页面"""
    st.title("风控中心")

    # 风险状态
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        risk_level = "low"  # 从引擎获取

        if risk_level == "low":
            st.success("### 风险状态: 安全")
        elif risk_level == "medium":
            st.warning("### 风险状态: 警告")
        elif risk_level == "high":
            st.warning("### 风险状态: 高风险")
        else:
            st.error("### 风险状态: 危险")

    st.markdown("---")

    # 风险指标
    st.subheader("风险指标")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("保证金占用", "21.7%", "限制: 80%")
    with col2:
        st.metric("日亏损", "-0.5%", "限制: 5%")
    with col3:
        st.metric("最大回撤", "3.2%", "限制: 15%")
    with col4:
        st.metric("连续亏损", "1次", "限制: 5次")

    st.markdown("---")

    # 风控设置
    st.subheader("风控设置")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**持仓限制**")
        max_pos_per_symbol = st.number_input("单品种最大持仓", value=10, min_value=1)
        max_pos_total = st.number_input("总最大持仓", value=50, min_value=1)

        st.write("**资金风控**")
        max_margin_ratio = st.slider("最大保证金占用比例", 0.0, 1.0, 0.8)
        min_available = st.number_input("最小可用资金", value=10000, min_value=0)

    with col2:
        st.write("**亏损控制**")
        max_daily_loss = st.slider("日最大亏损比例", 0.0, 0.2, 0.05)
        max_drawdown = st.slider("最大回撤比例", 0.0, 0.3, 0.15)
        max_consecutive = st.number_input("最大连续亏损次数", value=5, min_value=1)

        st.write("**其他设置**")
        force_close = st.checkbox("达到限制时强制平仓", value=True)
        allow_open = st.checkbox("高风险时允许开仓", value=False)

    if st.button("保存设置", use_container_width=True):
        st.success("风控设置已保存!")

    st.markdown("---")

    # 风控日志
    st.subheader("风控日志")

    logs_df = pd.DataFrame({
        '时间': ['14:35:20', '14:20:15', '11:30:00'],
        '级别': ['INFO', 'WARNING', 'INFO'],
        '消息': [
            '订单风控检查通过: RB2505 买开2手',
            '日亏损接近限制: -4.2% (限制: -5%)',
            '新策略加入: WaveTrend'
        ]
    })

    st.dataframe(logs_df, hide_index=True, use_container_width=True)


def render_backtest():
    """渲染回测系统页面"""
    st.title("回测系统")

    # 回测配置
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("回测配置")

        strategy = st.selectbox(
            "选择策略",
            ["WaveTrend趋势策略", "MACD动量策略", "EMA突破策略"]
        )

        symbols = st.multiselect(
            "交易品种",
            ["RB", "I", "AU", "CU", "AL", "NI", "TA", "MA", "PP"],
            default=["RB", "I"]
        )

        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input("开始日期", datetime(2024, 1, 1))
        with col_b:
            end_date = st.date_input("结束日期", datetime(2025, 1, 1))

        timeframe = st.selectbox(
            "K线周期",
            ["日线", "60分钟", "30分钟", "15分钟", "5分钟", "1分钟"]
        )

        initial_capital = st.number_input("初始资金", value=100000, min_value=10000)

    with col2:
        st.subheader("策略参数")

        # 根据策略动态显示参数
        if "WaveTrend" in strategy:
            st.number_input("WT Length", value=10)
            st.number_input("WT AvgLength", value=21)
            st.number_input("超买阈值", value=60)
            st.number_input("超卖阈值", value=-60)
        elif "MACD" in strategy:
            st.number_input("Fast Period", value=12)
            st.number_input("Slow Period", value=26)
            st.number_input("Signal Period", value=9)
        else:
            st.number_input("EMA Short", value=12)
            st.number_input("EMA Long", value=50)

    st.markdown("---")

    # 运行回测按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("运行回测", use_container_width=True, type="primary"):
            with st.spinner("回测运行中..."):
                import time
                time.sleep(2)  # 模拟回测时间
                st.session_state['backtest_done'] = True
                st.rerun()

    # 显示回测结果
    if st.session_state.get('backtest_done', False):
        st.markdown("---")
        st.subheader("回测结果")

        # 指标卡片
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric("总收益", "+28.5%")
        with col2:
            st.metric("年化收益", "+35.2%")
        with col3:
            st.metric("最大回撤", "-8.6%")
        with col4:
            st.metric("夏普比率", "1.85")
        with col5:
            st.metric("胜率", "56.3%")

        # 权益曲线
        st.subheader("权益曲线")

        dates = pd.date_range(start='2024-01-01', periods=250, freq='D')
        equity = 100000 * (1 + np.cumsum(np.random.randn(250) * 0.01))
        benchmark = 100000 * (1 + np.cumsum(np.random.randn(250) * 0.008))

        chart_df = pd.DataFrame({
            '策略': equity,
            '基准': benchmark
        }, index=dates)

        st.line_chart(chart_df, height=400)

        # 详细统计
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("收益统计")
            stats_df = pd.DataFrame({
                '指标': ['总收益率', '年化收益率', '月均收益', '日均收益', '最大单日收益', '最大单日亏损'],
                '数值': ['28.5%', '35.2%', '2.4%', '0.11%', '3.2%', '-2.1%']
            })
            st.dataframe(stats_df, hide_index=True, use_container_width=True)

        with col2:
            st.subheader("风险统计")
            risk_df = pd.DataFrame({
                '指标': ['最大回撤', '回撤恢复天数', '波动率', '下行波动率', '胜率', '盈亏比'],
                '数值': ['-8.6%', '15天', '18.5%', '12.3%', '56.3%', '1.45']
            })
            st.dataframe(risk_df, hide_index=True, use_container_width=True)

        # 交易记录
        st.subheader("交易记录")

        trades_df = pd.DataFrame({
            '日期': ['2024-03-15', '2024-03-18', '2024-04-02', '2024-04-10'],
            '合约': ['RB2405', 'I2405', 'RB2405', 'I2405'],
            '方向': ['多', '多', '空', '多'],
            '开仓价': [3650, 850, 3720, 810],
            '平仓价': [3720, 830, 3680, 860],
            '手数': [3, 2, 3, 2],
            '盈亏': ['+2,100', '-400', '+1,200', '+1,000'],
            '盈亏%': ['+1.92%', '-2.35%', '+1.08%', '+6.17%']
        })

        st.dataframe(trades_df, hide_index=True, use_container_width=True)

        # 导出按钮
        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button(
                "导出回测报告",
                data="回测报告内容...",
                file_name="backtest_report.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                "导出交易记录",
                data="交易记录内容...",
                file_name="trades.csv",
                mime="text/csv"
            )
        with col3:
            if st.button("保存到Notion", use_container_width=True):
                st.success("已保存到Notion!")


def render_settings():
    """渲染系统设置页面"""
    st.title("系统设置")

    tab1, tab2, tab3, tab4 = st.tabs(["基础设置", "品种配置", "网关设置", "数据管理"])

    with tab1:
        st.subheader("基础设置")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**账户设置**")
            st.number_input("初始资金", value=100000, min_value=10000)
            st.selectbox("结算货币", ["CNY", "USD"])

            st.write("**显示设置**")
            st.checkbox("深色模式", value=True)
            st.selectbox("刷新频率", ["1秒", "3秒", "5秒", "10秒"])

        with col2:
            st.write("**通知设置**")
            st.checkbox("成交通知", value=True)
            st.checkbox("风控预警通知", value=True)
            st.checkbox("策略信号通知", value=False)

            st.write("**日志设置**")
            st.selectbox("日志级别", ["DEBUG", "INFO", "WARNING", "ERROR"])
            st.checkbox("保存日志到文件", value=True)

    with tab2:
        st.subheader("品种配置")

        instruments_df = pd.DataFrame({
            '品种代码': ['RB', 'I', 'AU', 'CU', 'AL'],
            '品种名称': ['螺纹钢', '铁矿石', '黄金', '沪铜', '沪铝'],
            '合约乘数': [10, 100, 1000, 5, 5],
            '保证金率': ['10%', '12%', '8%', '10%', '10%'],
            '手续费': ['成交额万分之一', '成交额万分之一', '10元/手', '成交额万分之0.5', '3元/手']
        })

        st.dataframe(instruments_df, hide_index=True, use_container_width=True)

        with st.expander("添加/编辑品种"):
            col1, col2 = st.columns(2)
            with col1:
                st.text_input("品种代码")
                st.text_input("品种名称")
                st.number_input("合约乘数", value=10)
            with col2:
                st.number_input("保证金率", value=0.1, format="%.2f")
                st.selectbox("手续费类型", ["按比例", "固定金额"])
                st.number_input("手续费", value=0.0001, format="%.4f")

            st.button("保存品种配置")

    with tab3:
        st.subheader("网关设置")

        gateway_type = st.selectbox("网关类型", ["模拟盘", "CTP实盘"])

        if gateway_type == "模拟盘":
            st.info("模拟盘模式，无需配置网关连接信息")
            st.number_input("模拟滑点(跳)", value=1, min_value=0)
            st.number_input("模拟延迟(ms)", value=100, min_value=0)

        else:
            st.write("**CTP连接配置**")
            st.text_input("交易前置地址", placeholder="tcp://180.168.146.187:10130")
            st.text_input("行情前置地址", placeholder="tcp://180.168.146.187:10131")
            st.text_input("Broker ID", placeholder="9999")
            st.text_input("用户名")
            st.text_input("密码", type="password")
            st.text_input("AppID")
            st.text_input("AuthCode")

            if st.button("测试连接"):
                with st.spinner("连接中..."):
                    import time
                    time.sleep(2)
                st.success("连接成功!")

    with tab4:
        st.subheader("数据管理")

        st.write("**数据库信息**")
        st.info("数据来源: TianQin量化数据库")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("K线数据量", "12.5GB")
            st.metric("交易记录数", "15,680")

        with col2:
            st.metric("数据时间范围", "2020-01 至 2025-12")
            st.metric("品种数量", "45")

        st.write("**数据操作**")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("更新数据", use_container_width=True):
                st.info("正在更新数据...")
        with col2:
            if st.button("清理缓存", use_container_width=True):
                st.success("缓存已清理")
        with col3:
            if st.button("备份数据库", use_container_width=True):
                st.info("正在备份...")


if __name__ == "__main__":
    main()
