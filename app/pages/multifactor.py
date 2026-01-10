# coding=utf-8
"""
多因子选股回测页面
支持沪深300/中证500/中证1000成分股
使用LightGBM排序模型
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import threading
import queue

# 添加项目路径（确保multifactor模块可导入）
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def render_multifactor_page():
    """渲染多因子选股页面"""
    st.title("多因子选股回测")

    # 子页面选择
    sub_page = st.radio(
        "功能选择",
        ["策略回测", "因子分析", "模型训练"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    if sub_page == "策略回测":
        _render_backtest_page()
    elif sub_page == "因子分析":
        _render_factor_analysis()
    else:
        _render_model_training()


def _render_backtest_page():
    """渲染回测页面"""
    # 三列布局
    col1, col2, col3 = st.columns([1, 1.2, 0.8])

    with col1:
        st.subheader("回测配置")

        # 指数选择
        index_options = {
            "zz1000": "中证1000 (1000只)",
            "zz500": "中证500 (500只)",
            "hs300": "沪深300 (300只)"
        }
        index_name = st.selectbox(
            "股票池",
            options=list(index_options.keys()),
            format_func=lambda x: index_options[x],
            index=0
        )

        # 时间范围
        col_a, col_b = st.columns(2)
        with col_a:
            start_date = st.date_input(
                "开始日期",
                value=datetime(2021, 1, 1),
                min_value=datetime(2015, 1, 1),
                max_value=datetime.now()
            )
        with col_b:
            end_date = st.date_input(
                "结束日期",
                value=datetime(2025, 12, 31),
                min_value=datetime(2015, 1, 1),
                max_value=datetime.now()
            )

        # 持仓设置
        hold_num = st.slider("持仓数量", min_value=5, max_value=50, value=20)
        rebalance_num = st.slider("每日换仓数", min_value=1, max_value=10, value=2)

        # 模型设置
        train_days = st.number_input("训练窗口(天)", min_value=60, max_value=500, value=120)
        retrain_freq = st.number_input("重训频率(天)", min_value=5, max_value=60, value=20)

        # 初始资金
        initial_capital = st.number_input(
            "初始资金",
            min_value=100000,
            max_value=10000000,
            value=1000000,
            step=100000
        )

        # 最大股票数限制
        max_stocks = st.number_input(
            "股票数限制 (0=不限)",
            min_value=0,
            max_value=1000,
            value=0,
            help="限制股票池大小，用于快速测试"
        )

        st.markdown("---")

        # 运行按钮
        run_btn = st.button("开始回测", type="primary", use_container_width=True)

    with col2:
        st.subheader("因子配置")

        # 因子分组
        factor_groups = {
            "动量因子": ["mom_5d", "mom_10d", "mom_20d", "mom_60d", "reversal_5d"],
            "波动因子": ["vol_5d", "vol_20d", "vol_60d", "vol_ratio"],
            "成交量因子": ["vol_ma_ratio", "amount_ma_ratio"],
            "价格因子": ["price_to_high", "price_to_low", "price_position"],
            "均线因子": ["ma_bias_5", "ma_bias_20", "ma_bias_60", "ma_bull"],
            "技术指标": ["rsi", "macd", "macd_hist"]
        }

        selected_factors = []
        for group_name, factors in factor_groups.items():
            with st.expander(group_name, expanded=True):
                cols = st.columns(3)
                for i, factor in enumerate(factors):
                    with cols[i % 3]:
                        if st.checkbox(factor, value=True, key=f"factor_{factor}"):
                            selected_factors.append(factor)

        st.caption(f"已选择 {len(selected_factors)} 个因子")

    with col3:
        st.subheader("策略说明")

        st.markdown("""
        **多因子选股策略**

        使用LightGBM排序模型预测股票未来收益，选取预测得分最高的股票持有。

        **核心逻辑**
        1. 计算所有股票的因子值
        2. 使用历史数据训练LightGBM模型
        3. 预测各股票未来5日收益
        4. 选取预测得分Top N持有
        5. 每日换仓M只

        **风险提示**
        - 历史回测不代表未来
        - 小盘股流动性风险
        - 换手率成本影响
        """)

        st.markdown("---")
        st.caption("模型: LightGBM GBDT")
        st.caption("目标: 5日收益率")
        st.caption("训练: 滚动窗口")

    st.markdown("---")

    # 运行回测
    if run_btn:
        _run_multifactor_backtest(
            index_name=index_name,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            hold_num=hold_num,
            rebalance_num=rebalance_num,
            train_days=train_days,
            retrain_freq=retrain_freq,
            initial_capital=initial_capital,
            max_stocks=max_stocks,
            selected_factors=selected_factors
        )

    # 显示结果
    if 'multifactor_result' in st.session_state and st.session_state.multifactor_result:
        _render_backtest_results()


def _run_multifactor_backtest(index_name, start_date, end_date, hold_num,
                               rebalance_num, train_days, retrain_freq,
                               initial_capital, max_stocks, selected_factors):
    """运行多因子回测"""
    # 确保路径正确
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    try:
        from multifactor.run_multifactor import run_multifactor_strategy
        from multifactor.data_loader import get_index_components

        # 获取股票池大小
        stock_pool = get_index_components(index_name)
        pool_size = len(stock_pool)
        if max_stocks > 0:
            pool_size = min(pool_size, max_stocks)

        st.info(f"准备回测: {index_name} ({pool_size}只), {start_date} ~ {end_date}")

        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("正在加载数据和计算因子...")
        progress_bar.progress(10)

        # 运行回测
        with st.spinner("正在运行多因子回测，这可能需要几分钟..."):
            result = run_multifactor_strategy(
                start_date=start_date,
                end_date=end_date,
                hold_num=hold_num,
                rebalance_num=rebalance_num,
                train_days=train_days,
                retrain_freq=retrain_freq,
                use_sample=False,
                index_name=index_name,
                max_stocks=max_stocks
            )

        progress_bar.progress(100)
        status_text.empty()

        if result:
            st.session_state.multifactor_result = result
            st.success(f"回测完成! 累计收益: {result.total_return*100:.2f}%, 夏普: {result.sharpe_ratio:.2f}")
        else:
            st.error("回测失败，请检查数据")

    except Exception as e:
        st.error(f"回测出错: {e}")
        import traceback
        st.code(traceback.format_exc())


def _render_backtest_results():
    """渲染回测结果"""
    result = st.session_state.multifactor_result

    # 标签页
    tabs = st.tabs(["概览", "资金曲线", "交易记录", "因子分析"])

    with tabs[0]:
        _render_overview(result)

    with tabs[1]:
        _render_equity_curve(result)

    with tabs[2]:
        _render_trades(result)

    with tabs[3]:
        _render_factor_importance(result)


def _render_overview(result):
    """渲染概览"""
    st.subheader("回测概览")

    # 关键指标
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("累计收益", f"{result.total_return*100:.2f}%")
    with col2:
        st.metric("年化收益", f"{result.annual_return*100:.2f}%")
    with col3:
        st.metric("最大回撤", f"{result.max_drawdown*100:.2f}%")
    with col4:
        st.metric("夏普比率", f"{result.sharpe_ratio:.2f}")
    with col5:
        st.metric("胜率", f"{result.win_rate*100:.1f}%")
    with col6:
        st.metric("年换手率", f"{result.turnover:.1f}x")

    st.markdown("---")

    # 详细指标
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**收益指标**")
        st.write(f"- 初始资金: ¥{result.initial_capital:,.0f}")
        st.write(f"- 期末资金: ¥{result.final_value:,.0f}")
        st.write(f"- 总收益率: {result.total_return*100:.2f}%")
        st.write(f"- 年化收益: {result.annual_return*100:.2f}%")

    with col2:
        st.write("**风险指标**")
        st.write(f"- 最大回撤: {result.max_drawdown*100:.2f}%")
        st.write(f"- 夏普比率: {result.sharpe_ratio:.2f}")
        st.write(f"- 卡玛比率: {result.calmar_ratio:.2f}")

    with col3:
        st.write("**交易统计**")
        st.write(f"- 总交易次数: {result.total_trades}")
        st.write(f"- 胜率: {result.win_rate*100:.1f}%")
        st.write(f"- 年换手率: {result.turnover:.1f}倍")


def _render_equity_curve(result):
    """渲染资金曲线"""
    st.subheader("资金曲线")

    if result.equity_curve is None or result.equity_curve.empty:
        st.warning("没有资金曲线数据")
        return

    df = result.equity_curve

    # 创建图表
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=('账户净值', '回撤')
    )

    # 净值曲线
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['total_value'],
            name='账户净值',
            line=dict(color='#2196F3', width=2),
            fill='tozeroy',
            fillcolor='rgba(33, 150, 243, 0.1)'
        ),
        row=1, col=1
    )

    # 回撤曲线
    rolling_max = df['total_value'].cummax()
    drawdown = (df['total_value'] - rolling_max) / rolling_max * 100

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=drawdown,
            name='回撤',
            line=dict(color='#F44336', width=1),
            fill='tozeroy',
            fillcolor='rgba(244, 67, 54, 0.3)'
        ),
        row=2, col=1
    )

    fig.update_layout(height=600, hovermode='x unified', showlegend=True)
    fig.update_yaxes(title_text="净值 (元)", row=1, col=1)
    fig.update_yaxes(title_text="回撤 (%)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


def _render_trades(result):
    """渲染交易记录"""
    st.subheader("交易记录")

    if not result.trades:
        st.warning("没有交易记录")
        return

    # 转换为DataFrame
    trades_data = []
    for t in result.trades:
        trades_data.append({
            '日期': t.date,
            '代码': t.code,
            '方向': t.direction,
            '价格': f"{t.price:.2f}",
            '数量': t.shares,
            '金额': f"¥{t.amount:,.0f}",
            '手续费': f"¥{t.commission:.2f}",
            '盈亏': f"¥{t.pnl:,.0f}" if hasattr(t, 'pnl') else "-"
        })

    df_trades = pd.DataFrame(trades_data)

    # 筛选
    col1, col2 = st.columns(2)
    with col1:
        direction_filter = st.multiselect(
            "方向筛选",
            options=df_trades['方向'].unique().tolist(),
            default=df_trades['方向'].unique().tolist()
        )

    df_filtered = df_trades[df_trades['方向'].isin(direction_filter)]

    st.dataframe(df_filtered, use_container_width=True, hide_index=True)

    # 统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("买入次数", len(df_trades[df_trades['方向'] == 'BUY']))
    with col2:
        st.metric("卖出次数", len(df_trades[df_trades['方向'] == 'SELL']))
    with col3:
        total_commission = sum(t.commission for t in result.trades)
        st.metric("总手续费", f"¥{total_commission:,.0f}")


def _render_factor_importance(result):
    """渲染因子重要性分析"""
    st.subheader("因子分析")

    st.info("因子重要性分析需要完整的回测结果数据")

    # 确保路径正确
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    # 显示使用的因子
    from multifactor.factors import get_factor_list
    factors = get_factor_list()

    st.write("**使用的因子列表**")

    factor_groups = {
        "动量因子": ["mom_5d", "mom_10d", "mom_20d", "mom_60d", "reversal_5d"],
        "波动因子": ["vol_5d", "vol_20d", "vol_60d", "vol_ratio"],
        "成交量因子": ["vol_ma_ratio", "amount_ma_ratio"],
        "价格因子": ["price_to_high", "price_to_low", "price_position"],
        "均线因子": ["ma_bias_5", "ma_bias_20", "ma_bias_60", "ma_bull"],
        "技术指标": ["rsi", "macd", "macd_hist"]
    }

    for group_name, group_factors in factor_groups.items():
        used = [f for f in group_factors if f in factors]
        st.write(f"**{group_name}**: {', '.join(used)}")


def _render_factor_analysis():
    """因子分析页面"""
    st.subheader("因子分析")

    st.info("此功能正在开发中，敬请期待...")

    st.markdown("""
    **计划功能**
    - 单因子IC分析
    - 因子相关性矩阵
    - 因子分层回测
    - 因子衰减分析
    """)


def _render_model_training():
    """模型训练页面"""
    st.subheader("模型训练")

    # 确保路径正确（Streamlit热重载可能丢失路径）
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    # 检查依赖
    try:
        from multifactor.model import (
            is_gpu_available, HAS_LIGHTGBM, HAS_OPTUNA,
            cross_validate_ranker, optimize_hyperparams,
            train_and_save_model, list_saved_models, load_saved_model
        )
        from multifactor.data_loader import StockDataLoader, get_index_components
        from multifactor.factors import calculate_all_factors, get_factor_list, prepare_factor_data
    except ImportError as e:
        st.error(f"导入模块失败: {e}")
        st.info(f"项目路径: {_project_root}")
        st.info(f"sys.path: {sys.path[:3]}")
        return

    # 状态显示
    col1, col2 = st.columns(2)
    with col1:
        st.metric("LightGBM", "已安装" if HAS_LIGHTGBM else "未安装")
    with col2:
        st.metric("Optuna", "已安装" if HAS_OPTUNA else "未安装")

    st.markdown("---")

    # 训练模式选择
    train_mode = st.radio(
        "训练模式",
        ["快速训练", "交叉验证", "超参数调优"],
        horizontal=True
    )

    # 三列布局
    col1, col2, col3 = st.columns([1, 1.2, 0.8])

    with col1:
        st.write("**数据设置**")

        # 指数选择
        index_options = {
            "zz1000": "中证1000",
            "zz500": "中证500",
            "hs300": "沪深300"
        }
        index_name = st.selectbox(
            "股票池",
            options=list(index_options.keys()),
            format_func=lambda x: index_options[x],
            key="train_index"
        )

        # 股票数限制
        max_stocks = st.number_input(
            "股票数限制 (0=不限)",
            min_value=0,
            max_value=1000,
            value=100,
            help="限制股票数量加快训练速度"
        )

        # 日期范围
        col_a, col_b = st.columns(2)
        with col_a:
            train_start = st.date_input(
                "开始日期",
                value=datetime(2023, 1, 1),
                key="train_start"
            )
        with col_b:
            train_end = st.date_input(
                "结束日期",
                value=datetime(2024, 12, 31),
                key="train_end"
            )

        # LightGBM CPU版本已足够快，无需GPU

    with col2:
        st.write("**模型参数**")

        if train_mode == "快速训练":
            num_leaves = st.slider("叶子节点数", 15, 127, 31)
            learning_rate = st.select_slider(
                "学习率",
                options=[0.01, 0.02, 0.05, 0.1, 0.2],
                value=0.05
            )
            n_estimators = st.slider("树数量", 50, 300, 100)
            train_ratio = st.slider("训练集比例", 0.6, 0.9, 0.8)

            model_params = {
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators
            }

        elif train_mode == "交叉验证":
            n_splits = st.slider("折数", 3, 10, 5)
            num_leaves = st.slider("叶子节点数", 15, 127, 31, key="cv_leaves")
            learning_rate = st.select_slider(
                "学习率",
                options=[0.01, 0.02, 0.05, 0.1, 0.2],
                value=0.05,
                key="cv_lr"
            )

            model_params = {
                "num_leaves": num_leaves,
                "learning_rate": learning_rate
            }

        else:  # 超参数调优
            if not HAS_OPTUNA:
                st.warning("需要安装Optuna: `pip install optuna`")

            n_trials = st.slider("试验次数", 10, 100, 30)
            n_cv_splits = st.slider("CV折数", 2, 5, 3)

            st.caption("搜索空间:")
            st.code("""
num_leaves: [15, 127]
learning_rate: [0.01, 0.2]
n_estimators: [50, 300]
feature_fraction: [0.5, 1.0]
reg_alpha/lambda: [1e-8, 10.0]
            """)

        # 因子选择
        st.write("**因子选择**")
        all_factors = get_factor_list()
        selected_factors = st.multiselect(
            "选择因子",
            options=all_factors,
            default=all_factors,
            key="train_factors"
        )

    with col3:
        st.write("**模型管理**")

        # 模型保存目录
        model_dir = st.text_input(
            "模型目录",
            value="D:/期货/回测改造/models",
            key="model_dir"
        )

        # 模型名称
        model_name = st.text_input(
            "模型名称 (留空自动生成)",
            value="",
            key="model_name"
        )

        st.markdown("---")

        # 已保存的模型
        st.write("**已保存模型**")
        saved_models = list_saved_models(model_dir)

        if saved_models:
            for m in saved_models[:5]:
                with st.expander(f"{m['model_name']} (IC: {m.get('val_ic', 0):.4f})"):
                    st.write(f"创建时间: {m.get('created_at', 'N/A')[:19]}")
                    st.write(f"训练样本: {m.get('train_samples', 0)}")
                    st.write(f"验证IC: {m.get('val_ic', 0):.4f}")
                    if st.button("加载", key=f"load_{m['model_name']}"):
                        st.session_state.loaded_model = m['model_path']
                        st.success(f"已选择: {m['model_name']}")
        else:
            st.caption("暂无保存的模型")

    st.markdown("---")

    # 训练按钮
    if st.button("开始训练", type="primary", use_container_width=True):
        _run_model_training(
            train_mode=train_mode,
            index_name=index_name,
            max_stocks=max_stocks,
            start_date=train_start.strftime("%Y-%m-%d"),
            end_date=train_end.strftime("%Y-%m-%d"),
            model_params=model_params if train_mode != "超参数调优" else None,
            selected_factors=selected_factors,
            model_dir=model_dir,
            model_name=model_name if model_name else None,
            n_splits=n_splits if train_mode == "交叉验证" else 5,
            n_trials=n_trials if train_mode == "超参数调优" else 30,
            n_cv_splits=n_cv_splits if train_mode == "超参数调优" else 3,
            train_ratio=train_ratio if train_mode == "快速训练" else 0.8
        )


def _run_model_training(train_mode, index_name, max_stocks, start_date, end_date,
                        model_params, selected_factors, model_dir, model_name,
                        n_splits, n_trials, n_cv_splits, train_ratio):
    """执行模型训练"""
    # 确保路径正确
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    try:
        from multifactor.model import (
            cross_validate_ranker, optimize_hyperparams,
            train_and_save_model
        )
        from multifactor.data_loader import StockDataLoader, get_index_components
        from multifactor.factors import calculate_all_factors, prepare_factor_data

        progress_bar = st.progress(0)
        status_text = st.empty()

        # 1. 加载数据
        status_text.text("正在加载数据...")
        progress_bar.progress(10)

        stock_pool = get_index_components(index_name)
        if max_stocks > 0:
            stock_pool = stock_pool[:max_stocks]

        loader = StockDataLoader()

        # 扩展日期范围
        from datetime import datetime, timedelta
        train_start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        extended_start = (train_start_dt - timedelta(days=200)).strftime("%Y-%m-%d")

        stock_data = {}
        for code in stock_pool:
            df = loader.get_stock_data(code, extended_start, end_date)
            if len(df) > 60:
                stock_data[code] = df

        status_text.text(f"已加载 {len(stock_data)} 只股票数据")
        progress_bar.progress(30)

        # 2. 计算因子
        status_text.text("正在计算因子...")
        for code in stock_data:
            stock_data[code] = calculate_all_factors(stock_data[code])

        progress_bar.progress(50)

        # 3. 准备训练数据
        status_text.text("正在准备训练数据...")

        all_dates = set()
        for df in stock_data.values():
            all_dates.update(df["date"].tolist())
        trading_dates = sorted([d for d in all_dates if start_date <= d <= end_date])

        # 收集因子数据
        factor_samples = []
        for date in trading_dates[::5]:  # 每5天采样一次加快速度
            day_data = prepare_factor_data(stock_data, date)
            if len(day_data) > 0:
                factor_samples.append(day_data)

        if not factor_samples:
            st.error("无法生成训练数据")
            return

        factor_data = pd.concat(factor_samples, ignore_index=True)
        st.info(f"训练数据: {len(factor_data)} 样本, {len(selected_factors)} 因子")

        progress_bar.progress(60)

        # 4. 执行训练
        if train_mode == "快速训练":
            status_text.text("正在训练模型...")

            result = train_and_save_model(
                factor_data=factor_data,
                factor_cols=selected_factors,
                save_dir=model_dir,
                model_name=model_name,
                model_params=model_params,
                use_gpu=False,
                train_ratio=train_ratio
            )

            progress_bar.progress(100)
            status_text.empty()

            st.success(f"训练完成! 验证IC: {result['val_ic']:.4f}")
            st.write(f"模型已保存: `{result['model_path']}`")

            # 显示特征重要性
            _display_feature_importance(result['meta'].get('feature_importance', {}))

        elif train_mode == "交叉验证":
            status_text.text("正在进行交叉验证...")

            def cv_callback(fold, total, ic):
                progress_bar.progress(60 + int(40 * fold / total))
                status_text.text(f"Fold {fold}/{total}: IC = {ic:.4f}")

            cv_result = cross_validate_ranker(
                factor_data=factor_data,
                factor_cols=selected_factors,
                n_splits=n_splits,
                model_params=model_params,
                use_gpu=False,
                callback=cv_callback
            )

            progress_bar.progress(100)
            status_text.empty()

            # 显示结果
            st.success(f"交叉验证完成! 平均IC: {cv_result['mean_ic']:.4f} ± {cv_result['std_ic']:.4f}")

            # IC曲线
            _display_cv_results(cv_result)

        else:  # 超参数调优
            status_text.text("正在进行超参数调优...")

            def optuna_callback(trial, total, ic):
                progress_bar.progress(60 + int(40 * trial / total))
                status_text.text(f"Trial {trial}/{total}: IC = {ic:.4f}")

            try:
                opt_result = optimize_hyperparams(
                    factor_data=factor_data,
                    factor_cols=selected_factors,
                    n_trials=n_trials,
                    n_cv_splits=n_cv_splits,
                    use_gpu=False,
                    callback=optuna_callback
                )

                progress_bar.progress(100)
                status_text.empty()

                st.success(f"调优完成! 最优IC: {opt_result['best_ic']:.4f}")

                # 显示最优参数
                st.write("**最优参数:**")
                st.json(opt_result['best_params'])

                # 用最优参数训练并保存
                if st.button("使用最优参数保存模型"):
                    save_result = train_and_save_model(
                        factor_data=factor_data,
                        factor_cols=selected_factors,
                        save_dir=model_dir,
                        model_name=model_name,
                        model_params=opt_result['best_params'],
                        use_gpu=use_gpu
                    )
                    st.success(f"模型已保存: {save_result['model_path']}")

            except ImportError:
                st.error("需要安装Optuna: pip install optuna")

    except Exception as e:
        st.error(f"训练出错: {e}")
        import traceback
        st.code(traceback.format_exc())


def _display_feature_importance(importance: dict):
    """显示特征重要性"""
    if not importance:
        return

    st.write("**特征重要性 Top 10:**")

    import plotly.graph_objects as go

    # 取Top 10
    top_features = list(importance.items())[:10]
    features = [f[0] for f in top_features]
    values = [f[1] for f in top_features]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker_color='#2196F3'
    ))
    fig.update_layout(
        height=300,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=100, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)


def _display_cv_results(cv_result: dict):
    """显示交叉验证结果"""
    import plotly.graph_objects as go

    # IC曲线
    fold_results = cv_result.get('fold_results', [])
    if fold_results:
        folds = [r['fold'] for r in fold_results]
        ics = [r['ic'] for r in fold_results]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=folds, y=ics,
            mode='lines+markers',
            name='IC',
            line=dict(color='#2196F3', width=2),
            marker=dict(size=10)
        ))
        fig.add_hline(y=cv_result['mean_ic'], line_dash="dash",
                      annotation_text=f"平均: {cv_result['mean_ic']:.4f}")
        fig.update_layout(
            title="各折IC值",
            xaxis_title="Fold",
            yaxis_title="IC",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

    # 特征重要性
    avg_importance = cv_result.get('avg_feature_importance', {})
    if avg_importance:
        _display_feature_importance(avg_importance)


if __name__ == "__main__":
    st.set_page_config(page_title="多因子选股", layout="wide")
    render_multifactor_page()
