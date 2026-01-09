# coding=utf-8
"""
回测历史记录页面
查看、对比、导出历史回测结果
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def render_backtest_history_page():
    """渲染回测历史页面"""
    st.markdown("## 📚 回测历史记录")

    from utils.backtest_storage import get_backtest_storage

    storage = get_backtest_storage()

    # 顶部统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_count = storage.get_record_count()
        st.metric("总记录数", total_count)
    with col2:
        etf_count = storage.get_record_count("ETF")
        st.metric("ETF回测", etf_count)
    with col3:
        futures_count = storage.get_record_count("期货")
        st.metric("期货回测", futures_count)
    with col4:
        strategies = storage.get_strategies()
        st.metric("策略数量", len(strategies))

    st.markdown("---")

    # 筛选区
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        type_filter = st.selectbox("类型", ["全部", "ETF", "期货"])
        type_value = None if type_filter == "全部" else type_filter

    with col2:
        all_strategies = ["全部"] + storage.get_strategies(type_value)
        strategy_filter = st.selectbox("策略", all_strategies)
        strategy_value = None if strategy_filter == "全部" else strategy_filter

    with col3:
        sort_by = st.selectbox("排序", ["时间", "收益率", "夏普比率", "最大回撤"])

    with col4:
        page_size = st.selectbox("每页显示", [10, 20, 50], index=0)

    # 获取记录
    records = storage.get_records(
        backtest_type=type_value,
        strategy_name=strategy_value,
        limit=page_size
    )

    if not records:
        st.info("暂无回测记录")
        st.markdown("运行ETF或期货回测后，结果会自动保存到这里。")
        return

    # 转换为DataFrame显示
    records_data = []
    for r in records:
        records_data.append({
            "选择": False,
            "ID": r.backtest_id,
            "类型": r.backtest_type,
            "策略": r.strategy_name,
            "标的": r.symbols[:30] + "..." if len(r.symbols) > 30 else r.symbols,
            "时间范围": f"{r.start_date} ~ {r.end_date}",
            "初始资金": f"¥{r.initial_capital:,.0f}",
            "收益率": f"{r.total_return*100:+.2f}%",
            "年化": f"{r.annual_return*100:.2f}%",
            "回撤": f"{r.max_drawdown*100:.2f}%",
            "夏普": f"{r.sharpe_ratio:.2f}",
            "胜率": f"{r.win_rate*100:.1f}%",
            "交易数": r.total_trades,
            "创建时间": r.created_at[:16],
        })

    df = pd.DataFrame(records_data)

    # 排序
    sort_map = {
        "时间": "创建时间",
        "收益率": "收益率",
        "夏普比率": "夏普",
        "最大回撤": "回撤"
    }

    # 多选对比
    st.markdown("### 回测列表")
    st.caption("选中多条记录可进行对比分析")

    # 使用data_editor实现多选
    edited_df = st.data_editor(
        df,
        column_config={
            "选择": st.column_config.CheckboxColumn("选", default=False, width="small"),
            "ID": st.column_config.TextColumn("ID", width="small"),
            "类型": st.column_config.TextColumn("类型", width="small"),
            "策略": st.column_config.TextColumn("策略", width="medium"),
            "收益率": st.column_config.TextColumn("收益率", width="small"),
        },
        disabled=["ID", "类型", "策略", "标的", "时间范围", "初始资金", "收益率", "年化", "回撤", "夏普", "胜率", "交易数", "创建时间"],
        hide_index=True,
        use_container_width=True,
        key="backtest_history_table"
    )

    # 获取选中的记录
    selected_ids = edited_df[edited_df["选择"] == True]["ID"].tolist()

    # 操作按钮
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 查看详情", disabled=len(selected_ids) != 1):
            if selected_ids:
                st.session_state['view_backtest_id'] = selected_ids[0]
                st.rerun()

    with col2:
        if st.button("📈 对比分析", disabled=len(selected_ids) < 2):
            if len(selected_ids) >= 2:
                st.session_state['compare_backtest_ids'] = selected_ids
                st.rerun()

    with col3:
        if st.button("📥 导出选中", disabled=len(selected_ids) == 0):
            if selected_ids:
                csv = storage.export_to_csv(selected_ids)
                st.download_button(
                    "下载CSV",
                    csv.encode('utf-8-sig'),
                    "backtest_records.csv",
                    "text/csv",
                    key="download_selected"
                )

    with col4:
        if st.button("🗑️ 删除选中", disabled=len(selected_ids) == 0):
            if selected_ids:
                st.session_state['delete_backtest_ids'] = selected_ids

    # 处理删除确认
    if 'delete_backtest_ids' in st.session_state:
        ids_to_delete = st.session_state['delete_backtest_ids']
        st.warning(f"确定要删除 {len(ids_to_delete)} 条记录吗？")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("确认删除", type="primary"):
                for bid in ids_to_delete:
                    storage.delete_record(bid)
                del st.session_state['delete_backtest_ids']
                st.success("删除成功")
                st.rerun()
        with col2:
            if st.button("取消"):
                del st.session_state['delete_backtest_ids']
                st.rerun()

    # 查看详情
    if 'view_backtest_id' in st.session_state:
        st.markdown("---")
        _render_record_detail(storage, st.session_state['view_backtest_id'])

    # 对比分析
    if 'compare_backtest_ids' in st.session_state:
        st.markdown("---")
        _render_comparison(storage, st.session_state['compare_backtest_ids'])


def _render_record_detail(storage, backtest_id: str):
    """渲染回测详情"""
    record = storage.get_record(backtest_id)
    if not record:
        st.error("记录不存在")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 回测详情: {record.strategy_name}")
    with col2:
        if st.button("关闭详情"):
            del st.session_state['view_backtest_id']
            st.rerun()

    # 基本信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("类型", record.backtest_type)
        st.metric("初始资金", f"¥{record.initial_capital:,.0f}")
    with col2:
        st.metric("时间范围", f"{record.start_date} ~ {record.end_date}")
        st.metric("最终权益", f"¥{record.final_value:,.0f}")
    with col3:
        st.metric("累计收益", f"{record.total_return*100:+.2f}%")
        st.metric("年化收益", f"{record.annual_return*100:.2f}%")
    with col4:
        st.metric("最大回撤", f"{record.max_drawdown*100:.2f}%")
        st.metric("夏普比率", f"{record.sharpe_ratio:.2f}")

    # 标的列表
    st.markdown("**标的:** " + record.symbols)

    # 策略参数
    st.markdown("**策略参数:**")
    params = json.loads(record.params_json)
    st.json(params)

    # 备注
    st.markdown("**备注:**")
    new_notes = st.text_area("", value=record.notes, key="record_notes", height=80)
    if new_notes != record.notes:
        if st.button("保存备注"):
            storage.update_notes(backtest_id, new_notes)
            st.success("备注已保存")

    # 资金曲线
    if record.equity_csv:
        st.markdown("### 资金曲线")
        try:
            import io
            equity_df = pd.read_csv(io.StringIO(record.equity_csv))

            # 查找日期列
            date_col = None
            for col in ['date', 'Date', 'datetime', 'time']:
                if col in equity_df.columns:
                    date_col = col
                    break

            if date_col is None and equity_df.columns[0] not in ['total_value', 'equity', 'value']:
                date_col = equity_df.columns[0]

            # 查找权益列
            value_col = None
            for col in ['total_value', 'equity', 'value', 'balance']:
                if col in equity_df.columns:
                    value_col = col
                    break

            if date_col and value_col:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=equity_df[date_col],
                    y=equity_df[value_col],
                    mode='lines',
                    name='权益',
                    line=dict(color='#1f77b4', width=2)
                ))
                fig.update_layout(height=300, margin=dict(l=50, r=50, t=30, b=30))
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"无法解析资金曲线: {e}")

    # 交易记录
    st.markdown("### 交易记录")
    trades = json.loads(record.trades_json)
    if trades:
        trades_df = pd.DataFrame(trades)
        st.dataframe(trades_df, use_container_width=True, hide_index=True)
    else:
        st.info("无交易记录")


def _render_comparison(storage, backtest_ids: list):
    """渲染对比分析"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"### 回测对比分析 ({len(backtest_ids)}条记录)")
    with col2:
        if st.button("关闭对比"):
            del st.session_state['compare_backtest_ids']
            st.rerun()

    records = [storage.get_record(bid) for bid in backtest_ids]
    records = [r for r in records if r is not None]

    if len(records) < 2:
        st.error("需要至少2条有效记录")
        return

    # 对比表格
    compare_data = {
        "指标": ["策略名称", "类型", "时间范围", "初始资金", "累计收益", "年化收益",
                 "最大回撤", "夏普比率", "胜率", "交易次数", "创建时间"]
    }

    for i, r in enumerate(records):
        compare_data[f"回测{i+1}"] = [
            r.strategy_name,
            r.backtest_type,
            f"{r.start_date}~{r.end_date}",
            f"¥{r.initial_capital:,.0f}",
            f"{r.total_return*100:+.2f}%",
            f"{r.annual_return*100:.2f}%",
            f"{r.max_drawdown*100:.2f}%",
            f"{r.sharpe_ratio:.2f}",
            f"{r.win_rate*100:.1f}%",
            str(r.total_trades),
            r.created_at[:16]
        ]

    compare_df = pd.DataFrame(compare_data)
    st.dataframe(compare_df, use_container_width=True, hide_index=True)

    # 对比图表
    st.markdown("### 指标对比图")

    metrics = ['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio', 'win_rate']
    metric_names = ['累计收益', '年化收益', '最大回撤', '夏普比率', '胜率']

    fig = make_subplots(rows=1, cols=len(metrics), subplot_titles=metric_names)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        values = [getattr(r, metric) * (100 if metric in ['total_return', 'annual_return', 'max_drawdown', 'win_rate'] else 1)
                  for r in records]
        labels = [f"回测{j+1}" for j in range(len(records))]

        fig.add_trace(
            go.Bar(x=labels, y=values, marker_color=colors[:len(records)], showlegend=False),
            row=1, col=i+1
        )

    fig.update_layout(height=300, margin=dict(l=50, r=50, t=50, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # 资金曲线对比
    st.markdown("### 资金曲线对比")

    fig_equity = go.Figure()
    for i, r in enumerate(records):
        if r.equity_csv:
            try:
                import io
                equity_df = pd.read_csv(io.StringIO(r.equity_csv))

                # 查找列
                date_col = None
                for col in ['date', 'Date', 'datetime', 'time']:
                    if col in equity_df.columns:
                        date_col = col
                        break
                if date_col is None:
                    date_col = equity_df.columns[0]

                value_col = None
                for col in ['total_value', 'equity', 'value', 'cumulative_return']:
                    if col in equity_df.columns:
                        value_col = col
                        break

                if date_col and value_col:
                    # 归一化为收益率
                    values = equity_df[value_col]
                    if value_col != 'cumulative_return':
                        values = (values / values.iloc[0] - 1) * 100

                    fig_equity.add_trace(go.Scatter(
                        x=equity_df[date_col],
                        y=values,
                        mode='lines',
                        name=f"{r.strategy_name} ({r.backtest_id[:6]})",
                        line=dict(color=colors[i % len(colors)], width=2)
                    ))
            except:
                pass

    fig_equity.update_layout(
        height=400,
        yaxis_title="收益率 (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=50, b=30)
    )
    st.plotly_chart(fig_equity, use_container_width=True)
