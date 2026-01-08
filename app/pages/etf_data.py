# coding=utf-8
"""
ETF数据管理页面
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def render_etf_data_page():
    """渲染ETF数据管理页面"""
    st.markdown("## 📊 ETF数据管理")

    tab1, tab2, tab3 = st.tabs(["📥 数据更新", "📋 数据统计", "🔍 数据查询"])

    with tab1:
        _render_data_update()

    with tab2:
        _render_data_stats()

    with tab3:
        _render_data_query()


def _render_data_update():
    """数据更新"""
    st.markdown("### 📥 更新ETF数据")

    from core.etf_data_service import ETF_POOLS, ALL_ETFS, BIGBROTHER_POOL

    col1, col2 = st.columns(2)

    with col1:
        pool_options = ["全部", "BigBrother V14 池"] + list(ETF_POOLS.keys())
        selected_pool = st.selectbox("选择ETF池", pool_options)

        if selected_pool == "全部":
            codes_to_update = list(ALL_ETFS.keys())
        elif selected_pool == "BigBrother V14 池":
            codes_to_update = BIGBROTHER_POOL
        else:
            codes_to_update = list(ETF_POOLS[selected_pool].keys())

        selected_codes = st.multiselect(
            "或选择具体ETF",
            options=list(ALL_ETFS.keys()),
            default=codes_to_update,
            format_func=lambda x: f"{x} - {ALL_ETFS.get(x, '')}"
        )

    with col2:
        st.markdown("**更新选项**")
        force_update = st.checkbox("强制全量更新", value=False)
        include_benchmark = st.checkbox("包含沪深300指数", value=True)

    if include_benchmark:
        if "000300.SH" not in selected_codes:
            selected_codes.append("000300.SH")
        if "510300.SH" not in selected_codes:
            selected_codes.append("510300.SH")

    st.markdown(f"**将更新 {len(selected_codes)} 个标的**")

    if st.button("🚀 开始更新", type="primary"):
        if not selected_codes:
            st.error("请选择至少一个ETF")
            return

        try:
            from core.etf_data_service import get_etf_data_service

            ds = get_etf_data_service()

            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []

            for i, code in enumerate(selected_codes):
                status_text.text(f"正在更新: {code} ({i+1}/{len(selected_codes)})")

                try:
                    rows = ds.update_data(code, force=force_update)
                    results.append({"代码": code, "名称": ALL_ETFS.get(code, ""), "新增": rows, "状态": "成功"})
                except Exception as e:
                    results.append({"代码": code, "名称": ALL_ETFS.get(code, ""), "新增": 0, "状态": f"失败: {e}"})

                progress_bar.progress((i + 1) / len(selected_codes))

            progress_bar.empty()
            status_text.empty()

            st.success("更新完成!")

            result_df = pd.DataFrame(results)
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            success_count = len([r for r in results if r["状态"] == "成功"])
            total_rows = sum([r["新增"] for r in results])
            st.info(f"成功: {success_count}/{len(results)}，新增数据: {total_rows}条")

        except ImportError:
            st.error("请安装akshare: pip install akshare")
        except Exception as e:
            st.error(f"更新失败: {e}")


def _render_data_stats():
    """数据统计"""
    st.markdown("### 📋 数据统计")

    try:
        from core.etf_data_service import get_etf_data_service, ALL_ETFS

        ds = get_etf_data_service()
        info = ds.get_data_info()

        if len(info) == 0:
            st.warning("数据库为空，请先更新数据")
            return

        info["名称"] = info["code"].map(ALL_ETFS)
        info = info[["code", "名称", "rows", "start_date", "end_date"]]
        info.columns = ["代码", "名称", "数据条数", "开始日期", "结束日期"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("ETF数量", f"{len(info)}个")
        with col2:
            st.metric("总数据条数", f"{info['数据条数'].sum():,}")
        with col3:
            st.metric("最早日期", info["开始日期"].min())
        with col4:
            st.metric("最新日期", info["结束日期"].max())

        st.markdown("---")
        st.dataframe(info, use_container_width=True, hide_index=True)

        csv = info.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载统计数据", csv, "etf_stats.csv", "text/csv")

    except Exception as e:
        st.error(f"加载统计失败: {e}")


def _render_data_query():
    """数据查询"""
    st.markdown("### 🔍 数据查询")

    from core.etf_data_service import ALL_ETFS

    col1, col2 = st.columns(2)

    with col1:
        codes = list(ALL_ETFS.keys())
        selected_code = st.selectbox(
            "选择ETF",
            options=codes,
            format_func=lambda x: f"{x} - {ALL_ETFS.get(x, '')}"
        )

    with col2:
        date_range = st.date_input(
            "日期范围",
            value=(datetime(2024, 1, 1), datetime.now()),
            max_value=datetime.now()
        )

    if st.button("🔍 查询"):
        if len(date_range) != 2:
            st.error("请选择完整的日期范围")
            return

        start_date = date_range[0].strftime("%Y-%m-%d")
        end_date = date_range[1].strftime("%Y-%m-%d")

        try:
            from core.etf_data_service import get_etf_data_service

            ds = get_etf_data_service()
            df = ds.get_data_with_indicators(selected_code, start_date, end_date)

            if len(df) == 0:
                st.warning("无数据，请先更新")
                return

            st.success(f"查询到 {len(df)} 条数据")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # K线图
            st.markdown("### 📈 K线图")

            fig = go.Figure(data=[
                go.Candlestick(
                    x=df["date"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="K线"
                )
            ])

            if "ema_fast" in df.columns:
                fig.add_trace(go.Scatter(x=df["date"], y=df["ema_fast"],
                                          mode="lines", name="EMA20", line=dict(color="orange", width=1)))

            if "ema_slow" in df.columns:
                fig.add_trace(go.Scatter(x=df["date"], y=df["ema_slow"],
                                          mode="lines", name="EMA60", line=dict(color="blue", width=1)))

            fig.update_layout(height=400, xaxis_rangeslider_visible=False,
                              title=f"{selected_code} - {ALL_ETFS.get(selected_code, '')}")

            st.plotly_chart(fig, use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📥 下载数据", csv, f"{selected_code}_data.csv", "text/csv")

        except Exception as e:
            st.error(f"查询失败: {e}")
            import traceback
            st.code(traceback.format_exc())
