# coding=utf-8
"""
参数扫描可视化工具
生成热力图、3D曲面图等
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def create_heatmap_data(
    results_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = 'sharpe_ratio'
) -> Tuple[np.ndarray, List, List]:
    """
    创建热力图数据

    Args:
        results_df: 参数扫描结果DataFrame
        x_param: X轴参数名
        y_param: Y轴参数名
        metric: 指标名

    Returns:
        (z_matrix, x_values, y_values)
    """
    if x_param not in results_df.columns or y_param not in results_df.columns:
        raise ValueError(f"参数 {x_param} 或 {y_param} 不在结果中")

    if metric not in results_df.columns:
        raise ValueError(f"指标 {metric} 不在结果中")

    # 获取唯一值
    x_values = sorted(results_df[x_param].unique())
    y_values = sorted(results_df[y_param].unique())

    # 创建矩阵
    z_matrix = np.zeros((len(y_values), len(x_values)))

    for i, y_val in enumerate(y_values):
        for j, x_val in enumerate(x_values):
            mask = (results_df[x_param] == x_val) & (results_df[y_param] == y_val)
            if mask.any():
                z_matrix[i, j] = results_df.loc[mask, metric].values[0]
            else:
                z_matrix[i, j] = np.nan

    return z_matrix, x_values, y_values


def create_plotly_heatmap(
    results_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = 'sharpe_ratio',
    title: str = None
) -> dict:
    """
    创建Plotly热力图配置

    Args:
        results_df: 参数扫描结果DataFrame
        x_param: X轴参数名
        y_param: Y轴参数名
        metric: 指标名
        title: 图表标题

    Returns:
        Plotly figure dict
    """
    z_matrix, x_values, y_values = create_heatmap_data(
        results_df, x_param, y_param, metric
    )

    metric_labels = {
        'sharpe_ratio': 'Sharpe Ratio',
        'total_return': '总收益率',
        'annual_return': '年化收益率',
        'max_drawdown': '最大回撤',
        'win_rate': '胜率',
        'profit_factor': '盈亏比'
    }

    fig_data = {
        'data': [{
            'type': 'heatmap',
            'z': z_matrix.tolist(),
            'x': [str(v) for v in x_values],
            'y': [str(v) for v in y_values],
            'colorscale': 'RdYlGn',
            'colorbar': {'title': metric_labels.get(metric, metric)},
            'hovertemplate': f'{x_param}: %{{x}}<br>{y_param}: %{{y}}<br>{metric}: %{{z:.4f}}<extra></extra>'
        }],
        'layout': {
            'title': title or f'参数扫描热力图 - {metric_labels.get(metric, metric)}',
            'xaxis': {'title': x_param, 'type': 'category'},
            'yaxis': {'title': y_param, 'type': 'category'},
            'height': 500
        }
    }

    return fig_data


def create_plotly_surface(
    results_df: pd.DataFrame,
    x_param: str,
    y_param: str,
    metric: str = 'sharpe_ratio',
    title: str = None
) -> dict:
    """
    创建Plotly 3D曲面图配置

    Args:
        results_df: 参数扫描结果DataFrame
        x_param: X轴参数名
        y_param: Y轴参数名
        metric: 指标名
        title: 图表标题

    Returns:
        Plotly figure dict
    """
    z_matrix, x_values, y_values = create_heatmap_data(
        results_df, x_param, y_param, metric
    )

    metric_labels = {
        'sharpe_ratio': 'Sharpe Ratio',
        'total_return': '总收益率',
        'annual_return': '年化收益率',
        'max_drawdown': '最大回撤',
        'win_rate': '胜率',
        'profit_factor': '盈亏比'
    }

    fig_data = {
        'data': [{
            'type': 'surface',
            'z': z_matrix.tolist(),
            'x': [float(v) if isinstance(v, (int, float)) else i for i, v in enumerate(x_values)],
            'y': [float(v) if isinstance(v, (int, float)) else i for i, v in enumerate(y_values)],
            'colorscale': 'Viridis',
            'colorbar': {'title': metric_labels.get(metric, metric)}
        }],
        'layout': {
            'title': title or f'参数扫描3D曲面 - {metric_labels.get(metric, metric)}',
            'scene': {
                'xaxis': {'title': x_param},
                'yaxis': {'title': y_param},
                'zaxis': {'title': metric_labels.get(metric, metric)}
            },
            'height': 600
        }
    }

    return fig_data


def create_param_comparison_chart(
    results_df: pd.DataFrame,
    param: str,
    metrics: List[str] = None
) -> dict:
    """
    创建单参数对比图

    Args:
        results_df: 参数扫描结果DataFrame
        param: 参数名
        metrics: 指标列表

    Returns:
        Plotly figure dict
    """
    if metrics is None:
        metrics = ['sharpe_ratio', 'total_return', 'max_drawdown']

    # 按参数分组平均
    grouped = results_df.groupby(param)[metrics].mean().reset_index()

    traces = []
    for metric in metrics:
        traces.append({
            'type': 'scatter',
            'x': grouped[param].tolist(),
            'y': grouped[metric].tolist(),
            'name': metric,
            'mode': 'lines+markers'
        })

    fig_data = {
        'data': traces,
        'layout': {
            'title': f'参数 {param} 对各指标的影响',
            'xaxis': {'title': param},
            'yaxis': {'title': '指标值'},
            'height': 400,
            'showlegend': True
        }
    }

    return fig_data


def get_top_params(
    results_df: pd.DataFrame,
    metric: str = 'sharpe_ratio',
    n: int = 10,
    ascending: bool = False
) -> pd.DataFrame:
    """
    获取最优参数组合

    Args:
        results_df: 参数扫描结果DataFrame
        metric: 排序指标
        n: 返回数量
        ascending: 升序排列

    Returns:
        Top N参数组合
    """
    sorted_df = results_df.sort_values(metric, ascending=ascending)
    return sorted_df.head(n)


def generate_scan_report(
    results_df: pd.DataFrame,
    param_names: List[str],
    target_metric: str = 'sharpe_ratio'
) -> str:
    """
    生成参数扫描报告

    Args:
        results_df: 参数扫描结果DataFrame
        param_names: 参数名列表
        target_metric: 目标指标

    Returns:
        Markdown格式报告
    """
    report = []
    report.append("# 参数扫描报告\n")

    # 基本统计
    report.append("## 扫描概况\n")
    report.append(f"- 总组合数: {len(results_df)}")
    report.append(f"- 成功率: {(results_df['success'].sum() / len(results_df) * 100):.1f}%")
    report.append(f"- 目标指标: {target_metric}\n")

    # 最优参数
    report.append("## 最优参数组合 (Top 5)\n")
    top5 = get_top_params(results_df, target_metric, n=5)

    report.append("| 排名 | " + " | ".join(param_names) + f" | {target_metric} | 收益率 | 回撤 |")
    report.append("|" + "|".join(["---"] * (len(param_names) + 4)) + "|")

    for i, (_, row) in enumerate(top5.iterrows(), 1):
        param_vals = " | ".join([str(row.get(p, 'N/A')) for p in param_names])
        report.append(
            f"| {i} | {param_vals} | "
            f"{row.get(target_metric, 0):.3f} | "
            f"{row.get('total_return', 0):.2%} | "
            f"{row.get('max_drawdown', 0):.2%} |"
        )

    report.append("")

    # 参数敏感度分析
    report.append("## 参数敏感度分析\n")

    for param in param_names:
        if param in results_df.columns:
            grouped = results_df.groupby(param)[target_metric].agg(['mean', 'std', 'min', 'max'])
            best_val = grouped['mean'].idxmax()
            report.append(f"### {param}")
            report.append(f"- 最优值: {best_val}")
            report.append(f"- 均值范围: {grouped['mean'].min():.3f} ~ {grouped['mean'].max():.3f}")
            report.append(f"- 标准差: {grouped['std'].mean():.3f}\n")

    return "\n".join(report)


class ParamScanVisualizer:
    """
    参数扫描可视化器

    提供Streamlit集成的可视化组件
    """

    def __init__(self, results_df: pd.DataFrame, param_names: List[str]):
        """
        初始化

        Args:
            results_df: 参数扫描结果DataFrame
            param_names: 参数名列表
        """
        self.results_df = results_df
        self.param_names = param_names

    def render_heatmap(self, x_param: str, y_param: str, metric: str = 'sharpe_ratio'):
        """渲染热力图（返回Plotly figure）"""
        return create_plotly_heatmap(self.results_df, x_param, y_param, metric)

    def render_surface(self, x_param: str, y_param: str, metric: str = 'sharpe_ratio'):
        """渲染3D曲面（返回Plotly figure）"""
        return create_plotly_surface(self.results_df, x_param, y_param, metric)

    def render_comparison(self, param: str, metrics: List[str] = None):
        """渲染参数对比图（返回Plotly figure）"""
        return create_param_comparison_chart(self.results_df, param, metrics)

    def get_best_params(self, metric: str = 'sharpe_ratio', n: int = 10) -> pd.DataFrame:
        """获取最优参数"""
        return get_top_params(self.results_df, metric, n)

    def generate_report(self, target_metric: str = 'sharpe_ratio') -> str:
        """生成报告"""
        return generate_scan_report(self.results_df, self.param_names, target_metric)
