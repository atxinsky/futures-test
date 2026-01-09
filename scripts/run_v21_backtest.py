# coding=utf-8
"""
BigBrother V21 完整回测脚本
使用优化后的参数，生成详细回测报告
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# 配置
START_DATE = "2021-01-01"
END_DATE = "2025-01-08"
INITIAL_CAPITAL = 1000000

ETF_POOL = [
    "513100.SH",  # 纳指ETF
    "513050.SH",  # 中概互联
    "512480.SH",  # 半导体ETF
    "515030.SH",  # 新能车ETF
    "518880.SH",  # 黄金ETF
    "512890.SH",  # 红利低波
    "588000.SH",  # 科创50
    "516010.SH",  # 游戏动漫
]

ETF_NAMES = {
    "513100.SH": "纳指ETF",
    "513050.SH": "中概互联",
    "512480.SH": "半导体ETF",
    "515030.SH": "新能车ETF",
    "518880.SH": "黄金ETF",
    "512890.SH": "红利低波",
    "588000.SH": "科创50",
    "516010.SH": "游戏动漫",
}


def load_data():
    """加载ETF数据"""
    from core.etf_data_service import get_etf_data_service

    print("加载ETF数据...")
    ds = get_etf_data_service()

    data = {}
    for code in ETF_POOL:
        df = ds.get_data_with_indicators(code, START_DATE, END_DATE)
        if len(df) > 0:
            data[code] = df
            print(f"  {code} ({ETF_NAMES[code]}): {len(df)}行")

    return data


def run_backtest(data):
    """运行回测"""
    from core.etf_backtest_engine import ETFBacktestEngine
    from strategies.etf_bigbrother_v17_v21 import ETFBigBrotherV21

    print("\n运行回测...")
    print(f"策略: BigBrother V21 (优化参数)")
    print(f"时间: {START_DATE} ~ {END_DATE}")
    print(f"初始资金: {INITIAL_CAPITAL:,}")

    # 创建策略 (使用默认的优化参数)
    strategy = ETFBigBrotherV21(pool=ETF_POOL)

    print(f"\n策略参数:")
    for k, v in strategy.PARAMS.items():
        print(f"  {v['name']}: {v['default']}")

    # 创建引擎
    engine = ETFBacktestEngine(
        initial_capital=INITIAL_CAPITAL,
        commission_rate=0.0001,
        slippage=0.0001,
        benchmark="510300.SH"
    )

    engine.set_strategy(strategy.initialize, strategy.handle_data)

    # 运行回测
    result = engine.run(
        data=data,
        start_date=START_DATE,
        end_date=END_DATE
    )

    return result, strategy


def analyze_trades(result):
    """分析交易记录"""
    if not result.trades:
        return {}

    trades_by_etf = {}
    for t in result.trades:
        if t.code not in trades_by_etf:
            trades_by_etf[t.code] = {'buys': [], 'sells': [], 'pnl': 0, 'win': 0, 'lose': 0}

        if t.direction == "BUY":
            trades_by_etf[t.code]['buys'].append(t)
        else:
            trades_by_etf[t.code]['sells'].append(t)
            trades_by_etf[t.code]['pnl'] += t.pnl
            if t.pnl > 0:
                trades_by_etf[t.code]['win'] += 1
            else:
                trades_by_etf[t.code]['lose'] += 1

    return trades_by_etf


def analyze_yearly(result):
    """年度收益分析"""
    if result.equity_curve is None:
        return {}

    df = result.equity_curve.reset_index()
    if 'date' not in df.columns:
        return {}

    df['year'] = pd.to_datetime(df['date']).dt.year

    yearly = {}
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year]
        if len(year_df) > 1:
            start_val = year_df.iloc[0]['total_value']
            end_val = year_df.iloc[-1]['total_value']
            ret = (end_val - start_val) / start_val

            # 计算年度最大回撤
            rolling_max = year_df['total_value'].cummax()
            drawdown = (year_df['total_value'] - rolling_max) / rolling_max
            max_dd = drawdown.min()

            yearly[year] = {
                'return': ret,
                'max_drawdown': abs(max_dd),
                'start_value': start_val,
                'end_value': end_val
            }

    return yearly


def generate_report(result, strategy, trades_by_etf, yearly):
    """生成Markdown报告"""
    report = f"""# BigBrother V21 回测报告

**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 策略概述

| 项目 | 内容 |
|------|------|
| 策略名称 | BigBrother V21 (防跳空版) |
| 策略类型 | Donchian通道趋势突破 |
| 核心逻辑 | 收盘价突破N日高点买入，跌破M日低点卖出 |
| 特色功能 | VWAP成交价、高开过滤 |

## 优化后参数

| 参数 | 值 | 说明 |
|------|-----|------|
| risk_per_trade | **0.8%** | 单笔风险敞口 |
| max_position | **20%** | 单标的最大仓位 |
| donchian_high_period | **28** | 突破周期 |
| donchian_low_period | **14** | 跌破周期 |
| gap_up_limit | **3%** | 高开超过3%放弃追高 |

## 回测配置

| 配置 | 值 |
|------|-----|
| 回测区间 | {START_DATE} ~ {END_DATE} |
| 初始资金 | ¥{INITIAL_CAPITAL:,} |
| 手续费率 | 0.01% |
| 滑点 | 0.01% |

### ETF标的池 (8只)
"""

    for code in ETF_POOL:
        report += f"- {code} ({ETF_NAMES[code]})\n"

    report += f"""
---

## 绩效总览

| 指标 | 数值 |
|------|------|
| **累计收益** | **{result.total_return*100:.2f}%** |
| **年化收益** | **{result.annual_return*100:.2f}%** |
| **最大回撤** | {result.max_drawdown*100:.2f}% |
| **夏普比率** | **{result.sharpe_ratio:.2f}** |
| **卡玛比率** | {result.calmar_ratio:.2f} |
| 胜率 | {result.win_rate*100:.1f}% |
| 盈亏比 | {result.profit_loss_ratio:.2f} |
| 总交易次数 | {result.total_trades} |
| 盈利交易 | {result.win_trades} |
| 亏损交易 | {result.lose_trades} |

---

## 年度表现

| 年份 | 收益率 | 最大回撤 | 评价 |
|------|--------|----------|------|
"""

    for year, stats in sorted(yearly.items()):
        ret = stats['return']
        dd = stats['max_drawdown']
        if ret > 0.20:
            rating = "优秀"
        elif ret > 0.10:
            rating = "良好"
        elif ret > 0:
            rating = "一般"
        else:
            rating = "亏损"

        report += f"| {year} | {ret*100:+.1f}% | {dd*100:.1f}% | {rating} |\n"

    report += f"""
---

## 分标的表现

| ETF | 名称 | 交易次数 | 盈利 | 亏损 | 胜率 | 总盈亏 |
|-----|------|----------|------|------|------|--------|
"""

    for code in ETF_POOL:
        if code in trades_by_etf:
            stats = trades_by_etf[code]
            total = stats['win'] + stats['lose']
            win_rate = stats['win'] / total * 100 if total > 0 else 0
            report += f"| {code} | {ETF_NAMES[code]} | {total} | {stats['win']} | {stats['lose']} | {win_rate:.0f}% | ¥{stats['pnl']:+,.0f} |\n"
        else:
            report += f"| {code} | {ETF_NAMES[code]} | 0 | 0 | 0 | - | ¥0 |\n"

    report += f"""
---

## 风险分析

### 回撤分析
- 最大回撤: {result.max_drawdown*100:.2f}%
- 最大回撤持续: {result.max_drawdown_duration}天
- 波动率: {result.volatility*100:.2f}%

### 策略特点
1. **趋势跟踪**: Donchian通道是海龟交易法核心，适合捕捉中长期趋势
2. **防跳空设计**: 高开超过3%不追，避免追高被套
3. **VWAP成交**: 使用成交均价，更贴近实际交易成本
4. **多标的分散**: 8只ETF覆盖美股、中概、科技、商品、防守

---

## 结论

BigBrother V21策略在{START_DATE[:4]}-{END_DATE[:4]}年回测中：

- **累计收益{result.total_return*100:.1f}%**，年化{result.annual_return*100:.1f}%
- 夏普比率{result.sharpe_ratio:.2f}，风险调整后收益可接受
- 最大回撤{result.max_drawdown*100:.1f}%，在可控范围内

**建议**: 先用模拟盘跑1-2个月验证，确认信号执行无误后再考虑实盘。

---

*报告由Claude Code自动生成*
"""

    return report


def main():
    print("="*60)
    print("BigBrother V21 完整回测")
    print("="*60)

    # 加载数据
    data = load_data()
    if not data:
        print("数据加载失败")
        return

    # 运行回测
    result, strategy = run_backtest(data)

    print("\n" + "="*60)
    print("回测完成!")
    print("="*60)

    print(f"\n累计收益: {result.total_return*100:.2f}%")
    print(f"年化收益: {result.annual_return*100:.2f}%")
    print(f"最大回撤: {result.max_drawdown*100:.2f}%")
    print(f"夏普比率: {result.sharpe_ratio:.2f}")
    print(f"总交易: {result.total_trades}次")

    # 分析
    trades_by_etf = analyze_trades(result)
    yearly = analyze_yearly(result)

    # 生成报告
    report = generate_report(result, strategy, trades_by_etf, yearly)

    # 保存报告
    report_path = os.path.join(os.path.dirname(__file__), "v21_backtest_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n报告已保存: {report_path}")

    return report


if __name__ == "__main__":
    main()
