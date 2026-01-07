# coding=utf-8
"""
回测引擎
支持策略回测和绩效分析
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.base import BaseStrategy, Signal
from utils.data_loader import DataLoader, get_data_loader

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """交易记录"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    direction: int  # 1=多, -1=空
    entry_price: float
    exit_price: float
    volume: int
    pnl: float
    pnl_pct: float
    holding_bars: int
    tag: str = ""


@dataclass
class BacktestResult:
    """回测结果"""
    # 基本信息
    strategy_name: str
    symbol: str
    period: str
    start_date: datetime
    end_date: datetime
    initial_capital: float

    # 收益指标
    final_capital: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0

    # 风险指标
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0

    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    max_profit: float = 0.0
    max_loss: float = 0.0
    avg_holding_bars: float = 0.0

    # 详细数据
    trades: List[TradeRecord] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_returns: pd.Series = field(default_factory=pd.Series)


class BacktestEngine:
    """
    回测引擎

    功能:
    1. 策略回测
    2. 绩效分析
    3. 多品种/多周期回测
    4. 参数优化
    """

    def __init__(self, data_loader: DataLoader = None):
        """
        初始化回测引擎

        Args:
            data_loader: 数据加载器
        """
        self.data_loader = data_loader or get_data_loader()

        # 品种配置（合约乘数和手续费）
        self.instrument_configs = {
            'RB': {'multiplier': 10, 'margin_rate': 0.10, 'commission': 0.0001},
            'I': {'multiplier': 100, 'margin_rate': 0.12, 'commission': 0.0001},
            'AU': {'multiplier': 1000, 'margin_rate': 0.08, 'commission': 10},
            'CU': {'multiplier': 5, 'margin_rate': 0.10, 'commission': 0.00005},
            'AL': {'multiplier': 5, 'margin_rate': 0.10, 'commission': 3},
            'NI': {'multiplier': 1, 'margin_rate': 0.12, 'commission': 6},
            'TA': {'multiplier': 5, 'margin_rate': 0.08, 'commission': 3},
            'MA': {'multiplier': 10, 'margin_rate': 0.08, 'commission': 2},
            'PP': {'multiplier': 5, 'margin_rate': 0.08, 'commission': 0.0001},
            'M': {'multiplier': 10, 'margin_rate': 0.08, 'commission': 1.5},
        }

    def set_instrument_config(self, symbol: str, config: dict):
        """设置品种配置"""
        self.instrument_configs[symbol] = config

    def run(
        self,
        strategy: BaseStrategy,
        symbol: str,
        period: str = '1d',
        start_date: datetime = None,
        end_date: datetime = None,
        initial_capital: float = 100000.0,
        volume: int = 1,
        commission_rate: float = None
    ) -> BacktestResult:
        """
        运行回测

        Args:
            strategy: 策略实例
            symbol: 交易品种
            period: K线周期
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            volume: 每笔交易数量
            commission_rate: 手续费率（覆盖默认值）

        Returns:
            BacktestResult
        """
        logger.info(f"开始回测: {strategy.name} - {symbol} - {period}")

        # 加载数据
        df = self.data_loader.load_bars(symbol, period, start_date, end_date)

        if df.empty:
            logger.error(f"无法加载数据: {symbol} {period}")
            return BacktestResult(
                strategy_name=strategy.name,
                symbol=symbol,
                period=period,
                start_date=start_date or datetime.now(),
                end_date=end_date or datetime.now(),
                initial_capital=initial_capital
            )

        # 获取品种配置
        product = ''.join([c for c in symbol if c.isalpha()]).upper()
        config = self.instrument_configs.get(product, {
            'multiplier': 10,
            'margin_rate': 0.10,
            'commission': 0.0001
        })

        multiplier = config['multiplier']
        if commission_rate is None:
            commission_rate = config['commission']

        # 计算指标
        df = strategy.calculate_indicators(df)

        # 重置策略状态
        strategy.reset()

        # 回测变量
        capital = initial_capital
        position = 0
        entry_price = 0.0
        entry_time = None
        entry_idx = 0

        trades: List[TradeRecord] = []
        equity_history = []

        # 逐K线回测
        for idx in range(strategy.warmup_num, len(df)):
            current_bar = df.iloc[idx]
            current_time = current_bar['time'] if 'time' in current_bar else datetime.now()
            current_price = current_bar['close']

            # 记录权益
            if position != 0:
                if position > 0:
                    unrealized = (current_price - entry_price) * volume * multiplier
                else:
                    unrealized = (entry_price - current_price) * volume * multiplier
                equity = capital + unrealized
            else:
                equity = capital

            equity_history.append({
                'time': current_time,
                'equity': equity,
                'position': position
            })

            # 获取信号
            signal = strategy.on_bar(idx, df, capital)

            if signal is None:
                continue

            # 处理信号
            if signal.action in ['buy', 'sell'] and position == 0:
                # 开仓
                position = 1 if signal.action == 'buy' else -1
                entry_price = signal.price
                entry_time = current_time
                entry_idx = idx

                # 扣除手续费
                if commission_rate < 1:
                    commission = entry_price * volume * multiplier * commission_rate
                else:
                    commission = commission_rate * volume

                capital -= commission

                # 更新策略状态
                strategy.position = position
                strategy.entry_price = entry_price
                strategy.entry_time = entry_time

            elif signal.action in ['close', 'close_long', 'close_short'] and position != 0:
                # 平仓
                exit_price = signal.price

                # 计算盈亏
                if position > 0:
                    pnl = (exit_price - entry_price) * volume * multiplier
                else:
                    pnl = (entry_price - exit_price) * volume * multiplier

                # 扣除手续费
                if commission_rate < 1:
                    commission = exit_price * volume * multiplier * commission_rate
                else:
                    commission = commission_rate * volume

                pnl -= commission

                # 记录交易
                trade = TradeRecord(
                    entry_time=entry_time,
                    exit_time=current_time,
                    symbol=symbol,
                    direction=position,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    volume=volume,
                    pnl=pnl,
                    pnl_pct=pnl / (entry_price * volume * multiplier),
                    holding_bars=idx - entry_idx,
                    tag=signal.tag
                )
                trades.append(trade)

                capital += pnl

                # 更新策略状态
                position = 0
                entry_price = 0
                entry_time = None

                strategy.position = 0
                strategy.entry_price = 0
                strategy.entry_time = None

            # 处理反手信号
            if signal.action == 'buy' and position == -1:
                # 先平空
                exit_price = signal.price
                pnl = (entry_price - exit_price) * volume * multiplier

                if commission_rate < 1:
                    commission = exit_price * volume * multiplier * commission_rate
                else:
                    commission = commission_rate * volume
                pnl -= commission

                trade = TradeRecord(
                    entry_time=entry_time,
                    exit_time=current_time,
                    symbol=symbol,
                    direction=-1,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    volume=volume,
                    pnl=pnl,
                    pnl_pct=pnl / (entry_price * volume * multiplier),
                    holding_bars=idx - entry_idx,
                    tag="反手平空"
                )
                trades.append(trade)
                capital += pnl

                # 再开多
                position = 1
                entry_price = signal.price
                entry_time = current_time
                entry_idx = idx

                if commission_rate < 1:
                    commission = entry_price * volume * multiplier * commission_rate
                else:
                    commission = commission_rate * volume
                capital -= commission

                strategy.position = 1
                strategy.entry_price = entry_price

            elif signal.action == 'sell' and position == 1:
                # 先平多
                exit_price = signal.price
                pnl = (exit_price - entry_price) * volume * multiplier

                if commission_rate < 1:
                    commission = exit_price * volume * multiplier * commission_rate
                else:
                    commission = commission_rate * volume
                pnl -= commission

                trade = TradeRecord(
                    entry_time=entry_time,
                    exit_time=current_time,
                    symbol=symbol,
                    direction=1,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    volume=volume,
                    pnl=pnl,
                    pnl_pct=pnl / (entry_price * volume * multiplier),
                    holding_bars=idx - entry_idx,
                    tag="反手平多"
                )
                trades.append(trade)
                capital += pnl

                # 再开空
                position = -1
                entry_price = signal.price
                entry_time = current_time
                entry_idx = idx

                if commission_rate < 1:
                    commission = entry_price * volume * multiplier * commission_rate
                else:
                    commission = commission_rate * volume
                capital -= commission

                strategy.position = -1
                strategy.entry_price = entry_price

        # 平掉剩余持仓
        if position != 0:
            last_price = df.iloc[-1]['close']
            last_time = df.iloc[-1]['time'] if 'time' in df.iloc[-1] else datetime.now()

            if position > 0:
                pnl = (last_price - entry_price) * volume * multiplier
            else:
                pnl = (entry_price - last_price) * volume * multiplier

            if commission_rate < 1:
                commission = last_price * volume * multiplier * commission_rate
            else:
                commission = commission_rate * volume
            pnl -= commission

            trade = TradeRecord(
                entry_time=entry_time,
                exit_time=last_time,
                symbol=symbol,
                direction=position,
                entry_price=entry_price,
                exit_price=last_price,
                volume=volume,
                pnl=pnl,
                pnl_pct=pnl / (entry_price * volume * multiplier),
                holding_bars=len(df) - 1 - entry_idx,
                tag="回测结束平仓"
            )
            trades.append(trade)
            capital += pnl

        # 计算绩效指标
        result = self._calculate_metrics(
            strategy_name=strategy.name,
            symbol=symbol,
            period=period,
            initial_capital=initial_capital,
            final_capital=capital,
            trades=trades,
            equity_history=equity_history,
            start_date=df.iloc[0]['time'] if 'time' in df.columns else start_date,
            end_date=df.iloc[-1]['time'] if 'time' in df.columns else end_date
        )

        logger.info(f"回测完成: 总收益={result.total_return:.2%}, "
                    f"交易次数={result.total_trades}, 胜率={result.win_rate:.2%}")

        return result

    def _calculate_metrics(
        self,
        strategy_name: str,
        symbol: str,
        period: str,
        initial_capital: float,
        final_capital: float,
        trades: List[TradeRecord],
        equity_history: List[dict],
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """计算绩效指标"""
        result = BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            period=period,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_capital=final_capital,
            trades=trades
        )

        # 基本收益
        result.total_return = (final_capital - initial_capital) / initial_capital

        # 年化收益（按250交易日计算）
        if start_date and end_date:
            days = (end_date - start_date).days
            if days > 0:
                years = days / 365
                result.annual_return = (1 + result.total_return) ** (1 / years) - 1 if years > 0 else 0

        # 权益曲线
        if equity_history:
            result.equity_curve = pd.DataFrame(equity_history)

            # 计算最大回撤
            equity_series = result.equity_curve['equity']
            rolling_max = equity_series.expanding().max()
            drawdown = (rolling_max - equity_series) / rolling_max

            result.max_drawdown_pct = drawdown.max()
            result.max_drawdown = (rolling_max - equity_series).max()

            # 日收益率
            if 'time' in result.equity_curve.columns:
                result.equity_curve.set_index('time', inplace=True)
                result.daily_returns = result.equity_curve['equity'].pct_change().dropna()

                # 波动率
                result.volatility = result.daily_returns.std() * np.sqrt(250)

                # Sharpe Ratio (假设无风险利率3%)
                rf_daily = 0.03 / 250
                excess_return = result.daily_returns - rf_daily
                if result.daily_returns.std() > 0:
                    result.sharpe_ratio = excess_return.mean() / result.daily_returns.std() * np.sqrt(250)

                # Sortino Ratio
                negative_returns = result.daily_returns[result.daily_returns < 0]
                if len(negative_returns) > 0 and negative_returns.std() > 0:
                    result.sortino_ratio = excess_return.mean() / negative_returns.std() * np.sqrt(250)

        # Calmar Ratio
        if result.max_drawdown_pct > 0:
            result.calmar_ratio = result.annual_return / result.max_drawdown_pct

        # 交易统计
        result.total_trades = len(trades)

        if trades:
            profits = [t.pnl for t in trades if t.pnl > 0]
            losses = [t.pnl for t in trades if t.pnl <= 0]

            result.winning_trades = len(profits)
            result.losing_trades = len(losses)
            result.win_rate = result.winning_trades / result.total_trades if result.total_trades > 0 else 0

            result.avg_profit = np.mean(profits) if profits else 0
            result.avg_loss = np.mean(losses) if losses else 0
            result.max_profit = max(profits) if profits else 0
            result.max_loss = min(losses) if losses else 0

            # Profit Factor
            total_profit = sum(profits) if profits else 0
            total_loss = abs(sum(losses)) if losses else 0
            result.profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            # 平均持仓时间
            result.avg_holding_bars = np.mean([t.holding_bars for t in trades])

        return result

    def run_multi_symbol(
        self,
        strategy: BaseStrategy,
        symbols: List[str],
        period: str = '1d',
        start_date: datetime = None,
        end_date: datetime = None,
        initial_capital: float = 100000.0,
        volume: int = 1
    ) -> Dict[str, BacktestResult]:
        """
        多品种回测

        Args:
            strategy: 策略实例
            symbols: 品种列表
            period: K线周期
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金（每个品种）
            volume: 每笔交易数量

        Returns:
            {symbol: BacktestResult}
        """
        results = {}

        for symbol in symbols:
            # 为每个品种创建策略副本
            strategy_copy = strategy.__class__(strategy.params.copy())
            result = self.run(
                strategy_copy, symbol, period,
                start_date, end_date, initial_capital, volume
            )
            results[symbol] = result

        return results

    def optimize(
        self,
        strategy_class: type,
        symbol: str,
        param_grid: Dict[str, List[Any]],
        period: str = '1d',
        start_date: datetime = None,
        end_date: datetime = None,
        initial_capital: float = 100000.0,
        optimize_target: str = 'sharpe_ratio'
    ) -> Tuple[Dict, BacktestResult]:
        """
        参数优化

        Args:
            strategy_class: 策略类
            symbol: 交易品种
            param_grid: 参数网格 {param_name: [values]}
            period: K线周期
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            optimize_target: 优化目标

        Returns:
            (最优参数, 回测结果)
        """
        from itertools import product

        # 生成参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        logger.info(f"参数优化: {len(combinations)} 种组合")

        best_params = None
        best_result = None
        best_score = float('-inf')

        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))

            try:
                strategy = strategy_class(params)
                result = self.run(
                    strategy, symbol, period,
                    start_date, end_date, initial_capital
                )

                # 获取优化目标值
                score = getattr(result, optimize_target, 0)

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result

                if (i + 1) % 10 == 0:
                    logger.info(f"进度: {i + 1}/{len(combinations)}, "
                               f"当前最优: {optimize_target}={best_score:.4f}")

            except Exception as e:
                logger.warning(f"参数组合 {params} 回测失败: {e}")

        logger.info(f"优化完成: 最优参数={best_params}, {optimize_target}={best_score:.4f}")
        return best_params, best_result


def generate_report(result: BacktestResult) -> str:
    """生成回测报告"""
    report = f"""
# 回测报告

## 基本信息
- 策略名称: {result.strategy_name}
- 交易品种: {result.symbol}
- K线周期: {result.period}
- 回测时间: {result.start_date} 至 {result.end_date}
- 初始资金: {result.initial_capital:,.2f}

## 收益指标
| 指标 | 数值 |
|------|------|
| 最终资金 | {result.final_capital:,.2f} |
| 总收益率 | {result.total_return:.2%} |
| 年化收益率 | {result.annual_return:.2%} |
| 最大回撤 | {result.max_drawdown_pct:.2%} |

## 风险指标
| 指标 | 数值 |
|------|------|
| 夏普比率 | {result.sharpe_ratio:.2f} |
| 索提诺比率 | {result.sortino_ratio:.2f} |
| 卡尔玛比率 | {result.calmar_ratio:.2f} |
| 波动率 | {result.volatility:.2%} |

## 交易统计
| 指标 | 数值 |
|------|------|
| 总交易次数 | {result.total_trades} |
| 盈利次数 | {result.winning_trades} |
| 亏损次数 | {result.losing_trades} |
| 胜率 | {result.win_rate:.2%} |
| 盈亏比 | {result.profit_factor:.2f} |
| 平均盈利 | {result.avg_profit:,.2f} |
| 平均亏损 | {result.avg_loss:,.2f} |
| 最大盈利 | {result.max_profit:,.2f} |
| 最大亏损 | {result.max_loss:,.2f} |
| 平均持仓 | {result.avg_holding_bars:.1f} 根K线 |
"""
    return report
