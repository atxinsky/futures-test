# coding=utf-8
"""
回测引擎
支持多品种、多策略回测
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Type
from dataclasses import dataclass, field
from config import get_instrument, calculate_commission, calculate_pnl, INSTRUMENTS

# 使用统一的回测数据模型
from models.backtest_models import BacktestTrade, BacktestResult

# 兼容性别名（保持旧代码可用）
Trade = BacktestTrade


def run_backtest_with_strategy(
    df: pd.DataFrame,
    symbol: str,
    strategy,  # BaseStrategy实例
    initial_capital: float = 1000000,
) -> BacktestResult:
    """
    使用策略类运行回测（支持多空双向交易）

    df: 包含OHLCV数据的DataFrame
    symbol: 品种代码
    strategy: 策略实例
    initial_capital: 初始资金
    """
    inst = get_instrument(symbol)
    if not inst:
        raise ValueError(f"未知品种: {symbol}")

    multiplier = inst['multiplier']
    margin_rate = inst['margin_rate']

    # 计算指标
    df = strategy.calculate_indicators(df)
    strategy.reset()

    # 交易状态
    trades = []
    trade_id = 0
    shares = 0
    last_direction = 0  # 记录上一次开仓方向

    capital = initial_capital
    equity_curve = []
    max_equity = capital
    max_drawdown_pct = 0
    max_drawdown_val = 0

    warmup = strategy.warmup_num

    for i in range(warmup, len(df)):
        curr_close = df['close'].iloc[i]
        curr_high = df['high'].iloc[i]
        current_time = df['time'].iloc[i]

        # 计算当前权益
        if strategy.position != 0:
            unrealized_pnl = (curr_close - strategy.entry_price) * shares * multiplier * strategy.position
            current_equity = capital + unrealized_pnl
        else:
            current_equity = capital

        # 更新最大回撤
        if current_equity > max_equity:
            max_equity = current_equity

        dd_pct = (max_equity - current_equity) / max_equity * 100
        dd_val = max_equity - current_equity
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct
            max_drawdown_val = dd_val

        equity_curve.append({
            'time': current_time,
            'equity': current_equity,
            'drawdown_pct': dd_pct
        })

        # 调用策略逻辑
        signal = strategy.on_bar(i, df, capital)

        if signal is None:
            continue

        # 处理平仓信号（多空通用）
        if signal.action == "close" and strategy.position == 0:
            # 策略已重置position，计算盈亏（方向由last_direction决定）
            pnl = (signal.price - strategy.entry_price) * shares * multiplier * last_direction
            comm = calculate_commission(symbol, signal.price, shares) * 2
            net_pnl = pnl - comm
            capital += net_pnl

            # 计算持仓天数
            if strategy.entry_time:
                holding_days = (current_time - strategy.entry_time).days
            else:
                holding_days = 0

            pnl_pct = (signal.price - strategy.entry_price) / strategy.entry_price * 100 * last_direction if strategy.entry_price > 0 else 0

            trade = Trade(
                trade_id=trade_id,
                symbol=symbol,
                direction=last_direction,
                entry_time=strategy.entry_time if strategy.entry_time else current_time,
                entry_price=strategy.entry_price,
                entry_tag="long" if last_direction == 1 else "short",
                volume=shares,
                exit_time=current_time,
                exit_price=signal.price,
                exit_tag=signal.tag,
                pnl=net_pnl,
                pnl_pct=pnl_pct,
                commission=comm,
                holding_days=holding_days,
                capital_after=capital
            )
            trades.append(trade)
            trade_id += 1
            shares = 0
            last_direction = 0
            # 重置entry_price用于下次记录
            strategy.entry_price = 0
            strategy.entry_time = None

        # 处理开多信号
        elif signal.action == "buy" and strategy.position == 1:
            # 计算合理仓位（考虑品种乘数）
            stake_amt = capital * strategy.params.get('capital_rate', 1.0)
            risk_per_trade = stake_amt * strategy.params.get('risk_rate', 0.02)
            # 判断 stop_loss 是价格还是距离
            if signal.stop_loss > 0:
                if signal.stop_loss > signal.price * 0.5:
                    stop_dist = abs(signal.price - signal.stop_loss)
                else:
                    stop_dist = signal.stop_loss
            else:
                stop_dist = signal.price * 0.02
            # 根据风险计算手数
            if stop_dist > 0:
                shares = max(1, int(risk_per_trade / (stop_dist * multiplier)))
            else:
                shares = 1

            # 保证金检查
            required_margin = signal.price * multiplier * shares * margin_rate
            max_margin = capital * 0.8
            if required_margin > max_margin:
                shares = max(1, int(max_margin / (signal.price * multiplier * margin_rate)))

            strategy.entry_time = current_time
            last_direction = 1  # 记录多头方向

        # 处理开空信号
        elif signal.action == "sell" and strategy.position == -1:
            # 计算合理仓位（考虑品种乘数）
            stake_amt = capital * strategy.params.get('capital_rate', 1.0)
            risk_per_trade = stake_amt * strategy.params.get('risk_rate', 0.02)
            # 判断 stop_loss 是价格还是距离
            if signal.stop_loss > 0:
                if signal.stop_loss > signal.price * 0.5:
                    stop_dist = abs(signal.stop_loss - signal.price)
                else:
                    stop_dist = signal.stop_loss
            else:
                stop_dist = signal.price * 0.02
            # 根据风险计算手数
            if stop_dist > 0:
                shares = max(1, int(risk_per_trade / (stop_dist * multiplier)))
            else:
                shares = 1

            # 保证金检查
            required_margin = signal.price * multiplier * shares * margin_rate
            max_margin = capital * 0.8
            if required_margin > max_margin:
                shares = max(1, int(max_margin / (signal.price * multiplier * margin_rate)))

            strategy.entry_time = current_time
            last_direction = -1  # 记录空头方向

    # 平掉最后持仓
    if strategy.position != 0 and shares > 0:
        curr_close = df['close'].iloc[-1]
        current_time = df['time'].iloc[-1]
        pnl = (curr_close - strategy.entry_price) * shares * multiplier * strategy.position
        comm = calculate_commission(symbol, curr_close, shares) * 2
        net_pnl = pnl - comm
        capital += net_pnl

        holding_days = (current_time - strategy.entry_time).days if strategy.entry_time else 0
        pnl_pct = (curr_close - strategy.entry_price) / strategy.entry_price * 100 * strategy.position if strategy.entry_price > 0 else 0

        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            direction=strategy.position,
            entry_time=strategy.entry_time if strategy.entry_time else current_time,
            entry_price=strategy.entry_price,
            entry_tag="long" if strategy.position == 1 else "short",
            volume=shares,
            exit_time=current_time,
            exit_price=curr_close,
            exit_tag="backtest_end",
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=comm,
            holding_days=holding_days,
            capital_after=capital
        )
        trades.append(trade)

    # 计算统计指标
    return _calculate_statistics(
        df, symbol, strategy.name, initial_capital, capital,
        trades, equity_curve, max_drawdown_pct, max_drawdown_val
    )


def _calculate_statistics(
    df: pd.DataFrame,
    symbol: str,
    strategy_name: str,
    initial_capital: float,
    final_capital: float,
    trades: List[Trade],
    equity_curve: List[dict],
    max_drawdown_pct: float,
    max_drawdown_val: float
) -> BacktestResult:
    """计算回测统计指标"""

    df_equity = pd.DataFrame(equity_curve)
    if len(df_equity) == 0:
        df_equity = pd.DataFrame({'time': [df['time'].min()], 'equity': [initial_capital], 'drawdown_pct': [0]})

    df_equity['time'] = pd.to_datetime(df_equity['time'])
    df_equity.set_index('time', inplace=True)

    # 日收益率
    daily_returns = df_equity['equity'].pct_change().dropna()

    # 年化收益
    years = (df['time'].max() - df['time'].min()).days / 365
    total_return = (final_capital - initial_capital) / initial_capital
    annual_return = ((1 + total_return) ** (1/years) - 1) * 100 if years > 0 else 0

    # 夏普比率 (假设无风险利率3%)
    rf = 0.03 / 252
    excess_returns = daily_returns - rf
    sharpe = np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-10) if len(excess_returns) > 0 else 0

    # 索提诺比率
    downside_returns = daily_returns[daily_returns < 0]
    sortino = np.sqrt(252) * excess_returns.mean() / (downside_returns.std() + 1e-10) if len(downside_returns) > 0 else 0

    # 卡尔玛比率
    calmar = annual_return / max_drawdown_pct if max_drawdown_pct > 0 else 0

    # 交易统计
    df_trades = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame()

    if len(df_trades) > 0:
        win_trades = df_trades[df_trades['pnl'] > 0]
        loss_trades = df_trades[df_trades['pnl'] <= 0]

        win_rate = len(win_trades) / len(df_trades) * 100
        avg_win = win_trades['pnl'].mean() if len(win_trades) > 0 else 0
        avg_loss = loss_trades['pnl'].mean() if len(loss_trades) > 0 else 0
        total_win = win_trades['pnl'].sum() if len(win_trades) > 0 else 0
        total_loss = abs(loss_trades['pnl'].sum()) if len(loss_trades) > 0 else 0
        profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
        max_win = df_trades['pnl'].max()
        max_loss = df_trades['pnl'].min()
        avg_holding = df_trades['holding_days'].mean()
        total_comm = df_trades['commission'].sum()

        # 按出场标签统计
        exit_tag_stats = df_trades.groupby('exit_tag').agg({
            'pnl': ['count', 'sum', 'mean'],
            'pnl_pct': 'mean'
        }).round(2)

        # 按月统计
        df_trades['exit_month'] = pd.to_datetime(df_trades['exit_time']).dt.to_period('M')
        monthly_stats = df_trades.groupby('exit_month').agg({
            'pnl': ['count', 'sum'],
            'trade_id': lambda x: (df_trades.loc[x.index, 'pnl'] > 0).sum()
        })
        monthly_stats.columns = ['trades', 'pnl', 'wins']
        monthly_stats['win_rate'] = monthly_stats['wins'] / monthly_stats['trades'] * 100

        # 按年统计
        df_trades['exit_year'] = pd.to_datetime(df_trades['exit_time']).dt.year
        yearly_stats = df_trades.groupby('exit_year').agg({
            'pnl': ['count', 'sum'],
            'trade_id': lambda x: (df_trades.loc[x.index, 'pnl'] > 0).sum()
        })
        yearly_stats.columns = ['trades', 'pnl', 'wins']
        yearly_stats['win_rate'] = yearly_stats['wins'] / yearly_stats['trades'] * 100
    else:
        win_rate = avg_win = avg_loss = profit_factor = max_win = max_loss = avg_holding = total_comm = 0
        exit_tag_stats = monthly_stats = yearly_stats = None

    result = BacktestResult(
        symbol=symbol,
        strategy=strategy_name,
        start_date=df['time'].min(),
        end_date=df['time'].max(),
        initial_capital=initial_capital,
        final_capital=final_capital,
        trades=trades,
        equity_curve=df_equity.reset_index(),
        daily_returns=daily_returns,
        total_pnl=final_capital - initial_capital,
        total_return_pct=total_return * 100,
        annual_return_pct=annual_return,
        max_drawdown_pct=max_drawdown_pct,
        max_drawdown_val=max_drawdown_val,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        max_win=max_win,
        max_loss=max_loss,
        avg_holding_days=avg_holding,
        total_commission=total_comm,
        monthly_stats=monthly_stats,
        yearly_stats=yearly_stats,
        exit_tag_stats=exit_tag_stats
    )

    return result


# 保留原有函数以兼容旧代码
def calculate_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """计算技术指标 (兼容旧代码)"""
    df = df.copy()

    # EMA
    df['ema_short'] = df['close'].ewm(span=params['sml_len'], adjust=False).mean()
    df['ema_long'] = df['close'].ewm(span=params['big_len'], adjust=False).mean()

    # 突破线 (用High)
    df['high_line'] = df['high'].rolling(window=params['break_len']).max()

    # ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=params['atr_len']).mean()

    # ADX
    up_move = df['high'].diff()
    down_move = -df['low'].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    atr_adx = tr.rolling(window=params['adx_len']).mean()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=params['adx_len']).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=params['adx_len']).mean()

    plus_di = 100 * plus_dm_smooth / (atr_adx + 1e-10)
    minus_di = 100 * minus_dm_smooth / (atr_adx + 1e-10)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    df['adx'] = dx.rolling(window=params['adx_len']).mean()

    return df


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    params: dict,
    initial_capital: float = 1000000,
    strategy_name: str = "brother2v5"
) -> BacktestResult:
    """
    运行回测 (兼容旧代码)
    """
    inst = get_instrument(symbol)
    if not inst:
        raise ValueError(f"未知品种: {symbol}")

    multiplier = inst['multiplier']
    margin_rate = inst['margin_rate']

    # 计算指标
    df = calculate_indicators(df, params)

    # 交易状态
    trades = []
    trade_id = 0
    position = 0
    entry_price = 0
    entry_time = None
    entry_tag = ""
    shares = 0
    record_high = 0

    capital = initial_capital
    equity_curve = []
    max_equity = capital
    max_drawdown_pct = 0
    max_drawdown_val = 0

    warmup = max(params['big_len'], params['break_len'], params['adx_len'] * 2) + 10

    for i in range(warmup, len(df)):
        curr_close = df['close'].iloc[i]
        curr_high = df['high'].iloc[i]
        curr_low = df['low'].iloc[i]
        ema_short = df['ema_short'].iloc[i]
        ema_long = df['ema_long'].iloc[i]
        high_line_prev = df['high_line'].iloc[i-1]
        atr = df['atr'].iloc[i]
        adx = df['adx'].iloc[i]
        current_time = df['time'].iloc[i]

        if pd.isna(atr) or pd.isna(adx) or atr == 0:
            equity_curve.append({
                'time': current_time,
                'equity': capital,
                'drawdown_pct': 0
            })
            continue

        # 计算当前权益
        if position != 0:
            unrealized_pnl = (curr_close - entry_price) * shares * multiplier * position
            current_equity = capital + unrealized_pnl
        else:
            current_equity = capital

        # 更新最大回撤
        if current_equity > max_equity:
            max_equity = current_equity

        dd_pct = (max_equity - current_equity) / max_equity * 100
        dd_val = max_equity - current_equity
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct
            max_drawdown_val = dd_val

        equity_curve.append({
            'time': current_time,
            'equity': current_equity,
            'drawdown_pct': dd_pct
        })

        # ========== 多头持仓检查 ==========
        if position == 1:
            if curr_high > record_high:
                record_high = curr_high
            if record_high < entry_price:
                record_high = entry_price

            exit_tag = None

            # 趋势反转
            if ema_short < ema_long:
                exit_tag = "trend_reverse"

            # 移动止损
            stop_line = record_high - (atr * params['stop_n'])
            if curr_close < stop_line:
                exit_tag = "trailing_stop"

            if exit_tag:
                pnl = (curr_close - entry_price) * shares * multiplier
                comm = calculate_commission(symbol, curr_close, shares) * 2
                net_pnl = pnl - comm
                capital += net_pnl

                holding_days = (current_time - entry_time).days
                pnl_pct = (curr_close - entry_price) / entry_price * 100

                trade = Trade(
                    trade_id=trade_id,
                    symbol=symbol,
                    direction=1,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    entry_tag="long_breakout",
                    volume=shares,
                    exit_time=current_time,
                    exit_price=curr_close,
                    exit_tag=exit_tag,
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    commission=comm,
                    holding_days=holding_days,
                    capital_after=capital
                )
                trades.append(trade)
                trade_id += 1

                position = 0
                shares = 0
                record_high = 0
                continue

        # ========== 开仓信号检查 ==========
        if position == 0:
            is_bullish = ema_short > ema_long
            is_trend_strong = adx > params['adx_thres']
            is_breakout = curr_close > high_line_prev

            if is_bullish and is_trend_strong and is_breakout:
                # 仓位计算
                stake_amt = capital * params['capital_rate']
                risk_per_trade = stake_amt * params['risk_rate']
                stop_dist = atr * params['stop_n']
                if stop_dist <= 0:
                    stop_dist = curr_close * 0.01

                amount = risk_per_trade / stop_dist
                shares = max(1, int(amount / multiplier))

                # 保证金检查
                required_margin = curr_close * multiplier * shares * margin_rate
                max_margin = capital * 0.8
                if required_margin > max_margin:
                    shares = max(1, int(max_margin / (curr_close * multiplier * margin_rate)))

                if shares >= 1:
                    position = 1
                    entry_price = curr_close
                    entry_time = current_time
                    entry_tag = "long_breakout"
                    record_high = curr_high

    # 平掉最后持仓
    if position != 0:
        curr_close = df['close'].iloc[-1]
        current_time = df['time'].iloc[-1]
        pnl = (curr_close - entry_price) * shares * multiplier * position
        comm = calculate_commission(symbol, curr_close, shares) * 2
        net_pnl = pnl - comm
        capital += net_pnl

        holding_days = (current_time - entry_time).days
        pnl_pct = (curr_close - entry_price) / entry_price * 100 * position

        trade = Trade(
            trade_id=trade_id,
            symbol=symbol,
            direction=position,
            entry_time=entry_time,
            entry_price=entry_price,
            entry_tag=entry_tag,
            volume=shares,
            exit_time=current_time,
            exit_price=curr_close,
            exit_tag="backtest_end",
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            commission=comm,
            holding_days=holding_days,
            capital_after=capital
        )
        trades.append(trade)

    return _calculate_statistics(
        df, symbol, strategy_name, initial_capital, capital,
        trades, equity_curve, max_drawdown_pct, max_drawdown_val
    )
