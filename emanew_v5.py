# coding=utf-8
"""
============================================================================
emanew V5 策略 - 分批止盈版
============================================================================

核心改进：分批止盈，解决"卖飞焦虑"问题

交易心理学背景：
- 问题：卖出后行情继续涨 → 懊恼 → 追高杀回 → 顶部震荡止损 → 心态资金双回撤
- 解决：分批止盈，第一次锁定利润消除焦虑，剩余仓位让利润奔跑

分批止盈逻辑：
┌─────────────────────────────────────────────────────────────────────┐
│ 第一次止盈（平仓50%）                                                 │
│ - 触发：盈利≥15% 且 从高点回撤≥6%                                    │
│ - 目的：锁定一半利润，消除"完全踏空"的焦虑                            │
├─────────────────────────────────────────────────────────────────────┤
│ 第二次止盈（平仓剩余全部）                                            │
│ - 触发：从高点回撤≥12% 或 EMA死叉                                    │
│ - 目的：给利润足够的奔跑空间，不怕"卖飞"                              │
└─────────────────────────────────────────────────────────────────────┘

其他止损机制：
- 信号K线止损：连续3天收盘低于金叉K线最低价
- 固定止损：亏损超过8%立即出场
- 保本止损：盈利超10%后跌回入场价

使用方法：
    from emanew_v5 import EmaNewV5

    strategy = EmaNewV5()
    trades, equity, final_capital = strategy.backtest(df)
    metrics = strategy.calculate_metrics(trades, equity)
============================================================================
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class StrategyParams:
    """策略参数配置"""
    # 技术指标参数
    ema_fast: int = 9
    ema_slow: int = 21
    ma_len: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_smooth: int = 9

    # 止损参数（百分比）
    stop_loss: float = 0.08           # 固定止损 8%
    break_even: float = 0.10          # 保本触发 10%
    signal_low_days: int = 3          # 信号K线止损天数
    signal_low_buffer: float = 0.0    # 信号K线缓冲区

    # 分批止盈参数
    partial_trigger: float = 0.15     # 第一次止盈触发 15%
    partial_drawdown: float = 0.06    # 第一次止盈回撤 6%
    partial_rate: float = 0.50        # 第一次平仓比例 50%
    full_drawdown: float = 0.12       # 第二次止盈回撤 12%

    # 资金管理
    initial_capital: float = 1000000
    contract_multiplier: float = 300  # 合约乘数（IF=300）
    margin_rate: float = 0.12         # 保证金比例
    commission_rate: float = 0.000023 # 手续费率


@dataclass
class Position:
    """持仓状态"""
    entry_price: float = 0
    entry_time: Optional[str] = None
    high_since: float = 0
    signal_low: float = 0
    days_below_signal: int = 0
    initial_shares: float = 1
    current_shares: float = 1
    has_partial_exit: bool = False
    high_after_partial: float = 0


class EmaNewV5:
    """
    emanew V5 策略 - 分批止盈
    """

    def __init__(self, params: StrategyParams = None):
        self.params = params or StrategyParams()
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.capital = self.params.initial_capital

    def reset(self):
        """重置策略状态"""
        self.position = None
        self.trades = []
        self.equity_curve = []
        self.capital = self.params.initial_capital

    @staticmethod
    def calculate_ema(series: pd.Series, period: int) -> pd.Series:
        """计算EMA"""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(series: pd.Series, period: int) -> pd.Series:
        """计算SMA"""
        return series.rolling(window=period).mean()

    @staticmethod
    def calculate_macd(close: pd.Series, fast: int, slow: int, smooth: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=smooth, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    @staticmethod
    def detect_cross(fast: pd.Series, slow: pd.Series) -> np.ndarray:
        """检测金叉死叉: 1=金叉, -1=死叉, 0=无"""
        cross = np.zeros(len(fast))
        for i in range(1, len(fast)):
            if fast.iloc[i] > slow.iloc[i] and fast.iloc[i-1] <= slow.iloc[i-1]:
                cross[i] = 1
            elif fast.iloc[i] < slow.iloc[i] and fast.iloc[i-1] >= slow.iloc[i-1]:
                cross[i] = -1
        return cross

    def prepare_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        p = self.params

        df['ema_fast'] = self.calculate_ema(df['close'], p.ema_fast)
        df['ema_slow'] = self.calculate_ema(df['close'], p.ema_slow)
        df['ma'] = self.calculate_sma(df['close'], p.ma_len)
        df['macd'], df['signal'], df['hist'] = self.calculate_macd(
            df['close'], p.macd_fast, p.macd_slow, p.macd_smooth
        )
        df['ema_cross'] = self.detect_cross(df['ema_fast'], df['ema_slow'])

        return df

    def check_entry_signal(self, row: pd.Series, prev_row: pd.Series) -> bool:
        """检查入场信号"""
        return (
            row['ema_cross'] == 1 and
            row['hist'] > 0 and
            (row['hist'] > prev_row['hist'] or prev_row['hist'] < 0) and
            row['close'] > row['ma']
        )

    def check_exit_signal(self, row: pd.Series, prev_row: pd.Series) -> Tuple[Optional[str], float]:
        """
        检查出场信号

        返回:
            exit_tag: 出场原因，如果没有则返回None
            exit_shares: 出场手数
        """
        if self.position is None or self.position.current_shares <= 0:
            return None, 0

        p = self.params
        price = row['close']
        pos = self.position

        # 更新持仓期间最高价
        if price > pos.high_since:
            pos.high_since = price

        # 更新第一次止盈后的最高价
        if pos.has_partial_exit and price > pos.high_after_partial:
            pos.high_after_partial = price

        # 计算盈亏比例
        profit_rate = (price - pos.entry_price) / pos.entry_price
        drawdown_from_high = (pos.high_since - price) / pos.high_since if pos.high_since > 0 else 0
        max_profit_rate = (pos.high_since - pos.entry_price) / pos.entry_price

        # 信号K线止损阈值
        signal_low_threshold = pos.signal_low * (1 - p.signal_low_buffer)

        # 检查今天收盘价是否低于信号K线最低价
        if price < signal_low_threshold:
            pos.days_below_signal += 1
        else:
            pos.days_below_signal = 0

        # ============ 信号K线止损（全部出场）============
        if pos.days_below_signal >= p.signal_low_days:
            return "signal_low_break", pos.current_shares

        # ============ 固定止损（全部出场）============
        if profit_rate <= -p.stop_loss:
            return "stop_loss", pos.current_shares

        # ============ 分批止盈逻辑 ============
        if not pos.has_partial_exit:
            # 检查第一次止盈条件：盈利≥15% 且 从高点回撤≥6%
            if max_profit_rate >= p.partial_trigger and drawdown_from_high >= p.partial_drawdown:
                return "partial_stop", pos.current_shares * p.partial_rate
        else:
            # 已经第一次止盈，检查第二次
            if pos.high_after_partial > 0:
                drawdown_after_partial = (pos.high_after_partial - price) / pos.high_after_partial
                if drawdown_after_partial >= p.full_drawdown:
                    return "trail_stop_full", pos.current_shares

        # ============ 保本止损（全部出场）============
        if max_profit_rate >= p.break_even and profit_rate <= 0:
            return "break_even", pos.current_shares

        # ============ 技术信号出场（死叉）============
        if row['ema_cross'] == -1 and price < row['ma']:
            return "death_cross", pos.current_shares

        return None, 0

    def execute_entry(self, row: pd.Series):
        """执行入场"""
        self.position = Position(
            entry_price=row['close'],
            entry_time=row['time'],
            high_since=row['close'],
            signal_low=row['low'],
            days_below_signal=0,
            initial_shares=1,
            current_shares=1,
            has_partial_exit=False,
            high_after_partial=0
        )

    def execute_exit(self, row: pd.Series, exit_tag: str, exit_shares: float):
        """执行出场"""
        p = self.params
        pos = self.position
        price = row['close']

        # 计算盈亏
        pnl = (price - pos.entry_price) * exit_shares * p.contract_multiplier
        commission = price * exit_shares * p.contract_multiplier * p.commission_rate * 2
        net_pnl = pnl - commission

        self.capital += net_pnl

        # 记录交易
        max_profit_rate = (pos.high_since - pos.entry_price) / pos.entry_price
        self.trades.append({
            'entry_time': pos.entry_time,
            'exit_time': row['time'],
            'entry_price': pos.entry_price,
            'exit_price': price,
            'shares': exit_shares,
            'pnl': net_pnl,
            'pnl_pct': (price - pos.entry_price) / pos.entry_price * 100,
            'exit_tag': exit_tag,
            'max_profit_pct': max_profit_rate * 100,
            'high_since': pos.high_since,
            'is_partial': exit_tag == "partial_stop"
        })

        # 更新持仓
        pos.current_shares -= exit_shares

        # 如果是第一次止盈，标记并记录当前价格
        if exit_tag == "partial_stop":
            pos.has_partial_exit = True
            pos.high_after_partial = price

        # 如果完全清仓，重置position
        if pos.current_shares <= 0.01:
            self.position = None

    def backtest(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict], float]:
        """
        运行回测

        参数:
            df: 包含 time, open, high, low, close, volume 的DataFrame

        返回:
            trades: 交易记录列表
            equity_curve: 权益曲线
            final_capital: 最终资金
        """
        self.reset()

        # 标准化列名
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # 计算指标
        df = self.prepare_indicators(df)

        # 预热期
        p = self.params
        warmup = max(p.ema_slow, p.ma_len, p.macd_slow) + 10

        for i in range(warmup, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]

            # 检查出场
            if self.position is not None and self.position.current_shares > 0:
                exit_tag, exit_shares = self.check_exit_signal(row, prev_row)
                if exit_tag and exit_shares > 0:
                    self.execute_exit(row, exit_tag, exit_shares)

            # 检查入场（只有完全清仓后才能再次入场）
            if self.position is None:
                if self.check_entry_signal(row, prev_row):
                    self.execute_entry(row)

            # 记录权益
            unrealized = 0
            if self.position is not None and self.position.current_shares > 0:
                unrealized = (row['close'] - self.position.entry_price) * \
                            self.position.current_shares * p.contract_multiplier

            self.equity_curve.append({
                'time': row['time'],
                'equity': self.capital + unrealized
            })

        return self.trades, self.equity_curve, self.capital

    def calculate_metrics(self, trades: List[Dict] = None, equity_curve: List[Dict] = None) -> Dict:
        """计算回测指标"""
        trades = trades or self.trades
        equity_curve = equity_curve or self.equity_curve

        if not trades:
            return {
                'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
                'total_return': 0, 'max_drawdown': 0, 'sharpe_ratio': 0,
                'profit_factor': 0, 'avg_win': 0, 'avg_loss': 0, 'exit_tags': {},
                'partial_exits': 0, 'full_exits': 0
            }

        df_trades = pd.DataFrame(trades)
        df_equity = pd.DataFrame(equity_curve)

        # 基础统计
        total_trades = len(df_trades)
        wins = df_trades[df_trades['pnl'] > 0]
        losses = df_trades[df_trades['pnl'] <= 0]
        win_rate = len(wins) / total_trades * 100

        total_pnl = df_trades['pnl'].sum()
        total_return = (total_pnl / self.params.initial_capital) * 100

        # 最大回撤
        df_equity['peak'] = df_equity['equity'].cummax()
        df_equity['drawdown'] = (df_equity['peak'] - df_equity['equity']) / df_equity['peak']
        max_drawdown = df_equity['drawdown'].max() * 100

        # 盈亏比
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = abs(losses['pnl'].mean()) if len(losses) > 0 else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

        # 夏普比率
        if len(df_equity) > 1:
            returns = df_equity['equity'].pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # 出场原因统计
        exit_tags = df_trades['exit_tag'].value_counts().to_dict()

        # 分批止盈统计
        partial_exits = len(df_trades[df_trades['exit_tag'] == 'partial_stop'])
        full_exits = len(df_trades[df_trades['exit_tag'] == 'trail_stop_full'])

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'exit_tags': exit_tags,
            'partial_exits': partial_exits,
            'full_exits': full_exits
        }


# ==================== 独立运行示例 ====================
if __name__ == "__main__":
    import os

    # 读取数据
    data_path = r"D:\期货\股指期货\CFFEX_DLY_IF1!, 1D_6ce25.csv"

    if os.path.exists(data_path):
        print(f"读取数据: {data_path}")
        df = pd.read_csv(data_path)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').reset_index(drop=True)

        print(f"数据范围: {df['time'].min()} ~ {df['time'].max()}")
        print(f"数据条数: {len(df)}")

        # 创建策略实例
        strategy = EmaNewV5()

        # 运行回测
        print("\n运行 emanew V5 回测...")
        trades, equity, capital = strategy.backtest(df)
        metrics = strategy.calculate_metrics()

        # 打印结果
        print(f"\n{'='*50}")
        print("回测结果:")
        print(f"{'='*50}")
        print(f"总交易次数: {metrics['total_trades']}")
        print(f"胜率: {metrics['win_rate']:.1f}%")
        print(f"总收益: {metrics['total_pnl']:,.0f} 元")
        print(f"收益率: {metrics['total_return']:.2f}%")
        print(f"最大回撤: {metrics['max_drawdown']:.2f}%")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"盈亏比: {metrics['profit_factor']:.2f}")

        print(f"\n分批止盈统计:")
        print(f"  第一次止盈(50%): {metrics['partial_exits']} 次")
        print(f"  第二次止盈(全部): {metrics['full_exits']} 次")

        print(f"\n出场原因统计:")
        for tag, count in metrics['exit_tags'].items():
            print(f"  {tag}: {count}")
    else:
        print(f"数据文件不存在: {data_path}")
        print("请修改 data_path 变量指向你的数据文件")
