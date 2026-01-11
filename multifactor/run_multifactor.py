# coding=utf-8
"""
多因子选股策略 - 主运行脚本

使用沪深300成分股，LightGBM排序模型，
每日持仓10只，每日换仓1只
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

from multifactor.data_loader import StockDataLoader, get_hs300_components, get_zz1000_components, get_index_components
from multifactor.factors import (
    calculate_all_factors, get_factor_list, prepare_factor_data,
    calculate_factors_parallel, prepare_factor_data_parallel
)
from multifactor.model import StockRanker, train_ranker
from multifactor.backtest import MultifactorBacktest
import multiprocessing

# CPU核心数
CPU_COUNT = multiprocessing.cpu_count()


def run_multifactor_strategy(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    hold_num: int = 10,
    rebalance_num: int = 1,
    train_days: int = 250,  # 训练窗口
    retrain_freq: int = 20,  # 重新训练频率
    use_sample: bool = False,  # 是否使用样本数据（快速测试）
    index_name: str = "zz1000",  # 指数成分股: hs300 / zz500 / zz1000
    max_stocks: int = 0,  # 最大股票数（0=不限制）
    # === 换手率控制参数 ===
    rebalance_days: int = 1,  # 调仓周期（每N天调一次）
    position_sticky: float = 0.0,  # 持仓粘性（0-1）
    min_holding_days: int = 0,  # 最小持仓天数
    # === 风控参数 ===
    vol_timing: bool = False,  # 波动率择时
    vol_threshold: float = 0.30,  # 波动率阈值
    drawdown_stop: bool = False,  # 整体止损
    max_drawdown_limit: float = 0.15,  # 最大回撤阈值
    # === 市场择时参数 ===
    market_timing: bool = False,  # 市场均线择时
    market_ma_days: int = 20,  # 均线天数
    # === 性能参数 ===
    n_jobs: int = 0  # 并行数（0=自动，使用CPU核心数）
):
    """
    运行多因子选股策略

    Args:
        start_date: 回测开始日期
        end_date: 回测结束日期
        hold_num: 持仓数量
        rebalance_num: 每日换仓数量
        train_days: 模型训练使用的历史天数
        retrain_freq: 模型重新训练的频率（天）
        use_sample: 使用样本股票池快速测试
        index_name: 指数成分股（hs300/zz500/zz1000）
        max_stocks: 最大股票数限制（0=不限制）
        rebalance_days: 调仓周期（每N天调一次，默认1=每天）
        position_sticky: 持仓粘性（0-1，越高越不易换出，默认0）
        min_holding_days: 最小持仓天数（默认0=不限制）
        vol_timing: 波动率择时（高波动时减仓）
        vol_threshold: 波动率阈值（年化，默认30%）
        drawdown_stop: 整体止损（回撤超阈值清仓）
        max_drawdown_limit: 最大回撤阈值（默认15%）
        market_timing: 市场均线择时（熊市减仓）
        market_ma_days: 均线天数（默认20）
        n_jobs: 并行线程数（0=自动）
    """
    # 设置并行数
    if n_jobs <= 0:
        n_jobs = CPU_COUNT
    print(f"并行线程数: {n_jobs} (CPU核心: {CPU_COUNT})")

    print("="*60)
    print("多因子选股策略回测")
    print("="*60)

    index_names = {"hs300": "沪深300", "zz500": "中证500", "zz1000": "中证1000"}

    # 1. 获取股票池
    print("\n[1/6] 获取股票池...")
    if use_sample:
        # 样本股票池（快速测试）
        stock_pool = [
            "600519", "601318", "600036", "000858", "601166",
            "600276", "601398", "600900", "000333", "002415",
            "600030", "601888", "600887", "000651", "601012",
            "600309", "002304", "600585", "601628", "000725",
            "002594", "600104", "601668", "000002", "600048",
            "002142", "600000", "601601", "000568", "002027"
        ]
        print(f"  使用样本股票池: {len(stock_pool)}只")
    else:
        stock_pool = get_index_components(index_name)
        if not stock_pool:
            print(f"  无法获取{index_names.get(index_name, index_name)}成分股，使用样本股票池")
            stock_pool = [
                "600519", "601318", "600036", "000858", "601166",
                "600276", "601398", "600900", "000333", "002415"
            ]
        else:
            if max_stocks > 0 and len(stock_pool) > max_stocks:
                stock_pool = stock_pool[:max_stocks]
                print(f"  {index_names.get(index_name, index_name)}成分股 (限制{max_stocks}只): {len(stock_pool)}只")
            else:
                print(f"  {index_names.get(index_name, index_name)}成分股: {len(stock_pool)}只")

    # 2. 加载数据（批量SQL查询优化）
    print("\n[2/6] 加载股票数据...")
    loader = StockDataLoader()

    # 扩展日期范围用于训练
    train_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=train_days + 100)).strftime("%Y-%m-%d")

    import time as time_module
    load_start = time_module.time()

    # 批量加载所有股票数据（一次SQL查询）
    stock_data = loader.get_stocks_data_batch(stock_pool, train_start, end_date)

    # 过滤数据不足的股票
    stock_data = {code: df for code, df in stock_data.items() if len(df) > 60}
    load_time = time_module.time() - load_start

    print(f"  批量加载完成: {len(stock_data)}只股票 (耗时: {load_time:.2f}秒)")

    # 下载缺失数据
    missing_codes = [c for c in stock_pool if c not in stock_data]
    if missing_codes and len(missing_codes) < len(stock_pool):
        print(f"  下载缺失数据: {len(missing_codes)}只")
        loader.update_stock_data(missing_codes[:50], train_start, end_date)  # 限制下载数量

        # 重新批量加载缺失的
        new_data = loader.get_stocks_data_batch(missing_codes[:50], train_start, end_date)
        for code, df in new_data.items():
            if len(df) > 60:
                stock_data[code] = df

    print(f"  最终数据: {len(stock_data)}只股票")

    if len(stock_data) < 10:
        print("错误: 股票数据不足，无法运行回测")
        return None

    # 3. 计算因子（并行）
    print(f"\n[3/6] 计算因子... (并行: {n_jobs}线程)")
    stock_data = calculate_factors_parallel(stock_data, n_jobs=n_jobs)

    # 获取交易日
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df["date"].tolist())
    trading_dates = sorted([d for d in all_dates if start_date <= d <= end_date])

    print(f"  交易日: {len(trading_dates)}天")
    print(f"  因子数量: {len(get_factor_list())}")

    # 4. 预计算所有日期的因子截面数据（避免重复计算）
    print(f"\n[4/6] 预计算因子截面... (并行: {n_jobs}线程)")
    factor_cols = get_factor_list()
    all_factor_data = {}

    # 扩展到训练所需的日期
    all_needed_dates = set()
    for df in stock_data.values():
        all_needed_dates.update(df["date"].tolist())
    all_needed_dates = sorted([d for d in all_needed_dates])

    # 并行预计算所有日期的因子数据
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def prepare_single_date(date):
        return date, prepare_factor_data(stock_data, date)

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(prepare_single_date, d): d for d in all_needed_dates}
        completed = 0
        total_dates = len(all_needed_dates)
        for future in as_completed(futures):
            try:
                date, df = future.result()
                if len(df) > 0:
                    all_factor_data[date] = df
                completed += 1
                if completed % 200 == 0:
                    print(f"  预计算进度: {completed}/{total_dates}")
            except Exception as e:
                logger.warning(f"预计算因子失败: {e}")

    print(f"  预计算完成: {len(all_factor_data)}天")

    # 5. 滚动训练模型 + 生成每日选股
    print("\n[5/6] 滚动训练模型并生成选股...")
    daily_selections = {}
    ranker = None
    last_train_date = None

    for i, date in enumerate(trading_dates):
        # 是否需要重新训练
        need_train = (ranker is None or
                      (last_train_date and i % retrain_freq == 0))

        if need_train:
            # 准备训练数据
            train_end_idx = i
            train_start_idx = max(0, i - train_days)
            train_dates_list = trading_dates[train_start_idx:train_end_idx]

            if len(train_dates_list) < 30:
                # 数据不足，使用简单动量排序
                if date in all_factor_data:
                    today_data = all_factor_data[date]
                    today_data = today_data.dropna(subset=["mom_20d"])
                    ranked = today_data.sort_values("mom_20d", ascending=False)
                    daily_selections[date] = ranked["code"].head(hold_num).tolist()
                continue

            # 构建训练数据（从预计算数据中获取）
            train_samples = []
            for d in train_dates_list[-60:]:  # 最近60天
                if d in all_factor_data:
                    train_samples.append(all_factor_data[d])

            if not train_samples:
                continue

            train_df = pd.concat(train_samples, ignore_index=True)
            train_df = train_df.dropna(subset=["future_ret_5d"])

            if len(train_df) < 100:
                continue

            # 训练模型
            try:
                ranker, _ = train_ranker(train_df, factor_cols)
                last_train_date = date
            except Exception as e:
                logger.warning(f"训练失败: {e}")
                continue

        # 今日选股（从预计算数据中获取）
        if date not in all_factor_data:
            continue
        today_data = all_factor_data[date]
        if len(today_data) < 5 or ranker is None:
            continue

        today_data = today_data.dropna(subset=factor_cols, how="all")
        if len(today_data) < 5:
            continue

        try:
            X = today_data[factor_cols]
            codes = today_data["code"].tolist()
            selected = ranker.rank(X, codes, top_n=hold_num)
            daily_selections[date] = selected
        except Exception as e:
            logger.warning(f"{date} 选股失败: {e}")

        if (i + 1) % 50 == 0:
            print(f"  进度: {i+1}/{len(trading_dates)}")

    print(f"  生成选股结果: {len(daily_selections)}天")

    # 6. 运行回测
    print("\n[6/6] 运行回测...")
    risk_info = []
    if vol_timing:
        risk_info.append(f"波动率择时(>{vol_threshold*100:.0f}%减仓)")
    if drawdown_stop:
        risk_info.append(f"整体止损(>{max_drawdown_limit*100:.0f}%清仓)")
    if risk_info:
        print(f"  风控: {', '.join(risk_info)}")

    backtest = MultifactorBacktest(
        initial_capital=1000000,
        commission_rate=0.001,
        slippage=0.001,
        hold_num=hold_num,
        rebalance_num=rebalance_num,
        # 换手率控制
        rebalance_days=rebalance_days,
        position_sticky=position_sticky,
        min_holding_days=min_holding_days,
        # 风控
        vol_timing=vol_timing,
        vol_threshold=vol_threshold,
        drawdown_stop=drawdown_stop,
        max_drawdown_limit=max_drawdown_limit,
        # 市场择时
        market_timing=market_timing,
        market_ma_days=market_ma_days
    )

    result = backtest.run(stock_data, daily_selections, start_date, end_date)

    # 输出结果
    print("\n" + "="*60)
    print("回测结果")
    print("="*60)
    print(f"累计收益:     {result.total_return*100:>10.2f}%")
    print(f"年化收益:     {result.annual_return*100:>10.2f}%")
    print(f"最大回撤:     {result.max_drawdown*100:>10.2f}%")
    print(f"夏普比率:     {result.sharpe_ratio:>10.2f}")
    print(f"卡玛比率:     {result.calmar_ratio:>10.2f}")
    print(f"胜率:         {result.win_rate*100:>10.1f}%")
    print(f"总交易次数:   {result.total_trades:>10}")
    print(f"年换手率:     {result.turnover:>10.1f}倍")
    print("="*60)

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多因子选股回测")
    parser.add_argument("--start", default="2021-01-01", help="开始日期")
    parser.add_argument("--end", default="2026-01-01", help="结束日期")
    parser.add_argument("--index", default="zz1000", choices=["hs300", "zz500", "zz1000"], help="指数成分股")
    parser.add_argument("--hold", type=int, default=20, help="持仓数量")
    parser.add_argument("--rebalance", type=int, default=2, help="每日换仓数量")
    parser.add_argument("--train-days", type=int, default=120, help="训练窗口天数")
    parser.add_argument("--retrain-freq", type=int, default=20, help="重训频率")
    parser.add_argument("--max-stocks", type=int, default=0, help="最大股票数(0=不限)")
    parser.add_argument("--sample", action="store_true", help="使用样本股票池")
    # 换手率控制参数
    parser.add_argument("--rebalance-days", type=int, default=1, help="调仓周期(每N天,默认1)")
    parser.add_argument("--position-sticky", type=float, default=0.0, help="持仓粘性(0-1,默认0)")
    parser.add_argument("--min-hold-days", type=int, default=0, help="最小持仓天数(默认0)")
    # 风控参数
    parser.add_argument("--vol-timing", action="store_true", help="启用波动率择时")
    parser.add_argument("--vol-threshold", type=float, default=0.30, help="波动率阈值(默认0.30)")
    parser.add_argument("--drawdown-stop", action="store_true", help="启用整体止损")
    parser.add_argument("--max-dd", type=float, default=0.15, help="最大回撤阈值(默认0.15)")
    # 市场择时参数
    parser.add_argument("--market-timing", action="store_true", help="启用市场均线择时")
    parser.add_argument("--market-ma", type=int, default=20, help="均线天数(默认20)")
    # 性能参数
    parser.add_argument("--n-jobs", type=int, default=0, help="并行线程数(0=自动)")

    args = parser.parse_args()

    result = run_multifactor_strategy(
        start_date=args.start,
        end_date=args.end,
        hold_num=args.hold,
        rebalance_num=args.rebalance,
        train_days=args.train_days,
        retrain_freq=args.retrain_freq,
        use_sample=args.sample,
        index_name=args.index,
        max_stocks=args.max_stocks,
        # 换手率控制
        rebalance_days=args.rebalance_days,
        position_sticky=args.position_sticky,
        min_holding_days=args.min_hold_days,
        # 风控
        vol_timing=args.vol_timing,
        vol_threshold=args.vol_threshold,
        drawdown_stop=args.drawdown_stop,
        max_drawdown_limit=args.max_dd,
        # 市场择时
        market_timing=args.market_timing,
        market_ma_days=args.market_ma,
        # 性能
        n_jobs=args.n_jobs
    )
