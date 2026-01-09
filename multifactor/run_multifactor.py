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
from multifactor.factors import calculate_all_factors, get_factor_list, prepare_factor_data
from multifactor.model import StockRanker, train_ranker
from multifactor.backtest import MultifactorBacktest


def run_multifactor_strategy(
    start_date: str = "2022-01-01",
    end_date: str = "2024-12-31",
    hold_num: int = 10,
    rebalance_num: int = 1,
    train_days: int = 250,  # 训练窗口
    retrain_freq: int = 20,  # 重新训练频率
    use_sample: bool = False,  # 是否使用样本数据（快速测试）
    index_name: str = "zz1000",  # 指数成分股: hs300 / zz500 / zz1000
    max_stocks: int = 0  # 最大股票数（0=不限制）
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
    """
    print("="*60)
    print("多因子选股策略回测")
    print("="*60)

    index_names = {"hs300": "沪深300", "zz500": "中证500", "zz1000": "中证1000"}

    # 1. 获取股票池
    print("\n[1/5] 获取股票池...")
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

    # 2. 加载数据
    print("\n[2/5] 加载股票数据...")
    loader = StockDataLoader()

    # 扩展日期范围用于训练
    train_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=train_days + 100)).strftime("%Y-%m-%d")

    # 检查本地数据
    stock_data = {}
    missing_codes = []

    for code in stock_pool:
        df = loader.get_stock_data(code, train_start, end_date)
        if len(df) > 60:
            stock_data[code] = df
        else:
            missing_codes.append(code)

    print(f"  本地已有数据: {len(stock_data)}只")

    # 下载缺失数据
    if missing_codes:
        print(f"  下载缺失数据: {len(missing_codes)}只")
        loader.update_stock_data(missing_codes[:50], train_start, end_date)  # 限制下载数量

        # 重新加载
        for code in missing_codes[:50]:
            df = loader.get_stock_data(code, train_start, end_date)
            if len(df) > 60:
                stock_data[code] = df

    print(f"  最终数据: {len(stock_data)}只股票")

    if len(stock_data) < 10:
        print("错误: 股票数据不足，无法运行回测")
        return None

    # 3. 计算因子
    print("\n[3/5] 计算因子...")
    for code in stock_data:
        stock_data[code] = calculate_all_factors(stock_data[code])

    # 获取交易日
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(df["date"].tolist())
    trading_dates = sorted([d for d in all_dates if start_date <= d <= end_date])

    print(f"  交易日: {len(trading_dates)}天")
    print(f"  因子数量: {len(get_factor_list())}")

    # 4. 滚动训练模型 + 生成每日选股
    print("\n[4/5] 滚动训练模型并生成选股...")
    factor_cols = get_factor_list()
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
            train_dates = trading_dates[train_start_idx:train_end_idx]

            if len(train_dates) < 30:
                # 数据不足，使用简单动量排序
                today_data = prepare_factor_data(stock_data, date)
                if len(today_data) > 0:
                    today_data = today_data.dropna(subset=["mom_20d"])
                    ranked = today_data.sort_values("mom_20d", ascending=False)
                    daily_selections[date] = ranked["code"].head(hold_num).tolist()
                continue

            # 构建训练数据
            train_samples = []
            for d in train_dates[-60:]:  # 最近60天
                day_data = prepare_factor_data(stock_data, d)
                if len(day_data) > 0:
                    train_samples.append(day_data)

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

        # 今日选股
        today_data = prepare_factor_data(stock_data, date)
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

    # 5. 运行回测
    print("\n[5/5] 运行回测...")
    backtest = MultifactorBacktest(
        initial_capital=1000000,
        commission_rate=0.001,
        slippage=0.001,
        hold_num=hold_num,
        rebalance_num=rebalance_num
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
        max_stocks=args.max_stocks
    )
