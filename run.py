# coding=utf-8
"""
启动脚本
支持启动Web界面、回测、模拟盘等模式
"""

import argparse
import sys
import os

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)


def run_web():
    """启动Web界面"""
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", os.path.join(ROOT_DIR, "app", "main.py"),
                "--server.port", "8502",
                "--server.headless", "true"]
    sys.exit(stcli.main())


def run_backtest(args):
    """运行回测"""
    from datetime import datetime
    from core.backtest_engine import BacktestEngine, generate_report
    from strategies.wavetrend_final import WaveTrendFinalStrategy

    print(f"运行回测: {args.symbol} {args.period}")

    # 创建策略
    strategy = WaveTrendFinalStrategy()
    strategy.set_symbol(args.symbol)

    # 创建回测引擎
    engine = BacktestEngine()

    # 解析日期
    start_date = datetime.strptime(args.start, "%Y-%m-%d") if args.start else None
    end_date = datetime.strptime(args.end, "%Y-%m-%d") if args.end else None

    # 运行回测
    result = engine.run(
        strategy=strategy,
        symbol=args.symbol,
        period=args.period,
        start_date=start_date,
        end_date=end_date,
        initial_capital=args.capital
    )

    # 输出结果
    report = generate_report(result)
    print(report)

    # 保存结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n报告已保存到: {args.output}")


def run_sim(args):
    """运行模拟盘"""
    from core.live_engine import LiveEngine
    from strategies.wavetrend_final import WaveTrendFinalStrategy

    print(f"启动模拟盘: {args.symbol}")

    # 创建引擎
    engine = LiveEngine()

    # 设置品种配置
    engine.set_instrument_config(args.symbol, {
        'multiplier': 10,
        'margin_rate': 0.10
    })

    # 初始化网关
    engine.init_gateway("sim")

    # 创建并添加策略
    strategy = WaveTrendFinalStrategy()
    strategy.set_symbol(args.symbol)
    engine.add_strategy(strategy, [args.symbol])

    # 启动
    engine.start(args.capital)

    print("模拟盘已启动，按Ctrl+C停止")

    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止...")
        engine.stop()
        print("模拟盘已停止")


def main():
    parser = argparse.ArgumentParser(description="期货量化交易系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # Web界面
    web_parser = subparsers.add_parser("web", help="启动Web界面")

    # 回测
    backtest_parser = subparsers.add_parser("backtest", help="运行回测")
    backtest_parser.add_argument("-s", "--symbol", required=True, help="交易品种")
    backtest_parser.add_argument("-p", "--period", default="1d", help="K线周期")
    backtest_parser.add_argument("--start", help="开始日期 (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", help="结束日期 (YYYY-MM-DD)")
    backtest_parser.add_argument("-c", "--capital", type=float, default=100000, help="初始资金")
    backtest_parser.add_argument("-o", "--output", help="输出文件")

    # 模拟盘
    sim_parser = subparsers.add_parser("sim", help="运行模拟盘")
    sim_parser.add_argument("-s", "--symbol", required=True, help="交易品种")
    sim_parser.add_argument("-c", "--capital", type=float, default=100000, help="初始资金")

    args = parser.parse_args()

    if args.command == "web":
        run_web()
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "sim":
        run_sim(args)
    else:
        # 默认启动Web界面
        run_web()


if __name__ == "__main__":
    main()
