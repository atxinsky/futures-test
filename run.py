# coding=utf-8
"""
启动脚本
支持启动Web界面、回测、模拟盘、实盘等模式
"""

import argparse
import sys
import os
import json

# 添加项目根目录到路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)


def run_web(port: int = 8504):
    """启动Web界面"""
    import streamlit.web.cli as stcli
    sys.argv = ["streamlit", "run", os.path.join(ROOT_DIR, "app", "main.py"),
                "--server.port", str(port),
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
    from config import get_instrument

    print(f"启动模拟盘: {args.symbol}")

    # 创建引擎
    engine = LiveEngine()

    # 设置品种配置
    inst = get_instrument(args.symbol)
    if inst:
        engine.set_instrument_config(args.symbol, inst)
    else:
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


def run_live(args):
    """运行实盘交易（TqSdk）"""
    from core.live_engine import LiveEngine
    from config import get_instrument

    # 加载配置
    config_file = os.path.join(ROOT_DIR, "tq_config.json")
    if os.path.exists(config_file):
        with open(config_file, 'r', encoding='utf-8') as f:
            tq_config = json.load(f)
    else:
        print("错误: 未找到配置文件 tq_config.json")
        print("请先在Web界面配置天勤账号，或手动创建配置文件")
        return

    # 检查配置
    if not tq_config.get('tq_user') or not tq_config.get('tq_password'):
        print("错误: 天勤账号未配置")
        return

    symbols = args.symbols.split(',') if args.symbols else tq_config.get('default_symbols', ['RB'])
    sim_mode = args.sim if args.sim is not None else tq_config.get('sim_mode', True)

    mode_str = "模拟盘" if sim_mode else "实盘"
    print(f"启动TqSdk {mode_str}: {symbols}")

    # 创建引擎
    engine = LiveEngine()

    # 设置品种配置
    for symbol in symbols:
        inst = get_instrument(symbol)
        if inst:
            engine.set_instrument_config(symbol, inst)

    # 初始化TqSdk网关
    gateway_type = "tq_sim" if sim_mode else "tq_live"
    gateway_config = {
        'tq_user': tq_config['tq_user'],
        'tq_password': tq_config['tq_password'],
        'sim_mode': sim_mode,
        'broker_id': tq_config.get('broker_id', ''),
        'td_account': tq_config.get('td_account', ''),
        'td_password': tq_config.get('td_password', '')
    }

    try:
        engine.init_gateway(gateway_type, gateway_config)
    except Exception as e:
        print(f"网关初始化失败: {e}")
        return

    # 如果指定了策略，则加载策略
    if args.strategy:
        from strategies.base import create_strategy
        strategy = create_strategy(args.strategy)
        if strategy:
            engine.add_strategy(strategy, symbols)
            print(f"已加载策略: {args.strategy}")
        else:
            print(f"警告: 未找到策略 {args.strategy}")

    # 启动
    engine.start(args.capital)

    print(f"\nTqSdk {mode_str}已启动")
    print(f"交易品种: {symbols}")
    print(f"初始资金: {args.capital:,.0f}")
    print("按Ctrl+C停止\n")

    try:
        import time
        while True:
            time.sleep(1)
            # 可以添加状态打印
            if engine.is_running:
                account = engine.get_account()
                if account:
                    print(f"\r权益: {account.balance:,.2f} | 可用: {account.available:,.2f}", end="", flush=True)
    except KeyboardInterrupt:
        print("\n\n正在停止...")
        engine.stop()
        print("交易已停止")


def main():
    parser = argparse.ArgumentParser(description="期货量化交易系统 v2.0")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # Web界面
    web_parser = subparsers.add_parser("web", help="启动Web界面")
    web_parser.add_argument("-p", "--port", type=int, default=8504, help="Web服务端口 (默认: 8504)")

    # 回测
    backtest_parser = subparsers.add_parser("backtest", help="运行回测")
    backtest_parser.add_argument("-s", "--symbol", required=True, help="交易品种")
    backtest_parser.add_argument("-p", "--period", default="1d", help="K线周期")
    backtest_parser.add_argument("--start", help="开始日期 (YYYY-MM-DD)")
    backtest_parser.add_argument("--end", help="结束日期 (YYYY-MM-DD)")
    backtest_parser.add_argument("-c", "--capital", type=float, default=100000, help="初始资金")
    backtest_parser.add_argument("-o", "--output", help="输出文件")

    # 模拟盘 (简单本地模拟)
    sim_parser = subparsers.add_parser("sim", help="运行本地模拟盘")
    sim_parser.add_argument("-s", "--symbol", required=True, help="交易品种")
    sim_parser.add_argument("-c", "--capital", type=float, default=100000, help="初始资金")

    # TqSdk实盘/模拟盘
    live_parser = subparsers.add_parser("live", help="运行TqSdk实盘/模拟盘")
    live_parser.add_argument("-s", "--symbols", help="交易品种，逗号分隔 (如: RB,AU,IF)")
    live_parser.add_argument("-c", "--capital", type=float, default=100000, help="初始资金")
    live_parser.add_argument("--strategy", help="策略名称")
    live_parser.add_argument("--sim", action="store_true", default=None, help="使用模拟盘")
    live_parser.add_argument("--real", action="store_false", dest="sim", help="使用实盘")

    args = parser.parse_args()

    if args.command == "web":
        run_web(args.port)
    elif args.command == "backtest":
        run_backtest(args)
    elif args.command == "sim":
        run_sim(args)
    elif args.command == "live":
        run_live(args)
    else:
        # 默认启动Web界面
        print("期货量化交易系统 v2.0")
        print("=" * 40)
        print("可用命令:")
        print("  web       启动Web界面 (默认端口 8504)")
        print("  backtest  运行策略回测")
        print("  sim       运行本地模拟盘")
        print("  live      运行TqSdk实盘/模拟盘")
        print("")
        print("使用 'python run.py <命令> -h' 查看详细帮助")
        print("")
        print("正在启动Web界面...")
        run_web()


if __name__ == "__main__":
    main()
