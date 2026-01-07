# coding=utf-8
"""
Docker模拟盘服务
7x24持续运行，实时接收TqSdk数据，执行策略交易
"""

import os
import sys
import json
import time
import signal
import logging
from datetime import datetime, time as dtime
from typing import List, Optional

# 添加项目路径
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/trading.log' if os.path.exists('/app/logs') else 'trading.log', encoding='utf-8')
    ]
)
logger = logging.getLogger('DockerService')


class TradingService:
    """
    Docker模拟盘交易服务

    功能:
    1. 自动连接TqSdk获取实时数据
    2. 加载并运行指定策略
    3. 交易时段自动启停
    4. 每日结算和报告
    5. 异常自动重连
    """

    def __init__(self, config_file: str = "tq_config.json"):
        self.config_file = config_file
        self.config = self._load_config()

        self.engine = None
        self.running = False
        self.start_time = None

        # 交易时段 (中国期货)
        self.trading_sessions = [
            (dtime(9, 0), dtime(10, 15)),   # 上午第一节
            (dtime(10, 30), dtime(11, 30)), # 上午第二节
            (dtime(13, 30), dtime(15, 0)),  # 下午
            (dtime(21, 0), dtime(23, 0)),   # 夜盘第一段
            (dtime(23, 0), dtime(2, 30)),   # 夜盘跨日(特殊处理)
        ]

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self) -> dict:
        """加载配置"""
        config_path = os.path.join(ROOT_DIR, self.config_file)

        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # 从环境变量读取
        return {
            'tq_user': os.getenv('TQ_USER', ''),
            'tq_password': os.getenv('TQ_PASSWORD', ''),
            'sim_mode': os.getenv('SIM_MODE', 'true').lower() == 'true',
            'default_symbols': os.getenv('SYMBOLS', 'RB,AU,IF').split(','),
            'strategy': os.getenv('STRATEGY', 'brother2v6'),
            'initial_capital': float(os.getenv('INITIAL_CAPITAL', '100000')),
            'risk_config': {
                'max_position_per_symbol': int(os.getenv('MAX_POSITION', '10')),
                'max_daily_loss': float(os.getenv('MAX_DAILY_LOSS', '0.05')),
                'max_drawdown': float(os.getenv('MAX_DRAWDOWN', '0.15'))
            }
        }

    def _signal_handler(self, signum, frame):
        """信号处理"""
        logger.info(f"收到信号 {signum}，准备停止...")
        self.stop()

    def _is_trading_time(self) -> bool:
        """判断是否在交易时段"""
        now = datetime.now().time()

        for start, end in self.trading_sessions:
            if start <= end:
                if start <= now <= end:
                    return True
            else:
                # 跨日时段
                if now >= start or now <= end:
                    return True

        return False

    def _is_trading_day(self) -> bool:
        """判断是否交易日（简单判断：周一到周五）"""
        return datetime.now().weekday() < 5

    def start(self):
        """启动服务"""
        logger.info("=" * 50)
        logger.info("期货量化交易服务启动")
        logger.info("=" * 50)

        # 检查配置
        if not self.config.get('tq_user') or not self.config.get('tq_password'):
            logger.error("天勤账号未配置，请设置环境变量 TQ_USER 和 TQ_PASSWORD")
            return

        self.running = True
        self.start_time = datetime.now()

        symbols = self.config.get('default_symbols', ['RB'])
        strategy_name = self.config.get('strategy', 'brother2v6')
        initial_capital = self.config.get('initial_capital', 100000)

        logger.info(f"交易品种: {symbols}")
        logger.info(f"策略: {strategy_name}")
        logger.info(f"初始资金: {initial_capital:,.0f}")
        logger.info(f"模式: {'模拟盘' if self.config.get('sim_mode', True) else '实盘'}")

        # 主循环
        self._main_loop(symbols, strategy_name, initial_capital)

    def _main_loop(self, symbols: List[str], strategy_name: str, initial_capital: float):
        """主循环"""
        engine_started = False
        last_status_time = time.time()
        daily_report_done = False

        while self.running:
            try:
                now = datetime.now()
                is_trading = self._is_trading_time() and self._is_trading_day()

                # 交易时段开始，启动引擎
                if is_trading and not engine_started:
                    logger.info("进入交易时段，启动交易引擎...")
                    self._start_engine(symbols, strategy_name, initial_capital)
                    engine_started = True
                    daily_report_done = False

                # 交易时段结束，停止引擎
                if not is_trading and engine_started:
                    logger.info("交易时段结束，停止交易引擎...")
                    self._stop_engine()
                    engine_started = False

                    # 生成日报
                    if not daily_report_done:
                        self._generate_daily_report()
                        daily_report_done = True

                # 定时状态报告（每5分钟）
                if engine_started and time.time() - last_status_time > 300:
                    self._print_status()
                    last_status_time = time.time()

                # 休眠
                time.sleep(10)

            except Exception as e:
                logger.error(f"主循环异常: {e}")

                # 尝试重启引擎
                if engine_started:
                    try:
                        self._stop_engine()
                    except:
                        pass
                    engine_started = False

                time.sleep(60)  # 等待1分钟后重试

    def _start_engine(self, symbols: List[str], strategy_name: str, initial_capital: float):
        """启动交易引擎"""
        try:
            from core.live_engine import LiveEngine
            from config import get_instrument
            from strategies.base import create_strategy

            self.engine = LiveEngine()

            # 设置品种配置
            for symbol in symbols:
                inst = get_instrument(symbol)
                if inst:
                    self.engine.set_instrument_config(symbol, inst)

            # 初始化TqSdk网关
            gateway_type = "tq_sim" if self.config.get('sim_mode', True) else "tq_live"
            gateway_config = {
                'tq_user': self.config['tq_user'],
                'tq_password': self.config['tq_password'],
                'sim_mode': self.config.get('sim_mode', True)
            }

            self.engine.init_gateway(gateway_type, gateway_config)

            # 加载策略
            strategy = create_strategy(strategy_name)
            if strategy:
                self.engine.add_strategy(strategy, symbols)
                logger.info(f"已加载策略: {strategy_name}")
            else:
                logger.warning(f"未找到策略: {strategy_name}，将只接收数据")

            # 启动
            self.engine.start(initial_capital)
            logger.info("交易引擎启动成功")

        except Exception as e:
            logger.error(f"启动引擎失败: {e}")
            self.engine = None
            raise

    def _stop_engine(self):
        """停止交易引擎"""
        if self.engine:
            try:
                self.engine.stop()
                logger.info("交易引擎已停止")
            except Exception as e:
                logger.error(f"停止引擎失败: {e}")
            finally:
                self.engine = None

    def _print_status(self):
        """打印状态"""
        if not self.engine:
            return

        try:
            account = self.engine.get_account()
            positions = self.engine.get_positions()

            if account:
                logger.info(f"账户状态 | 权益: {account.balance:,.2f} | 可用: {account.available:,.2f} | 持仓: {len(positions)}个")

            for pos in positions:
                logger.info(f"  持仓: {pos.symbol} {'多' if pos.direction.value == 'long' else '空'} {pos.volume}手 浮盈: {pos.unrealized_pnl:+,.2f}")

        except Exception as e:
            logger.debug(f"获取状态失败: {e}")

    def _generate_daily_report(self):
        """生成日报"""
        if not self.engine:
            return

        try:
            stats = self.engine.get_statistics()
            trades = self.engine.get_trades()

            logger.info("=" * 50)
            logger.info("每日交易报告")
            logger.info("=" * 50)

            if 'account' in stats:
                acc = stats['account']
                logger.info(f"期末权益: {acc.get('balance', 0):,.2f}")
                logger.info(f"今日盈亏: {acc.get('daily_pnl', 0):+,.2f}")

            logger.info(f"今日成交: {len(trades)}笔")

            for trade in trades[-10:]:  # 最近10笔
                logger.info(f"  {trade.symbol} {trade.direction.value} {trade.volume}@{trade.price}")

            logger.info("=" * 50)

        except Exception as e:
            logger.error(f"生成日报失败: {e}")

    def stop(self):
        """停止服务"""
        logger.info("正在停止服务...")
        self.running = False
        self._stop_engine()

        if self.start_time:
            runtime = datetime.now() - self.start_time
            logger.info(f"服务运行时长: {runtime}")

        logger.info("服务已停止")


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Docker模拟盘服务")
    parser.add_argument("--config", default="tq_config.json", help="配置文件")
    parser.add_argument("--web", action="store_true", help="同时启动Web界面")
    parser.add_argument("--web-port", type=int, default=8504, help="Web端口")

    args = parser.parse_args()

    # 如果需要Web界面，在后台启动
    if args.web:
        import subprocess
        import threading

        def run_web():
            subprocess.run([
                "streamlit", "run", "app/main.py",
                "--server.port", str(args.web_port),
                "--server.address", "0.0.0.0",
                "--server.headless", "true"
            ])

        web_thread = threading.Thread(target=run_web, daemon=True)
        web_thread.start()
        logger.info(f"Web界面已启动: http://0.0.0.0:{args.web_port}")

    # 启动交易服务
    service = TradingService(args.config)
    service.start()


if __name__ == "__main__":
    main()
