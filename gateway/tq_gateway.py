# coding=utf-8
"""
天勤TqSdk网关
支持实盘交易和模拟交易
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional, Callable
from collections import defaultdict
import logging
import threading
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.base import (
    TickData, BarData, OrderRequest, Order, Trade, Position, Account,
    Direction, Offset, OrderStatus, OrderType, EventType, Event
)
from core.event_engine import EventEngine
from gateway.base_gateway import BaseGateway
from utils.rate_limiter import GatewayRateLimiter

logger = logging.getLogger(__name__)

# 延迟导入订单重试处理器
_retry_handler = None
def _get_order_retry_handler():
    global _retry_handler
    if _retry_handler is None:
        try:
            from utils.order_retry import get_order_retry_handler, OrderRetryConfig
            config = OrderRetryConfig(
                max_retries=3,
                retry_on_limit_price=True,
                use_limit_price=True,
                price_offset_ticks=0
            )
            _retry_handler = get_order_retry_handler(config)
        except ImportError:
            _retry_handler = None
    return _retry_handler

# 延迟导入订单号生成器
_order_id_gen = None
def _get_order_id_generator():
    global _order_id_gen
    if _order_id_gen is None:
        try:
            from utils.order_id import get_order_id_generator
            _order_id_gen = get_order_id_generator()
        except ImportError:
            _order_id_gen = None
    return _order_id_gen

# TqSdk映射
EXCHANGE_MAP = {
    "CFFEX": "CFFEX",
    "SHFE": "SHFE",
    "DCE": "DCE",
    "CZCE": "CZCE",
    "INE": "INE",
    "GFEX": "GFEX",
}

# 天勤方向映射
TQ_DIRECTION_MAP = {
    Direction.LONG: "BUY",
    Direction.SHORT: "SELL",
}

TQ_OFFSET_MAP = {
    Offset.OPEN: "OPEN",
    Offset.CLOSE: "CLOSE",
    Offset.CLOSE_TODAY: "CLOSETODAY",
}


class TqGateway(BaseGateway):
    """
    天勤TqSdk网关

    支持:
    1. 实盘交易 (TqAccount)
    2. 模拟交易 (TqSim/TqKq)
    3. 实时行情订阅
    4. K线数据获取
    5. 自动止损/止盈
    """

    def __init__(self, event_engine: EventEngine):
        super().__init__(event_engine, "TQ")

        # TqSdk API
        self.api = None
        self.account = None

        # 配置
        self.tq_user: str = ""
        self.tq_password: str = ""
        self.broker_id: str = ""
        self.td_account: str = ""
        self.td_password: str = ""
        self.sim_mode: bool = True  # 默认模拟盘

        # 订阅管理
        self.quotes: Dict[str, any] = {}  # symbol -> quote对象
        self.klines: Dict[str, Dict[str, any]] = {}  # symbol -> {interval: kline对象}

        # 订单管理
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        self.tq_orders: Dict[str, any] = {}  # order_id -> tq_order

        # 持仓管理
        self.positions: Dict[str, Dict[Direction, Position]] = defaultdict(dict)

        # 成交记录
        self.trades: List[Trade] = []

        # 品种配置
        self.instrument_configs: Dict[str, dict] = {}

        # 计数器
        self._order_count = 0
        self._trade_count = 0

        # 线程锁 - 保护订单操作
        self._lock = threading.Lock()
        self._order_lock = threading.Lock()  # 订单专用锁

        # 数据线程
        self._data_thread: Optional[threading.Thread] = None
        self._running = False

        # 止损单管理
        self.stop_orders: Dict[str, dict] = {}  # order_id -> {symbol, direction, stop_price, volume, tag}

        # 订单同步计数器
        self._sync_counter = 0
        self._sync_interval = 10  # 每10次循环同步一次

        # 流控器
        self.rate_limiter = GatewayRateLimiter()

        # 智能重报配置
        self.enable_auto_retry: bool = True  # 是否启用涨跌停自动重报
        self._pending_retries: Dict[str, dict] = {}  # 待重报的订单信息

    def connect(self, config: dict) -> bool:
        """
        连接天勤

        Args:
            config: {
                'tq_user': '天勤用户名',
                'tq_password': '天勤密码',
                'sim_mode': True,  # True=模拟盘, False=实盘
                'broker_id': '期货公司代码',  # 实盘需要
                'td_account': '期货账号',     # 实盘需要
                'td_password': '期货密码',    # 实盘需要
                'instrument_configs': {}      # 品种配置
            }
        """
        try:
            # 动态导入TqSdk
            try:
                from tqsdk import TqApi, TqAuth, TqSim, TqKq, TqAccount
                from tqsdk.tafunc import ma, ema, atr
            except ImportError:
                self.on_error("TqSdk未安装，请执行: pip install tqsdk")
                return False

            self.tq_user = config.get('tq_user', '')
            self.tq_password = config.get('tq_password', '')
            self.sim_mode = config.get('sim_mode', True)
            self.broker_id = config.get('broker_id', '')
            self.td_account = config.get('td_account', '')
            self.td_password = config.get('td_password', '')

            # 加载品种配置
            if 'instrument_configs' in config:
                self.instrument_configs = config['instrument_configs']

            # 创建认证
            auth = TqAuth(self.tq_user, self.tq_password) if self.tq_user else None

            # 创建账户
            if self.sim_mode:
                # 模拟盘
                initial_balance = config.get('initial_capital', 1000000)
                self.account = TqSim(init_balance=initial_balance)
                self.on_log(f"使用模拟盘, 初始资金: {initial_balance:,.0f}")
            else:
                # 实盘
                if not all([self.broker_id, self.td_account, self.td_password]):
                    self.on_error("实盘模式需要提供期货公司代码、账号和密码")
                    return False
                self.account = TqAccount(self.broker_id, self.td_account, self.td_password)
                self.on_log(f"使用实盘, 期货公司: {self.broker_id}, 账号: {self.td_account}")

            # 创建API
            self.api = TqApi(account=self.account, auth=auth)

            self.connected = True
            self._running = True

            # 启动数据更新线程
            self._data_thread = threading.Thread(target=self._data_loop, daemon=True)
            self._data_thread.start()

            self.on_log("天勤网关连接成功")
            return True

        except Exception as e:
            self.on_error(f"天勤连接失败: {str(e)}")
            return False

    def disconnect(self):
        """断开连接"""
        self._running = False

        if self._data_thread:
            self._data_thread.join(timeout=5)

        if self.api:
            try:
                self.api.close()
            except:
                pass
            self.api = None

        self.connected = False
        self.on_log("天勤网关已断开")

    def _data_loop(self):
        """数据更新循环"""
        while self._running and self.api:
            try:
                # 等待数据更新
                self.api.wait_update()

                # 检查止损单
                self._check_stop_orders()

                # 更新行情
                for symbol, quote in self.quotes.items():
                    if self.api.is_changing(quote):
                        self._on_quote_update(symbol, quote)

                # 更新K线
                for symbol, intervals in self.klines.items():
                    for interval, kline in intervals.items():
                        if self.api.is_changing(kline):
                            self._on_kline_update(symbol, interval, kline)

                # 更新账户
                self._update_account()

                # 更新持仓
                self._update_positions()

                # 更新订单状态
                self._update_orders()

                # 定期订单同步检查
                self._sync_counter += 1
                if self._sync_counter >= self._sync_interval:
                    self._sync_counter = 0
                    self._sync_orders()

            except Exception as e:
                if self._running:
                    logger.error(f"数据更新错误: {e}")
                    time.sleep(1)

    def _to_tq_symbol(self, symbol: str, exchange: str = None) -> str:
        """
        转换为天勤合约代码

        Args:
            symbol: 如 'RB2505' 或 'RB'
            exchange: 交易所代码

        Returns:
            天勤格式: 'SHFE.rb2505'
        """
        # 提取品种和月份
        product = ''.join([c for c in symbol if c.isalpha()]).upper()
        month = ''.join([c for c in symbol if c.isdigit()])

        # 获取交易所
        if not exchange:
            cfg = self.instrument_configs.get(product, {})
            exchange = cfg.get('exchange', 'SHFE')

        # 构建天勤代码
        if month:
            return f"{exchange}.{product.lower()}{month}"
        else:
            # 主力合约
            return f"{exchange}.{product.lower()}@MAIN"

    def _from_tq_symbol(self, tq_symbol: str) -> tuple:
        """
        从天勤代码解析

        Args:
            tq_symbol: 'SHFE.rb2505'

        Returns:
            (symbol, exchange): ('RB2505', 'SHFE')
        """
        parts = tq_symbol.split('.')
        exchange = parts[0]
        symbol = parts[1].upper() if len(parts) > 1 else ''
        return symbol, exchange

    def subscribe(self, symbols: List[str]):
        """
        订阅行情

        Args:
            symbols: 合约代码列表，如 ['RB2505', 'AU2506']
        """
        if not self.api:
            self.on_error("API未初始化")
            return

        for symbol in symbols:
            if symbol in self.subscribed_symbols:
                continue

            try:
                tq_symbol = self._to_tq_symbol(symbol)

                # 订阅Quote
                quote = self.api.get_quote(tq_symbol)
                self.quotes[symbol] = quote

                # 订阅常用K线周期
                self.klines[symbol] = {}
                for interval, seconds in [('1m', 60), ('5m', 300), ('15m', 900),
                                          ('1h', 3600), ('4h', 14400), ('1d', 86400)]:
                    kline = self.api.get_kline_serial(tq_symbol, seconds, 200)
                    self.klines[symbol][interval] = kline

                self.subscribed_symbols.append(symbol)
                self.on_log(f"订阅行情: {symbol} -> {tq_symbol}")

            except Exception as e:
                self.on_error(f"订阅失败 {symbol}: {e}")

    def _on_quote_update(self, symbol: str, quote):
        """处理行情更新"""
        try:
            tick = TickData(
                symbol=symbol,
                exchange=quote.exchange_id if hasattr(quote, 'exchange_id') else '',
                datetime=datetime.fromtimestamp(quote.datetime / 1e9) if quote.datetime else datetime.now(),
                last_price=quote.last_price,
                volume=int(quote.volume),
                turnover=quote.amount if hasattr(quote, 'amount') else 0,
                open_interest=quote.open_interest if hasattr(quote, 'open_interest') else 0,
                bid_price_1=quote.bid_price1,
                bid_volume_1=int(quote.bid_volume1),
                ask_price_1=quote.ask_price1,
                ask_volume_1=int(quote.ask_volume1),
                high_price=quote.highest,
                low_price=quote.lowest,
                open_price=quote.open,
                pre_close=quote.pre_close,
                upper_limit=quote.upper_limit,
                lower_limit=quote.lower_limit
            )
            self.on_tick(tick)
        except Exception as e:
            logger.debug(f"行情处理错误 {symbol}: {e}")

    def _on_kline_update(self, symbol: str, interval: str, kline):
        """处理K线更新"""
        try:
            # 获取最新一根K线
            if len(kline) > 0:
                last_idx = len(kline) - 1
                bar = BarData(
                    symbol=symbol,
                    exchange='',
                    datetime=datetime.fromtimestamp(kline.iloc[last_idx]['datetime'] / 1e9),
                    interval=interval,
                    open=float(kline.iloc[last_idx]['open']),
                    high=float(kline.iloc[last_idx]['high']),
                    low=float(kline.iloc[last_idx]['low']),
                    close=float(kline.iloc[last_idx]['close']),
                    volume=float(kline.iloc[last_idx]['volume']),
                    turnover=float(kline.iloc[last_idx].get('amount', 0)),
                    open_interest=float(kline.iloc[last_idx].get('close_oi', 0))
                )
                self.on_bar(bar)
        except Exception as e:
            logger.debug(f"K线处理错误 {symbol}: {e}")

    def set_instrument_config(self, symbol: str, config: dict):
        """设置品种配置"""
        self.instrument_configs[symbol] = config

    def get_instrument_config(self, symbol: str) -> dict:
        """获取品种配置"""
        product = ''.join([c for c in symbol if c.isalpha()]).upper()
        return self.instrument_configs.get(product, self.instrument_configs.get(symbol, {
            'multiplier': 10,
            'margin_rate': 0.1,
            'commission_rate': 0.0001,
        }))

    def _generate_order_id(self, strategy_name: str = "", signal_id: str = "") -> str:
        """
        生成订单ID

        Args:
            strategy_name: 策略名称（用于生成有意义的订单号）
            signal_id: 关联的信号/交易ID

        Returns:
            订单号
        """
        gen = _get_order_id_generator()
        if gen and strategy_name:
            # 使用新的订单号生成器
            return gen.generate(strategy_name, signal_id)

        # 降级：使用原有格式
        with self._lock:
            self._order_count += 1
            return f"TQ_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._order_count:06d}"

    def _generate_trade_id(self) -> str:
        """生成成交ID"""
        with self._lock:
            self._trade_count += 1
            return f"TRD_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._trade_count:06d}"

    def send_order(self, request: OrderRequest) -> str:
        """
        发送订单（线程安全，带流控）

        Args:
            request: 订单请求

        Returns:
            订单ID
        """
        if not self.api or not self.connected:
            self.on_error("网关未连接")
            return ""

        # 流控 - 下单使用order限流器
        if not self.rate_limiter.acquire_for_order():
            self.on_error("下单被流控拒绝，请稍后重试")
            return ""

        try:
            from tqsdk import tafunc

            order_id = self._generate_order_id(request.strategy_name, request.signal_id)
            tq_symbol = self._to_tq_symbol(request.symbol, request.exchange)

            # 转换方向和开平
            tq_direction = TQ_DIRECTION_MAP.get(request.direction, "BUY")
            tq_offset = TQ_OFFSET_MAP.get(request.offset, "OPEN")

            # 下单
            if request.order_type == OrderType.MARKET:
                # 市价单
                tq_order = self.api.insert_order(
                    symbol=tq_symbol,
                    direction=tq_direction,
                    offset=tq_offset,
                    volume=request.volume
                )
            else:
                # 限价单
                tq_order = self.api.insert_order(
                    symbol=tq_symbol,
                    direction=tq_direction,
                    offset=tq_offset,
                    volume=request.volume,
                    limit_price=request.price
                )

            # 创建订单对象
            order = Order(
                order_id=order_id,
                symbol=request.symbol,
                exchange=request.exchange,
                direction=request.direction,
                offset=request.offset,
                order_type=request.order_type,
                price=request.price,
                volume=request.volume,
                status=OrderStatus.SUBMITTED,
                strategy_name=request.strategy_name,
                signal_id=request.signal_id,
                create_time=datetime.now()
            )

            # 线程安全地更新订单存储
            with self._order_lock:
                self.orders[order_id] = order
                self.active_orders[order_id] = order
                self.tq_orders[order_id] = tq_order

            self.on_order(order)
            self.on_log(f"订单已提交: {order_id} {request.direction.value} {request.offset.value} "
                       f"{request.symbol} {request.volume}@{request.price}")

            return order_id

        except Exception as e:
            self.on_error(f"下单失败: {e}")
            return ""

    def cancel_order(self, order_id: str) -> bool:
        """撤销订单（线程安全）"""
        with self._order_lock:
            if order_id not in self.tq_orders:
                self.on_error(f"订单不存在: {order_id}")
                return False

            try:
                tq_order = self.tq_orders[order_id]
                self.api.cancel_order(tq_order)

                if order_id in self.active_orders:
                    order = self.active_orders[order_id]
                    order.status = OrderStatus.CANCELLED
                    order.update_time = datetime.now()
                    del self.active_orders[order_id]
                    self.on_order(order)

                self.on_log(f"订单已撤销: {order_id}")
                return True

            except Exception as e:
                self.on_error(f"撤单失败: {e}")
                return False

    def set_stop_loss(self, order_id: str, symbol: str, direction: Direction,
                      stop_price: float, volume: int, tag: str = "stop_loss"):
        """
        设置止损单

        Args:
            order_id: 关联订单ID
            symbol: 品种
            direction: 持仓方向（止损时反向平仓）
            stop_price: 止损价格
            volume: 止损数量
            tag: 标签
        """
        self.stop_orders[order_id] = {
            'symbol': symbol,
            'direction': direction,
            'stop_price': stop_price,
            'volume': volume,
            'tag': tag,
            'triggered': False
        }
        self.on_log(f"设置止损: {symbol} {direction.value} @ {stop_price} x{volume}")

    def cancel_stop_loss(self, order_id: str):
        """取消止损单"""
        if order_id in self.stop_orders:
            del self.stop_orders[order_id]
            self.on_log(f"取消止损: {order_id}")

    def _check_stop_orders(self):
        """检查止损单触发"""
        to_trigger = []

        for order_id, stop_info in self.stop_orders.items():
            if stop_info['triggered']:
                continue

            symbol = stop_info['symbol']
            if symbol not in self.quotes:
                continue

            quote = self.quotes[symbol]
            current_price = quote.last_price

            direction = stop_info['direction']
            stop_price = stop_info['stop_price']

            # 检查是否触发
            triggered = False
            if direction == Direction.LONG:
                # 多头持仓，价格跌破止损价
                if current_price <= stop_price:
                    triggered = True
            else:
                # 空头持仓，价格涨破止损价
                if current_price >= stop_price:
                    triggered = True

            if triggered:
                to_trigger.append((order_id, stop_info, current_price))

        # 执行止损
        for order_id, stop_info, trigger_price in to_trigger:
            stop_info['triggered'] = True

            # 平仓方向与持仓方向相反
            close_direction = Direction.SHORT if stop_info['direction'] == Direction.LONG else Direction.LONG

            request = OrderRequest(
                symbol=stop_info['symbol'],
                exchange='',
                direction=close_direction,
                offset=Offset.CLOSE,
                order_type=OrderType.MARKET,
                price=trigger_price,
                volume=stop_info['volume'],
                strategy_name=f"stop_{stop_info['tag']}",
                signal_id=order_id
            )

            self.on_log(f"止损触发: {stop_info['symbol']} @ {trigger_price} ({stop_info['tag']})")
            self.send_order(request)

            # 移除止损单
            del self.stop_orders[order_id]

    def _update_account(self):
        """更新账户信息"""
        if not self.api:
            return

        try:
            tq_account = self.api.get_account()

            account = Account(
                account_id="TQ_ACCOUNT",
                balance=tq_account.balance,
                available=tq_account.available,
                margin=tq_account.margin,
                unrealized_pnl=tq_account.float_profit,
                realized_pnl=tq_account.close_profit,
                commission=tq_account.commission,
                pre_balance=tq_account.pre_balance,
                risk_ratio=tq_account.risk_ratio if hasattr(tq_account, 'risk_ratio') else 0,
                update_time=datetime.now()
            )

            self.on_account(account)

        except Exception as e:
            logger.debug(f"账户更新错误: {e}")

    def _update_positions(self):
        """更新持仓"""
        if not self.api:
            return

        try:
            for symbol in self.subscribed_symbols:
                tq_symbol = self._to_tq_symbol(symbol)
                tq_pos = self.api.get_position(tq_symbol)

                # 多头持仓
                if tq_pos.pos_long > 0:
                    pos = Position(
                        symbol=symbol,
                        exchange='',
                        direction=Direction.LONG,
                        volume=tq_pos.pos_long,
                        avg_price=tq_pos.open_price_long,
                        unrealized_pnl=tq_pos.float_profit_long,
                        margin=tq_pos.margin_long if hasattr(tq_pos, 'margin_long') else 0
                    )
                    self.positions[symbol][Direction.LONG] = pos
                    self.on_position(pos)

                # 空头持仓
                if tq_pos.pos_short > 0:
                    pos = Position(
                        symbol=symbol,
                        exchange='',
                        direction=Direction.SHORT,
                        volume=tq_pos.pos_short,
                        avg_price=tq_pos.open_price_short,
                        unrealized_pnl=tq_pos.float_profit_short,
                        margin=tq_pos.margin_short if hasattr(tq_pos, 'margin_short') else 0
                    )
                    self.positions[symbol][Direction.SHORT] = pos
                    self.on_position(pos)

        except Exception as e:
            logger.debug(f"持仓更新错误: {e}")

    def _update_orders(self):
        """更新订单状态（线程安全，支持智能重报）"""
        with self._order_lock:
            orders_to_process = list(self.tq_orders.items())

        for order_id, tq_order in orders_to_process:
            with self._order_lock:
                if order_id not in self.orders:
                    continue
                order = self.orders[order_id]

            try:
                # 检查订单状态
                if tq_order.status == "FINISHED":
                    # 获取拒绝/错误消息
                    reject_msg = getattr(tq_order, 'last_msg', '') or ''

                    if tq_order.volume_left == 0:
                        # 全部成交
                        order.status = OrderStatus.FILLED
                        order.filled_volume = tq_order.volume_orign
                        order.avg_price = tq_order.trade_price if hasattr(tq_order, 'trade_price') else order.price

                        # 标记重试成功
                        retry_handler = _get_order_retry_handler()
                        if retry_handler:
                            retry_handler.record_success(order_id)

                    elif self._is_rejected_order(reject_msg):
                        # 订单被拒绝
                        order.status = OrderStatus.REJECTED
                        order.error_msg = reject_msg

                        # 尝试智能重报
                        if self.enable_auto_retry:
                            self._handle_rejected_order(order, reject_msg)
                    else:
                        # 普通撤销
                        order.status = OrderStatus.CANCELLED

                    order.update_time = datetime.now()

                    with self._order_lock:
                        if order_id in self.active_orders:
                            del self.active_orders[order_id]

                    self.on_order(order)

                    # 如果成交，创建成交记录
                    if order.status == OrderStatus.FILLED:
                        trade = Trade(
                            trade_id=self._generate_trade_id(),
                            order_id=order_id,
                            symbol=order.symbol,
                            exchange=order.exchange,
                            direction=order.direction,
                            offset=order.offset,
                            price=order.avg_price,
                            volume=order.filled_volume,
                            commission=0,  # 天勤账户会自动计算
                            trade_time=datetime.now(),
                            strategy_name=order.strategy_name
                        )
                        with self._order_lock:
                            self.trades.append(trade)
                        self.on_trade(trade)

            except Exception as e:
                logger.debug(f"订单状态更新错误: {e}")

    def _is_rejected_order(self, msg: str) -> bool:
        """判断是否为被拒绝的订单"""
        if not msg:
            return False
        reject_keywords = ['拒绝', 'reject', '失败', 'fail', '错误', 'error',
                          '涨停', '跌停', 'limit', '超出', '无效']
        msg_lower = msg.lower()
        return any(kw in msg_lower for kw in reject_keywords)

    def _handle_rejected_order(self, order: Order, reject_msg: str):
        """
        处理被拒绝的订单，尝试智能重报

        Args:
            order: 被拒绝的订单
            reject_msg: 拒绝原因
        """
        retry_handler = _get_order_retry_handler()
        if not retry_handler:
            return

        # 判断是否应该重试
        should_retry, reason, msg = retry_handler.should_retry(order.order_id, reject_msg)

        if not should_retry:
            logger.info(f"[订单重报] 跳过 {order.order_id}: {msg}")
            retry_handler.record_final_fail(order.order_id, msg)
            return

        # 获取涨跌停价格
        limit_up, limit_down = self._get_limit_prices(order.symbol)
        if limit_up <= 0 or limit_down <= 0:
            logger.warning(f"[订单重报] 无法获取 {order.symbol} 涨跌停价格")
            return

        # 获取最小变动价位
        config = self.get_instrument_config(order.symbol)
        price_tick = config.get('price_tick', 1.0)

        # 计算调整后的价格
        adjusted_price = retry_handler.calculate_adjusted_price(
            original_price=order.price,
            direction=order.direction.value,
            limit_up=limit_up,
            limit_down=limit_down,
            price_tick=price_tick,
            reason=reason
        )

        # 如果价格没有变化，不重报
        if abs(adjusted_price - order.price) < price_tick * 0.5:
            logger.info(f"[订单重报] 价格无需调整 {order.order_id}")
            return

        # 创建新订单请求
        new_request = OrderRequest(
            symbol=order.symbol,
            exchange=order.exchange,
            direction=order.direction,
            offset=order.offset,
            order_type=order.order_type,
            price=adjusted_price,
            volume=order.volume - order.filled_volume,  # 剩余数量
            strategy_name=order.strategy_name,
            signal_id=order.signal_id
        )

        # 发送新订单
        new_order_id = self.send_order(new_request)

        if new_order_id:
            # 记录重试
            retry_handler.record_retry(
                original_order_id=order.order_id,
                new_order_id=new_order_id,
                original_price=order.price,
                adjusted_price=adjusted_price,
                reason=reason
            )
            self.on_log(f"[智能重报] {order.symbol} 价格 {order.price:.2f} -> {adjusted_price:.2f}")
        else:
            retry_handler.record_final_fail(order.order_id, "重报下单失败")

    def _get_limit_prices(self, symbol: str) -> tuple:
        """
        获取品种涨跌停价格

        Args:
            symbol: 品种代码

        Returns:
            (涨停价, 跌停价)
        """
        # 从行情中获取
        if symbol in self.quotes:
            quote = self.quotes[symbol]
            limit_up = getattr(quote, 'upper_limit', 0) or 0
            limit_down = getattr(quote, 'lower_limit', 0) or 0
            if limit_up > 0 and limit_down > 0:
                return limit_up, limit_down

        # 从缓存管理器获取
        try:
            from utils.limit_price import get_limit_price_manager
            manager = get_limit_price_manager()
            info = manager.get_limit_price(symbol)
            if info:
                return info.limit_up, info.limit_down
        except ImportError:
            pass

        return 0, 0

    def _sync_orders(self):
        """
        订单同步检查 - 确保本地订单和TqSdk订单一致

        定期检查并清理已完成但未正确处理的订单
        """
        with self._order_lock:
            # 清理已完成的订单（从active_orders中）
            completed_ids = []
            for order_id, order in self.active_orders.items():
                if order.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
                    completed_ids.append(order_id)

            for order_id in completed_ids:
                del self.active_orders[order_id]
                logger.debug(f"[订单同步] 清理已完成订单: {order_id}")

            # 检查tq_orders中已完成的订单
            tq_completed = []
            for order_id, tq_order in self.tq_orders.items():
                try:
                    if tq_order.status == "FINISHED":
                        if order_id in self.active_orders:
                            # 本地还是活动状态但TqSdk已完成，需要同步
                            order = self.active_orders[order_id]
                            if tq_order.volume_left == 0:
                                order.status = OrderStatus.FILLED
                            else:
                                order.status = OrderStatus.CANCELLED
                            order.update_time = datetime.now()
                            tq_completed.append(order_id)
                            logger.warning(f"[订单同步] 修复订单状态漂移: {order_id}")
                except:
                    pass

            for order_id in tq_completed:
                if order_id in self.active_orders:
                    del self.active_orders[order_id]

    def query_account(self) -> Optional[Account]:
        """查询账户（带流控）"""
        if not self.api:
            return None

        # 流控
        if not self.rate_limiter.acquire_for_query():
            logger.warning("查询账户被流控拒绝")
            return None

        try:
            tq_account = self.api.get_account()
            return Account(
                account_id="TQ_ACCOUNT",
                balance=tq_account.balance,
                available=tq_account.available,
                margin=tq_account.margin,
                unrealized_pnl=tq_account.float_profit,
                realized_pnl=tq_account.close_profit,
                commission=tq_account.commission,
                pre_balance=tq_account.pre_balance,
                update_time=datetime.now()
            )
        except:
            return None

    def query_positions(self) -> List[Position]:
        """查询持仓"""
        result = []
        for symbol_positions in self.positions.values():
            for pos in symbol_positions.values():
                if pos.volume > 0:
                    result.append(pos)
        return result

    def query_orders(self) -> List[Order]:
        """查询订单"""
        return list(self.orders.values())

    def query_active_orders(self) -> List[Order]:
        """查询活动订单"""
        return list(self.active_orders.values())

    def get_kline_data(self, symbol: str, interval: str = '1h', count: int = 200) -> List[dict]:
        """
        获取K线数据

        Args:
            symbol: 品种代码
            interval: 周期
            count: 数量

        Returns:
            K线数据列表
        """
        if symbol not in self.klines or interval not in self.klines[symbol]:
            return []

        kline = self.klines[symbol][interval]
        result = []

        try:
            for i in range(max(0, len(kline) - count), len(kline)):
                result.append({
                    'datetime': datetime.fromtimestamp(kline.iloc[i]['datetime'] / 1e9),
                    'open': float(kline.iloc[i]['open']),
                    'high': float(kline.iloc[i]['high']),
                    'low': float(kline.iloc[i]['low']),
                    'close': float(kline.iloc[i]['close']),
                    'volume': float(kline.iloc[i]['volume'])
                })
        except:
            pass

        return result

    def get_quote(self, symbol: str) -> Optional[dict]:
        """
        获取最新行情

        Args:
            symbol: 品种代码

        Returns:
            行情数据
        """
        if symbol not in self.quotes:
            return None

        quote = self.quotes[symbol]
        try:
            return {
                'symbol': symbol,
                'last_price': quote.last_price,
                'bid_price': quote.bid_price1,
                'ask_price': quote.ask_price1,
                'high': quote.highest,
                'low': quote.lowest,
                'open': quote.open,
                'volume': quote.volume,
                'datetime': datetime.fromtimestamp(quote.datetime / 1e9) if quote.datetime else None
            }
        except:
            return None
