# 期货回测系统升级改造计划

**目标**：将现有回测系统升级为支持模拟盘/实盘的完整交易系统
**参考**：D:\all code\trader-master\trader-master（专业级CTP交易框架）
**当前项目**：D:\期货\回测改造

---

## 一、现状评估

### 1.1 现有系统能力

| 模块 | 完成度 | 说明 |
|------|--------|------|
| 策略框架 | 90% | 20+策略、参数化、自动发现 |
| 数据管理 | 85% | 70+品种、多源、SQLite |
| 回测引擎 | 70% | 多空支持、风险管理、统计 |
| Web界面 | 80% | Streamlit、可视化完整 |
| **实盘能力** | **0%** | 完全缺失 |

### 1.2 trader-master核心设计借鉴

```
trader-master 架构
├── 异步事件驱动 (asyncio + Redis Pub/Sub)
├── Django ORM 数据模型 (Strategy/Signal/Order/Trade)
├── CTP接口分离 (独立进程 + 消息队列)
├── 定时任务调度 (@RegisterCallback + crontab)
├── 多层风控 (仓位/保证金/ATR止损/风险度)
└── 完整交易记录 (订单/成交/持仓/绩效)
```

---

## 二、升级架构设计

### 2.1 目标架构

```
期货交易系统 v2.0
├── 核心层 (Core)
│   ├── EventEngine           # 事件驱动引擎（新增）
│   ├── DataManager           # 数据管理（升级）
│   ├── BacktestEngine        # 回测引擎（保留）
│   └── LiveEngine            # 实盘引擎（新增）
│
├── 策略层 (Strategy)
│   ├── BaseStrategy          # 策略基类（升级）
│   ├── StrategyManager       # 策略管理器（新增）
│   └── strategies/*.py       # 具体策略（保留）
│
├── 交易层 (Trading)
│   ├── OrderManager          # 订单管理器（新增）
│   ├── PositionManager       # 持仓管理器（新增）
│   ├── RiskManager           # 风控管理器（新增）
│   └── AccountManager        # 账户管理器（新增）
│
├── 接口层 (Gateway)
│   ├── SimGateway            # 模拟盘接口（新增）
│   ├── CtpGateway            # CTP期货接口（新增）
│   └── BaseGateway           # 接口基类（新增）
│
├── 数据层 (Data)
│   ├── BarGenerator          # K线合成器（新增）
│   ├── DataFeed              # 数据订阅（新增）
│   └── DatabaseManager       # 数据库管理（升级）
│
└── 应用层 (App)
    ├── Streamlit Web         # Web界面（升级）
    ├── CLI Runner            # 命令行运行（新增）
    └── Scheduler             # 定时任务（新增）
```

### 2.2 运行模式

| 模式 | 数据源 | 执行方式 | 用途 |
|------|--------|----------|------|
| **回测模式** | 历史数据库 | 批量回放 | 策略验证 |
| **模拟盘** | 实时行情 | 虚拟下单 | 策略验证 |
| **实盘模式** | 实时行情 | CTP下单 | 真实交易 |

---

## 三、模块详细设计

### 3.1 事件驱动引擎 (EventEngine)

**文件**: `core/event_engine.py`

```python
# 事件类型
class EventType(Enum):
    TICK = "tick"              # Tick数据
    BAR = "bar"                # K线数据
    SIGNAL = "signal"          # 交易信号
    ORDER = "order"            # 订单事件
    TRADE = "trade"            # 成交事件
    POSITION = "position"      # 持仓变化
    ACCOUNT = "account"        # 账户变化
    LOG = "log"                # 日志事件
    TIMER = "timer"            # 定时器

class EventEngine:
    def __init__(self):
        self._queue = Queue()
        self._handlers = defaultdict(list)
        self._thread = None
        self._active = False

    def register(self, event_type: EventType, handler: Callable):
        """注册事件处理器"""

    def put(self, event: Event):
        """发送事件到队列"""

    def start(self):
        """启动事件循环"""

    def stop(self):
        """停止事件循环"""
```

**借鉴trader-master**:
- `BaseModule._msg_reader()` 消息循环
- `CallbackFunctionContainer` 回调注册机制

### 3.2 交易网关 (Gateway)

**文件**: `gateway/base_gateway.py`, `gateway/sim_gateway.py`, `gateway/ctp_gateway.py`

```python
class BaseGateway(ABC):
    """交易网关基类"""

    @abstractmethod
    def connect(self, config: dict):
        """连接"""

    @abstractmethod
    def subscribe(self, symbols: List[str]):
        """订阅行情"""

    @abstractmethod
    def send_order(self, order: OrderRequest) -> str:
        """发送订单，返回order_id"""

    @abstractmethod
    def cancel_order(self, order_id: str):
        """撤单"""

    @abstractmethod
    def query_account(self) -> AccountData:
        """查询账户"""

    @abstractmethod
    def query_position(self) -> List[PositionData]:
        """查询持仓"""


class SimGateway(BaseGateway):
    """模拟盘网关"""

    def __init__(self, event_engine: EventEngine):
        self.event_engine = event_engine
        self.account = SimAccount(initial_capital=100000)
        self.positions = {}
        self.orders = {}

    def on_bar(self, bar: BarData):
        """收到K线时模拟撮合"""
        for order in self.pending_orders:
            if self._can_fill(order, bar):
                self._fill_order(order, bar)

    def _can_fill(self, order: OrderRequest, bar: BarData) -> bool:
        """判断是否可以成交"""
        if order.direction == Direction.LONG:
            return bar.low <= order.price <= bar.high
        else:
            return bar.low <= order.price <= bar.high


class CtpGateway(BaseGateway):
    """CTP期货网关（参考trader-master实现）"""

    def __init__(self, event_engine: EventEngine):
        self.event_engine = event_engine
        self.redis_client = None  # Redis消息队列

    def connect(self, config: dict):
        """连接CTP"""
        # 启动Redis订阅
        # 发送登录请求

    def send_order(self, order: OrderRequest) -> str:
        """通过Redis发送订单到CTP"""
        param = {
            'InstrumentID': order.symbol,
            'Direction': order.direction.value,
            'VolumeTotalOriginal': order.volume,
            'LimitPrice': order.price,
            # ...
        }
        self.redis_client.publish('MSG:CTP:REQ:ReqOrderInsert', json.dumps(param))
```

**借鉴trader-master**:
- `brother2.py: ReqOrderInsert()` 订单发送
- `brother2.py: OnRtnTrade()` 成交回调
- Redis Pub/Sub通信机制

### 3.3 订单管理器 (OrderManager)

**文件**: `trading/order_manager.py`

```python
class OrderStatus(Enum):
    PENDING = "pending"         # 待发送
    SUBMITTED = "submitted"     # 已提交
    PARTIAL = "partial"         # 部分成交
    FILLED = "filled"           # 全部成交
    CANCELLED = "cancelled"     # 已撤单
    REJECTED = "rejected"       # 被拒绝

@dataclass
class Order:
    order_id: str
    symbol: str
    direction: Direction
    offset: Offset
    price: float
    volume: int
    status: OrderStatus
    filled_volume: int = 0
    avg_price: float = 0.0
    create_time: datetime = None
    update_time: datetime = None
    signal_id: str = None       # 关联的信号ID
    strategy_name: str = None

class OrderManager:
    def __init__(self, event_engine: EventEngine, gateway: BaseGateway):
        self.event_engine = event_engine
        self.gateway = gateway
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}

    def send_order(self, signal: Signal) -> Order:
        """根据信号创建并发送订单"""

    def cancel_order(self, order_id: str):
        """撤单"""

    def on_order(self, order_data: dict):
        """订单状态更新回调"""

    def on_trade(self, trade_data: dict):
        """成交回调"""
```

**借鉴trader-master**:
- `panel/models.py: Order` 模型
- `brother2.py: OnRtnOrder()` 订单状态处理

### 3.4 持仓管理器 (PositionManager)

**文件**: `trading/position_manager.py`

```python
@dataclass
class Position:
    symbol: str
    direction: Direction
    volume: int
    frozen: int = 0              # 冻结数量
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    margin: float = 0.0

    # 止损追踪
    highest_price: float = 0.0
    lowest_price: float = 0.0
    entry_time: datetime = None

class PositionManager:
    def __init__(self, event_engine: EventEngine):
        self.positions: Dict[str, Position] = {}

    def update_on_trade(self, trade: Trade):
        """成交后更新持仓"""

    def update_on_tick(self, tick: TickData):
        """Tick更新持仓盈亏"""

    def get_position(self, symbol: str) -> Optional[Position]:
        """获取持仓"""

    def get_available(self, symbol: str, direction: Direction) -> int:
        """获取可平仓数量"""
```

**借鉴trader-master**:
- `panel/models.py: Trade` 模型
- `brother2.py: refresh_position()` 持仓刷新

### 3.5 风控管理器 (RiskManager)

**文件**: `trading/risk_manager.py`

```python
@dataclass
class RiskConfig:
    max_position_per_symbol: int = 10       # 单品种最大持仓
    max_order_per_symbol: int = 5           # 单品种最大挂单
    max_margin_ratio: float = 0.8           # 最大保证金占用
    max_daily_loss: float = 0.05            # 日最大亏损比例
    max_drawdown: float = 0.15              # 最大回撤
    stop_loss_atr_mult: float = 3.0         # ATR止损倍数

class RiskManager:
    def __init__(self, config: RiskConfig):
        self.config = config
        self.daily_pnl = 0.0
        self.peak_equity = 0.0

    def check_order(self, order: OrderRequest, account: Account) -> Tuple[bool, str]:
        """订单风控检查"""
        # 1. 保证金检查
        if self._margin_check(order, account):
            return False, "保证金不足"

        # 2. 持仓限制检查
        if self._position_limit_check(order):
            return False, "超过持仓限制"

        # 3. 日亏损限制检查
        if self._daily_loss_check(account):
            return False, "触发日亏损限制"

        return True, ""

    def check_stop_loss(self, position: Position, bar: BarData) -> bool:
        """检查是否触发止损"""
        if position.direction == Direction.LONG:
            stop_price = position.highest_price - self.atr * self.config.stop_loss_atr_mult
            return bar.close < stop_price
        else:
            stop_price = position.lowest_price + self.atr * self.config.stop_loss_atr_mult
            return bar.close > stop_price
```

**借鉴trader-master**:
- `brother2.py: calc_signal()` 中的风险检查
- 保证金占用80%告警逻辑
- ATR动态止损逻辑

### 3.6 策略基类升级 (BaseStrategy)

**文件**: `strategies/base.py` (升级)

```python
class BaseStrategy(ABC):
    """升级后的策略基类 - 同时支持回测和实盘"""

    name: str = ""
    display_name: str = ""
    version: str = "1.0"
    warmup_num: int = 60

    def __init__(self, params: dict = None):
        self.params = params or {}
        self.position = 0
        self.positions: Dict[str, Position] = {}  # 多品种持仓

        # 实盘模式需要
        self.event_engine: EventEngine = None
        self.order_manager: OrderManager = None
        self.is_live: bool = False

    def set_live_mode(self, event_engine: EventEngine, order_manager: OrderManager):
        """设置实盘模式"""
        self.event_engine = event_engine
        self.order_manager = order_manager
        self.is_live = True

    @abstractmethod
    def on_bar(self, idx: int, df: pd.DataFrame, capital: float) -> Optional[Signal]:
        """回测模式：逐K线调用"""
        pass

    def on_bar_live(self, bar: BarData):
        """实盘模式：收到K线时调用"""
        signal = self._generate_signal(bar)
        if signal:
            self._send_signal(signal)

    def _send_signal(self, signal: Signal):
        """发送信号"""
        if self.is_live:
            # 实盘模式：通过OrderManager下单
            self.order_manager.send_order(signal)
        else:
            # 回测模式：返回信号
            return signal
```

### 3.7 实盘引擎 (LiveEngine)

**文件**: `core/live_engine.py`

```python
class LiveEngine:
    """实盘交易引擎"""

    def __init__(self, config: dict):
        # 核心组件
        self.event_engine = EventEngine()
        self.gateway = self._create_gateway(config['gateway'])

        # 交易组件
        self.order_manager = OrderManager(self.event_engine, self.gateway)
        self.position_manager = PositionManager(self.event_engine)
        self.risk_manager = RiskManager(config['risk'])
        self.account_manager = AccountManager(self.event_engine)

        # 策略组件
        self.strategies: Dict[str, BaseStrategy] = {}
        self.bar_generators: Dict[str, BarGenerator] = {}

        # 定时任务
        self.scheduler = Scheduler()

    def add_strategy(self, strategy: BaseStrategy, symbols: List[str]):
        """添加策略"""
        strategy.set_live_mode(self.event_engine, self.order_manager)
        self.strategies[strategy.name] = strategy

        # 订阅行情
        self.gateway.subscribe(symbols)

    def start(self):
        """启动引擎"""
        self.event_engine.start()
        self.gateway.connect(self.config)

        # 注册定时任务（参考trader-master）
        self._setup_scheduler()

    def _setup_scheduler(self):
        """设置定时任务"""
        # 08:55 处理日盘信号
        self.scheduler.add_job(self._process_day_signals, '08:55')
        # 20:55 处理夜盘信号
        self.scheduler.add_job(self._process_night_signals, '20:55')
        # 17:00 盘后计算
        self.scheduler.add_job(self._after_market, '17:00')

    def _on_bar(self, bar: BarData):
        """K线事件处理"""
        for strategy in self.strategies.values():
            strategy.on_bar_live(bar)
```

**借鉴trader-master**:
- `brother2.py: TradeStrategy` 整体结构
- 定时任务调度 `processing_signal1/2/3`
- 盘后处理 `collect_quote`

---

## 四、数据模型设计

### 4.1 数据库表结构 (SQLite/MySQL)

```sql
-- 策略配置表
CREATE TABLE strategy (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT,
    params TEXT,  -- JSON格式
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- 信号表
CREATE TABLE signal (
    id INTEGER PRIMARY KEY,
    strategy_id INTEGER,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,  -- buy/sell/close
    price REAL,
    volume INTEGER,
    tag TEXT,
    stop_loss REAL,
    trigger_time TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (strategy_id) REFERENCES strategy(id)
);

-- 订单表
CREATE TABLE order (
    id INTEGER PRIMARY KEY,
    order_id TEXT UNIQUE,
    signal_id INTEGER,
    symbol TEXT NOT NULL,
    direction TEXT,  -- long/short
    offset TEXT,     -- open/close
    price REAL,
    volume INTEGER,
    filled_volume INTEGER DEFAULT 0,
    avg_price REAL,
    status TEXT,
    create_time TIMESTAMP,
    update_time TIMESTAMP,
    FOREIGN KEY (signal_id) REFERENCES signal(id)
);

-- 成交表
CREATE TABLE trade (
    id INTEGER PRIMARY KEY,
    order_id TEXT,
    symbol TEXT NOT NULL,
    direction TEXT,
    offset TEXT,
    price REAL,
    volume INTEGER,
    commission REAL,
    trade_time TIMESTAMP,
    FOREIGN KEY (order_id) REFERENCES order(order_id)
);

-- 持仓表
CREATE TABLE position (
    id INTEGER PRIMARY KEY,
    strategy_id INTEGER,
    symbol TEXT NOT NULL,
    direction TEXT,
    volume INTEGER,
    frozen INTEGER DEFAULT 0,
    avg_price REAL,
    unrealized_pnl REAL,
    margin REAL,
    highest_price REAL,
    lowest_price REAL,
    entry_time TIMESTAMP,
    update_time TIMESTAMP,
    UNIQUE(strategy_id, symbol, direction)
);

-- 账户表
CREATE TABLE account (
    id INTEGER PRIMARY KEY,
    balance REAL,
    available REAL,
    margin REAL,
    unrealized_pnl REAL,
    realized_pnl REAL,
    commission REAL,
    update_time TIMESTAMP
);

-- 绩效表
CREATE TABLE performance (
    id INTEGER PRIMARY KEY,
    strategy_id INTEGER,
    date DATE,
    equity REAL,
    daily_pnl REAL,
    daily_return REAL,
    cumulative_return REAL,
    max_drawdown REAL,
    sharpe_ratio REAL,
    UNIQUE(strategy_id, date)
);
```

---

## 五、开发阶段规划

### Phase 1: 基础架构（2周）

| 任务 | 优先级 | 工作量 | 产出 |
|------|--------|--------|------|
| 事件驱动引擎 | P0 | 3天 | core/event_engine.py |
| 数据模型定义 | P0 | 2天 | models/*.py |
| 模拟盘网关 | P0 | 3天 | gateway/sim_gateway.py |
| 订单管理器 | P0 | 2天 | trading/order_manager.py |
| 持仓管理器 | P0 | 2天 | trading/position_manager.py |

**里程碑**: 能在模拟盘运行WaveTrend策略

### Phase 2: 风控与策略升级（1周）

| 任务 | 优先级 | 工作量 | 产出 |
|------|--------|--------|------|
| 风控管理器 | P0 | 2天 | trading/risk_manager.py |
| 策略基类升级 | P0 | 1天 | strategies/base.py |
| WaveTrend策略适配 | P1 | 1天 | 适配实盘接口 |
| 账户管理器 | P1 | 1天 | trading/account_manager.py |

**里程碑**: 完整的风控+策略运行

### Phase 3: 实盘接口（2周）

| 任务 | 优先级 | 工作量 | 产出 |
|------|--------|--------|------|
| CTP接口研究 | P0 | 3天 | 理解trader-master实现 |
| Redis集成 | P0 | 2天 | 消息队列搭建 |
| CTP网关实现 | P0 | 5天 | gateway/ctp_gateway.py |
| 连接测试 | P0 | 2天 | 期货公司模拟账号测试 |

**里程碑**: 能连接期货公司模拟环境

### Phase 4: 定时任务与监控（1周）

| 任务 | 优先级 | 工作量 | 产出 |
|------|--------|--------|------|
| 定时调度器 | P0 | 2天 | core/scheduler.py |
| 盘后处理 | P0 | 1天 | 数据更新、信号计算 |
| 日志系统 | P1 | 1天 | 完善日志记录 |
| 微信/钉钉通知 | P2 | 1天 | 告警推送 |

**里程碑**: 自动化运行

### Phase 5: Web界面升级（1周）

| 任务 | 优先级 | 工作量 | 产出 |
|------|--------|--------|------|
| 实时行情展示 | P1 | 2天 | 行情面板 |
| 持仓监控 | P1 | 1天 | 持仓面板 |
| 订单管理 | P1 | 1天 | 订单面板 |
| 策略控制 | P1 | 1天 | 启停控制 |

**里程碑**: 完整的管理界面

---

## 六、文件结构规划

```
D:\期货\回测改造\
├── core/
│   ├── __init__.py
│   ├── event_engine.py      # 事件驱动引擎
│   ├── backtest_engine.py   # 回测引擎（现有engine.py重命名）
│   ├── live_engine.py       # 实盘引擎
│   └── scheduler.py         # 定时调度
│
├── gateway/
│   ├── __init__.py
│   ├── base_gateway.py      # 网关基类
│   ├── sim_gateway.py       # 模拟盘网关
│   └── ctp_gateway.py       # CTP网关
│
├── trading/
│   ├── __init__.py
│   ├── order_manager.py     # 订单管理
│   ├── position_manager.py  # 持仓管理
│   ├── risk_manager.py      # 风控管理
│   └── account_manager.py   # 账户管理
│
├── models/
│   ├── __init__.py
│   ├── base.py              # 基础数据类
│   ├── order.py             # 订单模型
│   ├── trade.py             # 成交模型
│   ├── position.py          # 持仓模型
│   └── signal.py            # 信号模型
│
├── strategies/              # 现有策略目录
│   ├── base.py              # 升级后的基类
│   ├── wavetrend_final.py   # WaveTrend策略
│   └── ...
│
├── data/
│   ├── data_manager.py      # 现有数据管理
│   ├── bar_generator.py     # K线合成器
│   └── data_feed.py         # 数据订阅
│
├── utils/
│   ├── __init__.py
│   ├── logger.py            # 日志工具
│   ├── notify.py            # 通知推送
│   └── config.py            # 配置管理
│
├── app/
│   ├── streamlit_app.py     # 现有Web应用
│   ├── live_runner.py       # 实盘运行入口
│   └── cli.py               # 命令行工具
│
├── configs/
│   ├── strategy_config.yaml # 策略配置
│   ├── gateway_config.yaml  # 网关配置
│   └── risk_config.yaml     # 风控配置
│
└── tests/
    ├── test_sim_gateway.py
    ├── test_order_manager.py
    └── ...
```

---

## 七、快速启动方案（推荐）

如果想快速验证WaveTrend策略，可以先做一个**轻量级模拟盘**：

### 7.1 简化版模拟盘架构

```
WaveTrend模拟盘 (1周实现)
├── 数据源：天勤实时行情推送
├── 策略：现有WaveTrend策略
├── 信号：生成买卖信号
├── 执行：模拟撮合（无真实下单）
├── 记录：SQLite记录交易
└── 通知：微信推送信号
```

### 7.2 实现步骤

```python
# 1. 安装天勤SDK
pip install tqsdk

# 2. 简化版实时运行脚本
from tqsdk import TqApi, TqAuth
from strategies.wavetrend_final import WaveTrendFinalStrategy

api = TqApi(auth=TqAuth("用户名", "密码"))
strategy = WaveTrendFinalStrategy()
strategy.set_symbol('I')  # 铁矿石

# 订阅60分钟K线
klines = api.get_kline_serial("DCE.i2505", 60*60)

while True:
    api.wait_update()
    if api.is_changing(klines.iloc[-1], "close"):
        # 转换数据格式
        df = convert_klines(klines)
        # 计算信号
        signal = strategy.on_bar(len(df)-1, df, 100000)
        if signal:
            # 记录信号
            save_signal(signal)
            # 推送通知
            send_wechat(f"WaveTrend信号: {signal.action} {signal.tag}")
```

---

## 八、成本与资源估算

### 8.1 开发资源

| 阶段 | 工时 | 说明 |
|------|------|------|
| Phase 1 | 10人天 | 基础架构 |
| Phase 2 | 5人天 | 风控策略 |
| Phase 3 | 10人天 | CTP接口 |
| Phase 4 | 5人天 | 定时监控 |
| Phase 5 | 5人天 | Web升级 |
| **总计** | **35人天** | 约7周 |

### 8.2 基础设施

| 项目 | 费用 | 说明 |
|------|------|------|
| 期货模拟账号 | 免费 | 期货公司提供 |
| 天勤行情 | 免费/付费 | 基础功能免费 |
| 云服务器 | ¥100/月 | 阿里云2核4G |
| Redis | 包含 | 云服务器内 |
| MySQL | 包含 | 或继续用SQLite |

---

## 九、风险与建议

### 9.1 主要风险

1. **CTP接口复杂度**：需要独立的C++程序处理CTP回调
2. **网络稳定性**：实盘需要7x24小时稳定运行
3. **资金安全**：实盘前必须充分测试

### 9.2 建议路径

```
推荐路径：
1. 先做模拟盘验证 (Phase 1-2, 3周)
2. 用天勤SDK获取实时行情
3. 策略信号推送到手机
4. 手动执行交易（半自动）
5. 积累经验后再做全自动实盘
```

### 9.3 WaveTrend策略执行建议

| 周期 | 品种 | 执行方式 |
|------|------|----------|
| 日线 | M/RB/CU/AU | 每天收盘后计算，次日开盘手动执行 |
| 60分钟 | I/TA/AU/NI | 模拟盘运行，信号推送，手动确认 |
| 15分钟 | TA/AG/AU | 盯盘+信号提示 |

---

## 十、总结

### 现有系统评估
- 回测能力：**优秀** (90%)
- 策略框架：**优秀** (90%)
- 实盘能力：**缺失** (0%)

### 升级目标
- 实盘能力：**完整** (100%)
- 预计工期：**7周**
- 推荐起点：**简化版模拟盘** (1周)

### 下一步行动
1. 确认开发路径（全功能 vs 快速验证）
2. 申请期货公司模拟账号
3. 安装天勤SDK测试行情
4. 开始Phase 1开发

---

*文档版本：1.0*
*创建日期：2026-01-07*
*参考项目：trader-master*
