# Futures Quantitative Trading System v2.0

专业的中国期货市场量化交易系统，支持回测与模拟盘交易。

## Features

### Core Features
- **Backtesting**: 专业回测引擎，支持70+期货品种
- **ETF Backtesting**: ETF趋势轮动策略回测（BigBrother V14）
- **Simulation Trading**: 模拟盘实时交易
- **Risk Management**: 完整风控体系（持仓限制、日亏限制、最大回撤）
- **Professional Web UI**: 基于Streamlit的专业交易界面

### Trading Components
- **Event Engine**: 事件驱动架构
- **Order Manager**: 订单管理与状态跟踪
- **Position Manager**: 持仓管理与盈亏计算
- **Risk Manager**: 风险控制与合规检查
- **Account Manager**: 账户资金与绩效统计
- **Trade Manager**: 完整交易生命周期管理（分批建仓/平仓）

### Data Support
- **TianQin Integration**: 天勤数据接口
- **Local Database**: SQLite本地存储
- **Multi-Period**: 支持1m/5m/15m/1h/1d等多周期

## Quick Start

### Docker (Recommended)

```bash
# Build and start
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

Access: http://localhost:8502

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start Web UI
python run.py web

# Run backtest (CLI)
python run.py backtest -s RB -p 1d --start 2024-01-01 --end 2024-12-31

# Start simulation trading
python run.py sim -s RB -c 100000
```

Port: http://localhost:8502

### Windows

Double-click `start_web.bat`

## Architecture

```
futures-backtest/
├── app/                    # Web界面
│   └── main.py            # Streamlit主程序
├── core/                   # 核心引擎
│   ├── event_engine.py    # 事件引擎
│   ├── backtest_engine.py # 回测引擎
│   ├── live_engine.py     # 实盘/模拟盘引擎
│   └── scheduler.py       # 任务调度器
├── gateway/                # 网关接口
│   ├── base_gateway.py    # 网关基类
│   └── sim_gateway.py     # 模拟盘网关
├── trading/                # 交易管理
│   ├── order_manager.py   # 订单管理
│   ├── position_manager.py# 持仓管理
│   ├── risk_manager.py    # 风控管理
│   ├── account_manager.py # 账户管理
│   └── trade_manager.py   # 交易生命周期管理
├── strategies/             # 策略模块
│   ├── base.py            # 策略基类
│   ├── wavetrend_final.py # WaveTrend策略
│   ├── emanew_v5.py       # EmaNew V5策略
│   ├── macdema_v3.py      # MACD+EMA V3策略
│   └── ...                # 更多策略
├── models/                 # 数据模型
│   └── base.py            # 基础数据类
├── utils/                  # 工具模块
│   └── data_loader.py     # 数据加载器
├── data/                   # 数据存储
├── logs/                   # 日志目录
├── config.py              # 品种配置
├── system_config.py       # 系统配置
├── run.py                 # CLI启动脚本
├── Dockerfile
└── docker-compose.yml
```

## Built-in Strategies

### WaveTrend Final
基于WaveTrend指标的趋势策略，支持日线/小时线。

### EmaNew V5 (分批止盈)
解决"卖飞焦虑"的分批止盈策略。

**Entry Signal:**
- EMA(9) 金叉 EMA(21)
- MACD柱状图 > 0 且动量增强
- 收盘价 > MA(20)

**Exit Rules:**
| Type | Condition |
|------|-----------|
| 第一次止盈(50%) | 盈利≥15% 且 从高点回撤≥6% |
| 第二次止盈(100%) | 从高点回撤≥12% 或 EMA死叉 |
| 信号K线止损 | 连续3天收盘<金叉K线最低价 |
| 固定止损 | 亏损≥8% |
| 保本止损 | 盈利超10%后跌回入场价 |

### MACD+EMA V3 (追踪止盈)
一次性止盈版本，追踪止损保护利润。

### Brother2 V6
趋势突破策略，参考Banbot BigBrother架构。

## Web UI Pages

| Page | Description |
|------|-------------|
| 仪表盘 | 账户概览、持仓盈亏、风控状态 |
| 策略管理 | 策略列表、启停控制、参数配置 |
| 持仓监控 | 实时持仓、盈亏跟踪 |
| 订单管理 | 订单记录、状态查询 |
| 风控中心 | 风控指标、限制配置 |
| 回测系统 | 策略回测、参数优化 |
| 系统设置 | 系统配置、日志查看 |

## Supported Futures

| Category | Symbols |
|----------|---------|
| Stock Index | IF(沪深300), IH(上证50), IC(中证500), IM(中证1000) |
| Treasury | T(10年), TF(5年), TS(2年), TL(30年) |
| Precious Metals | AU(黄金), AG(白银) |
| Base Metals | CU(铜), AL(铝), ZN(锌), PB(铅), NI(镍), SN(锡) |
| Ferrous | RB(螺纹), HC(热卷), I(铁矿), J(焦炭), JM(焦煤) |
| Energy & Chemicals | SC(原油), FU(燃油), TA(PTA), MA(甲醇), PP, L, V |
| Agricultural | M(豆粕), Y(豆油), C(玉米), CF(棉花), SR(白糖) |

## Risk Management

### Risk Controls
| Control | Description | Default |
|---------|-------------|---------|
| max_position_per_symbol | 单品种最大持仓 | 10 |
| max_position_total | 总持仓上限 | 50 |
| max_margin_ratio | 最大保证金占用 | 80% |
| max_daily_loss_ratio | 日亏损限制 | 5% |
| max_drawdown_ratio | 最大回撤限制 | 15% |
| default_stop_loss | 默认止损比例 | 3% |

## CLI Commands

```bash
# Start Web UI
python run.py web

# Run backtest
python run.py backtest -s RB -p 1d --start 2024-01-01 --end 2024-12-31 -c 100000

# Start simulation
python run.py sim -s RB -c 100000
```

## Custom Strategy

```python
from strategies.base import BaseStrategy, StrategyParam, Signal

class MyStrategy(BaseStrategy):
    name = "my_strategy"
    display_name = "My Strategy"
    description = "Strategy description"
    version = "1.0"

    @classmethod
    def get_params(cls):
        return [
            StrategyParam("period", "Period", 20, 5, 50, 1, "int"),
        ]

    def calculate_indicators(self, df):
        df['ma'] = df['close'].rolling(self.params['period']).mean()
        return df

    def on_bar(self, idx, df, capital):
        row = df.iloc[idx]

        # Entry
        if self.position == 0 and row['close'] > row['ma']:
            self.position = 1
            self.entry_price = row['close']
            return Signal("buy", row['close'], tag="ma_breakout")

        # Exit
        if self.position == 1:
            profit = (row['close'] - self.entry_price) / self.entry_price
            if profit >= 0.05:
                self.position = 0
                return Signal("close", row['close'], tag="take_profit")

        return None
```

## Docker Configuration

### Volumes
| Volume | Description |
|--------|-------------|
| ./data | 数据持久化 |
| ./logs | 日志文件 |
| ./strategies | 策略文件（热更新） |
| ./core | 核心模块（热更新） |
| ./app | Web界面（热更新） |

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| TZ | 时区 | Asia/Shanghai |
| PYTHONUNBUFFERED | Python输出缓冲 | 1 |

## Performance Metrics

| Metric | Description |
|--------|-------------|
| Total Return | 总收益率 |
| Annual Return | 年化收益率 |
| Sharpe Ratio | 夏普比率 |
| Sortino Ratio | 索提诺比率 |
| Calmar Ratio | 卡尔玛比率 |
| Max Drawdown | 最大回撤 |
| Win Rate | 胜率 |
| Profit Factor | 盈亏比 |

## Changelog

### v2.1 (2026-01-08)
- **StrategyTrade Model**: 完整交易生命周期管理
  - 支持分批建仓（多笔开仓成交自动计算加权入场均价）
  - 支持分批平仓（多笔平仓成交自动计算盈亏）
  - 交易状态机：PENDING → OPENING → HOLDING → CLOSING → CLOSED
  - 极值追踪（持仓期间最高/最低价，用于追踪止损）
- **TradeManager**: 新增交易管理器
  - 交易创建与生命周期管理
  - 成交事件自动处理
  - 统计功能（胜率、盈亏汇总）
- **OrderManager增强**:
  - `open_trade()` 创建交易并发送开仓订单
  - `close_trade()` 平仓指定交易
  - `close_all_trades()` 批量平仓

### v2.0 (2026-01-07)
- Added live/simulation trading support
- New event-driven architecture
- Added risk management system
- Professional Web UI with multiple pages
- Order/Position/Account management
- Task scheduler for automated trading

### v1.0
- Initial backtest system
- Basic Web UI
- Multiple strategies

## License

MIT

## Author

atxinsky
