# 期货量化交易系统 v2.0

## 升级完成概述

基于 `trader-master` 专业框架的设计理念，对原有回测系统进行了全面升级改造。

### 升级内容

| 模块 | 状态 | 说明 |
|------|------|------|
| 事件驱动引擎 | ✅ 完成 | `core/event_engine.py` |
| 模拟盘网关 | ✅ 完成 | `gateway/sim_gateway.py` |
| 订单管理器 | ✅ 完成 | `trading/order_manager.py` |
| 持仓管理器 | ✅ 完成 | `trading/position_manager.py` |
| 风控管理器 | ✅ 完成 | `trading/risk_manager.py` |
| 账户管理器 | ✅ 完成 | `trading/account_manager.py` |
| 实盘引擎 | ✅ 完成 | `core/live_engine.py` |
| 定时调度器 | ✅ 完成 | `core/scheduler.py` |
| 回测引擎 | ✅ 完成 | `core/backtest_engine.py` |
| 专业Web界面 | ✅ 完成 | `app/main.py` |

---

## 目录结构

```
D:\期货\回测改造\
├── core/                    # 核心模块
│   ├── event_engine.py      # 事件驱动引擎
│   ├── live_engine.py       # 实盘/模拟盘引擎
│   ├── backtest_engine.py   # 回测引擎
│   └── scheduler.py         # 定时调度器
│
├── gateway/                 # 网关层
│   ├── base_gateway.py      # 网关基类
│   └── sim_gateway.py       # 模拟盘网关
│
├── trading/                 # 交易管理
│   ├── order_manager.py     # 订单管理
│   ├── position_manager.py  # 持仓管理
│   ├── risk_manager.py      # 风控管理
│   └── account_manager.py   # 账户管理
│
├── models/                  # 数据模型
│   └── base.py              # 基础数据结构
│
├── strategies/              # 策略
│   ├── base.py              # 策略基类（支持回测+实盘）
│   ├── wavetrend_final.py   # WaveTrend策略
│   └── ...                  # 其他策略
│
├── utils/                   # 工具
│   └── data_loader.py       # 数据加载器
│
├── app/                     # Web界面
│   └── main.py              # Streamlit主界面
│
├── config.py                # 品种配置
├── system_config.py         # 系统配置
├── run.py                   # 启动脚本
├── start_web.bat            # Windows启动
└── requirements.txt         # 依赖
```

---

## 快速开始

### 1. 启动Web界面

```bash
# 方式一：双击启动
start_web.bat

# 方式二：命令行
python run.py web

# 方式三：直接streamlit
streamlit run app/main.py
```

访问: http://localhost:8501

### 2. 运行回测

```bash
# 命令行回测
python run.py backtest -s RB -p 1d --start 2024-01-01 --end 2025-01-01

# 参数说明
# -s/--symbol: 品种代码
# -p/--period: K线周期 (1d/60m/15m等)
# --start: 开始日期
# --end: 结束日期
# -c/--capital: 初始资金
# -o/--output: 输出文件
```

### 3. 启动模拟盘

```bash
python run.py sim -s RB -c 100000
```

---

## 核心功能

### 事件驱动架构

```python
from core.event_engine import EventEngine
from models.base import EventType

# 创建引擎
engine = EventEngine()

# 注册处理器
engine.register(EventType.BAR, on_bar_handler)
engine.register(EventType.ORDER, on_order_handler)

# 启动
engine.start()
```

### 模拟盘交易

```python
from core.live_engine import LiveEngine
from strategies.wavetrend_final import WaveTrendFinalStrategy

# 创建引擎
engine = LiveEngine()

# 配置品种
engine.set_instrument_config('RB', {
    'multiplier': 10,
    'margin_rate': 0.10
})

# 初始化网关
engine.init_gateway("sim")

# 添加策略
strategy = WaveTrendFinalStrategy()
strategy.set_symbol('RB')
engine.add_strategy(strategy, ['RB'])

# 启动
engine.start(initial_capital=100000)
```

### 风控管理

```python
from trading.risk_manager import RiskManager, RiskConfig

# 配置风控
config = RiskConfig(
    max_position_per_symbol=10,
    max_margin_ratio=0.8,
    max_daily_loss_ratio=0.05,
    max_drawdown_ratio=0.15
)

# 检查订单
passed, reason = risk_manager.check_order(order_request)
```

---

## Web界面功能

- **仪表盘**: 账户概览、权益曲线、持仓分布
- **策略管理**: 添加/移除策略、参数配置、启停控制
- **持仓监控**: 实时持仓、浮动盈亏、快捷平仓
- **订单管理**: 活动订单、成交记录、历史查询
- **风控中心**: 风险状态、限制配置、风控日志
- **回测系统**: 策略回测、绩效分析、报告导出
- **系统设置**: 品种配置、网关设置、数据管理

---

## 备份位置

原系统备份: `D:\期货\回测改造_backup_20260107`

---

## 下一步

1. **接入CTP实盘**: 实现 `CtpGateway` 连接期货公司
2. **数据实时推送**: 对接行情数据源
3. **策略热加载**: 支持运行时添加/修改策略
4. **移动端通知**: 接入微信/钉钉推送

---

*升级完成时间: 2026-01-07*
