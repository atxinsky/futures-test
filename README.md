# Futures Backtest System

专业的中国期货市场策略回测平台，支持70+期货品种。

## Features

- **Multi-Variety Support**: 股指期货、国债期货、贵金属、有色金属、黑色系、能化、农产品等
- **Multiple Strategies**: 内置多种经过验证的交易策略
- **Data Management**: 一键下载历史数据，SQLite数据库存储
- **Professional Metrics**: 夏普比率、索提诺比率、卡尔玛比率、最大回撤等
- **Visualization**: K线图、资金曲线、月度热力图、交易记录

## Quick Start

### Docker (Recommended)

```bash
# Start
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

# Run
streamlit run app.py --server.port 8502
```

### Windows

Double-click `启动.bat`

## Built-in Strategies

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

**Entry Signal:**
- EMA(9) 金叉 EMA(21)
- MACD柱状图 > 0 且动量增强
- 收盘价 > MA(20)

**Exit Rules (by priority):**
1. 信号K线止损：连续N天收盘<金叉K线最低价
2. 固定止损：亏损≥8%
3. 追踪止盈：盈利≥18%后，从高点回撤10%
4. 保本止损：盈利≥10%后跌回入场价
5. 技术信号：EMA死叉 + 跌破MA20

### Other Strategies

- **Dual MA**: 双均线交叉策略
- **Bollinger**: 布林带突破策略
- **Turtle**: 海龟交易策略
- **Brother2V5**: 趋势突破策略

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

## Project Structure

```
futures-backtest/
├── app.py                 # Streamlit web application
├── config.py              # Futures variety configuration
├── engine.py              # Backtest engine
├── data_manager.py        # Data download & storage
├── strategies/            # Strategy modules
│   ├── __init__.py        # Strategy auto-discovery
│   ├── base.py            # BaseStrategy abstract class
│   ├── emanew_v5.py       # EmaNew V5 strategy
│   ├── macdema_v3.py      # MACD+EMA V3 strategy
│   ├── dual_ma.py         # Dual MA strategy
│   ├── bollinger.py       # Bollinger band strategy
│   └── turtle.py          # Turtle trading strategy
├── data/                  # SQLite database storage
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Custom Strategy

Extend `BaseStrategy` to create your own strategy:

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
            StrategyParam("threshold", "Threshold", 0.02, 0.01, 0.10, 0.01, "float"),
        ]

    def calculate_indicators(self, df):
        df['ma'] = df['close'].rolling(self.params['period']).mean()
        return df

    def on_bar(self, idx, df, capital):
        row = df.iloc[idx]

        # Entry logic
        if self.position == 0 and row['close'] > row['ma']:
            self.position = 1
            self.entry_price = row['close']
            return Signal("buy", row['close'], tag="ma_breakout")

        # Exit logic
        if self.position == 1:
            profit = (row['close'] - self.entry_price) / self.entry_price
            if profit >= self.params['threshold']:
                self.position = 0
                return Signal("close", row['close'], tag="take_profit")

        return None
```

## Configuration

### Docker Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| TZ | Timezone | Asia/Shanghai |
| DATA_DIR | Data directory | /app/data |

### Streamlit Config

Edit `.streamlit/config.toml`:

```toml
[server]
port = 8502
headless = true

[theme]
primaryColor = "#667eea"
```

## API

### Strategy Discovery

```python
from strategies import get_all_strategies, list_strategies

# Get all strategy classes
strategies = get_all_strategies()

# Get strategy info list
info = list_strategies()
```

### Run Backtest

```python
from engine import run_backtest_with_strategy
from strategies import get_strategy

# Get strategy class
StrategyClass = get_strategy("emanew_v5")

# Run backtest
results = run_backtest_with_strategy(
    df=price_data,
    strategy_class=StrategyClass,
    params={
        "ema_fast": 9,
        "ema_slow": 21,
        "stop_loss": 0.08
    },
    initial_capital=1000000,
    contract_multiplier=300,
    commission_rate=0.000023
)
```

## Performance Metrics

| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted return |
| Sortino Ratio | Downside risk-adjusted return |
| Calmar Ratio | Return / Max Drawdown |
| Max Drawdown | Maximum peak-to-trough decline |
| Win Rate | Percentage of winning trades |
| Profit Factor | Gross profit / Gross loss |

## License

MIT

## Author

atxinsky
