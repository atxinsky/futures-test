# 期货量化交易系统 v2.0

## 项目信息

- 端口：8504
- 启动：`python run.py web` 或 `start_web.bat`
- 技术栈：Streamlit + TqSdk + 事件驱动架构

## 核心模块

| 模块 | 文件 | 说明 |
|------|------|------|
| StrategyTrade | `models/base.py` | 交易生命周期管理 |
| TradeManager | `trading/trade_manager.py` | 分批建仓/平仓追踪 |
| OrderRetryHandler | `utils/order_retry.py` | 涨跌停智能重报 |
| ContractRoller | `utils/contract_roller.py` | 合约换月管理 |
| OrderIdGenerator | `utils/order_id.py` | 订单号生成 |

## TqSdk 配置

使用环境变量：
- `TQSDK_USER`：账号
- `TQSDK_PASSWORD`：密码

## UI 结构

```
仪表盘 | 模拟交易 | 实盘交易 | 风控中心 | 回测系统 | 系统设置
```

## 设计原则

- 交易界面三列布局：设置|参数|合约规格
- 持仓/订单在交易页面内，不是一级菜单
- 连接设置放系统设置
- 启动按钮在所有配置之后

## 订单号格式

```
{策略简码}_{日期}_{6位序号}_{信号ID后4位}
示例：B6_0108_000001_1234
```
