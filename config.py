# coding=utf-8
"""
品种配置文件
支持股指期货和商品期货的各种参数配置
"""

# 品种配置字典
INSTRUMENTS = {
    # ========== 股指期货 (CFFEX) ==========
    "IF": {
        "name": "沪深300股指",
        "exchange": "CFFEX",
        "multiplier": 300,        # 合约乘数
        "price_tick": 0.2,        # 最小变动价位
        "margin_rate": 0.12,      # 保证金率
        "commission_rate": 0.000023,  # 手续费率 (万分之0.23)
        "commission_fixed": 0,    # 固定手续费
        "night_trade": False,     # 是否有夜盘
        "currency": "CNY",
    },
    "IH": {
        "name": "上证50股指",
        "exchange": "CFFEX",
        "multiplier": 300,
        "price_tick": 0.2,
        "margin_rate": 0.12,
        "commission_rate": 0.000023,
        "commission_fixed": 0,
        "night_trade": False,
        "currency": "CNY",
    },
    "IC": {
        "name": "中证500股指",
        "exchange": "CFFEX",
        "multiplier": 200,
        "price_tick": 0.2,
        "margin_rate": 0.14,
        "commission_rate": 0.000023,
        "commission_fixed": 0,
        "night_trade": False,
        "currency": "CNY",
    },
    "IM": {
        "name": "中证1000股指",
        "exchange": "CFFEX",
        "multiplier": 200,
        "price_tick": 0.2,
        "margin_rate": 0.15,
        "commission_rate": 0.000023,
        "commission_fixed": 0,
        "night_trade": False,
        "currency": "CNY",
    },

    # ========== 商品期货示例 ==========
    "RB": {
        "name": "螺纹钢",
        "exchange": "SHFE",
        "multiplier": 10,
        "price_tick": 1,
        "margin_rate": 0.10,
        "commission_rate": 0.0001,
        "commission_fixed": 0,
        "night_trade": True,
        "currency": "CNY",
    },
    "AU": {
        "name": "黄金",
        "exchange": "SHFE",
        "multiplier": 1000,
        "price_tick": 0.02,
        "margin_rate": 0.08,
        "commission_rate": 0,
        "commission_fixed": 10,  # 固定10元/手
        "night_trade": True,
        "currency": "CNY",
    },
    "CU": {
        "name": "沪铜",
        "exchange": "SHFE",
        "multiplier": 5,
        "price_tick": 10,
        "margin_rate": 0.10,
        "commission_rate": 0.00005,
        "commission_fixed": 0,
        "night_trade": True,
        "currency": "CNY",
    },
    "M": {
        "name": "豆粕",
        "exchange": "DCE",
        "multiplier": 10,
        "price_tick": 1,
        "margin_rate": 0.08,
        "commission_rate": 0,
        "commission_fixed": 1.5,
        "night_trade": True,
        "currency": "CNY",
    },
    "TA": {
        "name": "PTA",
        "exchange": "CZCE",
        "multiplier": 5,
        "price_tick": 2,
        "margin_rate": 0.07,
        "commission_rate": 0,
        "commission_fixed": 3,
        "night_trade": True,
        "currency": "CNY",
    },
}

# 策略参数默认值
DEFAULT_STRATEGY_PARAMS = {
    "brother2v5": {
        "sml_len": 10,       # 短期EMA
        "big_len": 40,       # 长期EMA
        "break_len": 40,     # 突破周期
        "atr_len": 20,       # ATR周期
        "adx_len": 14,       # ADX周期
        "adx_thres": 25.0,   # ADX阈值
        "stop_n": 4.0,       # 止损ATR倍数
        "capital_rate": 1.0, # 资金使用比例
        "risk_rate": 0.03,   # 风险比例
    }
}

# 交易所信息
EXCHANGES = {
    "CFFEX": {"name": "中国金融期货交易所", "timezone": "Asia/Shanghai"},
    "SHFE": {"name": "上海期货交易所", "timezone": "Asia/Shanghai"},
    "DCE": {"name": "大连商品交易所", "timezone": "Asia/Shanghai"},
    "CZCE": {"name": "郑州商品交易所", "timezone": "Asia/Shanghai"},
    "INE": {"name": "上海国际能源交易中心", "timezone": "Asia/Shanghai"},
    "GFEX": {"name": "广州期货交易所", "timezone": "Asia/Shanghai"},
}

def get_instrument(symbol: str) -> dict:
    """获取品种配置"""
    return INSTRUMENTS.get(symbol.upper(), None)

def calculate_commission(symbol: str, price: float, volume: int) -> float:
    """计算手续费"""
    inst = get_instrument(symbol)
    if not inst:
        return 0

    if inst["commission_fixed"] > 0:
        return inst["commission_fixed"] * volume
    else:
        return price * inst["multiplier"] * volume * inst["commission_rate"]

def calculate_margin(symbol: str, price: float, volume: int) -> float:
    """计算保证金"""
    inst = get_instrument(symbol)
    if not inst:
        return 0
    return price * inst["multiplier"] * volume * inst["margin_rate"]

def calculate_pnl(symbol: str, entry_price: float, exit_price: float, volume: int, direction: int = 1) -> float:
    """
    计算盈亏
    direction: 1=多头, -1=空头
    """
    inst = get_instrument(symbol)
    if not inst:
        return 0
    return (exit_price - entry_price) * volume * inst["multiplier"] * direction
