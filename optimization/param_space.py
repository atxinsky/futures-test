# coding=utf-8
"""
参数空间定义与管理
为每个策略提供默认参数空间，支持自定义
"""

from typing import Dict, List, Optional
from .base import ParamSpace

# 路径已在 __init__.py 中统一设置


class ParamSpaceManager:
    """参数空间管理器"""

    # 预定义参数空间（基于策略历史最优区间）
    PREDEFINED_SPACES = {
        "brother2v6": {
            # ========== 趋势参数 ==========
            "sml_len": ParamSpace("sml_len", 8, 18, param_type="int", default=12,
                                  label="短期EMA", description="短期EMA周期"),
            "big_len": ParamSpace("big_len", 35, 70, step=5, param_type="int", default=50,
                                  label="长期EMA", description="长期EMA周期"),
            "break_len": ParamSpace("break_len", 20, 45, step=5, param_type="int", default=30,
                                    label="突破周期", description="N日高点突破周期"),

            # ========== 指标参数 ==========
            "atr_len": ParamSpace("atr_len", 14, 30, param_type="int", default=20,
                                  label="ATR周期", description="ATR计算周期"),
            "adx_len": ParamSpace("adx_len", 7, 21, param_type="int", default=14,
                                  label="ADX周期", description="ADX指标周期"),
            "adx_thres": ParamSpace("adx_thres", 18.0, 28.0, step=1.0, default=22.0,
                                    label="ADX阈值", description="ADX大于此值才开仓"),
            "chop_len": ParamSpace("chop_len", 10, 20, param_type="int", default=14,
                                   label="CHOP周期", description="CHOP指标周期"),
            "chop_thres": ParamSpace("chop_thres", 45.0, 55.0, step=1.0, default=50.0,
                                     label="CHOP阈值", description="CHOP小于此值表示趋势市场"),

            # ========== 成交量 ==========
            "vol_len": ParamSpace("vol_len", 15, 30, param_type="int", default=20,
                                  label="均量周期", description="成交量均线周期"),
            "vol_multi": ParamSpace("vol_multi", 1.1, 2.0, step=0.1, default=1.3,
                                    label="放量倍数", description="成交量需大于均量的倍数"),

            # ========== 止损参数 ==========
            "stop_n": ParamSpace("stop_n", 2.0, 4.5, step=0.5, default=3.0,
                                 label="初始止损ATR倍数", description="初始止损距离=ATR×此倍数"),
            "min_stop_n": ParamSpace("min_stop_n", 1.5, 3.0, step=0.5, default=2.0,
                                     label="最小止损ATR倍数", description="盈利较大时的最小止损倍数"),

            # ========== 止盈参数 ==========
            "break_even_pct": ParamSpace("break_even_pct", 8.0, 15.0, step=1.0, default=10.0,
                                         label="保本触发%", description="盈利超过此比例后启动保本机制"),
            "partial_trigger_pct": ParamSpace("partial_trigger_pct", 8.0, 18.0, step=1.0, default=12.0,
                                              label="分批止盈触发%", description="盈利达到此比例后可触发分批止盈"),
            "partial_drawdown_pct": ParamSpace("partial_drawdown_pct", 3.0, 8.0, step=1.0, default=4.0,
                                               label="分批止盈回撤%", description="从高点回撤此比例时触发分批止盈"),
            "partial_rate": ParamSpace("partial_rate", 30.0, 70.0, step=10.0, default=50.0,
                                       label="分批止盈比例%", description="第一次止盈平仓的比例"),
            "full_drawdown_pct": ParamSpace("full_drawdown_pct", 5.0, 12.0, step=1.0, default=8.0,
                                            label="剩余止盈回撤%", description="剩余仓位从高点回撤此比例时全部平仓"),

            # ========== 仓位管理 ==========
            "capital_rate": ParamSpace("capital_rate", 0.1, 0.5, step=0.1, default=0.2,
                                       label="资金使用比例", description="用于计算仓位的资金比例"),
            "risk_rate": ParamSpace("risk_rate", 0.01, 0.10, step=0.01, default=0.05,
                                    label="单次风险比例", description="每次交易最大风险占资金比例"),
        },

        "brother2v5": {
            "sml_len": ParamSpace("sml_len", 8, 15, param_type="int", default=10,
                                  label="短期EMA", description="短期EMA周期"),
            "big_len": ParamSpace("big_len", 30, 50, step=5, param_type="int", default=40,
                                  label="长期EMA", description="长期EMA周期"),
            "break_len": ParamSpace("break_len", 30, 50, step=5, param_type="int", default=40,
                                    label="突破周期", description="N日高点突破周期"),
            "atr_len": ParamSpace("atr_len", 14, 30, param_type="int", default=20,
                                  label="ATR周期", description="ATR计算周期"),
            "adx_len": ParamSpace("adx_len", 10, 20, param_type="int", default=14,
                                  label="ADX周期", description="ADX指标周期"),
            "adx_thres": ParamSpace("adx_thres", 20.0, 30.0, step=1.0, default=25.0,
                                    label="ADX阈值", description="ADX大于此值才开仓"),
            "stop_n": ParamSpace("stop_n", 3.0, 5.0, step=0.5, default=4.0,
                                 label="止损ATR倍数", description="止损距离=ATR×此倍数"),
            "capital_rate": ParamSpace("capital_rate", 0.5, 1.0, step=0.1, default=1.0,
                                       label="资金使用比例", description="用于计算仓位的资金比例"),
            "risk_rate": ParamSpace("risk_rate", 0.02, 0.05, step=0.01, default=0.03,
                                    label="单次风险比例", description="每次交易最大风险占资金比例"),
        },

        "brother2_enhanced": {
            "sml_len": ParamSpace("sml_len", 8, 18, param_type="int", default=12,
                                  label="短期EMA", description="短期EMA周期"),
            "big_len": ParamSpace("big_len", 35, 65, step=5, param_type="int", default=50,
                                  label="长期EMA", description="长期EMA周期"),
            "break_len": ParamSpace("break_len", 25, 45, step=5, param_type="int", default=35,
                                    label="突破周期", description="N日高点突破周期"),
            "atr_len": ParamSpace("atr_len", 14, 28, param_type="int", default=20,
                                  label="ATR周期", description="ATR计算周期"),
            "adx_thres": ParamSpace("adx_thres", 18.0, 28.0, step=1.0, default=22.0,
                                    label="ADX阈值", description="ADX大于此值才开仓"),
            "stop_n": ParamSpace("stop_n", 2.5, 4.0, step=0.5, default=3.0,
                                 label="止损ATR倍数", description="止损距离=ATR×此倍数"),
        },

        "brother2v6_dual": {
            # 双向版本参数与v6基本相同
            "sml_len": ParamSpace("sml_len", 8, 18, param_type="int", default=12,
                                  label="短期EMA", description="短期EMA周期"),
            "big_len": ParamSpace("big_len", 35, 70, step=5, param_type="int", default=50,
                                  label="长期EMA", description="长期EMA周期"),
            "break_len": ParamSpace("break_len", 20, 45, step=5, param_type="int", default=30,
                                    label="突破周期", description="N日高/低点突破周期"),
            "atr_len": ParamSpace("atr_len", 14, 30, param_type="int", default=20,
                                  label="ATR周期", description="ATR计算周期"),
            "adx_thres": ParamSpace("adx_thres", 18.0, 28.0, step=1.0, default=22.0,
                                    label="ADX阈值", description="ADX大于此值才开仓"),
            "chop_thres": ParamSpace("chop_thres", 45.0, 55.0, step=1.0, default=50.0,
                                     label="CHOP阈值", description="CHOP小于此值表示趋势市场"),
            "vol_multi": ParamSpace("vol_multi", 1.1, 2.0, step=0.1, default=1.3,
                                    label="放量倍数", description="成交量需大于均量的倍数"),
            "stop_n": ParamSpace("stop_n", 2.0, 4.5, step=0.5, default=3.0,
                                 label="止损ATR倍数", description="止损距离=ATR×此倍数"),
        },
    }

    # 参数分组定义（用于UI展示）
    PARAM_GROUPS = {
        "brother2v6": {
            "趋势参数": ["sml_len", "big_len", "break_len"],
            "指标参数": ["atr_len", "adx_len", "adx_thres", "chop_len", "chop_thres"],
            "成交量": ["vol_len", "vol_multi"],
            "止损参数": ["stop_n", "min_stop_n"],
            "止盈参数": ["break_even_pct", "partial_trigger_pct", "partial_drawdown_pct",
                       "partial_rate", "full_drawdown_pct"],
            "仓位管理": ["capital_rate", "risk_rate"]
        },
        "brother2v5": {
            "趋势参数": ["sml_len", "big_len", "break_len"],
            "指标参数": ["atr_len", "adx_len", "adx_thres"],
            "止损参数": ["stop_n"],
            "仓位管理": ["capital_rate", "risk_rate"]
        },
        "brother2_enhanced": {
            "趋势参数": ["sml_len", "big_len", "break_len"],
            "指标参数": ["atr_len", "adx_thres"],
            "止损参数": ["stop_n"]
        },
        "brother2v6_dual": {
            "趋势参数": ["sml_len", "big_len", "break_len"],
            "指标参数": ["atr_len", "adx_thres", "chop_thres"],
            "成交量": ["vol_multi"],
            "止损参数": ["stop_n"]
        },
    }

    # 关键参数（减少搜索空间时使用）
    KEY_PARAMS = {
        "brother2v6": [
            "sml_len", "big_len", "break_len",      # 趋势核心
            "adx_thres", "chop_thres",              # 过滤核心
            "stop_n", "min_stop_n",                 # 止损核心
            "partial_trigger_pct"                   # 止盈核心
        ],
        "brother2v5": ["sml_len", "big_len", "adx_thres", "stop_n"],
        "brother2_enhanced": ["sml_len", "big_len", "adx_thres", "stop_n"],
        "brother2v6_dual": ["sml_len", "big_len", "adx_thres", "stop_n"],
    }

    @classmethod
    def get_all_params(cls, strategy_name: str) -> Dict[str, ParamSpace]:
        """获取策略的所有参数空间"""
        return cls.PREDEFINED_SPACES.get(strategy_name, {}).copy()

    @classmethod
    def get_key_params(cls, strategy_name: str) -> Dict[str, ParamSpace]:
        """获取关键参数（减少搜索空间）"""
        all_params = cls.get_all_params(strategy_name)
        key_names = cls.KEY_PARAMS.get(strategy_name, list(all_params.keys()))
        return {k: v for k, v in all_params.items() if k in key_names}

    @classmethod
    def get_param_groups(cls, strategy_name: str) -> Dict[str, List[str]]:
        """获取参数分组（用于UI展示）"""
        return cls.PARAM_GROUPS.get(strategy_name, {"参数": list(cls.get_all_params(strategy_name).keys())})

    @classmethod
    def get_supported_strategies(cls) -> List[str]:
        """获取支持优化的策略列表"""
        return list(cls.PREDEFINED_SPACES.keys())

    @classmethod
    def auto_detect_params(cls, strategy_name: str) -> Dict[str, ParamSpace]:
        """
        自动检测策略参数
        通过调用策略类的get_params()方法获取参数定义，
        然后转换为ParamSpace格式
        """
        try:
            from strategies import get_strategy

            strategy_class = get_strategy(strategy_name)
            if not strategy_class:
                return cls.get_all_params(strategy_name)

            # 获取策略定义的参数
            strategy_params = strategy_class.get_params()
            if not strategy_params:
                return cls.get_all_params(strategy_name)

            # 转换为ParamSpace格式
            param_spaces = {}
            for sp in strategy_params:
                # 使用策略定义的范围，如果没有则使用预定义的范围
                predefined = cls.PREDEFINED_SPACES.get(strategy_name, {}).get(sp.name)

                if predefined:
                    # 优先使用预定义的范围（经过调优）
                    param_spaces[sp.name] = predefined
                elif sp.min_val is not None and sp.max_val is not None:
                    # 使用策略自身定义的范围
                    param_spaces[sp.name] = ParamSpace(
                        name=sp.name,
                        low=sp.min_val,
                        high=sp.max_val,
                        step=sp.step,
                        param_type=sp.param_type,
                        default=sp.default,
                        label=sp.label,
                        description=sp.description
                    )

            return param_spaces

        except Exception as e:
            # 如果自动检测失败，返回预定义的参数
            return cls.get_all_params(strategy_name)
