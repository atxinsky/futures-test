# coding=utf-8
"""
基于Optuna的参数优化器
支持TPE采样、并行优化、多目标优化
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

# 路径已在 __init__.py 中统一设置

from .base import BaseOptimizer, OptimizationConfig, ParamSpace, OptimizationResult

logger = logging.getLogger(__name__)


class OptunaOptimizer(BaseOptimizer):
    """Optuna优化器"""

    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self._trial_results: List[Dict] = []
        self._loaded_data: Dict[str, pd.DataFrame] = {}

    def optimize(self, param_spaces: Dict[str, ParamSpace]) -> OptimizationResult:
        """综合多品种优化（参数共享）"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna未安装，请运行: pip install optuna")

        self._log(f"开始优化策略: {self.config.strategy_name}")
        self._log(f"品种: {', '.join(self.config.symbols)}")
        self._log(f"训练集: {self.config.train_start} ~ {self.config.train_end}")
        self._log(f"参数数量: {len(param_spaces)}")

        # 1. 加载数据
        self._load_data()
        if not self._loaded_data:
            raise ValueError("无法加载任何品种数据")

        # 2. 创建Study（带早停Pruner）
        self._update_progress(0.1, "创建优化器...")
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,      # 前5次不剪枝
                n_warmup_steps=0,        # 不需要warmup步骤
                interval_steps=1         # 每步检查
            )
        )

        # 3. 定义目标函数
        def objective(trial):
            return self._objective_function(trial, param_spaces, self._loaded_data)

        # 4. 执行优化
        self._update_progress(0.2, "开始优化...")
        self._trial_results.clear()

        def callback(study, trial):
            progress = 0.2 + (trial.number + 1) / self.config.n_trials * 0.7
            self._update_progress(progress, f"Trial {trial.number + 1}/{self.config.n_trials}")
            if trial.value and trial.value > -900:
                self._log(f"Trial {trial.number}: {self.config.objective}={trial.value:.3f}")

        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            callbacks=[callback],
            show_progress_bar=False
        )

        # 5. 构建结果
        self._update_progress(0.95, "构建结果...")
        result = self._build_result(study, param_spaces, None)

        self._update_progress(1.0, "优化完成!")
        self._log(f"最优{self.config.objective}: {result.best_value:.3f}")

        return result

    def optimize_per_symbol(self, param_spaces: Dict[str, ParamSpace]) -> Dict[str, OptimizationResult]:
        """每品种独立优化"""
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("Optuna未安装，请运行: pip install optuna")

        self._log(f"开始每品种独立优化: {self.config.strategy_name}")

        # 1. 加载数据
        self._load_data()
        if not self._loaded_data:
            raise ValueError("无法加载任何品种数据")

        results = {}
        total_symbols = len(self._loaded_data)

        for idx, (symbol, data) in enumerate(self._loaded_data.items()):
            progress_base = idx / total_symbols
            progress_span = 1.0 / total_symbols

            self._log(f"优化品种 {symbol} ({idx + 1}/{total_symbols})...")
            self._update_progress(progress_base, f"优化 {symbol}...")

            # 创建Study
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=42)
            )

            # 单品种目标函数 - 使用工厂函数避免闭包变量捕获问题
            single_data = {symbol: data}
            objective = self._make_objective(param_spaces, single_data)

            # 执行优化
            self._trial_results.clear()

            # 回调函数也需要避免闭包问题
            callback = self._make_callback(progress_base, progress_span, symbol)

            study.optimize(
                objective,
                n_trials=self.config.n_trials,
                callbacks=[callback],
                show_progress_bar=False
            )

            # 构建结果
            result = self._build_result(study, param_spaces, symbol)
            results[symbol] = result

            self._log(f"{symbol} 最优{self.config.objective}: {result.best_value:.3f}")

        self._update_progress(1.0, "全部优化完成!")
        return results

    def _make_objective(self, param_spaces: Dict[str, ParamSpace], data_dict: Dict[str, pd.DataFrame]):
        """工厂函数：创建目标函数，避免闭包变量捕获问题"""
        def objective(trial):
            return self._objective_function(trial, param_spaces, data_dict)
        return objective

    def _make_callback(self, progress_base: float, progress_span: float, symbol: str):
        """工厂函数：创建回调函数，避免闭包变量捕获问题"""
        def callback(study, trial):
            progress = progress_base + (trial.number + 1) / self.config.n_trials * progress_span
            self._update_progress(progress, f"{symbol} Trial {trial.number + 1}")
        return callback

    def _load_data(self):
        """加载所有品种数据"""
        self._update_progress(0.0, "加载数据...")
        self._loaded_data.clear()

        # 尝试导入数据加载器
        try:
            from utils.data_loader import load_futures_data
        except ImportError:
            # 如果导入失败，尝试使用data_manager
            try:
                from data_manager import load_from_database as load_futures_data
            except ImportError:
                raise ImportError("无法导入数据加载器")

        total = len(self.config.symbols)
        for idx, symbol in enumerate(self.config.symbols):
            self._log(f"加载 {symbol} ({idx + 1}/{total})...")

            try:
                # 尝试加载数据（使用配置的时间周期）
                timeframe = getattr(self.config, 'timeframe', '1d')
                df = load_futures_data(
                    symbol,
                    self.config.train_start,
                    self.config.val_end,
                    period=timeframe
                )

                if df is not None and len(df) > 100:
                    # 确保索引是日期格式 - 复制以避免修改原始数据
                    df = df.copy()
                    if not isinstance(df.index, pd.DatetimeIndex):
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                            df.set_index('date', inplace=True)
                        elif 'time' in df.columns:
                            df['time'] = pd.to_datetime(df['time'])
                            df.set_index('time', inplace=True)
                        else:
                            self._log(f"  {symbol}: 无法识别时间列，跳过")
                            continue

                    self._loaded_data[symbol] = df
                    self._log(f"  {symbol}: {len(df)}行")
                else:
                    self._log(f"  {symbol}: 数据不足，跳过")

            except Exception as e:
                self._log(f"  {symbol}: 加载失败 - {e}")

        self._log(f"数据加载完成，共 {len(self._loaded_data)} 个品种")

    def _objective_function(self, trial, param_spaces: Dict[str, ParamSpace],
                            data_dict: Dict[str, pd.DataFrame]) -> float:
        """目标函数"""
        # 1. 采样参数
        params = {}
        for name, space in param_spaces.items():
            if space.param_type == "int":
                params[name] = trial.suggest_int(name, int(space.low), int(space.high))
            else:
                step = space.step if space.step else round((space.high - space.low) / 10, 4)
                if step < 0.001:
                    step = 0.001
                params[name] = trial.suggest_float(
                    name, space.low, space.high,
                    step=step
                )

        # 2. 获取策略类
        try:
            from strategies import get_strategy
            strategy_class = get_strategy(self.config.strategy_name)
            if not strategy_class:
                return -999
        except Exception as e:
            logger.warning(f"获取策略失败: {e}")
            return -999

        # 3. 多品种回测
        total_sharpe = 0
        total_sortino = 0
        total_return = 0
        total_trades = 0
        max_dd = 0
        valid_count = 0

        # 导入回测引擎
        try:
            from core.backtest_engine import BacktestEngine
        except ImportError:
            from engine import run_backtest_with_strategy
            BacktestEngine = None

        import numpy as np

        for symbol, df in data_dict.items():
            try:
                # 获取策略的warmup_num
                warmup = getattr(strategy_class, 'warmup_num', 100)

                # 使用loc更安全地筛选数据
                train_start_ts = pd.Timestamp(self.config.train_start)
                train_end_ts = pd.Timestamp(self.config.train_end)

                # 找到开始位置，向前偏移warmup条
                mask = df.index < train_start_ts
                start_idx = mask.sum()
                warmup_start_idx = max(0, start_idx - warmup)

                # 筛选数据（包含预热期）
                train_df = df.iloc[warmup_start_idx:]
                train_df = train_df[train_df.index <= train_end_ts]

                if len(train_df) < warmup + 50:  # 确保有足够数据
                    continue

                # 执行回测
                strategy = strategy_class(params=params)

                if BacktestEngine:
                    engine = BacktestEngine()
                    result = engine.run(
                        strategy=strategy,
                        symbol=symbol,
                        data=train_df,
                        initial_capital=self.config.initial_capital,
                        check_limit_price=False
                    )
                else:
                    result = run_backtest_with_strategy(
                        train_df, symbol, strategy,
                        self.config.initial_capital
                    )

                if result and hasattr(result, 'total_trades') and result.total_trades > 0:
                    sharpe = getattr(result, 'sharpe_ratio', 0) or 0
                    ret = getattr(result, 'total_return', 0) or 0
                    trades = getattr(result, 'total_trades', 0) or 0
                    dd = getattr(result, 'max_drawdown_pct', 0) or 0

                    # 计算Sortino（如果有equity_curve）
                    sortino = sharpe  # 默认用sharpe
                    if hasattr(result, 'equity_curve') and result.equity_curve is not None:
                        try:
                            equity = result.equity_curve
                            if isinstance(equity, pd.DataFrame) and 'equity' in equity.columns:
                                returns = equity['equity'].pct_change().dropna()
                            elif isinstance(equity, pd.Series):
                                returns = equity.pct_change().dropna()
                            else:
                                returns = None

                            if returns is not None and len(returns) > 10:
                                neg_returns = returns[returns < 0]
                                if len(neg_returns) > 0:
                                    downside_std = np.sqrt(np.mean(neg_returns ** 2)) * np.sqrt(252)
                                    if downside_std > 0:
                                        # 复利年化收益率
                                        trading_days = len(returns)
                                        annual_return = (1 + ret) ** (252 / trading_days) - 1 if trading_days > 0 else 0
                                        sortino = (annual_return - 0.03) / downside_std
                        except Exception:
                            pass

                    total_sharpe += sharpe
                    total_sortino += sortino
                    total_return += ret
                    total_trades += trades
                    max_dd = max(max_dd, dd)
                    valid_count += 1

                    # 早停检查：只在极端差的情况下终止（sharpe < -5且亏损超过20%）
                    if valid_count == 1 and sharpe < -5 and ret < -0.2:
                        return -999

            except Exception as e:
                logger.warning(f"回测 {symbol} 失败: {e}")
                continue

        if valid_count == 0:
            return -999

        # 4. 计算平均指标
        avg_sharpe = total_sharpe / valid_count
        avg_sortino = total_sortino / valid_count
        avg_return = total_return / valid_count

        # 5. 约束条件惩罚
        if total_trades < self.config.min_trades:
            return -999
        if max_dd > self.config.max_drawdown:
            return -999

        # 6. 记录结果（回撤为小数形式，如0.18表示18%）
        self._trial_results.append({
            'trial': trial.number,
            'params': params.copy(),
            'sharpe': avg_sharpe,
            'sortino': avg_sortino,
            'return': avg_return,
            'max_drawdown': max_dd,  # 小数形式
            'trades': total_trades
        })

        # 7. 返回目标值
        if self.config.objective == 'sharpe':
            return avg_sharpe
        elif self.config.objective == 'calmar':
            return avg_return / max_dd if max_dd > 0 else avg_return
        elif self.config.objective == 'return':
            return avg_return
        elif self.config.objective == 'sortino':
            return avg_sortino
        else:
            return avg_sharpe

    def _build_result(self, study, param_spaces: Dict[str, ParamSpace],
                      symbol: Optional[str]) -> OptimizationResult:
        """构建优化结果"""
        import optuna

        # 1. 最优参数
        best_params = study.best_params
        best_value = study.best_value

        # 2. 参数重要性
        try:
            param_importance = optuna.importance.get_param_importances(study)
        except Exception:
            param_importance = {}

        # 3. 优化历史
        history_df = pd.DataFrame(self._trial_results) if self._trial_results else pd.DataFrame()

        # 4. 验证集测试
        train_metrics, val_metrics = self._validate_params(best_params, symbol)

        return OptimizationResult(
            strategy_name=self.config.strategy_name,
            symbol=symbol,
            best_params=best_params,
            best_value=best_value,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            param_importance=param_importance,
            optimization_history=history_df,
            created_at=datetime.now(),
            config=self.config
        )

    def _validate_params(self, params: Dict, symbol: Optional[str]) -> tuple:
        """验证集测试"""
        train_metrics = {}
        val_metrics = {}

        try:
            from strategies import get_strategy
            strategy_class = get_strategy(self.config.strategy_name)
            if not strategy_class:
                return train_metrics, val_metrics
        except Exception:
            return train_metrics, val_metrics

        # 导入回测引擎
        try:
            from core.backtest_engine import BacktestEngine
        except ImportError:
            from engine import run_backtest_with_strategy
            BacktestEngine = None

        data_dict = self._loaded_data if symbol is None else {symbol: self._loaded_data[symbol]}

        for period_name, start, end in [
            ("train", self.config.train_start, self.config.train_end),
            ("val", self.config.val_start, self.config.val_end)
        ]:
            total_sharpe = 0
            total_return = 0
            total_drawdown = 0
            total_trades = 0
            valid_count = 0

            for sym, df in data_dict.items():
                try:
                    # 获取策略的warmup_num，向前多取数据
                    warmup = getattr(strategy_class, 'warmup_num', 100)
                    start_ts = pd.Timestamp(start)
                    end_ts = pd.Timestamp(end)
                    
                    start_idx = df.index.searchsorted(start_ts)
                    warmup_start_idx = max(0, start_idx - warmup)
                    
                    period_df = df.iloc[warmup_start_idx:]
                    period_df = period_df[period_df.index <= end_ts]
                    
                    if len(period_df) < warmup + 20:
                        continue

                    strategy = strategy_class(params=params)

                    if BacktestEngine:
                        engine = BacktestEngine()
                        result = engine.run(
                            strategy=strategy,
                            symbol=sym,
                            data=period_df,
                            initial_capital=self.config.initial_capital,
                            check_limit_price=False
                        )
                    else:
                        result = run_backtest_with_strategy(
                            period_df, sym, strategy,
                            self.config.initial_capital
                        )

                    if result:
                        total_sharpe += getattr(result, 'sharpe_ratio', 0) or 0
                        total_return += getattr(result, 'total_return', 0) or 0
                        # 使用max_drawdown_pct（百分比）
                        total_drawdown += getattr(result, 'max_drawdown_pct', 0) or 0
                        total_trades += getattr(result, 'total_trades', 0) or 0
                        valid_count += 1

                except Exception as e:
                    logger.warning(f"验证 {sym} {period_name} 失败: {e}")

            if valid_count > 0:
                # 所有指标说明：return和max_drawdown为小数形式（0.18=18%）
                metrics = {
                    'sharpe': total_sharpe / valid_count,
                    'return': total_return / valid_count,
                    'max_drawdown': total_drawdown / valid_count,  # 小数形式
                    'trades': total_trades
                }

                if period_name == "train":
                    train_metrics = metrics
                else:
                    val_metrics = metrics

        return train_metrics, val_metrics
