# coding=utf-8
"""
并行回测模块
支持多品种、多参数并行回测
"""

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import multiprocessing
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ParallelBacktestTask:
    """并行回测任务"""
    task_id: str
    strategy_class: type
    strategy_params: Dict[str, Any]
    symbol: str
    period: str = "1d"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 100000.0
    volume: int = 1


@dataclass
class ParallelBacktestResult:
    """并行回测结果"""
    task_id: str
    symbol: str
    params: Dict[str, Any]
    success: bool = True
    error: str = ""
    # 核心指标
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    profit_factor: float = 0.0
    # 完整结果（可选）
    full_result: Any = None


def _run_single_backtest(task: ParallelBacktestTask) -> ParallelBacktestResult:
    """
    执行单个回测任务（进程内函数）
    """
    try:
        # 延迟导入，避免进程fork问题
        from core.backtest_engine import BacktestEngine

        engine = BacktestEngine()
        strategy = task.strategy_class(task.strategy_params)

        result = engine.run(
            strategy=strategy,
            symbol=task.symbol,
            period=task.period,
            start_date=task.start_date,
            end_date=task.end_date,
            initial_capital=task.initial_capital,
            volume=task.volume
        )

        return ParallelBacktestResult(
            task_id=task.task_id,
            symbol=task.symbol,
            params=task.strategy_params,
            success=True,
            total_return=result.total_return,
            annual_return=result.annual_return,
            max_drawdown=result.max_drawdown_pct,
            sharpe_ratio=result.sharpe_ratio,
            win_rate=result.win_rate,
            total_trades=result.total_trades,
            profit_factor=result.profit_factor,
            full_result=result
        )

    except Exception as e:
        logger.error(f"回测任务 {task.task_id} 失败: {e}")
        return ParallelBacktestResult(
            task_id=task.task_id,
            symbol=task.symbol,
            params=task.strategy_params,
            success=False,
            error=str(e)
        )


class ParallelBacktestRunner:
    """
    并行回测运行器

    功能:
    1. 多品种并行回测
    2. 参数网格搜索并行化
    3. 进度回调
    4. 结果汇总
    """

    def __init__(
        self,
        max_workers: int = None,
        use_process: bool = True
    ):
        """
        初始化

        Args:
            max_workers: 最大并行数，默认为CPU核心数
            use_process: 使用进程池(True)还是线程池(False)
        """
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        self.use_process = use_process
        self._progress_callback: Optional[Callable] = None

    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """
        设置进度回调

        Args:
            callback: 回调函数，参数为 (completed, total, message)
        """
        self._progress_callback = callback

    def run_multi_symbol(
        self,
        strategy_class: type,
        strategy_params: Dict[str, Any],
        symbols: List[str],
        period: str = "1d",
        start_date: datetime = None,
        end_date: datetime = None,
        initial_capital: float = 100000.0
    ) -> Dict[str, ParallelBacktestResult]:
        """
        多品种并行回测

        Args:
            strategy_class: 策略类
            strategy_params: 策略参数
            symbols: 品种列表
            period: K线周期
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金

        Returns:
            {symbol: ParallelBacktestResult}
        """
        tasks = []
        for symbol in symbols:
            task = ParallelBacktestTask(
                task_id=f"{symbol}_{period}",
                strategy_class=strategy_class,
                strategy_params=strategy_params,
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital
            )
            tasks.append(task)

        results = self._run_tasks(tasks)

        return {r.symbol: r for r in results}

    def run_param_scan(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        symbol: str,
        period: str = "1d",
        start_date: datetime = None,
        end_date: datetime = None,
        initial_capital: float = 100000.0
    ) -> List[ParallelBacktestResult]:
        """
        参数扫描并行回测

        Args:
            strategy_class: 策略类
            param_grid: 参数网格 {param_name: [values]}
            symbol: 品种
            period: K线周期
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金

        Returns:
            List[ParallelBacktestResult] 按sharpe_ratio降序
        """
        from itertools import product

        # 生成参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        tasks = []
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            task = ParallelBacktestTask(
                task_id=f"scan_{i}",
                strategy_class=strategy_class,
                strategy_params=params,
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital
            )
            tasks.append(task)

        logger.info(f"参数扫描: {len(tasks)} 种组合")

        results = self._run_tasks(tasks)

        # 按sharpe_ratio排序
        results.sort(key=lambda x: x.sharpe_ratio if x.success else float('-inf'), reverse=True)

        return results

    def run_full_scan(
        self,
        strategy_class: type,
        param_grid: Dict[str, List[Any]],
        symbols: List[str],
        period: str = "1d",
        start_date: datetime = None,
        end_date: datetime = None,
        initial_capital: float = 100000.0
    ) -> pd.DataFrame:
        """
        全量扫描（多品种 x 多参数）

        Returns:
            DataFrame with all results
        """
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        tasks = []
        for symbol in symbols:
            for i, combo in enumerate(combinations):
                params = dict(zip(param_names, combo))
                task = ParallelBacktestTask(
                    task_id=f"{symbol}_scan_{i}",
                    strategy_class=strategy_class,
                    strategy_params=params,
                    symbol=symbol,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital
                )
                tasks.append(task)

        logger.info(f"全量扫描: {len(symbols)} 品种 x {len(combinations)} 参数 = {len(tasks)} 任务")

        results = self._run_tasks(tasks)

        # 转换为DataFrame
        rows = []
        for r in results:
            row = {
                'symbol': r.symbol,
                'success': r.success,
                'total_return': r.total_return,
                'annual_return': r.annual_return,
                'max_drawdown': r.max_drawdown,
                'sharpe_ratio': r.sharpe_ratio,
                'win_rate': r.win_rate,
                'total_trades': r.total_trades,
                'profit_factor': r.profit_factor,
                'error': r.error
            }
            row.update(r.params)
            rows.append(row)

        return pd.DataFrame(rows)

    def _run_tasks(self, tasks: List[ParallelBacktestTask]) -> List[ParallelBacktestResult]:
        """执行任务列表"""
        results = []
        total = len(tasks)

        if total == 0:
            return results

        # 单任务直接执行
        if total == 1:
            result = _run_single_backtest(tasks[0])
            if self._progress_callback:
                self._progress_callback(1, 1, f"完成: {tasks[0].symbol}")
            return [result]

        # 多任务并行
        ExecutorClass = ProcessPoolExecutor if self.use_process else ThreadPoolExecutor

        with ExecutorClass(max_workers=min(self.max_workers, total)) as executor:
            futures = {executor.submit(_run_single_backtest, task): task for task in tasks}

            completed = 0
            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"任务 {task.task_id} 执行异常: {e}")
                    results.append(ParallelBacktestResult(
                        task_id=task.task_id,
                        symbol=task.symbol,
                        params=task.strategy_params,
                        success=False,
                        error=str(e)
                    ))

                completed += 1
                if self._progress_callback:
                    self._progress_callback(
                        completed, total,
                        f"进度: {completed}/{total} - {task.symbol}"
                    )

        return results


def run_parallel_backtest(
    strategy_class: type,
    symbols: List[str],
    params: Dict[str, Any] = None,
    period: str = "1d",
    start_date: datetime = None,
    end_date: datetime = None,
    initial_capital: float = 100000.0,
    max_workers: int = None
) -> Dict[str, ParallelBacktestResult]:
    """
    便捷函数：并行回测多品种

    Args:
        strategy_class: 策略类
        symbols: 品种列表
        params: 策略参数
        period: K线周期
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金
        max_workers: 最大并行数

    Returns:
        {symbol: ParallelBacktestResult}
    """
    runner = ParallelBacktestRunner(max_workers=max_workers)
    return runner.run_multi_symbol(
        strategy_class=strategy_class,
        strategy_params=params or {},
        symbols=symbols,
        period=period,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital
    )
