# coding=utf-8
"""
回测记录存储服务
支持ETF回测和期货回测的结果持久化
"""

import sqlite3
import json
import hashlib
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BacktestRecord:
    """回测记录"""
    id: int = 0
    backtest_id: str = ""  # 唯一标识 (hash)
    backtest_type: str = ""  # ETF / 期货
    strategy_name: str = ""
    symbols: str = ""  # 逗号分隔的标的列表
    start_date: str = ""
    end_date: str = ""
    initial_capital: float = 0.0
    final_value: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    params_json: str = "{}"  # 策略参数
    result_json: str = "{}"  # 完整结果
    trades_json: str = "[]"  # 交易记录
    equity_csv: str = ""  # 资金曲线CSV
    notes: str = ""  # 用户备注
    created_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'BacktestRecord':
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class BacktestStorage:
    """
    回测记录存储服务

    使用SQLite存储回测历史记录，支持：
    - 保存回测结果
    - 查询历史记录
    - 对比多个回测
    - 导出记录
    """

    def __init__(self, db_path: str = None):
        """
        初始化存储服务

        Args:
            db_path: 数据库路径，默认为 data/backtest_history.db
        """
        if db_path is None:
            base_dir = Path(__file__).parent.parent / "data"
            base_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(base_dir / "backtest_history.db")

        self.db_path = db_path
        self._init_db()
        logger.info(f"回测记录存储初始化: {db_path}")

    def _init_db(self):
        """初始化数据库表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id TEXT UNIQUE NOT NULL,
                backtest_type TEXT NOT NULL,
                strategy_name TEXT NOT NULL,
                symbols TEXT,
                start_date TEXT,
                end_date TEXT,
                initial_capital REAL,
                final_value REAL,
                total_return REAL,
                annual_return REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                win_rate REAL,
                total_trades INTEGER,
                params_json TEXT,
                result_json TEXT,
                trades_json TEXT,
                equity_csv TEXT,
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
        ''')

        # 创建索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_backtest_type ON backtest_records(backtest_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_name ON backtest_records(strategy_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON backtest_records(created_at)')

        conn.commit()
        conn.close()

    def _generate_backtest_id(self, strategy_name: str, symbols: str, start_date: str,
                               end_date: str, params: dict) -> str:
        """生成回测唯一标识"""
        content = f"{strategy_name}_{symbols}_{start_date}_{end_date}_{json.dumps(params, sort_keys=True)}_{datetime.now().timestamp()}"
        return hashlib.md5(content.encode()).hexdigest()[:10]

    def save_etf_backtest(self, result, strategy_name: str, symbols: List[str],
                          params: dict, notes: str = "") -> str:
        """
        保存ETF回测结果

        Args:
            result: ETFBacktestResult对象
            strategy_name: 策略名称
            symbols: 标的列表
            params: 策略参数
            notes: 备注

        Returns:
            backtest_id
        """
        symbols_str = ",".join(symbols)
        backtest_id = self._generate_backtest_id(strategy_name, symbols_str,
                                                  result.start_date, result.end_date, params)

        # 构建交易记录JSON
        trades_data = []
        if result.trades:
            for t in result.trades:
                trades_data.append({
                    "date": t.date,
                    "code": t.code,
                    "direction": t.direction,
                    "price": t.price,
                    "shares": t.shares,
                    "amount": t.amount,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct
                })

        # 构建资金曲线CSV
        equity_csv = ""
        if result.equity_curve is not None:
            equity_csv = result.equity_curve.to_csv()

        # 构建完整结果JSON
        result_data = {
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "benchmark_return": result.benchmark_return,
            "excess_return": result.excess_return,
            "max_drawdown": result.max_drawdown,
            "max_drawdown_duration": result.max_drawdown_duration,
            "sharpe_ratio": result.sharpe_ratio,
            "calmar_ratio": result.calmar_ratio,
            "volatility": result.volatility,
            "win_rate": result.win_rate,
            "profit_loss_ratio": result.profit_loss_ratio,
            "total_trades": result.total_trades,
            "win_trades": result.win_trades,
            "lose_trades": result.lose_trades,
            "avg_win": result.avg_win,
            "avg_loss": result.avg_loss,
            "avg_holding_days": result.avg_holding_days
        }

        record = BacktestRecord(
            backtest_id=backtest_id,
            backtest_type="ETF",
            strategy_name=strategy_name,
            symbols=symbols_str,
            start_date=result.start_date,
            end_date=result.end_date,
            initial_capital=result.initial_capital,
            final_value=result.final_value,
            total_return=result.total_return,
            annual_return=result.annual_return,
            max_drawdown=result.max_drawdown,
            sharpe_ratio=result.sharpe_ratio,
            win_rate=result.win_rate,
            total_trades=result.total_trades,
            params_json=json.dumps(params, ensure_ascii=False),
            result_json=json.dumps(result_data, ensure_ascii=False),
            trades_json=json.dumps(trades_data, ensure_ascii=False),
            equity_csv=equity_csv,
            notes=notes,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        self._save_record(record)
        logger.info(f"ETF回测已保存: {backtest_id}")
        return backtest_id

    def save_futures_backtest(self, result: dict, strategy_name: str, symbols: List[str],
                               params: dict, start_date: str, end_date: str,
                               initial_capital: float, notes: str = "") -> str:
        """
        保存期货回测结果

        Args:
            result: 回测结果字典
            strategy_name: 策略名称
            symbols: 品种列表
            params: 策略参数
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            notes: 备注

        Returns:
            backtest_id
        """
        symbols_str = ",".join(symbols)
        backtest_id = self._generate_backtest_id(strategy_name, symbols_str,
                                                  start_date, end_date, params)

        # 构建交易记录JSON
        trades_data = []
        trades = result.get('trades', [])
        if trades:
            for t in trades:
                if isinstance(t, dict):
                    trades_data.append(t)
                elif hasattr(t, 'to_dict'):
                    trades_data.append(t.to_dict())
                else:
                    trades_data.append(vars(t))

        # 构建资金曲线CSV
        equity_csv = ""
        equity_curve = result.get('equity_curve')
        if equity_curve is not None:
            import pandas as pd
            if isinstance(equity_curve, pd.DataFrame):
                equity_csv = equity_curve.to_csv()

        final_value = result.get('final_equity', result.get('final_value', initial_capital))
        total_return = result.get('total_return_pct', result.get('total_return', 0))

        record = BacktestRecord(
            backtest_id=backtest_id,
            backtest_type="期货",
            strategy_name=strategy_name,
            symbols=symbols_str,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_value=final_value,
            total_return=total_return,
            annual_return=result.get('annual_return', 0),
            max_drawdown=result.get('max_drawdown_pct', result.get('max_drawdown', 0)),
            sharpe_ratio=result.get('sharpe_ratio', 0),
            win_rate=result.get('win_rate', 0),
            total_trades=result.get('total_trades', 0),
            params_json=json.dumps(params, ensure_ascii=False),
            result_json=json.dumps(result, ensure_ascii=False, default=str),
            trades_json=json.dumps(trades_data, ensure_ascii=False, default=str),
            equity_csv=equity_csv,
            notes=notes,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

        self._save_record(record)
        logger.info(f"期货回测已保存: {backtest_id}")
        return backtest_id

    def _save_record(self, record: BacktestRecord):
        """保存记录到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT OR REPLACE INTO backtest_records
            (backtest_id, backtest_type, strategy_name, symbols, start_date, end_date,
             initial_capital, final_value, total_return, annual_return, max_drawdown,
             sharpe_ratio, win_rate, total_trades, params_json, result_json,
             trades_json, equity_csv, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            record.backtest_id, record.backtest_type, record.strategy_name, record.symbols,
            record.start_date, record.end_date, record.initial_capital, record.final_value,
            record.total_return, record.annual_return, record.max_drawdown, record.sharpe_ratio,
            record.win_rate, record.total_trades, record.params_json, record.result_json,
            record.trades_json, record.equity_csv, record.notes, record.created_at
        ))

        conn.commit()
        conn.close()

    def get_record(self, backtest_id: str) -> Optional[BacktestRecord]:
        """获取单条回测记录"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM backtest_records WHERE backtest_id = ?', (backtest_id,))
        row = cursor.fetchone()
        conn.close()

        if row:
            return BacktestRecord.from_dict(dict(row))
        return None

    def get_records(self, backtest_type: str = None, strategy_name: str = None,
                    limit: int = 100, offset: int = 0) -> List[BacktestRecord]:
        """
        获取回测记录列表

        Args:
            backtest_type: 回测类型筛选 (ETF/期货)
            strategy_name: 策略名称筛选
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            BacktestRecord列表
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = 'SELECT * FROM backtest_records WHERE 1=1'
        params = []

        if backtest_type:
            query += ' AND backtest_type = ?'
            params.append(backtest_type)

        if strategy_name:
            query += ' AND strategy_name = ?'
            params.append(strategy_name)

        query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [BacktestRecord.from_dict(dict(row)) for row in rows]

    def get_strategies(self, backtest_type: str = None) -> List[str]:
        """获取所有策略名称"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if backtest_type:
            cursor.execute('SELECT DISTINCT strategy_name FROM backtest_records WHERE backtest_type = ?',
                          (backtest_type,))
        else:
            cursor.execute('SELECT DISTINCT strategy_name FROM backtest_records')

        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def get_record_count(self, backtest_type: str = None) -> int:
        """获取记录总数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if backtest_type:
            cursor.execute('SELECT COUNT(*) FROM backtest_records WHERE backtest_type = ?',
                          (backtest_type,))
        else:
            cursor.execute('SELECT COUNT(*) FROM backtest_records')

        count = cursor.fetchone()[0]
        conn.close()
        return count

    def delete_record(self, backtest_id: str) -> bool:
        """删除回测记录"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM backtest_records WHERE backtest_id = ?', (backtest_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()

        if deleted:
            logger.info(f"回测记录已删除: {backtest_id}")
        return deleted

    def update_notes(self, backtest_id: str, notes: str) -> bool:
        """更新回测备注"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('UPDATE backtest_records SET notes = ? WHERE backtest_id = ?',
                      (notes, backtest_id))
        updated = cursor.rowcount > 0

        conn.commit()
        conn.close()
        return updated

    def export_to_csv(self, backtest_ids: List[str] = None) -> str:
        """
        导出回测记录为CSV

        Args:
            backtest_ids: 指定要导出的回测ID列表，None则导出全部

        Returns:
            CSV字符串
        """
        import pandas as pd

        conn = sqlite3.connect(self.db_path)

        if backtest_ids:
            placeholders = ','.join(['?' for _ in backtest_ids])
            query = f'SELECT * FROM backtest_records WHERE backtest_id IN ({placeholders})'
            df = pd.read_sql_query(query, conn, params=backtest_ids)
        else:
            df = pd.read_sql_query('SELECT * FROM backtest_records ORDER BY created_at DESC', conn)

        conn.close()

        # 移除大字段
        export_cols = ['backtest_id', 'backtest_type', 'strategy_name', 'symbols',
                       'start_date', 'end_date', 'initial_capital', 'final_value',
                       'total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio',
                       'win_rate', 'total_trades', 'notes', 'created_at']

        return df[export_cols].to_csv(index=False)

    def compare_records(self, backtest_ids: List[str]) -> Dict[str, Any]:
        """
        对比多个回测记录

        Args:
            backtest_ids: 要对比的回测ID列表

        Returns:
            对比结果字典
        """
        records = [self.get_record(bid) for bid in backtest_ids]
        records = [r for r in records if r is not None]

        if len(records) < 2:
            return {"error": "需要至少2条记录进行对比"}

        comparison = {
            "records": [r.to_dict() for r in records],
            "metrics": {}
        }

        # 对比指标
        metrics = ['total_return', 'annual_return', 'max_drawdown', 'sharpe_ratio',
                   'win_rate', 'total_trades']

        for metric in metrics:
            values = [getattr(r, metric) for r in records]
            comparison["metrics"][metric] = {
                "values": values,
                "best_idx": values.index(max(values)) if metric != 'max_drawdown' else values.index(min(values)),
                "avg": sum(values) / len(values)
            }

        return comparison


# 全局单例
_storage: Optional[BacktestStorage] = None


def get_backtest_storage() -> BacktestStorage:
    """获取回测存储服务单例"""
    global _storage
    if _storage is None:
        _storage = BacktestStorage()
    return _storage
