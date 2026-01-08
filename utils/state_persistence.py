# coding=utf-8
"""
状态持久化模块
将持仓、订单等关键状态持久化到SQLite，防止程序崩溃后状态丢失
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from threading import Lock
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class StatePersistence:
    """
    状态持久化管理器

    功能:
    1. 持仓状态持久化
    2. 订单状态持久化
    3. 账户状态持久化
    4. 崩溃恢复
    """

    def __init__(self, db_path: str = None):
        """
        初始化

        Args:
            db_path: 数据库路径，默认为 data/state.db
        """
        if db_path is None:
            base_dir = Path(__file__).parent.parent / 'data'
            base_dir.mkdir(exist_ok=True)
            db_path = str(base_dir / 'state.db')

        self.db_path = db_path
        self._lock = Lock()
        self._init_db()

        logger.info(f"状态持久化初始化: {self.db_path}")

    @contextmanager
    def _get_connection(self):
        """获取数据库连接（上下文管理器）"""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _init_db(self):
        """初始化数据库表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 持仓表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    exchange TEXT,
                    direction TEXT NOT NULL,
                    volume INTEGER NOT NULL,
                    avg_price REAL NOT NULL,
                    margin REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    highest_price REAL DEFAULT 0,
                    lowest_price REAL DEFAULT 0,
                    entry_time TEXT,
                    strategy_name TEXT,
                    update_time TEXT NOT NULL,
                    UNIQUE(symbol, direction)
                )
            ''')

            # 订单表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    exchange TEXT,
                    direction TEXT NOT NULL,
                    offset TEXT NOT NULL,
                    price REAL,
                    volume INTEGER NOT NULL,
                    traded INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    order_time TEXT,
                    strategy_name TEXT,
                    update_time TEXT NOT NULL
                )
            ''')

            # 账户表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS account (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    account_id TEXT,
                    balance REAL DEFAULT 0,
                    available REAL DEFAULT 0,
                    margin REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    update_time TEXT NOT NULL
                )
            ''')

            # 成交记录表（用于对账）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    order_id TEXT,
                    symbol TEXT NOT NULL,
                    exchange TEXT,
                    direction TEXT NOT NULL,
                    offset TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    trade_time TEXT NOT NULL,
                    strategy_name TEXT
                )
            ''')

            # 系统状态表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_state (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    update_time TEXT NOT NULL
                )
            ''')

            logger.debug("数据库表初始化完成")

    # ============ 持仓操作 ============

    def save_position(self, position_data: dict):
        """
        保存单个持仓

        Args:
            position_data: 持仓数据字典
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO positions
                    (symbol, exchange, direction, volume, avg_price, margin,
                     unrealized_pnl, realized_pnl, highest_price, lowest_price,
                     entry_time, strategy_name, update_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    position_data.get('symbol'),
                    position_data.get('exchange', ''),
                    position_data.get('direction'),
                    position_data.get('volume', 0),
                    position_data.get('avg_price', 0),
                    position_data.get('margin', 0),
                    position_data.get('unrealized_pnl', 0),
                    position_data.get('realized_pnl', 0),
                    position_data.get('highest_price', 0),
                    position_data.get('lowest_price', 0),
                    position_data.get('entry_time', ''),
                    position_data.get('strategy_name', ''),
                    datetime.now().isoformat()
                ))

        logger.debug(f"保存持仓: {position_data.get('symbol')} {position_data.get('direction')}")

    def save_all_positions(self, positions: List[dict]):
        """批量保存持仓"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # 先清空现有持仓
                cursor.execute('DELETE FROM positions')

                # 批量插入
                for pos in positions:
                    cursor.execute('''
                        INSERT INTO positions
                        (symbol, exchange, direction, volume, avg_price, margin,
                         unrealized_pnl, realized_pnl, highest_price, lowest_price,
                         entry_time, strategy_name, update_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        pos.get('symbol'),
                        pos.get('exchange', ''),
                        pos.get('direction'),
                        pos.get('volume', 0),
                        pos.get('avg_price', 0),
                        pos.get('margin', 0),
                        pos.get('unrealized_pnl', 0),
                        pos.get('realized_pnl', 0),
                        pos.get('highest_price', 0),
                        pos.get('lowest_price', 0),
                        pos.get('entry_time', ''),
                        pos.get('strategy_name', ''),
                        datetime.now().isoformat()
                    ))

        logger.info(f"批量保存 {len(positions)} 个持仓")

    def load_positions(self) -> List[dict]:
        """加载所有持仓"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM positions WHERE volume > 0')
                rows = cursor.fetchall()

                positions = []
                for row in rows:
                    positions.append({
                        'symbol': row['symbol'],
                        'exchange': row['exchange'],
                        'direction': row['direction'],
                        'volume': row['volume'],
                        'avg_price': row['avg_price'],
                        'margin': row['margin'],
                        'unrealized_pnl': row['unrealized_pnl'],
                        'realized_pnl': row['realized_pnl'],
                        'highest_price': row['highest_price'],
                        'lowest_price': row['lowest_price'],
                        'entry_time': row['entry_time'],
                        'strategy_name': row['strategy_name']
                    })

                logger.info(f"加载 {len(positions)} 个持仓")
                return positions

    def delete_position(self, symbol: str, direction: str):
        """删除持仓记录"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'DELETE FROM positions WHERE symbol = ? AND direction = ?',
                    (symbol, direction)
                )
        logger.debug(f"删除持仓: {symbol} {direction}")

    # ============ 订单操作 ============

    def save_order(self, order_data: dict):
        """保存订单"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO orders
                    (order_id, symbol, exchange, direction, offset, price,
                     volume, traded, status, order_time, strategy_name, update_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    order_data.get('order_id'),
                    order_data.get('symbol'),
                    order_data.get('exchange', ''),
                    order_data.get('direction'),
                    order_data.get('offset'),
                    order_data.get('price', 0),
                    order_data.get('volume', 0),
                    order_data.get('traded', 0),
                    order_data.get('status'),
                    order_data.get('order_time', ''),
                    order_data.get('strategy_name', ''),
                    datetime.now().isoformat()
                ))

    def load_active_orders(self) -> List[dict]:
        """加载活跃订单"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM orders
                    WHERE status IN ('SUBMITTING', 'SUBMITTED', 'PARTIAL')
                ''')
                rows = cursor.fetchall()

                orders = []
                for row in rows:
                    orders.append(dict(row))

                return orders

    def clear_orders(self):
        """清空订单表（日终清理）"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM orders')

    # ============ 成交操作 ============

    def save_trade(self, trade_data: dict):
        """保存成交记录"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR IGNORE INTO trades
                    (trade_id, order_id, symbol, exchange, direction, offset,
                     price, volume, trade_time, strategy_name)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data.get('trade_id'),
                    trade_data.get('order_id', ''),
                    trade_data.get('symbol'),
                    trade_data.get('exchange', ''),
                    trade_data.get('direction'),
                    trade_data.get('offset'),
                    trade_data.get('price', 0),
                    trade_data.get('volume', 0),
                    trade_data.get('trade_time', datetime.now().isoformat()),
                    trade_data.get('strategy_name', '')
                ))

    def get_today_trades(self) -> List[dict]:
        """获取今日成交"""
        today = datetime.now().strftime('%Y-%m-%d')
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    'SELECT * FROM trades WHERE trade_time LIKE ?',
                    (f'{today}%',)
                )
                return [dict(row) for row in cursor.fetchall()]

    # ============ 账户操作 ============

    def save_account(self, account_data: dict):
        """保存账户状态"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO account
                    (id, account_id, balance, available, margin,
                     unrealized_pnl, realized_pnl, update_time)
                    VALUES (1, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    account_data.get('account_id', ''),
                    account_data.get('balance', 0),
                    account_data.get('available', 0),
                    account_data.get('margin', 0),
                    account_data.get('unrealized_pnl', 0),
                    account_data.get('realized_pnl', 0),
                    datetime.now().isoformat()
                ))

    def load_account(self) -> Optional[dict]:
        """加载账户状态"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM account WHERE id = 1')
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None

    # ============ 系统状态 ============

    def set_state(self, key: str, value: Any):
        """设置系统状态"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO system_state (key, value, update_time)
                    VALUES (?, ?, ?)
                ''', (key, json.dumps(value), datetime.now().isoformat()))

    def get_state(self, key: str, default: Any = None) -> Any:
        """获取系统状态"""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT value FROM system_state WHERE key = ?', (key,))
                row = cursor.fetchone()
                if row:
                    return json.loads(row['value'])
                return default

    def get_last_sync_time(self) -> Optional[datetime]:
        """获取上次同步时间"""
        time_str = self.get_state('last_sync_time')
        if time_str:
            return datetime.fromisoformat(time_str)
        return None

    def set_last_sync_time(self, dt: datetime = None):
        """设置上次同步时间"""
        if dt is None:
            dt = datetime.now()
        self.set_state('last_sync_time', dt.isoformat())


# 全局单例
_persistence: Optional[StatePersistence] = None


def get_state_persistence(db_path: str = None) -> StatePersistence:
    """获取状态持久化单例"""
    global _persistence
    if _persistence is None:
        _persistence = StatePersistence(db_path)
    return _persistence
