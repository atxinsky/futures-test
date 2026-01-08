# coding=utf-8
"""
滑点统计与分析模块
记录实盘滑点数据，用于回测时更真实地模拟交易成本
"""

import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from threading import Lock
from pathlib import Path
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


@dataclass
class SlippageRecord:
    """滑点记录"""
    trade_id: str
    symbol: str
    direction: str          # LONG/SHORT
    offset: str             # OPEN/CLOSE
    expected_price: float   # 预期成交价（信号价格）
    actual_price: float     # 实际成交价
    volume: int
    slippage: float         # 滑点金额 = (actual - expected) * direction_sign
    slippage_ticks: float   # 滑点跳数
    slippage_pct: float     # 滑点百分比
    trade_time: datetime
    market_condition: str = ""  # 市场状态：normal, volatile, limit_up, limit_down


@dataclass
class SlippageStats:
    """滑点统计"""
    symbol: str
    sample_count: int = 0

    # 基础统计
    avg_slippage: float = 0.0           # 平均滑点（金额）
    avg_slippage_ticks: float = 0.0     # 平均滑点（跳数）
    avg_slippage_pct: float = 0.0       # 平均滑点（百分比）

    # 分布统计
    median_slippage_ticks: float = 0.0  # 中位数滑点
    std_slippage_ticks: float = 0.0     # 滑点标准差
    max_slippage_ticks: float = 0.0     # 最大滑点
    min_slippage_ticks: float = 0.0     # 最小滑点

    # 方向统计
    positive_rate: float = 0.0          # 正滑点比例（对我不利）
    negative_rate: float = 0.0          # 负滑点比例（对我有利）
    zero_rate: float = 0.0              # 零滑点比例

    # 分类统计
    open_avg_ticks: float = 0.0         # 开仓平均滑点
    close_avg_ticks: float = 0.0        # 平仓平均滑点

    update_time: datetime = field(default_factory=datetime.now)


class SlippageTracker:
    """
    滑点追踪器

    功能:
    1. 记录每笔交易的滑点
    2. 统计分析滑点分布
    3. 提供回测滑点模型参数
    4. 持久化存储历史数据
    """

    def __init__(self, db_path: str = None):
        """
        初始化滑点追踪器

        Args:
            db_path: 数据库路径，默认 data/slippage.db
        """
        if db_path is None:
            base_dir = Path(__file__).parent.parent / 'data'
            base_dir.mkdir(exist_ok=True)
            db_path = str(base_dir / 'slippage.db')

        self.db_path = db_path
        self._lock = Lock()
        self._cache: Dict[str, List[SlippageRecord]] = defaultdict(list)
        self._stats_cache: Dict[str, SlippageStats] = {}

        self._init_db()
        logger.info(f"滑点追踪器初始化: {self.db_path}")

    def _get_connection(self):
        """获取数据库连接"""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """初始化数据库"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # 滑点记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS slippage_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    offset TEXT NOT NULL,
                    expected_price REAL NOT NULL,
                    actual_price REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    slippage REAL NOT NULL,
                    slippage_ticks REAL NOT NULL,
                    slippage_pct REAL NOT NULL,
                    trade_time TEXT NOT NULL,
                    market_condition TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_slippage_symbol
                ON slippage_records(symbol)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_slippage_time
                ON slippage_records(trade_time)
            ''')

            conn.commit()
        finally:
            conn.close()

    def record_slippage(
        self,
        trade_id: str,
        symbol: str,
        direction: str,
        offset: str,
        expected_price: float,
        actual_price: float,
        volume: int,
        price_tick: float = 1.0,
        market_condition: str = ""
    ) -> SlippageRecord:
        """
        记录一笔交易的滑点

        Args:
            trade_id: 成交ID
            symbol: 品种代码
            direction: 方向 (LONG/SHORT)
            offset: 开平 (OPEN/CLOSE)
            expected_price: 预期价格（信号价格）
            actual_price: 实际成交价
            volume: 成交量
            price_tick: 最小变动价位
            market_condition: 市场状态

        Returns:
            SlippageRecord
        """
        # 计算滑点
        # 买入时：实际价 > 预期价 = 正滑点（不利）
        # 卖出时：实际价 < 预期价 = 正滑点（不利）
        price_diff = actual_price - expected_price

        if direction == "LONG":
            # 买入：高于预期为正滑点
            if offset == "OPEN":
                slippage = price_diff  # 开多：买入
            else:
                slippage = -price_diff  # 平多：卖出
        else:
            # 卖出：低于预期为正滑点
            if offset == "OPEN":
                slippage = -price_diff  # 开空：卖出
            else:
                slippage = price_diff  # 平空：买入

        slippage_ticks = slippage / price_tick if price_tick > 0 else 0
        slippage_pct = slippage / expected_price * 100 if expected_price > 0 else 0

        record = SlippageRecord(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            offset=offset,
            expected_price=expected_price,
            actual_price=actual_price,
            volume=volume,
            slippage=slippage,
            slippage_ticks=slippage_ticks,
            slippage_pct=slippage_pct,
            trade_time=datetime.now(),
            market_condition=market_condition
        )

        # 保存到数据库
        self._save_record(record)

        # 更新缓存
        with self._lock:
            self._cache[symbol].append(record)
            # 限制缓存大小
            if len(self._cache[symbol]) > 1000:
                self._cache[symbol] = self._cache[symbol][-500:]
            # 清除统计缓存
            self._stats_cache.pop(symbol, None)

        logger.debug(f"记录滑点: {symbol} {direction} {offset} "
                    f"预期={expected_price:.2f} 实际={actual_price:.2f} "
                    f"滑点={slippage_ticks:.1f}跳")

        return record

    def _save_record(self, record: SlippageRecord):
        """保存记录到数据库"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO slippage_records
                (trade_id, symbol, direction, offset, expected_price, actual_price,
                 volume, slippage, slippage_ticks, slippage_pct, trade_time, market_condition)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record.trade_id,
                record.symbol,
                record.direction,
                record.offset,
                record.expected_price,
                record.actual_price,
                record.volume,
                record.slippage,
                record.slippage_ticks,
                record.slippage_pct,
                record.trade_time.isoformat(),
                record.market_condition
            ))
            conn.commit()
        finally:
            conn.close()

    def get_stats(self, symbol: str, days: int = 30) -> SlippageStats:
        """
        获取品种的滑点统计

        Args:
            symbol: 品种代码
            days: 统计最近N天

        Returns:
            SlippageStats
        """
        # 检查缓存
        cache_key = f"{symbol}_{days}"
        with self._lock:
            if cache_key in self._stats_cache:
                cached = self._stats_cache[cache_key]
                # 缓存5分钟有效
                if (datetime.now() - cached.update_time).seconds < 300:
                    return cached

        # 从数据库查询
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM slippage_records
                WHERE symbol = ? AND trade_time >= ?
                ORDER BY trade_time DESC
            ''', (symbol, cutoff))

            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            return SlippageStats(symbol=symbol, sample_count=0)

        # 计算统计
        slippage_ticks = [row['slippage_ticks'] for row in rows]
        slippage_pcts = [row['slippage_pct'] for row in rows]

        open_ticks = [row['slippage_ticks'] for row in rows if row['offset'] == 'OPEN']
        close_ticks = [row['slippage_ticks'] for row in rows if row['offset'] == 'CLOSE']

        positive_count = sum(1 for t in slippage_ticks if t > 0)
        negative_count = sum(1 for t in slippage_ticks if t < 0)
        zero_count = sum(1 for t in slippage_ticks if t == 0)
        total = len(slippage_ticks)

        stats = SlippageStats(
            symbol=symbol,
            sample_count=total,
            avg_slippage=sum(row['slippage'] for row in rows) / total,
            avg_slippage_ticks=statistics.mean(slippage_ticks),
            avg_slippage_pct=statistics.mean(slippage_pcts),
            median_slippage_ticks=statistics.median(slippage_ticks),
            std_slippage_ticks=statistics.stdev(slippage_ticks) if total > 1 else 0,
            max_slippage_ticks=max(slippage_ticks),
            min_slippage_ticks=min(slippage_ticks),
            positive_rate=positive_count / total,
            negative_rate=negative_count / total,
            zero_rate=zero_count / total,
            open_avg_ticks=statistics.mean(open_ticks) if open_ticks else 0,
            close_avg_ticks=statistics.mean(close_ticks) if close_ticks else 0,
            update_time=datetime.now()
        )

        # 更新缓存
        with self._lock:
            self._stats_cache[cache_key] = stats

        return stats

    def get_slippage_model(self, symbol: str, days: int = 30) -> dict:
        """
        获取滑点模型参数（用于回测）

        Args:
            symbol: 品种代码
            days: 统计天数

        Returns:
            {
                'mode': 'statistical',  # 统计模型
                'avg_ticks': 1.5,       # 平均滑点跳数
                'std_ticks': 0.8,       # 标准差
                'open_ticks': 1.2,      # 开仓滑点
                'close_ticks': 1.8,     # 平仓滑点
                'sample_count': 100,    # 样本数
                'confidence': 'high'    # 置信度
            }
        """
        stats = self.get_stats(symbol, days)

        # 确定置信度
        if stats.sample_count >= 100:
            confidence = 'high'
        elif stats.sample_count >= 30:
            confidence = 'medium'
        elif stats.sample_count >= 10:
            confidence = 'low'
        else:
            confidence = 'insufficient'

        return {
            'mode': 'statistical',
            'avg_ticks': stats.avg_slippage_ticks,
            'std_ticks': stats.std_slippage_ticks,
            'median_ticks': stats.median_slippage_ticks,
            'open_ticks': stats.open_avg_ticks,
            'close_ticks': stats.close_avg_ticks,
            'max_ticks': stats.max_slippage_ticks,
            'sample_count': stats.sample_count,
            'confidence': confidence,
            'positive_rate': stats.positive_rate
        }

    def get_all_symbols_stats(self, days: int = 30) -> Dict[str, SlippageStats]:
        """获取所有品种的滑点统计"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT symbol FROM slippage_records')
            symbols = [row['symbol'] for row in cursor.fetchall()]
        finally:
            conn.close()

        return {symbol: self.get_stats(symbol, days) for symbol in symbols}

    def get_recent_records(self, symbol: str = None, limit: int = 50) -> List[SlippageRecord]:
        """获取最近的滑点记录"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if symbol:
                cursor.execute('''
                    SELECT * FROM slippage_records
                    WHERE symbol = ?
                    ORDER BY trade_time DESC
                    LIMIT ?
                ''', (symbol, limit))
            else:
                cursor.execute('''
                    SELECT * FROM slippage_records
                    ORDER BY trade_time DESC
                    LIMIT ?
                ''', (limit,))

            records = []
            for row in cursor.fetchall():
                records.append(SlippageRecord(
                    trade_id=row['trade_id'],
                    symbol=row['symbol'],
                    direction=row['direction'],
                    offset=row['offset'],
                    expected_price=row['expected_price'],
                    actual_price=row['actual_price'],
                    volume=row['volume'],
                    slippage=row['slippage'],
                    slippage_ticks=row['slippage_ticks'],
                    slippage_pct=row['slippage_pct'],
                    trade_time=datetime.fromisoformat(row['trade_time']),
                    market_condition=row['market_condition'] or ''
                ))
            return records
        finally:
            conn.close()

    def generate_report(self, days: int = 30) -> str:
        """生成滑点分析报告"""
        all_stats = self.get_all_symbols_stats(days)

        if not all_stats:
            return "暂无滑点数据"

        report = f"# 滑点分析报告 (最近{days}天)\n\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        report += "## 品种滑点统计\n\n"
        report += "| 品种 | 样本数 | 平均滑点(跳) | 中位数 | 标准差 | 最大 | 正滑点率 |\n"
        report += "|------|--------|--------------|--------|--------|------|----------|\n"

        for symbol, stats in sorted(all_stats.items()):
            if stats.sample_count > 0:
                report += f"| {symbol} | {stats.sample_count} | "
                report += f"{stats.avg_slippage_ticks:.2f} | "
                report += f"{stats.median_slippage_ticks:.2f} | "
                report += f"{stats.std_slippage_ticks:.2f} | "
                report += f"{stats.max_slippage_ticks:.1f} | "
                report += f"{stats.positive_rate*100:.1f}% |\n"

        report += "\n## 开仓/平仓对比\n\n"
        report += "| 品种 | 开仓滑点(跳) | 平仓滑点(跳) | 差异 |\n"
        report += "|------|--------------|--------------|------|\n"

        for symbol, stats in sorted(all_stats.items()):
            if stats.sample_count > 0:
                diff = stats.close_avg_ticks - stats.open_avg_ticks
                report += f"| {symbol} | {stats.open_avg_ticks:.2f} | "
                report += f"{stats.close_avg_ticks:.2f} | "
                report += f"{diff:+.2f} |\n"

        report += "\n## 回测建议\n\n"
        for symbol, stats in sorted(all_stats.items()):
            if stats.sample_count >= 30:
                model = self.get_slippage_model(symbol, days)
                report += f"- **{symbol}**: 建议使用 {model['avg_ticks']:.1f} 跳滑点 "
                report += f"(置信度: {model['confidence']})\n"

        return report


# 全局单例
_slippage_tracker: Optional[SlippageTracker] = None


def get_slippage_tracker() -> SlippageTracker:
    """获取滑点追踪器单例"""
    global _slippage_tracker
    if _slippage_tracker is None:
        _slippage_tracker = SlippageTracker()
    return _slippage_tracker


# ============ 回测滑点模拟器 ============

class SlippageSimulator:
    """
    滑点模拟器（用于回测）

    支持多种模拟模式:
    1. fixed: 固定滑点
    2. random: 随机滑点
    3. statistical: 基于历史统计
    """

    def __init__(self, mode: str = 'fixed', default_ticks: float = 1.0):
        """
        初始化滑点模拟器

        Args:
            mode: 模拟模式 (fixed/random/statistical)
            default_ticks: 默认滑点跳数
        """
        self.mode = mode
        self.default_ticks = default_ticks
        self._symbol_models: Dict[str, dict] = {}

        # 如果是统计模式，加载历史数据
        if mode == 'statistical':
            self._load_statistical_models()

    def _load_statistical_models(self):
        """加载统计模型"""
        try:
            tracker = get_slippage_tracker()
            all_stats = tracker.get_all_symbols_stats(days=30)

            for symbol, stats in all_stats.items():
                if stats.sample_count >= 10:
                    self._symbol_models[symbol] = tracker.get_slippage_model(symbol)

            logger.info(f"加载滑点统计模型: {len(self._symbol_models)} 个品种")
        except Exception as e:
            logger.warning(f"加载滑点统计模型失败: {e}")

    def simulate(
        self,
        symbol: str,
        price: float,
        direction: str,
        offset: str,
        price_tick: float = 1.0
    ) -> float:
        """
        模拟滑点后的成交价

        Args:
            symbol: 品种代码
            price: 原始价格
            direction: 方向 (LONG/SHORT)
            offset: 开平 (OPEN/CLOSE)
            price_tick: 最小变动价位

        Returns:
            模拟滑点后的价格
        """
        import random

        # 获取滑点跳数
        if self.mode == 'fixed':
            slippage_ticks = self.default_ticks

        elif self.mode == 'random':
            # 随机滑点 [0, 2*default]
            slippage_ticks = random.uniform(0, self.default_ticks * 2)

        elif self.mode == 'statistical':
            model = self._symbol_models.get(symbol)
            if model and model['sample_count'] >= 10:
                # 使用正态分布模拟
                avg = model['open_ticks'] if offset == 'OPEN' else model['close_ticks']
                std = model['std_ticks']
                slippage_ticks = max(0, random.gauss(avg, std))
            else:
                slippage_ticks = self.default_ticks
        else:
            slippage_ticks = self.default_ticks

        # 计算滑点金额
        slippage = slippage_ticks * price_tick

        # 应用滑点（总是对交易者不利的方向）
        if direction == "LONG":
            if offset == "OPEN":
                return price + slippage  # 买入时价格上滑
            else:
                return price - slippage  # 卖出时价格下滑
        else:
            if offset == "OPEN":
                return price - slippage  # 卖出时价格下滑
            else:
                return price + slippage  # 买入时价格上滑

    def get_model_info(self, symbol: str) -> Optional[dict]:
        """获取品种的滑点模型信息"""
        return self._symbol_models.get(symbol)
