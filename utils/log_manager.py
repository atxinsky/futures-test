# coding=utf-8
"""
日志管理模块
支持日志归档、清理、查询
"""

import os
import gzip
import shutil
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class LogFileInfo:
    """日志文件信息"""
    path: str
    name: str
    size_bytes: int
    modified_time: datetime
    is_archived: bool = False

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1024 / 1024

    @property
    def age_days(self) -> int:
        return (datetime.now() - self.modified_time).days


class LogManager:
    """
    日志管理器

    功能:
    1. 日志归档（gzip压缩）
    2. 过期日志清理
    3. 日志统计
    4. 日志查询
    """

    def __init__(
        self,
        log_dir: str = None,
        archive_dir: str = None,
        max_age_days: int = 30,
        max_size_mb: float = 100,
        archive_after_days: int = 7
    ):
        """
        初始化

        Args:
            log_dir: 日志目录
            archive_dir: 归档目录
            max_age_days: 最大保留天数
            max_size_mb: 最大总大小(MB)
            archive_after_days: 多少天后归档
        """
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'logs'
            )

        self.log_dir = log_dir
        self.archive_dir = archive_dir or os.path.join(log_dir, 'archive')
        self.max_age_days = max_age_days
        self.max_size_mb = max_size_mb
        self.archive_after_days = archive_after_days

        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.archive_dir, exist_ok=True)

    def get_log_files(self, include_archived: bool = False) -> List[LogFileInfo]:
        """
        获取日志文件列表

        Args:
            include_archived: 是否包含归档文件

        Returns:
            日志文件信息列表
        """
        files = []

        # 主日志目录
        if os.path.exists(self.log_dir):
            for name in os.listdir(self.log_dir):
                path = os.path.join(self.log_dir, name)
                if os.path.isfile(path) and name.endswith('.log'):
                    stat = os.stat(path)
                    files.append(LogFileInfo(
                        path=path,
                        name=name,
                        size_bytes=stat.st_size,
                        modified_time=datetime.fromtimestamp(stat.st_mtime),
                        is_archived=False
                    ))

        # 归档目录
        if include_archived and os.path.exists(self.archive_dir):
            for name in os.listdir(self.archive_dir):
                path = os.path.join(self.archive_dir, name)
                if os.path.isfile(path) and name.endswith('.gz'):
                    stat = os.stat(path)
                    files.append(LogFileInfo(
                        path=path,
                        name=name,
                        size_bytes=stat.st_size,
                        modified_time=datetime.fromtimestamp(stat.st_mtime),
                        is_archived=True
                    ))

        return sorted(files, key=lambda x: x.modified_time, reverse=True)

    def get_stats(self) -> Dict[str, Any]:
        """
        获取日志统计

        Returns:
            统计信息
        """
        files = self.get_log_files(include_archived=True)

        active_files = [f for f in files if not f.is_archived]
        archived_files = [f for f in files if f.is_archived]

        return {
            'total_files': len(files),
            'active_files': len(active_files),
            'archived_files': len(archived_files),
            'active_size_mb': sum(f.size_mb for f in active_files),
            'archived_size_mb': sum(f.size_mb for f in archived_files),
            'total_size_mb': sum(f.size_mb for f in files),
            'oldest_file': files[-1].name if files else None,
            'newest_file': files[0].name if files else None
        }

    def archive_old_logs(self) -> List[str]:
        """
        归档旧日志

        Returns:
            归档的文件列表
        """
        archived = []
        cutoff_date = datetime.now() - timedelta(days=self.archive_after_days)

        for log_file in self.get_log_files():
            if log_file.modified_time < cutoff_date:
                archived_path = self._archive_file(log_file.path)
                if archived_path:
                    archived.append(log_file.name)
                    logger.info(f"已归档: {log_file.name}")

        return archived

    def _archive_file(self, file_path: str) -> Optional[str]:
        """
        归档单个文件

        Args:
            file_path: 文件路径

        Returns:
            归档后的路径
        """
        try:
            file_name = os.path.basename(file_path)
            archive_path = os.path.join(self.archive_dir, f"{file_name}.gz")

            with open(file_path, 'rb') as f_in:
                with gzip.open(archive_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            # 删除原文件
            os.remove(file_path)

            return archive_path

        except Exception as e:
            logger.error(f"归档失败 {file_path}: {e}")
            return None

    def cleanup_old_logs(self) -> List[str]:
        """
        清理过期日志

        Returns:
            删除的文件列表
        """
        deleted = []
        cutoff_date = datetime.now() - timedelta(days=self.max_age_days)

        for log_file in self.get_log_files(include_archived=True):
            if log_file.modified_time < cutoff_date:
                try:
                    os.remove(log_file.path)
                    deleted.append(log_file.name)
                    logger.info(f"已删除过期日志: {log_file.name}")
                except Exception as e:
                    logger.error(f"删除失败 {log_file.name}: {e}")

        return deleted

    def cleanup_by_size(self) -> List[str]:
        """
        按大小清理日志

        Returns:
            删除的文件列表
        """
        deleted = []
        files = self.get_log_files(include_archived=True)

        total_size = sum(f.size_mb for f in files)

        # 从最旧的开始删除
        files_by_age = sorted(files, key=lambda x: x.modified_time)

        for log_file in files_by_age:
            if total_size <= self.max_size_mb:
                break

            try:
                os.remove(log_file.path)
                total_size -= log_file.size_mb
                deleted.append(log_file.name)
                logger.info(f"已删除(超限): {log_file.name}")
            except Exception as e:
                logger.error(f"删除失败 {log_file.name}: {e}")

        return deleted

    def run_maintenance(self) -> Dict[str, List[str]]:
        """
        运行维护任务

        Returns:
            {'archived': [...], 'deleted': [...]}
        """
        result = {
            'archived': self.archive_old_logs(),
            'deleted_old': self.cleanup_old_logs(),
            'deleted_size': self.cleanup_by_size()
        }

        logger.info(f"日志维护完成: 归档 {len(result['archived'])} 个, "
                    f"删除 {len(result['deleted_old']) + len(result['deleted_size'])} 个")

        return result

    def search_logs(
        self,
        pattern: str,
        file_pattern: str = None,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        搜索日志内容

        Args:
            pattern: 搜索模式（正则）
            file_pattern: 文件名模式
            max_results: 最大结果数

        Returns:
            匹配结果列表
        """
        results = []
        regex = re.compile(pattern, re.IGNORECASE)

        for log_file in self.get_log_files():
            if file_pattern and not re.match(file_pattern, log_file.name):
                continue

            try:
                with open(log_file.path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append({
                                'file': log_file.name,
                                'line': line_num,
                                'content': line.strip()
                            })

                            if len(results) >= max_results:
                                return results

            except Exception as e:
                logger.debug(f"读取日志失败 {log_file.name}: {e}")

        return results

    def read_log_tail(self, file_name: str, lines: int = 100) -> List[str]:
        """
        读取日志尾部

        Args:
            file_name: 日志文件名
            lines: 行数

        Returns:
            日志行列表
        """
        file_path = os.path.join(self.log_dir, file_name)

        if not os.path.exists(file_path):
            return []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()
                return [l.strip() for l in all_lines[-lines:]]
        except Exception as e:
            logger.error(f"读取日志失败: {e}")
            return []


def setup_logging(
    log_dir: str = None,
    log_level: int = logging.INFO,
    log_format: str = None
) -> logging.Logger:
    """
    配置日志系统

    Args:
        log_dir: 日志目录
        log_level: 日志级别
        log_format: 日志格式

    Returns:
        根Logger
    """
    if log_dir is None:
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'logs'
        )

    os.makedirs(log_dir, exist_ok=True)

    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # 文件名按日期
    log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")

    # 配置根Logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 文件Handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(file_handler)

    # 控制台Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    root_logger.addHandler(console_handler)

    return root_logger


def get_log_manager(log_dir: str = None) -> LogManager:
    """便捷函数：获取日志管理器"""
    return LogManager(log_dir)
