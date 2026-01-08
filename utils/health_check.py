# coding=utf-8
"""
系统健康检查模块
检查系统各组件状态
"""

import os
import sys
import logging
import importlib
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import platform

logger = logging.getLogger(__name__)


@dataclass
class CheckResult:
    """检查结果"""
    name: str
    status: str  # "ok", "warning", "error"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """健康报告"""
    timestamp: datetime = field(default_factory=datetime.now)
    overall_status: str = "ok"
    checks: List[CheckResult] = field(default_factory=list)
    system_info: Dict[str, Any] = field(default_factory=dict)

    def add_check(self, result: CheckResult):
        """添加检查结果"""
        self.checks.append(result)
        if result.status == "error":
            self.overall_status = "error"
        elif result.status == "warning" and self.overall_status != "error":
            self.overall_status = "warning"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'overall_status': self.overall_status,
            'system_info': self.system_info,
            'checks': [
                {
                    'name': c.name,
                    'status': c.status,
                    'message': c.message,
                    'details': c.details
                }
                for c in self.checks
            ]
        }


class HealthChecker:
    """
    系统健康检查器

    检查项目:
    1. Python环境
    2. 依赖包安装
    3. 数据目录
    4. 数据库连接
    5. TqSdk配置
    6. 磁盘空间
    """

    def __init__(self, project_root: str = None):
        """
        初始化

        Args:
            project_root: 项目根目录
        """
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.project_root = project_root

    def run_all_checks(self) -> HealthReport:
        """运行所有检查"""
        report = HealthReport()

        # 系统信息
        report.system_info = self._get_system_info()

        # 运行各项检查
        report.add_check(self._check_python())
        report.add_check(self._check_dependencies())
        report.add_check(self._check_data_dir())
        report.add_check(self._check_database())
        report.add_check(self._check_tqsdk_config())
        report.add_check(self._check_disk_space())
        report.add_check(self._check_log_dir())

        return report

    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': platform.python_version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'project_root': self.project_root
        }

    def _check_python(self) -> CheckResult:
        """检查Python版本"""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version.major < 3 or (version.major == 3 and version.minor < 8):
            return CheckResult(
                name="Python版本",
                status="error",
                message=f"Python版本过低: {version_str}，需要 >= 3.8",
                details={'current': version_str, 'required': '>=3.8'}
            )

        return CheckResult(
            name="Python版本",
            status="ok",
            message=f"Python {version_str}",
            details={'version': version_str}
        )

    def _check_dependencies(self) -> CheckResult:
        """检查依赖包"""
        required_packages = {
            'streamlit': '1.28.0',
            'pandas': '2.0.0',
            'numpy': '1.24.0',
            'plotly': '5.0.0',
            'tqsdk': '3.0.0',
        }

        optional_packages = {
            'akshare': '1.0.0',
            'keyring': '24.0.0',
        }

        missing = []
        outdated = []
        installed = {}

        for pkg, min_ver in required_packages.items():
            try:
                mod = importlib.import_module(pkg)
                version = getattr(mod, '__version__', 'unknown')
                installed[pkg] = version
            except ImportError:
                missing.append(pkg)

        for pkg, min_ver in optional_packages.items():
            try:
                mod = importlib.import_module(pkg)
                version = getattr(mod, '__version__', 'unknown')
                installed[pkg] = version
            except ImportError:
                pass  # 可选包不报错

        if missing:
            return CheckResult(
                name="依赖包",
                status="error",
                message=f"缺少必需包: {', '.join(missing)}",
                details={'missing': missing, 'installed': installed}
            )

        return CheckResult(
            name="依赖包",
            status="ok",
            message=f"已安装 {len(installed)} 个包",
            details={'installed': installed}
        )

    def _check_data_dir(self) -> CheckResult:
        """检查数据目录"""
        data_dir = os.path.join(self.project_root, 'data')

        if not os.path.exists(data_dir):
            try:
                os.makedirs(data_dir)
                return CheckResult(
                    name="数据目录",
                    status="ok",
                    message=f"已创建数据目录: {data_dir}",
                    details={'path': data_dir, 'created': True}
                )
            except Exception as e:
                return CheckResult(
                    name="数据目录",
                    status="error",
                    message=f"无法创建数据目录: {e}",
                    details={'path': data_dir, 'error': str(e)}
                )

        # 检查写权限
        test_file = os.path.join(data_dir, '.write_test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            return CheckResult(
                name="数据目录",
                status="error",
                message=f"数据目录无写权限: {e}",
                details={'path': data_dir, 'writable': False}
            )

        # 统计文件
        file_count = len([f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))])

        return CheckResult(
            name="数据目录",
            status="ok",
            message=f"数据目录正常，{file_count} 个文件",
            details={'path': data_dir, 'file_count': file_count, 'writable': True}
        )

    def _check_database(self) -> CheckResult:
        """检查数据库"""
        db_path = os.path.join(self.project_root, 'data', 'futures_data.db')

        if not os.path.exists(db_path):
            return CheckResult(
                name="数据库",
                status="warning",
                message="数据库文件不存在，首次运行时将自动创建",
                details={'path': db_path, 'exists': False}
            )

        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # 检查表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

            # 统计数据量
            stats = {}
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[table] = cursor.fetchone()[0]

            conn.close()

            db_size = os.path.getsize(db_path) / 1024 / 1024  # MB

            return CheckResult(
                name="数据库",
                status="ok",
                message=f"数据库正常，{len(tables)} 个表，{db_size:.1f} MB",
                details={'path': db_path, 'tables': tables, 'stats': stats, 'size_mb': db_size}
            )

        except Exception as e:
            return CheckResult(
                name="数据库",
                status="error",
                message=f"数据库连接失败: {e}",
                details={'path': db_path, 'error': str(e)}
            )

    def _check_tqsdk_config(self) -> CheckResult:
        """检查TqSdk配置"""
        config_path = os.path.join(self.project_root, 'tq_config.json')

        if not os.path.exists(config_path):
            return CheckResult(
                name="TqSdk配置",
                status="warning",
                message="未找到配置文件，请在系统设置中配置",
                details={'path': config_path, 'exists': False}
            )

        try:
            import json
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            has_user = bool(config.get('tq_user'))
            has_pass = bool(config.get('tq_password'))
            sim_mode = config.get('sim_mode', True)

            if not has_user or not has_pass:
                return CheckResult(
                    name="TqSdk配置",
                    status="warning",
                    message="天勤账号未配置",
                    details={'has_user': has_user, 'has_password': has_pass}
                )

            mode = "模拟盘" if sim_mode else "实盘"
            return CheckResult(
                name="TqSdk配置",
                status="ok",
                message=f"配置正常，当前模式: {mode}",
                details={'mode': mode, 'has_user': True}
            )

        except Exception as e:
            return CheckResult(
                name="TqSdk配置",
                status="error",
                message=f"配置文件读取失败: {e}",
                details={'error': str(e)}
            )

    def _check_disk_space(self) -> CheckResult:
        """检查磁盘空间"""
        try:
            if platform.system() == 'Windows':
                import ctypes
                free_bytes = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    ctypes.c_wchar_p(self.project_root),
                    None, None, ctypes.pointer(free_bytes)
                )
                free_gb = free_bytes.value / (1024 ** 3)
            else:
                import shutil
                total, used, free = shutil.disk_usage(self.project_root)
                free_gb = free / (1024 ** 3)

            if free_gb < 1:
                return CheckResult(
                    name="磁盘空间",
                    status="error",
                    message=f"磁盘空间不足: {free_gb:.1f} GB",
                    details={'free_gb': free_gb}
                )
            elif free_gb < 5:
                return CheckResult(
                    name="磁盘空间",
                    status="warning",
                    message=f"磁盘空间较低: {free_gb:.1f} GB",
                    details={'free_gb': free_gb}
                )

            return CheckResult(
                name="磁盘空间",
                status="ok",
                message=f"可用空间: {free_gb:.1f} GB",
                details={'free_gb': free_gb}
            )

        except Exception as e:
            return CheckResult(
                name="磁盘空间",
                status="warning",
                message=f"无法检查磁盘空间: {e}",
                details={'error': str(e)}
            )

    def _check_log_dir(self) -> CheckResult:
        """检查日志目录"""
        log_dir = os.path.join(self.project_root, 'logs')

        if not os.path.exists(log_dir):
            return CheckResult(
                name="日志目录",
                status="ok",
                message="日志目录不存在，将在需要时创建",
                details={'path': log_dir, 'exists': False}
            )

        # 统计日志文件
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        total_size = sum(
            os.path.getsize(os.path.join(log_dir, f))
            for f in log_files
        ) / 1024 / 1024  # MB

        if total_size > 100:
            return CheckResult(
                name="日志目录",
                status="warning",
                message=f"日志文件较大: {total_size:.1f} MB，建议清理",
                details={'path': log_dir, 'files': len(log_files), 'size_mb': total_size}
            )

        return CheckResult(
            name="日志目录",
            status="ok",
            message=f"{len(log_files)} 个日志文件，{total_size:.1f} MB",
            details={'path': log_dir, 'files': len(log_files), 'size_mb': total_size}
        )


def run_health_check() -> HealthReport:
    """便捷函数：运行健康检查"""
    checker = HealthChecker()
    return checker.run_all_checks()


def print_health_report(report: HealthReport):
    """打印健康报告"""
    status_icons = {
        'ok': '✓',
        'warning': '⚠',
        'error': '✗'
    }

    print("\n" + "=" * 50)
    print("系统健康检查报告")
    print("=" * 50)
    print(f"时间: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总体状态: {status_icons.get(report.overall_status, '?')} {report.overall_status.upper()}")
    print("-" * 50)

    for check in report.checks:
        icon = status_icons.get(check.status, '?')
        print(f"{icon} {check.name}: {check.message}")

    print("=" * 50 + "\n")
