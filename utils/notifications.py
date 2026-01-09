# coding=utf-8
"""
通知服务模块
支持桌面通知、声音提示、自动刷新
"""

import logging
import threading
import platform
from datetime import datetime
from typing import Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationType(Enum):
    """通知类型"""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    TRADE = "trade"      # 成交
    SIGNAL = "signal"    # 信号
    RISK = "risk"        # 风控


@dataclass
class Notification:
    """通知消息"""
    title: str
    message: str
    type: NotificationType = NotificationType.INFO
    timestamp: datetime = field(default_factory=datetime.now)
    sound: bool = True
    desktop: bool = True


class NotificationService:
    """
    通知服务

    功能:
    1. 桌面通知（Windows/Mac/Linux）
    2. 声音提示
    3. 通知历史记录
    """

    # 声音文件（Windows系统声音）
    SOUNDS = {
        NotificationType.INFO: "SystemAsterisk",
        NotificationType.SUCCESS: "SystemExclamation",
        NotificationType.WARNING: "SystemHand",
        NotificationType.ERROR: "SystemHand",
        NotificationType.TRADE: "SystemExclamation",
        NotificationType.SIGNAL: "SystemAsterisk",
        NotificationType.RISK: "SystemHand",
    }

    def __init__(
        self,
        enable_sound: bool = True,
        enable_desktop: bool = True,
        wecom_webhook: str = None,
        dingtalk_webhook: str = None
    ):
        """
        初始化通知服务

        Args:
            enable_sound: 是否启用声音
            enable_desktop: 是否启用桌面通知
            wecom_webhook: 企业微信机器人Webhook URL
            dingtalk_webhook: 钉钉机器人Webhook URL
        """
        self.enable_sound = enable_sound
        self.enable_desktop = enable_desktop
        self.wecom_webhook = wecom_webhook
        self.dingtalk_webhook = dingtalk_webhook
        self._history: List[Notification] = []
        self._max_history = 100
        self._lock = threading.Lock()

        # 检测平台
        self._platform = platform.system()
        self._sound_available = self._check_sound()
        self._desktop_available = self._check_desktop()

        # Webhook可用性
        self._wecom_available = bool(wecom_webhook)
        self._dingtalk_available = bool(dingtalk_webhook)

        logger.info(f"通知服务初始化: 平台={self._platform}, "
                   f"声音={self._sound_available}, 桌面={self._desktop_available}, "
                   f"企业微信={self._wecom_available}, 钉钉={self._dingtalk_available}")

    def _check_sound(self) -> bool:
        """检查声音支持"""
        if self._platform == "Windows":
            try:
                import winsound
                return True
            except ImportError:
                return False
        else:
            try:
                # Linux/Mac 可以用 playsound
                import playsound
                return True
            except ImportError:
                return False

    def _check_desktop(self) -> bool:
        """检查桌面通知支持"""
        try:
            from plyer import notification
            return True
        except ImportError:
            # 尝试 Windows toast
            if self._platform == "Windows":
                try:
                    from win10toast import ToastNotifier
                    return True
                except ImportError:
                    pass
            return False

    def notify(self, notification: Notification):
        """
        发送通知

        Args:
            notification: 通知对象
        """
        # 记录历史
        with self._lock:
            self._history.append(notification)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        # 异步发送，不阻塞主线程
        threading.Thread(
            target=self._send_notification,
            args=(notification,),
            daemon=True
        ).start()

    def _send_notification(self, notification: Notification):
        """实际发送通知"""
        # 桌面通知
        if notification.desktop and self.enable_desktop and self._desktop_available:
            self._send_desktop(notification)

        # 声音提示
        if notification.sound and self.enable_sound and self._sound_available:
            self._play_sound(notification.type)

        # Webhook通知（风险/成交等重要消息）
        important_types = [NotificationType.RISK, NotificationType.TRADE, NotificationType.ERROR]
        if notification.type in important_types:
            if self._wecom_available:
                self._send_wecom(notification)
            if self._dingtalk_available:
                self._send_dingtalk(notification)

    def _send_desktop(self, notification: Notification):
        """发送桌面通知"""
        try:
            if self._platform == "Windows":
                self._send_windows_toast(notification)
            else:
                self._send_plyer_notification(notification)
        except Exception as e:
            logger.debug(f"桌面通知失败: {e}")

    def _send_windows_toast(self, notification: Notification):
        """Windows Toast通知"""
        try:
            from win10toast import ToastNotifier
            toaster = ToastNotifier()
            toaster.show_toast(
                notification.title,
                notification.message,
                duration=5,
                threaded=True
            )
        except ImportError:
            # 回退到 plyer
            self._send_plyer_notification(notification)

    def _send_plyer_notification(self, notification: Notification):
        """Plyer跨平台通知"""
        try:
            from plyer import notification as plyer_notify
            plyer_notify.notify(
                title=notification.title,
                message=notification.message,
                timeout=5
            )
        except Exception as e:
            logger.debug(f"Plyer通知失败: {e}")

    def _play_sound(self, ntype: NotificationType):
        """播放提示音"""
        try:
            if self._platform == "Windows":
                import winsound
                sound_name = self.SOUNDS.get(ntype, "SystemAsterisk")
                winsound.PlaySound(sound_name, winsound.SND_ALIAS | winsound.SND_ASYNC)
            else:
                # Linux/Mac 需要音频文件
                pass
        except Exception as e:
            logger.debug(f"播放声音失败: {e}")

    def _send_wecom(self, notification: Notification):
        """发送企业微信机器人消息"""
        if not self.wecom_webhook:
            return

        try:
            import requests

            # 构建消息
            content = f"**{notification.title}**\n\n{notification.message}\n\n时间: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

            data = {
                "msgtype": "markdown",
                "markdown": {
                    "content": content
                }
            }

            response = requests.post(
                self.wecom_webhook,
                json=data,
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') != 0:
                    logger.warning(f"企业微信发送失败: {result.get('errmsg')}")
            else:
                logger.warning(f"企业微信请求失败: {response.status_code}")

        except ImportError:
            logger.debug("requests库未安装，无法发送企业微信通知")
        except Exception as e:
            logger.debug(f"企业微信通知失败: {e}")

    def _send_dingtalk(self, notification: Notification):
        """发送钉钉机器人消息"""
        if not self.dingtalk_webhook:
            return

        try:
            import requests

            # 构建消息
            content = f"**{notification.title}**\n\n{notification.message}\n\n时间: {notification.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

            data = {
                "msgtype": "markdown",
                "markdown": {
                    "title": notification.title,
                    "text": content
                }
            }

            response = requests.post(
                self.dingtalk_webhook,
                json=data,
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('errcode') != 0:
                    logger.warning(f"钉钉发送失败: {result.get('errmsg')}")
            else:
                logger.warning(f"钉钉请求失败: {response.status_code}")

        except ImportError:
            logger.debug("requests库未安装，无法发送钉钉通知")
        except Exception as e:
            logger.debug(f"钉钉通知失败: {e}")

    def set_wecom_webhook(self, webhook_url: str):
        """设置企业微信Webhook"""
        self.wecom_webhook = webhook_url
        self._wecom_available = bool(webhook_url)
        logger.info(f"企业微信Webhook已{'启用' if webhook_url else '禁用'}")

    def set_dingtalk_webhook(self, webhook_url: str):
        """设置钉钉Webhook"""
        self.dingtalk_webhook = webhook_url
        self._dingtalk_available = bool(webhook_url)
        logger.info(f"钉钉Webhook已{'启用' if webhook_url else '禁用'}")

    def get_history(self, limit: int = 20) -> List[Notification]:
        """获取通知历史"""
        with self._lock:
            return list(self._history[-limit:])

    # ============ 便捷方法 ============

    def trade_filled(self, symbol: str, direction: str, price: float, volume: int):
        """成交通知"""
        self.notify(Notification(
            title="成交通知",
            message=f"{symbol} {direction} {volume}手 @ {price:.2f}",
            type=NotificationType.TRADE
        ))

    def signal_generated(self, strategy: str, symbol: str, action: str):
        """信号通知"""
        self.notify(Notification(
            title="策略信号",
            message=f"[{strategy}] {symbol} {action}",
            type=NotificationType.SIGNAL
        ))

    def risk_alert(self, message: str):
        """风控告警"""
        self.notify(Notification(
            title="风控告警",
            message=message,
            type=NotificationType.RISK
        ))

    def info(self, title: str, message: str):
        """普通通知"""
        self.notify(Notification(
            title=title,
            message=message,
            type=NotificationType.INFO
        ))


# 全局单例
_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """获取通知服务单例"""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


# ============ Streamlit 自动刷新组件 ============

def streamlit_auto_refresh(interval_seconds: int = 5, key: str = "auto_refresh"):
    """
    Streamlit 自动刷新组件

    Args:
        interval_seconds: 刷新间隔（秒）
        key: 组件key

    Usage:
        import streamlit as st
        from utils.notifications import streamlit_auto_refresh

        # 页面顶部调用
        streamlit_auto_refresh(interval_seconds=5)
    """
    import streamlit as st

    # 方法1: 使用 streamlit-autorefresh（如果安装了）
    try:
        from streamlit_autorefresh import st_autorefresh
        st_autorefresh(interval=interval_seconds * 1000, key=key)
        return
    except ImportError:
        pass

    # 方法2: 使用 JavaScript 注入
    st.markdown(
        f"""
        <script>
            setTimeout(function() {{
                window.location.reload();
            }}, {interval_seconds * 1000});
        </script>
        """,
        unsafe_allow_html=True
    )


def create_refresh_toggle(default_interval: int = 5):
    """
    创建刷新控制开关

    Returns:
        (enabled, interval)
    """
    import streamlit as st

    col1, col2 = st.columns([1, 3])
    with col1:
        enabled = st.checkbox("自动刷新", value=False, key="auto_refresh_enabled")
    with col2:
        if enabled:
            interval = st.select_slider(
                "刷新间隔",
                options=[3, 5, 10, 30, 60],
                value=default_interval,
                format_func=lambda x: f"{x}秒",
                key="auto_refresh_interval"
            )
        else:
            interval = default_interval

    if enabled:
        streamlit_auto_refresh(interval)

    return enabled, interval
