"""
Observer pattern: alert subscribers get notified when a high-severity
impact assessment is created. Kept provider-agnostic - console/log by
default (works with zero config for a demo), with a clean extension point
for a real Slack/email webhook in production.
"""
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger("regwatch.alerts")
logging.basicConfig(level=logging.INFO)


class AlertChannel(ABC):
    @abstractmethod
    def notify(self, message: str, severity: str, context: dict):
        ...


class ConsoleAlertChannel(AlertChannel):
    """Default channel - logs to stdout. Zero config, works everywhere."""

    def notify(self, message: str, severity: str, context: dict):
        logger.warning(f"[ALERT:{severity.upper()}] {message} | context={context}")


class WebhookAlertChannel(AlertChannel):
    """Stretch: posts to a Slack/generic webhook URL. Opt-in via config -
    not wired up by default since it needs a real webhook URL to be useful."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def notify(self, message: str, severity: str, context: dict):
        import requests
        try:
            requests.post(self.webhook_url, json={"text": f"[{severity.upper()}] {message}"}, timeout=5)
        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")


_subscribers: list[AlertChannel] = [ConsoleAlertChannel()]


def subscribe(channel: AlertChannel):
    _subscribers.append(channel)


def notify_all(message: str, severity: str, context: dict | None = None):
    for channel in _subscribers:
        channel.notify(message, severity, context or {})


def maybe_alert_on_impact(policy_name: str, severity: str, circular_title: str, reasoning: str):
    """Called from the pipeline after each impact assessment is persisted."""
    if severity == "urgent":
        notify_all(
            message=f"Urgent impact detected: '{circular_title}' affects policy '{policy_name}'",
            severity=severity,
            context={"policy": policy_name, "circular": circular_title, "reasoning": reasoning},
        )
