import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.alerts import subscribe, notify_all, maybe_alert_on_impact, AlertChannel, _subscribers


class _CapturingChannel(AlertChannel):
    def __init__(self):
        self.received = []

    def notify(self, message, severity, context):
        self.received.append((message, severity, context))


def test_subscriber_receives_notification():
    channel = _CapturingChannel()
    subscribe(channel)
    notify_all("test message", "urgent", {"key": "value"})
    assert ("test message", "urgent", {"key": "value"}) in channel.received
    _subscribers.remove(channel)


def test_maybe_alert_only_fires_on_urgent():
    channel = _CapturingChannel()
    subscribe(channel)

    maybe_alert_on_impact("Some Policy", "info", "Some Circular", "reasoning")
    maybe_alert_on_impact("Some Policy", "action_required", "Some Circular", "reasoning")
    assert len(channel.received) == 0

    maybe_alert_on_impact("Some Policy", "urgent", "Some Circular", "reasoning text")
    assert len(channel.received) == 1
    assert "Some Policy" in channel.received[0][2]["policy"]

    _subscribers.remove(channel)
