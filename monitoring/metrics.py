import time
from collections import Counter
from threading import Lock


class Metrics:
    def __init__(self):
        self.started_at = time.time()
        self._counters = Counter()
        self._lock = Lock()

    def increment(self, name, value=1):
        with self._lock:
            self._counters[name] += value

    def snapshot(self):
        with self._lock:
            counters = dict(self._counters)

        return {
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "counters": counters,
        }


metrics = Metrics()


def log_metric(name, value):
    metrics.increment(name, value)
