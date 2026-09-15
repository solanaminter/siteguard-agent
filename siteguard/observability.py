"""Structured JSON logging (CloudWatch-compatible).

Every line is a single JSON object: {"ts","level","service","event",...}.
CloudWatch Logs Insights can query these directly, e.g.:
  fields @timestamp, event, incident_id | filter service="siteguard-agent"
"""
import json
import sys
import time


class JsonLogger:
    def __init__(self, service: str = "siteguard-agent", stream=None):
        self.service = service
        self.stream = stream or sys.stdout

    def _emit(self, level: str, event: str, **fields):
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "level": level,
            "service": self.service,
            "event": event,
            **fields,
        }
        self.stream.write(json.dumps(record) + "\n")
        self.stream.flush()

    def info(self, event: str, **fields):
        self._emit("INFO", event, **fields)

    def warn(self, event: str, **fields):
        self._emit("WARN", event, **fields)

    def error(self, event: str, **fields):
        self._emit("ERROR", event, **fields)


logger = JsonLogger()
