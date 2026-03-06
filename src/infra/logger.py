"""Structured JSON logging configuration."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)
        for key in ("signal_score", "trade_id", "order_id", "market_id",
                     "action", "pnl", "position_size", "confidence",
                     "underlying", "strike", "option_type", "greeks"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)


def setup_logging(log_level: str = "INFO", error_log_path: str | None = None) -> None:
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root.handlers.clear()

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(JSONFormatter())
    root.addHandler(stdout_handler)

    if error_log_path:
        path = Path(error_log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        error_handler = logging.FileHandler(str(path))
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JSONFormatter())
        root.addHandler(error_handler)

    for name in ("websockets", "aiohttp", "urllib3", "alpaca", "yfinance"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
