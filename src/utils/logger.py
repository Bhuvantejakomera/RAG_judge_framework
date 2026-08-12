from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Any

import yaml


def _ensure_log_directories(config: dict[str, Any]) -> None:
    """Ensure that all handler log directories exist before applying configuration."""
    handler_configs = config.get("handlers", {})
    for handler in handler_configs.values():
        filename = handler.get("filename")
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)


def setup_logging(config_path: str | Path | None = None) -> None:
    """Load and configure logging from a YAML configuration file."""
    path = Path(config_path or Path.cwd() / "config" / "logging.yaml")
    if not path.exists():
        raise FileNotFoundError(f"Logging configuration file not found at {path}")

    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Logging configuration file must contain a YAML mapping.")

    _ensure_log_directories(config)
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger instance by name."""
    return logging.getLogger(name)
