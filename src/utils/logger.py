from __future__ import annotations

import logging
import traceback
from typing import Optional

import constant
from utils.error_log_handler_utils import ErrorLogHandler


class Logger:
    """
    Per-name logger with a singleton registry.

    Each unique name gets its own logging.Logger instance so per-module
    log lines can be filtered independently.  Every line carries:
      - timestamp
      - filename    (auto-captured by Python logging from the call site)
      - funcName    (auto-captured by Python logging from the call site)
      - level
      - message + any extra context kwargs
    """

    _instances: dict[str, "Logger"] = {}

    def __new__(cls, name: str = constant.LOGGER_NAME) -> "Logger":
        if name not in cls._instances:
            instance = super().__new__(cls)
            instance._setup(name)
            cls._instances[name] = instance
        return cls._instances[name]

    def __init__(self, name: str = constant.LOGGER_NAME) -> None:
        # Guard against re-initialising the error handler on repeated __init__ calls
        if not hasattr(self, "_error_log"):
            self._error_log = ErrorLogHandler()

    def _setup(self, name: str) -> None:
        """Configure handlers for this named logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s - %(funcName)s - %(levelname)s - %(message)s"
        )

        # Avoid duplicate handlers if _setup is called more than once
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            file_handler = logging.FileHandler("logs.txt", mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _format_msg(self, message: str, kwargs: dict) -> str:
        return f"{message} | context: {kwargs}" if kwargs else message

    # ── Sync log methods ───────────────────────────────────────────────────────
    def info(self, message: str, **kwargs) -> None:
        self.logger.info(self._format_msg(message, kwargs), stacklevel=2)

    def debug(self, message: str, **kwargs) -> None:
        self.logger.debug(self._format_msg(message, kwargs), stacklevel=2)

    def warning(self, message: str, **kwargs) -> None:
        self.logger.warning(self._format_msg(message, kwargs), stacklevel=2)

    # ── Async error method (persists to DB via ErrorLogHandler) ───────────────
    async def error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs,
    ) -> None:
        """Log an error with full stack trace and persist to DB."""
        filename  = "unknown"
        func_name = "unknown"

        if exception is not None:
            tb = exception.__traceback__
            if tb is not None:
                stack = traceback.extract_tb(tb)
                if stack:
                    filename, _line_no, func_name, _ = stack[-1]
            kwargs["exception_msg"] = str(exception)
            kwargs["filename"]  = filename
            kwargs["func_name"] = func_name

        self.logger.error(
            self._format_msg(message, kwargs), exc_info=True, stacklevel=2
        )

        # Persist to DB — failures here are caught so logging never crashes the caller
        try:
            await self._error_log.log_error_details(
                filename, func_name, str(exception) if exception else message
            )
        except Exception as db_err:
            self.logger.error(
                f"ErrorLogHandler failed to persist: {db_err}",
                exc_info=True,
                stacklevel=2,
            )
