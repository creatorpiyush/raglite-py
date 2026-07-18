import sys
from typing import Literal, Any

LogLevel = Literal["silent", "info", "debug"]


class Logger:
    def __init__(self, level: LogLevel = "info"):
        self.level = level

    def _should_log(self, target: LogLevel) -> bool:
        if self.level == "silent":
            return False
        if self.level == "info":
            return target != "debug"
        return True

    def info(self, message: str, *args: Any) -> None:
        if self._should_log("info"):
            suffix = " " + " ".join(map(str, args)) if args else ""
            print(f"[raglite] {message}{suffix}", file=sys.stderr)

    def debug(self, message: str, *args: Any) -> None:
        if self._should_log("debug"):
            suffix = " " + " ".join(map(str, args)) if args else ""
            print(f"[raglite:debug] {message}{suffix}", file=sys.stderr)

    def warn(self, message: str, *args: Any) -> None:
        if self.level != "silent":
            suffix = " " + " ".join(map(str, args)) if args else ""
            print(f"[raglite] {message}{suffix}", file=sys.stderr)

    def error(self, message: str, *args: Any) -> None:
        if self.level != "silent":
            suffix = " " + " ".join(map(str, args)) if args else ""
            print(f"[raglite] {message}{suffix}", file=sys.stderr)


def create_logger(level: LogLevel = "info") -> Logger:
    return Logger(level)


# Alias for TS parity
createLogger = create_logger
