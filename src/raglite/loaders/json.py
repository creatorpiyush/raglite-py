import json
from typing import Any, List

from ..errors import LoaderError
from .base import BaseLoader


class JsonLoader(BaseLoader):
    def load(self) -> str:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            lines: List[str] = []
            self._flatten(data, "", lines)
            return "\n".join(lines).strip()
        except Exception as cause:
            raise LoaderError(f"Failed to load JSON: {self.file_path}", cause=cause)

    def _flatten(self, value: Any, path: str, lines: List[str]) -> None:
        if value is None:
            lines.append(f"{path}: null")
            return
        if isinstance(value, list):
            if len(value) == 0:
                lines.append(f"{path}: []")
                return
            for index, item in enumerate(value):
                self._flatten(item, f"{path}[{index}]", lines)
            return
        if isinstance(value, dict):
            if len(value) == 0:
                lines.append(f"{path}: {{}}")
                return
            for key, child in value.items():
                next_path = f"{path}.{key}" if path else key
                self._flatten(child, next_path, lines)
            return
        if isinstance(value, bool):
            lines.append(f"{path}: {str(value).lower()}")
            return
        lines.append(f"{path}: {str(value)}")
