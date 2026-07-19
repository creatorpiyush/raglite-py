from ..errors import LoaderError
from .base import BaseLoader


class TxtLoader(BaseLoader):
    def load(self) -> str:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            try:
                with open(self.file_path, "r", encoding="latin1") as f:
                    return f.read().strip()
            except Exception as inner_cause:
                raise LoaderError(f"Failed to load TXT: {self.file_path}", cause=inner_cause)
