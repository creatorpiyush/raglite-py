import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import LoaderError
from ..utils.logger import create_logger
from .base import BaseLoader


@dataclass
class LoadedDirectoryItem:
    file_path: str
    text: str


@dataclass
class DirectoryLoadErrorItem:
    file_path: str
    error: str


@dataclass
class DirectoryLoadResult:
    loaded: list[LoadedDirectoryItem] = field(default_factory=list)
    errors: list[DirectoryLoadErrorItem] = field(default_factory=list)


class DirectoryLoader(BaseLoader):
    """Recursively scans directories and loads all supported documents."""

    def __init__(
        self,
        dir_path: str | Path,
        glob: str | None = None,
        recursive: bool = True,
        ignore_errors: bool = True,
    ):
        path_str = str(dir_path)
        super().__init__(path_str)
        self.dir_path = Path(path_str).resolve()
        self.glob_pattern = glob
        self.recursive = recursive
        self.ignore_errors = ignore_errors
        self.logger = create_logger("info")

    def load(self) -> str:
        """Load and concatenate all supported files from the directory."""
        result = self.load_files()
        sections = [
            f"--- SOURCE: {item.file_path} ---\n{item.text}" for item in result.loaded
        ]
        return "\n\n".join(sections)

    def load_files(self) -> DirectoryLoadResult:
        """Scan directory and return individual loaded document contents and errors."""
        from . import get_loader, is_supported_file

        if not self.dir_path.exists():
            raise LoaderError(f'Directory does not exist: "{self.dir_path}"')

        if not self.dir_path.is_dir():
            raise LoaderError(f'Path is not a directory: "{self.dir_path}"')

        result = DirectoryLoadResult()

        for root, dirs, files in os.walk(self.dir_path):
            # Like the TypeScript SDK: skip hidden entries (.git, .venv, the
            # .raglite store) and node_modules.
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "node_modules"]
            if not self.recursive and Path(root) != self.dir_path:
                continue

            for file in sorted(f for f in files if not f.startswith(".")):
                full_path = Path(root) / file
                rel_path = str(full_path.relative_to(self.dir_path))
                abs_path_str = str(full_path)

                if self.glob_pattern:
                    if not (
                        fnmatch.fnmatch(file, self.glob_pattern)
                        or fnmatch.fnmatch(rel_path, self.glob_pattern)
                    ):
                        continue

                if not is_supported_file(abs_path_str):
                    self.logger.warning(f"Skipping unsupported file type: {abs_path_str}")
                    result.errors.append(
                        DirectoryLoadErrorItem(
                            file_path=abs_path_str,
                            error=f"Unsupported file extension: {full_path.suffix}",
                        )
                    )
                    continue

                try:
                    loader = get_loader(abs_path_str)
                    text = loader.load()
                    result.loaded.append(
                        LoadedDirectoryItem(file_path=abs_path_str, text=text)
                    )
                except Exception as err:
                    errMsg = str(err)
                    self.logger.error(f'Failed loading document "{abs_path_str}": {errMsg}')
                    result.errors.append(
                        DirectoryLoadErrorItem(file_path=abs_path_str, error=errMsg)
                    )
                    if not self.ignore_errors:
                        raise LoaderError(
                            f'Error loading file "{abs_path_str}": {errMsg}', cause=err
                        ) from err

        return result
