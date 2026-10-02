import html
import re

from ..constants import PACKAGE_VERSION
from ..errors import LoaderError
from .base import BaseLoader


class WebLoader(BaseLoader):
    """Loader for fetching and extracting plain text / markdown content from HTTP/HTTPS web URLs."""

    def __init__(self, url: str):
        super().__init__(url)
        self.url = url

    def load(self) -> str:
        """Fetch URL content over HTTP/HTTPS and return plain text or markdown."""
        try:
            import httpx

            headers = {
                "User-Agent": f"RAGLite/{PACKAGE_VERSION} (Python/3.10+)",
                "Accept": "text/html,text/plain,application/xhtml+xml;q=0.9,*/*;q=0.8",
            }
            response = httpx.get(self.url, headers=headers, follow_redirects=True, timeout=15.0)
            if response.status_code >= 400:
                raise LoaderError(
                    f'HTTP error {response.status_code} {response.reason_phrase} fetching "{self.url}"'
                )

            content_type = response.headers.get("content-type", "")
            raw_text = response.text

            if "application/json" in content_type:
                try:
                    import json

                    data = json.loads(raw_text)
                    return json.dumps(data, indent=2)
                except Exception:
                    return raw_text

            if "text/plain" in content_type or "text/markdown" in content_type:
                return raw_text

            return self._extract_text_from_html(raw_text)
        except LoaderError:
            raise
        except Exception as err:
            raise LoaderError(f'Failed to load web URL "{self.url}": {err}', cause=err) from err

    def _extract_text_from_html(self, html_content: str) -> str:
        clean = re.sub(r"(?i)<script\b[^<]*>([\s\S]*?)<\/script>", "", html_content)
        clean = re.sub(r"(?i)<style\b[^<]*>([\s\S]*?)<\/style>", "", clean)
        clean = re.sub(r"(?i)<svg\b[^<]*>([\s\S]*?)<\/svg>", "", clean)
        clean = re.sub(r"(?i)<head\b[^<]*>([\s\S]*?)<\/head>", "", clean)

        clean = re.sub(r"(?i)<(h[1-6])\b[^>]*>(.*?)<\/\1>", r"\n\n# \2\n\n", clean)
        clean = re.sub(r"(?i)<p\b[^>]*>(.*?)<\/p>", r"\n\n\1\n\n", clean)
        clean = re.sub(r"(?i)<br\s*\/?>", "\n", clean)
        clean = re.sub(r"(?i)<li\b[^>]*>(.*?)<\/li>", r"\n* \1", clean)
        clean = re.sub(r"<[^>]+>", "", clean)

        clean = html.unescape(clean)

        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in clean.splitlines()]
        return "\n".join(line for line in lines if line)
