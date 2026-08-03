from unittest.mock import MagicMock, patch

from raglite.loaders.web import WebLoader


def test_web_loader_extracts_html_text():
    html_content = """
    <!DOCTYPE html>
    <html>
      <head><title>Test Page</title></head>
      <body>
        <h1>Main Heading</h1>
        <p>This is a test paragraph with <a href="#">a link</a>.</p>
        <script>console.log("ignore script");</script>
        <style>body { color: red; }</style>
      </body>
    </html>
    """

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/html"}
    mock_response.text = html_content

    with patch("httpx.get", return_value=mock_response):
        loader = WebLoader("https://example.com/test")
        text = loader.load()

        assert "# Main Heading" in text
        assert "This is a test paragraph with a link." in text
        assert "ignore script" not in text
        assert "color: red" not in text


def test_web_loader_handles_plain_text():
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "text/plain"}
    mock_response.text = "Plain text content from web server"

    with patch("httpx.get", return_value=mock_response):
        loader = WebLoader("https://example.com/raw.txt")
        text = loader.load()

        assert text == "Plain text content from web server"
