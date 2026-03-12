"""Tests for web_crawl tool registration and schema."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json


class TestWebCrawlRegistration:
    """Verify web_crawl is properly registered in the tool system."""

    def test_tool_registered_in_registry(self):
        from tools.registry import registry
        from tools import web_tools  # noqa: F401
        assert "web_crawl" in registry._tools

    def test_tool_metadata(self):
        from tools.registry import registry
        from tools import web_tools  # noqa: F401
        tool = registry._tools["web_crawl"]
        assert tool.toolset == "web"
        assert tool.is_async is True
        assert "FIRECRAWL_API_KEY" in tool.requires_env

    def test_tool_schema_has_required_fields(self):
        from tools.web_tools import WEB_CRAWL_SCHEMA
        assert WEB_CRAWL_SCHEMA["name"] == "web_crawl"
        assert "url" in WEB_CRAWL_SCHEMA["parameters"]["properties"]
        assert "url" in WEB_CRAWL_SCHEMA["parameters"]["required"]

    def test_tool_schema_optional_fields(self):
        from tools.web_tools import WEB_CRAWL_SCHEMA
        props = WEB_CRAWL_SCHEMA["parameters"]["properties"]
        assert "instructions" in props
        assert "depth" in props
        assert props["depth"]["enum"] == ["basic", "advanced"]


class TestWebCrawlInToolset:
    """Verify web_crawl is included in the web toolset."""

    def test_web_toolset_includes_crawl(self):
        from toolsets import TOOLSETS
        assert "web_crawl" in TOOLSETS["web"]["tools"]

    def test_legacy_map_includes_crawl(self):
        from model_tools import _LEGACY_TOOLSET_MAP
        assert "web_crawl" in _LEGACY_TOOLSET_MAP["web_tools"]


def _make_mock_page(markdown, source_url, title, status_code=200):
    """Helper: create a mock Firecrawl Document with model_dump support."""
    page = MagicMock()
    page.model_dump.return_value = {
        "markdown": markdown,
        "metadata": {
            "sourceURL": source_url,
            "title": title,
            "statusCode": status_code,
        },
    }
    return page


def _make_crawl_result(pages, status="completed"):
    """Helper: create a mock CrawlJob object."""
    crawl = MagicMock()
    crawl.data = pages
    crawl.status = status
    return crawl


class TestWebCrawlHandler:
    """Test the crawl handler dispatches correctly."""

    @pytest.mark.asyncio
    async def test_crawl_adds_https_prefix(self):
        """URL without protocol should get https:// prepended."""
        with patch("tools.web_tools._get_firecrawl_client") as mock_client:
            mock_client.return_value.crawl.return_value = _make_crawl_result([])

            from tools.web_tools import web_crawl_tool
            await web_crawl_tool("example.com", use_llm_processing=False)

            call_kwargs = mock_client.return_value.crawl.call_args[1]
            assert call_kwargs["url"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_crawl_preserves_http_prefix(self):
        """URL with http:// should NOT be changed to https://."""
        with patch("tools.web_tools._get_firecrawl_client") as mock_client:
            mock_client.return_value.crawl.return_value = _make_crawl_result([])

            from tools.web_tools import web_crawl_tool
            await web_crawl_tool("http://example.com", use_llm_processing=False)

            call_kwargs = mock_client.return_value.crawl.call_args[1]
            assert call_kwargs["url"] == "http://example.com"

    @pytest.mark.asyncio
    async def test_crawl_params_passed_to_firecrawl(self):
        """Firecrawl client should receive limit=20 and markdown format."""
        with patch("tools.web_tools._get_firecrawl_client") as mock_client:
            mock_client.return_value.crawl.return_value = _make_crawl_result([])

            from tools.web_tools import web_crawl_tool
            await web_crawl_tool("https://example.com", use_llm_processing=False)

            call_kwargs = mock_client.return_value.crawl.call_args[1]
            assert call_kwargs["limit"] == 20
            assert call_kwargs["scrape_options"]["formats"] == ["markdown"]

    @pytest.mark.asyncio
    async def test_crawl_returns_trimmed_fields(self):
        """Response should only contain title, content, error per page (url stripped)."""
        page = _make_mock_page("# Hello", "https://example.com/page1", "Hello Page")
        with patch("tools.web_tools._get_firecrawl_client") as mock_client:
            mock_client.return_value.crawl.return_value = _make_crawl_result([page])

            from tools.web_tools import web_crawl_tool
            result = await web_crawl_tool("https://example.com", use_llm_processing=False)
            data = json.loads(result)

            entry = data["results"][0]
            assert set(entry.keys()) == {"title", "content", "error"}
            assert entry["title"] == "Hello Page"
            assert entry["content"] == "# Hello"

    @pytest.mark.asyncio
    async def test_crawl_multiple_pages(self):
        """Multiple crawled pages should all appear in results."""
        pages = [
            _make_mock_page("Page 1 content", "https://a.com/1", "Page 1"),
            _make_mock_page("Page 2 content", "https://a.com/2", "Page 2"),
            _make_mock_page("Page 3 content", "https://a.com/3", "Page 3"),
        ]
        with patch("tools.web_tools._get_firecrawl_client") as mock_client:
            mock_client.return_value.crawl.return_value = _make_crawl_result(pages)

            from tools.web_tools import web_crawl_tool
            result = await web_crawl_tool("https://a.com", use_llm_processing=False)
            data = json.loads(result)

            assert len(data["results"]) == 3
            titles = [r["title"] for r in data["results"]]
            assert titles == ["Page 1", "Page 2", "Page 3"]

    @pytest.mark.asyncio
    async def test_crawl_empty_results(self):
        """Crawl with no pages should return empty results list."""
        with patch("tools.web_tools._get_firecrawl_client") as mock_client:
            mock_client.return_value.crawl.return_value = _make_crawl_result([])

            from tools.web_tools import web_crawl_tool
            result = await web_crawl_tool("https://example.com", use_llm_processing=False)
            data = json.loads(result)

            assert data["results"] == []

    @pytest.mark.asyncio
    async def test_crawl_dict_item_type(self):
        """Items returned as plain dicts (not Pydantic models) should work."""
        dict_item = {
            "markdown": "# Dict Page",
            "metadata": {
                "sourceURL": "https://example.com/dict",
                "title": "Dict Title",
            },
        }
        mock_crawl = MagicMock()
        # data returns dicts, not objects with model_dump
        mock_crawl.data = [dict_item]
        mock_crawl.status = "completed"

        with patch("tools.web_tools._get_firecrawl_client") as mock_client:
            mock_client.return_value.crawl.return_value = mock_crawl

            from tools.web_tools import web_crawl_tool
            result = await web_crawl_tool("https://example.com", use_llm_processing=False)
            data = json.loads(result)

            assert len(data["results"]) == 1
            assert data["results"][0]["title"] == "Dict Title"
            assert data["results"][0]["content"] == "# Dict Page"

    @pytest.mark.asyncio
    async def test_crawl_strips_base64_images(self):
        """Base64 encoded images in content should be replaced with placeholders."""
        markdown_with_b64 = (
            "# Page\n"
            "![img](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEA"
            "AAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJ"
            "RU5ErkJggg==)\n"
            "Some text after"
        )
        page = _make_mock_page(markdown_with_b64, "https://example.com", "B64 Test")
        with patch("tools.web_tools._get_firecrawl_client") as mock_client:
            mock_client.return_value.crawl.return_value = _make_crawl_result([page])

            from tools.web_tools import web_crawl_tool
            result = await web_crawl_tool("https://example.com", use_llm_processing=False)

            # The raw base64 string should NOT be in the output
            assert "iVBORw0KGgo" not in result
            # But the surrounding text should remain
            assert "Some text after" in result

    @pytest.mark.asyncio
    async def test_crawl_error_returns_error_json(self):
        """Exception from Firecrawl should return error JSON, not raise."""
        with patch("tools.web_tools._get_firecrawl_client", side_effect=Exception("Connection refused")):
            from tools.web_tools import web_crawl_tool
            result = await web_crawl_tool("https://example.com")
            data = json.loads(result)

            assert "error" in data
            assert "Connection refused" in data["error"]

    @pytest.mark.asyncio
    async def test_crawl_interrupted_returns_early(self):
        """If interrupted before crawl starts, return immediately."""
        with patch("tools.interrupt.is_interrupted", return_value=True):
            from tools.web_tools import web_crawl_tool
            result = await web_crawl_tool("https://example.com")
            data = json.loads(result)

            assert data["success"] is False
            assert "Interrupted" in data["error"]

    @pytest.mark.asyncio
    async def test_crawl_with_llm_processing(self):
        """When use_llm_processing=True (default), process_content_with_llm is called per page."""
        page = _make_mock_page(
            "x" * 6000,  # content longer than default min_length
            "https://example.com/long",
            "Long Page",
        )
        with patch("tools.web_tools._get_firecrawl_client") as mock_client, \
             patch("tools.web_tools.process_content_with_llm", new_callable=AsyncMock) as mock_llm:
            mock_client.return_value.crawl.return_value = _make_crawl_result([page])
            mock_llm.return_value = "## Summarized content"

            from tools.web_tools import web_crawl_tool
            result = await web_crawl_tool("https://example.com", use_llm_processing=True)
            data = json.loads(result)

            mock_llm.assert_called_once()
            assert data["results"][0]["content"] == "## Summarized content"
