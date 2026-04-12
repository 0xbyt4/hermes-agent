"""Tests for the gateway's native vision content routing.

When the active model declares native vision support, the gateway
should build a list of typed content blocks (text + image_url) instead
of calling vision_analyze_tool to flatten images to a text description.
Models without native vision still get the legacy text-flatten path.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.run import GatewayRunner


@pytest.fixture()
def runner():
    """Bare GatewayRunner without full init — fine for unit-testing helpers."""
    r = object.__new__(GatewayRunner)
    return r


@pytest.fixture()
def png_file(tmp_path):
    """Tiny valid PNG file for base64-encoding tests."""
    # 1x1 transparent PNG
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6300010000000500010d0a2db40000"
        "000049454e44ae426082"
    )
    path = tmp_path / "test.png"
    path.write_bytes(png_bytes)
    return path


# ---------------------------------------------------------------------------
# _message_preview_for_hook
# ---------------------------------------------------------------------------


class TestMessagePreviewForHook:
    def test_string_message(self):
        assert GatewayRunner._message_preview_for_hook("hello") == "hello"

    def test_none_message(self):
        assert GatewayRunner._message_preview_for_hook(None) == ""

    def test_text_only_list(self):
        content = [{"type": "text", "text": "Look at this"}]
        assert GatewayRunner._message_preview_for_hook(content) == "Look at this"

    def test_multimodal_list(self):
        content = [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "https://x.com/y.png"}},
        ]
        result = GatewayRunner._message_preview_for_hook(content)
        assert "What is this?" in result
        assert "[image]" in result

    def test_audio_block(self):
        content = [
            {"type": "text", "text": "Listen:"},
            {"type": "input_audio", "audio_url": {"url": "data:audio/wav;base64,X"}},
        ]
        result = GatewayRunner._message_preview_for_hook(content)
        assert "Listen:" in result
        assert "[audio]" in result


# ---------------------------------------------------------------------------
# _should_use_native_vision_for_source
# ---------------------------------------------------------------------------


class TestShouldUseNativeVision:
    def test_force_env_var_returns_true(self, runner, monkeypatch):
        """HERMES_FORCE_NATIVE_VISION=1 short-circuits the lookup."""
        monkeypatch.setenv("HERMES_FORCE_NATIVE_VISION", "1")
        # No need to mock _resolve_session_agent_runtime — should not be called
        assert runner._should_use_native_vision_for_source(MagicMock()) is True

    def test_returns_true_when_capability_says_vision(self, runner, monkeypatch):
        monkeypatch.delenv("HERMES_FORCE_NATIVE_VISION", raising=False)
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("claude-opus-4-6", {"provider": "anthropic"})
        )
        with patch("agent.models_dev.get_model_capabilities") as mock_caps:
            mock_caps.return_value = MagicMock(supports_vision=True)
            assert runner._should_use_native_vision_for_source(MagicMock()) is True
            mock_caps.assert_called_once_with("anthropic", "claude-opus-4-6")

    def test_returns_false_when_capability_says_no_vision(self, runner, monkeypatch):
        monkeypatch.delenv("HERMES_FORCE_NATIVE_VISION", raising=False)
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("plain-text-model", {"provider": "custom"})
        )
        with patch("agent.models_dev.get_model_capabilities") as mock_caps:
            mock_caps.return_value = MagicMock(supports_vision=False)
            assert runner._should_use_native_vision_for_source(MagicMock()) is False

    def test_returns_false_when_capability_lookup_returns_none(self, runner, monkeypatch):
        """Unknown model defaults to safe legacy fallback."""
        monkeypatch.delenv("HERMES_FORCE_NATIVE_VISION", raising=False)
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("unknown-model", {"provider": "custom"})
        )
        with patch("agent.models_dev.get_model_capabilities", return_value=None):
            assert runner._should_use_native_vision_for_source(MagicMock()) is False

    def test_returns_false_when_no_model(self, runner, monkeypatch):
        """Empty model name returns False without lookup."""
        monkeypatch.delenv("HERMES_FORCE_NATIVE_VISION", raising=False)
        runner._resolve_session_agent_runtime = MagicMock(
            return_value=("", {"provider": "anthropic"})
        )
        assert runner._should_use_native_vision_for_source(MagicMock()) is False

    def test_returns_false_when_runtime_resolution_fails(self, runner, monkeypatch):
        """Exception in resolver is caught and we fall back to legacy."""
        monkeypatch.delenv("HERMES_FORCE_NATIVE_VISION", raising=False)
        runner._resolve_session_agent_runtime = MagicMock(side_effect=RuntimeError("boom"))
        assert runner._should_use_native_vision_for_source(MagicMock()) is False


# ---------------------------------------------------------------------------
# _build_native_vision_content
# ---------------------------------------------------------------------------


class TestBuildNativeVisionContent:
    def test_builds_text_and_image_blocks(self, runner, png_file):
        result = runner._build_native_vision_content("What's this?", [str(png_file)])
        assert isinstance(result, list)
        # First block: user text
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "What's this?"
        # Second block: image as data URL
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"].startswith("data:image/")
        assert "base64," in result[1]["image_url"]["url"]

    def test_skips_text_block_when_caption_empty(self, runner, png_file):
        result = runner._build_native_vision_content("", [str(png_file)])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"

    def test_handles_multiple_images(self, runner, png_file):
        result = runner._build_native_vision_content("two pics", [str(png_file), str(png_file)])
        assert len(result) == 3  # 1 text + 2 images
        image_blocks = [b for b in result if b.get("type") == "image_url"]
        assert len(image_blocks) == 2

    def test_skips_unreadable_images(self, runner, tmp_path):
        """Bad image path is skipped, doesn't crash the build."""
        bad_path = tmp_path / "nonexistent.png"
        result = runner._build_native_vision_content("hi", [str(bad_path)])
        # Text block remains, no image block
        assert isinstance(result, list)
        text_blocks = [b for b in result if b.get("type") == "text"]
        assert len(text_blocks) == 1
        image_blocks = [b for b in result if b.get("type") == "image_url"]
        assert len(image_blocks) == 0
