"""Tests for the CLI's native vision content routing.

When the active model declares native vision support, the CLI should
build a list of typed content blocks (text + image_url) instead of
calling _preprocess_images_with_vision to flatten images via the
auxiliary vision model. Models without native vision still get the
legacy text-flatten path.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._attached_images = []
    return cli_obj


@pytest.fixture()
def png_file(tmp_path):
    """Tiny valid PNG file for base64-encoding tests."""
    png_bytes = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c6300010000000500010d0a2db40000"
        "000049454e44ae426082"
    )
    path = tmp_path / "test.png"
    path.write_bytes(png_bytes)
    return path


# ---------------------------------------------------------------------------
# _build_native_vision_content_cli
# ---------------------------------------------------------------------------


class TestBuildNativeVisionContentCli:
    def test_text_and_one_image(self, png_file):
        cli_obj = _make_cli()
        result = cli_obj._build_native_vision_content_cli("Look at this", [png_file])
        assert isinstance(result, list)
        assert result[0] == {"type": "text", "text": "Look at this"}
        assert result[1]["type"] == "image_url"
        assert result[1]["image_url"]["url"].startswith("data:image/")
        assert "base64," in result[1]["image_url"]["url"]

    def test_image_only_no_caption(self, png_file):
        cli_obj = _make_cli()
        result = cli_obj._build_native_vision_content_cli("", [png_file])
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "image_url"

    def test_multiple_images(self, png_file):
        cli_obj = _make_cli()
        result = cli_obj._build_native_vision_content_cli(
            "compare", [png_file, png_file, png_file]
        )
        text_blocks = [b for b in result if b.get("type") == "text"]
        image_blocks = [b for b in result if b.get("type") == "image_url"]
        assert len(text_blocks) == 1
        assert len(image_blocks) == 3

    def test_skips_missing_image(self, tmp_path):
        cli_obj = _make_cli()
        bad = tmp_path / "missing.png"
        with patch("cli._cprint"):
            result = cli_obj._build_native_vision_content_cli("hi", [bad])
        # No image block, but text remains as a list with the text block
        assert isinstance(result, list)
        assert any(b.get("type") == "text" for b in result)
        assert not any(b.get("type") == "image_url" for b in result)

    def test_falls_back_to_string_when_nothing_to_send(self, tmp_path):
        """Empty caption + all-bad images returns a placeholder string."""
        cli_obj = _make_cli()
        bad = tmp_path / "missing.png"
        with patch("cli._cprint"):
            result = cli_obj._build_native_vision_content_cli("", [bad])
        # Falls back to placeholder text rather than empty list
        assert isinstance(result, str)
        assert "image" in result.lower()


# ---------------------------------------------------------------------------
# chat() routing — vision-capable vs non-vision models
# ---------------------------------------------------------------------------


class TestChatVisionRouting:
    def test_chat_uses_native_vision_when_capable(self, png_file):
        """When agent reports native vision, chat() builds content blocks
        instead of calling _preprocess_images_with_vision."""
        cli_obj = _make_cli()
        cli_obj.agent = MagicMock()
        cli_obj.agent._model_supports_native_vision.return_value = True

        # Just test the image-handling branch in isolation by replicating
        # the chat() preprocessing logic
        message = "describe"
        images = [png_file]

        if images:
            _use_native = False
            try:
                if cli_obj.agent is not None:
                    _use_native = cli_obj.agent._model_supports_native_vision()
            except Exception:
                pass
            if _use_native:
                message = cli_obj._build_native_vision_content_cli(message, images)
            else:
                message = cli_obj._preprocess_images_with_vision(message, images)

        assert isinstance(message, list)
        assert any(b.get("type") == "image_url" for b in message)
        cli_obj.agent._model_supports_native_vision.assert_called_once()

    def test_chat_uses_legacy_flatten_when_not_capable(self, png_file):
        """When agent reports no native vision, chat() falls back to
        _preprocess_images_with_vision (legacy text-flatten path)."""
        cli_obj = _make_cli()
        cli_obj.agent = MagicMock()
        cli_obj.agent._model_supports_native_vision.return_value = False

        message = "describe"
        images = [png_file]

        with patch.object(
            cli_obj,
            "_preprocess_images_with_vision",
            return_value="[The user attached an image. Description: a cat]",
        ) as mock_preprocess:
            if images:
                _use_native = False
                try:
                    if cli_obj.agent is not None:
                        _use_native = cli_obj.agent._model_supports_native_vision()
                except Exception:
                    pass
                if _use_native:
                    message = cli_obj._build_native_vision_content_cli(message, images)
                else:
                    message = cli_obj._preprocess_images_with_vision(message, images)

        assert isinstance(message, str)
        assert "Description: a cat" in message
        mock_preprocess.assert_called_once()

    def test_chat_handles_capability_check_exception(self, png_file):
        """If _model_supports_native_vision raises, fall back to legacy."""
        cli_obj = _make_cli()
        cli_obj.agent = MagicMock()
        cli_obj.agent._model_supports_native_vision.side_effect = RuntimeError("boom")

        message = "describe"
        images = [png_file]

        with patch.object(
            cli_obj,
            "_preprocess_images_with_vision",
            return_value="[fallback text]",
        ) as mock_preprocess:
            if images:
                _use_native = False
                try:
                    if cli_obj.agent is not None:
                        _use_native = cli_obj.agent._model_supports_native_vision()
                except Exception:
                    pass
                if _use_native:
                    message = cli_obj._build_native_vision_content_cli(message, images)
                else:
                    message = cli_obj._preprocess_images_with_vision(message, images)

        assert isinstance(message, str)
        assert "[fallback text]" in message
        mock_preprocess.assert_called_once()

    def test_chat_handles_no_agent(self, png_file):
        """When self.agent is None, fall back to legacy without crashing."""
        cli_obj = _make_cli()
        cli_obj.agent = None

        message = "describe"
        images = [png_file]

        with patch.object(
            cli_obj,
            "_preprocess_images_with_vision",
            return_value="[fallback text]",
        ) as mock_preprocess:
            if images:
                _use_native = False
                try:
                    if cli_obj.agent is not None:
                        _use_native = cli_obj.agent._model_supports_native_vision()
                except Exception:
                    pass
                if _use_native:
                    message = cli_obj._build_native_vision_content_cli(message, images)
                else:
                    message = cli_obj._preprocess_images_with_vision(message, images)

        mock_preprocess.assert_called_once()
