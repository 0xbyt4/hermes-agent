"""Tests for computer_use_tool module."""

import json
from unittest.mock import MagicMock, patch

import pytest


class TestCoordinateScaling:
    """Test coordinate scaling from Claude's image space to actual screen."""

    def test_no_scaling_needed(self):
        from tools.computer_use_tool import scale_coordinates_to_screen
        x, y = scale_coordinates_to_screen(100, 200, 1024, 768, 1024, 768)
        assert x == 100
        assert y == 200

    def test_2x_upscale(self):
        from tools.computer_use_tool import scale_coordinates_to_screen
        # Screen is 2048x1536, image is 1024x768
        x, y = scale_coordinates_to_screen(100, 200, 2048, 1536, 1024, 768)
        assert x == 200
        assert y == 400

    def test_retina_scaling(self):
        from tools.computer_use_tool import scale_coordinates_to_screen
        # Typical macOS: 2560x1440 actual, downsampled to 1568x882
        x, y = scale_coordinates_to_screen(784, 441, 2560, 1440, 1568, 882)
        assert abs(x - 1280) < 2
        assert abs(y - 720) < 2

    def test_zero_image_size_no_crash(self):
        from tools.computer_use_tool import scale_coordinates_to_screen
        x, y = scale_coordinates_to_screen(100, 200, 1920, 1080, 0, 0)
        assert x == 100
        assert y == 200


class TestComputeScale:
    """Test image downscaling calculation."""

    def test_small_screen_no_downscale(self):
        from tools.computer_use_tool import _compute_scale
        w, h, scale = _compute_scale(1024, 768)
        assert w == 1024
        assert h == 768
        assert scale == 1.0

    def test_large_screen_downscale(self):
        from tools.computer_use_tool import _compute_scale
        w, h, scale = _compute_scale(2560, 1440)
        assert w <= 1568
        assert h <= 1568
        assert scale < 1.0

    def test_max_edge_respected(self):
        from tools.computer_use_tool import _compute_scale
        w, h, _ = _compute_scale(3840, 2160)
        assert max(w, h) <= 1568


class TestNativeToolDefinition:
    """Test the Anthropic native tool definition generation."""

    @patch("tools.computer_use_tool._get_screen_size", return_value=(1920, 1080))
    def test_returns_correct_format(self, _mock_size):
        from tools.computer_use_tool import get_native_tool_definition
        defn = get_native_tool_definition()
        assert defn["type"] == "computer_20251124"
        assert defn["name"] == "computer"
        assert "display_width_px" in defn
        assert "display_height_px" in defn

    @patch("tools.computer_use_tool._get_screen_size", return_value=(1920, 1080))
    def test_dimensions_within_limits(self, _mock_size):
        from tools.computer_use_tool import get_native_tool_definition
        defn = get_native_tool_definition()
        assert defn["display_width_px"] <= 1568
        assert defn["display_height_px"] <= 1568


class TestActionExecution:
    """Test action execution with mocked pyautogui."""

    @pytest.fixture(autouse=True)
    def _mock_pyautogui(self):
        """Inject a mock pyautogui into the module before each test."""
        self.mock_pag = MagicMock()
        self.mock_pag.FAILSAFE = True
        with patch.dict("sys.modules", {"pyautogui": self.mock_pag}):
            yield

    def test_left_click(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("left_click", {"coordinate": [500, 300]})
        self.mock_pag.click.assert_called_once_with(500, 300)
        assert "clicked" in result

    def test_type_text(self):
        from tools.computer_use_tool import _execute_action
        from unittest.mock import patch as _patch
        with _patch("subprocess.run") as mock_run:
            result = _execute_action("type", {"text": "hello world"})
            # Type uses clipboard paste: pbcopy + Cmd+V
            mock_run.assert_called_once()
            assert mock_run.call_args[0][0] == ["pbcopy"]
            self.mock_pag.hotkey.assert_called_once_with("command", "v")
            assert "typed" in result

    def test_key_combo(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("key", {"key": "ctrl+c"})
        self.mock_pag.hotkey.assert_called_once_with("ctrl", "c")
        assert "pressed" in result

    def test_single_key(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("key", {"key": "Return"})
        self.mock_pag.press.assert_called_once_with("Return")
        assert "pressed" in result

    def test_scroll_down(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("scroll", {"scroll_direction": "down", "scroll_amount": 5})
        self.mock_pag.scroll.assert_called_once_with(-5)
        assert "scrolled" in result

    def test_mouse_move(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("mouse_move", {"coordinate": [100, 200]})
        self.mock_pag.moveTo.assert_called_once_with(100, 200, duration=0.3)
        assert "moved" in result

    def test_unknown_action(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("nonexistent", {})
        assert "unknown" in result.lower()

    def test_wait_capped(self):
        from tools.computer_use_tool import _execute_action
        import time
        start = time.time()
        _execute_action("wait", {"duration": 100})  # Request 100s
        elapsed = time.time() - start
        assert elapsed < 12  # Capped at 10s + margin


class TestHandleComputerUse:
    """Test the main handler function."""

    def test_unknown_action_returns_error(self):
        from tools.computer_use_tool import handle_computer_use
        result = handle_computer_use({"action": "fly"})
        parsed = json.loads(result)
        assert "error" in parsed

    @patch("tools.computer_use_tool._take_screenshot", return_value=("AAAA", 1024, 768, "image/jpeg"))
    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    def test_screenshot_returns_multimodal(self, _size, _screenshot):
        from tools.computer_use_tool import handle_computer_use
        result = handle_computer_use({"action": "screenshot"})
        assert isinstance(result, dict)
        assert result["_multimodal"] is True
        assert result["content_blocks"][0]["type"] == "image"
        assert result["content_blocks"][0]["source"]["data"] == "AAAA"
        assert result["content_blocks"][0]["source"]["media_type"] == "image/jpeg"
        assert "MEDIA:" in result["text_summary"]


class TestCoordinateParsing:
    """Test JSON string coordinate parsing."""

    @patch("tools.computer_use_tool._take_screenshot", return_value=("AAAA", 1024, 768, "image/jpeg"))
    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    @patch("tools.computer_use_tool._cached_screenshot_size", (1024, 768))
    def test_string_coordinate_parsed(self, _size, _screenshot):
        """Claude sometimes sends coordinates as JSON string '[89, 863]'."""
        from tools.computer_use_tool import handle_computer_use
        with patch.dict("sys.modules", {"pyautogui": MagicMock()}):
            result = handle_computer_use({"action": "left_click", "coordinate": "[500, 300]"})
            parsed = json.loads(result)
            assert parsed.get("success") is True

    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    @patch("tools.computer_use_tool._cached_screenshot_size", (1024, 768))
    def test_string_list_coordinate_parsed(self, _size):
        """Coordinates as list of strings ['500', '300']."""
        from tools.computer_use_tool import handle_computer_use
        with patch.dict("sys.modules", {"pyautogui": MagicMock()}):
            result = handle_computer_use({"action": "left_click", "coordinate": ["500", "300"]})
            parsed = json.loads(result)
            assert parsed.get("success") is True


class TestActionResults:
    """Test that actions return correct result format."""

    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    @patch("tools.computer_use_tool._cached_screenshot_size", (1024, 768))
    def test_click_returns_json_not_multimodal(self, _size):
        """Non-screenshot actions return JSON string, not multimodal dict."""
        from tools.computer_use_tool import handle_computer_use
        with patch.dict("sys.modules", {"pyautogui": MagicMock()}):
            result = handle_computer_use({"action": "left_click", "coordinate": [500, 300]})
            assert isinstance(result, str)
            parsed = json.loads(result)
            assert parsed.get("success") is True

    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    @patch("tools.computer_use_tool._cached_screenshot_size", (1024, 768))
    def test_type_empty_text_returns_error(self, _size):
        """Type with empty text should return error."""
        from tools.computer_use_tool import handle_computer_use
        mock_pag = MagicMock()
        mock_pag.FAILSAFE = True
        with patch.dict("sys.modules", {"pyautogui": mock_pag}):
            result = handle_computer_use({"action": "type", "text": ""})
            parsed = json.loads(result)
            assert "error" in parsed.get("status", "")

    @patch("tools.computer_use_tool._take_screenshot", return_value=("AAAA", 1024, 768, "image/jpeg"))
    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    def test_screenshot_saves_file(self, _size, _screenshot):
        """Screenshot should save to /tmp/hermes_screenshot.jpg."""
        import os
        from tools.computer_use_tool import handle_computer_use
        result = handle_computer_use({"action": "screenshot"})
        assert isinstance(result, dict)
        assert os.path.exists("/tmp/hermes_screenshot.jpg")

    @patch("tools.computer_use_tool._take_screenshot", return_value=("AAAA", 1024, 768, "image/jpeg"))
    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    def test_screenshot_media_tag_has_correct_path(self, _size, _screenshot):
        """MEDIA: tag should contain /tmp/hermes_screenshot_ prefix."""
        from tools.computer_use_tool import handle_computer_use
        result = handle_computer_use({"action": "screenshot"})
        assert "MEDIA:/tmp/hermes_screenshot_" in result["text_summary"]
        assert ".jpg" in result["text_summary"]


class TestDragCoordinates:
    """Test drag action coordinate handling."""

    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    @patch("tools.computer_use_tool._cached_screenshot_size", (1024, 768))
    def test_drag_coordinates_scaled(self, _size):
        """start_coordinate and end_coordinate should be parsed and scaled."""
        from tools.computer_use_tool import handle_computer_use
        mock_pag = MagicMock()
        mock_pag.FAILSAFE = True
        with patch.dict("sys.modules", {"pyautogui": mock_pag}):
            result = handle_computer_use({
                "action": "left_click_drag",
                "coordinate": [100, 200],
                "start_coordinate": [100, 200],
                "end_coordinate": [400, 500],
            })
            parsed = json.loads(result)
            assert parsed.get("success") is True
            mock_pag.moveTo.assert_called()
            mock_pag.mouseDown.assert_called_once()
            mock_pag.mouseUp.assert_called_once()


class TestScrollDirection:
    """Test scroll direction handling."""

    @pytest.fixture(autouse=True)
    def _mock_pyautogui(self):
        self.mock_pag = MagicMock()
        self.mock_pag.FAILSAFE = True
        with patch.dict("sys.modules", {"pyautogui": self.mock_pag}):
            yield

    def test_scroll_up_positive(self):
        from tools.computer_use_tool import _execute_action
        _execute_action("scroll", {"scroll_direction": "up", "scroll_amount": 3})
        self.mock_pag.scroll.assert_called_once_with(3)

    def test_scroll_down_negative(self):
        from tools.computer_use_tool import _execute_action
        _execute_action("scroll", {"scroll_direction": "down", "scroll_amount": 3})
        self.mock_pag.scroll.assert_called_once_with(-3)


class TestRequirementsCheck:
    """Test platform requirements detection."""

    @patch("sys.platform", "darwin")
    def test_macos_with_pyautogui(self):
        with patch.dict("sys.modules", {"pyautogui": MagicMock()}):
            from tools.computer_use_tool import check_computer_use_requirements
            # Re-import to pick up patched platform
            import importlib
            import tools.computer_use_tool as mod
            importlib.reload(mod)
            assert mod.check_computer_use_requirements() is True

    @patch("sys.platform", "linux")
    def test_linux_rejected(self):
        from tools.computer_use_tool import check_computer_use_requirements
        import importlib
        import tools.computer_use_tool as mod
        importlib.reload(mod)
        assert mod.check_computer_use_requirements() is False
