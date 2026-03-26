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
        result = _execute_action("type", {"text": "hello world"})
        self.mock_pag.write.assert_called_once_with("hello world", interval=0.02)
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

    @patch("tools.computer_use_tool._take_screenshot", return_value=("base64data", 1024, 768, "image/jpeg"))
    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    def test_screenshot_returns_multimodal(self, _size, _screenshot):
        from tools.computer_use_tool import handle_computer_use
        result = handle_computer_use({"action": "screenshot"})
        assert isinstance(result, dict)
        assert result["_multimodal"] is True
        assert result["content_blocks"][0]["type"] == "image"
        assert result["content_blocks"][0]["source"]["data"] == "base64data"
        assert result["content_blocks"][0]["source"]["media_type"] == "image/jpeg"


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
