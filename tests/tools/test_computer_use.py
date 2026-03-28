"""Tests for computer_use_tool module."""

import json
import os
import re
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
            self.mock_pag.hotkey.assert_called_once_with("command", "v", interval=0.04)
            assert "typed" in result

    def test_key_combo(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("key", {"key": "ctrl+c"})
        self.mock_pag.hotkey.assert_called_once_with("ctrl", "c", interval=0.04)
        assert "pressed" in result

    def test_single_key(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("key", {"key": "Return"})
        self.mock_pag.press.assert_called_once_with("return")
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

    @patch("tools.computer_use_tool._take_screenshot", side_effect=RuntimeError("screencapture failed"))
    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    def test_screenshot_error_returns_json(self, _size, _screenshot):
        """Screenshot exception should return JSON error, not crash."""
        from tools.computer_use_tool import handle_computer_use
        result = handle_computer_use({"action": "screenshot"})
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "error" in parsed
        assert "screencapture failed" in parsed["error"]


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

    @patch("tools.computer_use_tool._cleanup_temp_files")
    @patch("tools.computer_use_tool._take_screenshot", return_value=("AAAA", 1024, 768, "image/jpeg"))
    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    def test_screenshot_saves_file(self, _size, _screenshot, _cleanup):
        """Screenshot should save to a unique /tmp/hermes_screenshot_<id>.jpg path."""
        from tools.computer_use_tool import handle_computer_use
        result = handle_computer_use({"action": "screenshot"})
        assert isinstance(result, dict)
        # Extract the file path from text_summary
        match = re.search(r"MEDIA:(/tmp/hermes_screenshot_[a-f0-9]+\.jpg)", result["text_summary"])
        assert match is not None, f"No MEDIA path found in: {result['text_summary']}"
        assert os.path.exists(match.group(1))

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
            with patch("tools.computer_use_tool._quartz_drag") as mock_drag:
                result = handle_computer_use({
                    "action": "left_click_drag",
                    "coordinate": [100, 200],
                    "start_coordinate": [100, 200],
                    "end_coordinate": [400, 500],
                })
                parsed = json.loads(result)
                assert parsed.get("success") is True
                mock_drag.assert_called_once_with(100, 200, 400, 500)


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


class TestHorizontalScroll:
    """Test horizontal scroll via hscroll."""

    @pytest.fixture(autouse=True)
    def _mock_pyautogui(self):
        self.mock_pag = MagicMock()
        self.mock_pag.FAILSAFE = True
        with patch.dict("sys.modules", {"pyautogui": self.mock_pag}):
            yield

    def test_scroll_left_positive(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("scroll", {"scroll_direction": "left", "scroll_amount": 3})
        self.mock_pag.hscroll.assert_called_once_with(3)
        assert "scrolled left" in result

    def test_scroll_right_negative(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("scroll", {"scroll_direction": "right", "scroll_amount": 3})
        self.mock_pag.hscroll.assert_called_once_with(-3)
        assert "scrolled right" in result

    def test_scroll_at_coordinate(self):
        from tools.computer_use_tool import _execute_action
        _execute_action("scroll", {"scroll_direction": "left", "scroll_amount": 2, "coordinate": [500, 300]})
        self.mock_pag.moveTo.assert_called_once_with(500, 300)
        self.mock_pag.hscroll.assert_called_once_with(2)


class TestMiddleClick:
    """Test middle click action."""

    @pytest.fixture(autouse=True)
    def _mock_pyautogui(self):
        self.mock_pag = MagicMock()
        self.mock_pag.FAILSAFE = True
        with patch.dict("sys.modules", {"pyautogui": self.mock_pag}):
            yield

    def test_middle_click_with_coordinate(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("middle_click", {"coordinate": [500, 300]})
        self.mock_pag.middleClick.assert_called_once_with(500, 300)
        assert "middle-clicked" in result

    def test_middle_click_without_coordinate(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("middle_click", {})
        self.mock_pag.middleClick.assert_called_once()
        assert "middle-clicked" in result


class TestMouseDownUp:
    """Test left_mouse_down and left_mouse_up actions (Quartz-based)."""

    @pytest.fixture(autouse=True)
    def _mock_deps(self):
        """Inject mock pyautogui and Quartz into the module before each test."""
        self.mock_pag = MagicMock()
        self.mock_pag.FAILSAFE = True
        self.mock_quartz = MagicMock()
        with patch.dict("sys.modules", {
            "pyautogui": self.mock_pag,
            "Quartz": self.mock_quartz,
        }):
            yield

    def test_mouse_down_with_coordinate(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("left_mouse_down", {"coordinate": [200, 400]})
        # Quartz sends MouseMoved + LeftMouseDown = 2 events
        assert self.mock_quartz.CGEventCreateMouseEvent.call_count == 2
        assert self.mock_quartz.CGEventPost.call_count == 2
        assert "pressed down" in result

    def test_mouse_down_without_coordinate(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("left_mouse_down", {})
        self.mock_pag.position.assert_called_once()
        assert self.mock_quartz.CGEventCreateMouseEvent.call_count == 2
        assert self.mock_quartz.CGEventPost.call_count == 2
        assert "pressed down" in result

    def test_mouse_up(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("left_mouse_up", {})
        self.mock_pag.position.assert_called_once()
        self.mock_quartz.CGEventCreateMouseEvent.assert_called_once()
        self.mock_quartz.CGEventPost.assert_called_once()
        assert "released" in result


class TestHoldKey:
    """Test hold_key action."""

    @pytest.fixture(autouse=True)
    def _mock_pyautogui(self):
        self.mock_pag = MagicMock()
        self.mock_pag.FAILSAFE = True
        with patch.dict("sys.modules", {"pyautogui": self.mock_pag}):
            yield

    def test_hold_key_with_duration(self):
        from tools.computer_use_tool import _execute_action
        import time
        start = time.time()
        result = _execute_action("hold_key", {"key": "shift", "duration": 0.1})
        elapsed = time.time() - start
        self.mock_pag.keyDown.assert_called_once_with("shift")
        self.mock_pag.keyUp.assert_called_once_with("shift")
        assert "held shift" in result
        assert elapsed < 2  # Should be very fast (0.1s + overhead)

    def test_hold_key_duration_capped(self):
        from tools.computer_use_tool import _execute_action
        # Duration should be capped at 5 seconds
        result = _execute_action("hold_key", {"key": "a", "duration": 100})
        assert "held a for 5" in result

    def test_hold_key_no_key_returns_error(self):
        from tools.computer_use_tool import _execute_action
        result = _execute_action("hold_key", {})
        assert "error" in result


class TestModifierKeys:
    """Test modifier key handling during click/scroll actions."""

    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    @patch("tools.computer_use_tool._cached_screenshot_size", (1024, 768))
    def test_shift_held_during_click(self, _size):
        """Modifier key should be held during click action."""
        from tools.computer_use_tool import handle_computer_use
        mock_pag = MagicMock()
        mock_pag.FAILSAFE = True
        with patch.dict("sys.modules", {"pyautogui": mock_pag}):
            result = handle_computer_use({
                "action": "left_click",
                "coordinate": [500, 300],
                "text": "shift",
            })
            parsed = json.loads(result)
            assert parsed.get("success") is True
            mock_pag.keyDown.assert_called_once_with("shift")
            mock_pag.keyUp.assert_called_once_with("shift")

    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    @patch("tools.computer_use_tool._cached_screenshot_size", (1024, 768))
    def test_ctrl_modifier(self, _size):
        """Ctrl modifier maps to ctrl key."""
        from tools.computer_use_tool import handle_computer_use
        mock_pag = MagicMock()
        mock_pag.FAILSAFE = True
        with patch.dict("sys.modules", {"pyautogui": mock_pag}):
            handle_computer_use({
                "action": "left_click",
                "coordinate": [500, 300],
                "text": "ctrl",
            })
            mock_pag.keyDown.assert_called_once_with("ctrl")
            mock_pag.keyUp.assert_called_once_with("ctrl")

    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    @patch("tools.computer_use_tool._cached_screenshot_size", (1024, 768))
    def test_super_maps_to_command(self, _size):
        """Super modifier maps to command on macOS."""
        from tools.computer_use_tool import handle_computer_use
        mock_pag = MagicMock()
        mock_pag.FAILSAFE = True
        with patch.dict("sys.modules", {"pyautogui": mock_pag}):
            handle_computer_use({
                "action": "left_click",
                "coordinate": [500, 300],
                "text": "super",
            })
            mock_pag.keyDown.assert_called_once_with("command")
            mock_pag.keyUp.assert_called_once_with("command")

    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    @patch("tools.computer_use_tool._cached_screenshot_size", (1024, 768))
    def test_modifier_released_on_action_error(self, _size):
        """Modifier should be released even if action raises an exception."""
        from tools.computer_use_tool import handle_computer_use
        mock_pag = MagicMock()
        mock_pag.FAILSAFE = True
        mock_pag.click.side_effect = RuntimeError("pyautogui error")
        with patch.dict("sys.modules", {"pyautogui": mock_pag}):
            result = handle_computer_use({
                "action": "left_click",
                "coordinate": [500, 300],
                "text": "alt",
            })
            parsed = json.loads(result)
            assert "error" in parsed
            # Modifier should still be released in finally block
            mock_pag.keyDown.assert_called_once_with("alt")
            mock_pag.keyUp.assert_called_once_with("alt")

    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    @patch("tools.computer_use_tool._cached_screenshot_size", (1024, 768))
    def test_no_modifier_for_type_action(self, _size):
        """Type action should not use text param as modifier."""
        from tools.computer_use_tool import handle_computer_use
        mock_pag = MagicMock()
        mock_pag.FAILSAFE = True
        with patch.dict("sys.modules", {"pyautogui": mock_pag}):
            with patch("subprocess.run"):
                handle_computer_use({
                    "action": "type",
                    "text": "shift",  # This is text to type, not a modifier
                })
                # keyDown should NOT be called — "shift" is text to type
                mock_pag.keyDown.assert_not_called()


class TestZoomAction:
    """Test zoom action for region-based screenshots."""

    @patch("tools.computer_use_tool._take_screenshot", return_value=("AAAA", 1024, 768, "image/jpeg"))
    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    def test_zoom_missing_region_returns_error(self, _size, _screenshot):
        from tools.computer_use_tool import handle_computer_use
        result = handle_computer_use({"action": "zoom"})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "region" in parsed["error"]

    @patch("tools.computer_use_tool._take_screenshot", return_value=("AAAA", 1024, 768, "image/jpeg"))
    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    def test_zoom_invalid_region_length(self, _size, _screenshot):
        from tools.computer_use_tool import handle_computer_use
        result = handle_computer_use({"action": "zoom", "region": [10, 20]})
        parsed = json.loads(result)
        assert "error" in parsed

    @patch("tools.computer_use_tool.subprocess")
    @patch("tools.computer_use_tool._take_screenshot", return_value=("AAAA", 1024, 768, "image/jpeg"))
    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    def test_zoom_valid_region_returns_multimodal(self, _size, _screenshot, mock_subprocess):
        """Zoom with valid region should return multimodal dict."""
        from tools.computer_use_tool import handle_computer_use
        # Mock subprocess.run for sips crop command
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        # Create the expected crop output file before handle_computer_use reads it
        import base64
        crop_data = base64.b64encode(b"\xff\xd8\xff\xe0test").decode("ascii")

        original_open = open

        def mock_open_side_effect(path, mode="r", **kwargs):
            if "hermes_zoom_crop_" in str(path) and mode == "rb":
                import io
                return io.BytesIO(b"\xff\xd8\xff\xe0test")
            return original_open(path, mode, **kwargs)

        with patch("builtins.open", side_effect=mock_open_side_effect):
            result = handle_computer_use({"action": "zoom", "region": [100, 200, 500, 600]})

        assert isinstance(result, dict)
        assert result["_multimodal"] is True
        assert result["content_blocks"][0]["type"] == "image"
        assert "Zoomed region" in result["text_summary"]

    @patch("tools.computer_use_tool._take_screenshot", side_effect=RuntimeError("capture failed"))
    @patch("tools.computer_use_tool._get_screen_size", return_value=(1024, 768))
    def test_zoom_screenshot_error(self, _size, _screenshot):
        """Zoom should return error JSON if screenshot fails."""
        from tools.computer_use_tool import handle_computer_use
        result = handle_computer_use({"action": "zoom", "region": [0, 0, 100, 100]})
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "error" in parsed


class TestTempFileCleanup:
    """Test temporary file cleanup mechanism."""

    def test_cleanup_removes_old_files(self):
        """Cleanup should remove old files, keeping the latest ones."""
        import time
        from tools.computer_use_tool import _cleanup_temp_files, _MAX_TEMP_FILES

        # Create test files with unique prefix to avoid collision with parallel tests
        prefix = f"hermes_screenshot_cleanup{os.getpid()}"
        files = []
        for i in range(_MAX_TEMP_FILES + 3):
            f = f"/tmp/{prefix}_{i:04d}.jpg"
            with open(f, "w") as fh:
                fh.write("test")
            # Stagger mtime so ordering is deterministic
            os.utime(f, (time.time() - (_MAX_TEMP_FILES + 3 - i), time.time() - (_MAX_TEMP_FILES + 3 - i)))
            files.append(f)

        # Mock glob so only the first pattern returns our files, rest return empty.
        # This prevents the 4 glob patterns from quadrupling the file count.
        call_count = {"n": 0}

        def mock_glob(pattern):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return list(files)
            return []

        try:
            with patch("glob.glob", side_effect=mock_glob):
                _cleanup_temp_files()
            remaining = [f for f in files if os.path.exists(f)]
            assert len(remaining) == _MAX_TEMP_FILES
            # The newest files should survive
            for f in files[-_MAX_TEMP_FILES:]:
                assert os.path.exists(f), f"Expected {f} to survive cleanup"
        finally:
            for f in files:
                try:
                    os.unlink(f)
                except OSError:
                    pass

    def test_cleanup_no_crash_when_no_files(self):
        """Cleanup should not crash if no temp files exist."""
        from tools.computer_use_tool import _cleanup_temp_files
        with patch("glob.glob", return_value=[]):
            _cleanup_temp_files()


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


class TestStubSchema:
    """Test the tool registration stub schema completeness."""

    def test_schema_has_drag_coordinates(self):
        from tools.computer_use_tool import _COMPUTER_USE_SCHEMA
        props = _COMPUTER_USE_SCHEMA["parameters"]["properties"]
        assert "start_coordinate" in props
        assert "end_coordinate" in props

    def test_schema_has_all_params(self):
        from tools.computer_use_tool import _COMPUTER_USE_SCHEMA
        props = _COMPUTER_USE_SCHEMA["parameters"]["properties"]
        expected = ["action", "coordinate", "text", "scroll_direction",
                    "scroll_amount", "duration", "region",
                    "start_coordinate", "end_coordinate"]
        for param in expected:
            assert param in props, f"Missing parameter: {param}"
