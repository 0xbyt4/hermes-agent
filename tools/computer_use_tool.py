"""
Computer Use Tool Module

Enables Claude to control the desktop via screenshots, mouse, and keyboard
using Anthropic's Computer Use API (beta). macOS only.

Screenshots are taken via the native `screencapture` command (no dependencies).
Mouse and keyboard actions use `pyautogui` (optional dependency).

The tool definition uses Anthropic's native format (`computer_20251124`),
not the standard OpenAI function-calling schema. A stub schema is registered
in the normal tool registry for dispatch; the native definition is injected
separately into Anthropic API calls via `get_native_tool_definition()`.

Environment:
    Requires macOS and `pyautogui` (install: `uv pip install -e '.[computer-use]'`).
    Mouse/keyboard actions require macOS Accessibility permission
    (System Settings > Privacy & Security > Accessibility).

Usage:
    hermes -t computer_use   # enable the computer use toolset
"""

import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Approval callback — registered by CLI at startup for prompt_toolkit integration.
# Same pattern as terminal_tool._approval_callback.
_approval_callback = None


def set_approval_callback(cb):
    """Register a callback for computer_use approval prompts (used by CLI)."""
    global _approval_callback
    _approval_callback = cb


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Anthropic recommends max 1568px on longest edge for screenshots
_MAX_SCREENSHOT_EDGE = 1568

# Actions that modify system state — require user approval
_DESTRUCTIVE_ACTIONS = frozenset({
    "left_click", "right_click", "double_click", "triple_click",
    "middle_click", "left_click_drag", "left_mouse_down", "left_mouse_up",
    "type", "key", "scroll", "hold_key",
})

# Actions that are read-only
_SAFE_ACTIONS = frozenset({"screenshot", "mouse_move", "wait", "zoom"})

ALL_ACTIONS = sorted(_DESTRUCTIVE_ACTIONS | _SAFE_ACTIONS)

# Maximum number of screenshot/zoom temp files to keep in /tmp
_MAX_TEMP_FILES = 5


def _cleanup_temp_files() -> None:
    """Remove old hermes screenshot/zoom temp files, keeping the latest ones."""
    import glob
    patterns = ["/tmp/hermes_screenshot_*.jpg", "/tmp/hermes_screenshot_*.png",
                "/tmp/hermes_zoom_*.jpg", "/tmp/hermes_zoom_full_*.jpg"]
    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat))
    if len(all_files) <= _MAX_TEMP_FILES:
        return
    # Sort by modification time, oldest first. Use 0 for files deleted between
    # glob and getmtime (race with concurrent sessions).
    def _safe_mtime(f: str) -> float:
        try:
            return os.path.getmtime(f)
        except OSError:
            return 0
    all_files.sort(key=_safe_mtime)
    for f in all_files[:-_MAX_TEMP_FILES]:
        try:
            os.unlink(f)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Screen resolution helpers
# ---------------------------------------------------------------------------

_cached_screen_size: Optional[Tuple[int, int]] = None
_cached_screenshot_size: Optional[Tuple[int, int]] = None  # Actual image dimensions sent to Claude


def _get_screen_size() -> Tuple[int, int]:
    """Return logical screen resolution (width, height) on macOS."""
    global _cached_screen_size
    if _cached_screen_size:
        return _cached_screen_size
    try:
        import pyautogui
        _cached_screen_size = pyautogui.size()
        return _cached_screen_size
    except Exception:
        # Fallback: assume standard resolution
        return (1920, 1080)


def _compute_scale(actual_w: int, actual_h: int) -> Tuple[int, int, float]:
    """Compute the downsampled image size and scale factor.

    Returns (image_width, image_height, scale_factor) where scale_factor
    is applied to the actual dimensions to produce the image dimensions
    that Claude will see.
    """
    long_edge = max(actual_w, actual_h)
    if long_edge <= _MAX_SCREENSHOT_EDGE:
        return actual_w, actual_h, 1.0
    scale = _MAX_SCREENSHOT_EDGE / long_edge
    return int(actual_w * scale), int(actual_h * scale), scale


def scale_coordinates_to_screen(
    claude_x: int, claude_y: int,
    actual_w: int, actual_h: int,
    image_w: int, image_h: int,
) -> Tuple[int, int]:
    """Scale coordinates from Claude's downsampled image space to actual screen."""
    scale_x = actual_w / image_w if image_w else 1.0
    scale_y = actual_h / image_h if image_h else 1.0
    return int(claude_x * scale_x), int(claude_y * scale_y)


# ---------------------------------------------------------------------------
# Screenshot capture
# ---------------------------------------------------------------------------

def _take_screenshot() -> Tuple[str, int, int, str]:
    """Capture screenshot, resize, convert to JPEG, return (base64_data, image_w, image_h, media_type).

    Uses macOS native `screencapture` for capture and `sips` for resizing/conversion.
    No Python imaging dependencies required.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        tmp_path = f.name

    try:
        # Capture screenshot silently (-x = no sound)
        subprocess.run(
            ["screencapture", "-x", "-C", "-t", "png", tmp_path],
            capture_output=True, timeout=10,
        )
        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
            raise RuntimeError("screencapture produced no output")

        # Get actual image dimensions via sips
        result = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", tmp_path],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.strip().splitlines()
        img_w = img_h = 0
        for line in lines:
            if "pixelWidth" in line:
                img_w = int(line.split(":")[-1].strip())
            elif "pixelHeight" in line:
                img_h = int(line.split(":")[-1].strip())

        # Resize to logical resolution (pyautogui coordinate space).
        # screencapture captures at physical/Retina pixels (e.g. 2940x1912)
        # but pyautogui works in logical points (e.g. 1470x956).
        # Resizing to logical ensures Claude's coordinates map 1:1 to
        # pyautogui without any scaling math.
        logical_w, logical_h = _get_screen_size()
        if img_w != logical_w or img_h != logical_h:
            subprocess.run(
                ["sips", "--resampleWidth", str(logical_w), tmp_path],
                capture_output=True, timeout=10,
            )
            img_w, img_h = logical_w, logical_h

        # Further downscale if logical resolution exceeds Anthropic's max
        long_edge = max(img_w, img_h)
        if long_edge > _MAX_SCREENSHOT_EDGE:
            scale = _MAX_SCREENSHOT_EDGE / long_edge
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)
            subprocess.run(
                ["sips", "--resampleWidth", str(new_w), tmp_path],
                capture_output=True, timeout=10,
            )
            img_w, img_h = new_w, new_h

        # Convert to JPEG for smaller size (5-10x smaller than PNG).
        # Token cost is the same (based on pixels, not bytes) but
        # transfer size and context estimation are much better.
        jpg_path = tmp_path.replace(".png", ".jpg")
        subprocess.run(
            ["sips", "-s", "format", "jpeg", "-s", "formatOptions", "70",
             tmp_path, "--out", jpg_path],
            capture_output=True, timeout=10,
        )
        read_path = jpg_path if os.path.exists(jpg_path) else tmp_path

        # Read and base64 encode
        with open(read_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        media_type = "image/jpeg" if read_path.endswith(".jpg") else "image/png"

        # Cache actual screenshot dimensions for native tool definition
        global _cached_screenshot_size
        _cached_screenshot_size = (img_w, img_h)

        return data, img_w, img_h, media_type

    finally:
        for _p in (tmp_path, tmp_path.replace(".png", ".jpg")):
            try:
                os.unlink(_p)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Action execution
# ---------------------------------------------------------------------------

def _execute_action(action: str, args: Dict[str, Any]) -> str:
    """Execute a computer use action. Returns status message."""
    import pyautogui
    pyautogui.FAILSAFE = True  # Move mouse to corner to abort


    coordinate = args.get("coordinate")
    text = args.get("text", "")

    if action == "screenshot":
        return "screenshot_taken"

    if action == "mouse_move":
        if not coordinate:
            return "error: coordinate required for mouse_move"
        pyautogui.moveTo(coordinate[0], coordinate[1], duration=0.3)
        return f"moved to ({coordinate[0]}, {coordinate[1]})"

    if action == "left_click":
        if coordinate:
            pyautogui.click(coordinate[0], coordinate[1])
            return f"clicked at ({coordinate[0]}, {coordinate[1]})"
        pyautogui.click()
        return "clicked at current position"

    if action == "right_click":
        if coordinate:
            pyautogui.rightClick(coordinate[0], coordinate[1])
            return f"right-clicked at ({coordinate[0]}, {coordinate[1]})"
        pyautogui.rightClick()
        return "right-clicked at current position"

    if action == "double_click":
        if coordinate:
            pyautogui.doubleClick(coordinate[0], coordinate[1])
            return f"double-clicked at ({coordinate[0]}, {coordinate[1]})"
        pyautogui.doubleClick()
        return "double-clicked at current position"

    if action == "triple_click":
        if coordinate:
            pyautogui.tripleClick(coordinate[0], coordinate[1])
            return f"triple-clicked at ({coordinate[0]}, {coordinate[1]})"
        pyautogui.tripleClick()
        return "triple-clicked at current position"

    if action == "middle_click":
        if coordinate:
            pyautogui.middleClick(coordinate[0], coordinate[1])
            return f"middle-clicked at ({coordinate[0]}, {coordinate[1]})"
        pyautogui.middleClick()
        return "middle-clicked at current position"

    if action == "left_click_drag":
        start = args.get("start_coordinate", coordinate)
        end = args.get("end_coordinate")
        if not start or not end:
            return "error: start_coordinate and end_coordinate required for drag"
        pyautogui.moveTo(start[0], start[1], duration=0.2)
        pyautogui.mouseDown()
        pyautogui.moveTo(end[0], end[1], duration=0.5)
        pyautogui.mouseUp()
        return f"dragged from ({start[0]}, {start[1]}) to ({end[0]}, {end[1]})"

    if action == "type":
        if not text:
            return "error: text required for type action"
        # Always use clipboard paste — pyautogui.write() depends on the active
        # keyboard layout (e.g. Turkish layout maps '.' differently) and only
        # supports ASCII. Clipboard paste works with any layout and any charset.
        import subprocess as _sp
        _sp.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        pyautogui.hotkey("command", "v")
        return f"typed {len(text)} characters"

    if action == "key":
        key_combo = args.get("key", text)
        if not key_combo:
            return "error: key required for key action"
        keys = [k.strip() for k in key_combo.replace("+", " ").split()]
        if len(keys) == 1:
            pyautogui.press(keys[0])
        else:
            pyautogui.hotkey(*keys)
        return f"pressed {key_combo}"

    if action == "scroll":
        direction = args.get("scroll_direction", "down")
        amount = args.get("scroll_amount", 3)
        if coordinate:
            pyautogui.moveTo(coordinate[0], coordinate[1])
        clicks = amount if direction in ("up", "left") else -amount
        if direction in ("up", "down"):
            pyautogui.scroll(clicks)
        else:
            pyautogui.hscroll(clicks)
        return f"scrolled {direction} by {amount}"

    if action == "wait":
        duration = args.get("duration", 1)
        time.sleep(min(duration, 10))  # Cap at 10 seconds
        return f"waited {duration}s"

    if action == "left_mouse_down":
        if coordinate:
            pyautogui.moveTo(coordinate[0], coordinate[1])
        pyautogui.mouseDown()
        return "mouse button pressed down"

    if action == "left_mouse_up":
        pyautogui.mouseUp()
        return "mouse button released"

    if action == "hold_key":
        key = args.get("key", text)
        duration = min(args.get("duration", 1), 5)
        if key:
            pyautogui.keyDown(key)
            time.sleep(duration)
            pyautogui.keyUp(key)
            return f"held {key} for {duration}s"
        return "error: key required for hold_key"

    return f"error: unknown action '{action}'"


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def handle_computer_use(args: Dict[str, Any], **kwargs) -> Any:
    """Handle a computer use tool call from Claude.

    Returns either a JSON string (for text-only results) or a dict with
    `_multimodal: True` for results containing screenshots.
    """
    action = args.get("action", "")

    if action not in ALL_ACTIONS:
        return json.dumps({"error": f"Unknown action: {action}. Valid: {ALL_ACTIONS}"})

    # NOTE: Approval for destructive actions is disabled during beta.
    # The computer_use tool is gated behind an explicit toolset flag and
    # gateway allowed_users filtering provides access control.
    # TODO: Re-enable approval with proper gateway async flow support.

    # Coordinate scaling: Screenshots are resized to logical resolution
    # (pyautogui coordinate space). If further downscaled beyond that,
    # we need to scale back up. Otherwise coordinates are 1:1.
    actual_w, actual_h = _get_screen_size()
    if _cached_screenshot_size:
        image_w, image_h = _cached_screenshot_size
    else:
        image_w, image_h = actual_w, actual_h

    needs_scaling = (image_w != actual_w or image_h != actual_h)

    def _scale_coord(x: int, y: int) -> Tuple[int, int]:
        if not needs_scaling:
            return int(x), int(y)
        return scale_coordinates_to_screen(x, y, actual_w, actual_h, image_w, image_h)

    for coord_key in ("coordinate", "start_coordinate", "end_coordinate"):
        coord = args.get(coord_key)
        if not coord:
            continue
        # Claude may send coordinates as a JSON string "[89, 863]" instead of a list.
        if isinstance(coord, str):
            try:
                coord = json.loads(coord)
                args[coord_key] = coord
            except (json.JSONDecodeError, ValueError):
                continue
        if isinstance(coord, (list, tuple)) and len(coord) == 2:
            # Cast to int — pyautogui treats string args as image filenames.
            args[coord_key] = list(_scale_coord(int(coord[0]), int(coord[1])))

    # Execute the action
    if action == "screenshot":
        try:
            _cleanup_temp_files()
            b64_data, img_w, img_h, img_media = _take_screenshot()
            # Get current mouse position for Claude's awareness
            try:
                import pyautogui as _pag
                _mx, _my = _pag.position()
                _cursor_info = f" Cursor at ({_mx}, {_my})."
            except Exception:
                _cursor_info = ""
            # Save to file for gateway MEDIA: tag (sends image to Telegram/Discord)
            # Use session-unique path to avoid race between concurrent gateway sessions
            import uuid as _uuid
            ext = "jpg" if "jpeg" in img_media else "png"
            screenshot_path = f"/tmp/hermes_screenshot_{_uuid.uuid4().hex[:8]}.{ext}"
            with open(screenshot_path, "wb") as f:
                f.write(base64.b64decode(b64_data))
            return {
                "_multimodal": True,
                "content_blocks": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": img_media,
                            "data": b64_data,
                        },
                    },
                ],
                "text_summary": f"Screenshot taken ({img_w}x{img_h}).{_cursor_info} MEDIA:{screenshot_path}",
            }
        except Exception as e:
            logger.error("Screenshot failed: %s", e)
            return json.dumps({"error": f"Screenshot failed: {e}"})

    if action == "zoom":
        # Zoom captures a specific region at full resolution for detailed inspection.
        # Takes a region [x1, y1, x2, y2] and returns a cropped, full-res screenshot.
        region = args.get("region")
        if not region or len(region) != 4:
            return json.dumps({"error": "zoom requires region: [x1, y1, x2, y2]"})
        try:
            _cleanup_temp_files()
            b64_data, img_w, img_h, img_media = _take_screenshot()
            # Crop the region using sips (save full screenshot, crop, re-encode)
            import uuid as _uuid
            tmp_full = f"/tmp/hermes_zoom_full_{_uuid.uuid4().hex[:8]}.jpg"
            tmp_crop = f"/tmp/hermes_zoom_crop_{_uuid.uuid4().hex[:8]}.jpg"
            with open(tmp_full, "wb") as f:
                f.write(base64.b64decode(b64_data))
            x1, y1, x2, y2 = int(region[0]), int(region[1]), int(region[2]), int(region[3])
            crop_w, crop_h = x2 - x1, y2 - y1
            subprocess.run(
                ["sips", "--cropOffset", str(y1), str(x1),
                 "--cropToHeightWidth", str(crop_h), str(crop_w),
                 "-s", "format", "jpeg", "-s", "formatOptions", "90",
                 tmp_full, "--out", tmp_crop],
                capture_output=True, timeout=10,
            )
            with open(tmp_crop, "rb") as f:
                crop_b64 = base64.b64encode(f.read()).decode("ascii")
            # Cleanup
            for p in (tmp_full, tmp_crop):
                try:
                    os.unlink(p)
                except OSError:
                    pass
            screenshot_path = f"/tmp/hermes_zoom_{_uuid.uuid4().hex[:8]}.jpg"
            with open(screenshot_path, "wb") as f:
                f.write(base64.b64decode(crop_b64))
            return {
                "_multimodal": True,
                "content_blocks": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": crop_b64,
                        },
                    },
                ],
                "text_summary": f"Zoomed region ({x1},{y1})-({x2},{y2}) = {crop_w}x{crop_h}px MEDIA:{screenshot_path}",
            }
        except Exception as e:
            logger.error("Zoom failed: %s", e)
            return json.dumps({"error": f"Zoom failed: {e}"})

    # Execute the action — no auto-screenshot. Claude will send a
    # separate 'screenshot' action when it wants to see the result.
    # This avoids doubling token/transfer cost on every interaction.
    #
    # Modifier keys (shift, ctrl, alt, super) are held during click/scroll
    # actions per Anthropic spec — the "text" param on these actions holds
    # the modifier name, not text to type.
    import pyautogui as _pag
    _MODIFIER_MAP = {"shift": "shift", "ctrl": "ctrl", "alt": "alt", "super": "command"}
    _mod_text = args.get("text", "").strip().lower() if action not in ("type", "key") else ""
    _modifier = _MODIFIER_MAP.get(_mod_text)
    try:
        if _modifier:
            _pag.keyDown(_modifier)
        status = _execute_action(action, args)
    except Exception as e:
        logger.error("Action %s failed: %s", action, e)
        return json.dumps({"error": f"Action '{action}' failed: {e}"})
    finally:
        if _modifier:
            _pag.keyUp(_modifier)

    return json.dumps({"success": True, "status": status})


# ---------------------------------------------------------------------------
# Native Anthropic tool definition
# ---------------------------------------------------------------------------

def get_native_tool_definition() -> Dict[str, Any]:
    """Return the native Anthropic computer use tool definition.

    Uses cached screenshot dimensions if available (actual image size sent
    to Claude), otherwise estimates from logical screen size. This ensures
    the declared display size matches the actual screenshot pixels Claude sees.
    """
    if _cached_screenshot_size:
        image_w, image_h = _cached_screenshot_size
    else:
        w, h = _get_screen_size()
        image_w, image_h, _ = _compute_scale(w, h)
    return {
        "type": "computer_20251124",
        "name": "computer",
        "display_width_px": image_w,
        "display_height_px": image_h,
        "enable_zoom": True,
    }


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

def check_computer_use_requirements() -> bool:
    """Return True if computer use is available (macOS + pyautogui)."""
    if sys.platform != "darwin":
        return False
    try:
        import pyautogui  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Tool registration (stub schema for dispatch only)
# ---------------------------------------------------------------------------

_COMPUTER_USE_SCHEMA = {
    "name": "computer",
    "description": (
        "Control the computer desktop — take screenshots, click, type, scroll, "
        "and use keyboard shortcuts. Use 'screenshot' action first to see the "
        "current screen, then interact with elements by their coordinates. "
        "When sharing screenshots with the user, include MEDIA:<path> from the "
        "screenshot result's text_summary in your response to deliver it as an image."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ALL_ACTIONS,
                "description": "The action to perform",
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "[x, y] screen coordinate for click/move actions",
            },
            "text": {
                "type": "string",
                "description": "Text to type, or key combo to press (e.g. 'ctrl+c')",
            },
            "scroll_direction": {
                "type": "string",
                "enum": ["up", "down", "left", "right"],
                "description": "Scroll direction",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Number of scroll clicks",
            },
            "duration": {
                "type": "number",
                "description": "Duration in seconds for wait/hold_key",
            },
            "start_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "[x, y] start coordinate for left_click_drag",
            },
            "end_coordinate": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "[x, y] end coordinate for left_click_drag",
            },
            "region": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "[x1, y1, x2, y2] region to zoom into for detailed view",
            },
        },
        "required": ["action"],
    },
}

try:
    from tools.registry import registry

    registry.register(
        name="computer",
        toolset="computer_use",
        schema=_COMPUTER_USE_SCHEMA,
        handler=lambda args, **kw: handle_computer_use(args, **kw),
        check_fn=check_computer_use_requirements,
        emoji="\U0001f5a5",  # desktop computer emoji
    )
except Exception as e:
    logger.debug("Computer use tool registration skipped: %s", e)
