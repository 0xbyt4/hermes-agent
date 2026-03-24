"""Comprehensive tests for the image_process_tool module.

Covers all 24 actions, error handling, edge cases, and the dispatch layer.
Test images are created in-memory with PIL.Image.new() to avoid external deps.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from PIL import Image, ImageDraw

from tools.image_process_tool import (
    _ALLOWED_OUTPUT_FORMATS,
    _MAX_INPUT_SIZE,
    _OUTPUT_DIR,
    _action_blur,
    _action_brightness,
    _action_colors,
    _action_compress,
    _action_contrast,
    _action_convert,
    _action_crop,
    _action_draw,
    _action_filter,
    _action_flip,
    _action_grayscale,
    _action_histogram,
    _action_info,
    _action_invert,
    _action_merge,
    _action_paste,
    _action_resize,
    _action_rotate,
    _action_saturation,
    _action_sharpen,
    _action_strip,
    _action_text,
    _action_thumbnail,
    _action_transparent,
    image_process_tool,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def rgb_image_path(tmp_path):
    """Create a 200x100 RGB JPEG image with some color variation."""
    img = Image.new("RGB", (200, 100), color=(128, 64, 32))
    # Draw a colored rectangle so pixel data isn't completely uniform
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 25, 150, 75], fill=(255, 0, 0))
    path = tmp_path / "test_rgb.jpg"
    img.save(str(path), format="JPEG", quality=95)
    img.close()
    return str(path)


@pytest.fixture()
def rgba_image_path(tmp_path):
    """Create a 100x100 RGBA PNG image."""
    img = Image.new("RGBA", (100, 100), color=(0, 128, 255, 200))
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 20, 80, 80], fill=(255, 255, 0, 255))
    path = tmp_path / "test_rgba.png"
    img.save(str(path), format="PNG")
    img.close()
    return str(path)


@pytest.fixture()
def white_image_path(tmp_path):
    """Create a 50x50 solid white image (useful for transparent action test)."""
    img = Image.new("RGB", (50, 50), color=(255, 255, 255))
    # Put a small colored square in the center
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 30, 30], fill=(10, 20, 30))
    path = tmp_path / "test_white.png"
    img.save(str(path), format="PNG")
    img.close()
    return str(path)


@pytest.fixture()
def small_overlay_path(tmp_path):
    """Create a 30x30 overlay image for paste tests."""
    img = Image.new("RGBA", (30, 30), color=(255, 0, 0, 128))
    path = tmp_path / "overlay.png"
    img.save(str(path), format="PNG")
    img.close()
    return str(path)


@pytest.fixture()
def second_image_path(tmp_path):
    """Create a second 200x100 image for merge tests."""
    img = Image.new("RGB", (200, 100), color=(0, 200, 100))
    path = tmp_path / "test_second.jpg"
    img.save(str(path), format="JPEG")
    img.close()
    return str(path)


@pytest.fixture()
def grayscale_image_path(tmp_path):
    """Create a grayscale image."""
    img = Image.new("L", (80, 80), color=128)
    path = tmp_path / "test_gray.png"
    img.save(str(path), format="PNG")
    img.close()
    return str(path)


def _parse(result: str) -> dict:
    """Helper: parse JSON string returned by action functions."""
    return json.loads(result)


# ---------------------------------------------------------------------------
# 1. info
# ---------------------------------------------------------------------------

class TestActionInfo:
    def test_basic_fields(self, rgb_image_path):
        result = _parse(_action_info(rgb_image_path))
        assert result["format"] == "JPEG"
        assert result["width"] == 200
        assert result["height"] == 100
        assert result["mode"] == "RGB"
        assert "file_size_kb" in result
        assert result["has_alpha"] is False
        assert result["is_animated"] is False
        assert result["n_frames"] == 1

    def test_rgba_info(self, rgba_image_path):
        result = _parse(_action_info(rgba_image_path))
        assert result["format"] == "PNG"
        assert result["has_alpha"] is True
        assert result["mode"] == "RGBA"

    def test_mean_rgb_present(self, rgb_image_path):
        result = _parse(_action_info(rgb_image_path))
        assert "mean_rgb" in result
        assert len(result["mean_rgb"]) == 3

    def test_exif_section_optional(self, rgb_image_path):
        """Synthetic images have no EXIF; the key should be absent, not error."""
        result = _parse(_action_info(rgb_image_path))
        # No exif expected on a PIL.Image.new()-created file
        # Just ensure no crash; key may or may not exist
        assert "error" not in result


# ---------------------------------------------------------------------------
# 2. resize
# ---------------------------------------------------------------------------

class TestActionResize:
    def test_resize_width_only_keep_aspect(self, rgb_image_path):
        result = _parse(_action_resize(rgb_image_path, width=100, keep_aspect=True))
        assert result["width"] == 100
        assert result["height"] == 50  # 200x100 -> 100x50

    def test_resize_height_only_keep_aspect(self, rgb_image_path):
        result = _parse(_action_resize(rgb_image_path, height=50, keep_aspect=True))
        assert result["height"] == 50
        assert result["width"] == 100

    def test_resize_both_keep_aspect(self, rgb_image_path):
        # thumbnail mode: fits within the box while keeping aspect
        result = _parse(_action_resize(rgb_image_path, width=80, height=80, keep_aspect=True))
        assert result["width"] <= 80
        assert result["height"] <= 80

    def test_resize_no_keep_aspect(self, rgb_image_path):
        result = _parse(_action_resize(rgb_image_path, width=50, height=50, keep_aspect=False))
        assert result["width"] == 50
        assert result["height"] == 50

    def test_resize_no_dimensions_error(self, rgb_image_path):
        result = _parse(_action_resize(rgb_image_path, width=0, height=0))
        assert "error" in result

    def test_output_file_created(self, rgb_image_path):
        result = _parse(_action_resize(rgb_image_path, width=100))
        assert Path(result["output"]).is_file()


# ---------------------------------------------------------------------------
# 3. convert
# ---------------------------------------------------------------------------

class TestActionConvert:
    def test_jpeg_to_png(self, rgb_image_path):
        result = _parse(_action_convert(rgb_image_path, target_format="PNG"))
        assert result["format"] == "PNG"
        assert result["output"].endswith(".png")

    def test_png_to_webp(self, rgba_image_path):
        result = _parse(_action_convert(rgba_image_path, target_format="WEBP"))
        assert result["format"] == "WEBP"
        assert result["output"].endswith(".webp")

    def test_invalid_format(self, rgb_image_path):
        result = _parse(_action_convert(rgb_image_path, target_format="INVALID"))
        assert "error" in result
        assert "Unsupported format" in result["error"]

    def test_jpg_alias(self, rgb_image_path):
        result = _parse(_action_convert(rgb_image_path, target_format="JPG"))
        assert result["format"] == "JPEG"


# ---------------------------------------------------------------------------
# 4. compress
# ---------------------------------------------------------------------------

class TestActionCompress:
    def test_quality_30_vs_90(self, rgb_image_path):
        r30 = _parse(_action_compress(rgb_image_path, quality=30))
        r90 = _parse(_action_compress(rgb_image_path, quality=90))
        # Low quality should produce a smaller file
        assert r30["file_size_kb"] <= r90["file_size_kb"]
        assert r30["quality"] == 30
        assert r90["quality"] == 90

    def test_reduction_percent_present(self, rgb_image_path):
        result = _parse(_action_compress(rgb_image_path, quality=50))
        assert "reduction_percent" in result
        assert "original_kb" in result

    def test_quality_clamped(self, rgb_image_path):
        result = _parse(_action_compress(rgb_image_path, quality=200))
        assert result["quality"] == 100  # max(1, min(100, 200)) = 100


# ---------------------------------------------------------------------------
# 5. crop
# ---------------------------------------------------------------------------

class TestActionCrop:
    def test_valid_crop(self, rgb_image_path):
        result = _parse(_action_crop(rgb_image_path, left=10, top=10, right=100, bottom=50))
        assert result["width"] == 90
        assert result["height"] == 40

    def test_invalid_crop_right_less_than_left(self, rgb_image_path):
        result = _parse(_action_crop(rgb_image_path, left=100, top=0, right=50, bottom=50))
        assert "error" in result
        assert "Invalid crop region" in result["error"]

    def test_invalid_crop_bottom_less_than_top(self, rgb_image_path):
        result = _parse(_action_crop(rgb_image_path, left=0, top=80, right=100, bottom=20))
        assert "error" in result

    def test_crop_clamped_to_image_bounds(self, rgb_image_path):
        # right/bottom exceed image size; should be clamped
        result = _parse(_action_crop(rgb_image_path, left=0, top=0, right=9999, bottom=9999))
        assert result["width"] == 200
        assert result["height"] == 100


# ---------------------------------------------------------------------------
# 6. rotate
# ---------------------------------------------------------------------------

class TestActionRotate:
    def test_rotate_90(self, rgb_image_path):
        result = _parse(_action_rotate(rgb_image_path, degrees=90))
        # 200x100 rotated 90 degrees -> ~100x200
        assert result["degrees"] == 90
        assert result["width"] == 100
        assert result["height"] == 200

    def test_rotate_45(self, rgb_image_path):
        result = _parse(_action_rotate(rgb_image_path, degrees=45))
        assert result["degrees"] == 45
        # Expanded canvas should be larger than original
        assert result["width"] > 200 or result["height"] > 100

    def test_rotate_0(self, rgb_image_path):
        result = _parse(_action_rotate(rgb_image_path, degrees=0))
        assert result["width"] == 200
        assert result["height"] == 100

    def test_rotate_360(self, rgb_image_path):
        result = _parse(_action_rotate(rgb_image_path, degrees=360))
        assert result["width"] == 200
        assert result["height"] == 100


# ---------------------------------------------------------------------------
# 7. flip
# ---------------------------------------------------------------------------

class TestActionFlip:
    def test_horizontal(self, rgb_image_path):
        result = _parse(_action_flip(rgb_image_path, direction="horizontal"))
        assert result["direction"] == "horizontal"
        assert result["width"] == 200

    def test_vertical(self, rgb_image_path):
        result = _parse(_action_flip(rgb_image_path, direction="vertical"))
        assert result["direction"] == "vertical"
        assert result["height"] == 100

    def test_default_is_horizontal(self, rgb_image_path):
        result = _parse(_action_flip(rgb_image_path))
        assert result["direction"] == "horizontal"


# ---------------------------------------------------------------------------
# 8. strip
# ---------------------------------------------------------------------------

class TestActionStrip:
    def test_metadata_removed_flag(self, rgb_image_path):
        result = _parse(_action_strip(rgb_image_path))
        assert result["metadata_removed"] is True
        assert "original_kb" in result
        assert Path(result["output"]).is_file()

    def test_stripped_image_has_no_exif(self, rgb_image_path):
        result = _parse(_action_strip(rgb_image_path))
        output = result["output"]
        img = Image.open(output)
        exif = img._getexif()
        # Newly created image from putdata should have no EXIF
        assert exif is None
        img.close()


# ---------------------------------------------------------------------------
# 9. thumbnail
# ---------------------------------------------------------------------------

class TestActionThumbnail:
    def test_default_size(self, rgb_image_path):
        result = _parse(_action_thumbnail(rgb_image_path))
        assert result["width"] <= 256
        assert result["height"] <= 256

    def test_custom_size(self, rgb_image_path):
        result = _parse(_action_thumbnail(rgb_image_path, size=64))
        assert result["width"] <= 64
        assert result["height"] <= 64

    def test_size_clamped_minimum(self, rgb_image_path):
        # size < 16 should be clamped to 16
        result = _parse(_action_thumbnail(rgb_image_path, size=5))
        assert result["width"] <= 16
        assert result["height"] <= 16

    def test_output_is_jpeg(self, rgb_image_path):
        result = _parse(_action_thumbnail(rgb_image_path))
        assert result["output"].endswith(".jpg")


# ---------------------------------------------------------------------------
# 10. blur
# ---------------------------------------------------------------------------

class TestActionBlur:
    def test_default_radius(self, rgb_image_path):
        result = _parse(_action_blur(rgb_image_path))
        assert result["blur_radius"] == 2.0
        assert Path(result["output"]).is_file()

    def test_large_radius(self, rgb_image_path):
        result = _parse(_action_blur(rgb_image_path, radius=10.0))
        assert result["blur_radius"] == 10.0

    def test_radius_clamped(self, rgb_image_path):
        result = _parse(_action_blur(rgb_image_path, radius=100.0))
        assert result["blur_radius"] == 50.0  # clamped to max 50


# ---------------------------------------------------------------------------
# 11. sharpen
# ---------------------------------------------------------------------------

class TestActionSharpen:
    def test_default_factor(self, rgb_image_path):
        result = _parse(_action_sharpen(rgb_image_path))
        assert result["sharpen_factor"] == 2.0

    def test_custom_factor(self, rgb_image_path):
        result = _parse(_action_sharpen(rgb_image_path, factor=5.0))
        assert result["sharpen_factor"] == 5.0

    def test_factor_clamped(self, rgb_image_path):
        result = _parse(_action_sharpen(rgb_image_path, factor=20.0))
        assert result["sharpen_factor"] == 10.0


# ---------------------------------------------------------------------------
# 12. filter
# ---------------------------------------------------------------------------

class TestActionFilter:
    @pytest.mark.parametrize("name", ["contour", "edge", "emboss", "smooth", "detail", "edge_enhance"])
    def test_all_valid_filters(self, rgb_image_path, name):
        result = _parse(_action_filter(rgb_image_path, filter_name=name))
        assert result["filter"] == name
        assert Path(result["output"]).is_file()

    def test_invalid_filter_name(self, rgb_image_path):
        result = _parse(_action_filter(rgb_image_path, filter_name="nonexistent"))
        assert "error" in result
        assert "Unknown filter" in result["error"]
        assert "available" in result


# ---------------------------------------------------------------------------
# 13. brightness
# ---------------------------------------------------------------------------

class TestActionBrightness:
    def test_increase(self, rgb_image_path):
        result = _parse(_action_brightness(rgb_image_path, factor=2.0))
        assert result["brightness_factor"] == 2.0
        assert Path(result["output"]).is_file()

    def test_decrease(self, rgb_image_path):
        result = _parse(_action_brightness(rgb_image_path, factor=0.5))
        assert result["brightness_factor"] == 0.5

    def test_factor_clamped_to_max(self, rgb_image_path):
        result = _parse(_action_brightness(rgb_image_path, factor=10.0))
        assert result["brightness_factor"] == 5.0


# ---------------------------------------------------------------------------
# 14. contrast
# ---------------------------------------------------------------------------

class TestActionContrast:
    def test_increase(self, rgb_image_path):
        result = _parse(_action_contrast(rgb_image_path, factor=2.0))
        assert result["contrast_factor"] == 2.0

    def test_decrease(self, rgb_image_path):
        result = _parse(_action_contrast(rgb_image_path, factor=0.3))
        assert result["contrast_factor"] == 0.3


# ---------------------------------------------------------------------------
# 15. saturation
# ---------------------------------------------------------------------------

class TestActionSaturation:
    def test_increase(self, rgb_image_path):
        result = _parse(_action_saturation(rgb_image_path, factor=2.5))
        assert result["saturation_factor"] == 2.5

    def test_decrease(self, rgb_image_path):
        result = _parse(_action_saturation(rgb_image_path, factor=0.3))
        assert result["saturation_factor"] == 0.3

    def test_zero_saturation(self, rgb_image_path):
        result = _parse(_action_saturation(rgb_image_path, factor=0.0))
        assert result["saturation_factor"] == 0.0
        # factor 0 = fully desaturated
        assert Path(result["output"]).is_file()


# ---------------------------------------------------------------------------
# 16. grayscale
# ---------------------------------------------------------------------------

class TestActionGrayscale:
    def test_mode_change(self, rgb_image_path):
        result = _parse(_action_grayscale(rgb_image_path))
        assert result["mode"] == "grayscale"
        output = result["output"]
        img = Image.open(output)
        assert img.mode == "L"
        img.close()

    def test_output_is_jpeg(self, rgb_image_path):
        result = _parse(_action_grayscale(rgb_image_path))
        assert result["output"].endswith(".jpg")


# ---------------------------------------------------------------------------
# 17. invert
# ---------------------------------------------------------------------------

class TestActionInvert:
    def test_rgb_invert(self, rgb_image_path):
        result = _parse(_action_invert(rgb_image_path))
        assert Path(result["output"]).is_file()
        # Verify at least one pixel is inverted
        original = Image.open(rgb_image_path)
        inverted = Image.open(result["output"])
        op = original.getpixel((0, 0))
        ip = inverted.getpixel((0, 0))
        # Each channel should be ~(255 - original) with JPEG compression tolerance
        for o, i in zip(op, ip):
            assert abs((255 - o) - i) < 10
        original.close()
        inverted.close()

    def test_rgba_invert(self, rgba_image_path):
        result = _parse(_action_invert(rgba_image_path))
        assert Path(result["output"]).is_file()


# ---------------------------------------------------------------------------
# 18. text
# ---------------------------------------------------------------------------

class TestActionText:
    def test_basic_text_overlay(self, rgb_image_path):
        result = _parse(_action_text(rgb_image_path, text="Hello", x=10, y=10))
        assert result["text"] == "Hello"
        assert Path(result["output"]).is_file()

    def test_long_text_truncated_in_meta(self, rgb_image_path):
        long = "A" * 100
        result = _parse(_action_text(rgb_image_path, text=long))
        assert len(result["text"]) == 50  # truncated to 50 chars

    def test_output_has_correct_dimensions(self, rgb_image_path):
        result = _parse(_action_text(rgb_image_path, text="Test"))
        assert result["width"] == 200
        assert result["height"] == 100


# ---------------------------------------------------------------------------
# 19. draw
# ---------------------------------------------------------------------------

class TestActionDraw:
    def test_rectangle(self, rgb_image_path):
        result = _parse(_action_draw(rgb_image_path, shape="rectangle", x1=10, y1=10, x2=50, y2=50))
        assert result["shape"] == "rectangle"
        assert Path(result["output"]).is_file()

    def test_ellipse(self, rgb_image_path):
        result = _parse(_action_draw(rgb_image_path, shape="ellipse", x1=10, y1=10, x2=80, y2=80))
        assert result["shape"] == "ellipse"

    def test_line(self, rgb_image_path):
        result = _parse(_action_draw(rgb_image_path, shape="line", x1=0, y1=0, x2=100, y2=100))
        assert result["shape"] == "line"

    def test_invalid_shape(self, rgb_image_path):
        result = _parse(_action_draw(rgb_image_path, shape="star"))
        assert "error" in result
        assert "Unknown shape" in result["error"]

    def test_with_fill(self, rgb_image_path):
        result = _parse(_action_draw(
            rgb_image_path, shape="rectangle",
            x1=5, y1=5, x2=50, y2=50, color="blue", fill="yellow",
        ))
        assert result["shape"] == "rectangle"


# ---------------------------------------------------------------------------
# 20. paste
# ---------------------------------------------------------------------------

class TestActionPaste:
    def test_basic_paste(self, rgb_image_path, small_overlay_path):
        result = _parse(_action_paste(rgb_image_path, overlay_path=small_overlay_path, x=10, y=10))
        assert "overlay" in result
        assert Path(result["output"]).is_file()
        # Base image dimensions preserved
        assert result["width"] == 200
        assert result["height"] == 100

    def test_paste_with_opacity(self, rgb_image_path, small_overlay_path):
        result = _parse(_action_paste(
            rgb_image_path, overlay_path=small_overlay_path,
            x=0, y=0, opacity=0.5,
        ))
        assert Path(result["output"]).is_file()

    def test_paste_overlay_not_found(self, rgb_image_path):
        with pytest.raises(ValueError, match="(File not found|Image file not found)"):
            _action_paste(rgb_image_path, overlay_path="/nonexistent/overlay.png")


# ---------------------------------------------------------------------------
# 21. merge
# ---------------------------------------------------------------------------

class TestActionMerge:
    def test_horizontal_merge(self, rgb_image_path, second_image_path):
        result = _parse(_action_merge(rgb_image_path, image_paths=second_image_path, direction="horizontal"))
        assert result["direction"] == "horizontal"
        assert result["image_count"] == 2
        # Horizontal: widths sum, height = max
        assert result["width"] == 400  # 200 + 200
        assert result["height"] == 100

    def test_vertical_merge(self, rgb_image_path, second_image_path):
        result = _parse(_action_merge(rgb_image_path, image_paths=second_image_path, direction="vertical"))
        assert result["direction"] == "vertical"
        assert result["image_count"] == 2
        assert result["width"] == 200
        assert result["height"] == 200  # 100 + 100

    def test_insufficient_images(self, rgb_image_path):
        result = _parse(_action_merge(rgb_image_path, image_paths=""))
        assert "error" in result
        assert "at least 2" in result["error"].lower()


# ---------------------------------------------------------------------------
# 22. colors
# ---------------------------------------------------------------------------

class TestActionColors:
    def test_extract_dominant_colors(self, rgb_image_path):
        result = _parse(_action_colors(rgb_image_path, count=3))
        assert "dominant_colors" in result
        colors = result["dominant_colors"]
        assert len(colors) == 3
        for c in colors:
            assert "rgb" in c
            assert "hex" in c
            assert len(c["rgb"]) == 3
            assert c["hex"].startswith("#")
            assert len(c["hex"]) == 7

    def test_count_clamped(self, rgb_image_path):
        result = _parse(_action_colors(rgb_image_path, count=50))
        # Clamped to 20 max
        assert len(result["dominant_colors"]) <= 20


# ---------------------------------------------------------------------------
# 23. histogram
# ---------------------------------------------------------------------------

class TestActionHistogram:
    def test_rgb_stats(self, rgb_image_path):
        result = _parse(_action_histogram(rgb_image_path))
        for channel in ("red", "green", "blue"):
            assert channel in result
            assert "min" in result[channel]
            assert "max" in result[channel]
            assert "mean" in result[channel]
            assert isinstance(result[channel]["mean"], float)

    def test_grayscale_converted_to_rgb(self, grayscale_image_path):
        # Grayscale images are converted to RGB internally
        result = _parse(_action_histogram(grayscale_image_path))
        assert "red" in result
        assert "green" in result
        assert "blue" in result


# ---------------------------------------------------------------------------
# 24. transparent
# ---------------------------------------------------------------------------

class TestActionTransparent:
    def test_make_white_transparent(self, white_image_path):
        result = _parse(_action_transparent(white_image_path, target_color="white", tolerance=30))
        assert result["pixels_made_transparent"] > 0
        assert result["total_pixels"] == 50 * 50
        assert result["percent_transparent"] > 0
        assert result["output"].endswith(".png")

    def test_hex_color_target(self, white_image_path):
        result = _parse(_action_transparent(white_image_path, target_color="#ffffff", tolerance=10))
        assert result["pixels_made_transparent"] > 0

    def test_no_matching_pixels(self, rgb_image_path):
        # Try to make blue transparent on an image that has no blue
        result = _parse(_action_transparent(rgb_image_path, target_color="blue", tolerance=5))
        assert result["pixels_made_transparent"] == 0

    def test_named_colors(self, white_image_path):
        # "black" should not match a mostly white image with low tolerance
        result = _parse(_action_transparent(white_image_path, target_color="black", tolerance=5))
        assert result["pixels_made_transparent"] == 0


# ---------------------------------------------------------------------------
# Dispatch layer: image_process_tool()
# ---------------------------------------------------------------------------

class TestImageProcessToolDispatch:
    def test_unknown_action(self, rgb_image_path):
        result = _parse(image_process_tool(action="nonexistent", image_path=rgb_image_path))
        assert "error" in result
        assert "Unknown action" in result["error"]
        assert "available_actions" in result

    def test_empty_image_path(self):
        result = _parse(image_process_tool(action="info", image_path=""))
        assert "error" in result
        assert "image_path is required" in result["error"]

    def test_missing_file(self):
        result = _parse(image_process_tool(action="info", image_path="/nonexistent/file.jpg"))
        assert "error" in result
        assert "not found" in result["error"].lower() or "Image file" in result["error"]

    def test_dispatch_info(self, rgb_image_path):
        result = _parse(image_process_tool(action="info", image_path=rgb_image_path))
        assert result["width"] == 200
        assert result["height"] == 100

    def test_dispatch_resize_with_kwargs(self, rgb_image_path):
        result = _parse(image_process_tool(action="resize", image_path=rgb_image_path, width=50))
        assert result["width"] == 50

    def test_dispatch_convert(self, rgb_image_path):
        result = _parse(image_process_tool(action="convert", image_path=rgb_image_path, format="PNG"))
        assert result["format"] == "PNG"

    def test_dispatch_crop(self, rgb_image_path):
        result = _parse(image_process_tool(
            action="crop", image_path=rgb_image_path,
            left=0, top=0, right=100, bottom=50,
        ))
        assert result["width"] == 100

    def test_dispatch_all_actions_have_entries(self):
        """Ensure dispatch table covers all 24 documented actions (excluding batch)."""
        expected = {
            "info", "resize", "convert", "compress", "crop", "rotate", "flip",
            "strip", "thumbnail", "blur", "sharpen", "filter", "brightness",
            "contrast", "saturation", "grayscale", "invert", "text", "draw",
            "paste", "merge", "colors", "histogram", "transparent",
        }
        # Calling with a dummy action to see the available list
        result = _parse(image_process_tool(action="__bogus__", image_path="/tmp/x"))
        available = set(result["available_actions"])
        assert expected.issubset(available), f"Missing actions: {expected - available}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_file_not_found(self):
        with pytest.raises(ValueError, match="(File not found|Image file not found)"):
            _action_info("/definitely/not/a/real/file.jpg")

    def test_oversized_file(self, tmp_path):
        """Simulate a file that exceeds _MAX_INPUT_SIZE."""
        big_file = tmp_path / "big.jpg"
        # Create a valid JPEG then inflate the file with trailing bytes
        img = Image.new("RGB", (10, 10), color=(0, 0, 0))
        img.save(str(big_file), format="JPEG")
        img.close()
        # Truncate to a size just over the limit to guarantee st_size exceeds it
        with open(str(big_file), "r+b") as f:
            f.truncate(_MAX_INPUT_SIZE + 1)
        with pytest.raises(ValueError, match="too large"):
            _action_info(str(big_file))

    def test_pillow_not_available(self, rgb_image_path):
        with patch("tools.image_process_tool._PILLOW_AVAILABLE", False):
            result = _parse(image_process_tool(action="info", image_path=rgb_image_path))
            assert "error" in result
            assert "Pillow" in result["error"]

    def test_processing_exception_caught(self, rgb_image_path):
        """Exceptions inside action handlers are caught and returned as JSON."""
        with patch("tools.image_process_tool._action_info", side_effect=RuntimeError("boom")):
            result = _parse(image_process_tool(action="info", image_path=rgb_image_path))
            assert "error" in result
            assert "RuntimeError" in result["error"] or "boom" in result["error"]


# ---------------------------------------------------------------------------
# Output file creation
# ---------------------------------------------------------------------------

class TestOutputFiles:
    def test_output_dir_created(self, rgb_image_path, monkeypatch):
        """Ensure the output directory is created when saving a processed image."""
        result = _parse(_action_resize(rgb_image_path, width=50))
        output_path = Path(result["output"])
        assert output_path.is_file()
        assert output_path.parent.exists()

    def test_output_has_correct_extension_jpeg(self, rgb_image_path):
        result = _parse(_action_compress(rgb_image_path, quality=80))
        assert result["output"].endswith(".jpg")

    def test_output_has_correct_extension_png(self, rgba_image_path):
        result = _parse(_action_convert(rgba_image_path, target_format="PNG"))
        assert result["output"].endswith(".png")

    def test_multiple_operations_unique_files(self, rgb_image_path):
        r1 = _parse(_action_blur(rgb_image_path, radius=1.0))
        r2 = _parse(_action_blur(rgb_image_path, radius=5.0))
        assert r1["output"] != r2["output"]
        assert Path(r1["output"]).is_file()
        assert Path(r2["output"]).is_file()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_resize_width_only_no_aspect(self, rgb_image_path):
        """Width-only resize with keep_aspect=False keeps original height."""
        result = _parse(_action_resize(rgb_image_path, width=80, keep_aspect=False))
        assert result["width"] == 80
        assert result["height"] == 100  # original height preserved

    def test_compress_quality_zero_clamped(self, rgb_image_path):
        result = _parse(_action_compress(rgb_image_path, quality=0))
        assert result["quality"] == 1  # clamped to 1

    def test_blur_tiny_radius(self, rgb_image_path):
        result = _parse(_action_blur(rgb_image_path, radius=0.01))
        assert result["blur_radius"] == 0.1  # clamped

    def test_thumbnail_huge_size_clamped(self, rgb_image_path):
        result = _parse(_action_thumbnail(rgb_image_path, size=9999))
        # Clamped to 1024 max, and image is only 200x100 so thumbnail stays within original
        assert result["width"] <= 200
        assert result["height"] <= 100

    def test_transparent_tolerance_clamped(self, white_image_path):
        result = _parse(_action_transparent(white_image_path, target_color="white", tolerance=999))
        # tolerance clamped to 255, so everything should match
        assert result["pixels_made_transparent"] == 50 * 50

    def test_colors_count_1(self, rgb_image_path):
        result = _parse(_action_colors(rgb_image_path, count=1))
        assert len(result["dominant_colors"]) == 1

    def test_grayscale_image_histogram(self, grayscale_image_path):
        result = _parse(_action_histogram(grayscale_image_path))
        # After conversion to RGB, all channels should have the same mean
        assert abs(result["red"]["mean"] - result["green"]["mean"]) < 1
        assert abs(result["green"]["mean"] - result["blue"]["mean"]) < 1

    def test_invert_grayscale_image(self, grayscale_image_path):
        result = _parse(_action_invert(grayscale_image_path))
        assert Path(result["output"]).is_file()

    def test_convert_rgba_to_jpeg(self, rgba_image_path):
        """RGBA images must be converted to RGB when saving as JPEG."""
        result = _parse(_action_convert(rgba_image_path, target_format="JPEG"))
        assert result["format"] == "JPEG"
        img = Image.open(result["output"])
        assert img.mode == "RGB"
        img.close()
