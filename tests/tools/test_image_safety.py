"""Tests for tools/image_safety.py — image format validation and size limits."""

import tempfile
from pathlib import Path

import pytest

from tools.image_safety import (
    MAX_IMAGE_SIZE_BYTES,
    validate_image_file,
    check_content_length,
    get_real_mime_type,
)


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


# ---------------------------------------------------------------------------
# Magic bytes validation
# ---------------------------------------------------------------------------


class TestValidateImageFile:
    def test_real_jpeg(self, tmp_dir):
        f = tmp_dir / "real.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        ok, err, mime = validate_image_file(f)
        assert ok is True
        assert mime == "image/jpeg"

    def test_real_png(self, tmp_dir):
        f = tmp_dir / "real.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        ok, err, mime = validate_image_file(f)
        assert ok is True
        assert mime == "image/png"

    def test_real_gif(self, tmp_dir):
        f = tmp_dir / "real.gif"
        f.write_bytes(b"GIF89a" + b"\x00" * 100)
        ok, err, mime = validate_image_file(f)
        assert ok is True
        assert mime == "image/gif"

    def test_real_webp(self, tmp_dir):
        f = tmp_dir / "real.webp"
        # WebP needs proper RIFF container with VP8 chunk
        f.write_bytes(b"RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x00" + b"\x00" * 100)
        ok, err, mime = validate_image_file(f)
        assert ok is True
        assert mime == "image/webp"

    def test_fake_jpeg_html(self, tmp_dir):
        """A .jpg file that is actually HTML should be rejected."""
        f = tmp_dir / "fake.jpg"
        f.write_bytes(b"<html><script>alert(1)</script></html>")
        ok, err, mime = validate_image_file(f)
        assert ok is False
        assert "not a recognized image" in err.lower() or "not an image" in err.lower()

    def test_fake_jpeg_executable(self, tmp_dir):
        """A .jpg file that is actually an ELF binary should be rejected."""
        f = tmp_dir / "fake.jpg"
        f.write_bytes(b"\x7fELF" + b"\x00" * 100)
        ok, err, mime = validate_image_file(f)
        assert ok is False

    def test_fake_png_pdf(self, tmp_dir):
        """A .png file that is actually a PDF should be rejected."""
        f = tmp_dir / "fake.png"
        f.write_bytes(b"%PDF-1.4" + b"\x00" * 100)
        ok, err, mime = validate_image_file(f)
        assert ok is False
        assert "not an image" in err.lower()

    def test_svg_blocked(self, tmp_dir):
        """SVG files should be blocked (can contain scripts)."""
        f = tmp_dir / "test.svg"
        f.write_bytes(b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>')
        ok, err, mime = validate_image_file(f)
        assert ok is False
        assert "svg" in err.lower()

    def test_svgz_blocked(self, tmp_dir):
        """Compressed SVG should also be blocked."""
        f = tmp_dir / "test.svgz"
        f.write_bytes(b"\x1f\x8b" + b"\x00" * 50)  # gzip header
        ok, err, mime = validate_image_file(f)
        assert ok is False
        assert "svg" in err.lower()

    def test_empty_file(self, tmp_dir):
        f = tmp_dir / "empty.jpg"
        f.write_bytes(b"")
        ok, err, mime = validate_image_file(f)
        assert ok is False
        assert "empty" in err.lower()

    def test_missing_file(self, tmp_dir):
        f = tmp_dir / "nonexistent.jpg"
        ok, err, mime = validate_image_file(f)
        assert ok is False
        assert "not found" in err.lower()

    def test_oversized_file(self, tmp_dir):
        """Files exceeding MAX_IMAGE_SIZE_BYTES should be rejected."""
        from unittest.mock import patch
        f = tmp_dir / "huge.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        # Temporarily lower the limit to test rejection without writing 20MB
        with patch("tools.image_safety.MAX_IMAGE_SIZE_BYTES", 50):
            ok, err, mime = validate_image_file(f)
            assert ok is False
            assert "too large" in err.lower()

    def test_random_bytes(self, tmp_dir):
        """Random bytes that don't match any format should be rejected."""
        f = tmp_dir / "random.jpg"
        f.write_bytes(bytes(range(256)) * 4)
        ok, err, mime = validate_image_file(f)
        assert ok is False


# ---------------------------------------------------------------------------
# Content-Length pre-flight check
# ---------------------------------------------------------------------------


class TestCheckContentLength:
    def test_small_file(self):
        ok, err = check_content_length(1024)
        assert ok is True

    def test_at_limit(self):
        ok, err = check_content_length(MAX_IMAGE_SIZE_BYTES)
        assert ok is True

    def test_over_limit(self):
        ok, err = check_content_length(MAX_IMAGE_SIZE_BYTES + 1)
        assert ok is False
        assert "too large" in err.lower()

    def test_none_allows(self):
        """Unknown Content-Length should be allowed (checked after download)."""
        ok, err = check_content_length(None)
        assert ok is True

    def test_zero(self):
        ok, err = check_content_length(0)
        assert ok is True


# ---------------------------------------------------------------------------
# MIME type detection
# ---------------------------------------------------------------------------


class TestGetRealMimeType:
    def test_jpeg(self, tmp_dir):
        f = tmp_dir / "test.jpg"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        assert get_real_mime_type(f) == "image/jpeg"

    def test_png(self, tmp_dir):
        f = tmp_dir / "test.png"
        f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        assert get_real_mime_type(f) == "image/png"

    def test_wrong_extension(self, tmp_dir):
        """A .png file with JPEG magic bytes should return image/jpeg."""
        f = tmp_dir / "misleading.png"
        f.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
        assert get_real_mime_type(f) == "image/jpeg"

    def test_fallback_on_unknown(self, tmp_dir):
        """Unknown content falls back to extension-based detection."""
        f = tmp_dir / "unknown.webp"
        f.write_bytes(b"not real webp content but has extension")
        mime = get_real_mime_type(f)
        assert mime == "image/webp"  # extension fallback
