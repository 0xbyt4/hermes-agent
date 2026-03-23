"""Image safety checks — validates file format and enforces size limits.

Prevents:
  - Fake images: files with image extensions but non-image content (HTML, scripts, executables)
  - Oversized files: large downloads that could exhaust memory or disk
  - SVG with embedded scripts: SVG can contain <script>, <iframe>, event handlers

Uses the `filetype` library for magic-byte detection (pure Python, no system deps).
Falls back to extension-based detection if filetype is unavailable.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Maximum image file size (20 MB)
MAX_IMAGE_SIZE_BYTES = 20 * 1024 * 1024

# Allowed image MIME types (magic-byte verified)
_ALLOWED_IMAGE_MIMES = frozenset({
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "image/heic",       # iPhone default camera format
    "image/heif",       # HEIF container (related to HEIC)
    "image/avif",       # Next-gen format (Chrome, Firefox)
})

# SVG is text-based and can contain scripts — blocked by default
_SVG_EXTENSIONS = frozenset({".svg", ".svgz"})


def validate_image_file(path: Path) -> Tuple[bool, str, Optional[str]]:
    """Validate that a file is a real image with an allowed format.

    Args:
        path: Path to the downloaded image file.

    Returns:
        (is_valid, error_message, detected_mime)
        - is_valid: True if the file is a safe image
        - error_message: empty string if valid, description of problem if not
        - detected_mime: the real MIME type detected from content, or None
    """
    if not path.is_file():
        return False, "File not found", None

    # Check file size
    size = path.stat().st_size
    if size == 0:
        return False, "File is empty (0 bytes)", None
    if size > MAX_IMAGE_SIZE_BYTES:
        size_mb = size / (1024 * 1024)
        limit_mb = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
        return False, f"File too large ({size_mb:.1f} MB, limit {limit_mb:.0f} MB)", None

    # Block SVG by extension (SVG is text-based, filetype can't detect it via magic bytes)
    if path.suffix.lower() in _SVG_EXTENSIONS:
        return False, "SVG files are not supported (may contain embedded scripts)", None

    # Detect real file type from magic bytes
    try:
        import filetype
        kind = filetype.guess(str(path))
        if kind is None:
            return False, "Unable to detect file type — not a recognized image format", None
        if kind.mime not in _ALLOWED_IMAGE_MIMES:
            return False, f"Not an image file (detected: {kind.mime})", kind.mime
        return True, "", kind.mime
    except ImportError:
        # filetype not installed — fall back to extension check with warning
        logger.warning("filetype library not available — using extension-based image validation only")
        return _fallback_extension_check(path)
    except Exception as e:
        logger.warning("Image validation error: %s", e)
        return False, f"Image validation failed: {e}", None


def check_content_length(content_length: Optional[int]) -> Tuple[bool, str]:
    """Pre-flight check on Content-Length header before downloading.

    Args:
        content_length: Value from the Content-Length HTTP header, or None.

    Returns:
        (is_ok, error_message)
    """
    if content_length is None:
        return True, ""  # Unknown size — proceed and check after download
    if content_length > MAX_IMAGE_SIZE_BYTES:
        size_mb = content_length / (1024 * 1024)
        limit_mb = MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
        return False, f"Image too large ({size_mb:.1f} MB, limit {limit_mb:.0f} MB)"
    return True, ""


def get_real_mime_type(path: Path) -> str:
    """Detect the real MIME type from file content, not extension.

    Falls back to extension-based detection if filetype is unavailable.
    """
    try:
        import filetype
        kind = filetype.guess(str(path))
        if kind and kind.mime.startswith("image/"):
            return kind.mime
    except (ImportError, Exception):
        pass

    # Fallback: extension-based
    ext_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".gif": "image/gif",
        ".webp": "image/webp", ".bmp": "image/bmp",
        ".heic": "image/heic", ".heif": "image/heif",
        ".avif": "image/avif",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    return ext_map.get(path.suffix.lower(), "image/jpeg")


def _fallback_extension_check(path: Path) -> Tuple[bool, str, Optional[str]]:
    """Extension-only validation when filetype library is not available."""
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif", ".avif"}
    ext = path.suffix.lower()
    if ext not in allowed_extensions:
        return False, f"Unsupported image extension: {ext}", None
    return True, "", None
