"""
Image Processing Tool — full Pillow-powered image manipulation for the agent.

Actions:
  info        — Get image metadata (dimensions, format, EXIF, GPS, color info)
  resize      — Resize image to specified dimensions
  convert     — Convert between formats (HEIC->JPEG, PNG->WebP, etc.)
  compress    — Reduce file size with quality setting
  crop        — Crop a region from the image
  rotate      — Rotate image by degrees
  flip        — Flip/mirror horizontally or vertically
  strip       — Remove EXIF/metadata (re-encode clean)
  thumbnail   — Generate a small thumbnail
  blur        — Apply Gaussian blur
  sharpen     — Sharpen image
  filter      — Apply filters (contour, edge, emboss, smooth, detail)
  brightness  — Adjust brightness
  contrast    — Adjust contrast
  saturation  — Adjust color saturation
  grayscale   — Convert to grayscale
  invert      — Invert colors (negative)
  text        — Add text overlay (watermark, annotation)
  draw        — Draw shapes (rectangle, circle, line)
  paste       — Paste/overlay one image onto another
  merge       — Merge multiple images side by side or in a grid
  colors      — Extract dominant colors
  histogram   — Get color histogram data
  transparent — Make a color transparent (simple background removal)

Requires: Pillow (optional dependency — pip install hermes-agent[image])
"""

import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tools.registry import registry

logger = logging.getLogger(__name__)

_PILLOW_AVAILABLE = False
try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont, ImageStat, ExifTags
    _PILLOW_AVAILABLE = True
except ImportError:
    pass

# Output directory for processed images
_OUTPUT_DIR = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")) / "image_output"

# Safety limits
_MAX_DIMENSION = 8192
_MAX_INPUT_SIZE = 50 * 1024 * 1024  # 50MB
_ALLOWED_OUTPUT_FORMATS = {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF"}


def _ensure_output_dir() -> Path:
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTPUT_DIR


_ALLOWED_INPUT_DIRS = None  # Populated lazily


def _get_allowed_dirs():
    """Return directories from which images can be read."""
    global _ALLOWED_INPUT_DIRS
    if _ALLOWED_INPUT_DIRS is None:
        home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes")).resolve()
        import tempfile
        _ALLOWED_INPUT_DIRS = [
            home,                                    # hermes home (image_cache, image_output)
            Path.home().resolve(),                   # user home directory
            Path.cwd().resolve(),                    # current working directory
            Path("/tmp").resolve(),                   # temp files
            Path(tempfile.gettempdir()).resolve(),    # system temp (macOS: /private/var/folders/...)
        ]
    return _ALLOWED_INPUT_DIRS


# Max pixels after decompression (prevents decompression bombs)
_MAX_IMAGE_PIXELS = 64 * 1024 * 1024  # 64M pixels


def _validate_path(path_str: str) -> Path:
    path = Path(os.path.expanduser(path_str)).resolve()
    path_lower = str(path).lower()

    # Block sensitive system paths BEFORE existence check (prevent oracle)
    blocked = ("/etc", "/private/etc", "/proc", "/sys", "/dev",
               "/var/run", "/private/var/run", "/var/log", "/private/var/log",
               "/boot", "/root")
    if any(path_lower.startswith(b) for b in blocked):
        raise ValueError("Access denied: system path not allowed")

    # Block ALL dotfiles/dirs (except .hermes) regardless of location
    for part in path.parts:
        if part.startswith(".") and part not in (".hermes", "."):
            raise ValueError("Access denied: hidden files not allowed")

    if not path.is_file():
        raise ValueError("Image file not found")

    size = path.stat().st_size
    if size > _MAX_INPUT_SIZE:
        raise ValueError(f"File too large (limit {_MAX_INPUT_SIZE // 1024 // 1024} MB)")

    # Allowlist check: must be under an allowed directory
    allowed = _get_allowed_dirs()
    if not any(str(path).startswith(str(d)) for d in allowed):
        raise ValueError("Access denied: path outside allowed directories")

    return path


def _check_pixel_limit(img):
    """Reject images that decompress to excessive pixel counts."""
    pixels = img.width * img.height
    if pixels > _MAX_IMAGE_PIXELS:
        img.close()
        raise ValueError(f"Image too large ({img.width}x{img.height} = {pixels // 1_000_000}M pixels, limit {_MAX_IMAGE_PIXELS // 1_000_000}M)")


def _output_path(fmt: str) -> Path:
    ext = fmt.lower()
    if ext == "jpeg":
        ext = "jpg"
    return _ensure_output_dir() / f"processed_{uuid.uuid4().hex[:8]}.{ext}"


def _open_image(path_str: str) -> Tuple[Path, Image.Image]:
    """Validate path, open image, check pixel limit."""
    path = _validate_path(path_str)
    img = Image.open(path)
    _check_pixel_limit(img)
    return path, img


def _save_image(img: Image.Image, fmt: str = None, quality: int = 90) -> Tuple[Path, dict]:
    """Save image and return (path, metadata)."""
    fmt = fmt or img.format or "JPEG"
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt == "JPEG" and img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    out = _output_path(fmt)
    save_kwargs = {"format": fmt}
    if fmt in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
    if fmt == "PNG":
        save_kwargs["optimize"] = True
    img.save(out, **save_kwargs)
    w, h = img.size
    return out, {"output": str(out), "width": w, "height": h, "file_size_kb": round(out.stat().st_size / 1024, 1)}


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def _action_info(image_path: str) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    info = {
        "file": str(path),
        "format": img.format,
        "mode": img.mode,
        "width": img.width,
        "height": img.height,
        "file_size_kb": round(path.stat().st_size / 1024, 1),
        "has_alpha": img.mode in ("RGBA", "LA", "PA"),
        "is_animated": getattr(img, "is_animated", False),
        "n_frames": getattr(img, "n_frames", 1),
    }

    # Color stats
    try:
        stat = ImageStat.Stat(img)
        info["mean_rgb"] = [round(v, 1) for v in stat.mean[:3]]
    except Exception:
        pass

    # EXIF
    exif_data = {}
    try:
        raw_exif = img.getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                if isinstance(value, (bytes, bytearray)):
                    continue
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                exif_data[tag_name] = str(value)
            gps_info = raw_exif.get(34853)
            if gps_info:
                gps = {}
                for key, val in gps_info.items():
                    gps[ExifTags.GPSTAGS.get(key, str(key))] = str(val)
                exif_data["GPS"] = gps
    except Exception:
        pass
    if exif_data:
        info["exif"] = exif_data

    img.close()
    return json.dumps(info, indent=2, default=str)


def _action_resize(image_path: str, width: int = 0, height: int = 0, keep_aspect: bool = True) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    if width <= 0 and height <= 0:
        return json.dumps({"error": "Specify width and/or height"})
    width = min(width, _MAX_DIMENSION) if width > 0 else 0
    height = min(height, _MAX_DIMENSION) if height > 0 else 0
    if keep_aspect:
        if width > 0 and height > 0:
            img.thumbnail((width, height), Image.LANCZOS)
        elif width > 0:
            ratio = width / img.width
            img = img.resize((width, int(img.height * ratio)), Image.LANCZOS)
        else:
            ratio = height / img.height
            img = img.resize((int(img.width * ratio), height), Image.LANCZOS)
    else:
        w = width if width > 0 else img.width
        h = height if height > 0 else img.height
        img = img.resize((w, h), Image.LANCZOS)
    out, meta = _save_image(img)
    img.close()
    return json.dumps(meta)


def _action_convert(image_path: str, target_format: str) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    fmt = target_format.upper()
    if fmt == "JPG":
        fmt = "JPEG"
    if fmt not in _ALLOWED_OUTPUT_FORMATS:
        return json.dumps({"error": f"Unsupported format: {target_format}. Allowed: {', '.join(sorted(_ALLOWED_OUTPUT_FORMATS))}"})
    out, meta = _save_image(img, fmt=fmt)
    meta["format"] = fmt
    img.close()
    return json.dumps(meta)


def _action_compress(image_path: str, quality: int = 70) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    quality = max(1, min(100, quality))
    original_size = path.stat().st_size
    out, meta = _save_image(img, quality=quality)
    new_size = out.stat().st_size
    meta["original_kb"] = round(original_size / 1024, 1)
    meta["reduction_percent"] = round((1 - new_size / original_size) * 100, 1) if original_size > 0 else 0
    meta["quality"] = quality
    img.close()
    return json.dumps(meta)


def _action_crop(image_path: str, left: int, top: int, right: int, bottom: int) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    left, top = max(0, left), max(0, top)
    right, bottom = min(img.width, right), min(img.height, bottom)
    if right <= left or bottom <= top:
        return json.dumps({"error": "Invalid crop region"})
    cropped = img.crop((left, top, right, bottom))
    out, meta = _save_image(cropped)
    img.close()
    cropped.close()
    return json.dumps(meta)


def _action_rotate(image_path: str, degrees: float) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    _check_pixel_limit(img)
    rotated = img.rotate(degrees, expand=True, resample=Image.BICUBIC)
    _check_pixel_limit(rotated)  # expand=True can grow canvas beyond limit
    out, meta = _save_image(rotated)
    meta["degrees"] = degrees
    img.close()
    rotated.close()
    return json.dumps(meta)


def _action_flip(image_path: str, direction: str = "horizontal") -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    if direction == "vertical":
        flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
    else:
        flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    out, meta = _save_image(flipped)
    meta["direction"] = direction
    img.close()
    flipped.close()
    return json.dumps(meta)


def _action_strip(image_path: str) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    # Re-encode without metadata — use frombytes instead of list(getdata()) to avoid memory bomb
    clean = Image.frombytes(img.mode, img.size, img.tobytes())
    original_size = path.stat().st_size
    out, meta = _save_image(clean, quality=95)
    meta["original_kb"] = round(original_size / 1024, 1)
    meta["metadata_removed"] = True
    img.close()
    clean.close()
    return json.dumps(meta)


def _action_thumbnail(image_path: str, size: int = 256) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    size = max(16, min(size, 1024))
    img.thumbnail((size, size), Image.LANCZOS)
    out, meta = _save_image(img, fmt="JPEG", quality=80)
    img.close()
    return json.dumps(meta)


def _action_blur(image_path: str, radius: float = 2.0) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    radius = max(0.1, min(radius, 50.0))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
    out, meta = _save_image(blurred)
    meta["blur_radius"] = radius
    img.close()
    blurred.close()
    return json.dumps(meta)


def _action_sharpen(image_path: str, factor: float = 2.0) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    factor = max(0.0, min(factor, 10.0))
    enhanced = ImageEnhance.Sharpness(img).enhance(factor)
    out, meta = _save_image(enhanced)
    meta["sharpen_factor"] = factor
    img.close()
    return json.dumps(meta)


def _action_filter(image_path: str, filter_name: str = "contour") -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    filters = {
        "contour": ImageFilter.CONTOUR,
        "edge": ImageFilter.FIND_EDGES,
        "emboss": ImageFilter.EMBOSS,
        "smooth": ImageFilter.SMOOTH_MORE,
        "detail": ImageFilter.DETAIL,
        "edge_enhance": ImageFilter.EDGE_ENHANCE_MORE,
    }
    if filter_name not in filters:
        return json.dumps({"error": f"Unknown filter: {filter_name}", "available": list(filters.keys())})
    filtered = img.filter(filters[filter_name])
    out, meta = _save_image(filtered)
    meta["filter"] = filter_name
    img.close()
    filtered.close()
    return json.dumps(meta)


def _action_brightness(image_path: str, factor: float = 1.5) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    factor = max(0.0, min(factor, 5.0))
    enhanced = ImageEnhance.Brightness(img).enhance(factor)
    out, meta = _save_image(enhanced)
    meta["brightness_factor"] = factor
    img.close()
    return json.dumps(meta)


def _action_contrast(image_path: str, factor: float = 1.5) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    factor = max(0.0, min(factor, 5.0))
    enhanced = ImageEnhance.Contrast(img).enhance(factor)
    out, meta = _save_image(enhanced)
    meta["contrast_factor"] = factor
    img.close()
    return json.dumps(meta)


def _action_saturation(image_path: str, factor: float = 1.5) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    factor = max(0.0, min(factor, 5.0))
    enhanced = ImageEnhance.Color(img).enhance(factor)
    out, meta = _save_image(enhanced)
    meta["saturation_factor"] = factor
    img.close()
    return json.dumps(meta)


def _action_grayscale(image_path: str) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    gray = img.convert("L")
    out, meta = _save_image(gray, fmt="JPEG")
    meta["mode"] = "grayscale"
    img.close()
    gray.close()
    return json.dumps(meta)


def _action_invert(image_path: str) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    from PIL import ImageOps
    if img.mode == "RGBA":
        r, g, b, a = img.split()
        rgb = Image.merge("RGB", (r, g, b))
        inverted = ImageOps.invert(rgb)
        r2, g2, b2 = inverted.split()
        result = Image.merge("RGBA", (r2, g2, b2, a))
    elif img.mode in ("RGB", "L"):
        result = ImageOps.invert(img)
    else:
        result = ImageOps.invert(img.convert("RGB"))
    out, meta = _save_image(result)
    img.close()
    result.close()
    return json.dumps(meta)


def _action_text(image_path: str, text: str, x: int = 10, y: int = 10,
                 font_size: int = 24, color: str = "white") -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    font_size = max(1, min(font_size, 500))
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()
    draw.text((x, y), text, fill=color, font=font)
    result = Image.alpha_composite(img, overlay)
    out, meta = _save_image(result)
    meta["text"] = text[:50]
    img.close()
    result.close()
    return json.dumps(meta)


def _action_draw(image_path: str, shape: str = "rectangle",
                 x1: int = 0, y1: int = 0, x2: int = 100, y2: int = 100,
                 color: str = "red", width: int = 2, fill: str = "") -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    draw = ImageDraw.Draw(img)
    fill_color = fill if fill else None
    if shape == "rectangle":
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width, fill=fill_color)
    elif shape == "ellipse":
        draw.ellipse([x1, y1, x2, y2], outline=color, width=width, fill=fill_color)
    elif shape == "line":
        draw.line([x1, y1, x2, y2], fill=color, width=width)
    else:
        return json.dumps({"error": f"Unknown shape: {shape}. Use: rectangle, ellipse, line"})
    out, meta = _save_image(img)
    meta["shape"] = shape
    img.close()
    return json.dumps(meta)


def _action_paste(image_path: str, overlay_path: str, x: int = 0, y: int = 0,
                  opacity: float = 1.0) -> str:
    base_path = _validate_path(image_path)
    over_path = _validate_path(overlay_path)
    base = Image.open(base_path)
    overlay = Image.open(over_path)
    if opacity < 1.0:
        if overlay.mode != "RGBA":
            overlay = overlay.convert("RGBA")
        alpha = overlay.split()[3]
        alpha = alpha.point(lambda p: int(p * opacity))
        overlay.putalpha(alpha)
    if overlay.mode == "RGBA":
        base = base.convert("RGBA")
        base.paste(overlay, (x, y), overlay)
    else:
        base.paste(overlay, (x, y))
    out, meta = _save_image(base)
    meta["overlay"] = str(over_path)
    base.close()
    overlay.close()
    return json.dumps(meta)


def _action_merge(image_path: str, image_paths: str = "", direction: str = "horizontal") -> str:
    """Merge multiple images. image_paths is comma-separated."""
    all_paths = [image_path] + [p.strip() for p in image_paths.split(",") if p.strip()]
    if len(all_paths) < 2:
        return json.dumps({"error": "Need at least 2 images to merge"})
    if len(all_paths) > 20:
        return json.dumps({"error": "Maximum 20 images for merge"})
    _MAX_MERGE_PIXELS = 64 * 1024 * 1024  # 64M pixels max for merged canvas
    images = []
    for p in all_paths:
        path = _validate_path(p)
        images.append(Image.open(path))
    if direction == "vertical":
        max_w = max(img.width for img in images)
        total_h = sum(img.height for img in images)
        if max_w * total_h > _MAX_MERGE_PIXELS:
            for i in images: i.close()
            return json.dumps({"error": f"Merged canvas too large ({max_w}x{total_h}). Resize images first."})
        merged = Image.new("RGB", (max_w, total_h), (0, 0, 0))
        y_offset = 0
        for img in images:
            merged.paste(img, (0, y_offset))
            y_offset += img.height
    else:
        total_w = sum(img.width for img in images)
        max_h = max(img.height for img in images)
        if total_w * max_h > _MAX_MERGE_PIXELS:
            for i in images: i.close()
            return json.dumps({"error": f"Merged canvas too large ({total_w}x{max_h}). Resize images first."})
        merged = Image.new("RGB", (total_w, max_h), (0, 0, 0))
        x_offset = 0
        for img in images:
            merged.paste(img, (x_offset, 0))
            x_offset += img.width
    out, meta = _save_image(merged)
    meta["image_count"] = len(images)
    meta["direction"] = direction
    for img in images:
        img.close()
    merged.close()
    return json.dumps(meta)


def _action_colors(image_path: str, count: int = 5) -> str:
    """Extract dominant colors."""
    path = _validate_path(image_path)
    img = Image.open(path)
    # Resize for speed
    small = img.copy()
    small.thumbnail((150, 150))
    if small.mode != "RGB":
        small = small.convert("RGB")
    count = max(1, min(count, 20))
    quantized = small.quantize(colors=count, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()[:count * 3]
    colors = []
    for i in range(0, len(palette), 3):
        r, g, b = palette[i], palette[i + 1], palette[i + 2]
        colors.append({"rgb": [r, g, b], "hex": f"#{r:02x}{g:02x}{b:02x}"})
    img.close()
    small.close()
    return json.dumps({"dominant_colors": colors})


def _action_histogram(image_path: str) -> str:
    path = _validate_path(image_path)
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    hist = img.histogram()
    r_hist = hist[0:256]
    g_hist = hist[256:512]
    b_hist = hist[512:768]
    img.close()
    return json.dumps({
        "red": {"min": min(r_hist), "max": max(r_hist), "mean": round(sum(i * v for i, v in enumerate(r_hist)) / max(sum(r_hist), 1), 1)},
        "green": {"min": min(g_hist), "max": max(g_hist), "mean": round(sum(i * v for i, v in enumerate(g_hist)) / max(sum(g_hist), 1), 1)},
        "blue": {"min": min(b_hist), "max": max(b_hist), "mean": round(sum(i * v for i, v in enumerate(b_hist)) / max(sum(b_hist), 1), 1)},
    })


def _action_transparent(image_path: str, target_color: str = "white", tolerance: int = 30) -> str:
    """Make a specific color transparent."""
    path = _validate_path(image_path)
    img = Image.open(path)
    # Limit pixel count to prevent memory bomb (getdata materializes all pixels)
    max_pixels = 4096 * 4096
    if img.width * img.height > max_pixels:
        img.thumbnail((4096, 4096), Image.LANCZOS)
    img = img.convert("RGBA")
    # Parse target color
    color_map = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0),
                 "green": (0, 255, 0), "blue": (0, 0, 255)}
    if target_color.startswith("#") and len(target_color) == 7:
        tc = (int(target_color[1:3], 16), int(target_color[3:5], 16), int(target_color[5:7], 16))
    else:
        tc = color_map.get(target_color.lower(), (255, 255, 255))
    tolerance = max(0, min(tolerance, 255))
    data = img.getdata() if not hasattr(img, 'get_flattened_data') else img.get_flattened_data()
    new_data = []
    removed = 0
    for item in data:
        if (abs(item[0] - tc[0]) <= tolerance and
            abs(item[1] - tc[1]) <= tolerance and
            abs(item[2] - tc[2]) <= tolerance):
            new_data.append((item[0], item[1], item[2], 0))
            removed += 1
        else:
            new_data.append(item)
    img.putdata(new_data)
    out, meta = _save_image(img, fmt="PNG")
    meta["pixels_made_transparent"] = removed
    meta["total_pixels"] = len(data)
    meta["percent_transparent"] = round(removed / len(data) * 100, 1) if data else 0
    img.close()
    return json.dumps(meta)


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def image_process_tool(action: str, image_path: str = "", **kwargs) -> str:
    if not _PILLOW_AVAILABLE:
        return json.dumps({"error": "Pillow is not installed. Run: pip install hermes-agent[image]"})

    actions = {
        "info": lambda: _action_info(image_path),
        "resize": lambda: _action_resize(image_path, width=kwargs.get("width", 0), height=kwargs.get("height", 0), keep_aspect=kwargs.get("keep_aspect", True)),
        "convert": lambda: _action_convert(image_path, target_format=kwargs.get("format", "JPEG")),
        "compress": lambda: _action_compress(image_path, quality=kwargs.get("quality", 70)),
        "crop": lambda: _action_crop(image_path, left=kwargs.get("left", 0), top=kwargs.get("top", 0), right=kwargs.get("right", 0), bottom=kwargs.get("bottom", 0)),
        "rotate": lambda: _action_rotate(image_path, degrees=kwargs.get("degrees", 90)),
        "flip": lambda: _action_flip(image_path, direction=kwargs.get("direction", "horizontal")),
        "strip": lambda: _action_strip(image_path),
        "thumbnail": lambda: _action_thumbnail(image_path, size=kwargs.get("size", 256)),
        "blur": lambda: _action_blur(image_path, radius=kwargs.get("radius", 2.0)),
        "sharpen": lambda: _action_sharpen(image_path, factor=kwargs.get("factor", 2.0)),
        "filter": lambda: _action_filter(image_path, filter_name=kwargs.get("filter_name", "contour")),
        "brightness": lambda: _action_brightness(image_path, factor=kwargs.get("factor", 1.5)),
        "contrast": lambda: _action_contrast(image_path, factor=kwargs.get("factor", 1.5)),
        "saturation": lambda: _action_saturation(image_path, factor=kwargs.get("factor", 1.5)),
        "grayscale": lambda: _action_grayscale(image_path),
        "invert": lambda: _action_invert(image_path),
        "text": lambda: _action_text(image_path, text=kwargs.get("text", ""), x=kwargs.get("x", 10), y=kwargs.get("y", 10), font_size=kwargs.get("font_size", 24), color=kwargs.get("color", "white")),
        "draw": lambda: _action_draw(image_path, shape=kwargs.get("shape", "rectangle"), x1=kwargs.get("x1", 0), y1=kwargs.get("y1", 0), x2=kwargs.get("x2", 100), y2=kwargs.get("y2", 100), color=kwargs.get("color", "red"), width=kwargs.get("width", 2), fill=kwargs.get("fill", "")),
        "paste": lambda: _action_paste(image_path, overlay_path=kwargs.get("overlay_path", ""), x=kwargs.get("x", 0), y=kwargs.get("y", 0), opacity=kwargs.get("opacity", 1.0)),
        "merge": lambda: _action_merge(image_path, image_paths=kwargs.get("image_paths", ""), direction=kwargs.get("direction", "horizontal")),
        "colors": lambda: _action_colors(image_path, count=kwargs.get("count", 5)),
        "histogram": lambda: _action_histogram(image_path),
        "transparent": lambda: _action_transparent(image_path, target_color=kwargs.get("target_color", "white"), tolerance=kwargs.get("tolerance", 30)),
    }

    if action not in actions:
        return json.dumps({"error": f"Unknown action: {action}", "available_actions": sorted(actions.keys())})

    if not image_path:
        return json.dumps({"error": "image_path is required"})

    try:
        return actions[action]()
    except ValueError as e:
        return json.dumps({"error": str(e)})
    except Exception as e:
        logger.error("Image processing error: %s", e)
        return json.dumps({"error": f"Processing failed: {type(e).__name__}"})


# ---------------------------------------------------------------------------
# Tool schema and registration
# ---------------------------------------------------------------------------

def check_image_process_requirements() -> bool:
    return _PILLOW_AVAILABLE


IMAGE_PROCESS_SCHEMA = {
    "name": "image_process",
    "description": (
            "Full image processing toolkit. Actions: info (EXIF/metadata), resize, convert (format), "
            "compress, crop, rotate, flip, strip (metadata), thumbnail, blur, sharpen, filter "
            "(contour/edge/emboss/smooth/detail), brightness, contrast, saturation, grayscale, "
            "invert, text (overlay), draw (shapes), paste (overlay image), merge (combine images), "
            "colors (dominant), histogram, transparent (remove background color). "
            "Input: local file path. Output: processed image path + metadata."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["info", "resize", "convert", "compress", "crop", "rotate", "flip",
                             "strip", "thumbnail", "blur", "sharpen", "filter", "brightness",
                             "contrast", "saturation", "grayscale", "invert", "text", "draw",
                             "paste", "merge", "colors", "histogram", "transparent"],
                    "description": "Action to perform",
                },
                "image_path": {"type": "string", "description": "Path to the image file"},
                "width": {"type": "integer", "description": "Target width (resize)"},
                "height": {"type": "integer", "description": "Target height (resize)"},
                "keep_aspect": {"type": "boolean", "description": "Maintain aspect ratio (resize, default true)"},
                "format": {"type": "string", "description": "Target format (convert: JPEG, PNG, WEBP, GIF, BMP, TIFF)"},
                "quality": {"type": "integer", "description": "Quality 1-100 (compress, default 70)"},
                "left": {"type": "integer", "description": "Left bound (crop)"},
                "top": {"type": "integer", "description": "Top bound (crop)"},
                "right": {"type": "integer", "description": "Right bound (crop)"},
                "bottom": {"type": "integer", "description": "Bottom bound (crop)"},
                "degrees": {"type": "number", "description": "Rotation degrees (rotate)"},
                "direction": {"type": "string", "description": "horizontal or vertical (flip/merge)"},
                "size": {"type": "integer", "description": "Max dimension (thumbnail, default 256)"},
                "radius": {"type": "number", "description": "Blur radius (blur, default 2.0)"},
                "factor": {"type": "number", "description": "Enhancement factor (sharpen/brightness/contrast/saturation)"},
                "filter_name": {"type": "string", "description": "Filter name (filter: contour/edge/emboss/smooth/detail/edge_enhance)"},
                "text": {"type": "string", "description": "Text to add (text)"},
                "x": {"type": "integer", "description": "X position (text/paste)"},
                "y": {"type": "integer", "description": "Y position (text/paste)"},
                "font_size": {"type": "integer", "description": "Font size (text, default 24)"},
                "color": {"type": "string", "description": "Color name or hex (text/draw)"},
                "shape": {"type": "string", "description": "Shape (draw: rectangle/ellipse/line)"},
                "x1": {"type": "integer", "description": "Start X (draw)"},
                "y1": {"type": "integer", "description": "Start Y (draw)"},
                "x2": {"type": "integer", "description": "End X (draw)"},
                "y2": {"type": "integer", "description": "End Y (draw)"},
                "fill": {"type": "string", "description": "Fill color (draw, empty=no fill)"},
                "overlay_path": {"type": "string", "description": "Overlay image path (paste)"},
                "opacity": {"type": "number", "description": "Overlay opacity 0-1 (paste, default 1.0)"},
                "image_paths": {"type": "string", "description": "Comma-separated paths (merge)"},
                "target_color": {"type": "string", "description": "Color to make transparent (transparent, default white)"},
                "tolerance": {"type": "integer", "description": "Color match tolerance 0-255 (transparent, default 30)"},
                "count": {"type": "integer", "description": "Number of colors (colors, default 5)"},
            },
            "required": ["action", "image_path"],
        },
}

registry.register(
    name="image_process",
    toolset="vision",
    schema=IMAGE_PROCESS_SCHEMA,
    handler=lambda args, **kw: image_process_tool(
        action=args.get("action", ""),
        image_path=args.get("image_path", ""),
        **{k: v for k, v in args.items() if k not in ("action", "image_path")},
    ),
    check_fn=check_image_process_requirements,
    emoji="🖼️",
)
