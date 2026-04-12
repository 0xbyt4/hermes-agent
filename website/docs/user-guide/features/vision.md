---
title: Vision & Image Paste
description: Paste images from your clipboard into the Hermes CLI for multimodal vision analysis.
sidebar_label: Vision & Image Paste
sidebar_position: 7
---

# Vision & Image Paste

Hermes Agent supports **multimodal vision** — you can paste images from your clipboard directly into the CLI and ask the agent to analyze, describe, or work with them. When the active model has native vision support, images are sent as base64-encoded content blocks so the model receives the actual pixels. Models without native vision automatically fall back to a text-description path that uses an auxiliary vision model to summarize each image.

## How It Works

1. Copy an image to your clipboard (screenshot, browser image, etc.)
2. Attach it using one of the methods below
3. Type your question and press Enter
4. The image appears as a `[📎 Image #1]` badge above the input
5. On submit, Hermes checks the active model's capability:
   - **Native vision** (Claude 3+, Opus 4.6, Sonnet 4.6, GPT-5.4, Gemini 2.5/3, Grok, Qwen-VL, Llava, etc.) → image sent as a typed content block
   - **No native vision** → image goes through `vision_analyze` (auxiliary vision model produces a text description that gets prepended to your message)

You can attach multiple images before sending — each gets its own badge. Press `Ctrl+C` to clear all attached images.

Images are saved to `~/.hermes/images/` as PNG files with timestamped filenames.

## Messaging Platforms (Telegram, Discord, Matrix, etc.)

The same capability-aware routing applies when users send images to Hermes via messaging gateways:

- Send a photo + caption to your Hermes bot on Telegram, Discord, Matrix, Signal, Feishu, Slack, WhatsApp, or any other supported platform
- The gateway detects the active model for that session and either passes the image natively or falls back to the legacy text-description path
- Image-bearing messages are persisted with their full multimodal structure so session reload preserves the visual context

This means **webdev workflows work end-to-end through messaging**: drop a screenshot of a broken UI into Telegram, ask "what's wrong with this layout?", and Claude/GPT/Gemini sees the actual pixels — not a lossy text summary.

## Self-Hosted & Uncatalogued Vision Models

Capability detection uses [models.dev](https://models.dev) data. If you're running a self-hosted vision model (vLLM serving Llama 3.2 Vision, a custom multimodal model not yet in the catalog, etc.), set the env var to force native vision routing:

```bash
export HERMES_FORCE_NATIVE_VISION=1
```

When unset, Hermes auto-detects native support per model. When set to `1`, the gateway and CLI both treat the active model as vision-capable regardless of catalog state.

## Paste Methods

How you attach an image depends on your terminal environment. Not all methods work everywhere — here's the full breakdown:

### `/paste` Command

**The most reliable method. Works everywhere.**

```
/paste
```

Type `/paste` and press Enter. Hermes checks your clipboard for an image and attaches it. This works in every environment because it explicitly calls the clipboard backend — no terminal keybinding interception to worry about.

### Ctrl+V / Cmd+V (Bracketed Paste)

When you paste text that's on the clipboard alongside an image, Hermes automatically checks for an image too. This works when:
- Your clipboard contains **both text and an image** (some apps put both on the clipboard when you copy)
- Your terminal supports bracketed paste (most modern terminals do)

:::warning
If your clipboard has **only an image** (no text), Ctrl+V does nothing in most terminals. Terminals can only paste text — there's no standard mechanism to paste binary image data. Use `/paste` or Alt+V instead.
:::

### Alt+V

Alt key combinations pass through most terminal emulators (they're sent as ESC + key rather than being intercepted). Press `Alt+V` to check the clipboard for an image.

:::caution
**Does not work in VSCode's integrated terminal.** VSCode intercepts many Alt+key combos for its own UI. Use `/paste` instead.
:::

### Ctrl+V (Raw — Linux Only)

On Linux desktop terminals (GNOME Terminal, Konsole, Alacritty, etc.), `Ctrl+V` is **not** the paste shortcut — `Ctrl+Shift+V` is. So `Ctrl+V` sends a raw byte to the application, and Hermes catches it to check the clipboard. This only works on Linux desktop terminals with X11 or Wayland clipboard access.

## Platform Compatibility

| Environment | `/paste` | Ctrl+V text+image | Alt+V | Notes |
|---|:---:|:---:|:---:|---|
| **macOS Terminal / iTerm2** | ✅ | ✅ | ✅ | Best experience — `osascript` always available |
| **Linux X11 desktop** | ✅ | ✅ | ✅ | Requires `xclip` (`apt install xclip`) |
| **Linux Wayland desktop** | ✅ | ✅ | ✅ | Requires `wl-paste` (`apt install wl-clipboard`) |
| **WSL2 (Windows Terminal)** | ✅ | ✅¹ | ✅ | Uses `powershell.exe` — no extra install needed |
| **VSCode Terminal (local)** | ✅ | ✅¹ | ❌ | VSCode intercepts Alt+key |
| **VSCode Terminal (SSH)** | ❌² | ❌² | ❌ | Remote clipboard not accessible |
| **SSH terminal (any)** | ❌² | ❌² | ❌² | Remote clipboard not accessible |

¹ Only when clipboard has both text and an image (image-only clipboard = nothing happens)
² See [SSH & Remote Sessions](#ssh--remote-sessions) below

## Platform-Specific Setup

### macOS

**No setup required.** Hermes uses `osascript` (built into macOS) to read the clipboard. For faster performance, optionally install `pngpaste`:

```bash
brew install pngpaste
```

### Linux (X11)

Install `xclip`:

```bash
# Ubuntu/Debian
sudo apt install xclip

# Fedora
sudo dnf install xclip

# Arch
sudo pacman -S xclip
```

### Linux (Wayland)

Modern Linux desktops (Ubuntu 22.04+, Fedora 34+) often use Wayland by default. Install `wl-clipboard`:

```bash
# Ubuntu/Debian
sudo apt install wl-clipboard

# Fedora
sudo dnf install wl-clipboard

# Arch
sudo pacman -S wl-clipboard
```

:::tip How to check if you're on Wayland
```bash
echo $XDG_SESSION_TYPE
# "wayland" = Wayland, "x11" = X11, "tty" = no display server
```
:::

### WSL2

**No extra setup required.** Hermes detects WSL2 automatically (via `/proc/version`) and uses `powershell.exe` to access the Windows clipboard through .NET's `System.Windows.Forms.Clipboard`. This is built into WSL2's Windows interop — `powershell.exe` is available by default.

The clipboard data is transferred as base64-encoded PNG over stdout, so no file path conversion or temp files are needed.

:::info WSLg Note
If you're running WSLg (WSL2 with GUI support), Hermes tries the PowerShell path first, then falls back to `wl-paste`. WSLg's clipboard bridge only supports BMP format for images — Hermes auto-converts BMP to PNG using Pillow (if installed) or ImageMagick's `convert` command.
:::

#### Verify WSL2 clipboard access

```bash
# 1. Check WSL detection
grep -i microsoft /proc/version

# 2. Check PowerShell is accessible
which powershell.exe

# 3. Copy an image, then check
powershell.exe -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.Clipboard]::ContainsImage()"
# Should print "True"
```

## SSH & Remote Sessions

**Clipboard paste does not work over SSH.** When you SSH into a remote machine, the Hermes CLI runs on the remote host. All clipboard tools (`xclip`, `wl-paste`, `powershell.exe`, `osascript`) read the clipboard of the machine they run on — which is the remote server, not your local machine. Your local clipboard is inaccessible from the remote side.

### Workarounds for SSH

1. **Upload the image file** — Save the image locally, upload it to the remote server via `scp`, VSCode's file explorer (drag-and-drop), or any file transfer method. Then reference it by path. *(A `/attach <filepath>` command is planned for a future release.)*

2. **Use a URL** — If the image is accessible online, just paste the URL in your message. The agent can use `vision_analyze` to look at any image URL directly.

3. **X11 forwarding** — Connect with `ssh -X` to forward X11. This lets `xclip` on the remote machine access your local X11 clipboard. Requires an X server running locally (XQuartz on macOS, built-in on Linux X11 desktops). Slow for large images.

4. **Use a messaging platform** — Send images to Hermes via Telegram, Discord, Slack, or WhatsApp. These platforms handle image upload natively and are not affected by clipboard/terminal limitations.

## Why Terminals Can't Paste Images

This is a common source of confusion, so here's the technical explanation:

Terminals are **text-based** interfaces. When you press Ctrl+V (or Cmd+V), the terminal emulator:

1. Reads the clipboard for **text content**
2. Wraps it in [bracketed paste](https://en.wikipedia.org/wiki/Bracketed-paste) escape sequences
3. Sends it to the application through the terminal's text stream

If the clipboard contains only an image (no text), the terminal has nothing to send. There is no standard terminal escape sequence for binary image data. The terminal simply does nothing.

This is why Hermes uses a separate clipboard check — instead of receiving image data through the terminal paste event, it calls OS-level tools (`osascript`, `powershell.exe`, `xclip`, `wl-paste`) directly via subprocess to read the clipboard independently.

## Supported Models

Image input works with **any model** — Hermes routes the image differently based on the model's capability:

### Native vision models (preferred path)

Image is sent as an OpenAI-style `image_url` content block; the provider adapter converts to the right native format (Anthropic image blocks, OpenAI image_url, Codex input_image, etc.):

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:image/png;base64,..."
  }
}
```

Confirmed native vision support: Claude Opus 4.6, Sonnet 4.6, Haiku 4.5; GPT-5, GPT-5.4, GPT-4o; Gemini 2.5, Gemini 3 Flash; Grok 4; Qwen-VL; Llava and other open-source multimodal models served through OpenRouter or self-hosted endpoints.

### Non-vision models (fallback path)

For text-only models (older GPTs without vision, smaller open-source models, etc.), Hermes routes the image through the auxiliary vision model (default: Gemini Flash) to produce a text description. The description is prepended to your message so the main model can answer questions about the image, just less accurately than native vision and with the cost/latency of an extra LLM call.

You can override the fallback vision model via `AUXILIARY_VISION_PROVIDER` / `AUXILIARY_VISION_MODEL` (see [environment variables](/docs/reference/environment-variables)).
