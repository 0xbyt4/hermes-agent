---
name: macos-computer-use
description: Guide for using the computer_use tool effectively on macOS — app switching, dock navigation, keyboard shortcuts, typing, and reliable interaction patterns.
version: 1.0.0
author: 0xbyt4
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [computer-use, macos, desktop, automation, screenshots, mouse, keyboard]
    category: apple
    requires_toolsets: [computer_use]
---

# macOS Computer Use Guide

Guide for controlling a macOS desktop via the `computer` tool. This tool uses Anthropic's Computer Use API to take screenshots, click, type, and use keyboard shortcuts.

## Golden Rules

1. **Always screenshot first** before clicking anything — you need to see the current screen state
2. **After every click, take another screenshot** to verify what happened
3. **Never assume focus** — always verify which app is active before typing
4. **Prefer keyboard shortcuts** over clicking small UI elements — they are 100% reliable
5. **When sharing screenshots with users, include `MEDIA:<path>` from the result text in your response**

## App Switching & Focus

**CRITICAL**: The computer tool types into whatever app is currently focused. Always ensure the correct app has focus before typing.

### Reliable app switching methods (best to worst):

| Method | Command | Reliability |
|--------|---------|-------------|
| **osascript (terminal)** | `osascript -e 'tell application "AppName" to activate'` | Best |
| **open command (terminal)** | `open -a "Google Chrome"` | Great |
| **Cmd+Tab** | `key: command+Tab` | Good (cycles, unpredictable order) |
| **Click on app window** | `left_click` on visible window | OK (need accurate coordinates) |
| **Click dock icon** | `left_click` at bottom of screen | Tricky (small targets) |

### Recommended pattern for switching to an app:

```
1. Use terminal: osascript -e 'tell application "Google Chrome" to activate'
2. Wait 0.5s: computer action=wait, duration=0.5
3. Take screenshot to confirm: computer action=screenshot
4. Now you can type/click in that app
```

## Keyboard Shortcuts (macOS)

These are 100% reliable — prefer them over clicking:

### System
| Action | Shortcut |
|--------|----------|
| Spotlight search | `command+space` |
| Open Finder | `open -a Finder` via terminal |
| Force quit menu | `command+option+Escape` |
| Screenshot (system) | `command+shift+3` |
| Lock screen | `command+control+q` |
| Switch app | `command+Tab` |
| Close window | `command+w` |
| Quit app | `command+q` |
| Minimize | `command+m` |
| Full screen | `command+control+f` |

### Browser (Chrome/Firefox/Safari)
| Action | Shortcut |
|--------|----------|
| Address bar | `command+l` |
| New tab | `command+t` |
| Close tab | `command+w` |
| Refresh | `command+r` |
| Back | `command+[` |
| Forward | `command+]` |
| Find | `command+f` |
| Bookmark | `command+d` |

### Text editing
| Action | Shortcut |
|--------|----------|
| Select all | `command+a` |
| Copy | `command+c` |
| Paste | `command+v` |
| Cut | `command+x` |
| Undo | `command+z` |
| Redo | `command+shift+z` |

## Typing Text

The `type` action uses clipboard paste (Cmd+V) — works with ALL keyboard layouts and Unicode.

### Reliable typing pattern:
```
1. Click on the text field (or use keyboard to focus it)
2. Take screenshot to verify cursor is in the field
3. Use type action with your text
4. Take screenshot to verify text was entered
```

### For browser address bar:
```
1. Focus browser: osascript via terminal, or click on browser window
2. Cmd+L to focus address bar: computer action=key, key=command+l
3. Type URL: computer action=type, text=https://example.com
4. Press Enter: computer action=key, key=Return
```

## Scrolling

The `scroll` action may not work in all contexts. Reliable alternatives:

| Method | When to use |
|--------|-------------|
| `scroll` action | General scrolling (may fail) |
| `key: space` | Scroll down in browser pages |
| `key: shift+space` | Scroll up in browser pages |
| `key: Page_Down` | Scroll down (any app) |
| `key: Page_Up` | Scroll up (any app) |
| `key: Home` | Scroll to top |
| `key: End` | Scroll to bottom |
| `key: Down Down Down` | Arrow keys for small scrolls |

## Clicking Accuracy

The coordinate system is the logical screen resolution (e.g., 1470x956). Tips for accurate clicking:

1. **Dock icons are at the very bottom** (y > 930 on a 956px screen)
2. **Menu bar is at y=0 to y=25**
3. **Small icons need precise coordinates** — if you miss, take a screenshot and adjust
4. **Move mouse first** with `mouse_move` to verify position, then `left_click`
5. **Double-click** for opening files in Finder

## Opening URLs

**Best method** — use terminal, not browser clicks:
```
1. Terminal: open "https://example.com" (opens in default browser)
   OR
   Terminal: osascript -e 'tell application "Google Chrome" to open location "https://example.com"'
2. Wait: computer action=wait, duration=2
3. Screenshot: computer action=screenshot
```

## Common macOS App Names (for osascript/open)

| App | Name for osascript |
|-----|-------------------|
| Chrome | "Google Chrome" |
| Firefox | "Firefox" |
| Safari | "Safari" |
| Finder | "Finder" |
| Terminal | "Terminal" |
| VS Code | "Visual Studio Code" |
| Spotify | "Spotify" |
| Discord | "Discord" |
| Telegram | "Telegram" |
| Slack | "Slack" |
| Notes | "Notes" |
| Messages | "Messages" |

## Error Recovery

If something goes wrong:
1. **Escape key** (`key: Escape`) — closes dialogs, cancels operations
2. **Cmd+Z** — undo last action
3. **Cmd+W** — close current window/tab
4. **Take screenshot** — always check what happened
5. **Use terminal as fallback** — `open`, `osascript`, `pbcopy`/`pbpaste` are always available

## Gateway Usage (Telegram/Discord)

When running via gateway, the agent controls the host machine's desktop remotely. The user sees screenshots sent as images. Always include `MEDIA:<path>` from the screenshot result to deliver images to the user.
