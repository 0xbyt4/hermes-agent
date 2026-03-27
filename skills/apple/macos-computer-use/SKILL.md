---
name: macos-computer-use
description: Guide for using the computer_use tool effectively on macOS — app switching, keyboard shortcuts, typing, clicking, scrolling, drag-and-drop, and reliable interaction patterns for CLI and gateway modes.
version: 2.0.0
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

Control a macOS desktop via the `computer` tool — screenshots, mouse, keyboard, scrolling, drag-and-drop. This tool uses Anthropic's Computer Use API.

## Golden Rules

1. **Screenshot first** — always see the screen before acting
2. **Screenshot after** — verify every action worked
3. **Never assume focus** — verify which app is active before typing
4. **Keyboard shortcuts over clicks** — shortcuts are 100% reliable, clicks can miss by a few pixels
5. **MEDIA tag for gateway** — extract the `MEDIA:/tmp/hermes_screenshot_<id>.jpg` path from the screenshot result's `text_summary` and include it in your response
6. **Terminal as fallback** — `osascript`, `open`, `pbcopy`/`pbpaste` are always available when GUI fails

## DO NOT (Safety)

- DO NOT type passwords or secrets — tell the user to handle login dialogs
- DO NOT close windows without checking for unsaved work
- DO NOT interact with System Settings > Privacy/Security sections autonomously
- DO NOT lock the screen (`command+control+q`) — you lose all control
- DO NOT click "Allow" on permission dialogs — the user must do this
- DO NOT use `command+shift+4` (interactive screenshot) — it blocks execution
- DO NOT run destructive terminal commands (`rm -rf`, `sudo`) without user approval

## CLI Mode vs Gateway Mode

**CLI mode**: Terminal running Hermes has focus. After using terminal tool (osascript, open), Terminal takes focus back. If you then `type`, text goes to Terminal, not target app. **Workaround**: after every terminal command, re-activate the target app with osascript and verify with screenshot.

**Gateway mode** (Telegram/Discord): Agent runs in background, no terminal window steals focus. This is the reliable mode for multi-step GUI workflows. Always extract the `MEDIA:` path from the screenshot result's `text_summary` and include it in your response so the user sees screenshots.

## App Switching & Focus

**CRITICAL**: The `type` action types into whatever app is currently focused.

### Methods (best to worst):

| Method | Command | Reliability |
|--------|---------|-------------|
| osascript (terminal) | `osascript -e 'tell application "AppName" to activate'` | Best |
| open command (terminal) | `open -a "Google Chrome"` | Great |
| Cmd+Tab | `key: command+Tab` | Good (cycles, unpredictable order) |
| Click on window | `left_click` on visible window area | OK (need correct coordinates) |
| Click dock icon | `left_click` at bottom of screen | Tricky (small targets) |

### Recommended pattern:
1. Terminal: `osascript -e 'tell application "Google Chrome" to activate'`
2. `computer action=wait, duration=0.5`
3. `computer action=screenshot` — confirm correct app is focused
4. Now safe to type/click in that app

## Keyboard Shortcuts

100% reliable — always prefer over clicking.

### System
| Action | Shortcut |
|--------|----------|
| Spotlight search | `command+space` |
| Switch app | `command+Tab` |
| Close window | `command+w` |
| Quit app | `command+q` |
| Minimize | `command+m` |
| Full screen | `command+control+f` |
| Force quit menu | `command+option+Escape` |
| Undo | `command+z` |
| Redo | `command+shift+z` |

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
| Top of page | `command+Up` |
| Bottom of page | `command+Down` |

### Text editing
| Action | Shortcut |
|--------|----------|
| Select all | `command+a` |
| Copy | `command+c` |
| Paste | `command+v` |
| Cut | `command+x` |
| Select word | `option+shift+Right` |
| Select line | `command+shift+Right` |
| Delete word | `option+Delete` |

## Typing Text

The `type` action uses clipboard paste (`Cmd+V`) — works with ALL keyboard layouts and Unicode.

**WARNING**: Type action overwrites the user's clipboard. If you need to preserve clipboard content, read it first with `pbpaste` via terminal, then restore after typing.

### Pattern:
1. Ensure target field is focused (click or keyboard navigation)
2. `computer action=screenshot` — verify cursor is in the field
3. `computer action=type, text=your text here`
4. `computer action=screenshot` — verify text was entered

### For browser address bar:
1. Focus browser: `osascript -e 'tell application "Google Chrome" to activate'`
2. `computer action=key, key=command+l` — focus address bar
3. `computer action=type, text=https://example.com`
4. `computer action=key, key=Return`
5. `computer action=wait, duration=2`
6. `computer action=screenshot`

## Wait Action

Use `computer action=wait, duration=N` (max 10 seconds per call) for:
- App launch: 0.5-2s
- Page load: 1-3s
- Dialog appearance: 0.5-1s
- For longer waits: chain multiple waits with screenshot checks

## Scrolling

The `scroll` action may fail in some apps. Reliable alternatives:

| Method | When to use |
|--------|-------------|
| `key: space` | Scroll down in browser |
| `key: shift+space` | Scroll up in browser |
| `key: Page_Down` | Scroll down (most apps) |
| `key: Page_Up` | Scroll up (most apps) |
| `key: command+Up` | Top of page/document |
| `key: command+Down` | Bottom of page/document |
| `key: Down` | Small scroll (send multiple separate actions) |

**Note**: Each key press must be a separate `computer action=key` call. Do not combine like `Down Down Down`.

## Clicking

Coordinates are in logical screen resolution (e.g., 1470x956). The tool auto-scales from screenshot space.

### Tips:
- **Aim for center** of buttons/icons — edge clicks may miss
- **Dock**: icons are at y > 930 (on 956px screen height)
- **Menu bar**: y = 0 to 25
- **After a miss**: take screenshot, identify correct position, then click updated coordinate. Do NOT retry same coordinate.
- **`mouse_move` first** to verify position visually, then `left_click`
- **`double_click`** for opening files in Finder
- **`triple_click`** to select entire line/paragraph in text editors

## Drag and Drop

The tool supports `left_click_drag` with `start_coordinate` and `end_coordinate`.

```
computer action=left_click_drag, start_coordinate=[100, 200], end_coordinate=[400, 300]
```

Use cases:
- Move files in Finder: drag from file icon to target folder
- Move windows: drag from title bar
- Resize windows: drag from window edges (prefer keyboard/osascript alternatives)

**Alternative for file operations**: `mv`, `cp` via terminal is more reliable than drag.

## Reading Screen Content

- The agent reads text directly from screenshots via vision
- For large text, use `command+a, command+c` then `pbpaste` via terminal
- For web pages: `command+a, command+c` selects all page text
- For Finder: `osascript -e 'tell application "Finder" to get selection'` returns selected files

## Opening URLs

**Best method** — use terminal:
```
Terminal: open "https://example.com"
```
Or target specific browser:
```
Terminal: osascript -e 'tell application "Google Chrome" to open location "https://example.com"'
```
Then wait 2s and screenshot.

If Chrome is not running, `open location` launches it first (add extra wait time).

## Common App Names

| App | osascript name |
|-----|---------------|
| Chrome | "Google Chrome" |
| Firefox | "Firefox" |
| Safari | "Safari" |
| Finder | "Finder" |
| Terminal | "Terminal" |
| VS Code | "Visual Studio Code" |
| Discord | "Discord" |
| Telegram | "Telegram" |
| Slack | "Slack" |
| Notes | "Notes" |
| Messages | "Messages" |
| TextEdit | "TextEdit" |
| Preview | "Preview" |
| Calendar | "Calendar" |
| System Settings | "System Settings" |
| Activity Monitor | "Activity Monitor" |

## MEDIA: Gateway Screenshot Delivery

When the user requests a screenshot via gateway (Telegram/Discord):

1. `computer action=screenshot` returns `text_summary` containing a `MEDIA:/tmp/hermes_screenshot_<id>.jpg` path (unique per capture)
2. Extract the exact `MEDIA:` path from `text_summary` and include it in your response text
3. The gateway extracts this path and sends the image file to the chat
4. If you omit the MEDIA tag, the user sees no image
5. Each screenshot creates a new file with a unique ID — old files are cleaned up automatically

Example response: "Here's your screenshot MEDIA:/tmp/hermes_screenshot_a1b2c3d4.jpg — I can see Chrome open with X/Twitter."

## Notification and Dialog Handling

- System notifications appear top-right — wait 3-5s for auto-dismiss
- Permission dialogs ("App wants to access...") block interaction — tell the user to handle them
- "Save changes?" dialogs: `Return` to save, `command+d` for don't save, `Escape` to cancel
- Spotlight sometimes activates unexpectedly — press `Escape` to dismiss

## Error Recovery

1. `key: Escape` — close dialogs, cancel operations
2. `command+z` — undo last action
3. `command+w` — close current window/tab
4. `computer action=screenshot` — always check what happened
5. Terminal fallback: `osascript`, `open`, `pbcopy`/`pbpaste`
6. App not responding: `command+option+Escape` opens Force Quit, or `osascript -e 'tell application "AppName" to quit'`
7. For stuck states: take screenshot to diagnose, then decide next action

## Accessibility Permissions

The computer tool requires macOS permissions:
- **Screen Recording**: System Settings > Privacy & Security > Screen Recording — add Terminal/iTerm
- **Accessibility**: System Settings > Privacy & Security > Accessibility — add Terminal/iTerm
- Symptom of missing permission: screenshot returns empty or click/type fails silently
- After granting permission, Terminal must be **fully restarted** (not just new tab)
- For gateway: the Python process itself needs these permissions

## Limitations

- Cannot see content off-screen (must scroll)
- Cannot interact behind overlapping windows (must bring target to front)
- Scroll action unreliable in some apps (use keyboard alternatives)
- Wait capped at 10 seconds per call (chain for longer waits)
- Screenshots capture primary display only (multi-monitor: secondary displays invisible)
- Type action overwrites clipboard
- Cannot handle macOS full-screen Spaces/Mission Control
- Coordinate accuracy ~5-10px — small UI targets may need retry
- Cannot detect Touch Bar interactions

## Workflow Examples

### Open a website and search:
```
1. Terminal: osascript -e 'tell application "Google Chrome" to activate'
2. computer action=wait, duration=0.5
3. computer action=screenshot — verify Chrome active
4. computer action=key, key=command+l — focus address bar
5. computer action=type, text=https://x.com
6. computer action=key, key=Return
7. computer action=wait, duration=2
8. computer action=screenshot — verify page loaded
```

### Create and save a text file:
```
1. Terminal: open -a TextEdit
2. computer action=wait, duration=1
3. computer action=screenshot — verify TextEdit open
4. computer action=type, text=Hello World
5. computer action=key, key=command+s — save dialog
6. computer action=wait, duration=0.5
7. computer action=screenshot — verify dialog
8. computer action=type, text=myfile.txt
9. computer action=key, key=Return
```

### Find and open a file:
```
1. Terminal: osascript -e 'tell application "Finder" to activate'
2. computer action=wait, duration=0.5
3. computer action=key, key=command+shift+g — Go to Folder
4. computer action=type, text=/Users/username/Documents
5. computer action=key, key=Return
6. computer action=wait, duration=0.5
7. computer action=screenshot — see folder contents
8. computer action=double_click on target file
```
