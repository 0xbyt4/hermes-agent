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

## Clicking — Hover-Verify-Click Pattern

**CRITICAL**: Never blind-click. Always verify cursor position first.

The screenshot includes the mouse cursor and reports `Cursor at (x, y)` in the result. Use this to navigate precisely.

### Reliable click pattern (ALWAYS use this):
```
1. screenshot — see the screen, note where cursor is
2. mouse_move to target — move cursor to the element you want to click
3. screenshot — VERIFY cursor is on the correct element
4. left_click (no coordinate) — click at current cursor position
5. screenshot — verify the click had the expected effect
```

### Why this works:
- `mouse_move` is 100% accurate — cursor goes exactly where you say
- Screenshot shows cursor visually — you can SEE if it's on the right element
- `left_click` without coordinates clicks at current position — no guessing
- If cursor is wrong, adjust with another `mouse_move` before clicking

### DO NOT:
- Do NOT guess coordinates and click directly — you will miss small targets
- Do NOT retry the same coordinate after a miss — take screenshot and adjust
- Do NOT combine mouse_move + click in one step — always verify between them

### Coordinate reference:
- **Dock icons**: y > 930 (on 956px screen)
- **Menu bar**: y = 0 to 25
- **Aim for center** of buttons/icons — never edges
- **`double_click`** for opening files in Finder
- **`triple_click`** to select entire line/paragraph

### Context menus (right-click):
```
1. mouse_move to target element
2. screenshot — verify position
3. right_click — opens context menu
4. screenshot — see menu options
5. mouse_move to menu item
6. screenshot — verify on correct item
7. left_click — select menu item
```

### Focus management before clicking:
- Before clicking in an app window, make sure that app is FRONTMOST
- Use `osascript -e 'tell application "AppName" to activate'` first
- Or click on an empty area of the target window first to bring it to front
- Then use hover-verify-click on the specific element

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

### Click a specific UI element (hover-verify-click):
```
1. screenshot — see screen, note cursor position
2. mouse_move to [x, y] — move cursor to target button/icon
3. screenshot — VERIFY cursor is on the correct element
4. left_click — click at current position (no coordinates!)
5. screenshot — verify click had expected effect
```

### Create a new folder in Finder (GUI):
```
1. osascript -e 'tell application "Finder" to activate'
2. wait 0.5s
3. screenshot — verify Finder is frontmost
4. mouse_move to empty area in Finder window
5. screenshot — verify cursor in window
6. right_click — open context menu
7. screenshot — see menu
8. mouse_move to "New Folder" menu item
9. screenshot — verify on correct item
10. left_click — creates folder with editable name
11. type: MyNewFolder
12. key: Return — confirm name
13. screenshot — verify folder created
```

### Open a website:
```
1. osascript -e 'tell application "Google Chrome" to activate'
2. wait 0.5s
3. screenshot — verify Chrome active
4. key: command+l — focus address bar
5. type: https://x.com
6. key: Return
7. wait 2s
8. screenshot — verify page loaded
```

### Click a link on a webpage:
```
1. screenshot — see the page
2. mouse_move to the link text/button
3. screenshot — verify cursor is on the link
4. left_click — click the link
5. wait 1s
6. screenshot — verify navigation
```

### Fill a form field:
```
1. screenshot — see the form
2. mouse_move to the input field
3. screenshot — verify cursor on field
4. left_click — focus the field
5. screenshot — verify cursor blinking in field
6. type: field value
7. key: Tab — move to next field
8. screenshot — verify text entered
```

### Create and save a text file:
```
1. Terminal: open -a TextEdit
2. wait 1s
3. screenshot — verify TextEdit open
4. type: Hello World
5. key: command+s — save dialog
6. wait 0.5s
7. screenshot — verify dialog
8. type: myfile.txt
9. key: Return
```

### Drag a file:
```
1. screenshot — see files
2. mouse_move to source file icon
3. screenshot — verify on file
4. left_click_drag with start_coordinate and end_coordinate to target folder
5. screenshot — verify file moved
```
