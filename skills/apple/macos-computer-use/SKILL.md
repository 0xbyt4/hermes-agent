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

100% reliable — always prefer over clicking. But shortcuts only work when the correct app is focused and no overlay (dialog, menu, Spotlight) is blocking input.

### Pre-shortcut Checklist (MUST follow)

**Before ANY keyboard shortcut:**
1. `key: Escape` — dismiss any open menu, dialog, Spotlight, or overlay
2. `screenshot` — verify the correct app is frontmost and no overlay is blocking
3. Only THEN press the shortcut
4. `screenshot` — verify the shortcut worked

**If a shortcut does nothing:**
1. `key: Escape` — normalize state
2. `screenshot` — check what's on screen
3. Verify the correct app is in focus (check title bar, menu bar app name)
4. If wrong app: `osascript -e 'tell application "AppName" to activate'` + `wait 0.5`
5. Retry the shortcut
6. If still fails after 2 attempts: use terminal/osascript fallback, do NOT keep retrying the same shortcut

**DO NOT:**
- Press shortcuts without verifying focus first
- Retry the same shortcut more than 2 times — switch to terminal fallback
- Use non-standard shortcuts (e.g. `super`, `cmd+F3`) — stick to the list below
- Press `cmd+space` and then click elsewhere — use `Escape` to dismiss Spotlight first

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
| Screenshot | `command+shift+3` |
| Screenshot selection | `command+shift+4` (interactive — avoid) |
| Screenshot/record panel | `command+shift+5` |
| Lock screen | `command+control+q` (avoid — loses control) |

### macOS Tahoe 26 — Fn/Globe Key Shortcuts
| Action | Shortcut |
|--------|----------|
| Show Desktop | `fn+h` |
| Show/Hide Dock | `fn+a` |
| Show/Hide Apps (Launchpad) | `fn+shift+a` |
| Control Center | `fn+c` |
| Notification Center | `fn+n` |
| Start/Stop Dictation | `fn+d` |
| Emoji/Character Viewer | `fn+e` |
| Quick Note | `fn+q` |

### Mission Control & Spaces
| Action | Shortcut |
|--------|----------|
| Mission Control | `control+Up` |
| Application Windows | `control+Down` |
| Show Desktop (alt) | `fn+f11` |
| Move to Left Space | `control+Left` |
| Move to Right Space | `control+Right` |

**IMPORTANT**: Do NOT use `cmd+F3`, `super+F3`, `F11` alone, or `super` key — these are either media keys or invalid key names. Use the shortcuts listed above.

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

### Finder
| Action | Shortcut |
|--------|----------|
| New Finder window | `command+n` |
| New folder | `command+shift+n` |
| Rename (selected file) | `Return` (enters rename mode) |
| Get info | `command+i` |
| Duplicate | `command+d` |
| Move to trash | `command+Delete` |
| Go to folder | `command+shift+g` |
| Show hidden files | `command+shift+.` |
| Open selected | `command+Down` |
| Go to parent folder | `command+Up` |
| Quick Look | `space` |
| View as icons | `command+1` |
| View as list | `command+2` |
| View in columns | `command+3` |
| Connect to server | `command+k` |
| Open Home folder | `command+shift+h` |
| Open Desktop folder | `command+shift+d` |
| Open Downloads folder | `option+command+l` |

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
8. screenshot — verify action result (see Text Input State below)
```

### Text Input State (CRITICAL — read this carefully)

Some actions activate a **text input field** where the next step is typing, NOT clicking. Clicking on an active text field will **dismiss it** and you lose the state.

**Actions that activate text input:**
- Clicking "Rename" in a context menu → filename becomes an editable text field
- Pressing `Return` on a selected file in Finder → rename mode activates
- `command+l` in browser → address bar is focused
- Clicking a search box or form field → text cursor appears
- `command+f` → find bar opens with cursor ready
- Double-clicking text in a document → text becomes editable

**After activating text input, follow this pattern:**
```
1. screenshot — verify the text field is active (look for: blue border around text,
   blinking cursor, highlighted/selected text, or editable text area)
2. DO NOT click on the text field — this will DEACTIVATE it
3. cmd+a — select all existing text (optional, if you need to replace)
4. type: your new text
5. Return — confirm the input
6. screenshot — verify the change was applied
```

**How to recognize an active text field in a screenshot:**
- Text is highlighted/selected (blue background over text)
- A thin blue or white border appears around the filename or field
- The cursor (blinking line) is visible inside the field
- The text area looks slightly different from its normal state

**If you accidentally click and dismiss the text field:**
- Do NOT repeat the same sequence — you'll loop forever
- Instead: re-select the item, then use keyboard (Return for Finder rename) or re-open context menu

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

## Escape Normalization (CRITICAL)

`Escape` is your reset button. Use it aggressively to clear unknown state.

**When to press Escape:**
- Before ANY keyboard shortcut (clears menus, dialogs, Spotlight)
- After a failed action (resets state before retry)
- When you don't know what's on screen (normalize first, then screenshot)
- After closing Spotlight (`cmd+space`) — ALWAYS press Escape, never click away
- Before switching apps (clears any open overlay in current app)

**Escape sequence for stuck states:**
```
1. key: Escape          — dismiss overlay/menu/dialog
2. key: Escape          — press again (some dialogs need 2 presses)
3. screenshot           — see what state we're in now
4. Decide next action based on clean state
```

**Multiple Escape is safe** — pressing Escape when nothing is open does nothing. It never causes harm.

## Error Recovery

1. `key: Escape` (2x) — close dialogs, menus, cancel operations
2. `screenshot` — always check what happened
3. `command+z` — undo last action
4. `command+w` — close current window/tab
5. Terminal fallback: `osascript`, `open`, `pbcopy`/`pbpaste` — always available when GUI fails
6. App not responding: `command+option+Escape` opens Force Quit, or `osascript -e 'tell application "AppName" to quit'`
7. **Retry limit**: if an action fails 2 times, switch to a different approach (terminal, osascript, different shortcut). Do NOT keep retrying the same thing.

## Accessibility Permissions

The computer tool requires macOS permissions:
- **Screen Recording**: System Settings > Privacy & Security > Screen Recording — add Terminal/iTerm
- **Accessibility**: System Settings > Privacy & Security > Accessibility — add Terminal/iTerm
- Symptom of missing permission: screenshot returns empty or click/type fails silently
- After granting permission, Terminal must be **fully restarted** (not just new tab)
- For gateway: the Python process itself needs these permissions

## Zoom Action

Use `zoom` to inspect a small area at full resolution. Useful for reading small text, verifying icons, or checking UI details.

```
computer action=zoom, region=[x1, y1, x2, y2]
```

**Rules:**
- Region coordinates are in screenshot space (not screen space)
- Minimum region size: 30x30 pixels (smaller regions are rejected)
- Aim for regions of 100x100 to 400x300 for best results
- Do NOT use tiny strips (e.g. 1300x25) — they produce unusable images
- If you need to read text, capture a region that includes full line height plus padding

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
10. left_click — creates folder with editable name field active
    *** TEXT INPUT STATE — do NOT click again ***
11. screenshot — verify name field is editable (text highlighted)
12. type: MyNewFolder — (do NOT click the name field first!)
13. key: Return — confirm name
14. screenshot — verify folder created with correct name
```

### Rename a file or folder in Finder:
```
METHOD 1 — Keyboard (preferred):
1. Click file to select it
2. screenshot — verify file is selected (highlighted)
3. key: Return — activates rename mode
    *** TEXT INPUT STATE — do NOT click the filename ***
4. screenshot — verify name is editable (text highlighted in blue)
5. cmd+a — select all existing text
6. type: NewFileName
7. key: Return — confirm rename
8. screenshot — verify renamed

METHOD 2 — Context menu:
1. mouse_move to file, verify, right_click — context menu appears
2. screenshot — find "Rename" in menu
3. mouse_move to "Rename", verify, left_click
    *** TEXT INPUT STATE — do NOT click the filename ***
4. screenshot — verify name field is editable
5. cmd+a — select all existing text
6. type: NewFileName
7. key: Return — confirm rename
8. screenshot — verify renamed
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
