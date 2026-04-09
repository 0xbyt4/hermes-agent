"""
Hermes Self Watcher -- notify the active gateway when the agent's own
source code changes.

This is a dev-only feature. When enabled it runs a background thread that
polls `git status --porcelain` every N minutes against the hermes-agent
repo itself. When a change is detected it formats a `git diff --stat`
summary and ships it to the first reachable channel from a priority list
(e.g. telegram -> discord -> cli-stdout).

Design goals:
- Zero new dependencies: stdlib only (threading, subprocess, pathlib, json).
- Silent no-op if the repo is not a git checkout (pip-installed users).
- Daemon thread so it never blocks process shutdown.
- Failures inside the watcher never crash the host process.
- Uses the existing `tools.send_message_tool.send_message_tool` so it
  piggybacks on whatever platforms are already configured (home channel,
  credentials, etc.) without duplicating gateway logic.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Reasonable upper bound so a typo like "0" doesn't DoS the user's repo.
_MIN_INTERVAL_MIN = 0.1
_DEFAULT_INTERVAL_MIN = 1.0
_DEFAULT_PRIORITY: List[str] = ["telegram", "discord", "cli"]

# Git commands run from the watcher. Kept as lists (no shell) and capped by
# timeout so a hung git process can't stall the thread forever.
_GIT_TIMEOUT_SEC = 10


def _repo_root() -> Path:
    """Return the hermes-agent repo root (parent of the `agent/` dir)."""
    return Path(__file__).resolve().parent.parent


def _porcelain_path(line: str) -> str:
    """Extract the path from a `git status --porcelain` line.

    Porcelain v1 format: `XY <path>` where XY is 2 chars + space. For
    renames the path is `old -> new`; we keep the new path only.
    """
    if len(line) < 4:
        return line
    path = line[3:]
    if " -> " in path:
        path = path.split(" -> ", 1)[1]
    return path.strip().strip('"')


class SelfWatcher:
    """Background thread that polls git and forwards diffs to a notifier.

    Parameters
    ----------
    repo_root : Path
        Directory to run git commands in. Must contain a `.git` entry.
    interval_min : float
        Poll interval in minutes. Clamped to >= _MIN_INTERVAL_MIN.
    notify : Callable[[str], None]
        Function called with the formatted notification text whenever a
        change is detected. Must be thread-safe; exceptions are swallowed.
    """

    def __init__(
        self,
        repo_root: Path,
        interval_min: float,
        notify: Callable[[str], None],
    ) -> None:
        self.repo_root = repo_root
        self.interval_sec = max(interval_min, _MIN_INTERVAL_MIN) * 60.0
        self.notify = notify
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_state: Optional[frozenset] = None

    def start(self) -> bool:
        """Start the background thread. Returns True if the watcher is
        actually running; False if the repo is not a git checkout (in
        which case the feature is silently disabled)."""
        if not (self.repo_root / ".git").exists():
            logger.debug("self_watcher: %s is not a git repo, disabled", self.repo_root)
            return False
        # Prime the last-known state so the first iteration doesn't fire a
        # spurious notification for pre-existing unstaged changes.
        self._last_state = self._git_state()
        self._thread = threading.Thread(
            target=self._loop,
            name="hermes-self-watcher",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "self_watcher: started (interval=%.1fmin, repo=%s)",
            self.interval_sec / 60.0,
            self.repo_root,
        )
        return True

    def stop(self) -> None:
        """Signal the watcher to stop. Safe to call multiple times."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        # `Event.wait` returns True when set, False on timeout. We use the
        # timeout path as the poll tick.
        while not self._stop.wait(self.interval_sec):
            try:
                self._tick()
            except Exception as exc:  # never crash the thread
                logger.debug("self_watcher: tick error: %s", exc)

    def _tick(self) -> None:
        current = self._git_state()
        previous = self._last_state or frozenset()
        if current == previous:
            return

        # Compare at the PATH level, not at the raw porcelain-line level.
        # This makes staging flips invisible (" M foo.py" <-> "M  foo.py"
        # is the same path) while still catching real additions.
        old_paths = {_porcelain_path(line) for line in previous}
        new_paths = {_porcelain_path(line) for line in current}
        newly_seen = new_paths - old_paths

        # Untracked additions we haven't seen before. `git diff --stat`
        # only covers tracked files, so we collect these separately and
        # append them to the notification body.
        untracked_new = sorted(
            _porcelain_path(line)
            for line in (current - previous)
            if line.startswith("??") and _porcelain_path(line) in newly_seen
        )

        diff_stat = self._git_diff_stat()
        self._last_state = current

        if not diff_stat and not untracked_new:
            # Nothing meaningful to report: revert, stash, or staging flip.
            return

        parts: List[str] = []
        if diff_stat:
            parts.append(diff_stat)
        if untracked_new:
            parts.append(
                "Untracked:\n" + "\n".join(f"  {p}" for p in untracked_new)
            )

        msg = f"🔧 Source change detected:\n" + "\n\n".join(parts)
        try:
            self.notify(msg)
        except Exception as exc:
            logger.debug("self_watcher: notify failed: %s", exc)

    def _git_state(self) -> frozenset:
        """Return a hashable snapshot of `git status --porcelain`.

        The set comparison catches any change in tracked OR untracked
        files, including renames, deletions, and new files -- .gitignore
        is honored automatically.
        """
        out = self._run(["git", "status", "--porcelain"])
        return frozenset(out.splitlines())

    def _git_diff_stat(self) -> str:
        """Return `git diff --stat` output, stripped."""
        return self._run(["git", "diff", "--stat"]).strip()

    def _run(self, cmd: List[str]) -> str:
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT_SEC,
                check=False,
            )
            return result.stdout or ""
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.debug("self_watcher: git command failed: %s", exc)
            return ""


# ----------------------------------------------------------------------
# Notifier helpers -- turn a priority list into a single callable
# ----------------------------------------------------------------------

def _send_via_platform(platform: str, message: str) -> bool:
    """Attempt to send `message` to `platform`'s home channel via the
    existing send_message tool. Returns True on success.

    This deliberately uses the tool entry point so the watcher inherits
    credential loading, home channel resolution, and per-platform quirks
    without duplicating logic.
    """
    try:
        from tools.send_message_tool import send_message_tool
    except Exception as exc:
        logger.debug("self_watcher: send_message_tool import failed: %s", exc)
        return False

    try:
        raw = send_message_tool({"target": platform, "message": message})
    except Exception as exc:
        logger.debug("self_watcher: send_message_tool raised for %s: %s", platform, exc)
        return False

    if not isinstance(raw, str):
        # Tool returned a dict/other -- treat truthy as success.
        return bool(raw)
    try:
        payload = json.loads(raw)
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("error"):
        logger.debug(
            "self_watcher: %s send returned error: %s",
            platform,
            payload.get("error"),
        )
        return False
    return bool(payload.get("success", True))


def _print_to_cli(message: str) -> bool:
    """Fallback 'cli' channel: print to stdout with a blank-line buffer so
    the notification doesn't collide with the user's prompt line."""
    try:
        print(f"\n{message}\n", flush=True)
        return True
    except Exception:
        return False


def make_priority_notifier(priority: List[str]) -> Callable[[str], None]:
    """Build a notifier that walks `priority` and returns on the first
    successful delivery. Unknown channel names are skipped.
    """
    normalized = [p.strip().lower() for p in priority if p and p.strip()]
    if not normalized:
        normalized = list(_DEFAULT_PRIORITY)

    def notify(message: str) -> None:
        for channel in normalized:
            if channel == "cli":
                if _print_to_cli(message):
                    return
                continue
            if _send_via_platform(channel, message):
                return
        # Final fallback so we never drop a notification silently.
        logger.info("self_watcher: no notification channel worked; dropping\n%s", message)

    return notify


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------

def _load_config_section_from_yaml() -> Optional[dict]:
    """Look up the `self_watcher` section from the hermes config files.

    Precedence matches `cli.py`'s `load_cli_config`:
        1. $HERMES_HOME/config.yaml (or ~/.hermes/config.yaml)
        2. <repo>/cli-config.yaml

    Returns the parsed section or None. Never raises.
    """
    candidates: List[Path] = []
    import os as _os
    hermes_home = _os.environ.get("HERMES_HOME")
    if hermes_home:
        candidates.append(Path(hermes_home).expanduser() / "config.yaml")
    candidates.append(Path.home() / ".hermes" / "config.yaml")
    candidates.append(_repo_root() / "cli-config.yaml")

    for path in candidates:
        if not path.exists():
            continue
        try:
            import yaml  # PyYAML ships with hermes already
            with path.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
        except Exception as exc:
            logger.debug("self_watcher: failed to read %s: %s", path, exc)
            continue
        if isinstance(data, dict) and isinstance(data.get("self_watcher"), dict):
            return data["self_watcher"]
    return None


def start_if_enabled(repo_root: Optional[Path] = None) -> Optional[SelfWatcher]:
    """Convenience entry point: look up config from the usual files and
    start a watcher if it is enabled. Returns the watcher or None.

    Safe to call from any process start-up path -- it never raises.
    """
    try:
        section = _load_config_section_from_yaml()
    except Exception as exc:
        logger.debug("self_watcher: config lookup failed: %s", exc)
        section = None
    return start_from_config(section, repo_root=repo_root)


def start_from_config(
    config_section: Optional[dict],
    repo_root: Optional[Path] = None,
) -> Optional[SelfWatcher]:
    """Instantiate and start a SelfWatcher from a config dict slice.

    Expected shape (all keys optional):

        self_watcher:
          enabled: false
          poll_interval_min: 1
          priority: [telegram, discord, cli]

    Returns None when disabled, when config is missing, or when the repo
    is not a git checkout. This function never raises.
    """
    if not config_section or not isinstance(config_section, dict):
        return None
    if not config_section.get("enabled", False):
        return None

    try:
        interval_min = float(config_section.get("poll_interval_min", _DEFAULT_INTERVAL_MIN))
    except (TypeError, ValueError):
        interval_min = _DEFAULT_INTERVAL_MIN

    raw_priority = config_section.get("priority") or _DEFAULT_PRIORITY
    if isinstance(raw_priority, str):
        priority = [p.strip() for p in raw_priority.split(",") if p.strip()]
    elif isinstance(raw_priority, (list, tuple)):
        priority = [str(p).strip() for p in raw_priority if str(p).strip()]
    else:
        priority = list(_DEFAULT_PRIORITY)

    notifier = make_priority_notifier(priority)
    watcher = SelfWatcher(
        repo_root=repo_root or _repo_root(),
        interval_min=interval_min,
        notify=notifier,
    )
    try:
        started = watcher.start()
    except Exception as exc:
        logger.debug("self_watcher: start failed: %s", exc)
        return None
    if not started:
        return None
    return watcher
