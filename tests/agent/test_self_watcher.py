"""Tests for agent.self_watcher -- dev-only source tree change watcher.

The SelfWatcher polls `git status --porcelain` in the hermes repo and
forwards formatted diffs to a notifier. Tests use a tmp git repo so no
real subprocess dependencies leak into other tests, and they drive
`_tick()` directly to avoid timing flake from the background thread.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

from agent import self_watcher as sw
from agent.self_watcher import (
    SelfWatcher,
    _DEFAULT_PRIORITY,
    _porcelain_path,
    make_priority_notifier,
    start_from_config,
    start_if_enabled,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _run(cmd: List[str], cwd: Path) -> None:
    """Run a subprocess in `cwd` and raise on failure."""
    subprocess.run(cmd, cwd=str(cwd), check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Initialise a minimal git repo with one committed file."""
    _run(["git", "init", "-q", "-b", "main"], cwd=tmp_path)
    _run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path)
    _run(["git", "config", "user.name", "Test"], cwd=tmp_path)
    _run(["git", "config", "commit.gpgsign", "false"], cwd=tmp_path)

    (tmp_path / "foo.py").write_text("x = 1\n", encoding="utf-8")
    _run(["git", "add", "foo.py"], cwd=tmp_path)
    _run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path)
    return tmp_path


def _make_watcher(repo: Path, notify=None) -> SelfWatcher:
    """Build a watcher without starting its thread (for _tick-driven tests)."""
    calls: List[str] = []

    def default_notify(msg: str) -> None:
        calls.append(msg)

    watcher = SelfWatcher(
        repo_root=repo,
        interval_min=0.1,
        notify=notify or default_notify,
    )
    # Prime the last-known state the way .start() would.
    watcher._last_state = watcher._git_state()
    watcher._captured = calls  # type: ignore[attr-defined]
    return watcher


# ─────────────────────────────────────────────────────────────────────────
# _porcelain_path
# ─────────────────────────────────────────────────────────────────────────

class TestPorcelainPath:
    def test_modified_file(self):
        assert _porcelain_path(" M foo.py") == "foo.py"

    def test_staged_modified(self):
        assert _porcelain_path("M  foo.py") == "foo.py"

    def test_untracked(self):
        assert _porcelain_path("?? newfile.py") == "newfile.py"

    def test_rename_keeps_destination(self):
        assert _porcelain_path("R  old.py -> new.py") == "new.py"

    def test_quoted_path(self):
        assert _porcelain_path(' M "with space.py"') == "with space.py"

    def test_short_line_returned_as_is(self):
        # Defensive: malformed input should not crash.
        assert _porcelain_path("XX") == "XX"


# ─────────────────────────────────────────────────────────────────────────
# start_from_config / start_if_enabled -- disable paths
# ─────────────────────────────────────────────────────────────────────────

class TestStartFromConfigDisabled:
    def test_none_config_returns_none(self):
        assert start_from_config(None) is None

    def test_empty_dict_returns_none(self):
        assert start_from_config({}) is None

    def test_explicitly_disabled_returns_none(self):
        assert start_from_config({"enabled": False}) is None

    def test_non_dict_returns_none(self):
        # Defensive: bad yaml that parses to a list should not crash.
        assert start_from_config(["enabled", "true"]) is None  # type: ignore[arg-type]

    def test_enabled_but_no_git_returns_none(self, tmp_path: Path):
        # Enabled config but repo_root has no .git -- should silently disable.
        result = start_from_config(
            {"enabled": True, "poll_interval_min": 0.1},
            repo_root=tmp_path,
        )
        assert result is None

    def test_enabled_with_git_returns_watcher(self, git_repo: Path):
        # Use a notifier that does nothing so we don't touch send_message_tool.
        with patch.object(sw, "_send_via_platform", return_value=True), \
             patch.object(sw, "_print_to_cli", return_value=True):
            watcher = start_from_config(
                {"enabled": True, "poll_interval_min": 0.1, "priority": ["cli"]},
                repo_root=git_repo,
            )
            assert watcher is not None
            try:
                assert watcher._thread is not None
                assert watcher._thread.is_alive()
            finally:
                watcher.stop()

    def test_invalid_interval_falls_back_to_default(self, git_repo: Path):
        watcher = start_from_config(
            {"enabled": True, "poll_interval_min": "not-a-number", "priority": ["cli"]},
            repo_root=git_repo,
        )
        assert watcher is not None
        try:
            # Default is 1.0 min -> 60 sec.
            assert watcher.interval_sec == pytest.approx(60.0)
        finally:
            watcher.stop()

    def test_interval_clamped_to_minimum(self, git_repo: Path):
        watcher = start_from_config(
            {"enabled": True, "poll_interval_min": 0.0, "priority": ["cli"]},
            repo_root=git_repo,
        )
        assert watcher is not None
        try:
            # Clamped to 0.1 min -> 6 sec.
            assert watcher.interval_sec == pytest.approx(6.0)
        finally:
            watcher.stop()


class TestStartIfEnabled:
    def test_no_config_file_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # Point HERMES_HOME somewhere empty and make home() return an empty
        # dir so no cli-config.yaml / user config is ever found.
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        assert start_if_enabled(repo_root=tmp_path) is None


# ─────────────────────────────────────────────────────────────────────────
# make_priority_notifier
# ─────────────────────────────────────────────────────────────────────────

class TestPriorityNotifier:
    def test_first_platform_wins(self):
        calls: List[tuple] = []

        def fake_send(platform, msg):
            calls.append((platform, msg))
            return True  # success on first try

        with patch.object(sw, "_send_via_platform", side_effect=fake_send):
            notify = make_priority_notifier(["telegram", "discord", "cli"])
            notify("hello")

        assert calls == [("telegram", "hello")]

    def test_fallthrough_to_discord(self):
        attempted: List[str] = []

        def fake_send(platform, msg):
            attempted.append(platform)
            return platform == "discord"

        with patch.object(sw, "_send_via_platform", side_effect=fake_send):
            notify = make_priority_notifier(["telegram", "discord", "cli"])
            notify("msg")

        assert attempted == ["telegram", "discord"]

    def test_fallthrough_to_cli(self):
        printed: List[str] = []

        def fake_print(msg):
            printed.append(msg)
            return True

        with patch.object(sw, "_send_via_platform", return_value=False), \
             patch.object(sw, "_print_to_cli", side_effect=fake_print):
            notify = make_priority_notifier(["telegram", "discord", "cli"])
            notify("msg")

        assert printed == ["msg"]

    def test_all_fail_does_not_raise(self):
        with patch.object(sw, "_send_via_platform", return_value=False), \
             patch.object(sw, "_print_to_cli", return_value=False):
            notify = make_priority_notifier(["telegram", "cli"])
            notify("msg")  # must not raise

    def test_unknown_channel_name_is_skipped(self):
        # "mastodon" is not a supported send target; if the notifier
        # doesn't recognise it explicitly it tries as a platform (which
        # returns False via _send_via_platform mock) and falls through.
        with patch.object(sw, "_send_via_platform", return_value=False), \
             patch.object(sw, "_print_to_cli", return_value=True):
            notify = make_priority_notifier(["mastodon", "cli"])
            notify("msg")  # must not raise

    def test_empty_priority_uses_default(self):
        # make_priority_notifier with empty list should default internally.
        calls: List[str] = []

        def fake_send(platform, msg):
            calls.append(platform)
            return False

        with patch.object(sw, "_send_via_platform", side_effect=fake_send), \
             patch.object(sw, "_print_to_cli", return_value=True):
            notify = make_priority_notifier([])
            notify("msg")

        # Default priority contains telegram, discord, cli.
        assert "telegram" in calls
        assert "discord" in calls


# ─────────────────────────────────────────────────────────────────────────
# SelfWatcher._tick -- detection semantics
# ─────────────────────────────────────────────────────────────────────────

class TestTickDetection:
    def test_no_change_no_notification(self, git_repo: Path):
        watcher = _make_watcher(git_repo)
        watcher._tick()
        assert watcher._captured == []  # type: ignore[attr-defined]

    def test_modify_tracked_file_notifies(self, git_repo: Path):
        watcher = _make_watcher(git_repo)
        (git_repo / "foo.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        watcher._tick()
        assert len(watcher._captured) == 1  # type: ignore[attr-defined]
        msg = watcher._captured[0]  # type: ignore[attr-defined]
        assert "Source change detected" in msg
        assert "foo.py" in msg
        # git diff --stat should be present
        assert "insertion" in msg or "+" in msg

    def test_no_duplicate_notification_when_state_unchanged(self, git_repo: Path):
        watcher = _make_watcher(git_repo)
        (git_repo / "foo.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        watcher._tick()
        assert len(watcher._captured) == 1  # type: ignore[attr-defined]
        # Tick again with no new changes -- must stay silent.
        watcher._tick()
        watcher._tick()
        assert len(watcher._captured) == 1  # type: ignore[attr-defined]

    def test_second_file_triggers_new_notification(self, git_repo: Path):
        watcher = _make_watcher(git_repo)
        (git_repo / "foo.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        watcher._tick()
        (git_repo / "bar.py").write_text("z = 3\n", encoding="utf-8")
        watcher._tick()
        assert len(watcher._captured) == 2  # type: ignore[attr-defined]
        assert "bar.py" in watcher._captured[1]  # type: ignore[attr-defined]

    def test_untracked_file_notifies_with_path_list(self, git_repo: Path):
        watcher = _make_watcher(git_repo)
        (git_repo / "newfile.py").write_text("print('hi')\n", encoding="utf-8")
        watcher._tick()
        assert len(watcher._captured) == 1  # type: ignore[attr-defined]
        msg = watcher._captured[0]  # type: ignore[attr-defined]
        # Untracked files are not in `git diff --stat`; we fall back to
        # showing the path list, so the filename must appear.
        assert "newfile.py" in msg

    def test_revert_is_silent(self, git_repo: Path):
        watcher = _make_watcher(git_repo)
        foo = git_repo / "foo.py"
        original = foo.read_text(encoding="utf-8")

        foo.write_text(original + "# edit\n", encoding="utf-8")
        watcher._tick()  # first change → notification
        assert len(watcher._captured) == 1  # type: ignore[attr-defined]

        foo.write_text(original, encoding="utf-8")  # revert
        watcher._tick()  # state changed back, but diff is empty → silent
        assert len(watcher._captured) == 1  # type: ignore[attr-defined]

    def test_staging_flip_is_silent(self, git_repo: Path):
        watcher = _make_watcher(git_repo)
        (git_repo / "foo.py").write_text("x = 999\n", encoding="utf-8")
        watcher._tick()
        initial_count = len(watcher._captured)  # type: ignore[attr-defined]
        assert initial_count == 1

        # Stage the file -- porcelain changes from " M" to "M " but
        # `git diff --stat` (unstaged only) becomes empty. No notification.
        _run(["git", "add", "foo.py"], cwd=git_repo)
        watcher._tick()
        assert len(watcher._captured) == initial_count  # type: ignore[attr-defined]

    def test_delete_tracked_file_notifies(self, git_repo: Path):
        watcher = _make_watcher(git_repo)
        (git_repo / "foo.py").unlink()
        watcher._tick()
        assert len(watcher._captured) == 1  # type: ignore[attr-defined]
        assert "foo.py" in watcher._captured[0]  # type: ignore[attr-defined]

    def test_notify_exception_does_not_crash(self, git_repo: Path):
        def boom(msg: str) -> None:
            raise RuntimeError("notifier blew up")

        watcher = _make_watcher(git_repo, notify=boom)
        (git_repo / "foo.py").write_text("x = 2\n", encoding="utf-8")
        # Must not raise.
        watcher._tick()


# ─────────────────────────────────────────────────────────────────────────
# SelfWatcher lifecycle (thread start/stop)
# ─────────────────────────────────────────────────────────────────────────

class TestLifecycle:
    def test_start_on_non_git_dir_returns_false(self, tmp_path: Path):
        watcher = SelfWatcher(tmp_path, interval_min=0.1, notify=lambda _: None)
        assert watcher.start() is False
        assert watcher._thread is None

    def test_start_and_stop_on_git_dir(self, git_repo: Path):
        watcher = SelfWatcher(git_repo, interval_min=0.1, notify=lambda _: None)
        try:
            assert watcher.start() is True
            assert watcher._thread is not None
            assert watcher._thread.is_alive()
        finally:
            watcher.stop()
        assert not watcher._thread.is_alive()

    def test_stop_is_idempotent(self, git_repo: Path):
        watcher = SelfWatcher(git_repo, interval_min=0.1, notify=lambda _: None)
        watcher.start()
        watcher.stop()
        watcher.stop()  # must not raise

    def test_stop_without_start_is_safe(self, git_repo: Path):
        watcher = SelfWatcher(git_repo, interval_min=0.1, notify=lambda _: None)
        watcher.stop()  # never started — must not raise


# ─────────────────────────────────────────────────────────────────────────
# _load_config_section_from_yaml
# ─────────────────────────────────────────────────────────────────────────

class TestYamlLoader:
    def test_reads_hermes_home_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            "self_watcher:\n"
            "  enabled: true\n"
            "  poll_interval_min: 2\n"
            "  priority: [telegram, cli]\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        # Make sure ~/.hermes/config.yaml doesn't accidentally shadow us.
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nobody"))

        section = sw._load_config_section_from_yaml()
        assert section == {
            "enabled": True,
            "poll_interval_min": 2,
            "priority": ["telegram", "cli"],
        }

    def test_missing_files_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "nowhere"))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nobody"))
        # Prevent the repo-root cli-config.yaml fallback from matching by
        # pointing _repo_root at an empty directory.
        monkeypatch.setattr(sw, "_repo_root", lambda: tmp_path / "empty")
        assert sw._load_config_section_from_yaml() is None

    def test_yaml_without_section_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("model:\n  default: foo\n", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nobody"))
        monkeypatch.setattr(sw, "_repo_root", lambda: tmp_path / "empty")
        assert sw._load_config_section_from_yaml() is None

    def test_malformed_yaml_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        cfg = tmp_path / "config.yaml"
        cfg.write_text("::: not valid yaml [[[", encoding="utf-8")
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nobody"))
        monkeypatch.setattr(sw, "_repo_root", lambda: tmp_path / "empty")
        assert sw._load_config_section_from_yaml() is None


# ─────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────

class TestDefaults:
    def test_default_priority_shape(self):
        # Defensive: downstream code assumes these three entries exist.
        assert _DEFAULT_PRIORITY == ["telegram", "discord", "cli"]
