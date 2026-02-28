"""Tests for gateway/status.py -- PID-file based gateway detection.

Covers: write_pid_file, remove_pid_file, is_gateway_running including
stale PID cleanup and malformed file handling.
"""

import os

import pytest

import gateway.status as status_mod


@pytest.fixture(autouse=True)
def _use_tmp_pid(tmp_path, monkeypatch):
    """Redirect PID file to tmp_path so tests never touch ~/.hermes."""
    monkeypatch.setattr(status_mod, "_PID_FILE", tmp_path / "gateway.pid")


# =========================================================================
# write_pid_file / remove_pid_file
# =========================================================================


class TestPidFileLifecycle:
    def test_write_creates_file(self):
        status_mod.write_pid_file()
        assert status_mod._PID_FILE.exists()

    def test_write_contains_current_pid(self):
        status_mod.write_pid_file()
        content = status_mod._PID_FILE.read_text().strip()
        assert content == str(os.getpid())

    def test_remove_deletes_file(self):
        status_mod.write_pid_file()
        status_mod.remove_pid_file()
        assert not status_mod._PID_FILE.exists()

    def test_remove_no_file_is_noop(self):
        # Should not raise even if file doesn't exist
        status_mod.remove_pid_file()


# =========================================================================
# is_gateway_running
# =========================================================================


class TestIsGatewayRunning:
    def test_no_pid_file_returns_false(self):
        assert status_mod.is_gateway_running() is False

    def test_current_process_returns_true(self):
        status_mod.write_pid_file()
        assert status_mod.is_gateway_running() is True

    def test_stale_pid_cleaned_up(self):
        # Write a PID that definitely doesn't exist
        # PID 2^22 + random offset is extremely unlikely to be running
        fake_pid = 4194999
        status_mod._PID_FILE.write_text(str(fake_pid))

        result = status_mod.is_gateway_running()
        assert result is False
        # Stale PID file should be removed
        assert not status_mod._PID_FILE.exists()

    def test_malformed_pid_file_returns_false(self):
        status_mod._PID_FILE.write_text("not-a-number")
        result = status_mod.is_gateway_running()
        assert result is False
        # Malformed file should be cleaned up
        assert not status_mod._PID_FILE.exists()

    def test_empty_pid_file_returns_false(self):
        status_mod._PID_FILE.write_text("")
        assert status_mod.is_gateway_running() is False
