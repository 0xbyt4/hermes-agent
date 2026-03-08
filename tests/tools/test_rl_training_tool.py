"""Tests for rl_training_tool.py - file handle lifecycle and cleanup."""

import subprocess
from dataclasses import dataclass
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from tools.rl_training_tool import RunState, _stop_training_run


def _make_run_state(**overrides) -> RunState:
    """Create a minimal RunState for testing."""
    defaults = {
        "run_id": "test-run-001",
        "environment": "test_env",
        "config": {},
    }
    defaults.update(overrides)
    return RunState(**defaults)


class TestRunStateLogFiles:
    """Verify that RunState tracks log file handles."""

    def test_log_files_default_empty(self):
        state = _make_run_state()
        assert state._log_files == []

    def test_log_files_accumulates(self):
        state = _make_run_state()
        fh1 = StringIO()
        fh2 = StringIO()
        state._log_files.append(fh1)
        state._log_files.append(fh2)
        assert len(state._log_files) == 2

    def test_separate_instances_have_separate_lists(self):
        s1 = _make_run_state(run_id="a")
        s2 = _make_run_state(run_id="b")
        s1._log_files.append(StringIO())
        assert len(s2._log_files) == 0


class TestStopTrainingRun:
    """Verify that _stop_training_run closes file handles and terminates processes."""

    def test_closes_all_log_files(self):
        state = _make_run_state()
        files = [MagicMock() for _ in range(3)]
        state._log_files = files

        _stop_training_run(state)

        for fh in files:
            fh.close.assert_called_once()
        assert state._log_files == []

    def test_clears_log_files_list(self):
        state = _make_run_state()
        state._log_files = [MagicMock()]

        _stop_training_run(state)

        assert state._log_files == []

    def test_close_exception_does_not_propagate(self):
        """If a file handle .close() raises, it should not crash."""
        state = _make_run_state()
        bad_fh = MagicMock()
        bad_fh.close.side_effect = OSError("already closed")
        good_fh = MagicMock()
        state._log_files = [bad_fh, good_fh]

        _stop_training_run(state)

        # Both should be attempted, second should succeed
        bad_fh.close.assert_called_once()
        good_fh.close.assert_called_once()
        assert state._log_files == []

    def test_terminates_processes(self):
        state = _make_run_state()
        for attr in ("api_process", "trainer_process", "env_process"):
            proc = MagicMock()
            proc.poll.return_value = None  # still running
            setattr(state, attr, proc)

        _stop_training_run(state)

        for attr in ("api_process", "trainer_process", "env_process"):
            getattr(state, attr).terminate.assert_called_once()

    def test_no_crash_with_no_processes_and_no_files(self):
        state = _make_run_state()
        _stop_training_run(state)  # should not raise

    def test_sets_status_to_stopped_when_running(self):
        state = _make_run_state(status="running")
        _stop_training_run(state)
        assert state.status == "stopped"

    def test_does_not_change_status_when_failed(self):
        state = _make_run_state(status="failed")
        _stop_training_run(state)
        assert state.status == "failed"

    def test_handles_mixed_running_and_exited_processes(self):
        state = _make_run_state()
        # api still running
        api = MagicMock()
        api.poll.return_value = None
        state.api_process = api
        # trainer already exited
        trainer = MagicMock()
        trainer.poll.return_value = 0
        state.trainer_process = trainer
        # env is None
        state.env_process = None

        _stop_training_run(state)

        api.terminate.assert_called_once()
        trainer.terminate.assert_not_called()
