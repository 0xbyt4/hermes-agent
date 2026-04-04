"""Tests for the Claude Code CLI delegation tool."""

import json
import subprocess
from unittest.mock import patch, MagicMock

import pytest

from tools.claude_code_tool import (
    claude_code,
    _handle_claude_code_dispatch,
    check_claude_code_available,
    _build_claude_command,
    DEFAULT_TIMEOUT,
    MAX_PROMPT_LENGTH,
)


class TestCheckAvailability:
    def test_available_when_claude_on_path(self):
        with patch("shutil.which", return_value="/usr/local/bin/claude"):
            assert check_claude_code_available() is True

    def test_unavailable_when_not_on_path(self):
        with patch("shutil.which", return_value=None):
            assert check_claude_code_available() is False


class TestBuildCommand:
    def test_basic_command(self):
        cmd = _build_claude_command("hello")
        assert cmd == ["claude", "-p", "hello"]

    def test_with_model(self):
        cmd = _build_claude_command("hello", model="sonnet")
        assert cmd == ["claude", "-p", "--model", "sonnet", "hello"]

    def test_with_max_turns(self):
        cmd = _build_claude_command("hello", max_turns=5)
        assert cmd == ["claude", "-p", "--max-turns", "5", "hello"]

    def test_with_all_options(self):
        cmd = _build_claude_command("hello", model="opus", max_turns=3)
        assert cmd == ["claude", "-p", "--model", "opus", "--max-turns", "3", "hello"]

    def test_zero_max_turns_ignored(self):
        cmd = _build_claude_command("hello", max_turns=0)
        assert cmd == ["claude", "-p", "hello"]


class TestClaudeCode:
    def test_empty_prompt(self):
        result = claude_code(prompt="")
        assert result["success"] is False
        assert "empty" in result["error"].lower()

    def test_prompt_too_long(self):
        result = claude_code(prompt="x" * (MAX_PROMPT_LENGTH + 1))
        assert result["success"] is False
        assert "too long" in result["error"].lower()

    def test_cli_not_found(self):
        with patch("tools.claude_code_tool.check_claude_code_available", return_value=False):
            result = claude_code(prompt="hello")
            assert result["success"] is False
            assert "not found" in result["error"].lower()

    def test_successful_execution(self):
        mock_result = MagicMock()
        mock_result.stdout = "Hello world"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            result = claude_code(prompt="say hello")
            assert result["success"] is True
            assert result["output"] == "Hello world"

    def test_nonzero_exit_code(self):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "Authentication failed"
        mock_result.returncode = 1

        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            result = claude_code(prompt="do something")
            assert result["success"] is False
            assert "Authentication failed" in result["error"]

    def test_timeout(self):
        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="claude", timeout=10)):
            result = claude_code(prompt="long task", timeout=10)
            assert result["success"] is False
            assert "timed out" in result["error"].lower()

    def test_output_truncation(self):
        long_output = "x" * 60_000
        mock_result = MagicMock()
        mock_result.stdout = long_output
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            result = claude_code(prompt="generate long output")
            assert result["success"] is True
            assert "[Output truncated]" in result["output"]

    def test_timeout_capped_at_600(self):
        mock_result = MagicMock()
        mock_result.stdout = "ok"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            claude_code(prompt="test", timeout=9999)
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs["timeout"] == 600

    def test_custom_cwd(self):
        import os
        mock_result = MagicMock()
        mock_result.stdout = "ok"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", return_value=mock_result) as mock_run:
            claude_code(prompt="test", cwd="/tmp")
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs["cwd"] == os.path.realpath("/tmp")


    def test_invalid_cwd(self):
        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True):
            result = claude_code(prompt="test", cwd="/nonexistent/path/xyz")
            assert result["success"] is False
            assert "does not exist" in result["error"]


class TestDispatchHandler:
    def test_handler_returns_json(self):
        mock_result = MagicMock()
        mock_result.stdout = "result"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("subprocess.run", return_value=mock_result):
            output = _handle_claude_code_dispatch({"prompt": "test"})
            parsed = json.loads(output)
            assert parsed["success"] is True
            assert parsed["output"] == "result"

    def test_handler_receives_args_dict(self):
        with patch("tools.claude_code_tool.check_claude_code_available", return_value=True), \
             patch("tools.claude_code_tool.claude_code", return_value={"success": True, "output": "ok"}) as mock_cc:
            _handle_claude_code_dispatch({"prompt": "test", "model": "", "max_turns": 0, "cwd": "", "timeout": 0})
            mock_cc.assert_called_once_with(
                prompt="test", model=None, max_turns=None, cwd=None, timeout=None,
            )
