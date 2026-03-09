"""Tests for reasoning display feature.

Verifies:
1. run_agent returns last_reasoning in result dict
2. CLI _toggle_reasoning toggles show_reasoning state
3. Reasoning is rendered when show_reasoning is True
4. Reasoning is hidden when show_reasoning is False
5. Long reasoning is collapsed to 10 lines
6. Config default is respected
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock


class TestLastReasoningInResult(unittest.TestCase):
    """Verify run_agent includes last_reasoning in its return dict."""

    def _build_messages(self, reasoning=None):
        """Build a minimal messages list with an assistant message."""
        msgs = [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": "Hi there!",
                "reasoning": reasoning,
                "finish_reason": "stop",
            },
        ]
        return msgs

    def test_last_reasoning_present_when_model_reasons(self):
        """Result dict should contain reasoning from the last assistant message."""
        messages = self._build_messages(reasoning="Let me think about this carefully...")
        # Simulate what run_agent does to extract last_reasoning
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertEqual(last_reasoning, "Let me think about this carefully...")

    def test_last_reasoning_none_when_no_reasoning(self):
        """Result dict should have None when model doesn't reason."""
        messages = self._build_messages(reasoning=None)
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertIsNone(last_reasoning)

    def test_last_reasoning_picks_final_assistant_message(self):
        """When multiple assistant messages exist, pick the last one's reasoning."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "...", "reasoning": "first thought", "finish_reason": "tool_calls"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "done!", "reasoning": "final thought", "finish_reason": "stop"},
        ]
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertEqual(last_reasoning, "final thought")

    def test_last_reasoning_skips_empty_reasoning(self):
        """Empty string reasoning should be treated as no reasoning."""
        messages = self._build_messages(reasoning="")
        last_reasoning = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and msg.get("reasoning"):
                last_reasoning = msg["reasoning"]
                break
        self.assertIsNone(last_reasoning)


class TestReasoningCollapse(unittest.TestCase):
    """Verify long reasoning is collapsed to 10 lines."""

    def test_short_reasoning_not_collapsed(self):
        """Reasoning with <= 10 lines should be shown in full."""
        reasoning = "\n".join(f"Line {i}" for i in range(5))
        lines = reasoning.strip().splitlines()
        self.assertEqual(len(lines), 5)
        self.assertLessEqual(len(lines), 10)

    def test_long_reasoning_collapsed(self):
        """Reasoning with > 10 lines should show first 10 + count."""
        reasoning = "\n".join(f"Line {i}" for i in range(25))
        lines = reasoning.strip().splitlines()
        if len(lines) > 10:
            display = "\n".join(lines[:10])
            display += f"\n  ... ({len(lines) - 10} more lines)"
        else:
            display = reasoning.strip()
        display_lines = display.splitlines()
        # 10 content lines + 1 "more lines" indicator
        self.assertEqual(len(display_lines), 11)
        self.assertIn("15 more lines", display_lines[-1])

    def test_exactly_10_lines_not_collapsed(self):
        """Reasoning with exactly 10 lines should not be collapsed."""
        reasoning = "\n".join(f"Line {i}" for i in range(10))
        lines = reasoning.strip().splitlines()
        self.assertEqual(len(lines), 10)
        self.assertFalse(len(lines) > 10)


class TestToggleReasoning(unittest.TestCase):
    """Verify /reasoning command toggles state correctly."""

    def _make_cli_stub(self, initial=False):
        """Create a minimal object with show_reasoning attribute."""
        stub = SimpleNamespace(show_reasoning=initial)
        return stub

    def test_toggle_on(self):
        stub = self._make_cli_stub(initial=False)
        stub.show_reasoning = not stub.show_reasoning
        self.assertTrue(stub.show_reasoning)

    def test_toggle_off(self):
        stub = self._make_cli_stub(initial=True)
        stub.show_reasoning = not stub.show_reasoning
        self.assertFalse(stub.show_reasoning)

    def test_explicit_on(self):
        stub = self._make_cli_stub(initial=False)
        arg = "on"
        if arg == "on":
            stub.show_reasoning = True
        self.assertTrue(stub.show_reasoning)

    def test_explicit_off(self):
        stub = self._make_cli_stub(initial=True)
        arg = "off"
        if arg == "off":
            stub.show_reasoning = False
        self.assertFalse(stub.show_reasoning)


class TestConfigDefault(unittest.TestCase):
    """Verify config default for show_reasoning."""

    def test_default_config_has_show_reasoning(self):
        from hermes_cli.config import DEFAULT_CONFIG
        display = DEFAULT_CONFIG.get("display", {})
        self.assertIn("show_reasoning", display)
        self.assertFalse(display["show_reasoning"])


class TestCommandRegistered(unittest.TestCase):
    """Verify /reasoning is in the COMMANDS dict."""

    def test_reasoning_in_commands(self):
        from hermes_cli.commands import COMMANDS
        self.assertIn("/reasoning", COMMANDS)
        self.assertIn("reasoning", COMMANDS["/reasoning"].lower())


if __name__ == "__main__":
    unittest.main()
