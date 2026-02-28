"""Tests for tools/send_message_tool.py -- cross-platform messaging.

Covers: input validation, target parsing, platform lookup, home channel
fallback, channel name resolution, availability gating, and error paths.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tools.send_message_tool import (
    send_message_tool,
    _handle_send,
    _check_send_message,
    SEND_MESSAGE_SCHEMA,
)


def _parse(result_str: str) -> dict:
    """Parse JSON tool result string."""
    return json.loads(result_str)


# =========================================================================
# Schema sanity
# =========================================================================


class TestSchema:
    def test_schema_name(self):
        assert SEND_MESSAGE_SCHEMA["name"] == "send_message"

    def test_schema_has_action_enum(self):
        props = SEND_MESSAGE_SCHEMA["parameters"]["properties"]
        assert "send" in props["action"]["enum"]
        assert "list" in props["action"]["enum"]


# =========================================================================
# Input validation
# =========================================================================


class TestInputValidation:
    def test_missing_target(self):
        result = _parse(_handle_send({"message": "hi"}))
        assert "error" in result
        assert "target" in result["error"]

    def test_missing_message(self):
        result = _parse(_handle_send({"target": "telegram"}))
        assert "error" in result
        assert "message" in result["error"]

    def test_both_missing(self):
        result = _parse(_handle_send({}))
        assert "error" in result

    def test_empty_target(self):
        result = _parse(_handle_send({"target": "", "message": "hi"}))
        assert "error" in result

    def test_empty_message(self):
        result = _parse(_handle_send({"target": "telegram", "message": ""}))
        assert "error" in result


# =========================================================================
# Target parsing
# =========================================================================


class TestTargetParsing:
    @patch("tools.interrupt.is_interrupted", return_value=False)
    @patch("gateway.config.load_gateway_config")
    def test_platform_only_uses_home_channel(self, mock_config, mock_int):
        """'telegram' (no colon) should resolve to home channel."""
        from gateway.config import Platform, PlatformConfig, HomeChannel, GatewayConfig

        cfg = GatewayConfig(platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True, token="tok",
                home_channel=HomeChannel(
                    platform=Platform.TELEGRAM, chat_id="123", name="Home",
                ),
            ),
        })
        mock_config.return_value = cfg

        with patch("model_tools._run_async") as mock_run:
            mock_run.return_value = {"success": True}
            result = _parse(_handle_send({"target": "telegram", "message": "hi"}))

        assert result["success"] is True
        assert "home channel" in result.get("note", "")

    @patch("tools.interrupt.is_interrupted", return_value=False)
    @patch("gateway.config.load_gateway_config")
    def test_platform_with_numeric_id(self, mock_config, mock_int):
        """'telegram:123456' should send to that numeric chat_id."""
        from gateway.config import Platform, PlatformConfig, GatewayConfig

        cfg = GatewayConfig(platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok"),
        })
        mock_config.return_value = cfg

        with patch("model_tools._run_async") as mock_run:
            mock_run.return_value = {"success": True}
            result = _parse(_handle_send({"target": "telegram:123456", "message": "hi"}))

        assert result["success"] is True

    @patch("tools.interrupt.is_interrupted", return_value=False)
    @patch("gateway.config.load_gateway_config")
    def test_negative_telegram_id_treated_as_numeric(self, mock_config, mock_int):
        """'-100123456' (Telegram group ID) should be treated as numeric."""
        from gateway.config import Platform, PlatformConfig, GatewayConfig

        cfg = GatewayConfig(platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok"),
        })
        mock_config.return_value = cfg

        with patch("model_tools._run_async") as mock_run:
            mock_run.return_value = {"success": True}
            # Negative IDs should NOT trigger channel name resolution
            with patch("gateway.channel_directory.resolve_channel_name") as mock_resolve:
                result = _parse(_handle_send({
                    "target": "telegram:-100123456", "message": "hi",
                }))
                mock_resolve.assert_not_called()

        assert result["success"] is True

    @patch("tools.interrupt.is_interrupted", return_value=False)
    @patch("gateway.channel_directory.resolve_channel_name", return_value="C999")
    @patch("gateway.config.load_gateway_config")
    def test_channel_name_resolved(self, mock_config, mock_resolve, mock_int):
        """'discord:#bot-home' should resolve the name to a numeric ID."""
        from gateway.config import Platform, PlatformConfig, GatewayConfig

        cfg = GatewayConfig(platforms={
            Platform.DISCORD: PlatformConfig(enabled=True, token="tok"),
        })
        mock_config.return_value = cfg

        with patch("model_tools._run_async") as mock_run:
            mock_run.return_value = {"success": True}
            result = _parse(_handle_send({
                "target": "discord:#bot-home", "message": "hi",
            }))

        mock_resolve.assert_called_once_with("discord", "#bot-home")
        assert result["success"] is True

    @patch("tools.interrupt.is_interrupted", return_value=False)
    @patch("gateway.channel_directory.resolve_channel_name", return_value=None)
    def test_unresolvable_channel_name_errors(self, mock_resolve, mock_int):
        """If channel name can't be resolved, return a helpful error."""
        result = _parse(_handle_send({
            "target": "slack:#nonexistent", "message": "hi",
        }))
        assert "error" in result
        assert "resolve" in result["error"].lower()


# =========================================================================
# Platform validation
# =========================================================================


class TestPlatformValidation:
    @patch("tools.interrupt.is_interrupted", return_value=False)
    @patch("gateway.config.load_gateway_config")
    def test_unknown_platform(self, mock_config, mock_int):
        from gateway.config import GatewayConfig
        mock_config.return_value = GatewayConfig()

        result = _parse(_handle_send({"target": "matrix", "message": "hi"}))
        assert "error" in result
        assert "Unknown platform" in result["error"]
        assert "matrix" in result["error"]

    @patch("tools.interrupt.is_interrupted", return_value=False)
    @patch("gateway.config.load_gateway_config")
    def test_platform_not_configured(self, mock_config, mock_int):
        from gateway.config import GatewayConfig
        mock_config.return_value = GatewayConfig()

        result = _parse(_handle_send({"target": "telegram", "message": "hi"}))
        assert "error" in result
        assert "not configured" in result["error"]

    @patch("tools.interrupt.is_interrupted", return_value=False)
    @patch("gateway.config.load_gateway_config")
    def test_platform_disabled(self, mock_config, mock_int):
        from gateway.config import Platform, PlatformConfig, GatewayConfig

        cfg = GatewayConfig(platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=False, token="tok"),
        })
        mock_config.return_value = cfg

        result = _parse(_handle_send({"target": "telegram", "message": "hi"}))
        assert "error" in result
        assert "not configured" in result["error"]

    @patch("tools.interrupt.is_interrupted", return_value=False)
    @patch("gateway.config.load_gateway_config")
    def test_no_home_channel_error(self, mock_config, mock_int):
        from gateway.config import Platform, PlatformConfig, GatewayConfig

        cfg = GatewayConfig(platforms={
            Platform.TELEGRAM: PlatformConfig(enabled=True, token="tok"),
        })
        mock_config.return_value = cfg

        result = _parse(_handle_send({"target": "telegram", "message": "hi"}))
        assert "error" in result
        assert "home channel" in result["error"].lower()


# =========================================================================
# Action dispatch
# =========================================================================


class TestActionDispatch:
    def test_default_action_is_send(self):
        """No action specified defaults to 'send'."""
        result = _parse(send_message_tool({}))
        assert "error" in result
        assert "target" in result["error"]

    @patch("gateway.channel_directory.format_directory_for_display", return_value=[])
    def test_list_action(self, mock_dir):
        result = _parse(send_message_tool({"action": "list"}))
        assert "targets" in result

    @patch("gateway.channel_directory.format_directory_for_display",
           side_effect=RuntimeError("no dir"))
    def test_list_action_error(self, mock_dir):
        result = _parse(send_message_tool({"action": "list"}))
        assert "error" in result


# =========================================================================
# Interrupt handling
# =========================================================================


class TestInterruptHandling:
    @patch("tools.interrupt.is_interrupted", return_value=True)
    def test_interrupted_returns_error(self, mock_int):
        result = _parse(_handle_send({"target": "telegram:123", "message": "hi"}))
        assert "error" in result
        assert "Interrupted" in result["error"]


# =========================================================================
# _check_send_message (availability gate)
# =========================================================================


class TestCheckSendMessage:
    def test_messaging_platform_always_available(self, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
        assert _check_send_message() is True

    def test_messaging_platform_discord(self, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "discord")
        assert _check_send_message() is True

    def test_local_checks_gateway(self, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "local")
        with patch("gateway.status.is_gateway_running", return_value=True):
            assert _check_send_message() is True

    def test_local_no_gateway(self, monkeypatch):
        monkeypatch.setenv("HERMES_SESSION_PLATFORM", "local")
        with patch("gateway.status.is_gateway_running", return_value=False):
            assert _check_send_message() is False

    def test_no_platform_env_checks_gateway(self, monkeypatch):
        monkeypatch.delenv("HERMES_SESSION_PLATFORM", raising=False)
        with patch("gateway.status.is_gateway_running", return_value=False):
            assert _check_send_message() is False
