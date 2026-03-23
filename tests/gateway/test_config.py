"""Tests for gateway configuration management."""

import os
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from gateway.config import (
    GatewayConfig,
    HomeChannel,
    Platform,
    PlatformConfig,
    SessionResetPolicy,
    load_gateway_config,
    _apply_env_overrides,
)


class TestHomeChannelRoundtrip:
    def test_to_dict_from_dict(self):
        hc = HomeChannel(platform=Platform.DISCORD, chat_id="999", name="general")
        d = hc.to_dict()
        restored = HomeChannel.from_dict(d)

        assert restored.platform == Platform.DISCORD
        assert restored.chat_id == "999"
        assert restored.name == "general"


class TestPlatformConfigRoundtrip:
    def test_to_dict_from_dict(self):
        pc = PlatformConfig(
            enabled=True,
            token="tok_123",
            home_channel=HomeChannel(
                platform=Platform.TELEGRAM,
                chat_id="555",
                name="Home",
            ),
            extra={"foo": "bar"},
        )
        d = pc.to_dict()
        restored = PlatformConfig.from_dict(d)

        assert restored.enabled is True
        assert restored.token == "tok_123"
        assert restored.home_channel.chat_id == "555"
        assert restored.extra == {"foo": "bar"}

    def test_disabled_no_token(self):
        pc = PlatformConfig()
        d = pc.to_dict()
        restored = PlatformConfig.from_dict(d)
        assert restored.enabled is False
        assert restored.token is None


class TestGetConnectedPlatforms:
    def test_returns_enabled_with_token(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(enabled=True, token="t"),
                Platform.DISCORD: PlatformConfig(enabled=False, token="d"),
                Platform.SLACK: PlatformConfig(enabled=True),  # no token
            },
        )
        connected = config.get_connected_platforms()
        assert Platform.TELEGRAM in connected
        assert Platform.DISCORD not in connected
        assert Platform.SLACK not in connected

    def test_empty_platforms(self):
        config = GatewayConfig()
        assert config.get_connected_platforms() == []


class TestSessionResetPolicy:
    def test_roundtrip(self):
        policy = SessionResetPolicy(mode="idle", at_hour=6, idle_minutes=120)
        d = policy.to_dict()
        restored = SessionResetPolicy.from_dict(d)
        assert restored.mode == "idle"
        assert restored.at_hour == 6
        assert restored.idle_minutes == 120

    def test_defaults(self):
        policy = SessionResetPolicy()
        assert policy.mode == "both"
        assert policy.at_hour == 4
        assert policy.idle_minutes == 1440

    def test_from_dict_treats_null_values_as_defaults(self):
        restored = SessionResetPolicy.from_dict(
            {"mode": None, "at_hour": None, "idle_minutes": None}
        )
        assert restored.mode == "both"
        assert restored.at_hour == 4
        assert restored.idle_minutes == 1440


class TestGatewayConfigRoundtrip:
    def test_full_roundtrip(self):
        config = GatewayConfig(
            platforms={
                Platform.TELEGRAM: PlatformConfig(
                    enabled=True,
                    token="tok_123",
                    home_channel=HomeChannel(Platform.TELEGRAM, "123", "Home"),
                ),
            },
            reset_triggers=["/new"],
            quick_commands={"limits": {"type": "exec", "command": "echo ok"}},
            group_sessions_per_user=False,
        )
        d = config.to_dict()
        restored = GatewayConfig.from_dict(d)

        assert Platform.TELEGRAM in restored.platforms
        assert restored.platforms[Platform.TELEGRAM].token == "tok_123"
        assert restored.reset_triggers == ["/new"]
        assert restored.quick_commands == {"limits": {"type": "exec", "command": "echo ok"}}
        assert restored.group_sessions_per_user is False

    def test_roundtrip_preserves_unauthorized_dm_behavior(self):
        config = GatewayConfig(
            unauthorized_dm_behavior="ignore",
            platforms={
                Platform.WHATSAPP: PlatformConfig(
                    enabled=True,
                    extra={"unauthorized_dm_behavior": "pair"},
                ),
            },
        )

        restored = GatewayConfig.from_dict(config.to_dict())

        assert restored.unauthorized_dm_behavior == "ignore"
        assert restored.platforms[Platform.WHATSAPP].extra["unauthorized_dm_behavior"] == "pair"


class TestLoadGatewayConfig:
    def test_bridges_quick_commands_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "quick_commands:\n"
            "  limits:\n"
            "    type: exec\n"
            "    command: echo ok\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.quick_commands == {"limits": {"type": "exec", "command": "echo ok"}}

    def test_bridges_group_sessions_per_user_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("group_sessions_per_user: false\n", encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.group_sessions_per_user is False

    def test_invalid_quick_commands_in_config_yaml_are_ignored(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text("quick_commands: not-a-mapping\n", encoding="utf-8")

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.quick_commands == {}

    def test_bridges_unauthorized_dm_behavior_from_config_yaml(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / ".hermes"
        hermes_home.mkdir()
        config_path = hermes_home / "config.yaml"
        config_path.write_text(
            "unauthorized_dm_behavior: ignore\n"
            "whatsapp:\n"
            "  unauthorized_dm_behavior: pair\n",
            encoding="utf-8",
        )

        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        config = load_gateway_config()

        assert config.unauthorized_dm_behavior == "ignore"
        assert config.platforms[Platform.WHATSAPP].extra["unauthorized_dm_behavior"] == "pair"


# ---------------------------------------------------------------------------
# /sethome persistence tests
# ---------------------------------------------------------------------------


class TestSethomeEnvPersistence:
    """Verify /sethome writes to .env and survives restart."""

    def test_env_override_loads_home_channel(self, monkeypatch):
        """When HOME_CHANNEL env var is set, gateway config picks it up."""
        monkeypatch.setenv("DISCORD_HOME_CHANNEL", "123456789")
        config = GatewayConfig()
        config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True)
        _apply_env_overrides(config)
        assert config.platforms[Platform.DISCORD].home_channel is not None
        assert config.platforms[Platform.DISCORD].home_channel.chat_id == "123456789"

    def test_signal_home_channel_env(self, monkeypatch):
        """SIGNAL_HOME_CHANNEL env var should set Signal home channel."""
        monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:8080")
        monkeypatch.setenv("SIGNAL_ACCOUNT", "+1999999999")
        monkeypatch.setenv("SIGNAL_HOME_CHANNEL", "+1234567890")
        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.SIGNAL in config.platforms
        assert config.platforms[Platform.SIGNAL].home_channel is not None
        assert config.platforms[Platform.SIGNAL].home_channel.chat_id == "+1234567890"

    def test_telegram_home_channel_env(self, monkeypatch):
        """TELEGRAM_HOME_CHANNEL env var should set Telegram home channel."""
        monkeypatch.setenv("TELEGRAM_HOME_CHANNEL", "987654321")
        config = GatewayConfig()
        config.platforms[Platform.TELEGRAM] = PlatformConfig(enabled=True)
        _apply_env_overrides(config)
        assert config.platforms[Platform.TELEGRAM].home_channel is not None
        assert config.platforms[Platform.TELEGRAM].home_channel.chat_id == "987654321"

    def test_missing_env_no_home_channel(self, monkeypatch):
        """Without env var, no home channel is set."""
        monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)
        config = GatewayConfig()
        config.platforms[Platform.DISCORD] = PlatformConfig(enabled=True)
        _apply_env_overrides(config)
        assert config.platforms[Platform.DISCORD].home_channel is None

    def test_sethome_writes_to_env_file(self, tmp_path, monkeypatch):
        """Verify save_env_value writes HOME_CHANNEL to .env file."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("")

        from hermes_cli.config import save_env_value
        save_env_value("DISCORD_HOME_CHANNEL", "111222333")

        env_content = (tmp_path / ".env").read_text()
        assert "DISCORD_HOME_CHANNEL=111222333" in env_content

    def test_env_file_survives_reload(self, tmp_path, monkeypatch):
        """Written .env value is available after reload."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("")

        from hermes_cli.config import save_env_value
        save_env_value("TELEGRAM_HOME_CHANNEL", "444555666")

        # Clear env and reload
        monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)
        assert os.getenv("TELEGRAM_HOME_CHANNEL") is None

        from hermes_cli.env_loader import load_hermes_dotenv
        load_hermes_dotenv()
        assert os.getenv("TELEGRAM_HOME_CHANNEL") == "444555666"

    def test_signal_in_extra_env_keys(self):
        """SIGNAL_HOME_CHANNEL should be in _EXTRA_ENV_KEYS."""
        from hermes_cli.config import _EXTRA_ENV_KEYS
        assert "SIGNAL_HOME_CHANNEL" in _EXTRA_ENV_KEYS
        assert "SLACK_HOME_CHANNEL" in _EXTRA_ENV_KEYS
