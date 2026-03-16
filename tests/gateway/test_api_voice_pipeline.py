"""Full pipeline tests: API adapter + GatewayRunner voice mode.

Tests the real voice command processing through the runner,
then verifies the adapter's auto_tts_disabled state is correct
for subsequent voice/text input.

Unlike unit tests that mock handle_message, these use the real runner's
_handle_voice_command and _set_adapter_auto_tts_disabled to verify
the full state machine.
"""

import asyncio
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Mock discord before importing gateway
def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return
    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, danger=3, green=1, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    discord_mod.opus = SimpleNamespace(is_loaded=lambda: True, load_opus=lambda *a, **k: None)
    discord_mod.FFmpegPCMAudio = MagicMock
    discord_mod.PCMVolumeTransformer = MagicMock
    discord_mod.http = SimpleNamespace(Route=MagicMock)
    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod
    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)

_ensure_discord_mock()

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api import APIPlatformAdapter
from gateway.platforms.base import MessageEvent, MessageType, SessionSource


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_adapter():
    config = PlatformConfig(enabled=True)
    return APIPlatformAdapter(config)


def _make_runner(tmp_path, adapter):
    """Create a real GatewayRunner with the API adapter wired in."""
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.API: adapter}
    runner._voice_mode = {}
    runner._VOICE_MODE_PATH = tmp_path / "voice_mode.json"
    runner._session_db = None
    runner.session_store = MagicMock()
    runner._is_user_authorized = lambda source: True
    return runner


def _make_api_event(text, chat_id="api-chat-1", msg_type=MessageType.TEXT):
    source = SessionSource(
        chat_id=chat_id, user_id=chat_id,
        platform=Platform.API, chat_type="channel",
    )
    event = MessageEvent(text=text, message_type=msg_type, source=source)
    event.message_id = "msg-1"
    return event


# ═══════════════════════════════════════════════════════════════════════
# SPEC: /voice command via API SHOULD update adapter's auto_tts state
# ═══════════════════════════════════════════════════════════════════════


class TestVoiceCommandUpdatesAdapter:
    """Real runner processes /voice command -> adapter state changes."""

    @pytest.fixture
    def setup(self, tmp_path):
        adapter = _make_adapter()
        runner = _make_runner(tmp_path, adapter)
        return runner, adapter

    @pytest.mark.asyncio
    async def test_voice_off_should_add_to_disabled(self, setup):
        runner, adapter = setup
        event = _make_api_event("/voice off")
        await runner._handle_voice_command(event)
        assert "api-chat-1" in adapter._auto_tts_disabled_chats

    @pytest.mark.asyncio
    async def test_voice_on_should_remove_from_disabled(self, setup):
        runner, adapter = setup
        adapter._auto_tts_disabled_chats.add("api-chat-1")
        event = _make_api_event("/voice on")
        await runner._handle_voice_command(event)
        assert "api-chat-1" not in adapter._auto_tts_disabled_chats

    @pytest.mark.asyncio
    async def test_voice_tts_should_remove_from_disabled(self, setup):
        runner, adapter = setup
        adapter._auto_tts_disabled_chats.add("api-chat-1")
        event = _make_api_event("/voice tts")
        await runner._handle_voice_command(event)
        assert "api-chat-1" not in adapter._auto_tts_disabled_chats

    @pytest.mark.asyncio
    async def test_voice_off_should_set_mode_off(self, setup):
        runner, adapter = setup
        event = _make_api_event("/voice off")
        await runner._handle_voice_command(event)
        assert runner._voice_mode["api-chat-1"] == "off"

    @pytest.mark.asyncio
    async def test_voice_on_should_set_mode_voice_only(self, setup):
        runner, adapter = setup
        event = _make_api_event("/voice on")
        await runner._handle_voice_command(event)
        assert runner._voice_mode["api-chat-1"] == "voice_only"

    @pytest.mark.asyncio
    async def test_voice_tts_should_set_mode_all(self, setup):
        runner, adapter = setup
        event = _make_api_event("/voice tts")
        await runner._handle_voice_command(event)
        assert runner._voice_mode["api-chat-1"] == "all"


# ═══════════════════════════════════════════════════════════════════════
# SPEC: Full state machine - command sequence -> TTS decision
# ═══════════════════════════════════════════════════════════════════════


class TestFullVoiceStateMachine:
    """Test command sequences and verify _should_send_voice_reply decision."""

    @pytest.fixture
    def setup(self, tmp_path):
        adapter = _make_adapter()
        runner = _make_runner(tmp_path, adapter)
        return runner, adapter

    def _should_tts(self, runner, msg_type=MessageType.VOICE, chat_id="api-chat-1"):
        event = _make_api_event("test", chat_id=chat_id, msg_type=msg_type)
        return runner._should_send_voice_reply(event, "Hello!", [])

    # ── Streaming mode enter/exit cycle ──────────────────────────────

    @pytest.mark.asyncio
    async def test_enter_streaming_text_should_get_tts(self, setup):
        """enterVoiceMode sends /voice tts -> text input SHOULD get TTS from runner."""
        runner, adapter = setup
        await runner._handle_voice_command(_make_api_event("/voice tts"))
        assert self._should_tts(runner, MessageType.TEXT) is True

    @pytest.mark.asyncio
    async def test_enter_streaming_voice_should_skip_runner(self, setup):
        """enterVoiceMode /voice tts -> voice input: runner skips (K1 handles)."""
        runner, adapter = setup
        await runner._handle_voice_command(_make_api_event("/voice tts"))
        assert self._should_tts(runner, MessageType.VOICE) is False
        # But K1 fires because not disabled
        assert "api-chat-1" not in adapter._auto_tts_disabled_chats

    @pytest.mark.asyncio
    async def test_exit_streaming_voice_should_still_get_k1_tts(self, setup):
        """exitVoiceMode sends /voice on -> voice input SHOULD still get K1 TTS."""
        runner, adapter = setup
        await runner._handle_voice_command(_make_api_event("/voice tts"))
        await runner._handle_voice_command(_make_api_event("/voice on"))
        # voice_only mode: runner skips voice input (dedup with K1)
        assert self._should_tts(runner, MessageType.VOICE) is False
        # K1 fires: not disabled
        assert "api-chat-1" not in adapter._auto_tts_disabled_chats

    @pytest.mark.asyncio
    async def test_exit_streaming_text_should_not_get_tts(self, setup):
        """exitVoiceMode /voice on -> text input: NO TTS (voice_only doesn't cover text)."""
        runner, adapter = setup
        await runner._handle_voice_command(_make_api_event("/voice tts"))
        await runner._handle_voice_command(_make_api_event("/voice on"))
        assert self._should_tts(runner, MessageType.TEXT) is False

    # ── Full cycle: enter -> exit -> push-to-talk ────────────────────

    @pytest.mark.asyncio
    async def test_full_cycle_push_to_talk_after_streaming(self, setup):
        """Enter streaming -> exit -> push-to-talk SHOULD still get TTS."""
        runner, adapter = setup
        # Enter streaming
        await runner._handle_voice_command(_make_api_event("/voice tts"))
        assert runner._voice_mode["api-chat-1"] == "all"
        # Exit streaming
        await runner._handle_voice_command(_make_api_event("/voice on"))
        assert runner._voice_mode["api-chat-1"] == "voice_only"
        # Push-to-talk (voice input)
        assert "api-chat-1" not in adapter._auto_tts_disabled_chats  # K1 will fire

    @pytest.mark.asyncio
    async def test_full_cycle_off_then_streaming(self, setup):
        """Explicit /voice off -> enter streaming -> TTS SHOULD work."""
        runner, adapter = setup
        await runner._handle_voice_command(_make_api_event("/voice off"))
        assert "api-chat-1" in adapter._auto_tts_disabled_chats
        # Enter streaming
        await runner._handle_voice_command(_make_api_event("/voice tts"))
        assert "api-chat-1" not in adapter._auto_tts_disabled_chats
        assert runner._voice_mode["api-chat-1"] == "all"

    @pytest.mark.asyncio
    async def test_off_push_to_talk_should_not_get_tts(self, setup):
        """Explicit /voice off -> push-to-talk: NO TTS at all."""
        runner, adapter = setup
        await runner._handle_voice_command(_make_api_event("/voice off"))
        # K1: disabled -> skip
        assert "api-chat-1" in adapter._auto_tts_disabled_chats
        # K2: off -> skip
        assert self._should_tts(runner, MessageType.VOICE) is False

    # ── Per-chat isolation ───────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_per_chat_isolation(self, setup):
        """Different chat_ids SHOULD have independent voice mode state."""
        runner, adapter = setup
        await runner._handle_voice_command(_make_api_event("/voice tts", chat_id="chat-a"))
        await runner._handle_voice_command(_make_api_event("/voice off", chat_id="chat-b"))

        assert runner._voice_mode["chat-a"] == "all"
        assert runner._voice_mode["chat-b"] == "off"
        assert "chat-a" not in adapter._auto_tts_disabled_chats
        assert "chat-b" in adapter._auto_tts_disabled_chats

    # ── Persistence ──────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_voice_mode_persists_and_restores(self, setup):
        """Voice mode SHOULD persist to disk and restore on restart."""
        runner, adapter = setup
        await runner._handle_voice_command(_make_api_event("/voice off"))
        await runner._handle_voice_command(_make_api_event("/voice tts", chat_id="chat-2"))

        # Verify persisted
        data = json.loads(runner._VOICE_MODE_PATH.read_text())
        assert data["api-chat-1"] == "off"
        assert data["chat-2"] == "all"

        # Simulate restart: create new adapter, sync state
        new_adapter = _make_adapter()
        runner._voice_mode = json.loads(runner._VOICE_MODE_PATH.read_text())
        runner.adapters[Platform.API] = new_adapter
        runner._sync_voice_mode_state_to_adapter(new_adapter)

        # /voice off chat should be disabled
        assert "api-chat-1" in new_adapter._auto_tts_disabled_chats
        # /voice tts chat should NOT be disabled
        assert "chat-2" not in new_adapter._auto_tts_disabled_chats
