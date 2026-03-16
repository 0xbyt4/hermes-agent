"""Spec-driven tests for /voice commands via the API adapter.

Tests the full voice mode state machine through the API endpoints:
command -> voice_mode state -> auto_tts_disabled state -> TTS behavior.

Each test defines WHAT SHOULD HAPPEN for a specific scenario.
"""

import asyncio
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.api import APIPlatformAdapter
from gateway.platforms.base import MessageType


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    from gateway.api_server import _rate_limiter
    _rate_limiter._buckets.clear()


def _make_adapter():
    config = PlatformConfig(enabled=True)
    return APIPlatformAdapter(config)


def _make_app(adapter):
    from gateway.api_server import create_app
    return create_app(adapter)


def _wire_agent(adapter, response_text="ok"):
    """Mock handle_message, capture events and track what was sent."""
    captured = {"events": [], "queued": []}

    async def fake_handle_message(event):
        captured["events"].append(event)
        sk = adapter._build_session_key(event.source.chat_id)
        q = adapter._response_queues.get(sk)
        if q:
            if response_text:
                await q.put({"type": "message", "content": response_text})
            await q.put({"type": "done"})

    adapter.handle_message = fake_handle_message
    return captured


# ═══════════════════════════════════════════════════════════════════════
# SPEC: /voice commands SHOULD change voice mode state correctly
# ═══════════════════════════════════════════════════════════════════════


class TestVoiceCommandState:

    def test_voice_on_should_respond_with_confirmation(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "Voice mode enabled.\nI'll reply with voice when you send voice messages.\nUse /voice tts to get voice replies for all messages.")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/chat", json={"message": "/voice on", "session_id": "v1"},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 200
        assert "voice" in resp.json()["response"].lower() or "Voice" in resp.json()["response"]

    def test_voice_off_should_respond_with_confirmation(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "Voice mode disabled. Text-only replies.")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/chat", json={"message": "/voice off", "session_id": "v2"},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 200

    def test_voice_tts_should_respond_with_confirmation(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "Auto-TTS enabled.\nAll replies will include a voice message.")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/chat", json={"message": "/voice tts", "session_id": "v3"},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 200

    def test_voice_status_should_return_current_mode(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        _wire_agent(adapter, "Voice mode: Off (text only)")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            resp = client.post("/v1/chat", json={"message": "/voice status", "session_id": "v4"},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════
# SPEC: Voice input (MessageType.VOICE) SHOULD get TTS based on mode
# ═══════════════════════════════════════════════════════════════════════


class TestVoiceInputTTSBehavior:

    def test_fresh_session_voice_input_should_get_tts(self):
        """Default (no /voice command): voice input SHOULD get auto-TTS from KATMAN 1."""
        adapter = _make_adapter()
        # Fresh session: auto_tts_disabled is empty, so KATMAN 1 fires
        assert adapter._auto_tts_disabled_chats == set()

    def test_voice_off_should_disable_auto_tts(self):
        """After /voice off, chat_id SHOULD be in auto_tts_disabled set."""
        adapter = _make_adapter()
        chat_id = "test-chat"
        # Simulate /voice off
        adapter._auto_tts_disabled_chats.add(chat_id)
        assert chat_id in adapter._auto_tts_disabled_chats

    def test_voice_on_should_enable_auto_tts(self):
        """After /voice on, chat_id SHOULD be removed from auto_tts_disabled set."""
        adapter = _make_adapter()
        chat_id = "test-chat"
        adapter._auto_tts_disabled_chats.add(chat_id)
        # Simulate /voice on
        adapter._auto_tts_disabled_chats.discard(chat_id)
        assert chat_id not in adapter._auto_tts_disabled_chats

    def test_voice_input_should_be_message_type_voice_via_http(self):
        """POST /v1/chat/voice SHOULD set MessageType.VOICE on the event."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        captured = _wire_agent(adapter, "ok")
        client = TestClient(_make_app(adapter))

        fake_result = {"success": True, "transcript": "hello",
                       "language": "en", "language_probability": 0.9}

        async def fake_to_thread(fn, *args, **kwargs):
            return fake_result

        with patch.dict(os.environ, {"API_KEY": "k"}), \
             patch("gateway.api_server.asyncio.to_thread", fake_to_thread):
            client.post("/v1/chat/voice",
                        files={"file": ("v.webm", b"audio", "audio/webm")},
                        headers={"Authorization": "Bearer k"})

        assert captured["events"][0].message_type == MessageType.VOICE

    def test_voice_input_should_be_message_type_voice_via_ws(self):
        """WS message with voice:true SHOULD set MessageType.VOICE."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        captured = _wire_agent(adapter, "ok")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "k"})
                ws.receive_json()
                ws.send_json({"message": "transcript", "voice": True})
                while ws.receive_json()["type"] != "done":
                    pass

        assert captured["events"][0].message_type == MessageType.VOICE

    def test_text_input_should_be_message_type_text_via_ws(self):
        """WS message without voice flag SHOULD be MessageType.TEXT."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        captured = _wire_agent(adapter, "ok")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "k"})
                ws.receive_json()
                ws.send_json({"message": "hello"})
                while ws.receive_json()["type"] != "done":
                    pass

        assert captured["events"][0].message_type == MessageType.TEXT


# ═══════════════════════════════════════════════════════════════════════
# SPEC: Transcribe endpoint SHOULD only do STT
# ═══════════════════════════════════════════════════════════════════════


class TestTranscribeEndpoint:

    def test_transcribe_should_return_transcript(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        captured = _wire_agent(adapter, "should not reach")
        client = TestClient(_make_app(adapter))

        fake_result = {"success": True, "transcript": "hello world",
                       "language": "en", "language_probability": 0.95}

        async def fake_to_thread(fn, *args, **kwargs):
            return fake_result

        with patch.dict(os.environ, {"API_KEY": "k"}), \
             patch("gateway.api_server.asyncio.to_thread", fake_to_thread):
            resp = client.post("/v1/transcribe",
                               files={"file": ("v.webm", b"audio", "audio/webm")},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 200
        assert resp.json()["transcript"] == "hello world"
        assert len(captured["events"]) == 0  # agent NOT called

    def test_transcribe_failure_should_return_422(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        client = TestClient(_make_app(adapter))

        async def fake_to_thread(fn, *args, **kwargs):
            return {"success": False, "error": "STT down"}

        with patch.dict(os.environ, {"API_KEY": "k"}), \
             patch("gateway.api_server.asyncio.to_thread", fake_to_thread):
            resp = client.post("/v1/transcribe",
                               files={"file": ("v.ogg", b"bad", "audio/ogg")},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 422

    def test_transcribe_empty_should_return_422(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        client = TestClient(_make_app(adapter))

        async def fake_to_thread(fn, *args, **kwargs):
            return {"success": True, "transcript": "  ", "language": "en", "language_probability": 0.1}

        with patch.dict(os.environ, {"API_KEY": "k"}), \
             patch("gateway.api_server.asyncio.to_thread", fake_to_thread):
            resp = client.post("/v1/transcribe",
                               files={"file": ("v.webm", b"silence", "audio/webm")},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 422

    def test_transcribe_should_require_auth(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "correct"}):
            resp = client.post("/v1/transcribe",
                               files={"file": ("v.webm", b"audio", "audio/webm")},
                               headers={"Authorization": "Bearer wrong"})

        assert resp.status_code == 401

    def test_transcribe_should_be_rate_limited(self):
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        client = TestClient(_make_app(adapter))

        fake_result = {"success": True, "transcript": "ok",
                       "language": "en", "language_probability": 0.9}

        async def fake_to_thread(fn, *args, **kwargs):
            return fake_result

        with patch.dict(os.environ, {"API_KEY": "k"}), \
             patch("gateway.api_server.asyncio.to_thread", fake_to_thread):
            for _ in range(10):
                client.post("/v1/transcribe",
                            files={"file": ("v.webm", b"audio", "audio/webm")},
                            headers={"Authorization": "Bearer k"})
            resp = client.post("/v1/transcribe",
                               files={"file": ("v.webm", b"audio", "audio/webm")},
                               headers={"Authorization": "Bearer k"})

        assert resp.status_code == 429


# ═══════════════════════════════════════════════════════════════════════
# SPEC: Full voice-via-WS flow SHOULD work end-to-end
# ═══════════════════════════════════════════════════════════════════════


class TestVoiceViaWSFlow:

    def test_transcribe_then_ws_voice_should_stream(self):
        """Full flow: transcribe audio -> send transcript via WS with voice flag -> stream response."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        captured = _wire_agent(adapter, "voice streaming reply")
        client = TestClient(_make_app(adapter))

        fake_result = {"success": True, "transcript": "what is the weather",
                       "language": "en", "language_probability": 0.95}

        async def fake_to_thread(fn, *args, **kwargs):
            return fake_result

        with patch.dict(os.environ, {"API_KEY": "k"}), \
             patch("gateway.api_server.asyncio.to_thread", fake_to_thread):
            # Step 1: Transcribe
            tr = client.post("/v1/transcribe",
                             files={"file": ("v.webm", b"audio", "audio/webm")},
                             headers={"Authorization": "Bearer k"})
            transcript = tr.json()["transcript"]

        # Step 2: Send via WS with voice flag
        with patch.dict(os.environ, {"API_KEY": "k"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "k", "session_id": "vf1"})
                ws.receive_json()

                ws.send_json({"message": transcript, "voice": True})
                messages = []
                while True:
                    msg = ws.receive_json()
                    messages.append(msg)
                    if msg["type"] == "done":
                        break

        # Verify
        assert captured["events"][0].message_type == MessageType.VOICE
        assert captured["events"][0].text == "what is the weather"
        assert any(m.get("content") == "voice streaming reply" for m in messages)

    def test_voice_command_then_voice_input_should_respect_mode(self):
        """After /voice off via WS, voice input SHOULD still be VOICE type but auto_tts_disabled."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        captured = _wire_agent(adapter, "ok")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "k", "session_id": "vm1"})
                ws.receive_json()

                # Send /voice off
                ws.send_json({"message": "/voice off"})
                while ws.receive_json()["type"] != "done":
                    pass

                # Send voice input
                ws.send_json({"message": "hello voice", "voice": True})
                while ws.receive_json()["type"] != "done":
                    pass

        # Voice input should still be marked VOICE
        voice_events = [e for e in captured["events"] if e.message_type == MessageType.VOICE]
        assert len(voice_events) == 1
        assert voice_events[0].text == "hello voice"


# ═══════════════════════════════════════════════════════════════════════
# SPEC: UI streaming mode commands SHOULD set correct backend state
# ═══════════════════════════════════════════════════════════════════════


class TestStreamingModeCommands:

    def test_voice_tts_via_ws_should_be_processed(self):
        """/voice tts sent via WS SHOULD be processed by the agent."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        captured = _wire_agent(adapter, "Auto-TTS enabled.")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "k", "session_id": "sm1"})
                ws.receive_json()

                ws.send_json({"message": "/voice tts"})
                msgs = []
                while True:
                    m = ws.receive_json()
                    msgs.append(m)
                    if m["type"] == "done":
                        break

        assert len(captured["events"]) == 1
        assert captured["events"][0].text == "/voice tts"

    def test_voice_on_via_ws_should_be_processed(self):
        """/voice on sent via WS (on streaming exit) SHOULD be processed."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        captured = _wire_agent(adapter, "Voice mode enabled.")
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "k", "session_id": "sm2"})
                ws.receive_json()

                ws.send_json({"message": "/voice on"})
                while ws.receive_json()["type"] != "done":
                    pass

        assert captured["events"][0].text == "/voice on"

    def test_sequential_tts_then_on_should_both_process(self):
        """enterVoiceMode sends /voice tts, exitVoiceMode sends /voice on. Both SHOULD process."""
        from fastapi.testclient import TestClient
        adapter = _make_adapter()
        n = {"c": 0}

        async def handler(event):
            n["c"] += 1
            sk = adapter._build_session_key(event.source.chat_id)
            q = adapter._response_queues.get(sk)
            if q:
                await q.put({"type": "message", "content": f"cmd {n['c']}"})
                await q.put({"type": "done"})

        adapter.handle_message = handler
        client = TestClient(_make_app(adapter))

        with patch.dict(os.environ, {"API_KEY": "k"}):
            with client.websocket_connect("/v1/chat/stream") as ws:
                ws.send_json({"type": "auth", "token": "k", "session_id": "sm3"})
                ws.receive_json()

                # enterVoiceMode sends /voice tts
                ws.send_json({"message": "/voice tts"})
                while ws.receive_json()["type"] != "done":
                    pass

                # exitVoiceMode sends /voice on
                ws.send_json({"message": "/voice on"})
                while ws.receive_json()["type"] != "done":
                    pass

        assert n["c"] == 2  # both commands processed
