"""Tests for the dream engine — 5-stage memory processing pipeline."""

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tools.dream_engine import (
    DEFAULT_DREAM_CONFIG,
    DreamEngine,
    DreamState,
    get_dream_dir,
    load_dream_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_hermes(tmp_path, monkeypatch):
    """Set up a temporary HERMES_HOME with required directories."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "memories").mkdir()
    (tmp_path / "dreams").mkdir()
    return tmp_path


@pytest.fixture
def sample_state_db(tmp_hermes):
    """Create a minimal state.db with test sessions and messages."""
    db_path = tmp_hermes / "state.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY, source TEXT NOT NULL, title TEXT,
            message_count INTEGER DEFAULT 0, tool_call_count INTEGER DEFAULT 0,
            started_at REAL NOT NULL, ended_at REAL, end_reason TEXT,
            user_id TEXT, model TEXT, model_config TEXT, system_prompt TEXT,
            parent_session_id TEXT, input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0, cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0, reasoning_tokens INTEGER DEFAULT 0,
            billing_provider TEXT, billing_base_url TEXT, billing_mode TEXT,
            estimated_cost_usd REAL, actual_cost_usd REAL,
            cost_status TEXT, cost_source TEXT, pricing_version TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
            role TEXT NOT NULL, content TEXT, tool_call_id TEXT,
            tool_calls TEXT, tool_name TEXT, timestamp REAL NOT NULL,
            token_count INTEGER, finish_reason TEXT, reasoning TEXT,
            reasoning_details TEXT, codex_reasoning_items TEXT
        )
    """)
    # We also need the schema_version table
    conn.execute("CREATE TABLE schema_version (version INTEGER)")
    conn.execute("INSERT INTO schema_version VALUES (1)")

    now = datetime.now().timestamp()
    hour_ago = now - 3600
    two_hours_ago = now - 7200

    # Session 1: recent coding session
    conn.execute(
        "INSERT INTO sessions (id, source, title, message_count, tool_call_count, started_at, ended_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("session_001", "cli", "Debugging auth flow", 10, 5, hour_ago, now),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session_001", "user", "Fix the login bug in auth.py", hour_ago),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session_001", "user", "Also check the session timeout", hour_ago + 60),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session_001", "assistant", "Fixed the auth bug and session timeout.", now),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, tool_name, timestamp) VALUES (?, ?, ?, ?, ?)",
        ("session_001", "tool", "File contents...", "read_file", hour_ago + 30),
    )

    # Session 2: older session
    conn.execute(
        "INSERT INTO sessions (id, source, title, message_count, tool_call_count, started_at, ended_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("session_002", "discord", "Setting up CI pipeline", 15, 8, two_hours_ago, hour_ago),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session_002", "user", "Configure GitHub Actions for tests", two_hours_ago),
    )
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        ("session_002", "assistant", "CI pipeline configured with test and lint steps.", hour_ago),
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def engine(tmp_hermes):
    """Create a DreamEngine with test config."""
    config = dict(DEFAULT_DREAM_CONFIG)
    config["enabled"] = True
    config["sessions_to_process"] = 4
    return DreamEngine(config)


# ---------------------------------------------------------------------------
# DreamState tests
# ---------------------------------------------------------------------------


class TestDreamState:
    def test_load_empty(self, tmp_hermes):
        state = DreamState(tmp_hermes / "dreams")
        data = state.load()
        assert data["last_processed_session"] is None
        assert data["dream_count"] == 0

    def test_save_and_load(self, tmp_hermes):
        state = DreamState(tmp_hermes / "dreams")
        state.save({
            "last_processed_session": "session_001",
            "last_dream_at": "2026-04-06T12:00:00",
            "dream_count": 3,
        })
        data = state.load()
        assert data["last_processed_session"] == "session_001"
        assert data["dream_count"] == 3

    def test_atomic_write(self, tmp_hermes):
        """Verify state file exists after save (atomic rename)."""
        state = DreamState(tmp_hermes / "dreams")
        state.save({"dream_count": 1, "last_processed_session": None, "last_dream_at": None})
        assert (tmp_hermes / "dreams" / "state.json").exists()


# ---------------------------------------------------------------------------
# Harvest tests
# ---------------------------------------------------------------------------


class TestHarvest:
    def test_harvest_returns_digests(self, engine, sample_state_db):
        """Harvest should return session digests from state.db."""
        mock_db = MagicMock()
        mock_db._conn = sqlite3.connect(str(sample_state_db))
        mock_db.close = MagicMock()

        with patch("hermes_state.SessionDB", return_value=mock_db):
            digests = engine.harvest()

        assert len(digests) == 2
        assert digests[0]["session_id"] == "session_001"  # newest first
        assert digests[0]["title"] == "Debugging auth flow"
        assert digests[0]["platform"] == "cli"
        assert len(digests[0]["user_messages"]) == 2
        assert "login bug" in digests[0]["user_messages"][0]
        assert "read_file" in digests[0]["tools_used"]

    def test_harvest_respects_cursor(self, engine, sample_state_db):
        """Harvest should only return sessions newer than cursor."""
        engine.state.save({
            "last_processed_session": "session_002",
            "last_dream_at": None,
            "dream_count": 0,
        })

        mock_db = MagicMock()
        mock_db._conn = sqlite3.connect(str(sample_state_db))
        mock_db.close = MagicMock()

        with patch("hermes_state.SessionDB", return_value=mock_db):
            digests = engine.harvest()

        assert len(digests) == 1
        assert digests[0]["session_id"] == "session_001"

    def test_harvest_empty_db(self, engine, tmp_hermes):
        """Harvest should return empty list when no sessions exist."""
        db_path = tmp_hermes / "state.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY, source TEXT NOT NULL, title TEXT,
                message_count INTEGER DEFAULT 0, tool_call_count INTEGER DEFAULT 0,
                started_at REAL NOT NULL, ended_at REAL, end_reason TEXT,
                user_id TEXT, model TEXT, model_config TEXT, system_prompt TEXT,
                parent_session_id TEXT, input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0, cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0, reasoning_tokens INTEGER DEFAULT 0,
                billing_provider TEXT, billing_base_url TEXT, billing_mode TEXT,
                estimated_cost_usd REAL, actual_cost_usd REAL,
                cost_status TEXT, cost_source TEXT, pricing_version TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT, tool_call_id TEXT,
                tool_calls TEXT, tool_name TEXT, timestamp REAL NOT NULL,
                token_count INTEGER, finish_reason TEXT, reasoning TEXT,
                reasoning_details TEXT, codex_reasoning_items TEXT
            )
        """)
        conn.execute("CREATE TABLE schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version VALUES (1)")
        conn.commit()

        mock_db = MagicMock()
        mock_db._conn = sqlite3.connect(str(db_path))
        mock_db.close = MagicMock()

        with patch("hermes_state.SessionDB", return_value=mock_db):
            digests = engine.harvest()

        assert digests == []


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    def test_analysis_prompt_includes_sessions(self, engine):
        digests = [
            {
                "session_id": "s1", "platform": "cli", "title": "Test session",
                "message_count": 5, "tool_call_count": 2,
                "started_at": "2026-04-06 12:00", "ended_at": "2026-04-06 12:30",
                "end_reason": None, "tools_used": ["read_file"],
                "user_messages": ["Fix the bug"], "last_response": "Bug fixed.",
            }
        ]
        prompt = engine.build_analysis_prompt(digests, "existing memory", "user profile")
        assert "Test session" in prompt
        assert "Fix the bug" in prompt
        assert "existing memory" in prompt
        assert "user profile" in prompt
        assert "CONSOLIDATE" in prompt
        assert "CONNECT" in prompt
        assert "json" in prompt.lower()

    def test_analysis_prompt_handles_empty_memory(self, engine):
        prompt = engine.build_analysis_prompt([], "", "")
        assert "(empty)" in prompt

    def test_creative_prompt_includes_patterns(self, engine):
        analysis = {
            "patterns": ["User works late nights", "Focus on security"],
            "open_threads": ["PR review pending"],
            "session_summary": "Worked on auth and CI.",
        }
        prompt = engine.build_creative_prompt(analysis, "agent memory")
        assert "late nights" in prompt
        assert "security" in prompt
        assert "PR review" in prompt
        assert "auth and CI" in prompt
        assert "agent memory" in prompt


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_parse_valid_json(self, engine):
        response = '```json\n{"insights": [], "patterns": ["p1"], "open_threads": [], "session_summary": "test"}\n```'
        result = engine.parse_analysis_response(response)
        assert result["patterns"] == ["p1"]
        assert result["session_summary"] == "test"

    def test_parse_raw_json(self, engine):
        response = '{"insights": [], "patterns": [], "open_threads": ["t1"], "session_summary": "s"}'
        result = engine.parse_analysis_response(response)
        assert result["open_threads"] == ["t1"]

    def test_parse_json_in_text(self, engine):
        response = 'Here is my analysis:\n{"insights": [], "patterns": ["p"], "open_threads": [], "session_summary": "x"}\nThat is all.'
        result = engine.parse_analysis_response(response)
        assert result["patterns"] == ["p"]

    def test_parse_invalid_returns_fallback(self, engine):
        response = "This is not JSON at all, just plain text analysis."
        result = engine.parse_analysis_response(response)
        assert "session_summary" in result
        assert isinstance(result["insights"], list)

    def test_parse_empty_response(self, engine):
        result = engine.parse_analysis_response("")
        assert result["insights"] == []
        assert result["patterns"] == []


# ---------------------------------------------------------------------------
# Journal tests
# ---------------------------------------------------------------------------


class TestJournal:
    def test_write_journal_creates_file(self, engine, tmp_hermes):
        digests = [
            {
                "session_id": "s1", "platform": "cli", "title": "Test",
                "message_count": 5, "tool_call_count": 2,
            }
        ]
        analysis = {
            "session_summary": "Did some testing.",
            "patterns": ["Pattern A"],
            "open_threads": ["Thread 1"],
            "insights": ["New fact discovered about auth flow"],
        }
        path = engine.write_journal(digests, analysis, "I dreamed about code.")
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Dream" in content
        assert "Did some testing" in content
        assert "Pattern A" in content
        assert "Thread 1" in content
        assert "I dreamed about code" in content
        assert "Test" in content  # session title

    def test_advance_cursor(self, engine, tmp_hermes):
        digests = [{"session_id": "session_newest"}, {"session_id": "session_older"}]
        engine.advance_cursor(digests)
        state = engine.state.load()
        assert state["last_processed_session"] == "session_newest"
        assert state["dream_count"] == 1
        assert state["last_dream_at"] is not None


# ---------------------------------------------------------------------------
# Memory update tests
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Status and history tests
# ---------------------------------------------------------------------------


class TestStatusHistory:
    def test_get_status(self, engine, tmp_hermes):
        status = engine.get_status()
        assert "enabled" in status
        assert "model" in status
        assert "dream_count" in status
        assert status["dream_count"] == 0

    def test_list_dreams_empty(self, engine, tmp_hermes):
        dreams = engine.list_dreams()
        assert dreams == []

    def test_list_dreams_returns_logs(self, engine, tmp_hermes):
        dream_dir = tmp_hermes / "dreams"
        (dream_dir / "dream_20260406_120000.md").write_text(
            "# Dream\n\nSome dream content.", encoding="utf-8"
        )
        (dream_dir / "dream_20260405_120000.md").write_text(
            "# Dream\n\nOlder dream.", encoding="utf-8"
        )
        dreams = engine.list_dreams()
        assert len(dreams) == 2
        assert "2026-04-06" in dreams[0]["date"]  # newest first


# ---------------------------------------------------------------------------
# Full pipeline test (mocked LLM)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_run_with_no_sessions_returns_none(self, engine, tmp_hermes):
        with patch.object(engine, "harvest", return_value=[]):
            result = engine.run()
        assert result is None

    def test_run_full_pipeline(self, engine, tmp_hermes):
        """Test the complete pipeline with mocked LLM calls."""
        mem_dir = tmp_hermes / "memories"
        (mem_dir / "MEMORY.md").write_text("old memory", encoding="utf-8")
        (mem_dir / "USER.md").write_text("user info", encoding="utf-8")

        digests = [
            {
                "session_id": "s1", "platform": "cli", "title": "Auth work",
                "message_count": 10, "tool_call_count": 5,
                "started_at": "2026-04-06 12:00", "ended_at": "2026-04-06 13:00",
                "end_reason": "cli_close", "tools_used": ["read_file", "terminal"],
                "user_messages": ["Fix auth bug", "Run tests"],
                "last_response": "All tests pass.",
            }
        ]

        analysis_json = json.dumps({
            "insights": ["Auth system uses JWT tokens"],
            "patterns": ["User focuses on testing after fixes"],
            "open_threads": [],
            "session_summary": "Fixed auth and ran tests.",
        })

        with patch.object(engine, "harvest", return_value=digests), \
             patch.object(engine, "make_llm_call", side_effect=[
                 f"```json\n{analysis_json}\n```",
                 "I noticed the user always tests after fixing. Solid discipline.",
             ]), \
             patch.object(engine, "_load_memory_files", return_value=("old memory", "user info")):

            result = engine.run()

        assert result is not None
        assert result["sessions_processed"] == 1
        assert "testing" in result["patterns"][0].lower()
        assert "discipline" in result["dream_narrative"].lower()
        assert Path(result["log_path"]).exists()

        # Verify cursor advanced
        state = engine.state.load()
        assert state["last_processed_session"] == "s1"
        assert state["dream_count"] == 1
