"""
Dream Engine — 5-stage dream processing pipeline for Hermes Agent.

Processes recent session memories during idle time, mimicking human sleep stages:

  Stage 1: HARVEST      — Extract session digests from state.db (pure code, no LLM)
  Stage 2: CONSOLIDATE  — Find new facts, update memory (LLM call #1, cheap model)
  Stage 3: CONNECT      — Find cross-session patterns (same LLM call as stage 2)
  Stage 4: IMAGINE      — Creative connections and ideas (LLM call #2, creative model)
  Stage 5: JOURNAL      — Write dream log, update memory, advance cursor (pure code)

The engine separates code-only stages (1, 5) from LLM stages (2+3, 4) so cheap
models handle bulk analysis while the creative model only processes refined input.

Config section in config.yaml:

  dream:
    enabled: true
    model: claude-haiku-4-5-20251001
    creative_model: claude-sonnet-4-6
    provider: anthropic
    idle_minutes: 30
    sessions_to_process: 4
    max_messages_per_session: 50
    deliver: true
"""

import json
import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENTRY_DELIMITER = "\n§\n"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DREAM_CONFIG = {
    "enabled": False,
    "model": "claude-haiku-4-5-20251001",
    "creative_model": "",
    "provider": "",
    "base_url": "",
    "api_key": "",
    "idle_minutes": 30,
    "sessions_to_process": 4,
    "max_messages_per_session": 50,
    "deliver": True,
}


def get_dream_dir() -> Path:
    """Return dream output directory, creating it if needed."""
    home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    dream_dir = home / "dreams"
    dream_dir.mkdir(parents=True, exist_ok=True)
    return dream_dir


def load_dream_config() -> Dict[str, Any]:
    """Load dream config from config.yaml with defaults."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        return dict(DEFAULT_DREAM_CONFIG)
    dream_cfg = cfg.get("dream", {})
    if not isinstance(dream_cfg, dict):
        dream_cfg = {}
    merged = dict(DEFAULT_DREAM_CONFIG)
    merged.update({k: v for k, v in dream_cfg.items() if v is not None and v != ""})
    return merged


# ---------------------------------------------------------------------------
# Dream State (cursor tracking)
# ---------------------------------------------------------------------------

class DreamState:
    """Tracks which sessions have been processed and dream history."""

    def __init__(self, dream_dir: Optional[Path] = None):
        self._dir = dream_dir or get_dream_dir()
        self._path = self._dir / "state.json"

    def load(self) -> Dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "last_processed_session": None,
            "last_dream_at": None,
            "dream_count": 0,
        }

    def save(self, state: Dict[str, Any]):
        self._dir.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(self._dir), suffix=".tmp", prefix=".dream_state_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, str(self._path))
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# ---------------------------------------------------------------------------
# Dream Engine
# ---------------------------------------------------------------------------

class DreamEngine:
    """5-stage dream processing pipeline."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_dream_config()
        self.sessions_to_process = int(self.config.get("sessions_to_process", 4))
        self.max_messages = int(self.config.get("max_messages_per_session", 50))
        self.model = self.config.get("model", "claude-haiku-4-5-20251001")
        self.creative_model = self.config.get("creative_model", "") or self.model
        self.state = DreamState()

    # =====================================================================
    # Stage 1: HARVEST — Extract session digests (pure code, no LLM)
    # =====================================================================

    def harvest(self) -> List[Dict[str, Any]]:
        """Extract digests from recent unprocessed sessions.

        Reads state.db directly — user messages, tool names, metadata.
        Returns list of digest dicts, newest first.
        """
        cursor_state = self.state.load()
        last_processed = cursor_state.get("last_processed_session")

        try:
            from hermes_state import SessionDB
            db = SessionDB()
        except Exception as e:
            logger.warning("Dream harvest: cannot open state.db: %s", e)
            return []

        try:
            conn = db._conn
            sessions = self._query_new_sessions(conn, last_processed)
            if not sessions:
                return []

            digests = []
            for row in sessions:
                digest = self._extract_digest(conn, row)
                if digest:
                    digests.append(digest)

            return digests
        except Exception as e:
            logger.error("Dream harvest failed: %s", e)
            return []
        finally:
            db.close()

    def _query_new_sessions(self, conn, last_processed: Optional[str]) -> list:
        """Query sessions newer than the last processed one."""
        if last_processed:
            cursor = conn.execute(
                "SELECT started_at FROM sessions WHERE id = ?",
                (last_processed,),
            )
            row = cursor.fetchone()
            if row:
                return conn.execute(
                    "SELECT id, source, title, message_count, tool_call_count, "
                    "started_at, ended_at, end_reason "
                    "FROM sessions WHERE started_at > ? AND message_count > 0 "
                    "ORDER BY started_at DESC LIMIT ?",
                    (row[0], self.sessions_to_process),
                ).fetchall()

        # No cursor or cursor session not found — take most recent
        return conn.execute(
            "SELECT id, source, title, message_count, tool_call_count, "
            "started_at, ended_at, end_reason "
            "FROM sessions WHERE message_count > 0 "
            "ORDER BY started_at DESC LIMIT ?",
            (self.sessions_to_process,),
        ).fetchall()

    def _extract_digest(self, conn, session_row: tuple) -> Optional[Dict[str, Any]]:
        """Build a digest dict from a session row + its messages."""
        session_id = session_row[0]

        digest = {
            "session_id": session_id,
            "platform": session_row[1] or "unknown",
            "title": session_row[2] or "(untitled)",
            "message_count": session_row[3] or 0,
            "tool_call_count": session_row[4] or 0,
            "started_at": self._ts(session_row[5]),
            "ended_at": self._ts(session_row[6]),
            "end_reason": session_row[7],
        }

        # User messages — the most valuable signal for dream processing
        user_rows = conn.execute(
            "SELECT content FROM messages "
            "WHERE session_id = ? AND role = 'user' AND content != '' "
            "ORDER BY timestamp LIMIT ?",
            (session_id, self.max_messages),
        ).fetchall()
        digest["user_messages"] = [r[0] for r in user_rows if r[0]]

        # Last assistant response — the session outcome
        last_resp = conn.execute(
            "SELECT content FROM messages "
            "WHERE session_id = ? AND role = 'assistant' AND content != '' "
            "ORDER BY timestamp DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        digest["last_response"] = (
            last_resp[0][:500] if last_resp and last_resp[0] else ""
        )

        # Tool names used — what capabilities were exercised
        tool_rows = conn.execute(
            "SELECT DISTINCT tool_name FROM messages "
            "WHERE session_id = ? AND tool_name IS NOT NULL AND tool_name != ''",
            (session_id,),
        ).fetchall()
        digest["tools_used"] = [t[0] for t in tool_rows]

        return digest

    @staticmethod
    def _ts(unix_ts) -> Optional[str]:
        """Format unix timestamp to ISO string."""
        if unix_ts:
            try:
                return datetime.fromtimestamp(float(unix_ts)).strftime(
                    "%Y-%m-%d %H:%M"
                )
            except (ValueError, TypeError, OSError):
                pass
        return None

    # =====================================================================
    # Stage 2+3: CONSOLIDATE + CONNECT (LLM call #1)
    # =====================================================================

    def build_analysis_prompt(
        self,
        digests: List[Dict],
        memory_content: str,
        user_content: str,
    ) -> str:
        """Build prompt for consolidation + pattern detection."""
        session_text = "\n\n---\n\n".join(
            self._format_digest(d) for d in digests
        )

        return (
            "You are processing dream memories for an AI agent. "
            "Analyze recent sessions against current memory to find "
            "new knowledge and patterns.\n\n"
            "## Current Agent Memory\n"
            f"{memory_content or '(empty)'}\n\n"
            "## Current User Profile\n"
            f"{user_content or '(empty)'}\n\n"
            "## Recent Sessions\n"
            f"{session_text}\n\n"
            "## Instructions\n\n"
            "### CONSOLIDATE\n"
            "Compare sessions against current memory. Find:\n"
            "- New facts not in memory (topics, decisions, preferences)\n"
            "- Outdated memory entries that need updating\n"
            "- User behavior patterns (work hours, style, recurring topics)\n\n"
            "### CONNECT\n"
            "Find cross-session patterns:\n"
            "- Topics appearing across multiple sessions\n"
            "- Evolving projects or ongoing work\n"
            "- Unfinished tasks or recurring problems\n\n"
            "Respond in this exact JSON format:\n"
            "```json\n"
            "{\n"
            '  "memory_updates": [\n'
            '    {"action": "add", "target": "memory", '
            '"content": "new fact to remember"},\n'
            '    {"action": "add", "target": "user", '
            '"content": "user preference or pattern"},\n'
            '    {"action": "replace", "target": "memory", '
            '"old": "outdated substring", "content": "updated text"}\n'
            "  ],\n"
            '  "patterns": ["pattern 1", "pattern 2"],\n'
            '  "open_threads": ["unfinished task 1"],\n'
            '  "session_summary": "2-3 sentence summary"\n'
            "}\n"
            "```\n\n"
            "Only include genuine findings. Do not fabricate. "
            "If memory is already comprehensive, memory_updates can be empty."
        )

    def parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from analysis LLM response."""
        if not response:
            return self._empty_analysis()

        # Extract JSON from markdown code blocks
        match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, re.DOTALL)
        text = match.group(1) if match else response

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: find first JSON object
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning("Dream: could not parse analysis JSON, using raw text")
        return {
            "session_summary": response[:500],
            "memory_updates": [],
            "patterns": [],
            "open_threads": [],
        }

    @staticmethod
    def _empty_analysis() -> Dict[str, Any]:
        return {
            "session_summary": "",
            "memory_updates": [],
            "patterns": [],
            "open_threads": [],
        }

    # =====================================================================
    # Stage 4: IMAGINE (LLM call #2)
    # =====================================================================

    def build_creative_prompt(
        self, analysis: Dict[str, Any], memory_content: str
    ) -> str:
        """Build prompt for creative dream generation."""
        patterns = "\n".join(
            f"- {p}" for p in analysis.get("patterns", [])
        ) or "None detected"
        threads = "\n".join(
            f"- {t}" for t in analysis.get("open_threads", [])
        ) or "None"
        summary = analysis.get("session_summary", "No summary available")

        return (
            "You are an AI agent in dream mode. Your analytical mind has "
            "processed recent sessions and found these patterns. Now let "
            "your creative side make unexpected connections.\n\n"
            "## What happened recently\n"
            f"{summary}\n\n"
            "## Patterns found\n"
            f"{patterns}\n\n"
            "## Open threads\n"
            f"{threads}\n\n"
            "## Current memory\n"
            f"{memory_content or '(empty)'}\n\n"
            "## Dream instructions\n\n"
            "Think freely. Make connections between unrelated topics. "
            "Consider:\n"
            "- Could two separate projects benefit each other?\n"
            "- Is there a tool or approach from one context "
            "that applies elsewhere?\n"
            "- What is the user likely working toward?\n"
            "- Any creative ideas worth noting?\n\n"
            "Write a short dream narrative (3-8 sentences). "
            "Be genuine — if nothing interesting connects, say so briefly. "
            "Write in first person as the agent.\n\n"
            "Then add any actionable suggestions as a separate list "
            "(0-3 items max). Only include genuinely useful ideas."
        )

    # =====================================================================
    # Stage 5: JOURNAL — Write log, apply memory, advance cursor
    # =====================================================================

    def write_journal(
        self,
        digests: List[Dict],
        analysis: Dict[str, Any],
        dream_narrative: str,
    ) -> Path:
        """Write dream log file and return its path."""
        dream_dir = get_dream_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = dream_dir / f"dream_{timestamp}.md"

        lines = [
            f"# Dream — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"**Sessions processed:** {len(digests)}",
            "**Platforms:** "
            + ", ".join(sorted(set(d["platform"] for d in digests))),
            "",
        ]

        if analysis.get("session_summary"):
            lines.extend(["## Summary", "", analysis["session_summary"], ""])

        if analysis.get("patterns"):
            lines.append("## Patterns")
            lines.append("")
            for p in analysis["patterns"]:
                lines.append(f"- {p}")
            lines.append("")

        if analysis.get("open_threads"):
            lines.append("## Open Threads")
            lines.append("")
            for t in analysis["open_threads"]:
                lines.append(f"- {t}")
            lines.append("")

        if analysis.get("memory_updates"):
            lines.append("## Memory Updates")
            lines.append("")
            for u in analysis["memory_updates"]:
                action = u.get("action", "add")
                target = u.get("target", "memory")
                content = u.get("content", "")[:100]
                lines.append(f"- [{action}] {target}: {content}")
            lines.append("")

        if dream_narrative:
            lines.extend(["## Dream", "", dream_narrative, ""])

        # Session details appendix
        lines.append("## Sessions")
        lines.append("")
        for d in digests:
            lines.append(
                f"- **{d['title']}** ({d['platform']}, "
                f"{d['message_count']} msgs, "
                f"{d['tool_call_count']} tools)"
            )
        lines.append("")

        log_path.write_text("\n".join(lines), encoding="utf-8")
        return log_path

    def advance_cursor(self, digests: List[Dict]):
        """Update state cursor to the newest processed session."""
        state = self.state.load()
        if digests:
            state["last_processed_session"] = digests[0]["session_id"]
        state["last_dream_at"] = datetime.now().isoformat()
        state["dream_count"] = state.get("dream_count", 0) + 1
        self.state.save(state)

    def apply_memory_updates(self, updates: List[Dict]) -> int:
        """Apply memory updates from consolidation. Returns count applied."""
        if not updates:
            return 0

        try:
            from tools.memory_tool import MemoryStore, get_memory_dir
        except ImportError:
            logger.warning("Dream: memory_tool not available, skipping updates")
            return 0

        store = MemoryStore()
        store.load_from_disk()
        applied = 0

        for update in updates:
            action = update.get("action", "add")
            target = update.get("target", "memory")
            content = update.get("content", "").strip()
            if not content:
                continue

            entries = (
                store.memory_entries if target == "memory" else store.user_entries
            )
            char_limit = (
                store.memory_char_limit
                if target == "memory"
                else store.user_char_limit
            )

            if action == "add":
                # Skip if similar content exists
                content_lower = content.lower()
                if any(
                    content_lower in e.lower() or e.lower() in content_lower
                    for e in entries
                ):
                    continue
                # Check char limit
                current_chars = sum(len(e) for e in entries)
                if current_chars + len(content) > char_limit:
                    logger.info(
                        "Dream: skipping add to %s — would exceed %d char limit",
                        target,
                        char_limit,
                    )
                    continue
                entries.append(content)
                applied += 1

            elif action == "replace":
                old = update.get("old", "")
                if old:
                    for i, entry in enumerate(entries):
                        if old in entry:
                            entries[i] = entry.replace(old, content)
                            applied += 1
                            break

            elif action == "remove":
                old = update.get("old", content)
                before = len(entries)
                entries[:] = [e for e in entries if old not in e]
                applied += before - len(entries)

            store.save_to_disk(target)

        return applied

    # =====================================================================
    # LLM call
    # =====================================================================

    def make_llm_call(
        self, prompt: str, model: Optional[str] = None
    ) -> Optional[str]:
        """Make an LLM call for dream processing.

        Resolves provider and API key from dream config, falling back to
        the main model config and environment variables.
        """
        model = model or self.model
        provider = self.config.get("provider", "").strip()
        api_key = self.config.get("api_key", "").strip()
        base_url = self.config.get("base_url", "").strip()

        # Resolve provider from model name if not set
        if not provider:
            try:
                from hermes_cli.config import load_config
                cfg = load_config()
                provider = cfg.get("model", {}).get("provider", "anthropic")
            except Exception:
                provider = "anthropic"

        # Resolve API key from environment if not set
        if not api_key:
            if provider == "anthropic":
                api_key = os.getenv("ANTHROPIC_API_KEY", "")
            elif provider in ("openai", "codex"):
                api_key = os.getenv("OPENAI_API_KEY", "")
            elif provider == "openrouter":
                api_key = os.getenv("OPENROUTER_API_KEY", "")
            else:
                api_key = os.getenv("ANTHROPIC_API_KEY", "") or os.getenv(
                    "OPENAI_API_KEY", ""
                )

        if not api_key:
            logger.error("Dream: no API key available for provider %s", provider)
            return None

        try:
            return self._call_provider(provider, model, api_key, base_url, prompt)
        except Exception as e:
            logger.error("Dream LLM call failed (%s/%s): %s", provider, model, e)
            return None

    def _call_provider(
        self,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        prompt: str,
    ) -> Optional[str]:
        """Dispatch LLM call to the appropriate provider SDK."""
        if provider == "anthropic" and not base_url:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text if response.content else None

        # OpenAI-compatible (openai, openrouter, nous, custom)
        import openai

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        elif provider == "openrouter":
            client_kwargs["base_url"] = "https://openrouter.ai/api/v1"

        client = openai.OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        choice = response.choices[0] if response.choices else None
        return choice.message.content if choice else None

    # =====================================================================
    # Full pipeline
    # =====================================================================

    def run(self) -> Optional[Dict[str, Any]]:
        """Run the complete 5-stage dream pipeline.

        Returns a result dict on success, None if nothing to process.
        """
        # Stage 1: HARVEST
        digests = self.harvest()
        if not digests:
            logger.info("Dream: no new sessions to process")
            return None

        logger.info("Dream: harvested %d session digests", len(digests))

        # Load current memory for context
        memory_content, user_content = self._load_memory_files()

        # Stage 2+3: CONSOLIDATE + CONNECT
        analysis_prompt = self.build_analysis_prompt(
            digests, memory_content, user_content
        )
        analysis_response = self.make_llm_call(analysis_prompt, model=self.model)
        analysis = (
            self.parse_analysis_response(analysis_response)
            if analysis_response
            else self._empty_analysis()
        )

        logger.info(
            "Dream analysis: %d updates, %d patterns, %d threads",
            len(analysis.get("memory_updates", [])),
            len(analysis.get("patterns", [])),
            len(analysis.get("open_threads", [])),
        )

        # Stage 4: IMAGINE
        creative_prompt = self.build_creative_prompt(analysis, memory_content)
        dream_narrative = self.make_llm_call(
            creative_prompt, model=self.creative_model
        ) or ""

        # Stage 5: JOURNAL
        log_path = self.write_journal(digests, analysis, dream_narrative)
        self.advance_cursor(digests)

        # Apply memory updates
        updates_applied = self.apply_memory_updates(
            analysis.get("memory_updates", [])
        )
        logger.info(
            "Dream complete: %s (%d memory updates applied)", log_path, updates_applied
        )

        return {
            "log_path": str(log_path),
            "sessions_processed": len(digests),
            "memory_updates_applied": updates_applied,
            "patterns": analysis.get("patterns", []),
            "open_threads": analysis.get("open_threads", []),
            "session_summary": analysis.get("session_summary", ""),
            "dream_narrative": dream_narrative,
        }

    # =====================================================================
    # Helpers
    # =====================================================================

    def _format_digest(self, digest: Dict[str, Any]) -> str:
        """Format a single session digest for LLM consumption."""
        lines = [
            f"### {digest['title']}",
            f"Platform: {digest['platform']} | "
            f"Messages: {digest['message_count']} | "
            f"Tools: {digest['tool_call_count']}",
            f"Time: {digest['started_at'] or '?'} → "
            f"{digest['ended_at'] or 'ongoing'}",
        ]
        if digest.get("end_reason"):
            lines.append(f"End reason: {digest['end_reason']}")
        if digest.get("tools_used"):
            lines.append(f"Tools used: {', '.join(digest['tools_used'])}")

        if digest.get("user_messages"):
            lines.append("")
            lines.append("User messages:")
            for msg in digest["user_messages"]:
                preview = msg[:300].replace("\n", " ")
                if len(msg) > 300:
                    preview += "..."
                lines.append(f"  - {preview}")

        if digest.get("last_response"):
            lines.append("")
            resp_preview = digest["last_response"][:300].replace("\n", " ")
            lines.append(f"Final response: {resp_preview}")

        return "\n".join(lines)

    @staticmethod
    def _load_memory_files() -> Tuple[str, str]:
        """Read current MEMORY.md and USER.md contents."""
        try:
            from tools.memory_tool import get_memory_dir

            mem_dir = get_memory_dir()
        except ImportError:
            home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
            mem_dir = home / "memories"

        memory = ""
        user = ""
        mem_path = mem_dir / "MEMORY.md"
        user_path = mem_dir / "USER.md"
        if mem_path.exists():
            try:
                memory = mem_path.read_text(encoding="utf-8")
            except OSError:
                pass
        if user_path.exists():
            try:
                user = user_path.read_text(encoding="utf-8")
            except OSError:
                pass
        return memory, user

    # =====================================================================
    # Status / History
    # =====================================================================

    def get_status(self) -> Dict[str, Any]:
        """Return current dream state and config summary."""
        state = self.state.load()
        dream_dir = get_dream_dir()
        logs = sorted(dream_dir.glob("dream_*.md"), reverse=True)
        return {
            "enabled": self.config.get("enabled", False),
            "model": self.model,
            "creative_model": self.creative_model,
            "idle_minutes": self.config.get("idle_minutes", 30),
            "sessions_to_process": self.sessions_to_process,
            "last_dream_at": state.get("last_dream_at"),
            "dream_count": state.get("dream_count", 0),
            "last_processed_session": state.get("last_processed_session"),
            "log_count": len(logs),
            "latest_log": str(logs[0]) if logs else None,
        }

    @staticmethod
    def list_dreams(limit: int = 10) -> List[Dict[str, str]]:
        """List recent dream logs."""
        dream_dir = get_dream_dir()
        logs = sorted(dream_dir.glob("dream_*.md"), reverse=True)[:limit]
        results = []
        for log in logs:
            # Extract date from filename: dream_YYYYMMDD_HHMMSS.md
            name = log.stem  # dream_20260406_233000
            try:
                ts = datetime.strptime(name, "dream_%Y%m%d_%H%M%S")
                date_str = ts.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                date_str = name

            # Read first few lines for preview
            try:
                content = log.read_text(encoding="utf-8")
                # Find summary section
                summary = ""
                for line in content.split("\n"):
                    if line and not line.startswith("#") and not line.startswith("**"):
                        summary = line[:120]
                        break
            except OSError:
                summary = ""

            results.append({
                "path": str(log),
                "date": date_str,
                "preview": summary,
            })
        return results
