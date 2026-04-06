"""
Dream Tool — Agent dream processing during idle time.

Allows the agent (or user) to trigger dream processing manually,
check dream status, and browse dream history.

Actions:
  run      — Trigger a dream cycle (process recent sessions)
  status   — Show dream configuration and last dream info
  history  — List recent dream logs
  read     — Read a specific dream log
"""

import json
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)


DREAM_SCHEMA = {
    "name": "dream",
    "description": (
        "Process recent session memories through a 5-stage dream pipeline: "
        "harvest session digests, consolidate new knowledge, connect patterns, "
        "imagine creative links, and journal the results. "
        "Use 'run' to trigger a dream, 'status' to check state, "
        "'history' to list past dreams, 'read' to view a dream log."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["run", "status", "history", "read"],
                "description": "Dream action to perform",
            },
            "path": {
                "type": "string",
                "description": "Path to dream log file (for 'read' action)",
            },
            "limit": {
                "type": "integer",
                "description": "Number of dreams to list (for 'history', default 5)",
            },
        },
        "required": ["action"],
    },
}


def dream_tool(action: str, path: str = "", limit: int = 5) -> str:
    """Handle dream tool invocations."""
    try:
        from tools.dream_engine import DreamEngine, load_dream_config

        config = load_dream_config()
        engine = DreamEngine(config)

        if action == "run":
            if not config.get("enabled", False):
                return json.dumps({
                    "status": "disabled",
                    "message": "Dream is disabled. Enable in config.yaml: dream.enabled: true",
                })
            result = engine.run()
            if result is None:
                return json.dumps({
                    "status": "skipped",
                    "message": "No new sessions to process since last dream.",
                })
            return json.dumps({
                "status": "complete",
                "log_path": result["log_path"],
                "sessions_processed": result["sessions_processed"],
                "memory_updates_applied": result["memory_updates_applied"],
                "patterns": result["patterns"],
                "session_summary": result["session_summary"],
                "dream_narrative": result["dream_narrative"][:500],
            })

        elif action == "status":
            status = engine.get_status()
            return json.dumps(status)

        elif action == "history":
            dreams = engine.list_dreams(limit=max(1, min(limit, 50)))
            return json.dumps({"dreams": dreams, "count": len(dreams)})

        elif action == "read":
            if not path:
                return json.dumps({"error": "path is required for read action"})
            from pathlib import Path as P
            log = P(path)
            if not log.exists():
                return json.dumps({"error": f"Dream log not found: {path}"})
            # Only allow reading from dream directory
            from tools.dream_engine import get_dream_dir
            dream_dir = get_dream_dir()
            if not str(log.resolve()).startswith(str(dream_dir.resolve())):
                return json.dumps({"error": "Access denied: not a dream log"})
            content = log.read_text(encoding="utf-8")
            return json.dumps({"path": str(log), "content": content})

        else:
            return json.dumps({
                "error": f"Unknown action: {action}",
                "available": ["run", "status", "history", "read"],
            })

    except Exception as e:
        logger.error("Dream tool error: %s", e)
        return json.dumps({"error": f"Dream processing failed: {type(e).__name__}: {e}"})


def _check_dream_available() -> bool:
    """Check if dream tool dependencies are available."""
    try:
        from tools.dream_engine import DreamEngine
        return True
    except ImportError:
        return False


registry.register(
    name="dream",
    toolset="dream",
    schema=DREAM_SCHEMA,
    handler=lambda args, **kw: dream_tool(
        action=args.get("action", "status"),
        path=args.get("path", ""),
        limit=args.get("limit", 5),
    ),
    check_fn=_check_dream_available,
    emoji="dream",
)
