#!/usr/bin/env python3
"""
Claude Code Tool -- Delegate tasks to Claude Code CLI.

Allows Hermes to offload complex tasks (deep code analysis, refactoring,
multi-file edits, architecture review) to Claude Code via its print mode
(``claude -p``).  This uses the user's existing Claude subscription --
no API key or extra usage credits required.

The tool runs ``claude -p "<prompt>"`` as a subprocess and returns the
output.  The calling LLM (e.g. Qwen) decides when a task is too complex
for itself and delegates to Claude.

Requirements:
  - Claude Code CLI installed and authenticated (``claude`` on PATH)
"""

import logging
import os
import shutil
import subprocess
from typing import Dict, Any, Optional

from tools.registry import registry

logger = logging.getLogger(__name__)

TOOL_NAME = "claude_code"
TOOLSET = "claude_code"

DEFAULT_TIMEOUT = 300  # 5 minutes max
MAX_PROMPT_LENGTH = 100_000
MAX_OUTPUT_LENGTH = 50_000


def check_claude_code_available() -> bool:
    """Return True when the ``claude`` CLI binary is on PATH."""
    return shutil.which("claude") is not None


def _build_claude_command(
    prompt: str,
    model: Optional[str] = None,
    max_turns: Optional[int] = None,
) -> list:
    """Build the claude CLI command list."""
    cmd = ["claude", "-p"]

    if model:
        cmd.extend(["--model", model])

    if max_turns and max_turns > 0:
        cmd.extend(["--max-turns", str(max_turns)])

    cmd.append(prompt)
    return cmd


def claude_code(
    prompt: str,
    model: Optional[str] = None,
    max_turns: Optional[int] = None,
    cwd: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a task via Claude Code CLI in print mode.

    Args:
        prompt:     The task description / prompt for Claude.
        model:      Optional model override (e.g. "sonnet", "opus").
        max_turns:  Max agentic turns Claude can take (default: 1 for simple queries).
        cwd:        Working directory for Claude (defaults to current dir).
        timeout:    Max seconds to wait (default: 300).

    Returns:
        Dict with "success", "output", and optional "error" keys.
    """
    if not prompt or not prompt.strip():
        return {"success": False, "output": "", "error": "Prompt cannot be empty."}

    if len(prompt) > MAX_PROMPT_LENGTH:
        return {
            "success": False,
            "output": "",
            "error": f"Prompt too long ({len(prompt)} chars, max {MAX_PROMPT_LENGTH}).",
        }

    if not check_claude_code_available():
        return {
            "success": False,
            "output": "",
            "error": "Claude Code CLI not found. Install it: npm install -g @anthropic-ai/claude-code",
        }

    effective_timeout = min(timeout or DEFAULT_TIMEOUT, 600)
    work_dir = cwd or os.getcwd()

    cmd = _build_claude_command(prompt, model=model, max_turns=max_turns)

    logger.info("Running Claude Code: cwd=%s, model=%s, timeout=%ds", work_dir, model or "default", effective_timeout)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=effective_timeout,
            cwd=work_dir,
            env={**os.environ, "CLAUDE_CODE_DISABLE_NONINTERACTIVE_HINT": "1"},
        )

        output = result.stdout.strip()
        stderr = result.stderr.strip()

        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + "\n\n[Output truncated]"

        if result.returncode != 0:
            error_detail = stderr or f"Exit code {result.returncode}"
            logger.warning("Claude Code returned non-zero: %s", error_detail)
            return {
                "success": False,
                "output": output,
                "error": f"Claude Code failed: {error_detail}",
            }

        logger.info("Claude Code completed: %d chars output", len(output))
        return {"success": True, "output": output}

    except subprocess.TimeoutExpired:
        logger.error("Claude Code timed out after %ds", effective_timeout)
        return {
            "success": False,
            "output": "",
            "error": f"Claude Code timed out after {effective_timeout} seconds.",
        }
    except Exception as e:
        logger.error("Claude Code execution error: %s", e, exc_info=True)
        return {"success": False, "output": "", "error": f"Execution error: {e}"}


# ---------------------------------------------------------------------------
# Tool handler (called by model_tools dispatch)
# ---------------------------------------------------------------------------

def handle_claude_code(
    prompt: str,
    model: str = "",
    max_turns: int = 0,
    cwd: str = "",
    timeout: int = 0,
    **kwargs,
) -> str:
    """Tool handler entry point."""
    import json
    result = claude_code(
        prompt=prompt,
        model=model or None,
        max_turns=max_turns or None,
        cwd=cwd or None,
        timeout=timeout or None,
    )
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Schema + Registration
# ---------------------------------------------------------------------------

CLAUDE_CODE_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Delegate a complex task to Claude Code (Anthropic's coding agent). "
        "Use this for tasks that require deep code analysis, multi-file refactoring, "
        "architecture review, debugging complex issues, or writing substantial code. "
        "Claude runs locally using the user's existing subscription -- no API key needed. "
        "Provide a clear, detailed prompt describing the task."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": (
                    "Detailed task description for Claude. Include relevant context: "
                    "file paths, code snippets, expected behavior, and specific instructions."
                ),
            },
            "model": {
                "type": "string",
                "description": "Optional model override: 'sonnet', 'opus', or 'haiku'. Leave empty for default.",
                "default": "",
            },
            "max_turns": {
                "type": "integer",
                "description": "Max agentic turns (0 = default). Set higher for complex multi-step tasks.",
                "default": 0,
            },
            "cwd": {
                "type": "string",
                "description": "Working directory for Claude. Leave empty for current directory.",
                "default": "",
            },
            "timeout": {
                "type": "integer",
                "description": "Max seconds to wait (default 300, max 600).",
                "default": 0,
            },
        },
        "required": ["prompt"],
    },
}

registry.register(
    name=TOOL_NAME,
    toolset=TOOLSET,
    schema=CLAUDE_CODE_SCHEMA,
    handler=handle_claude_code,
    check_fn=check_claude_code_available,
    requires_env=[],
    description="Delegate complex tasks to Claude Code CLI",
    emoji="",
)
