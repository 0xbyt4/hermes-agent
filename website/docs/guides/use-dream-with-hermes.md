# Dream Mode — Idle-Time Memory Processing

Dream mode lets your Hermes agent process recent conversation memories during idle periods, inspired by how human sleep consolidates knowledge.

## How It Works

When enabled, the agent runs a 5-stage dream pipeline after being idle for a configurable period:

| Stage | Name | What It Does | Uses LLM? |
|-------|------|-------------|-----------|
| 1 | **Harvest** | Extracts session digests from recent conversations | No |
| 2 | **Consolidate** | Finds new facts not yet in memory | Yes (cheap model) |
| 3 | **Connect** | Identifies cross-session patterns | Yes (same call) |
| 4 | **Imagine** | Makes creative connections between topics | Yes (creative model) |
| 5 | **Journal** | Writes dream log, updates memory, advances cursor | No |

## Quick Start

Add to your `~/.hermes/config.yaml`:

```yaml
dream:
  enabled: true
  model: claude-haiku-4-5-20251001
```

That's it. The agent will start dreaming after 30 minutes of inactivity.

## Configuration

Full config options:

```yaml
dream:
  enabled: false                          # Enable dream processing
  model: claude-haiku-4-5-20251001        # Model for analysis (stages 2+3)
  creative_model: ""                       # Model for creative stage (default: same as model)
  provider: ""                             # Provider (default: use main provider)
  base_url: ""                             # Custom API endpoint
  api_key: ""                              # API key (default: use env var)
  idle_minutes: 30                         # Minutes idle before dreaming
  sessions_to_process: 4                   # Sessions to analyze per dream
  max_messages_per_session: 50             # Max user messages per session
  deliver: true                            # Send dream summary to chat
```

### Dual-Model Setup

Use a cheap model for bulk analysis and a stronger model for creative connections:

```yaml
dream:
  enabled: true
  model: claude-haiku-4-5-20251001         # cheap: analysis
  creative_model: claude-sonnet-4-6         # strong: creative
```

### Cost Control

Dream processing makes 2 LLM calls per cycle:
1. **Analysis call** — processes session digests + memory (~2-5K tokens input)
2. **Creative call** — processes analysis results (~1-2K tokens input)

Using Haiku for both keeps costs minimal (fraction of a cent per dream).

## CLI Commands

```bash
# Check dream status
hermes dream status

# Trigger a dream manually
hermes dream run

# List recent dreams
hermes dream history
hermes dream history -n 20

# Read the latest dream log
hermes dream read

# Read a specific dream log
hermes dream read dream_20260406_233000.md
```

## Gateway Commands

In Discord, Telegram, or any chat platform:

```
/dream           — Trigger a dream cycle now
/dream status    — Show dream state and config
/dream history   — List recent dreams
```

## Dream Output

Dreams are stored in `~/.hermes/dreams/` as markdown files:

```
~/.hermes/dreams/
  state.json                    # Cursor tracking
  dream_20260406_233000.md      # Dream logs
  dream_20260405_120000.md
```

Each dream log contains:
- **Summary** — What happened across processed sessions
- **Patterns** — Cross-session themes and behaviors
- **Open Threads** — Unfinished tasks or ongoing work
- **Memory Updates** — What was added/updated in memory
- **Dream** — Creative narrative and suggestions

## How Sessions Are Processed

The dream engine uses a cursor to track which sessions have been processed. Each dream cycle:

1. Checks `state.json` for the last processed session
2. Finds sessions newer than the cursor
3. For each session, extracts:
   - All user messages (what was asked)
   - Last assistant response (the outcome)
   - Tool names used (what capabilities were exercised)
   - Metadata (platform, duration, message/tool counts)
4. Sends digests to the LLM for analysis
5. Advances the cursor

This means:
- Sessions are never processed twice
- Only recent sessions are analyzed (configurable count)
- Full message content is read but truncated for token efficiency

## Memory Updates

The dream engine can update both memory files:
- `~/.hermes/memories/MEMORY.md` — Agent's personal notes
- `~/.hermes/memories/USER.md` — User profile

Safety guards:
- Duplicate content is skipped
- Character limits are respected
- Updates are logged in the dream journal

## Disabling

```yaml
dream:
  enabled: false
```

Or remove the `dream` section entirely. The feature is disabled by default.
