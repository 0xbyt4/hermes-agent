# Dream Mode — Idle-Time Memory Processing

Dream mode lets your Hermes agent process recent conversation memories during idle periods, inspired by how human sleep consolidates knowledge.

## Architecture

```
                    TRIGGER
                       |
          ┌────────────┴────────────┐
          │ Idle ticker (gateway)   │  Checks every 60s
          │ OR /dream command       │  Manual trigger
          │ OR hermes dream run     │  CLI trigger
          └────────────┬────────────┘
                       |
          ┌────────────┴────────────┐
          │  Stage 1: HARVEST       │  Pure code, no LLM
          │  state.db → digests     │  
          │  (user msgs, tools,     │  Reads last N sessions
          │   metadata per session) │  Skips already-processed (cursor)
          └────────────┬────────────┘
                       |
          ┌────────────┴────────────┐
          │  Stage 2+3: ANALYZE     │  LLM call #1 (cheap model)
          │  CONSOLIDATE + CONNECT  │  
          │  Input: digests +       │  Finds insights, patterns,
          │    MEMORY.md + USER.md  │  open threads
          │  Output: JSON           │  
          └────────────┬────────────┘
                       |
                   5s pause
                       |
          ┌────────────┴────────────┐
          │  Stage 4: IMAGINE       │  LLM call #2 (creative model)
          │  Input: analysis result │  
          │  (small, refined)       │  Creative narrative,
          │  Output: dream text     │  unexpected connections
          └────────────┬────────────┘
                       |
          ┌────────────┴────────────┐
          │  Stage 5: JOURNAL       │  Pure code, no LLM
          │  Write dream log (.md)  │  
          │  Advance cursor         │  Never touches MEMORY.md
          │  Deliver to chat (opt)  │  
          └────────────┬────────────┘
                       |
                ~/.hermes/dreams/dream_YYYYMMDD_HHMMSS.md
```

### Data Flow

```
state.db ──→ session digests ──→ LLM #1 (haiku) ──→ LLM #2 (sonnet) ──→ dream log
 (sessions     (user msgs,        (insights,         (creative           (~/.hermes/
  + messages)   tools, metadata)    patterns,          narrative)          dreams/)
                                    threads)
                    +
             MEMORY.md + USER.md
             (read-only context)
```

Key points:
- Memory files are **read** for context but **never written** by dream
- Each session is processed only once (cursor in `state.json`)
- Stage 1 and 5 are pure code (no LLM cost)
- Stage 2+3 share a single LLM call

## How It Works

| Stage | Name | What It Does | Uses LLM? |
|-------|------|-------------|-----------|
| 1 | **Harvest** | Extracts session digests from state.db | No |
| 2 | **Consolidate** | Finds new insights not yet captured | Yes (cheap model) |
| 3 | **Connect** | Identifies cross-session patterns | Yes (same call) |
| 4 | **Imagine** | Makes creative connections between topics | Yes (creative model) |
| 5 | **Journal** | Writes dream log, advances cursor | No |

## Triggering

### Automatic (Gateway)
When the gateway is running, a background `dream-ticker` thread checks every 60 seconds:
1. Are there any active sessions? If yes → reset idle timer
2. Has idle time exceeded `idle_minutes`? If yes → run dream cycle
3. After dream, reset timer to prevent back-to-back runs

### Manual
- **CLI chat:** Type `/dream` in an active hermes chat session
- **CLI command:** Run `hermes dream run` from terminal
- **Gateway:** Type `/dream` in Discord, Telegram, etc.

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

### Provider Support

Dream engine uses Hermes' auth chain:
- **Anthropic OAuth** — full support (auto-detects and adds required identity headers)
- **Anthropic API key** — works directly
- **OpenRouter** — works via OpenAI-compatible path
- **OpenAI/Codex** — works directly
- **Custom endpoint** — set `base_url` + `api_key` in config

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
  state.json                    # Cursor tracking (last processed session)
  dream_20260406_233000.md      # Dream logs
  dream_20260405_120000.md
```

Each dream log contains:
- **Summary** — What happened across processed sessions
- **Patterns** — Cross-session themes and behaviors
- **Open Threads** — Unfinished tasks or ongoing work
- **Insights** — New observations and knowledge discovered
- **Dream** — Creative narrative and suggestions

## How Sessions Are Processed

The dream engine uses a cursor to track which sessions have been processed. Each dream cycle:

1. Reads `state.json` for the last processed session ID
2. Queries `state.db` for sessions newer than the cursor
3. Takes the most recent N sessions (`sessions_to_process`, default 4)
4. For each session, extracts a digest:
   - All user messages (up to `max_messages_per_session`)
   - Last assistant response (truncated to 500 chars)
   - Distinct tool names used
   - Metadata (platform, message count, tool count, timestamps)
5. Sends digests + current memory to LLM for analysis
6. Advances the cursor to the newest processed session

This means:
- Sessions are never processed twice
- Only recent sessions are analyzed (configurable count)
- User message content is read in full for meaningful analysis
- Assistant responses are truncated to save tokens

## Memory Separation

Dream processing **never modifies** agent memory (`MEMORY.md`, `USER.md`).
All findings are written to the dream journal only. Insights and observations
are logged as suggestions — the user or agent can choose to act on them later.

This keeps memory under user control and avoids LLM hallucinations
contaminating persistent knowledge.

## Disabling

```yaml
dream:
  enabled: false
```

Or remove the `dream` section entirely. The feature is disabled by default.
