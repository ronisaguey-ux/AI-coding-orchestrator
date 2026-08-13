# AI Coding Orchestrator

A fully automated, self-adaptive master orchestrator for AI-driven software projects. It runs an infinite loop of **audit → cross-eval → execute → analyze → graphify rebuild** over your repository, powered by any Anthropic-compatible LLM API, with a Telegram bot for remote control.

## Features

- **Infinite improvement loop**: audits the repo for bugs/security issues, cross-validates findings with multi-agent review, executes plan steps (writing real code), and rebuilds project graphs after each step.
- **LLM self-adaptation**: a reasoning tier analyzes failures and rewrites workflow scripts when needed.
- **Telegram remote control**: start/pause/resume, live status, and a direct LLM chat interface from your phone.
- **State checkpointing**: every step is persisted — crash-safe resume with `--resume`.
- **Persistent task board** (`agent_todo_list.json`) tracking work across runs.
- **Error recovery**: LLM-based diagnosis and fixes when a plan step fails.

## Requirements

- Python 3.10+
- An LLM API that speaks the OpenAI-compatible `/v1` chat format (DeepSeek, OmniRoute, OpenAI-compatible gateways, etc.)
- Optional: a Telegram bot token for remote control

## Setup

```bash
git clone <this-repo>
cd AI-coding-orchestrator
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create the environment file (gitignored by default):

```bash
mkdir -p ~/.config/orchestrator
cat > ~/.config/orchestrator/orchestrator.env <<'EOF'
LLM_API_KEY=your-api-key
LLM_API_BASE=https://api.your-provider.com/v1
LLM_MODEL=your-model-name
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
REPO_DIR=/path/to/your/project
WORK_DIR=~/orchestrator_data
EOF
```

## Usage

```bash
./start_orchestrator.sh                 # resume from last state (execution phase)
./start_orchestrator.sh --phase audit   # start the full loop at the audit phase
./start_orchestrator.sh --status        # show orchestrator status and exit
```

Or run directly:

```bash
python3 run_workflow.py --config config_workflow.yaml
python3 run_workflow.py --resume
python3 run_workflow.py --status
```

## Configuration

All configuration lives in `config_workflow.yaml`. Secret values are never stored there — every secret is a `${ENV_VAR}` placeholder expanded from the environment.

| Variable | Required | Description |
|---|---|---|
| `LLM_API_KEY` | ✅ | API key for your LLM provider |
| `LLM_API_BASE` | ✅ | Base URL of the OpenAI-compatible `/v1` endpoint |
| `LLM_MODEL` | ✅ | Model name (e.g. `deepseek-v4-flash`, `gpt-4o`) |
| `TELEGRAM_BOT_TOKEN` | ❌ | Bot token (skip to run without remote control) |
| `TELEGRAM_CHAT_ID` | ❌ | Chat ID to send status messages to |
| `REPO_DIR` | ✅ | Absolute path of the project being orchestrated |
| `WORK_DIR` | ❌ | Working directory for state files (default: next to `REPO_DIR`) |

## Running it cheaply with OmniRoute

[OmniRoute](https://github.com/your-fork-or-upstream/omniroute) is a free, self-hosted LLM router that load-balances requests across free model providers. Point the orchestrator at it for near-zero-cost plan execution:

```bash
LLM_API_BASE=http://127.0.0.1:20128/v1   # OmniRoute's local endpoint
LLM_API_KEY=anything                     # OmniRoute does not validate keys locally
LLM_MODEL=auto                           # OmniRoute routes per its own config
```

Example: a full `audit → cross-eval → execute` cycle on a medium repo runs for pennies instead of dollars — OmniRoute only calls your paid provider for the reasoning-heavy steps you explicitly route there.

## How it works (high level)

1. **Audit** — spawns parallel reviewer agents over the repo's modules; findings are scored and deduplicated into an audit report.
2. **Cross-eval** — a synthesis pass validates the findings and produces the action list.
3. **Execute** — runs the plan steps one at a time, with git-safe staging, per-step verification (tests/lints), and auto-rollback on failure.
4. **Analyze** — measures progress and feeds the result back into the next plan.
5. **Graphify rebuild** — regenerates the project structure graph after each step.

## Safety notes

- The orchestrator commits to git — run it only on repositories you want it to modify, and keep the default `--resume` behavior so state is never lost.
- Pin `MASTER_PLAN_FILE` in `start_orchestrator.sh` when you want the executor locked to a specific plan.
- The pipeline is fail-closed on state corruption: it halts for manual review rather than overwriting.

## License

MIT
>>>>>>> 5774c6c (Sanitized orchestrator release: run_workflow + config + start script (no secrets))
