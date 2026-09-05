# AI Coding Orchestrator

A high-throughput, multi-lane autonomous fix-executor and escalation solver for code audit and remediation plans.

The orchestrator density-packs independent plan findings into disjoint-file batches, drives multi-lane LLM execution with automatic fallback and failover, applies byte-exact modifications with atomic file updates, performs strict in-process verification, and drains stalled steps via an autonomous 3-tier escalation solver.

---

## Key Components

- **`orch_config.py`**: Layered configuration engine (`defaults < profile < config file < environment < CLI`). Location-derived roots allow cloning and execution without path editing.
- **`orch_lanes.py`**: Multi-lane routing across local gateways, routers, and free pools (e.g. DeepSeek, Gemini, OmniRoute, OpenRouter Free). Distinguishes transport failure (`ok=False`) from considered empty model output, preventing premature escalations. Features per-model cooldowns, SSE parsing, and dead-lane parking.
- **`orch_verify.py`**: In-process AST and syntax verification (replaces unsafe execution mechanisms) with strict allowlisting for model-proposed verification commands.
- **`orchestrator.py`**: Main execution engine. Density-packs up to `group_cap` disjoint steps into a single model round, applies edits with atomic locking and rolling backups, and runs local test verification.
- **`escalation_solver.py`**: Autonomous 3-tier solver for steps that exceed normal retry bounds:
  - **Tier 1 (RETRY)**: Re-runs the step against healthy lanes with fresh file context.
  - **Tier 2 (REPAIR)**: Re-runs quoting previous verifier or anchor failures back to the model.
  - **Tier 3 (PERSONA)**: Bounded escalation persona that can repair plan steps, supply sandboxed verification commands, or declare moot findings `obsolete` with filesystem-verified evidence.

---

## Installation & Setup

```bash
git clone https://github.com/ronisaguey-ux/AI-coding-orchestrator.git
cd AI-coding-orchestrator
pip install -r requirements.txt
```

### Configuration

Copy the example configuration file and adjust for your workspace:

```bash
cp orch.example.yaml orch.yaml
# Or place in ~/.config/orch/orch.yaml
```

Check the resolved configuration at any time without side effects:

```bash
python3 orchestrator.py --dry-run
# Or print the resolved options:
python3 orch_config.py
```

---

## Usage

### Running the Orchestrator

```bash
# Run against the configured plan and state
python3 orchestrator.py

# Resume from existing state
python3 orchestrator.py --resume

# Run a specific batch
python3 orchestrator.py --batch 1

# Focus on a single finding ID
python3 orchestrator.py --only-step STEP-042

# Dry-run mode (zero writes, zero side effects)
python3 orchestrator.py --dry-run
```

### Running the Escalation Solver

```bash
# Solve up to 5 escalated steps
python3 escalation_solver.py --limit 5

# Dry-run solver on a specific step
python3 escalation_solver.py --step STEP-042 --dry-run

# Run live solver on a specific step
python3 escalation_solver.py --step STEP-042

# Continuous drain loop
python3 escalation_solver.py --watch --interval 300
```

---

## Configuration Profiles

Profiles can be selected via `ORCH_PROFILE=<name>` or the `--profile` flag:

| Profile | Group Cap | Parallel | Max Rounds | Lane Timeout | Escalation Batch | Description |
| --- | --- | --- | --- | --- | --- | --- |
| `default` | 5 | 3 | 3 | 240s | 5 | Balanced execution for standard workloads |
| `max-throughput` | 5 | 3 | 3 | 180s | 8 | Aggressive batching and shorter retry cycles |
| `low-memory` | 2 | 1 | 3 | 240s | 2 | Constrained memory footprints (`min_avail_mb=1200`, smaller context caps) |
| `conservative` | 1 | 1 | 3 | 300s | 5 | Single-step execution, 20s lane cooldown, wake-only escalation |
| `debug` | 1 | 1 | 3 | 240s | 5 | `dry_run=True`, verbose logging, state backups |

---

## Configuration Options

| Option | Default | Tuned by profiles | Description |
| --- | --- | --- | --- |
| `base_dir` | `'<repo>/.orch'` | — | Plans/state directory |
| `dry_run` | `False` | debug | Resolve and print actions without side effects |
| `escalation_allow_plan_edit` | `True` | conservative | Escalation persona may rewrite a broken plan step |
| `escalation_allow_verify_edit` | `True` | conservative | Escalation persona may relax/repair the verify command |
| `escalation_batch` | `5` | max-throughput, low-memory | Escalated steps handled per solver pass |
| `escalation_policy` | `'auto-fix'` | conservative | wake \| auto-fix \| queue-only |
| `escalation_queue` | `''` | — | Escalation queue JSON (default: <base>/escalation_queue.json) |
| `escalation_rounds` | `3` | — | Escalation-solver attempts per step |
| `exec_state` | `''` | — | Executor state JSON (default: <base>/exec_state.json) |
| `file_ctx_cap` | `24000` | low-memory | Per-file FILE CONTENTS cap in characters |
| `files_per_step` | `2` | — | Files inlined per step in a group prompt |
| `group_cap` | `5` | max-throughput, low-memory, conservative, debug | Max INDEPENDENT steps packed into one lane call |
| `lane_connect_timeout` | `15` | — | Connection timeout in seconds |
| `lane_health_probe` | `True` | — | Probe lanes at startup and park the dead ones |
| `lane_max_hops` | `4` | — | Different lanes tried before a call is declared failed |
| `lane_retries` | `3` | max-throughput | Attempts per model before moving to the next model |
| `lane_timeout` | `240` | max-throughput, conservative | Per-request lane timeout in seconds (was an unbounded 900s total) |
| `log_dir` | `'/tmp'` | — | Directory for orchestrator logs |
| `max_prompt_chars` | `80000` | low-memory | Hard cap on the user prompt sent to a lane |
| `max_rounds` | `3` | conservative | Execute/verify rounds before a step escalates |
| `min_avail_mb` | `600` | low-memory | Pause when available memory falls below this many MB |
| `min_lane_gap_seconds` | `0` | conservative | Minimum seconds between two calls on the same lane |
| `parallel` | `3` | max-throughput, low-memory, conservative, debug | Batch groups in flight at once |
| `plan_json` | `''` | — | Master plan JSON (default: <base>/plan.json) |
| `ram_pct_cap` | `90` | — | Hard ceiling: pause above this system RAM percentage |
| `repo_dir` | `'<repo>'` | — | Repository the fixes are applied to |
| `state_backups` | `True` | debug | Keep a .bak of state before each atomic write |
| `swarm_cap` | `4` | — | Hard concurrency ceiling — never raised at runtime |
| `verbose` | `False` | debug | Verbose lane/step logging |
| `wake_log` | `'/tmp/main_wake.log'` | — | Wake-chain log the escalation signal appends to |
| `workers` | `8` | low-memory | Legacy worker hint (batch packing width) |

---

## Safety & Guardrails

- **Hard Concurrency Ceiling**: `swarm_cap` is hard-limited to 4 to prevent resource exhaustion.
- **Memory Watchdog**: Pauses passes if available system memory falls below `min_avail_mb` or overall usage exceeds `ram_pct_cap`.
- **State Locking**: State operations utilize file locking (`fcntl.flock`) and atomic writes (temporary file + `fsync` + rename) with automatic `.bak` backups.
- **Process Mutual Exclusion**: Escalation solver actively detects running orchestrator instances to eliminate concurrent modification collisions.
- **Verification Sandboxing**: Model-suggested verification commands are strictly checked against a command allowlist and stripped of shell expansion characters.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
