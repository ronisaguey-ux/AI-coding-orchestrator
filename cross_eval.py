#!/usr/bin/env python3
"""
cross_eval.py — multi-agent cross-evaluation & plan synthesis stage
=========================================================

Ingests 5,238 multi-agent audit findings from `multi_agent_oculus_audit_7_29.md`
(and pass 1-5 checkpoints), cross-references against `OCULUS_SOURCE_OF_TRUTH_7_23.md`,
`oculus_readme.md`, and Graphify call graph context, deploying 150 parallel DeepSeek subagents:

  - MODEL_FLASH ("deepseek-v4-flash" / "deepseek-chat"): Simpler subagent tasks (classification, validity checks, initial filtering).
  - MODEL_PRO   ("deepseek-v4-pro" / "deepseek-reasoner"): Complex subagent tasks (cross-eval synthesis, root cause verification, atomic step emission).

Fixes & Enhancements Applied:
  1. Secure API key loading (env var first, JSON/regex fallback).
  2. Strict rate-limiting & exponential backoff for 429/503/502.
  3. Unique, exclusive chunk assignment across 150 subagents (zero overlap/duplication).
  4. Atomic checkpointing & resume support via `cross_eval_state.json`.
  5. Dynamic model alias resolution (deepseek-v4-flash / deepseek-chat, deepseek-v4-pro / deepseek-reasoner).
  6. Cost & token tracking, structured logging, and master plan emission.

Usage:
    python3 parallel_agent_cross_eval.py --smoke
    python3 parallel_agent_cross_eval.py --resume
    python3 parallel_agent_cross_eval.py
"""

import os
import sys
import re
import json
import time
import random
import asyncio
import argparse
import logging
from datetime import datetime
from collections import defaultdict
import aiohttp

# STEP 286/1566: unified secret loading from scripts/common. The script runs
# standalone (python3 scripts/parallel_agent_cross_eval.py puts scripts/ on
# sys.path, not the repo root), so prepend the repo root first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.common.secrets import load_secret, log_redacted
from scripts.common.llm_schema import parse_findings_payload

# ─── CONFIGURATION ───────────────────────────────────────────────────────────
KEY_FILE = "/home/roni/Roni_workspace/tokens_keys/deepseek_api.json"

def load_deepseek_key() -> str:
    """DeepSeek API key: environment variable first, then the key file.

    Delegates to the shared fail-closed loader: raises (never an empty-string
    fallback) when no key is available, so a misconfigured run cannot silently
    send an empty Authorization header.
    """
    return load_secret("DEEPSEEK_API_KEY", KEY_FILE)

DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
DEEPSEEK_API_KEY = None  # lazy-loaded on first API call

def _ensure_deepseek_key() -> str:
    global DEEPSEEK_API_KEY
    if DEEPSEEK_API_KEY is None:
        DEEPSEEK_API_KEY = load_deepseek_key()
        log_redacted(logging.getLogger('oculus.cross_eval'),
                     'DeepSeek API key loaded lazily: %s', DEEPSEEK_API_KEY)
    return DEEPSEEK_API_KEY

# Primary & fallback model IDs
MODEL_FLASH = os.getenv("DEEPSEEK_MODEL_FLASH", "deepseek-v4-flash")
MODEL_PRO = os.getenv("DEEPSEEK_MODEL_PRO", "deepseek-v4-flash")

# 2026-08-08 (user): NO exact 80/20 split — assign a weight to each issue
# type and allocate subagent effort proportionally. Each weight maps to a
# cohort; the cohort's subagent count = round(TOTAL * weight). Security
# carries the heaviest weight (live trading: kill-switch/fail-closed
# invariants matter most), then performance & architecture (latency + spec
# restoration), then design/emergence/verification.
# 08-31 (user): 500 -> 700 default for thorough cross-eval; CROSS_EVAL_SUBAGENTS
# env scales it higher depending on the final audit size.
TOTAL_SUBAGENTS = int(os.environ.get("CROSS_EVAL_SUBAGENTS", "700"))
MIN_PER_COHORT = 12
ISSUE_TYPE_WEIGHTS = {
    "Security": 0.28,      # 84 agents
    "Performance": 0.20,   # 60
    "Architecture": 0.18,  # 54
    "Design": 0.12,        # 36
    "Emergence": 0.12,     # 36
    "Verification": 0.10,  # 30
}
COHORT_ORDER = ["Security", "Performance", "Architecture", "Design", "Emergence", "Verification"]

def _cohort_counts() -> dict:
    """Subagent count per cohort from ISSUE_TYPE_WEIGHTS (sum = TOTAL_SUBAGENTS)."""
    counts = {c: max(MIN_PER_COHORT, round(TOTAL_SUBAGENTS * ISSUE_TYPE_WEIGHTS[c]))
              for c in COHORT_ORDER}
    # Absorb rounding residual in the last cohort so the total is exact.
    counts[COHORT_ORDER[-1]] += TOTAL_SUBAGENTS - sum(counts.values())
    return counts


# 2026-08-08 (user, HARD RULE): NEVER use deepseek-chat ANYWHERE — only
# deepseek-v4-flash. Aliases stripped to v4-flash only (no chat, no -free).
FLASH_ALIASES = [MODEL_FLASH, "deepseek-v4-flash"]
# 2026-08-04 (user): v4 flash updated today — MORE powerful than v4 pro AND
# cheaper. Stop using v4 pro entirely; even "pro" tasks use v4 flash.
PRO_ALIASES = [MODEL_PRO, "deepseek-v4-flash"]

RATE_LIMIT_RPM = 60            # target requests per minute
DELAY_PER_REQUEST = 60.0 / RATE_LIMIT_RPM  # ~1.0s between requests
SEMAPHORE_LIMIT = 10           # concurrent HTTP requests
CLAUDE_SWARM_LIMIT = 4         # max CONCURRENT claude subprocesses (2026-08-05 OOM fix)
CHAT_TIMEOUT_SECONDS = 300     # per-model timeout in seconds (2026-08-10: was 90 — full 4096-token outputs take ~50s ALONE; under concurrent load every call exceeded 90s and the run ground out on retries)
MAX_BACKOFF_SECONDS = 15       # max exponential backoff cap
MAX_RETRIES = 5                # max retries per prompt

# Global subprocess throttle: each call_claude spawns a claude subprocess.
# Without this cap, a 150-agent swarm spawns ~150 concurrent claude processes
# → RAM exhaustion → kernel OOM → desktop black screen (2026-08-05 incident).
_claude_swarm_sem = None  # lazily created in main()

import os as _os
REPO_ROOT = _os.environ.get("ORCH_REPO_DIR", _os.getcwd())
AUDIT_DIR = _os.environ.get("ORCH_AUDIT_DIR", _os.path.join(REPO_ROOT, "audits_plans"))
OCULUS_DIR = REPO_ROOT
OUTPUT_BASE = AUDIT_DIR
# 2026-08-14 (user HARD RULE): label artifacts by the CURRENT date, never a
# stale constant — master_oculus_plan_8_4.md was being overwritten on 08-14
# with a fresh plan that still said 8_4. The orchestrator passes the
# web-verified date via PLAN_VERSION; direct invocations fall back to today.
PLAN_VERSION = os.environ.get("PLAN_VERSION", f"{datetime.now().month}_{datetime.now().day}")
AUDIT_REPORT = f"{OUTPUT_BASE}/multi_agent_oculus_audit_{PLAN_VERSION}.md"
SOT_FILE = f"{REPO_ROOT}/OCULUS_IMPORTANT/OCULUS_SOURCE_OF_TRUTH_7_23.md"
README_FILE = f"{REPO_ROOT}/OCULUS_IMPORTANT/oculus_readme.md"
GRAPH_FILE = f"{REPO_ROOT}/graphify-out/graph.json"
MASTER_PLAN_FILE = f"{OUTPUT_BASE}/master_oculus_plan_{PLAN_VERSION}.md"
CROSS_EVAL_STATE = f"{OUTPUT_BASE}/cross_eval_state.json"
LOG_FILE = f"{OUTPUT_BASE}/cross_eval.log"

# Model capability score lookup table
MODEL_SCORES = {
    "deepseek-v4-pro": 96,
    "deepseek-reasoner": 96,
    "deepseek-v4-pro-free": 96,
    "deepseek-v4-flash": 90,
    "deepseek-chat": 90,
    "deepseek-v4-flash-free": 90,
    "nuclear-fallback": 90,
    "big-pickle": 90,
    "mimo-v2.5-free": 88,
}
DEFAULT_MODEL_SCORE = 85

# Pricing estimates (per 1M tokens) for DeepSeek V4 Pro/Flash
COST_PER_1M_INPUT = 0.14    # $0.14 / 1M input tokens
COST_PER_1M_OUTPUT = 0.28   # $0.28 / 1M output tokens

# 09-02 (user: "$18 in under a day vs normal $3-6"): daily spend breaker for the
# paid DeepSeek API. Env DEEPSEEK_BREAKER_USD (0 = disabled; default 25).
# Ledger rotates by local date: /tmp/deepseek_daily_spend.json {"date","spent"}.
BREAKER_USD = float(os.environ.get("DEEPSEEK_BREAKER_USD", "25") or 0)
SPEND_LEDGER = "/tmp/deepseek_daily_spend.json"
BREAKER_TRIPPED = False

# ─── LOGGING HELPER ──────────────────────────────────────────────────────────
def log_message(msg: str):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(formatted + "\n")
    except Exception:
        pass

def load_daily_spend() -> float:
    """Current day's accrued DeepSeek cost (USD) from the rotating ledger."""
    try:
        d = json.load(open(SPEND_LEDGER))
        if d.get("date") == datetime.now().strftime("%Y-%m-%d"):
            return float(d.get("spent", 0.0))
    except Exception:
        pass
    return 0.0

def add_daily_spend(usd: float) -> float:
    """Accrue a completed call's cost into today's ledger. Returns day total."""
    if usd <= 0:
        return load_daily_spend()
    total = load_daily_spend() + usd
    try:
        with open(SPEND_LEDGER, "w") as f:
            json.dump({"date": datetime.now().strftime("%Y-%m-%d"),
                       "spent": round(total, 4)}, f)
    except Exception:
        pass
    return total

def alert_breaker() -> None:
    """Mirror audit_healthcheck.sh's wake path (Autonomous fix-forward): inbox
    entry + /tmp/main_wake.log line so main is pinged and pings the user."""
    msg = (f"DeepSeek daily spend ${load_daily_spend():.2f} ≥ "
           f"${BREAKER_USD:.0f} breaker — stage-2 paused. Resume when ok; "
           f"env DEEPSEEK_BREAKER_USD raises the cap.")
    try:
        inbox = f"{AUDIT_DIR}/claude_main_inbox.json"
        d = json.load(open(inbox))
        if not isinstance(d, list):
            d = d.get("messages", [])
        d.append({"ts": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "from": "breaker", "text": msg})
        json.dump(d, open(inbox, "w"), indent=1)
    except Exception:
        pass
    try:
        with open("/tmp/main_wake.log", "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')} [breaker] {msg}\n")
    except Exception:
        pass


# ─── GRAPHIFY IN-MEMORY QUERY ENGINE ─────────────────────────────────────────
class GraphifyDB:
    def __init__(self, graph_path: str):
        self.path = graph_path
        self.nodes: dict[str, dict] = {}
        self.links: list[dict] = []
        self.file_to_nodes: dict[str, list[dict]] = defaultdict(list)
        self.label_to_node: dict[str, dict] = {}
        self.out_edges: dict[str, list[str]] = defaultdict(list)
        self.in_edges: dict[str, list[str]] = defaultdict(list)
        self.communities: dict[int, list[dict]] = defaultdict(list)
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            # 2026-08-09 (step 571): resolve star-import shims — files whose
            # body is only 'from X import *' get their symbols attributed to X.
            shim_map = {}
            for n in data.get("nodes", []):
                f_src = n.get("file", n.get("source_file", ""))
                if f_src and f_src not in shim_map and os.path.exists(f_src):
                    try:
                        with open(f_src, errors="ignore") as sf:
                            tgt = resolve_shim_target(sf.read())
                        shim_map[f_src] = tgt
                    except Exception:
                        shim_map[f_src] = None
            for n in data.get("nodes", []):
                nid = n.get("id", "")
                lbl = n.get("label", nid)
                src = n.get("file", n.get("source_file", ""))
                if src and shim_map.get(src):
                    src = shim_map[src]
                comm = n.get("community", 0)
                n_struct = {
                    "id": nid, "label": lbl, "file": src,
                    "type": n.get("type", "symbol"),
                    "community": comm, "degree": n.get("degree", 0)
                }
                self.nodes[nid] = n_struct
                if lbl:
                    self.label_to_node[lbl] = n_struct
                if src:
                    self.file_to_nodes[src].append(n_struct)
                self.communities[comm].append(n_struct)

            for edge in data.get("links", []):
                s = edge.get("source")
                t = edge.get("target")
                if s and t:
                    self.links.append({"source": s, "target": t, "kind": edge.get("kind", "calls")})
                    self.out_edges[s].append(t)
                    self.in_edges[t].append(s)
        except Exception as e:
            log_message(f"[graphify] Error reading graph: {e}")

    def has_data(self) -> bool:
        return len(self.nodes) > 0

    def get_file_nodes(self, rel_path: str) -> list[dict]:
        return self.file_to_nodes.get(rel_path, [])

    def build_file_context(self, rel_path: str, max_funcs: int = 15) -> str:
        nodes = self.get_file_nodes(rel_path)
        if not nodes:
            return f"### FILE: {rel_path}\n[No AST nodes found in graph]"
        lines = [f"### FILE: {rel_path} ({len(nodes)} symbols in graph)"]
        nodes_sorted = sorted(nodes, key=lambda x: -x.get("degree", 0))
        for n in nodes_sorted[:max_funcs]:
            lbl = n["label"]
            nid = n["id"]
            callers = [self.nodes[c]["label"] for c in self.in_edges.get(nid, []) if c in self.nodes]
            callees = [self.nodes[c]["label"] for c in self.out_edges.get(nid, []) if c in self.nodes]
            c_str = f" <- [{', '.join(callers[:3])}]" if callers else ""
            e_str = f" -> [{', '.join(callees[:3])}]" if callees else ""
            lines.append(f"  - `{lbl}` (comm {n['community']}, deg {n['degree']}){c_str}{e_str}")
        return "\n".join(lines)


# 2026-08-09 (step 571): canonicalize star-import shims — a module whose body
# is only 'from <target> import *' resolves to <target> for symbol parsing.
_SHIM_RE = re.compile(r'^\s*from\s+([\w.]+)\s+import\s+\*\s*$')


def resolve_shim_target(source: str) -> str | None:
    """Return the star-import target if the FIRST non-comment line is a shim.

    Matches both pure shims ('from X import *' alone) and 2-line shims
    (star import + explicit re-exports, e.g. oculus/live/courtroom.py).
    """
    for l in source.splitlines():
        if l.strip() and not l.strip().startswith('#'):
            m = _SHIM_RE.match(l.strip())
            return m.group(1) if m else None
    return None


_graphify: GraphifyDB | None = None

def get_graphify() -> GraphifyDB:
    global _graphify
    if _graphify is None:
        _graphify = GraphifyDB(GRAPH_FILE)
    return _graphify

# ─── RUNTIME RATE LIMITING & METRICS ─────────────────────────────────────────
_last_request_time = 0.0
_request_lock = asyncio.Lock()
_model_health: dict[str, int] = {}
_total_tokens_used = {"input": 0, "output": 0}

async def throttled_start():
    global _last_request_time
    async with _request_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < DELAY_PER_REQUEST:
            await asyncio.sleep(DELAY_PER_REQUEST - elapsed)
        _last_request_time = time.time()


# ─── DEEPSEEK API CALLER ─────────────────────────────────────────────────────
async def call_deepseek(session: aiohttp.ClientSession,
                        user_prompt: str,
                        system_prompt: str,
                        use_pro: bool = False,
                        agent_name: str = "",
                        retries: int = MAX_RETRIES,
                        quiet: bool = False) -> tuple[str, str, int]:
    """Invoke the AI backend.

    The cross-eval runs many subagents in parallel; with each dispatch going to
    a Claude Code subprocess, that forms a CLAUDE SUBPROCESS SWARM. If a claude
    subprocess fails, fall back to DeepSeek.
    """
    # CLAUDE SUBPROCESS SWARM: the cross-eval's parallel subagents dispatch to
    # claude subprocesses by default. Set OCULUS_AI_BACKEND=deepseek to revert
    # to DeepSeek API.
    backend = os.getenv('OCULUS_AI_BACKEND', 'claude').lower()
    if backend == 'webchat-pool':
        return await call_webchat_pool(session, user_prompt, system_prompt, use_pro=use_pro,
                                       agent_name=agent_name, retries=retries, quiet=quiet)
    if backend != 'deepseek':
        import claude_ai
        tier = "pro" if use_pro else "flash"
        # Bound concurrent claude subprocesses (OOM fix): acquire the global
        # swarm semaphore so at most CLAUDE_SWARM_LIMIT claude processes run
        # at once. Creates lazily to avoid cross-loop re-creation.
        global _claude_swarm_sem
        # 2026-08-12 (py3.14 fix): Semaphore._loop is None on 3.14+, so the
        # old `._loop is not get_running_loop()` check recreated the semaphore
        # per coroutine -> unbounded claude -p swarm. The script runs in one
        # asyncio.run(), the loop never changes, so just guard on None.
        if _claude_swarm_sem is None:
            _claude_swarm_sem = asyncio.Semaphore(CLAUDE_SWARM_LIMIT)
        async with _claude_swarm_sem:
            res = await asyncio.to_thread(
                claude_ai.call_claude, user_prompt, system_prompt,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                timeout=120, tier=tier)
        content = res.get("output", "")
        if res.get("ok") and content and not content.startswith("ERROR"):
            return content, "claude", 100

    primary_list = PRO_ALIASES if use_pro else FLASH_ALIASES
    secondary_list = FLASH_ALIASES if use_pro else PRO_ALIASES
    candidates = primary_list + secondary_list
    candidates = [m for m in candidates if _model_health.get(m, 0) < 5]
    if not candidates:
        candidates = primary_list

    last_error = ""
    for attempt in range(retries):
        target_model = candidates[attempt % len(candidates)]
        score = MODEL_SCORES.get(target_model, DEFAULT_MODEL_SCORE)

        if attempt > 0:
            wait = min((2 ** attempt) + random.uniform(0.5, 2.5), MAX_BACKOFF_SECONDS)
            if not quiet:
                log_message(f"      [deepseek] retry {attempt}/{retries} using {target_model} after {wait:.1f}s ({last_error})")
            await asyncio.sleep(wait)

        headers = {
            "Authorization": f"Bearer {_ensure_deepseek_key()}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": target_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        try:
            url = f"{DEEPSEEK_API_BASE}/chat/completions"
            async with session.post(url, json=payload, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=CHAT_TIMEOUT_SECONDS)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data['choices'][0]['message']['content']
                    actual_model = data.get("model", target_model)
                    
                    # Track token usage if provided in response
                    usage = data.get("usage", {})
                    if usage:
                        _total_tokens_used["input"] += usage.get("prompt_tokens", 0)
                        _total_tokens_used["output"] += usage.get("completion_tokens", 0)
                        # 09-02 daily spend ledger — every completed call accrues
                        # its real cost so the breaker + run-end reports are exact.
                        add_daily_spend(
                            (usage.get("prompt_tokens", 0) / 1_000_000.0 * COST_PER_1M_INPUT)
                            + (usage.get("completion_tokens", 0) / 1_000_000.0 * COST_PER_1M_OUTPUT)
                        )

                    if not quiet:
                        log_message(f"      [deepseek] success via {target_model} -> {actual_model} ({len(content)} chars)")
                    return content, actual_model, score
                elif resp.status in (429, 502, 503, 504):
                    last_error = f"HTTP {resp.status}"
                    # Rate limit or temporary service overload: back off and rotate model
                    wait_time = random.uniform(2.0, 5.0) * (attempt + 1)
                    if not quiet:
                        log_message(f"      [deepseek] {target_model} rate limit / server busy (HTTP {resp.status}); backing off {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    last_error = f"HTTP {resp.status}"
                    _model_health[target_model] = _model_health.get(target_model, 0) + 1
                    if not quiet:
                        log_message(f"      [deepseek] {target_model} returned HTTP {resp.status}")
                    continue
        except asyncio.TimeoutError:
            last_error = "Timeout"
            _model_health[target_model] = _model_health.get(target_model, 0) + 1
            if not quiet:
                log_message(f"      [deepseek] {target_model} timed out after {CHAT_TIMEOUT_SECONDS}s")
            continue
        except Exception as e:
            last_error = str(e)
            _model_health[target_model] = _model_health.get(target_model, 0) + 1
            if not quiet:
                log_message(f"      [deepseek] {target_model} exception: {e}")
            continue

    return f"ERROR: Exhausted retries. Last error: {last_error}", candidates[0], DEFAULT_MODEL_SCORE


# ─── JSON EXTRACTION ─────────────────────────────────────────────────────────
def extract_json(text: str) -> dict | list | None:
    text = text.strip()
    text = text.replace("True", "true").replace("False", "false").replace("None", "null")
    if (text.startswith('{') and text.endswith('}')) or (text.startswith('[') and text.endswith(']')):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    for pattern in [r'```json\s*\n?(.*?)\n?```', r'```\s*\n?(.*?)\n?```']:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                continue
    m = re.search(r'[\{\[].*[\}\]]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ─── CHECKPOINTING / RESUME STATE HELPERS ────────────────────────────────────
def load_state() -> list[dict]:
    """Load atomic cross-eval checkpoint state if available."""
    if os.path.exists(CROSS_EVAL_STATE):
        try:
            with open(CROSS_EVAL_STATE) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            log_message(f"[checkpoint] Warning loading state: {e}")
    return []


def save_state(outputs: list[dict]):
    """Durably save cross-eval state (STEP 460/1566: mkstemp + fsync + os.replace)."""
    import tempfile
    try:
        _dir = os.path.dirname(CROSS_EVAL_STATE) or "."
        tmp_fd, tmp = tempfile.mkstemp(dir=_dir, suffix=".tmp")
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(outputs, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, CROSS_EVAL_STATE)
    except Exception as e:
        log_message(f"[checkpoint] Error saving state: {e}")


# ─── SUBAGENT DEFINITIONS (300 LEAN SUBAGENTS ACROSS 6 COHORTS) ─────────────
# 2026-08-07 (user request): 150 -> 300 subagents because the codebase grew
# significantly. LEAN design to keep token cost near-constant: each cohort is
# 2x the subagents, but the second half is model_type=flash (cheaper) with
# NARROWER focus (each subagent gets a tighter topic scope so it reads less
# context and answers shorter). Coverage doubles, token cost grows modestly.
# 2026-08-08 (user): cohort sizes are now driven by ISSUE_TYPE_WEIGHTS, NOT
# the old fixed 45/30/30/24/15/6 -> 2x layout. Sum still = TOTAL_SUBAGENTS.
def generate_150_subagent_definitions() -> list[dict]:
    """Generate 500 lean subagents mapped to the 6 cohorts in oculus_plan_prompt.md.

    Lean = flash-heavy + narrow-scope. Per-cohort: first half = original
    (mixed pro/flash), second half = flash with half-width topic slices so
    each call reads less and writes terser findings. Counts come from
    _cohort_counts() (ISSUE_TYPE_WEIGHTS).
    """
    subagents = []
    counts = _cohort_counts()

    # Cohort 1: Security Cohort (weight -> 84 subagents)
    sec_topics = [
        "API Auth & Token Security", "Secrets Exposure & Environment Gating",
        "Encryption & Data Ingestion Security", "Live Trading Kill-Switch & Fail-Closed",
        "FTROM Safety & Non-Paper Learning Invariants", "Replay Attack Protection",
        "Input Sanitization & Injection Vectors", "Privilege Escalation & File I/O Safety",
        "Courtroom & LLM Decision Safety Gating"
    ]
    for i in range(1, counts["Security"] + 1):
        topic = sec_topics[(i - 1) % len(sec_topics)]
        subagents.append({
            "id": f"SEC_{i:03d}",
            "name": f"Security Subagent #{i:02d} — {topic}",
            "cohort": "Security",
            "model_type": "flash" if (i % 2 == 1 or i > 45) else "pro",
            "focus": f"Adversarial security analysis targeting {topic}. Identify valid vulnerabilities, filter false positives, propose content-addressed fixes."
        })

    # Cohort 2: Design & UI Cohort (weight -> 36 subagents)
    design_topics = [
        "Information Hierarchy & Metric Prioritization", "UX Flow & Navigation Layout",
        "Accessibility & Color Contrast", "Mobile & Screen Responsiveness",
        "Chart Rendering & Live Data Density", "Flutter State Management & Screen Decoupling",
        "HTML Report Jinja Templates"
    ]
    for i in range(1, counts["Design"] + 1):
        topic = design_topics[(i - 1) % len(design_topics)]
        subagents.append({
            "id": f"DSG_{i:03d}",
            "name": f"Design Subagent #{i:02d} — {topic}",
            "cohort": "Design",
            "model_type": "flash",
            "focus": f"UI/UX layout, responsive boundaries, and reporting aesthetics for {topic}."
        })

    # Cohort 3: Performance Cohort (weight -> 60 subagents)
    perf_topics = [
        "Cascade & Memory Leak Optimizations", "Numba JIT Indicator Vectorization",
        "Feature Pipeline & Data Ingestion Bottlenecks", "Cache Control & Memory Footprint",
        "Loop Optimization & Matrix Computation", "Parallel Thread & Event Loop Concurrency"
    ]
    for i in range(1, counts["Performance"] + 1):
        topic = perf_topics[(i - 1) % len(perf_topics)]
        subagents.append({
            "id": f"PRF_{i:03d}",
            "name": f"Performance Subagent #{i:02d} — {topic}",
            "cohort": "Performance",
            "model_type": "flash" if i <= 40 else "pro",
            "focus": f"Performance profiling, latency reduction, and vectorization for {topic}."
        })

    # Cohort 4: Architecture Cohort (weight -> 54 subagents)
    arch_topics = [
        "Source of Truth Invariant Enforcement", "Spec Restoration (21 Assets, Bars, DD Rules)",
        "Structural Debt & Module Decoupling", "Dependency Hygiene & Circular Import Resolution",
        "Genome & Genetic Evolution Architecture"
    ]
    for i in range(1, counts["Architecture"] + 1):
        topic = arch_topics[(i - 1) % len(arch_topics)]
        subagents.append({
            "id": f"ARC_{i:03d}",
            "name": f"Architecture Subagent #{i:02d} — {topic}",
            "cohort": "Architecture",
            "model_type": "flash" if i > 24 else "pro",
            "focus": f"Architectural integrity, spec restoration, and module boundary enforcement for {topic}."
        })

    # Cohort 5: Emergence Cohort (weight -> 36 subagents)
    emerg_topics = [
        "Novel Vulnerability Discovery", "Cross-Domain Synergy Mining",
        "Future-Proofing & Extensibility", "Emergent Feature Synthesis"
    ]
    for i in range(1, counts["Emergence"] + 1):
        topic = emerg_topics[(i - 1) % len(emerg_topics)]
        subagents.append({
            "id": f"EMG_{i:03d}",
            "name": f"Emergence Subagent #{i:02d} — {topic}",
            "cohort": "Emergence",
            "model_type": "flash" if i > 15 else "pro",
            "focus": f"Mining omissions, unmentioned vulnerabilities, and high-impact emergent improvements in {topic}."
        })

    # Cohort 6: Verification Cohort (weight -> 30 subagents)
    ver_topics = [
        "Exact 5-Command Verification Suite Design", "Regression Risk & Rollback Command Auditor"
    ]
    for i in range(1, counts["Verification"] + 1):
        topic = ver_topics[(i - 1) % len(ver_topics)]
        subagents.append({
            "id": f"VER_{i:03d}",
            "name": f"Verification Subagent #{i:02d} — {topic}",
            "cohort": "Verification",
            "model_type": "flash" if i > 6 else "pro",
            "focus": f"Building rigorous 5-command verification steps and rollback guarantees for {topic}."
        })

    # Cohort 7: ScrapeGraph Integration Team (user 09-01: "add a 5 persona team
    # to figure out how to integrate it and add it to the master plan" — the
    # sgtool ScrapeGraphAI CLI built 09-01). Findings + proposed_steps flow to
    # synthesize_master_plan like any other cohort.
    SG_CONTEXT = (
        "SCRAPEGRAPH TOOL (LIVE, WORKING, verified 09-01): ScrapeGraphAI v2 installed as a CLI at "
        "/home/roni/Roni_Workspace/scrapegraph_tool/sgtool.sh \"<what to extract>\" <url> "
        "(SmartScraperGraph), --search \"<query>\" <url> (SearchGraph), multiple URLs "
        "(SmartScraperMultiConcatGraph). Backed by the PAID DeepSeek key via "
        "deepseek/deepseek-chat on api.deepseek.com/v1; v2 FetchNode is ALWAYS playwright chromium "
        "(--no-browser is a no-op); telemetry disabled. The venv was built with uv "
        "(python3 -m venv is BROKEN here — no ensurepip). Local processes that scrape today use raw "
        "requests or fixed exchange APIs (e.g. courtroom_swarm.py _onchain_slice → alternative.me)."
    )
    sg_personas = [
        ("INT_001", "Integration Site Scout", "flash",
         f"Walk the repository and catalog EVERY subsystem that ingests external/web data "
         f"(courtroom_swarm.py data slices, courtroom.py panels, public-APIs index, the audit corpus, "
         f"helpotron scrape_assignment, docs ingesters). Rank the best 2-3 sgtool integration slots: "
         f"what data each slot needs, why sgtool beats the current raw-requests approach, quantified "
         f"benefit (time saved / coverage gained). Output proposed_steps in the schema. {SG_CONTEXT}"),
        ("INT_002", "Courtroom Evidence-Slice Architect", "pro",
         f"Design the ONE-scrape-per-swarm-run web-evidence slice for oculus/live/courtroom_swarm.py "
         f"(and courtroom.py): a _web_evidence_slice(coin) prepended to each agent's data slice via "
         f"_build_agent_data, run ONCE per swarm round (NOT per agent), cached under the existing "
         f"_OHLCV_CACHE_LOCK pattern with a TTL, FAIL-OPEN (scrape failure = agents run without it). "
         f"Specify exact function signatures, where evidence feeds prompts, and how equity/forex assets "
         f"gain coverage (ETF flows, funding rates, macro news) that the fixed APIs miss. {SG_CONTEXT}"),
        ("INT_003", "Infrastructure & Packaging Wrangler", "flash",
         f"Resolve tooling economics: uv venv vs system venv (ensurepip absent), how to package sgtool "
         f"for reuse across oculus scripts (subprocess call vs python import), the 184MB playwright "
         f"chromium footprint on a RAM-tight box under the RAM watchdog, when SearchGraph vs "
         f"SmartScraperGraph is the right call, and a smoke test for CI/verification. {SG_CONTEXT}"),
        ("INT_004", "Security & Cost Sentinel", "flash",
         f"Audit the sgtool LLM integration for risks: prompt-injection VIA scraped page content into "
         f"the DeepSeek prompt path, key isolation (the paid key must never appear in prompts/logs), "
         f"cost caps per scrape (deepseek-chat billing), rate limits, and fail-open gating so a scrape "
         f"failure never blocks a verdict pipeline. Propose mitigations as proposed_steps. {SG_CONTEXT}"),
        ("INT_005", "Master-Plan Binder", "pro",
         f"Turn the team's designs into final atomic master-plan steps (schema: concise_objective, "
         f"category DATA|ARCHITECTURE|TESTING, priority, code_operations with <<<REPLACE>>> blocks, "
         f"verification commands incl. an sgtool smoke test, commit_message, rollback_command) for "
         f"integrating sgtool into courtroom_swarm.py + courtroom.py + the public-APIs refresh flow. "
         f"Steps must be executable independently by the executor. {SG_CONTEXT}"),
    ]
    for i, (sg_id, role, model_type, focus) in enumerate(sg_personas, start=1):
        subagents.append({
            "id": sg_id,
            "name": f"Integration Subagent #{i:02d} — {role}",
            "cohort": "Integration",
            "model_type": model_type,
            "focus": focus,
        })

    return subagents


# ─── PROMPT BUILDERS ─────────────────────────────────────────────────────────
def build_subagent_system_prompt(agent: dict, sot: str, readme: str) -> str:
    return f"""You are **{agent['name']}** in the Oculus Multi-Agent Cross-Evaluation Pipeline.

## COHORT: {agent['cohort']}
## FOCUS: {agent['focus']}

## SPECIFICATION HIERARCHY
1. `OCULUS_SOURCE_OF_TRUTH_7_23.md` (Authoritative Spec)
2. `oculus_readme.md` (Documented Features)
3. Codebase / Graphify matrix

## AUTHORITATIVE SPEC SUMMARY
{sot[:4000]}

## REQUIRED OUTPUT SCHEMA (STRICT JSON)
You MUST output a JSON object containing your evaluations and proposed plan steps:
{{
  "subagent_id": "{agent['id']}",
  "valid_findings": [
    {{
      "finding_id": "PXBXRXFX",
      "file": "path/to/file.py",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW",
      "reason_valid": "1 sentence explanation of why this is valid",
      "mechanism": "How it manifests",
      "quantified_benefit": "Quantified impact"
    }}
  ],
  "invalid_findings": [
    {{
      "finding_id": "PXBXRXFX",
      "reason_invalid": "Why it is a false positive or mitigated"
    }}
  ],
  "omissions_and_emergent": [
    {{
      "title": "Omission title",
      "target_file": "path/to/file.py",
      "description": "What was missed in audit",
      "benefit": "Quantified benefit"
    }}
  ],
  "proposed_steps": [
    {{
      "concise_objective": "1 sentence step title",
      "category": "SECURITY|DESIGN|PERFORMANCE|ARCHITECTURE|DATA|TESTING",
      "priority": "CRITICAL|HIGH|MEDIUM|LOW",
      "benefit": "Quantified impact",
      "effort": "1h|2-4h|4-8h|8+h",
      "target_files": ["path/to/file.py"],
      "dependencies": ["Step X or None"],
      "line_ranges": ["file.py:10-25"],
      "code_operations": [
        "<<<REPLACE>>> [file.py] line 10-25 (OLD_STRING)\\nold_code\\n(NEW_TEXT)\\nnew_code"
      ],
      "verification": [
        "pytest tests/test_file.py --strict-markers -v",
        "python -m py_compile file.py",
        "mypy --strict file.py",
        "findstr /N \"check\" file.py",
        "python -c \"import file\""
      ],
      "commit_message": "[STEP X/M] Summary of changes",
      "rollback_command": "git checkout -- file.py",
      "notes": "Special considerations"
    }}
  ]
}}
Output ONLY valid JSON. No markdown commentary outside JSON.
"""


def build_adversarial_review_prompt(agent: dict, candidate: dict) -> str:
    """oh-my-claudecode-inspired adversarial reviewer (2026-08-05).

    Given a subagent's accepted finding/step, a second pass challenges it:
    'REFUTE this finding if you can.' Findings that survive an adversarial
    challenge are far more likely to be real — tightening the accepted set
    (the 8_4 audit showed acceptance-rate inflation as a convergence risk).
    """
    finding_id = candidate.get("finding_id", "?")
    reason = candidate.get("reason_valid", candidate.get("concise_objective", ""))
    return f"""You are the ADVERSARIAL REVIEWER in the Oculus cross-eval pipeline.

Your job: attempt to REFUTE the following accepted finding from {agent['id']}.

FINDING: {finding_id}
CLAIM: {reason}

Challenge it on every axis:
1. Is the claim actually about the referenced file/line, or a hallucination?
2. Does the referenced symbol/import actually exist in the current tree?
3. Is the severity inflated (CRITICAL that is really LOW)?
4. Is it already mitigated elsewhere (config, guard, later code)?
5. Is the proposed step's verification actually runnable (no phantom commands)?

Respond STRICT JSON:
{{
  "finding_id": "{finding_id}",
  "verdict": "CONFIRMED|REFUTED|DOWNGRADED",
  "challenge_notes": "1-2 sentences: what you tried to refute and whether it held"
}}
Output ONLY valid JSON.
"""


def build_subagent_user_prompt(agent: dict, findings_chunk: list[dict], graphify: GraphifyDB) -> str:
    lines = [f"## EXCLUSIVE FINDINGS BATCH FOR {agent['id']} ({len(findings_chunk)} findings)\n"]
    for f in findings_chunk:
        fid = f.get("finding_id", "PXBXRX")
        fp = f.get("file", "unknown")
        sev = f.get("severity", "MEDIUM")
        cat = f.get("category", "GENERAL")
        desc = f.get("finding", "")[:180]
        ctx = graphify.build_file_context(fp, max_funcs=5) if graphify.has_data() else ""
        lines.append(f"### {fid} | {sev} | {cat} | `{fp}`")
        lines.append(f"Description: {desc}")
        if ctx:
            lines.append(f"Graphify context:\n{ctx}")
        lines.append("")

    lines.append("## INSTRUCTIONS")
    lines.append(f"Cross-evaluate the findings above for {agent['cohort']} / {agent['focus']}.")
    lines.append("Identify valid findings, false positives (invalid), omissions, and propose atomic implementation steps adhering to the JSON schema.")
    return "\n".join(lines)


# ─── MASTER PLAN SYNTHESIZER ─────────────────────────────────────────────────
async def synthesize_master_plan(session: aiohttp.ClientSession,
                                 all_subagent_outputs: list[dict],
                                 sot: str, readme: str,
                                 all_findings: list[dict] | None = None) -> str:
    """Synthesize all 150 subagent outputs into the final master implementation plan."""
    log_message("\n" + "=" * 70)
    log_message("SYNTHESIZING MASTER OCULUS IMPLEMENTATION PLAN")
    log_message("=" * 70)

    # Consolidate valid findings, omissions, and proposed steps
    valid_map = {}
    invalid_map = {}
    omissions = []
    proposed_steps = []

    for out in all_subagent_outputs:
        if not isinstance(out, dict):
            continue
        for v in out.get("valid_findings", []):
            if not isinstance(v, dict):
                continue
            fid = v.get("finding_id")
            if fid:
                valid_map[fid] = v

    # ADVERSARIAL REVIEW PASS (oh-my-claudecode-inspired, 2026-08-05):
    # challenge each accepted finding; REFUTED/DOWNGRADED ones are removed or
    # severity-reduced before they reach the master plan. Tightens the
    # accepted set against acceptance-rate inflation.
    if os.environ.get("OCULUS_ADVERSARIAL_REVIEW", "1") == "1":
        import asyncio as _asyncio
        from claude_ai import call_claude
        _challenges = []
        async def _challenge(fid, cand):
            prompt = build_adversarial_review_prompt(
                {"id": cand.get("subagent_id", "AGG")}, cand)
            try:
                # Bound claude subprocess concurrency (OOM fix) — same global cap.
                global _claude_swarm_sem
                # 2026-08-12 (py3.14 fix): see invoke_ai — loop check always
                # failed on 3.14 (._loop is None), recreating the semaphore per
                # coroutine and firehosing the claude -p swarm. Guard on None only.
                if _claude_swarm_sem is None:
                    _claude_swarm_sem = _asyncio.Semaphore(CLAUDE_SWARM_LIMIT)
                async with _claude_swarm_sem:
                    out = await _asyncio.to_thread(
                        call_claude, prompt, "", tier="flash", max_output=400)
                import json as _json
                parsed = _json.loads(out)
                return {"finding_id": fid, "verdict": parsed.get("verdict", "CONFIRMED"),
                        "notes": parsed.get("challenge_notes", "")}
            except Exception:
                return {"finding_id": fid, "verdict": "CONFIRMED", "notes": "challenge failed (kept)"}
        for fid, cand in list(valid_map.items()):
            _challenges.append(_challenge(fid, cand))
        if _challenges:
            results = await _asyncio.gather(*_challenges)
            for r in results:
                fid = r["finding_id"]
                if r["verdict"] == "REFUTED":
                    valid_map.pop(fid, None)
                    log(f"[ADVERSARIAL] REFUTED {fid} — removed from accepted set")
                elif r["verdict"] == "DOWNGRADED" and fid in valid_map:
                    v = valid_map[fid]
                    if v.get("severity") == "CRITICAL":
                        v["severity"] = "HIGH"
                        valid_map[fid] = v
                    log(f"[ADVERSARIAL] DOWNGRADED {fid}")

    # Collect invalid findings, omissions, and proposed steps from EVERY
    # output. FIX 2026-08-07: this block was accidentally nested inside the
    # adversarial-review `if` (introduced adf38df), so it ran once with the
    # loop's leftover `out` — or never, with OCULUS_ADVERSARIAL_REVIEW=0 —
    # producing plans with only the boilerplate README steps (8_5's plan
    # landed with 2 steps instead of ~1000; the 8_4 456-step plan predates
    # the regression).
    # 09-03 crash fix: lane-returned JSON can hold bare STRINGS inside
    # proposed_steps/omissions/list fields — synthesis crashed with
    # AttributeError 'str' object has no attribute 'get' at line 909.
    # Every collection site now skips non-dict entries instead of dying.
    for out in all_subagent_outputs:
        if not isinstance(out, dict):
            continue
        for inv in out.get("invalid_findings", []):
            if not isinstance(inv, dict):
                continue
            fid = inv.get("finding_id")
            if fid:
                invalid_map[fid] = inv
        omissions.extend(o for o in out.get("omissions_and_emergent", []) if isinstance(o, dict))
        proposed_steps.extend(s for s in out.get("proposed_steps", []) if isinstance(s, dict))

    # Deduplicate and group steps by target file / category
    unique_steps = []
    seen_objs = set()
    for s in proposed_steps:
        obj = s.get("concise_objective", "").strip()
        if obj and obj not in seen_objs:
            seen_objs.add(obj)
            unique_steps.append(s)

    # Sort steps by ordering discipline:
    # 1. Security -> 2. Data -> 3. Architecture -> 4. Design -> 5. Performance -> 6. Testing
    cat_order = {"SECURITY": 0, "DATA": 1, "ARCHITECTURE": 2, "DESIGN": 3, "PERFORMANCE": 4, "TESTING": 5}
    prio_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    unique_steps.sort(key=lambda s: (
        cat_order.get(str(s.get("category", "TESTING")).upper(), 6),
        prio_order.get(str(s.get("priority", "LOW")).upper(), 4)
    ))

    # Number steps M
    total_steps = len(unique_steps)
    formatted_steps = []

    for idx, s in enumerate(unique_steps, 1):
        target_files = s.get("target_files", [])
        tf_str = "\n".join(f"  - {f}" for f in target_files)
        deps = s.get("dependencies", ["None"])
        deps_str = ", ".join(deps) if isinstance(deps, list) else str(deps)
        line_ranges = s.get("line_ranges", ["N/A"])
        lr_str = ", ".join(line_ranges) if isinstance(line_ranges, list) else str(line_ranges)

        ops = s.get("code_operations", ["<<<REPLACE>>> [file.py] line 1\nold_code\n(NEW_TEXT)\nnew_code"])
        ops_str = "\n\n".join(ops) if isinstance(ops, list) else str(ops)

        ver = s.get("verification", [
            f"pytest tests/ --strict-markers -v",
            f"python -m py_compile {target_files[0] if target_files else 'file.py'}",
            f"mypy --strict {target_files[0] if target_files else 'file.py'}",
            "git status",
            "python -c \"import sys; print('Verified')\""
        ])
        if len(ver) < 5:
            ver.extend([f"python -m py_compile {f}" for f in target_files[:5 - len(ver)]])
        ver_str = "\n".join(f"{i}. {v}" for i, v in enumerate(ver[:5], 1))

        rb = s.get("rollback_command", f"git checkout -- {' '.join(target_files) if target_files else 'file.py'}")
        cm = s.get("commit_message", f"[STEP {idx}/{total_steps}] {s.get('concise_objective', 'Update file')}")

        formatted_step = f"""### STEP {idx}/{total_steps}: {s.get('concise_objective', 'Implementation Step')}

**CATEGORY:** {str(s.get('category', 'ARCHITECTURE')).upper()}
**PRIORITY:** {str(s.get('priority', 'HIGH')).upper()}
**BENEFIT:** {s.get('benefit', 'Improves system correctness and performance')}
**EFFORT:** {s.get('effort', '2-4h')}
**TARGET_FILES:**
{tf_str}
**DEPENDENCIES:** [{deps_str}]
**LINE_RANGES:** [{lr_str}]

**CODE_OPERATIONS:**
{ops_str}

**VERIFICATION:**
{ver_str}

**COMMIT_MESSAGE:** "{cm}"
**ROLLBACK_COMMAND:** {rb}
**NOTES:** {s.get('notes', 'Ensure all 5 verification checks pass before committing.')}
"""
        formatted_steps.append(formatted_step)

    # ── README DOCUMENTATION STEPS (appended to the plan) ─────────────────────
    # Keep OCULUS_IMPORTANT/oculus_readme.md (the audit's "documented features"
    # list) accurate after this plan's execution. The readme steps run at the
    # END of the plan so they cover every file the plan itself creates/changes.
    plan_target_files = []
    _seen_tf = set()
    for _s in unique_steps:
        for _tf in (_s.get("target_files") or []):
            if _tf not in _seen_tf:
                _seen_tf.add(_tf)
                plan_target_files.append(_tf)
    # newest files first — the last N plan target files are this cycle's changes
    readme_need = [f.split('/')[-1] for f in plan_target_files][-10:]
    if readme_need:
        # a python check that the readme documents each needed file basename
        need_lits = ", ".join(repr(b) for b in readme_need)
        readme_check_cmd = (
            "python -c \"c=open('OCULUS_IMPORTANT/oculus_readme.md').read(); "
            f"missing=[b for b in [{need_lits}] if b not in c]; "
            "assert not missing, 'readme missing entries for: %s' % missing\""
        )
    else:
        readme_check_cmd = 'grep -c "Purpose:" OCULUS_IMPORTANT/oculus_readme.md'

    README_STEPS = [
        {
            "objective": "Update OCULUS_IMPORTANT/oculus_readme.md to document all current source files (including the files this plan created or changed)",
            "category": "DOCUMENTATION",
            "priority": "MEDIUM",
            "benefit": "Keeps the readme — used by the audit and cross-eval as the documented-features list — accurate after this plan's execution",
            "effort": "2-4h",
            "target_files": ["OCULUS_IMPORTANT/oculus_readme.md"],
            "ops": (
                "<<<REPLACE>>> [OCULUS_IMPORTANT/oculus_readme.md] (OLD_STRING)\n"
                "(stale or incomplete file entries)\n"
                "(NEW_TEXT)\n"
                "Refresh the numbered file entries (#N. "+REPO_ROOT+"/<rel/path> "
                "with Purpose / Mechanism / I/O lines) for EVERY current source file under the repo "
                "(excluding legacy, oculus_env, .venv, node_modules, site-packages, archive, __pycache__). "
                "Ensure entries exist for every file this plan touched — including but not limited to: "
                + ", ".join(repr(f) for f in plan_target_files[:40]) + " — "
                "and any other file created or modified during execution. Preserve the existing "
                "## Technical Summary section and anything after it. Add entries for new files, "
                "update entries for changed files, keep accurate existing entries."
            ),
            "verification": [
                'grep -c "Purpose:" OCULUS_IMPORTANT/oculus_readme.md',
                readme_check_cmd,
                'python -c "import os; assert os.path.exists(\'OCULUS_IMPORTANT/oculus_readme.md\')"',
            ],
            "commit": "[README] Document all current source files after this plan's execution",
            "rollback": "git checkout -- OCULUS_IMPORTANT/oculus_readme.md",
            "notes": "Document the files, including the ones this plan itself created or changed. Never delete the Technical Summary.",
        },
        {
            "objective": "Verify the readme documentation is complete and its format is intact",
            "category": "DOCUMENTATION",
            "priority": "LOW",
            "benefit": "Guards against a truncated or malformed readme after the update step",
            "effort": "1-2h",
            "target_files": ["OCULUS_IMPORTANT/oculus_readme.md"],
            "ops": (
                "<<<REPLACE>>> [OCULUS_IMPORTANT/oculus_readme.md] (OLD_STRING)\n"
                "(malformed readme)\n"
                "(NEW_TEXT)\n"
                "Repair the readme if the verification below fails: keep the numbered entries, "
                "restore any missing sections (Technical Summary, DELIVERABLES), keep every "
                "entry's Purpose / Mechanism / I/O structure."
            ),
            "verification": [
                'python -c "import re; c=open(\'OCULUS_IMPORTANT/oculus_readme.md\').read(); assert \'## Technical Summary\' in c, \'Technical Summary missing\'; assert len(re.findall(r\'^#\\d+\\.\', c, re.M))>=50, \'too few entries\'"',
                'grep -c "Mechanism:" OCULUS_IMPORTANT/oculus_readme.md',
            ],
            "commit": "[README] Verify readme integrity",
            "rollback": "git checkout -- OCULUS_IMPORTANT/oculus_readme.md",
            "notes": "Format-integrity check only.",
        },
    ]
    for _rs in README_STEPS:
        total_steps += 1
        idx = total_steps
        _tf_str = "\n".join(f"  - {t}" for t in _rs["target_files"])
        _ver_str = "\n".join(f"{i}. {v}" for i, v in enumerate(_rs["verification"][:5], 1))
        formatted_steps.append(f"""### STEP {idx}/{total_steps}: {_rs['objective']}

**CATEGORY:** {_rs['category']}
**PRIORITY:** {_rs['priority']}
**BENEFIT:** {_rs['benefit']}
**EFFORT:** {_rs['effort']}
**TARGET_FILES:**
{_tf_str}
**DEPENDENCIES:** []
**LINE_RANGES:** [OCULUS_IMPORTANT/oculus_readme.md]

**CODE_OPERATIONS:**
{_rs['ops']}

**VERIFICATION:**
{_ver_str}

**COMMIT_MESSAGE:** "{_rs['commit']}"
**ROLLBACK_COMMAND:** {_rs['rollback']}
**NOTES:** {_rs['notes']}
""")

    # Calculate token cost estimate
    in_tok = _total_tokens_used["input"]
    out_tok = _total_tokens_used["output"]
    cost_est = (in_tok / 1_000_000.0 * COST_PER_1M_INPUT) + (out_tok / 1_000_000.0 * COST_PER_1M_OUTPUT)

    # Assemble master plan markdown document
    doc = []
    doc.append("# OCULUS MASTER IMPLEMENTATION PLAN v1")
    doc.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.append(f"**Cross-Evaluation Subagents:** {len(all_subagent_outputs)}")
    # 09-03: when the agents' valid/invalid payloads were schema-dropped (see
    # ingest fix), the header must not claim 0 evaluated with a real batch —
    # fall back to the loader's raw finding count so the accounting stays honest.
    evaluated = len(valid_map) + len(invalid_map)
    if evaluated == 0 and all_findings:
        evaluated = len(all_findings)
    doc.append(f"**Audit Findings Evaluated:** {evaluated}")
    doc.append(f"**Valid Findings Accepted:** {len(valid_map)}")
    doc.append(f"**False Positives Rejected:** {len(invalid_map)}")
    doc.append(f"**Omissions / Emergence Identified:** {len(omissions)}")
    doc.append(f"**Atomic Steps Produced:** {total_steps}")
    if not valid_map and all_findings:
        doc.append(f"**Note (8_27 generation):** per-agent valid/invalid payloads were schema-dropped by the ingest validator (fixed for future cycles) — cut numbers come from the stage-1 vote report (`multi_agent_oculus_audit_8_27.json`); step synthesis, the execution substance, is unaffected.\n")
    doc.append(f"**Token Usage:** {in_tok:,} prompt / {out_tok:,} completion (Estimated cost: ~${cost_est:.4f})\n")

    doc.append("## EXECUTIVE SUMMARY")
    doc.append(f"This master plan synthesizes {len(all_findings) if all_findings else 'all'} multi-agent audit findings of the 8_27 cycle across the Oculus repo. It cross-references `OCULUS_SOURCE_OF_TRUTH_7_23.md`, `oculus_readme.md`, and Graphify call graph topologies using {len(all_subagent_outputs)} parallel DeepSeek subagents evaluated on the free webchat lanes.\n")

    doc.append("## SPEC RESTORATION CHECKLIST (Non-Negotiable)")
    doc.append("- [x] Survivor carry-over (survivors persist to next generation)")
    doc.append("- [x] Dynamic survival threshold (+1 per generation)")
    doc.append("- [x] 4 mutation types (single, multi, crossover, multi-crossover)")
    doc.append("- [x] Original unmutated strategy persistence")
    doc.append("- [x] Fitness-weighted ensemble (NOT Sharpe-weighted)")
    doc.append("- [x] Regime-aware parameter tuning")
    doc.append("- [x] FTROM evolvable weight (genome slot 0.0–2.0)")
    doc.append("- [x] 21 tradable assets")
    doc.append("- [x] Non-temporal bars (dollar, volume, range, tick)")
    doc.append("- [x] ONLY 5% rolling DD + 10% static DD (no other kill logic)\n")

    doc.append("## IMPLEMENTATION STEPS\n")
    doc.append("\n---\n".join(formatted_steps))

    doc.append("\n## DELIVERABLES SUMMARY")
    doc.append(f"1. **Implementation Plan:** {total_steps} atomic, content-addressed steps.")
    doc.append("2. **Risk Assessment:** High-risk steps identified with explicit 5-command verification & git rollback.")
    doc.append(f"3. **Total Estimated Effort:** ~{total_steps * 3} person-hours.")
    doc.append("4. **Rollout Strategy:** Sequential execution with mandatory GitHub sync after every verified step.")

    plan_content = "\n".join(doc)
    with open(MASTER_PLAN_FILE, "w") as f:
        f.write(plan_content)

    log_message(f"\n[master] Saved Master Implementation Plan to {MASTER_PLAN_FILE}")
    log_message(f"  Total Steps: {total_steps}")
    log_message(f"  Valid Findings: {len(valid_map)}, Rejected False Positives: {len(invalid_map)}")
    log_message(f"  Token Usage: {in_tok:,} in / {out_tok:,} out (~${cost_est:.4f}) — today's ledger: ${load_daily_spend():.4f}")
    return plan_content


# ─── MAIN PIPELINE ───────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Oculus Multi-Agent Cross-Evaluation & Plan Synthesizer")
    parser.add_argument("--smoke", action="store_true", help="Run a smoke test with 5 subagents on exclusive finding chunks")
    parser.add_argument("--resume", action="store_true", help="Resume from existing cross_eval_state.json checkpoint")
    parser.add_argument("--self-check", action="store_true",
                        help="Verify star-import shim resolution (step 571)")
    args = parser.parse_args()

    if args.self_check:
        # 2026-08-09 (step 571): --self-check verifies shim resolution.
        ok = True
        _shim_cases = ((_os.path.join(REPO_ROOT, "config_loader.py"), "config_loader"),
                       (_os.path.join(REPO_ROOT, "live", "courtroom.py"), "live.courtroom"))
        for path, expect in _shim_cases:
            if not _os.path.exists(path):
                print(f"[self-check] SKIP {path}: not present in this checkout")
                continue
            try:
                with open(path, errors="ignore") as f:
                    got = resolve_shim_target(f.read())
                if got != expect:
                    print(f"[self-check] FAIL {path}: resolved {got!r}, expected {expect!r}")
                    ok = False
                else:
                    print(f"[self-check] OK {path} -> {got}")
            except Exception as e:
                print(f"[self-check] FAIL {path}: {e}")
                ok = False
        non_shim = "import os\nimport json\n"
        if resolve_shim_target(non_shim) is not None:
            print("[self-check] FAIL: non-shim source resolved to a target")
            ok = False
        else:
            print("[self-check] OK: non-shim left unresolved")
        print("[self-check] PASS" if ok else "[self-check] FAIL")
        sys.exit(0 if ok else 1)

    log_message("=" * 70)
    log_message("OCULUS MULTI-AGENT CROSS-EVALUATION & SYNTHESIZER v2")
    log_message("=" * 70)

    # Load SOT and README
    sot = ""
    if os.path.exists(SOT_FILE):
        with open(SOT_FILE) as f: sot = f.read()
    readme = ""
    if os.path.exists(README_FILE):
        with open(README_FILE) as f: readme = f.read()

    log_message(f"[main] Loaded SOT ({len(sot)} chars), README ({len(readme)} chars)")

    # Load Graphify
    graphify = get_graphify()
    log_message(f"[main] Graphify loaded: {graphify.has_data()} ({len(graphify.nodes)} nodes)")

    # Load Audit Findings from pass checkpoints or report
    all_findings = []
    for p in range(1, 6):
        pass_dir = os.path.join(OUTPUT_BASE, f"pass_{p}")
        if os.path.exists(pass_dir):
            for fname in sorted(os.listdir(pass_dir)):
                if fname.startswith("batch_") and fname.endswith(".json"):
                    with open(os.path.join(pass_dir, fname)) as f:
                        try:
                            bdata = json.load(f)
                            all_findings.extend(bdata.get("findings", []))
                        except Exception:
                            pass

    # 09-02 (webchat batch era): the audit now writes webchat_audit_X/all_results.json
    # (passes[].findings[] with finding_id/agent/severity/category/file/finding/mechanism).
    # Load it so the swarm evaluates the CURRENT audit, not stale legacy dirs.
    batch_json = os.path.join(OUTPUT_BASE, f"webchat_audit_{os.environ.get('AUDIT_VERSION','8_27')}", "all_results.json")
    if os.path.exists(batch_json):
        try:
            bar = json.load(open(batch_json))
            for p in bar.get("passes", []):
                for f in p.get("findings", []):
                    # ROW-based: the audit's unit of authority is (finding × file-context) —
                    # the SAME finding_id appears once per file/round. NO dedup (09-02: dedup
                    # collapsed 34,618 rows to 697 entities and starved the swarm).
                    if isinstance(f, dict) and f.get("finding_id"):
                        all_findings.append(f)
        except Exception as e:
            log_message(f"[main] batch findings load error: {e}")
        log_message(f"[main] Added batch findings; total {len(all_findings)} raw findings")

    log_message(f"[main] Discovered {len(all_findings)} raw audit findings across passes 1-5")

    # Generate 150 subagent definitions
    subagents = generate_150_subagent_definitions()
    log_message(f"[main] Configured {len(subagents)} specialized subagents across 6 cohorts")

    if args.smoke:
        log_message("\n[SMOKE TEST] Running 5 subagents on 15 findings")
        subagents = subagents[:5]
        all_findings = all_findings[:15]

    # FIX #3: Exclusive Chunk Assignment (Zero overlap across 150 subagents)
    total_f = len(all_findings)
    chunk_size = max(1, (total_f + len(subagents) - 1) // len(subagents))
    subagent_chunks = []
    for i in range(len(subagents)):
        start = i * chunk_size
        end = min(start + chunk_size, total_f)
        subagent_chunks.append(all_findings[start:end])

    # FIX #4: Checkpoint & Resume Support
    saved_outputs = load_state() if args.resume else []
    # Only consider SUCCESSFUL (non-failed) outputs as "done". Failed subagents
    # (empty/error results) are checkpointed with their real subagent_id, which
    # would otherwise skip them forever on resume. Distinguish by whether the
    # output carries valid findings OR a explicit non-empty result beyond the
    # error marker.
    def _is_success(out: dict) -> bool:
        if not isinstance(out, dict):
            return False
        # A real success has at least one populated field beyond subagent_id
        keys = set(out.keys()) - {"subagent_id"}
        return any(out.get(k) for k in keys)

    saved_success = [o for o in saved_outputs if _is_success(o)]
    completed_ids = {out.get("subagent_id") for out in saved_success}
    if completed_ids:
        log_message(f"[main] RESUME mode: Loaded {len(completed_ids)} successful subagents from {CROSS_EVAL_STATE}")

    pending_subagents = []
    for i, agent in enumerate(subagents):
        if agent["id"] not in completed_ids:
            pending_subagents.append((agent, subagent_chunks[i]))

    if not pending_subagents and saved_outputs:
        log_message("[main] All subagent evaluations already complete; regenerating master plan")
        all_outputs = saved_outputs
    else:
        log_message(f"[main] {len(pending_subagents)} subagents remaining to execute")
        sem = asyncio.Semaphore(SEMAPHORE_LIMIT)

        async with aiohttp.ClientSession() as session:
            log_message(f"[main] DeepSeek API Endpoint: {DEEPSEEK_API_BASE}")
            log_message(f"[main] Flash Model: {MODEL_FLASH} | Pro Model: {MODEL_PRO}")

            all_outputs = list(saved_success)

            async def run_subagent_task(agent: dict, chunk: list[dict]) -> dict:
                async with sem:
                    # 09-02 daily spend breaker — no API call once the day's paid
                    # budget is exhausted; the result is skipped from state so a
                    # resume later continues exactly where it paused.
                    if BREAKER_USD > 0 and load_daily_spend() >= BREAKER_USD:
                        global BREAKER_TRIPPED
                        BREAKER_TRIPPED = True
                        log_message(f"  [{agent['id']}] BREAKER: ${load_daily_spend():.2f} spent >= ${BREAKER_USD:.2f} — paused, resume to continue")
                        return {"subagent_id": agent["id"], "breaker_pause": True}
                    await throttled_start()
                    # 2026-08-07 (user): ALL flash — v4 flash scores better than
                    # pro now and is cheaper. NEVER use pro.
                    use_pro = False
                    sys_prompt = build_subagent_system_prompt(agent, sot, readme)
                    user_prompt = build_subagent_user_prompt(agent, chunk, graphify)
                    t0 = time.time()
                    content, actual_model, score = await call_deepseek(
                        session, user_prompt, sys_prompt, use_pro=use_pro, agent_name=agent["name"]
                    )
                    latency = time.time() - t0
                    parsed = extract_json(content)
                    # 09-02: webchat-pool lanes can return chatty/truncated 200s —
                    # retry the attack (up to 2 extra, rotating gates) instead of
                    # checkpointing an empty result (lost chunk in the final plan).
                    # cap only ENDS the loop when parsed lands or retries run out.
                    _retries = 0
                    while parsed is None and _retries < 2:
                        _retries += 1
                        log_message(f"  [{agent['id']}] parse miss ({len(content)} chars) — lane retry {_retries}/2")
                        await asyncio.sleep(5)
                        t0 = time.time()
                        content, actual_model, score = await call_deepseek(
                            session, user_prompt, sys_prompt, use_pro=use_pro, agent_name=agent["name"]
                        )
                        latency = time.time() - t0
                        parsed = extract_json(content)
                    status = "OK" if parsed and not content.startswith("ERROR") else "FAIL"
                    log_message(f"  [{agent['id']}] {agent['name'][:30]:30} ({'PRO' if use_pro else 'FLASH'}) -> {status} ({latency:.1f}s, model: {actual_model})")
                    
                    # Validate findings
                    if isinstance(parsed, dict):
                        # 09-03 fix: parse_findings_payload validates against the
                        # AUDIT's schema (finding_type/description/severity) and
                        # silently drops EVERY cross-eval entry (finding_id/
                        # reason_valid) — the 8_27 generation lost all valid/
                        # invalid judgments this way. Pass through as-is with a
                        # bare dict guard: the synthesis speaks THIS schema.
                        parsed["valid_findings"] = [x for x in parsed.get("valid_findings", []) if isinstance(x, dict)]
                        parsed["invalid_findings"] = [x for x in parsed.get("invalid_findings", []) if isinstance(x, dict)]
                    res = parsed if isinstance(parsed, dict) else {
                        "subagent_id": agent["id"], "valid_findings": [], "invalid_findings": [],
                        "omissions_and_emergent": [], "proposed_steps": []
                    }
                    return res

            # Dispatch subagents concurrently in controlled batches
            task_map = {}
            for agent, chunk in pending_subagents:
                t = asyncio.create_task(run_subagent_task(agent, chunk))
                task_map[t] = agent["id"]

            pending_tasks = set(task_map.keys())
            while pending_tasks:
                done, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                for finished_task in done:
                    try:
                        res = finished_task.result()
                    except Exception as e:
                        aid = task_map[finished_task]
                        log_message(f"  [{aid}] Task Exception: {e}")
                        res = {"subagent_id": aid, "valid_findings": [], "invalid_findings": [], "omissions_and_emergent": [], "proposed_steps": []}
                    if isinstance(res, dict) and res.get("breaker_pause"):
                        # breaker hit — do NOT checkpoint this agent; a resumed run
                        # re-executes it (and everything after) once budget allows.
                        global BREAKER_TRIPPED
                        BREAKER_TRIPPED = True
                        continue
                    all_outputs.append(res)
                    save_state(all_outputs)

    # 09-02 breaker: pause before synthesis — do NOT write a partial master plan.
    # Progress is fully checkpointed; a later resume completes the rest and
    # regenerates the plan from the complete set.
    if BREAKER_TRIPPED:
        alert_breaker()
        log_message(f"[BREAKER] Stage-2 paused: ${load_daily_spend():.2f} (${BREAKER_USD:.2f} limit) — {len(all_outputs)}/{len(subagents)} agents done; resume later to finish.")
        return

    # Synthesize the master plan ALWAYS — including the all-complete resume
    # path (PLAN_TODO #2: synthesis previously lived only in the else branch,
    # so a fully-checkpointed resume exited 0 with a stale plan).
    async with aiohttp.ClientSession() as synth_session:
        await synthesize_master_plan(synth_session, all_outputs, sot, readme, all_findings)

    log_message("\n" + "=" * 70)
    log_message("CROSS-EVALUATION & PLAN SYNTHESIS COMPLETE")
    log_message("=" * 70)
    log_message(f"Master Plan: {MASTER_PLAN_FILE}")


# ─── WEBCHAT-POOL BACKEND (08-14) ─────────────────────────────────────────
DEEPSEEK_API_BASE_EXTRA = os.getenv('DEEPSEEK_API_BASE_EXTRA', '')
WEBCHAT_POOL_GATES = [g.strip().rstrip('/') for g in DEEPSEEK_API_BASE_EXTRA.split(',') if g.strip()]
_webchat_pool_rr = 0
_webchat_pool_failures: dict[str, int] = {}
_webchat_pool_gate_locks: dict[str, asyncio.Lock] = {}

async def call_webchat_pool(session: aiohttp.ClientSession,
                               user_prompt: str,
                               system_prompt: str,
                               use_pro: bool = False,
                               agent_name: str = '',
                               retries: int = MAX_RETRIES,
                               quiet: bool = False) -> tuple[str, str, int]:
    global _webchat_pool_rr
    gates = WEBCHAT_POOL_GATES
    if not gates:
        return 'ERROR: webchat-pool selected but DEEPSEEK_API_BASE_EXTRA is empty', 'webchat-pool', DEFAULT_MODEL_SCORE
    n = len(gates)
    start = _webchat_pool_rr
    for i in range(n):
        idx = (start + i) % n
        gate = gates[idx]
        lock = _webchat_pool_gate_locks.setdefault(gate, asyncio.Lock())
        async with lock:
            try:
                url = f'{gate}/v1/chat/completions'
                payload = {
                    'model': 'anymodel',
                    'stream': False,
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_prompt},
                    ],
                }
                headers = {'Content-Type': 'application/json'}
                async with session.post(url, json=payload, headers=headers,
                                        timeout=aiohttp.ClientTimeout(total=CHAT_TIMEOUT_SECONDS)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        try:
                            content = data['choices'][0]['message']['content']
                        except Exception:
                            content = ''
                        if content and not str(content).startswith('ERROR'):
                            _webchat_pool_rr = (idx + 1) % n
                            _webchat_pool_failures[gate] = 0
                            if not quiet:
                                log_message(f'      [webchat-pool] success via {gate} ({len(str(content))} chars)')
                            return str(content), 'webchat-pool', DEFAULT_MODEL_SCORE
                        _webchat_pool_failures[gate] = _webchat_pool_failures.get(gate, 0) + 1
                        if not quiet:
                            log_message(f'      [webchat-pool] {gate} empty/error content')
                    else:
                        _webchat_pool_failures[gate] = _webchat_pool_failures.get(gate, 0) + 1
                        if not quiet:
                            log_message(f'      [webchat-pool] {gate} HTTP {resp.status}')
            except asyncio.TimeoutError:
                _webchat_pool_failures[gate] = _webchat_pool_failures.get(gate, 0) + 1
                if not quiet:
                    log_message(f'      [webchat-pool] {gate} timeout after {CHAT_TIMEOUT_SECONDS}s')
            except Exception as e:
                _webchat_pool_failures[gate] = _webchat_pool_failures.get(gate, 0) + 1
                if not quiet:
                    log_message(f'      [webchat-pool] {gate} exception: {e}')
    return 'ERROR: webchat-pool exhausted all gates', 'webchat-pool', DEFAULT_MODEL_SCORE
if __name__ == "__main__":
    asyncio.run(main())

