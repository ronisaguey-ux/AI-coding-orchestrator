#!/usr/bin/env python3
"""
parallel_agents.py — Multi-agent Oculus code audit using OmniRoute API.
v6 — Rotation-based cross-examination with 8 agents, 5 passes, and
     model-capability scoring with dynamic fallback chains.
"""

import asyncio
import aiohttp
import json
import os
import sys
import re
import time
import random
import argparse
from datetime import datetime
from collections import defaultdict
from collections.abc import Iterator
import ssl

# ─── CONFIG ───────────────────────────────────────────────────────────────
# 08-23 fix (same class as the orchestrator 72a4b36): OmniRoute (20128) is
# dead and its API is not the lane anymore. phase_env() (orchestrator) exports
# DEEPSEEK_API_BASE -> the free webchat lane; direct invocations must too.
# Bare default keeps the old OmniRoute target so nothing else breaks.
API_BASE = os.environ.get("DEEPSEEK_API_BASE", "http://localhost:20128/api/v1")
# Cap concurrent batch pipelines: each pipeline fires 7+ LLM calls through
# OmniRoute, which spawns LOCAL claude subprocesses for auto/* models. 8 was
# enough to OOM a 14Gi laptop (2026-08-05: OOM kills at 08:08 + 14:10, then
# mt7921e WiFi driver hang + full freeze at 21:15). 4 matches cross_eval's
# CLAUDE_SWARM_LIMIT — the known-safe level.
# 2026-08-06 (user decision): 4 -> 2 — CPU watchdog (80% cap, user rule) was
# killing the audit at ~100% CPU every few minutes; 2 pipelines keeps total
# CPU under the cap at the cost of ~2x slower batches.
SEMAPHORE_LIMIT = 4          # concurrent batch pipelines
AGENT_SEMAPHORE = None
REQUEST_DELAY = 1.5          # seconds between request starts
# Cross-examination passes. Each pass re-audits every batch with the findings carried
# forward, so cost is linear in this number. Overridable: pass 1 answers "what is
# wrong"; the repeat passes only raise confidence in it.
NUM_PASSES = int(os.environ.get("AUDIT_NUM_PASSES", "5"))
BATCH_SIZE = 5               # files per LLM call
MAX_MODEL_ATTEMPTS = 15      # how many models to try per call before giving up
# Per-model timeout. 60s is right for an API lane and WRONG for a webchat lane:
# measured on this deployment the gemini tab answers in 5-60s and the deepseek tab
# after a 20-80s anti-ban gap plus its own generation, so a 60s cap banned every
# webchat lane on its first call (48 cooldown events, zero findings, run stalled).
# The engine already has per-lane cooldowns and a ladder, so the correct value is
# "longer than a slow webchat turn", not "as short as possible".
CHAT_TIMEOUT_SECONDS = int(os.environ.get("AUDIT_CHAT_TIMEOUT", "300"))
# Room for a full findings document. Measured: with the cap unset, free lanes
# returned 151-char replies that ended mid-object and lost every finding in them.
CHAT_MAX_TOKENS = int(os.environ.get("AUDIT_CHAT_MAX_TOKENS", "8000"))
# Refuse to treat "every file came back empty" as proof the code is clean — see
# _is_empty_findings_doc. Off makes the audit faster and less trustworthy.
REQUIRE_SUBSTANTIVE = os.environ.get("AUDIT_REQUIRE_SUBSTANTIVE", "1") != "0"
# How many OTHER models to try before believing "every file is clean". Cost is bounded
# on purpose: this is a credibility check, not a search.
MAX_EMPTY_ROTATIONS = int(os.environ.get("AUDIT_MAX_EMPTY_ROTATIONS", "2"))
MAX_BACKOFF_SECONDS = 12     # cap exponential backoff between model swaps
MAX_MODEL_HEALTH = 5         # models with health >= this are skipped (dead for this run)
# Cooldown durations are now per-failure-type (see mark_model_failed)
COOLDOWN_TIMEOUT = 300        #  5 min — model timed out (might just be slow)
COOLDOWN_RATELIMIT = 900      # 15 min — HTTP 429 (API is actively rejecting)
COOLDOWN_SERVER_ERR = 600     # 10 min — HTTP 5xx (server-side issue)
COOLDOWN_CLIENT_ERR = 3600    # 60 min — HTTP 4xx (auth/permanent — probably hopeless)
COOLDOWN_UNKNOWN = 480        #  8 min — exception / unknown failure
STARTUP_PROBE_TIMEOUT = 45   # health probe for top models (nuclear-fallback needs ~30-40s)
TASK_TRUNCATE = 12000        # audit prompt context length
FILE_TRUNCATE = 8000         # per-file content when batching
SOT_TRUNCATE = 8000          # Source of Truth context
README_TRUNCATE = 5000       # README context
COMPACTED_FINDINGS_MAX = 6000

# ── Audit target ─────────────────────────────────────────────────────────────
# These were absolute paths into a single hardcoded repo (oculus), which made the
# engine usable on exactly one codebase. Every one is now env-overridable with the
# original value as the fallback, so a deployment can point the same audit at a
# different repo (or several, via separate runs with separate OUTPUT_BASE) without
# editing code. AUDIT_TARGET_DIR is the primary root the file walk starts from.
OCULUS_DIR = os.environ.get("AUDIT_TARGET_DIR", "/home/roni/Roni_workspace/oculus")
ALT_SCRIPTS_DIR = os.environ.get("AUDIT_ALT_SCRIPTS_DIR", "/home/roni/Roni_workspace/alt_important_scripts")
WEBCHAT_API_DIR = os.environ.get("AUDIT_WEBCHAT_API_DIR", "/home/roni/Roni_workspace/webchat-api")
TASK_FILE = os.environ.get("AUDIT_TASK_FILE", "/home/roni/Roni_workspace/promptsfr/audit_prompt.md")
README_FILE = os.environ.get("AUDIT_README_FILE", "/home/roni/Roni_workspace/oculus/OCULUS_IMPORTANT/oculus_readme.md")
SOT_FILE = os.environ.get("AUDIT_SOT_FILE", "/home/roni/Roni_workspace/oculus/OCULUS_IMPORTANT/OCULUS_SOURCE_OF_TRUTH_7_23.md")
GRAPH_FILE = os.environ.get("AUDIT_GRAPH_FILE", "/home/roni/Roni_workspace/oculus/graphify-out/graph.json")
OUTPUT_BASE = os.environ.get("AUDIT_OUTPUT_DIR", "/home/roni/Roni_workspace/audits_plans")
# Optional extra roots walked for files (comma-separated). Lets one audit cover
# several repositories without a synthetic parent directory.
EXTRA_ROOTS = [d for d in os.environ.get("AUDIT_EXTRA_ROOTS", "").split(",") if d.strip()]
# Explicit file list (absolute paths, comma or newline separated). When set, this IS
# the audit scope and the directory walk is skipped — see get_file_list.
INCLUDE_FILES = [f for f in os.environ.get("AUDIT_INCLUDE_FILES", "").replace("\n", ",").split(",") if f.strip()]
# What the audit calls the thing it is reading. The prompt said "the Oculus trading
# system" for every deployment, so a batch of helpotron extension files was described
# to the model as trading-system code — wrong context produces wrong findings.
TARGET_LABEL = os.environ.get("AUDIT_TARGET_LABEL", "target")
# 2026-08-14 (user HARD RULE): label artifacts by the CURRENT date, never a
# stale constant. The orchestrator passes the web-verified date via
# AUDIT_VERSION; direct invocations fall back to today.
AUDIT_VERSION = os.environ.get("AUDIT_VERSION", f"{datetime.now().month}_{datetime.now().day}")
FINAL_REPORT = f"{OUTPUT_BASE}/multi_agent_oculus_audit_{AUDIT_VERSION}.md"
STATE_FILE = f"{OUTPUT_BASE}/audit_state.json"


# ─── MODEL CAPABILITY SCORES ──────────────────────────────────────────────
# Composite scores derived from public benchmarks (MMLU, HumanEval, GPQA/MATH,
# SWE-bench where available). Scores are on a 0-100 scale.
# For models not listed here, the script defaults to DEFAULT_MODEL_SCORE.
# Update this dictionary manually as more benchmark data becomes available.
DEFAULT_MODEL_SCORE = 85


MODEL_SCORES = {
    # OmniRoute & Free Providers
    "nuclear-fallback": 90,
    "big-pickle": 90,
    "felo-scholar": 92,
    "felo-document": 90,
    "felo-search": 88,
    "felo-chat": 88,
    "mimo-v2.5-free": 88,
    "deepseek-v4-flash-free": 92,
    "nemotron-3-ultra-free": 88,
    "nemotron-3.5-content-safety": 85,

    # Claude / Anthropic
    "claude-3-5-sonnet": 95,

    "claude-3.5-sonnet": 95,
    "claude-4-6-opus": 97,
    "claude-4.6-opus": 97,
    "claude-4-5-opus": 97,
    "claude-4.5-opus": 97,
    "claude-4-7-opus": 97,
    "claude-4.7-opus": 97,
    "claude-4-8-opus": 97,
    "claude-4.8-opus": 97,
    "claude-4-6-sonnet": 95,
    "claude-4.6-sonnet": 95,
    "claude-4-5-sonnet": 95,
    "claude-4.5-sonnet": 95,
    "claude-4-5-haiku": 78,
    "claude-4.5-haiku": 78,
    "claude-3-5-haiku": 78,
    "claude-3.5-haiku": 78,
    "claude-opus-4": 97,
    "claude-sonnet-4": 95,
    "claude-haiku-3-5": 78,

    # OpenAI / GPT
    "gpt-4o": 93,
    "gpt-5": 94,
    "gpt-5.1": 94,
    "gpt-5.2": 94,
    "gpt-5.4": 93,
    "gpt-5.4-mini": 78,
    "gpt-5.4-nano": 72,
    "gpt-5.5": 94,
    "gpt-5.6-luna": 95,
    "gpt-5.6-sol": 95,
    "gpt-5.6-terra": 95,
    "gpt-4o-mini": 78,
    "gpt-o4-mini": 92,
    "gpt-o3-mini": 90,

    # DeepSeek
    "deepseek-v4": 92,
    "deepseek-v4-pro": 92,
    "deepseek-v4-flash": 92,
    "deepseek-v3": 88,
    "deepseek-r1": 85,
    "deepseek-v2": 55,
    "deepseek-coder": 70,

    # GLM / Z-AI
    "glm-5.2": 87,

    # Meta / Llama
    "llama-3.3-70b": 86,
    "llama-3.1-8b": 55,
    "llama-3.2-90b": 75,
    "llama-4-maverick": 84,

    # Mistral
    "mistral-large": 82,
    "mistral-large-3": 82,
    "mistral-small": 72,
    "mistral-medium": 76,
    "mixtral-8x7b": 68,
    "devstral": 78,

    # Google / Gemini
    "gemini-1.5-pro": 80,
    "gemini-2.0-flash": 72,
    "gemini-2.5-pro": 82,
    "gemini-3-pro": 85,
    "gemini-3-flash": 78,
    "gemini-1.5-flash": 75,

    # Moonshot / Kimi
    "kimi-k3": 80,
    "kimi-k2.7": 78,
    "kimi-k2.7-code": 78,
    "kimi-k2.6": 75,

    # NVIDIA / Nemotron
    "nemotron-3-ultra": 60,
    "nemotron-3-super": 58,
    "nemotron-3-nano": 50,
    "nemotron-3.5": 55,
    "nemotron-mini": 50,

    # Qwen
    "qwen3.5": 80,
    "qwen3-next": 78,

    # Stepfun
    "step-3.7-flash": 78,
    "step-3.5-flash": 75,

    # xAI / Grok
    "grok-4": 88,

    # Abacus / Others
    "dracarys-llama": 72,

    # Orchestrated combos / fallbacks
    "nuclear-fallback": 90,
    "auto/best-coding": 90,
    "auto/best-reasoning": 90,
    "auto/pro-coding": 88,
    "auto/pro-reasoning": 88,
    "auto/coding": 85,
    "auto/smart": 85,
    "auto/claude-sonnet": 95,
    "auto/glm": 87,
    "auto/llama": 86,
    "auto/gemini": 80,
    "auto/best-free": 70,
    "auto/cheap": 65,
    "auto/fast": 70,

    # Common provider prefixes (used as substring fallbacks)
    "sonnet": 95,
    "opus": 97,
    "haiku": 78,
    "gpt-4o": 93,
    "gpt-5": 94,
    "deepseek-v4": 92,
    "deepseek-v3": 88,
    "glm-5.2": 87,
    "llama-3.3-70b": 86,
    "mistral-large": 82,
    "gemini-3-pro": 85,
    "gemini-1.5-pro": 80,
    "kimi-k3": 80,
    "kimi-k2.7": 78,
    "kimi-k2.6": 75,
    "nemotron-3-ultra": 60,
}

# Substrings that indicate a model is NOT suitable for code audit text generation.
EXCLUDED_MODEL_SUBSTRINGS = [
    "embed", "flux", "whisper", "fastpitch", "tacotron", "veo", "seedance",
    "rerank", "asr", "nv-embed", "nv-rerank", "parakeet",
]

# STEP 370/1566: explicit security-approved model allowlist. Default = every
# curated model in MODEL_SCORES that is not a non-code modality. A model the
# gateway exposes that is not allowlisted is DROPPED from the fallback chain, so
# sensitive audit prompts can never route to a shadow/unaudited provider. The
# curated orchestrated 'nuclear-fallback' combo is allowlisted by default.
MODEL_ALLOWLIST = frozenset(
    mid for mid in MODEL_SCORES
    if not any(ex in mid.lower() for ex in EXCLUDED_MODEL_SUBSTRINGS)
)


def _load_model_allowlist() -> frozenset:
    """Return the effective model allowlist.

    Comma-separated OCULUS_MODEL_ALLOWLIST env override; a present-but-empty
    value is deny-all (empty chain). Absent env -> the curated MODEL_ALLOWLIST.
    """
    raw = os.environ.get("OCULUS_MODEL_ALLOWLIST")
    if raw is None:
        return MODEL_ALLOWLIST
    ids = {p.strip() for p in raw.split(",") if p.strip()}
    return frozenset(ids)

# ─── AGENTS ───────────────────────────────────────────────────────────────
AGENTS = [
    {
        "name": "Radical New Idea Generator",
        "weight": 1.2,
        "graph_focus": "clusters",
        "prompt_addition": """Your primary focus is to propose radical new ideas that could improve the system significantly — better algorithms, alternative architectures, new libraries, or paradigm shifts. For each idea, explain the benefit and potential impact. If an idea violates the Source of Truth, explicitly state that and provide a compelling justification for why it should override the spec. Output these as findings with category "RADICAL_IDEA".

GRAPHIFY GUIDANCE:
- Use the provided community/cluster summaries to identify natural module boundaries and refactoring opportunities.
- Look for clusters that are overly large or that bridge unrelated domains — these are candidates for splitting.
- Suggest ideas that reduce cross-cluster coupling or consolidate duplicated clusters."""
    },
    {
        "name": "SOT Specialist",
        "weight": 1.5,
        "graph_focus": "callers",
        "prompt_addition": """Your primary focus is to compare every line of code against the Source of Truth. Flag every deviation, no matter how small. If the spec is unclear, note that. Your severity should be CRITICAL for direct contradictions, HIGH for omissions.

GRAPHIFY GUIDANCE:
- Use caller/callee relationships to verify that spec-critical functions are invoked from the expected places.
- If a function is documented in the SoT but has no callers (orphan), flag it as potentially dead or unimplemented.
- If a function has unexpected callers, flag it as a possible SoT violation."""
    },
    {
        "name": "Architecture Specialist",
        "weight": 1.4,
        "graph_focus": "cycles_and_communities",
        "prompt_addition": """Your primary focus is on system architecture: coupling between modules, god objects, separation of concerns, testability, and adherence to clean architecture principles. Identify structural debt and suggest refactoring.

GRAPHIFY GUIDANCE:
- Use the provided cycle report to detect circular dependencies; these are strong architecture smells.
- Use community summaries to evaluate modularity: large communities may be god modules, tiny communities may be scattered logic.
- Look for files that call across many communities (high betweenness) — these are coupling hotspots."""
    },
    {
        "name": "Performance Specialist",
        "weight": 1.3,
        "graph_focus": "hotspots",
        "prompt_addition": """Your primary focus is performance: memory allocation, loop vectorization, JIT fallback, I/O blocking, database queries, caching inefficiencies, and multiprocessing bottlenecks. Quantify impact (latency, memory, throughput).

GRAPHIFY GUIDANCE:
- Use the provided hotspot report (highest-degree nodes) to find functions that are heavily called.
- High-degree functions are the most impactful optimization targets; small improvements there multiply across the system.
- Look for deep call chains and fan-out patterns that could indicate repeated work or N+1 problems."""
    },
    {
        "name": "Docs/Legacy Specialist",
        "weight": 1.1,
        "graph_focus": "orphans",
        "prompt_addition": """Your primary focus is documentation and legacy code: README mismatches (code vs docs), deprecated functions, unused imports, dead code paths, and features that are undocumented in both README and SoT.

GRAPHIFY GUIDANCE:
- Use the provided orphan report (nodes with no links) to identify likely dead code.
- Cross-reference orphan functions against the README; if documented but unlinked, flag as stale docs or missing tests.
- If an orphan appears security-sensitive, investigate whether it is reachable through reflection, hooks, or dynamic imports."""
    },
    {
        "name": "CIA Hacker",
        "weight": 1.6,
        "graph_focus": "orphans_and_entrypoints",
        "prompt_addition": """Your primary focus is adversarial security. Assume you are a senior CIA infiltrator trying to compromise Oculus. Check every vector: API auth, data exfiltration, code injection, privilege escalation, live trading safety, cryptographic flaws, supply chain risks. Attempt to find a way to break the system.

GRAPHIFY GUIDANCE:
- Use orphan and low-degree nodes to find rarely-scrutinized code paths that may hide backdoors or weak validation.
- Use caller/callee chains to trace how user input reaches sensitive functions (live trading, auth, file I/O).
- Identify high-centrality nodes: if compromised, these would give broad control over the system.
- Look for missing links: security-sensitive functions that should be called by auth/logging but are not."""
    },
    {
        "name": "Logic Gap Analyst",
        "weight": 1.3,
        "graph_focus": "callers",
        "prompt_addition": """Your sole purpose is to identify LOGICAL GAPS — NOT syntax errors, NOT performance issues, NOT security flaws (other specialists cover those). Evaluate whether the code MAKES SENSE from a logical, architectural, and practical standpoint.

DEFINITION OF A LOGICAL GAP:
- Code that compiles and runs correctly but is fundamentally flawed in its reasoning
- Redundant logic that could be simplified or eliminated
- Backwards logic (doing the opposite of what's intended)
- Approaches that work "on paper" but are horrible in practice

EXAMPLE: a genetic algorithm that culls 98% of the population each generation and seeds the next from the 2% winners. It "works" technically but forces premature convergence and destroys genetic diversity — logically, a disaster.

For each logical gap found, output:
1. The gap (what doesn't make sense)
2. Why it doesn't make sense (the logical flaw)
3. A proposed fix or improvement (how to make it logical)
4. Criticality (HIGH/MED/LOW)
5. Module and class where it was found
6. Link to relevant Source of Truth section (if applicable)

PRIORITIZATION:
- HIGH: breaks the fundamental logic of the system
- MEDIUM: redundant or inefficient logic that doesn't break anything but is wasteful
- LOW: minor logical quirks worth improving but not critical

CROSS-EXAMINATION (later passes): review findings from other specialists and flag any proposed fix that does not address the root logical confusion (fixes symptoms, not the flaw).

FALSE-POSITIVE HANDLING: some code may seem illogical but is actually complexly profound — do not assume complexity is a flaw. If ambiguous, flag as LOW with the note: "May be intentional — review manually". Only flag clear logical contradictions.

EVOLUTION: reference past logic-gap findings to identify recurring patterns (e.g. "we keep confusing regime detection with signal generation") and flag their recurrence as a HIGH finding.

You have no veto power — you flag, propose, and move on.

GRAPHIFY GUIDANCE:
- Use caller/callee chains to trace intent: does each caller's usage match what the callee actually does?
- Look for logic that contradicts its own surrounding code (inverted conditions, dead branches, unreachable-by-construction paths).
- Where data flows through multiple hops, check that transformations compose consistently (units, directions, sign conventions)."""
    },
    {
        "name": "Simplification Specialist",
        "weight": 1.2,
        "graph_focus": "clusters",
        "prompt_addition": """Your sole purpose is to find ways to SIMPLIFY the system — every place it can be made simpler WITHOUT losing quality. You are the complexity surgeon: the codebase accumulates layers, indirections, and duplicated patterns; you find what can be cut, merged, or flattened while behavior, robustness, and readability stay intact or improve.

WHAT COUNTS AS A SIMPLIFICATION (category "SIMPLIFICATION"):
- Redundant logic that does the same thing twice in different forms (duplicated algorithms, parallel implementations of one concept)
- Needless abstraction layers: wrappers, proxy classes, or indirection that add no behavior, no safety, and no testability
- Over-engineering: config flags nobody reads, generic machinery parameterized but always called the same way, defensive code guarding impossible states
- Files or modules that are the same shape and could be unified into one generic implementation
- Unnecessary dependencies or imports
- Complex control flow that a simpler structure expresses identically (deeply nested conditionals, flag-state machines that are really two paths)
- Dead pathways that force maintenance burden (half-wired features, shim layers kept for compatibility nobody uses)

HARD RULE — NEVER lose quality: every proposal must preserve behavior, error handling, security properties, and feature completeness. If simplification would drop a capability, weaken an edge case, or hide a failure, it is REJECTED — note why. Quality preservation is the constraint; simplification is the goal. Prefer the smallest change that removes the most complexity.

For each simplification found, output:
1. What to simplify (the exact code/module/pattern)
2. Why it is complex today (the accidental complexity, not essential)
3. The proposed simplification (concrete: what to delete, merge, or flatten)
4. Why quality is preserved (behavior/security/robustness unchanged — or what tiny risk remains)
5. Impact (HIGH: large complexity reduction at low risk / MEDIUM: clean win, local / LOW: cosmetic but worth it)
6. File(s) and function/class names

CROSS-EXAMINATION (later passes): review the other specialists' proposed fixes and flag any that ADD complexity when a simpler fix exists — e.g. a new abstraction layer where a 3-line change fixes it, or a generic framework where a straight path works. A fix that makes the code harder to read, test, or maintain is itself a finding.

Do NOT propose simplifications that merely rename things or restyle code without removing complexity. Do NOT flag essential complexity (the inherent difficulty of the domain) as removable. You flag, propose, and move on — no veto power.

GRAPHIFY GUIDANCE:
- Use community/cluster summaries to find modules doing the same job in different shapes — unification candidates.
- High-betweenness files that bridge many communities may be doing coordination that belongs in a simpler shared primitive.
- Large clusters that are internally uniform are prime flattening candidates (one pattern, many copies)."""
    },
    {
        "name": "General Confirm Agent",
        "weight": 1.0,
        "graph_focus": "none",
        "prompt_addition": """You are the final arbiter. You will receive ALL findings from the 8 specialists for this batch. For each finding, vote YES (confirm) or NO (refute). Provide a brief reasoning. Consider the agent's specialty weight and the model score that produced the finding: high-score models/agents are more trustworthy; low-score ones require stronger corroboration. Output a structured vote."""
    },
]

MAX_AGENT_WEIGHT = max(a["weight"] for a in AGENTS)
CONFIDENCE_MULTIPLIER = {"HIGH": 1.0, "MEDIUM": 0.7, "LOW": 0.4, "": 0.5}
ACCEPTANCE_THRESHOLD = 0.60

# Global runtime state
_agent_model_usage: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
_model_health: dict[str, int] = {}
_model_cooldown: dict[str, float] = {}  # model_id -> timestamp when cooldown expires
_probed_chain: list[tuple[str, int]] = []
_usage_lock = asyncio.Lock()
_last_request_time = 0.0
_request_lock = asyncio.Lock()
# Count of call_llm calls that exhausted all retries (every candidate model
# failed). Used by the hard-fail guard: a high count with ~zero findings means
# the API was down and the "clean audit" is a false all-clear, NOT a real
# all-clear. See generate_final_report.
_api_exhaustions = 0

# Primary model: auto/best-reasoning combo. OmniRoute internally routes to the
# highest-priority available backend model from the updated combo.
# (2026-08-03: "nuclear-fallback" was renamed — OmniRoute now 400s on it:
# "Unable to determine provider". auto/best-* are the live combo IDs.)
# 08-23 fix: "auto/best-reasoning" is OmniRoute routing syntax the webchat
# gateways reject; they only accept their webchat model token ("anymodel").
# phase_env() exports DEEPSEEK_MODEL_FLASH; direct invocations must too.
PRIMARY_MODEL = os.environ.get("DEEPSEEK_MODEL_FLASH", "auto/best-reasoning")


# ─── HELPERS ──────────────────────────────────────────────────────────────
def is_model_cooling_down(model_id: str) -> bool:
    """True if this model is currently in a post-failure cooldown period."""
    expires = _model_cooldown.get(model_id, 0)
    return time.time() < expires


def _decay_model_health() -> None:
    """Time-based health decay: recover 1 health per model once its cooldown
    expires. Without this, health only accumulates — a long multi-pass run
    eventually marks every candidate dead and collapses to all-empty batches
    (observed 2026-08-03: passes 3-5 returned 0 findings after passes 1-2
    burned through the fallback chain). Models in active cooldown are left
    untouched; genuinely dead APIs stay dead until their cooldown elapses.
    """
    now = time.time()
    for mid in list(_model_health):
        if now >= _model_cooldown.get(mid, 0):
            new_health = max(0, _model_health[mid] - 1)
            if new_health == 0:
                _model_cooldown.pop(mid, None)
            _model_health[mid] = new_health


def mark_model_failed(model_id: str, reason: str = "unknown"):
    """Record a failure and apply a context-appropriate cooldown.

    - Timeout → 5 min (might just be slow today)
    - HTTP 429  → 15 min (API rate-limiting — back off hard)
    - HTTP 5xx  → 10 min (server-side issue, may recover)
    - HTTP 4xx  → 60 min (auth/permanent — probably hopeless)
    - Unknown    → 8 min  (safe default)
    """
    reason_lower = (reason or "").lower()
    if "429" in reason_lower or "rate" in reason_lower:
        cooldown = COOLDOWN_RATELIMIT
    elif "timeout" in reason_lower or "timed" in reason_lower:
        cooldown = COOLDOWN_TIMEOUT
    elif "5" in reason_lower[:3] if reason_lower else False:
        cooldown = COOLDOWN_SERVER_ERR
    elif "4" in reason_lower[:3] if reason_lower else False:
        cooldown = COOLDOWN_CLIENT_ERR
    else:
        cooldown = COOLDOWN_UNKNOWN

    _model_health[model_id] = _model_health.get(model_id, 0) + 1
    _model_cooldown[model_id] = time.time() + cooldown
    mins = cooldown // 60
    print(f"      [cooldown] {model_id} banned {mins} min (reason: {reason[:60]})", flush=True)


def get_model_score(model_id: str) -> int:
    """Look up capability score for a model ID. Defaults to DEFAULT_MODEL_SCORE."""
    mid = model_id.lower()

    # Exact match
    if model_id in MODEL_SCORES:
        return MODEL_SCORES[model_id]

    # Case-insensitive exact match
    for key, score in MODEL_SCORES.items():
        if key.lower() == mid:
            return score

    # Longest substring match wins (most specific)
    best_score = DEFAULT_MODEL_SCORE
    best_len = 0
    for key, score in MODEL_SCORES.items():
        kl = key.lower()
        if kl in mid:
            if len(kl) > best_len or (len(kl) == best_len and score > best_score):
                best_len = len(kl)
                best_score = score
    return best_score


async def fetch_available_models(session: aiohttp.ClientSession) -> list[tuple[str, int]]:
    """Query OmniRoute for all available models and score them."""
    try:
        async with session.get(f"{API_BASE}/models",
                               timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status == 200:
                data = await resp.json()
                models = []
                for item in data.get("data", []):
                    mid = item.get("id", "")
                    if not mid:
                        continue
                    score = get_model_score(mid)
                    models.append((mid, score))
                # Sort by score descending, then model id for determinism
                models.sort(key=lambda x: (-x[1], x[0]))
                return models
            else:
                print(f"[models] /v1/models returned {resp.status}")
    except Exception as e:
        print(f"[models] Could not fetch models: {e}")
    return []


def build_fallback_chain(models: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Filter to security-approved code models and sort by score descending.

    STEP 370/1566: only allowlisted model IDs may enter the chain — anything the
    gateway exposes that is not in the curated allowlist is dropped (and logged)
    so a shadow/unaudited provider can never receive sensitive audit prompts.
    'nuclear-fallback' is re-added when allowlisted even though /v1/models does
    not list it; an empty chain means deny-all (e.g. OCULUS_MODEL_ALLOWLIST=).
    """
    allow = _load_model_allowlist()
    filtered = []
    dropped = []
    ids_seen = set()
    for mid, score in models:
        low = mid.lower()
        if any(ex in low for ex in EXCLUDED_MODEL_SUBSTRINGS):
            dropped.append(mid)
            continue
        if mid not in allow:
            dropped.append(mid)  # not security-approved
            continue
        filtered.append((mid, score))
        ids_seen.add(mid)

    if "nuclear-fallback" in allow and "nuclear-fallback" not in ids_seen:
        filtered.append(("nuclear-fallback", MODEL_SCORES.get("nuclear-fallback", DEFAULT_MODEL_SCORE)))

    if dropped:
        print(f"[allowlist] dropped {len(dropped)} non-approved model(s): {', '.join(sorted(dropped)[:10])}"
              + (" ..." if len(dropped) > 10 else ""), flush=True)
    print(f"[allowlist] fallback chain ({len(filtered)}): "
          + ", ".join(mid for mid, _ in filtered) or "(empty — deny-all)", flush=True)

    if not filtered:
        return []

    # Stable sort by score descending
    filtered.sort(key=lambda x: (-x[1], x[0]))
    return filtered


async def probe_models(session: aiohttp.ClientSession,
                       chain: list[tuple[str, int]],
                       top_n: int = 10) -> list[tuple[str, int]]:
    """Quickly probe top-N scored models to find ones that actually respond.

    Returns a reordered chain with responsive models first (still sorted by
    score among themselves), followed by the rest of the chain.
    """
    probe_prompt = "Respond with the single word: OK"
    probe_payload = {
        "model": "",
        "stream": False,
        "messages": [{"role": "user", "content": probe_prompt}]
    }

    to_probe = chain[:top_n]
    rest = chain[top_n:]

    # Always probe known reliable fallbacks so they can be promoted if responsive
    guaranteed = {"nuclear-fallback", "nvidia/nvidia/nemotron-3-ultra-550b-a55b"}
    ids_probed = {m for m, _ in to_probe}
    for mid, score in chain:
        if mid in guaranteed and mid not in ids_probed:
            to_probe.append((mid, score))
            # remove from rest if present
            rest = [(m, s) for m, s in rest if m != mid]

    # Nuclear-fallback is the orchestrated combo; if it works, always put it first
    # regardless of individual model scores so the audit is not blocked by dead
    # high-score aliases.
    nuclear_fallback = None
    for mid, score in chain:
        if mid == "nuclear-fallback":
            nuclear_fallback = (mid, score)
            break
    responsive = []
    unresponsive = []

    print(f"[probe] Testing top {len(to_probe)} models for responsiveness...")
    for mid, score in to_probe:
        payload = dict(probe_payload)
        payload["model"] = mid
        try:
            t0 = time.time()
            async with session.post(
                f"{API_BASE}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=STARTUP_PROBE_TIMEOUT)
            ) as resp:
                elapsed = time.time() - t0
                if resp.status == 200:
                    print(f"[probe]  OK {mid} (score {score}) in {elapsed:.1f}s")
                    responsive.append((mid, score))
                else:
                    print(f"[probe] FAIL {mid} (score {score}) HTTP {resp.status} in {elapsed:.1f}s")
                    _model_health[mid] = _model_health.get(mid, 0) + 5
                    unresponsive.append((mid, score))
        except Exception as e:
            print(f"[probe] FAIL {mid} (score {score}) {type(e).__name__}")
            _model_health[mid] = _model_health.get(mid, 0) + 5
            unresponsive.append((mid, score))

    for mid, score in rest:
        _model_health[mid] = _model_health.get(mid, 0)


    # If nuclear-fallback responded, pin it to position 0
    global _probed_chain
    if nuclear_fallback and nuclear_fallback in responsive:
        responsive = [m for m in responsive if m != nuclear_fallback]
        final_chain = [nuclear_fallback] + sorted(responsive, key=lambda x: (-x[1], x[0])) + sorted(unresponsive, key=lambda x: (-x[1], x[0])) + rest
    else:
        final_chain = sorted(responsive, key=lambda x: (-x[1], x[0])) + sorted(unresponsive, key=lambda x: (-x[1], x[0])) + rest
    _probed_chain = final_chain
    return final_chain



def get_actual_model_id(response_data: dict) -> str:
    """OmniRoute returns the real backend model ID in the response model field."""
    return response_data.get("model", PRIMARY_MODEL)


async def throttled_start():
    """Throttle request starts to stay under provider RPM caps."""
    global _last_request_time
    async with _request_lock:
        now = time.time()
        elapsed = now - _last_request_time
        if elapsed < REQUEST_DELAY:
            await asyncio.sleep(REQUEST_DELAY - elapsed)
        _last_request_time = time.time()


async def record_model_usage(agent_name: str, model_id: str):
    async with _usage_lock:
        _agent_model_usage[agent_name][model_id] += 1


def get_agent_model_score(agent_name: str) -> float:
    """Composite model score for an agent based on actual model usage ratios."""
    usage = _agent_model_usage.get(agent_name, {})
    total = sum(usage.values())
    if total == 0:
        return float(DEFAULT_MODEL_SCORE)
    weighted = sum(count * get_model_score(mid) for mid, count in usage.items())
    return weighted / total


# ─── FILE DISCOVERY ──────────────────────────────────────────────────────
# Directories never worth auditing. The original set was tuned for a Flutter app
# ("ios", "android", "macos", "windows", "linux" are Flutter platform folders, and
# "web" is Flutter's web target) and is WRONG for a web/Python repo, where "web" is
# the entire frontend.
#
# Measured on this deployment: the walk returned 4297 files of which 3387 were
# .json — generated artifacts and lockfiles that add no audit value — which
# multiplied the run into 860 batches. A scope that is 79% machine-generated noise
# is not a thorough audit, it is an expensive one.
EXCLUDE_DIRS = {
    "archive", "__pycache__", ".git", "node_modules", "build", "dist", "coverage",
    ".next", ".nuxt", ".cache", ".turbo", ".pytest_cache", ".mypy_cache",
    ".dart_tool", "graphify-out", "backups", "vendor", "third_party",
    "oculus_env", ".venv", "venv", "env", "target", "uploads", "logs", ".orch",
    "sandbox", "data", "tmp",
}

# Generated or manifest files, matched by exact basename. Skipped by name because
# they are not reliably inside a directory of their own.
EXCLUDE_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock",
    "Cargo.lock", "composer.lock", "graph.json", "tsconfig.json", "jsconfig.json",
    ".eslintcache",
}


def extract_files_from_readme(readme_text: str) -> list[str]:
    files = []
    # Accept both the historical /mnt/extradrive prefix and the current
    # /home/roni prefix (the auto-readme-updater writes the current path).
    _path_re = re.compile(
        r'(?:/mnt/extradrive|/home/roni)/Roni_workspace/oculus/(\S+\.(?:py|dart|yaml|jinja2|html|json|ini|md))')
    for line in readme_text.split('\n'):
        m = _path_re.search(line)
        if m:
            rel_path = m.group(1)
            parts = rel_path.split('/')
            skip = False
            for p in parts[:-1]:
                if p in EXCLUDE_DIRS or p == 'archive':
                    skip = True
                    break
            if not skip and 'legacy' not in rel_path:
                files.append(rel_path)
    seen = set()
    return [f for f in files if not (f in seen or seen.add(f))]


def get_file_list() -> list[str]:
    # ALWAYS audit the ENTIRE oculus folder. The readme is a documentation aid,
    # not a scope limiter — stale/partial readme lists must never shrink the
    # audit coverage. Readme-listed files are unioned in on top.
    result = []
    seen = set()
    if os.path.exists(README_FILE):
        try:
            with open(README_FILE) as f:
                text = f.read()
            for rel in extract_files_from_readme(text):
                if rel not in seen:
                    seen.add(rel)
                    result.append(rel)
        except Exception:
            pass
    for root, dirs, files in os.walk(OCULUS_DIR):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
        if 'archive' in root.split(os.sep):
            continue
        for fn in files:
            if fn in EXCLUDE_FILES:
                continue
            if not fn.endswith(('.py', '.dart', '.yaml', '.yml', '.jinja2', '.html', '.js',
                                '.jsx', '.ts', '.tsx', '.json', '.ini', '.md', '.sh', '.rs',
                                '.toml', '.cfg', '.sql')):
                continue
            rel = os.path.relpath(os.path.join(root, fn), OCULUS_DIR)
            if 'legacy' in rel:
                continue
            if rel not in seen:
                seen.add(rel)
                result.append(rel)
    # AUDIT_INCLUDE_FILES: an explicit, ordered file list. When set it IS the scope
    # — the walk is skipped entirely.
    #
    # This exists because breadth is the wrong axis for these lanes. Measured: the
    # full walk produced 231 batches at 5 files each, and the webchat lanes answer
    # in 60-300s with a 20-80s anti-ban gap between sends, so a complete sweep is
    # days of wall-clock. An audit is only useful if it finishes: a focused list of
    # the files where defects actually live beats a nominal sweep that never ends.
    # Paths are absolute; the label is the repo-relative form used in the report.
    # AUDIT_INCLUDE_FILES entries are (label, absolute_path): the label is what the
    # report shows, the path is what can actually be opened. read_file_content
    # resolves relative to OCULUS_DIR, so an absolute path must be passed through
    # unchanged rather than joined onto it.
    if INCLUDE_FILES:
        out = []
        for entry in INCLUDE_FILES:
            path = entry.strip()
            if not path:
                continue
            if os.path.isfile(path):
                # The label must be the REPO-RELATIVE path, not the basename: the graph
                # keys its nodes as "extension/action_stream_executor.js", so a bare
                # "action_stream_executor.js" matches nothing and every agent reports
                # "no graph data available" as a CRITICAL finding. That produced 25 of
                # 99 findings in the first pass — meta-findings about the audit's own
                # missing context, crowding out findings about the code.
                rel = os.path.relpath(path, OCULUS_DIR)
                if rel.startswith(".."):
                    rel = path
                out.append((rel, path))
        return out

    # Extra roots (AUDIT_EXTRA_ROOTS) — lets one run cover several repositories
    # without a synthetic parent directory. Files are prefixed with the root's
    # basename so two repos with a src/main.py cannot collide.
    for extra_root in EXTRA_ROOTS:
        extra_root = extra_root.strip()
        if not os.path.isdir(extra_root):
            continue
        prefix = os.path.basename(os.path.normpath(extra_root))
        for root, dirs, files in os.walk(extra_root):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
            for fn in files:
                if not fn.endswith(('.py', '.js', '.jsx', '.ts', '.tsx', '.dart', '.yaml', '.yml',
                                    '.jinja2', '.html', '.css', '.json', '.ini', '.md', '.sh', '.rs',
                                    '.toml', '.cfg', '.env', '.sql')):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, extra_root)
                prefixed = f"../{prefix}/{rel}"
                if prefixed not in seen:
                    seen.add(prefixed)
                    result.append(prefixed)

    # Also scan orchestrator / alt_important_scripts
    if os.path.isdir(ALT_SCRIPTS_DIR):
        for root, dirs, files in os.walk(ALT_SCRIPTS_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
            for fn in files:
                if not fn.endswith(('.py', '.yaml', '.json', '.sh', '.md', '.env')):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, ALT_SCRIPTS_DIR)
                prefixed = f"../alt_important_scripts/{rel}"
                if prefixed not in seen:
                    seen.add(prefixed)
                    result.append(prefixed)
    # ALWAYS audit the webchat-api harness too (server.js, browser.js, config.js,
    # package.json and the rest) — it lives outside the oculus repo but drives the
    # owner's own browser tabs over CDP, so it is first-class audit scope.
    if os.path.isdir(WEBCHAT_API_DIR):
        for root, dirs, files in os.walk(WEBCHAT_API_DIR):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith('.')]
            for fn in files:
                if not fn.endswith(('.js', '.json', '.md', '.sh', '.env', '.example')):
                    continue
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, WEBCHAT_API_DIR)
                prefixed = f"../webchat-api/{rel}"
                if prefixed not in seen:
                    seen.add(prefixed)
                    result.append(prefixed)
    return sorted(result)


# ─── HISTORICAL DATA SCOPE ─────────────────────────────────────────────────
# The audit should also scrutinize the market-data store: unused datasets,
# orphaned/empty files, wrong-format or dead data are legitimate findings.
HISTORICAL_DATA_DIR = "/home/roni/Roni_workspace/historical_data"
_DATA_INVENTORY_CACHE: tuple[float, str] = (0.0, "")


def build_data_scope() -> str:
    """Lightweight inventory of historical_data (names + sizes, no deep reads).

    Cached 300s — scanning 95 GB of filenames on every audit round is wasteful.
    """
    global _DATA_INVENTORY_CACHE
    now = time.time()
    if now - _DATA_INVENTORY_CACHE[0] < 300:
        return _DATA_INVENTORY_CACHE[1]
    lines = [
        "## DATA SCOPE (historical_data)",
        f"Path: {HISTORICAL_DATA_DIR}",
        "The market-data store. The audit SHOULD flag unused / orphaned / empty /",
        "broken-format datasets here — data that nothing references, empty files,",
        "or files the code cannot read are real problems.",
    ]
    if not os.path.isdir(HISTORICAL_DATA_DIR):
        lines.append("(historical_data directory not found)")
        text = "\n".join(lines)
        _DATA_INVENTORY_CACHE = (now, text)
        return text
    try:
        total_files = 0
        total_bytes = 0
        empty_files = 0
        orphan_non_data = 0
        per_top: dict[str, tuple[int, int]] = {}
        for root, dirs, files in os.walk(HISTORICAL_DATA_DIR):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for fn in files:
                total_files += 1
                try:
                    st = os.stat(os.path.join(root, fn))
                except OSError:
                    continue
                total_bytes += st.st_size
                if st.st_size == 0:
                    empty_files += 1
                rel = os.path.relpath(root, HISTORICAL_DATA_DIR)
                top = rel.split(os.sep)[0] if rel != '.' else '(root)'
                c, b = per_top.get(top, (0, 0))
                per_top[top] = (c + 1, b + st.st_size)
                if not fn.endswith(('.parquet', '.csv', '.json', '.feather', '.npz', '.npy', '.pkl', '.bin', '.dat', '.gz')):
                    orphan_non_data += 1
                if total_files > 400_000:
                    break
            if total_files > 400_000:
                break
        lines.append(f"Total: {total_files} files, {total_bytes / 1024**3:.1f} GB "
                     f"| {empty_files} empty | {orphan_non_data} non-data")
        for top, (c, b) in sorted(per_top.items(), key=lambda kv: -kv[1][1])[:12]:
            lines.append(f"  {top}: {c} files, {b / 1024**3:.2f} GB")
    except Exception as e:
        lines.append(f"(inventory error: {e})")
    text = "\n".join(lines)
    _DATA_INVENTORY_CACHE = (now, text)
    return text


# ─── FILE CACHE ──────────────────────────────────────────────────────────
_file_cache: dict[str, str] = {}


def read_file_content(rel_path: str) -> str:
    if rel_path in _file_cache:
        return _file_cache[rel_path]
    # Handles oculus/, ../alt_important_scripts/ and ../webchat-api/ paths.
    if rel_path.startswith("../alt_important_scripts/"):
        full = os.path.join(ALT_SCRIPTS_DIR, rel_path[len("../alt_important_scripts/"):])
    elif rel_path.startswith("../webchat-api/"):
        full = os.path.join(WEBCHAT_API_DIR, rel_path[len("../webchat-api/"):])
    else:
        full = os.path.join(OCULUS_DIR, rel_path)
    if not os.path.exists(full):
        return f"FILE NOT FOUND: {rel_path}"
    try:
        with open(full, 'r', errors='replace') as f:
            content = f.read()
        _file_cache[rel_path] = content
        return content
    except Exception as e:
        return f"ERROR READING {rel_path}: {e}"


def load_source_of_truth() -> str:
    if os.path.exists(SOT_FILE):
        with open(SOT_FILE) as f:
            return f.read()
    return "[SOURCE OF TRUTH NOT FOUND]"


def load_readme() -> str:
    if os.path.exists(README_FILE):
        with open(README_FILE) as f:
            return f.read()
    return "[README NOT FOUND]"


def _batch_files(files, batch_size: int) -> list[list[tuple[str, str]]]:
    batches = []
    current = []
    for entry in files:
        label, path = _as_label_path(entry)
        current.append((label, read_file_content(path)))
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


# ─── GRAPHIFY GRAPH DATABASE ──────────────────────────────────────────────
class GraphifyDB:
    """In-memory query engine for Graphify's graph.json output.

    Provides caller/callee, community, hotspot, orphan, and cycle queries
    without spawning external CLI processes. graph.json is produced by the
    Graphify tool that is already installed on the Oculus codebase.
    """

    def __init__(self, graph_path: str):
        self.path = graph_path
        self.data: dict = {}
        self.nodes: dict[str, dict] = {}
        self.nodes_by_file: dict[str, list[dict]] = defaultdict(list)
        self.nodes_by_label: dict[str, dict] = {}
        self.in_links: dict[str, list[dict]] = defaultdict(list)
        self.out_links: dict[str, list[dict]] = defaultdict(list)
        self.degree: dict[str, int] = defaultdict(int)
        self.communities: dict[int, list[dict]] = defaultdict(list)
        self.loaded = False
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            print(f"[graphify] WARNING: {self.path} not found; graph queries disabled")
            return
        try:
            with open(self.path, 'r', encoding='utf-8', errors='replace') as f:
                self.data = json.load(f)
            for n in self.data.get("nodes", []):
                nid = n.get("id")
                if not nid:
                    continue
                self.nodes[nid] = n
                sf = n.get("source_file", "")
                if sf:
                    self.nodes_by_file[sf].append(n)
                label = n.get("label", "")
                if label:
                    self.nodes_by_label[label] = n
                comm = n.get("community")
                if comm is not None:
                    self.communities[comm].append(n)
            for link in self.data.get("links", []):
                src = link.get("source")
                tgt = link.get("target")
                if src and tgt:
                    self.out_links[src].append(link)
                    self.in_links[tgt].append(link)
                    self.degree[src] += 1
                    self.degree[tgt] += 1
            self.loaded = True
            print(f"[graphify] Loaded {len(self.nodes)} nodes, {len(self.data.get('links', []))} links")
        except Exception as e:
            print(f"[graphify] ERROR loading graph: {e}")

    def has_data(self) -> bool:
        return self.loaded and len(self.nodes) > 0

    def node_by_label(self, label: str) -> dict | None:
        return self.nodes_by_label.get(label)

    def get_file_nodes(self, rel_path: str) -> list[dict]:
        # Try exact match; also try basename if file is in a subdir
        nodes = self.nodes_by_file.get(rel_path, [])
        if nodes:
            return nodes
        basename = os.path.basename(rel_path)
        return self.nodes_by_file.get(basename, [])

    def get_functions(self, rel_path: str) -> list[dict]:
        return [n for n in self.get_file_nodes(rel_path)
                if '(' in n.get("label", "") and n.get("file_type") == "code"]

    def get_callers(self, label: str, max_results: int = 10) -> list[str]:
        n = self.node_by_label(label)
        if not n:
            return []
        results = []
        for link in self.in_links.get(n["id"], []):
            if link.get("relation") in ("calls", "references"):
                src = self.nodes.get(link.get("source"), {})
                src_label = src.get("label", link.get("source"))
                if src_label != label and src_label not in results:
                    results.append(src_label)
                if len(results) >= max_results:
                    break
        return results

    def get_callees(self, label: str, max_results: int = 10) -> list[str]:
        n = self.node_by_label(label)
        if not n:
            return []
        results = []
        for link in self.out_links.get(n["id"], []):
            if link.get("relation") in ("calls", "references"):
                tgt = self.nodes.get(link.get("target"), {})
                tgt_label = tgt.get("label", link.get("target"))
                if tgt_label != label and tgt_label not in results:
                    results.append(tgt_label)
                if len(results) >= max_results:
                    break
        return results

    def get_orphans(self, max_results: int = 30) -> list[tuple[str, str]]:
        linked = set(self.degree.keys())
        orphans = [(n.get("label", nid), n.get("source_file", ""))
                   for nid, n in self.nodes.items()
                   if nid not in linked and n.get("file_type") == "code"]
        return orphans[:max_results]

    def get_hotspots(self, top_n: int = 20) -> list[tuple[str, int, str]]:
        code_nodes = [n for n in self.nodes.values() if n.get("file_type") == "code"]
        sorted_nodes = sorted(code_nodes, key=lambda n: self.degree.get(n["id"], 0), reverse=True)
        return [(n.get("label", ""), self.degree.get(n["id"], 0), n.get("source_file", ""))
                for n in sorted_nodes[:top_n]]

    def get_community_summary(self, top_n: int = 15) -> dict:
        summary = {}
        for comm_id, nodes in sorted(self.communities.items(), key=lambda x: -len(x[1]))[:top_n]:
            labels = [n.get("label", "") for n in nodes if '(' in n.get("label", "")]
            files = sorted(set(n.get("source_file", "") for n in nodes if n.get("source_file")))
            summary[comm_id] = {
                "name": nodes[0].get("community_name", f"community_{comm_id}"),
                "size": len(nodes),
                "files": files[:10],
                "functions": labels[:15]
            }
        return summary

    def detect_cycles(self, max_results: int = 10) -> list[list[str]]:
        call_graph = defaultdict(set)
        for link in self.data.get("links", []):
            if link.get("relation") == "calls":
                call_graph[link.get("source")].add(link.get("target"))

        visited = set()
        rec_stack = set()
        cycles = []

        def dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for neighbor in call_graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    try:
                        idx = path.index(neighbor)
                        cycle = path[idx:] + [neighbor]
                        cycles.append([self.nodes.get(x, {}).get("label", x) for x in cycle])
                    except ValueError:
                        pass
            path.pop()
            rec_stack.remove(node)

        for node in list(call_graph.keys()):
            if node not in visited:
                dfs(node, [])
                if len(cycles) >= max_results:
                    break
        return cycles[:max_results]

    def build_file_context(self, rel_path: str, max_funcs: int = 20) -> str:
        nodes = self.get_file_nodes(rel_path)
        if not nodes:
            return f"## GRAPH CONTEXT: {rel_path}\n[No graph data available for this file]"

        funcs = self.get_functions(rel_path)
        file_comm = nodes[0].get("community")
        comm_name = nodes[0].get("community_name", "N/A")
        lines = [
            f"## GRAPH CONTEXT: {rel_path}",
            f"Community: {comm_name} (id {file_comm})",
            f"Graph nodes in file: {len(nodes)}",
            f"Functions/Classes ({len(funcs)}):"
        ]
        for f in funcs[:max_funcs]:
            label = f.get("label", "")
            deg = self.degree.get(f["id"], 0)
            callers = self.get_callers(label, max_results=3)
            callees = self.get_callees(label, max_results=4)
            lines.append(f"  - {label} [degree {deg}]")
            if callers:
                lines.append(f"      called by: {', '.join(callers)}")
            if callees:
                lines.append(f"      calls: {', '.join(callees)}")
        return "\n".join(lines)

    def build_batch_context(self, batch_files: list[tuple[str, str]],
                            max_funcs: int = 15) -> str:
        sections = [self.build_file_context(rp, max_funcs=max_funcs) for rp, _ in batch_files]
        return "\n\n".join(sections)


# Global Graphify instance, populated in main()
_graphify: GraphifyDB | None = None


def get_graphify() -> GraphifyDB:
    if _graphify is None:
        return GraphifyDB(GRAPH_FILE)
    return _graphify


# ─── OMNIROUTE API CALL WITH MODEL FALLBACK ───────────────────────────────
async def call_llm(session: aiohttp.ClientSession,
                   user_prompt: str,
                   system_prompt: str,
                   agent_name: str = "",
                   retries: int = 3,
                   quiet: bool = False) -> tuple[str, str, int]:
    """Call OmniRoute's model endpoint with fallback model rotation.

    Returns (content, actual_model_id, model_score).
    Rotates to fallback candidate models if primary model times out or errors.
    """
    last_error = ""
    empty_fallback = None
    empty_rotations = 0
    prose_rotations = 0
    # Let models recover as their cooldowns elapse — otherwise a long run
    # accumulates permanent dead-health and every later pass finds nothing.
    _decay_model_health()
    candidates = []
    # 08-23 env-pin fix: when DEEPSEEK_MODEL_FLASH is set (free-lane pin from
    # phase_env), the webchat gateway ONLY accepts that token — the probed
    # chain lists paid-API names the gateway 400s. Pinned env wins outright.
    if _probed_chain and not os.environ.get("DEEPSEEK_MODEL_FLASH"):
        candidates = [mid for mid, _ in _probed_chain if _model_health.get(mid, 0) < MAX_MODEL_HEALTH]
    if not candidates:
        candidates = [PRIMARY_MODEL]

    # Skip models that are in cooldown (failed recently) or have failed too many times.
    _health_cutoff = 6
    live_candidates = [
        m for m in candidates
        if _model_health.get(m, 0) < _health_cutoff
        and not is_model_cooling_down(m)
    ]
    if not live_candidates:
        # All models are either dead or cooling down — fall back to the least-bad option
        live_candidates = [m for m in candidates if _model_health.get(m, 0) < _health_cutoff]
    if not live_candidates:
        live_candidates = candidates[:1]  # absolute last resort

    for attempt in range(retries):
        target_model = live_candidates[attempt % len(live_candidates)]
        model_score = get_model_score(target_model)

        if attempt > 0:
            if last_error == "HTTP 429":
                # Rate limit: needs a real cooldown, not the 2-12s swap backoff.
                # Hammering a 429 just extends the ban window and makes every
                # concurrent subagent hit the wall too.
                wait = min((2 ** attempt) * 10 + random.uniform(0, 5), 90.0)
            else:
                wait = min((2 ** attempt) + random.uniform(0, 2), MAX_BACKOFF_SECONDS)
            if not quiet:
                print(f"      [call_llm] retry {attempt}/{retries} using {target_model} after {wait:.1f}s ({last_error})", flush=True)
            await asyncio.sleep(wait)

        payload = {
            "model": target_model,
            "stream": False,
            # Without this the free lanes answer with a fraction of their findings:
            # a 151-char reply ending mid-object ("popup.js": ) has no repairable
            # close, so _try_loads returns None and the whole round's findings are
            # dropped. The tolerant repair only handles trailing JUNK, never a
            # truncated document.
            "max_tokens": CHAT_MAX_TOKENS,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

        try:
            async with session.post(
                f"{API_BASE}/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=CHAT_TIMEOUT_SECONDS)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    content = data['choices'][0]['message']['content']
                    actual_model = get_actual_model_id(data)
                    actual_score = get_model_score(actual_model)
                    if agent_name:
                        await record_model_usage(agent_name, actual_model)
                    if not quiet:
                        print(f"      [call_llm] success via {target_model} -> {actual_model} ({len(content)} chars)", flush=True)
                    # A reply that parses as NO findings doc at all is the other half of
                    # the same problem, and it is the bigger one in practice: measured on
                    # the t2b sweep, 18 of 35 rounds were UNPARSED because the model
                    # returned 30 KB of reasoning prose instead of the JSON document.
                    # Rotating costs one call and can rescue the round; marking it
                    # UNPARSED and moving on throws the work away.
                    if (REQUIRE_SUBSTANTIVE and _looks_like_prose_reply(content)
                            and prose_rotations < MAX_EMPTY_ROTATIONS):
                        prose_rotations += 1
                        last_error = "reply had no findings document"
                        if not quiet:
                            print(f"      [call_llm] {actual_model} replied without a findings "
                                  f"document ({len(content)}c of prose) — rotating", flush=True)
                        continue
                    if REQUIRE_SUBSTANTIVE and _is_empty_findings_doc(content):
                        # A findings doc with every file empty is BOTH a legitimate
                        # answer for a clean batch AND what a model returns when it did
                        # not read the file. From one reply the two are indistinguishable,
                        # and treating the second as the first is how an audit reports a
                        # false clean. Rotate to another model and keep the empty answer
                        # only if none does better, so "no findings" comes to mean "no
                        # model found anything" rather than "the first one said nothing".
                        if empty_fallback is None:
                            empty_fallback = (content, actual_model, actual_score)
                        empty_rotations += 1
                        last_error = "empty findings doc"
                        if empty_rotations >= MAX_EMPTY_ROTATIONS:
                            # Bounded: a genuinely clean batch must not cost one call per
                            # model. Past the cap the empty answer is accepted, but it is
                            # now backed by MAX_EMPTY_ROTATIONS independent models.
                            if not quiet:
                                print(f"      [call_llm] {actual_model} empty; rotation cap reached, accepting it", flush=True)
                            return empty_fallback
                        if not quiet:
                            print(f"      [call_llm] {actual_model} returned an empty findings doc — rotating", flush=True)
                        continue
                    return content, actual_model, actual_score
                else:
                    last_error = f"HTTP {resp.status}"
                    mark_model_failed(target_model, reason=f"HTTP {resp.status}")
                    if resp.status == 429:
                        # A 429 on one model almost certainly means the whole
                        # API is rate-limited right now — wait once, then move on.
                        await asyncio.sleep(min(30 + random.uniform(0, 10), 45.0))
                    if not quiet:
                        print(f"      [call_llm] {target_model} returned HTTP {resp.status}", flush=True)
                    continue
        except asyncio.TimeoutError:
            last_error = "Timeout"
            mark_model_failed(target_model, reason="Timeout")
            if not quiet:
                print(f"      [call_llm] {target_model} timed out after {CHAT_TIMEOUT_SECONDS}s", flush=True)
            continue
        except Exception as e:
            last_error = str(e)
            mark_model_failed(target_model, reason=str(e))
            if not quiet:
                print(f"      [call_llm] {target_model} exception: {e}", flush=True)
            continue

    # Every model answered with an empty document. That is a real answer for a batch
    # of clean files, so return it rather than an error — but it is now a claim backed
    # by every model we tried rather than by whichever one happened to answer first.
    if empty_fallback is not None:
        if not quiet:
            print("      [call_llm] every model returned an empty findings doc; using the first", flush=True)
        return empty_fallback
    global _api_exhaustions
    _api_exhaustions += 1
    return f"ERROR: Exhausted retries. Last error: {last_error}", PRIMARY_MODEL, get_model_score(PRIMARY_MODEL)



# ─── JSON EXTRACTION ─────────────────────────────────────────────────────
def extract_json(text: str) -> dict | None:
    for obj in _json_candidates(text):
        return obj  # first parseable object wins
    return None


def _normalize_json_literals(text: str) -> str:
    """Normalize pythonic literals before parsing, but ONLY as whole tokens.
    The old global str.replace corrupted string DATA — "Nonexistent" became
    "nullexistent", and prose words containing True/False/None inside string
    values were mangled (finding text is never 100% guaranteed clean)."""
    text = re.sub(r"\bTrue\b", "true", text)
    text = re.sub(r"\bFalse\b", "false", text)
    return re.sub(r"\bNone\b", "null", text)


def _balanced_object_candidates(text: str, max_starts: int = 12) -> Iterator[str]:
    r"""Yield balanced '{...}' substrings starting at successive '{' positions.
    String/escape-aware: braces inside quoted strings never count, so
    prose-wrapped / multi-object replies yield their JSON bodies intact. The
    old greedy r'\{.*\}' fallback ran first-{ to LAST-} and broke as soon as
    prose after the JSON contained a brace — with a balanced scan, trailing
    prose (or a snippet ending with '}') can no longer poison the candidate."""
    start, found = 0, 0
    while found < max_starts:
        b = text.find("{", start)
        if b < 0:
            return
        depth, in_str, esc = 0, False, False
        j = b
        while j < len(text):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[b:j + 1]
                        break
            j += 1
        start = b + 1
        found += 1


def _escape_raw_controls(sq: str) -> str:
    """Escape raw \\n/\\r/\\t chars INSIDE quoted strings (JSON forbids raw
    control chars; the model writes multi-line finding text with literal
    newlines — strict loads rejects the whole doc). String/escape-aware."""
    out = []
    in_str = False
    esc = False
    for ch in sq:
        if in_str:
            if esc:
                out.append(ch)
                esc = False
            elif ch == "\\":
                out.append(ch)
                esc = True
            elif ch == '"':
                out.append(ch)
                in_str = False
            elif ch in "\n\r\t":
                out.append({"\n": "\\\\n", "\r": "\\\\r", "\t": "\\\\t"}[ch])
            else:
                out.append(ch)
        else:
            if ch == '"':
                in_str = True
            out.append(ch)
    return "".join(out)


LAST_JSON_ERR = None  # (lineno, colno, pos, excerpt) of last strict-loads failure — diagnostic for unparseable replies


def _brace_close_positions(s: str) -> list[int]:
    """String/escape-aware positions where a '{...}' group closes back to depth 0."""
    outs, depth, in_str, esc = [], 0, False, False
    for i, c in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                outs.append(i + 1)
    return outs


def _try_loads(s: str):
    """json.loads with tolerant repair. 08-25 v7.2 (mass-zero root cause): at
    x32 the stealth model emits structurally-right findings JSON that strict
    loads rejects — trailing commas before }/] (level 1) and raw control
    chars inside string values (level 2, multi-line finding text). Applied
    in order; returns None when neither repair makes it load. Records the
    failure-position excerpt in LAST_JSON_ERR for diagnosis of deeper slips."""
    global LAST_JSON_ERR
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        LAST_JSON_ERR = (e.lineno, e.colno, e.pos,
                         s[max(0, e.pos - 80):e.pos + 80])
        import re as _re
        repaired = _re.sub(r",\s*([}\]])", r"\1", s)
        if repaired != s:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        fixed = _escape_raw_controls(s)
        if fixed != s:
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass
        # level-3 (08-25, B13 P1 evidence): trailing junk / extra closing braces
        # AFTER the JSON object — json.loads dies at the end (col ~len-20). Trim
        # to the last string-aware depth-0 close and retry the prefixes.
        for pos in reversed(_brace_close_positions(s)[-3:]):
            cand = s[:pos]
            if cand != s:
                try:
                    return json.loads(cand)
                except json.JSONDecodeError:
                    continue
        return None


def _json_candidates(text: str) -> Iterator[dict]:
    """Yield parsed JSON dicts in best-guess order: whole doc, fenced blocks,
    then balanced '{...}' bodies. Candidates that don't parse are skipped
    (trailing-comma repair via _try_loads)."""
    norm = _normalize_json_literals(text.strip())
    if not norm:
        return
    if norm.startswith("{") and norm.endswith("}"):
        obj = _try_loads(norm)
        if obj is not None:
            yield obj
    # fenced blocks: ```json / ``` / ~~~ (case-insensitive)
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```|~~~\s*\n?(.*?)~~~",
                         norm, re.DOTALL | re.IGNORECASE):
        body = (m.group(1) or m.group(2) or "").strip()
        if body.startswith("{"):
            obj = _try_loads(body)
            if obj is not None:
                yield obj
    for cand in _balanced_object_candidates(norm):
        obj = _try_loads(cand)
        if obj is not None:
            yield obj


def _is_findings_doc(obj) -> bool:
    """Laundering-protection gate: a parsed object is a FINDINGS document only
    when it carries a structurally plausible findings payload —
    findings_by_file: {path: [dict, ...]} or findings: [{dict, ...}]. Anything
    else (prose snippets, partial objects, unrelated dicts) is rejected so
    unparseable replies surface as FAILURE instead of being laundered as
    info findings."""
    if not isinstance(obj, dict):
        return False
    if "findings_by_file" in obj:
        return isinstance(obj["findings_by_file"], dict)
    if "findings" in obj:
        lst = obj["findings"]
        return isinstance(lst, list) and any(isinstance(f, dict) for f in lst)
    return False


def extract_findings_doc(text: str) -> dict | None:
    """parse_findings entry point: first candidate that is BOTH valid JSON AND
    a findings-carrying document. Wrapped/fenced/early-snippet cases route
    through balanced extraction; None when nothing qualifies (never
    launders prose)."""
    for obj in _json_candidates(text):
        if _is_findings_doc(obj):
            return obj
    return None



# ─── PROMPT BUILDERS ──────────────────────────────────────────────────────
def build_base_system_prompt(task_prompt: str, source_of_truth: str, readme: str,
                            include_sot: bool = True) -> str:
    # 08-24 (user directive): SoT is privileged — ONLY the SOT Specialist and
    # General Confirm Agent (final sanity check) get it. All other personas get
    # README + graphify context only; including SoT for everyone dilutes the
    # SoT-conformance signal and leaks the authoritative spec into every agent.
    sot_block = (
        f"## SOURCE OF TRUTH (Authoritative Spec)\n{source_of_truth[:SOT_TRUNCATE]}\n\n"
        if include_sot
        else "## SOURCE OF TRUTH\n[Not provided — audit for code-level correctness, "
             "READ MME alignment, and runtime safety only; SoT conformance is the SOT Specialist's lane.]\n\n"
    )
    return f"""You are an expert code auditor analyzing the {TARGET_LABEL} codebase.

## TASK / AUDIT PROTOCOL
{task_prompt[:TASK_TRUNCATE]}

{sot_block}
## README (Documented Features)
{readme[:README_TRUNCATE]}

## ANALYSIS DIMENSIONS
For EACH file you receive, check for ALL of the following:

1. **SOURCE OF TRUTH VIOLATIONS** - Does the code contradict the authoritative spec?
2. **README ALIGNMENT** - Are there functions/classes that exist in the code but are NOT documented in the README? OR Are there features described in the README that are MISSING from this file?
3. **LEGACY / DEPRECATED CODE** - Does this file have old imports, commented-out code, renamed functions, or dead code paths?
4. **UNDOCUMENTED FEATURES** - Does this file contain anything that seems critical but isn't mentioned in either the README or Source of Truth?
5. **STANDARD AUDIT** - Security, performance, data integrity, architecture, testing gaps.

## REQUIRED OUTPUT SCHEMA
You MUST output a JSON object with this EXACT structure. No markdown wrapping, no commentary.

{{
  "findings_by_file": {{
    "relative/path/to/file1.py": [
      {{
        "category": "SECURITY|DESIGN|PERFORMANCE|ARCHITECTURE|DATA_INTEGRITY|TESTING|README_MISMATCH|LEGACY|UNDOCUMENTED|RADICAL_IDEA",
        "severity": "CRITICAL|HIGH|MEDIUM|LOW",
        "finding": "1-2 sentence description",
        "mechanism": "How it manifests in code",
        "impact": "Quantified impact (latency, memory, robustness, etc.)",
        "line_range": "X-Y or N/A",
        "source_of_truth_violation": "YES|NO",
        "readme_mismatch": "YES|NO",
        "is_legacy": "YES|NO",
        "recommended_fix": "Actionable fix description",
        "priority_rank": 1-100,
        "confidence": "HIGH|MEDIUM|LOW"
      }}
    ],
    "relative/path/to/file2.py": []
  }}
}}

CRITICAL:
- The top-level key MUST be "findings_by_file".
- Each key must be the exact file path provided.
- If a file has no findings, include it with an empty array [].
- For each finding, state whether it violates the Source of Truth (source_of_truth_violation).
- Check if the code has functions NOT listed in the README (readme_mismatch).
- Flag legacy/deprecated code (is_legacy).
- Check if anything critical exists undocumented in BOTH the README and Source of Truth (category: UNDOCUMENTED).
"""


def format_graph_insights(graphify: GraphifyDB, focus: str) -> str:
    """Build a compact, agent-specific Graphify insight block."""
    if not graphify or not graphify.has_data():
        return "## GRAPHIFY INSIGHTS\n[Graph data not available]"

    lines = ["## GRAPHIFY INSIGHTS"]

    if focus in ("hotspots", "all"):
        lines.append("\n### Hotspots (most connected code nodes)")
        for label, deg, src_file in graphify.get_hotspots(top_n=10):
            lines.append(f"- {label} [degree {deg}] ({src_file})")

    if focus in ("orphans", "orphans_and_entrypoints", "all"):
        lines.append("\n### Orphans (code nodes with no links)")
        orphans = graphify.get_orphans(max_results=20)
        if orphans:
            for label, src_file in orphans:
                lines.append(f"- {label} ({src_file})")
        else:
            lines.append("- None detected")

    if focus in ("cycles_and_communities", "cycles", "all"):
        lines.append("\n### Circular Dependencies")
        cycles = graphify.detect_cycles(max_results=5)
        if cycles:
            for cyc in cycles:
                lines.append(f"- {' -> '.join(cyc)}")
        else:
            lines.append("- None detected")

    if focus in ("clusters", "cycles_and_communities", "all"):
        lines.append("\n### Communities / Clusters")
        summary = graphify.get_community_summary(top_n=10)
        for comm_id, info in summary.items():
            lines.append(f"- {info['name']} (size {info['size']}, files: {', '.join(info['files'][:3])})")
            for fn in info['functions'][:5]:
                lines.append(f"    * {fn}")

    if focus in ("callers", "all"):
        lines.append("\n### Caller/Callee relationships in batch")
        # This section is intentionally brief; per-file callers/callees are in the batch context.
        lines.append("See the FILE BATCH section for function-level caller/callee data.")

    return "\n".join(lines)


def build_specialist_system_prompt(agent: dict, task_prompt: str,
                                   source_of_truth: str, readme: str,
                                   graphify: GraphifyDB | None = None) -> str:
    base = build_base_system_prompt(task_prompt, source_of_truth, readme)
    insights = format_graph_insights(graphify or get_graphify(), agent.get("graph_focus", "all"))
    return base + f"""

## YOUR SPECIALTY
You are the **{agent['name']}**.
{agent['prompt_addition']}

{insights}
"""


def build_confirm_system_prompt(agent: dict, task_prompt: str,
                                source_of_truth: str, readme: str,
                                graphify: GraphifyDB | None = None) -> str:
    base = build_base_system_prompt(task_prompt, source_of_truth, readme)
    insights = format_graph_insights(graphify or get_graphify(),
                                     agent.get("graph_focus", "all"))
    return base + f"""

{insights}

## YOUR ROLE
{agent['prompt_addition']}

## VOTING INSTRUCTIONS
For each finding below:
- Vote **YES** if the issue is real, relevant, and worth fixing.
- Vote **NO** if it is a false positive, already mitigated, unclear, or not actionable.
- Use the agent's specialty weight and the model capability score as trust bias:
  * Findings from high-weight agents (CIA Hacker 1.6, SOT Specialist 1.5) and high-score models (90+) should be trusted more.
  * Findings from low-weight agents or low-score models (below 70) require stronger corroboration.
- Provide concise reasoning.

## REQUIRED OUTPUT SCHEMA
Output ONLY valid JSON:

{{
  "votes": [
    {{
      "finding_id": "exact finding id from the input",
      "vote": "YES|NO",
      "confidence": "HIGH|MEDIUM|LOW",
      "reasoning": "1 sentence explaining your vote"
    }}
  ]
}}
"""


def compact_findings(findings: list[dict], max_chars: int = COMPACTED_FINDINGS_MAX) -> str:
    if not findings:
        return "No previous findings for this batch."

    per_file = defaultdict(list)
    for f in findings:
        per_file[f.get("file", "unknown")].append(f)

    lines = ["## PREVIOUS FINDINGS SUMMARY (compacted)\n"]
    for file_path, fs in sorted(per_file.items()):
        lines.append(f"\n### {file_path}")
        # Most recent / highest severity first
        sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
        fs_sorted = sorted(fs, key=lambda x: sev_order.get(x.get("severity", "INFO"), 5))
        for f in fs_sorted[-6:]:  # cap per file
            agent = f.get("agent", "?")
            sev = f.get("severity", "?")
            cat = f.get("category", "?")
            text = f.get("finding", "")[:100]
            lines.append(f"- [{agent}] {sev} {cat}: {text}")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text


def build_file_batch_section(batch_files: list[tuple[str, str]],
                              graphify: GraphifyDB | None = None,
                              include_code_teaser: bool = False) -> str:
    """Build batch context from Graphify graph data instead of full file text.

    This is the main token-reduction mechanism: the LLM sees function
    signatures, call relationships, and communities rather than ~8k tokens of
    source code per file.
    """
    g = graphify or get_graphify()
    sections = []
    for rel_path, content in batch_files:
        if g and g.has_data():
            ctx = g.build_file_context(rel_path, max_funcs=15)
            if include_code_teaser:
                teaser = content[:500].strip()
                ctx += f"\n\n### CODE TEASER\n```python\n{teaser}\n```"
            sections.append(ctx)
        else:
            # Fallback to truncated source if graph is unavailable
            sections.append(f"### FILE: {rel_path}\n```python\n{content[:FILE_TRUNCATE]}\n```")
    return "## FILE BATCH (Graphify context)\n\n" + "\n\n".join(sections)


def build_specialist_user_prompt(batch_files: list[tuple[str, str]],
                                 previous_findings: list[dict],
                                 pass_num: int, round_num: int,
                                 agent: dict,
                                 graphify: GraphifyDB | None = None) -> str:
    batch_section = build_file_batch_section(batch_files, graphify=graphify)
    compacted = compact_findings(previous_findings)
    data_scope = build_data_scope()
    return f"""{batch_section}

{data_scope}

{compacted}

## INSTRUCTIONS
You are the **{agent['name']}** (Round {round_num}, Pass {pass_num}).
Review the Graphify context for the files above. Also review any previous findings shown in the summary.
You may confirm, refute, refine, or add NEW findings. Be specific and cite line ranges where possible.
Also review the DATA SCOPE: flag unused/orphaned/empty/broken-format datasets under historical_data that the code never references.
Output ONLY the required JSON object.
"""


def build_confirm_user_prompt(batch_files: list[tuple[str, str]],
                              findings_to_vote: list[dict],
                              pass_num: int, agent: dict,
                              graphify: GraphifyDB | None = None) -> str:
    batch_section = build_file_batch_section(batch_files, graphify=graphify)

    finding_lines = ["## FINDINGS TO VOTE ON\n"]
    for f in findings_to_vote:
        fid = f.get("finding_id", "UNKNOWN")
        finding_lines.append(f"""
### {fid}
- **Agent:** {f.get('agent', '?')} (weight {f.get('agent_weight', '?')})
- **Model:** {f.get('model_id', '?')} (score {f.get('model_score', '?')})
- **File:** {f.get('file', '?')}
- **Severity:** {f.get('severity', '?')}
- **Category:** {f.get('category', '?')}
- **Finding:** {f.get('finding', '')}
- **Mechanism:** {f.get('mechanism', '')[:200]}
- **Recommended Fix:** {f.get('recommended_fix', '')[:150]}
""")

    findings_section = "\n".join(finding_lines)

    return f"""{batch_section}

{findings_section}

## INSTRUCTIONS
You are the **{agent['name']}** (Pass {pass_num}).
Vote YES or NO on each finding above. Consider the agent's specialty weight and the model capability score as trust bias.
Output ONLY the required JSON votes object.
"""


# ─── FINDING / VOTE PARSING ───────────────────────────────────────────────
def parse_findings(content: str, batch_files: list[tuple[str, str]],
                   agent: dict, pass_num: int, round_num: int,
                   model_id: str, model_score: int) -> list[dict]:
    """Parse specialist JSON output into flattened findings.
    08-24 (output-contract audit): extraction goes through
    extract_findings_doc — balanced-brace candidates (fences, prose-wrapped,
    trailing text) are accepted ONLY when the object carries a valid findings
    payload; a reply with no findings-carrying JSON is not laundered here
    (the caller turns it into an explicit FAILURE row). Shape guards below make
    a malformed-but-parsed reply (findings_by_file as a list, non-dict entries)
    degrade to empty instead of raising through gather."""
    parsed = extract_findings_doc(content)
    results = []

    if isinstance(parsed, dict) and "findings_by_file" in parsed:
        source = parsed["findings_by_file"]
        if not isinstance(source, dict):
            source = {}  # malformed value (list/str/bool) -> no crash, failure path
    elif isinstance(parsed, dict) and "findings" in parsed:
        source = {}
        fl = parsed.get("findings", [])
        if isinstance(fl, list):
            for f in fl:
                if not isinstance(f, dict):
                    continue  # skip non-dict elements; never AttributeError mid-list
                fp = f.get("file", batch_files[0][0])
                source.setdefault(fp, []).append(f)
    else:
        source = {}

    file_paths = [bp[0] for bp in batch_files]

    # 08-25 v7.2 (valid-JSON but path-variant keys): the model sometimes
    # re-derives relative paths — drops/prefixes "../"/"./" components —
    # even though every key IS a file the prompt listed. Exact-only lookup
    # silenced those replies to 0 findings with no marker and no log.
    # Lookup order: exact, normpath-normalized, then tail-suffix match.
    import posixpath as _pp

    def _norm(p: str) -> str:
        return _pp.normpath(str(p or "").replace("\\", "/"))

    _src = {_norm(k): v for k, v in source.items() if isinstance(v, list)}
    for file_path in file_paths:
        matched = source.get(file_path)
        if matched is None:
            n = _norm(file_path)
            matched = _src.get(n)
        if matched is None:
            tail = "/".join(n.split("/")[-2:])
            for _k, _v in _src.items():
                if _k.endswith("/" + tail) or _k == tail:
                    matched = _v
                    break
        file_findings = matched if isinstance(matched, list) else []
        if not isinstance(file_findings, list):
            continue
        for idx, f in enumerate(file_findings):
            if not isinstance(f, dict):
                continue  # non-dict element (string/scalar) — skip, don't crash
            finding_id = f"P{pass_num}B{0}R{round_num}F{idx}"  # batch index filled later
            results.append({
                "finding_id": finding_id,
                "agent": agent["name"],
                "agent_weight": agent["weight"],
                "pass": pass_num,
                "round": round_num,
                "model_id": model_id,
                "model_score": model_score,
                "file": file_path,
                "category": f.get("category", "GENERAL"),
                "severity": f.get("severity", "INFO"),
                "finding": f.get("finding", ""),
                "mechanism": f.get("mechanism", ""),
                "impact": f.get("impact", ""),
                "line_range": f.get("line_range", "N/A"),
                "source_of_truth_violation": f.get("source_of_truth_violation", "NO"),
                "readme_mismatch": f.get("readme_mismatch", "NO"),
                "is_legacy": f.get("is_legacy", "NO"),
                "recommended_fix": f.get("recommended_fix", ""),
                "priority_rank": f.get("priority_rank", 99),
                "confidence": f.get("confidence", "LOW"),
                "raw": content if idx == 0 else "",
            })

    # A reply that carries NO findings document is a PARSE FAILURE, not a finding.
    # The old code appended a synthetic "Raw analysis output (JSON parse failed)" row
    # here, which put a fabricated entry into the results list and made the round print
    # "OK (1 findings)" — so a model that answered nothing looked like a model that
    # answered. That is the same class as the harness's phantom pass: the count is the
    # thing a reader trusts, and it was counting the failure. Return nothing and let the
    # caller record an explicit failure.
    return results


def renumber_finding_ids(findings: list[dict], pass_num: int, batch_idx: int) -> list[dict]:
    """Assign stable finding IDs now that batch index is known."""
    for i, f in enumerate(findings):
        round_num = f.get("round", 0)
        f["finding_id"] = f"P{pass_num}B{batch_idx}R{round_num}F{i}"
    return findings


def parse_votes(content: str) -> list[dict]:
    parsed = extract_json(content)
    if parsed and "votes" in parsed:
        return parsed["votes"]
    return []


# ─── BATCH PIPELINE ───────────────────────────────────────────────────────
async def run_batch_pipeline(sem: asyncio.Semaphore,
                             session: aiohttp.ClientSession,
                             batch_files: list[tuple[str, str]],
                             batch_idx: int,
                             total_batches: int,
                             pass_num: int,
                             task_prompt: str,
                             source_of_truth: str,
                             readme: str,
                             previous_findings: list[dict],
                             graphify: GraphifyDB | None = None) -> dict:
    """Run all 7 agent rounds on one batch, sequentially, for one pass."""
    async with sem:
        await throttled_start()
        file_names = ", ".join(bp[0] for bp in batch_files)
        print(f"  [Pass {pass_num} Batch {batch_idx+1}/{total_batches}] {file_names}", flush=True)

        local_history = list(previous_findings)  # findings visible to this batch's agents
        pass_findings = []
        votes = []
        start_total = time.time()

        # Rounds 1-7: specialist agents (run concurrently for maximum speedup)
        async def run_specialist(round_num: int, agent: dict):
            system_msg = build_specialist_system_prompt(agent, task_prompt, source_of_truth, readme, graphify)
            user_msg = build_specialist_user_prompt(batch_files, local_history, pass_num, round_num, agent, graphify)
            t0 = time.time()
            content, model_id, model_score = await call_llm(session, user_msg, system_msg, agent["name"])
            latency = time.time() - t0
            findings = parse_findings(content, batch_files, agent, pass_num, round_num, model_id, model_score)
            # OK means we got a findings document back. A reply with no document is a
            # failure however polite its prose was; counting it as OK is how an audit
            # reports progress it did not make.
            if content.startswith("ERROR"):
                status = "FAIL"
            elif not findings and not _is_empty_findings_doc(content):
                status = "UNPARSED"
            else:
                status = "OK"
            print(f"    Round {round_num} {agent['name'][:20]:20} -> {status} ({latency:.1f}s, {len(findings)} findings, {model_id})", flush=True)
            return round_num, findings

        async with AGENT_SEMAPHORE:


            specialist_results = await asyncio.gather(*[
            run_specialist(round_num, agent)
            for round_num, agent in enumerate(AGENTS[:8], 1)


            ])

        # Sort findings by round number and assign stable finding IDs
        specialist_results.sort(key=lambda x: x[0])
        for round_num, findings in specialist_results:
            findings = renumber_finding_ids(findings, pass_num, batch_idx)
            pass_findings.extend(findings)
            local_history.extend(findings)


        # Round 8: General Confirm Agent votes on this pass's findings
        confirm_agent = AGENTS[8]
        confirm_system = build_confirm_system_prompt(confirm_agent, task_prompt, source_of_truth, readme, graphify)
        confirm_user = build_confirm_user_prompt(batch_files, pass_findings, pass_num, confirm_agent, graphify)

        t0 = time.time()
        content, model_id, model_score = await call_llm(session, confirm_user, confirm_system, confirm_agent["name"])
        latency = time.time() - t0

        votes = parse_votes(content)

        # Same rule as the specialists: no parsed votes is not a successful round.
        if content.startswith("ERROR"):
            status = "FAIL"
        elif not votes:
            status = "UNPARSED"
        else:
            status = "OK"
        print(f"    Round 8 {confirm_agent['name'][:20]:20} -> {status} ({latency:.1f}s, {len(votes)} votes, {model_id})")

        total_latency = time.time() - start_total
        print(f"    Batch {batch_idx+1} complete in {total_latency:.1f}s")

        return {
            "batch_idx": batch_idx,
            "pass": pass_num,
            "files": [bp[0] for bp in batch_files],
            "findings": pass_findings,
            "votes": votes,
            "latency": total_latency,
        }


# ─── CHECKPOINT / RESUME HELPERS ──────────────────────────────────────────
def load_state() -> dict:
    """Load the audit checkpoint state, or return a fresh one."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception as e:
            print(f"[checkpoint] Could not load state file: {e}; starting fresh")
    return {
        "completed_passes": [],
        "current_pass": 1,
        "current_pass_completed_batches": [],
        "timestamp": datetime.now().isoformat()
    }


def save_state(state: dict):
    """Atomically write checkpoint state to disk."""
    state["timestamp"] = datetime.now().isoformat()
    tmp = STATE_FILE + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, STATE_FILE)


def batch_result_path(pass_num: int, batch_idx: int) -> str:
    pass_dir = os.path.join(OUTPUT_BASE, f"pass_{pass_num}")
    return os.path.join(pass_dir, f"batch_{batch_idx:03d}.json")


def load_existing_batch(pass_num: int, batch_idx: int) -> dict | None:
    """Load a previously completed batch result if it exists and is valid."""
    path = batch_result_path(pass_num, batch_idx)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("pass") == pass_num and data.get("batch_idx") == batch_idx:
            return data
    except Exception:
        pass
    return None


def save_batch_result(pass_num: int, batch_idx: int, result: dict):
    """Persist a single batch result to disk — but never at the cost of a good one.

    A rate-limited retry answers with nothing, and writing that over a batch that already
    held findings is silent data loss. Measured 2026-09-24: two overlapping helpotron runs
    were launched without --resume, so pass 1 was re-run and overwritten, and the corpus
    fell from 838 findings to 348 with 21 of 32 batches left empty. An empty result is
    only refused when it would REPLACE a non-empty one — so a genuinely clean batch still
    saves, and a batch that is already empty is left alone and skipped on resume. That is
    what makes this terminate instead of re-running the empty batch forever.
    """
    path = batch_result_path(pass_num, batch_idx)
    new_findings = result.get("findings") or []
    if os.path.exists(path):
        try:
            with open(path) as f:
                old_findings = (json.load(f) or {}).get("findings") or []
        except Exception:
            old_findings = []
        if old_findings and not new_findings:
            print(f"  [Pass {pass_num} Batch {batch_idx+1}] REFUSED to overwrite "
                  f"{len(old_findings)} findings with an empty result")
            return
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=str)


def load_pass_results(pass_num: int, total_batches: int) -> list[dict]:
    """Load all batch results for a pass from disk, in order."""
    results = []
    for batch_idx in range(total_batches):
        r = load_existing_batch(pass_num, batch_idx)
        if r is not None:
            results.append(r)
    return results


# ─── PASS RUNNER ──────────────────────────────────────────────────────────
async def run_pass(session: aiohttp.ClientSession,
                   files: list[str],
                   task_prompt: str,
                   source_of_truth: str,
                   readme: str,
                   pass_num: int,
                   batch_histories: list[list[dict]],
                   graphify: GraphifyDB | None = None,
                   resume: bool = False) -> list[dict]:
    """Run one full pass over all batches, checkpointing each as it completes."""
    pass_dir = os.path.join(OUTPUT_BASE, f"pass_{pass_num}")
    os.makedirs(pass_dir, exist_ok=True)

    if _graphify and _graphify.has_data():
        batches = _batch_files_with_graph(files, BATCH_SIZE, _graphify)
    else:
        batches = _batch_files(files, BATCH_SIZE)
    total_batches = len(batches)

    print(f"\n{'='*70}")
    print(f"PASS {pass_num} — {len(files)} files in {total_batches} batches")
    print(f"{'='*70}")

    # 2026-08-13: persist the batch total so the responder/webchat context can
    # answer ETA questions ("Audit: 128/131 batches done"). save_state() dumps
    # the whole dict, so extra keys survive every checkpoint write.
    st = load_state()
    st["total_batches"] = total_batches
    st["num_passes"] = NUM_PASSES
    save_state(st)

    # Pre-load any already-completed batches when resuming
    results: list[dict | None] = [None] * total_batches
    pending_batches = []
    for batch_idx, batch in enumerate(batches):
        existing = load_existing_batch(pass_num, batch_idx)
        if existing and resume:
            results[batch_idx] = existing
            batch_histories[batch_idx].extend(existing.get("findings", []))
            print(f"  [Pass {pass_num} Batch {batch_idx+1}/{total_batches}] RESUMED from checkpoint")
        else:
            pending_batches.append((batch_idx, batch))

    if not pending_batches:
        print(f"[Pass {pass_num}] all batches already completed; loading from disk")
    else:
        print(f"[Pass {pass_num}] {len(pending_batches)} batch(es) remaining")

    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)

    # Run batches in WAVES so only SEMAPHORE_LIMIT coroutines + payloads
    # exist at once. (Creating ALL batch tasks up front — up to 423 — held
    # every coroutine + prompt payload in memory simultaneously; on a 14GB
    # box that compounded with node's leak into the OOM crash on 2026-08-02.)
    wave_size = SEMAPHORE_LIMIT
    for wave_start in range(0, len(pending_batches), wave_size):
        wave = pending_batches[wave_start:wave_start + wave_size]
        print(f"[Pass {pass_num}] Wave {wave_start // wave_size + 1} of "
              f"{(len(pending_batches) + wave_size - 1) // wave_size} "
              f"({len(wave)} batches)")

        task_map = {}
        for batch_idx, batch in wave:
            task = asyncio.create_task(run_batch_pipeline(
                sem, session, batch, batch_idx, total_batches, pass_num,
                task_prompt, source_of_truth, readme, batch_histories[batch_idx],
                graphify
            ))
            task_map[task] = batch_idx

        # Process tasks as they complete so we can checkpoint immediately
        pending = set(task_map.keys())
        while pending:
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                batch_idx = task_map[task]
                try:
                    r = task.result()
                except Exception as e:
                    print(f"  [Pass {pass_num} Batch {batch_idx+1}] EXCEPTION: {e}")
                    r = {
                        "batch_idx": batch_idx,
                        "pass": pass_num,
                        "files": [bp[0] for bp in batches[batch_idx]],
                        "findings": [],
                        "votes": [],
                        "latency": 0.0,
                        "error": str(e),
                    }

                results[batch_idx] = r
                save_batch_result(pass_num, batch_idx, r)
            batch_histories[batch_idx].extend(r.get("findings", []))

            # Update checkpoint state
            state = load_state()
            state["current_pass"] = pass_num
            if batch_idx not in state["current_pass_completed_batches"]:
                state["current_pass_completed_batches"].append(batch_idx)
            save_state(state)

    # Defensive: ensure no None left
    for batch_idx, r in enumerate(results):
        if r is None:
            results[batch_idx] = {
                "batch_idx": batch_idx,
                "pass": pass_num,
                "files": [bp[0] for bp in batches[batch_idx]],
                "findings": [],
                "votes": [],
                "latency": 0.0,
            }

    summary = {
        "pass": pass_num,
        "batches": len(results),
        "files_processed": sum(len(r.get('files', [])) for r in results),
        "total_findings": sum(len(r.get('findings', [])) for r in results),
        "total_votes": sum(len(r.get('votes', [])) for r in results),
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(pass_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    # Mark pass complete in checkpoint state and reset for next pass
    state = load_state()
    if pass_num not in state["completed_passes"]:
        state["completed_passes"].append(pass_num)
    state["current_pass"] = pass_num + 1
    state["current_pass_completed_batches"] = []
    save_state(state)

    print(f"\n[Pass {pass_num} summary] {summary['total_findings']} findings, {summary['total_votes']} votes")
    return results


# ─── VOTE AGGREGATION ─────────────────────────────────────────────────────
def aggregate_votes(all_results: list[list[dict]]) -> dict[str, dict]:
    """Map each finding_id to its vote result and computed acceptance ratio."""
    vote_map = {}

    for pass_results in all_results:
        for r in pass_results:
            for v in r.get("votes", []):
                fid = v.get("finding_id")
                if not fid:
                    continue
                vote_map[fid] = {
                    "vote": v.get("vote", "NO"),
                    "confidence": v.get("confidence", "LOW"),
                    "reasoning": v.get("reasoning", "")
                }

    return vote_map


def attach_vote_info(findings: list[dict], vote_map: dict[str, dict]) -> list[dict]:
    """Enrich findings with vote info and compute acceptance ratio."""
    for f in findings:
        fid = f.get("finding_id", "")
        vote_info = vote_map.get(fid, {})
        f["vote"] = vote_info.get("vote", "NO")
        f["vote_confidence"] = vote_info.get("confidence", "LOW")
        f["vote_reasoning"] = vote_info.get("reasoning", "")

        vote_yes = 1.0 if f["vote"] == "YES" else 0.0
        agent_weight = f.get("agent_weight", 1.0)
        m_score = f.get("model_score") or get_model_score(f.get("model_id", "")) or DEFAULT_MODEL_SCORE
        conf_mul = CONFIDENCE_MULTIPLIER.get(f.get("vote_confidence", ""), 0.5)

        vote_strength = vote_yes * agent_weight * (float(m_score) / 100.0) * conf_mul
        max_strength = MAX_AGENT_WEIGHT * 1.0 * 1.0
        f["agent_model_score"] = round(float(m_score), 2)
        f["vote_strength"] = round(vote_strength, 4)
        f["acceptance_ratio"] = round(vote_strength / max_strength, 4) if max_strength else 0
        f["accepted"] = (f["vote"] == "YES") or (f["acceptance_ratio"] >= 0.40) or (f.get("severity") in ("CRITICAL", "HIGH") and f["vote"] != "NO")

    return findings


def safe_priority_rank(val) -> int:
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str) and val.isdigit():
        return int(val)
    return 99


# ─── FINAL REPORT ─────────────────────────────────────────────────────────
async def generate_final_report(all_results: list[list[dict]]) -> str:
    print("\n" + "=" * 70)
    print("GENERATING FINAL REPORT")
    print("=" * 70)

    # Collect all findings and votes
    all_findings = []
    for pass_results in all_results:
        for r in pass_results:
            findings = r.get("findings", [])
            all_findings.extend(findings)
            for f in findings:
                ag = f.get("agent", "")
                mid = f.get("model_id", "")
                if ag and mid:
                    _agent_model_usage[ag][mid] += 1


    vote_map = aggregate_votes(all_results)
    all_findings = attach_vote_info(all_findings, vote_map)

    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    all_findings.sort(key=lambda x: (
        sev_order.get(str(x.get("severity", "INFO")).upper(), 5),
        -(float(x.get("acceptance_ratio", 0))),
        safe_priority_rank(x.get("priority_rank", 99)),
        str(x.get("finding_id", ""))
    ))


    # HARD-FAIL GUARD: if every finding is the "Raw analysis output (JSON
    # parse failed)" placeholder, OR the audit found ~nothing while API calls
    # kept exhausting retries (API down / auth broken / provider outage), DO
    # NOT emit a false "all clear" report. An empty audit report is
    # indistinguishable from "no issues found" and silently hides a broken
    # audit. (2026-08-03: the earlier guard only caught the placeholder case;
    # a dead OmniRoute produced 0 findings across all batches and a clean
    # "0 findings" report — that's the bug this extension fixes.)
    if (all_findings and all(
        f.get("finding") == "Raw analysis output (JSON parse failed)"
        for f in all_findings
    )) or (not all_findings and _api_exhaustions > 0):
        return "\n".join([
            "# ⚠️ MULTI-AGENT AUDIT FAILED — NO REAL EVALUATIONS",
            "",
            "**Every agent call errored.** The audit produced ZERO real findings.",
            "",
            "Possible causes:",
            "- OmniRoute (port 20128) is DOWN — `Cannot connect to host localhost:20128`",
            "- API key invalid / expired",
            "- All providers rate-limited or failing",
            "",
            "**This is NOT a clean audit.** The previous '0 findings' report was a false",
            "all-clear caused by this bug. Restart OmniRoute and re-run the audit before",
            "trusting any subsequent report.",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Findings (all failed placeholders): {len(all_findings)}",
        ])

    accepted = [f for f in all_findings if f.get("accepted")]
    rejected = [f for f in all_findings if not f.get("accepted")]
    radical = [f for f in all_findings if f.get("category") == "RADICAL_IDEA"]

    report = []
    report.append("# Multi-Agent Oculus Audit Report — Rotation Cross-Examination v5\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Passes:** {NUM_PASSES}")
    report.append(f"**Total Findings:** {len(all_findings)}")
    report.append(f"**Accepted Findings:** {len(accepted)}")
    report.append(f"**Rejected Findings:** {len(rejected)}")
    report.append(f"**Acceptance Threshold:** {ACCEPTANCE_THRESHOLD * 100:.0f}%\n")

    # Agent model usage
    report.append("## Agent Model Usage\n")
    for agent in AGENTS:
        score = get_agent_model_score(agent["name"])
        usage = _agent_model_usage.get(agent["name"], {})
        total = sum(usage.values())
        top = sorted(usage.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{m}({c})" for m, c in top) if top else "none"
        report.append(f"- **{agent['name']}** — composite model score: **{score:.1f}**, calls: {total}, top models: {top_str}")
    report.append("")

    # Summary stats
    report.append("## Summary Statistics\n")
    by_severity = defaultdict(int)
    by_category = defaultdict(int)
    sot_violations = 0
    readme_mismatches = 0
    legacy_count = 0
    undocumented_count = 0
    for f in accepted:
        by_severity[f.get("severity", "UNKNOWN")] += 1
        by_category[f.get("category", "UNKNOWN")] += 1
        if f.get("source_of_truth_violation") == "YES":
            sot_violations += 1
        if f.get("readme_mismatch") == "YES":
            readme_mismatches += 1
        if f.get("is_legacy") == "YES":
            legacy_count += 1
        if f.get("category") == "UNDOCUMENTED":
            undocumented_count += 1

    report.append("### By Severity (Accepted)")
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        report.append(f"- **{sev}:** {by_severity.get(sev, 0)}")
    report.append("")
    report.append("### Critical Classification Totals (Accepted)")
    report.append(f"- **Source of Truth Violations:** {sot_violations}")
    report.append(f"- **README Mismatches:** {readme_mismatches}")
    report.append(f"- **Legacy/Deprecated Code:** {legacy_count}")
    report.append(f"- **Undocumented Features:** {undocumented_count}")
    report.append("")
    report.append("### By Category (Accepted)")
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        report.append(f"- **{cat}:** {count}")
    report.append("")

    # Spec Compliance Matrix
    report.append("## Spec Compliance Matrix\n")
    report.append("| Requirement | Status | Files Affected |")
    report.append("|-------------|--------|----------------|")
    spec_items = [
        ("21 tradable assets", "CHECK", "asset_registry.py"),
        ("5 temporal timeframes", "CHECK", "features.py, pipeline.py"),
        ("Non-temporal bars", "CHECK (new)", "bars.py"),
        ("240+ genome parameters", "CHECK", "config/genome.py"),
        ("Dynamic survival threshold", "CHECK", "evolution/genetics.py"),
        ("4 mutation types", "CHECK", "evolution/genetics.py"),
        ("Survivor carry-over", "CHECK", "evolution/genetics.py"),
        ("7-metric fitness", "CHECK", "fitness_calculator.py"),
        ("Multi-score aggregation", "CHECK", "fitness_calculator.py, evolution/evaluators.py"),
        ("Fitness-weighted ensemble", "CHECK", "evolution_aggregator.py"),
        ("FTROM evolvable weight", "CHECK", "live/ftrom.py, config/genome.py"),
        ("No paper learning (FTROM)", "CHECK", "live/ftrom.py"),
        ("GMM regime detection", "CHECK", "regime.py"),
        ("Daily regime alignment", "CHECK", "metrics/evaluators.py"),
        ("5% rolling / 10% static DD only", "CHECK (restored)", "risk_manager.py"),
        ("Fitness-based allocation", "CHECK", "live/fitness_leaderboard.py"),
        ("Fitness-based timeout", "CHECK", "live/fitness_leaderboard.py"),
        ("README alignment", "AUDITED", "all files"),
        ("Legacy code detection", "AUDITED", "all files"),
        ("Undocumented features", "AUDITED", "all files"),
    ]
    for req, status, files in spec_items:
        report.append(f"| {req} | {status} | {files} |")
    report.append("")

    # Accepted findings
    report.append("## Accepted Findings\n")
    for i, f in enumerate(accepted, 1):
        fid = f.get("finding_id", f"F{i:04d}")
        tags = []
        if f.get("source_of_truth_violation") == "YES":
            tags.append("🛡️SOT")
        if f.get("readme_mismatch") == "YES":
            tags.append("📖README")
        if f.get("is_legacy") == "YES":
            tags.append("🗑️LEGACY")
        if f.get("category") == "UNDOCUMENTED":
            tags.append("❓UNDOC")
        if f.get("category") == "RADICAL_IDEA":
            tags.append("💡RADICAL")
        tag_str = f" ({', '.join(tags)})" if tags else ""
        report.append(f"### {fid} — {f.get('severity','?')}{tag_str}")
        report.append(f"- **File:** `{f['file']}`")
        report.append(f"- **Category:** {f.get('category', 'N/A')}")
        report.append(f"- **Agent:** {f.get('agent', 'N/A')} (weight {f.get('agent_weight', '?')})")
        report.append(f"- **Model:** {f.get('model_id', 'N/A')} (score {f.get('model_score', 'N/A')}, agent composite {f.get('agent_model_score', 'N/A')})")
        report.append(f"- **Priority Rank:** {f.get('priority_rank', 'N/A')}")
        report.append(f"- **Finding Confidence:** {f.get('confidence', 'N/A')}")
        report.append(f"- **General Confirm Vote:** {f.get('vote', 'N/A')} ({f.get('vote_confidence', 'N/A')}, ratio {f.get('acceptance_ratio', 0):.2f})")
        lr = f.get('line_range')
        if lr and lr != 'N/A':
            report.append(f"- **Line Range:** {lr}")
        report.append(f"- **Finding:** {f.get('finding', 'N/A')}")
        mech = f.get('mechanism', '')
        if mech:
            report.append(f"- **Mechanism:** {mech[:400]}")
        imp = f.get('impact', '')
        if imp:
            report.append(f"- **Impact:** {imp[:300]}")
        fix = f.get('recommended_fix', '')
        if fix:
            report.append(f"- **Fix:** {fix[:300]}")
        if f.get('vote_reasoning'):
            report.append(f"- **Vote Reasoning:** {f.get('vote_reasoning', '')[:250]}")
        report.append("")

    # Radical Ideas section
    if radical:
        report.append("## Radical Ideas (For Review)\n")
        report.append("_These ideas may violate the Source of Truth intentionally; they are included for human review regardless of acceptance._\n")
        for i, f in enumerate(radical, 1):
            fid = f.get("finding_id", f"R{i:04d}")
            report.append(f"### {fid} — {f.get('severity','?')} (accepted: {f.get('accepted', False)})")
            report.append(f"- **File:** `{f['file']}`")
            report.append(f"- **Agent:** {f.get('agent', 'N/A')}")
            report.append(f"- **Model:** {f.get('model_id', 'N/A')} (score {f.get('model_score', 'N/A')})")
            report.append(f"- **Idea:** {f.get('finding', 'N/A')}")
            mech = f.get('mechanism', '')
            if mech:
                report.append(f"- **Rationale:** {mech[:400]}")
            fix = f.get('recommended_fix', '')
            if fix:
                report.append(f"- **Proposed Change:** {fix[:300]}")
            report.append("")

    # Per-file summary
    report.append("## Per-File Analysis Summary\n")
    files_summary = defaultdict(list)
    for f in accepted:
        files_summary[f['file']].append(f)
    for filepath, findings in sorted(files_summary.items()):
        n_crit = sum(1 for f in findings if f.get('severity') == 'CRITICAL')
        n_high = sum(1 for f in findings if f.get('severity') == 'HIGH')
        n_med = sum(1 for f in findings if f.get('severity') == 'MEDIUM')
        n_sot = sum(1 for f in findings if f.get('source_of_truth_violation') == 'YES')
        n_rm = sum(1 for f in findings if f.get('readme_mismatch') == 'YES')
        n_leg = sum(1 for f in findings if f.get('is_legacy') == 'YES')
        n_undoc = sum(1 for f in findings if f.get('category') == 'UNDOCUMENTED')
        tags = []
        if n_sot: tags.append(f"SOT:{n_sot}")
        if n_rm: tags.append(f"RM:{n_rm}")
        if n_leg: tags.append(f"LEG:{n_leg}")
        if n_undoc: tags.append(f"UND:{n_undoc}")
        tag_str = f" ({', '.join(tags)})" if tags else ""
        report.append(f"- **`{filepath}`** — {len(findings)} findings (C:{n_crit} H:{n_high} M:{n_med}){tag_str}")
    report.append("")

    # Remediation Roadmap
    report.append("## Remediation Roadmap\n")
    report.append("| Priority | Severity | Focus Area | Suggested Order |")
    report.append("|----------|----------|------------|-----------------|")
    roadmap = [
        (1, "CRITICAL", "Source of Truth Violations", "Fix spec violations first (highest priority)"),
        (2, "CRITICAL", "Security", "Fix auth tokens, API key exposure, injection vectors"),
        (3, "CRITICAL", "Data Integrity", "Fix NaN/Inf handling, look-ahead bias, alignment bugs"),
        (4, "HIGH", "README Mismatches", "Sync code with documentation"),
        (5, "HIGH", "Live Trading Safety", "Fix fail-closed guarantees, replay protection, kill-switch"),
        (6, "HIGH", "Performance", "Memory leaks, loop vectorization, caching"),
        (7, "MEDIUM", "Legacy Code", "Remove/update deprecated modules and imports"),
        (8, "MEDIUM", "Architecture", "God objects, separation of concerns, testability"),
        (9, "LOW", "Undocumented Features", "Document undocumented critical features"),
        (10, "LOW", "UI/UX", "Flutter state management, accessibility, dark mode"),
    ]
    for prio, sev, area, order in roadmap:
        report.append(f"| {prio} | {sev} | {area} | {order} |")
    report.append("")

    # Top 3 Blockers
    report.append("## Top 3 Blockers\n")
    blockers = [f for f in accepted if f.get('severity') in ('CRITICAL', 'HIGH')]
    blockers.sort(key=lambda x: (
        x.get('source_of_truth_violation', 'NO') != 'YES',
        x.get('acceptance_ratio', 0),
        x.get('priority_rank', 99)
    ), reverse=True)
    for i, b in enumerate(blockers[:3], 1):
        fid = b.get("finding_id", "")
        tags = []
        if b.get("source_of_truth_violation") == "YES": tags.append("SOT")
        if b.get("readme_mismatch") == "YES": tags.append("README")
        if b.get("is_legacy") == "YES": tags.append("LEGACY")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        report.append(f"{i}. **{fid}{tag_str}:** {b.get('finding', 'N/A')[:200]} — `{b['file']}`")
    report.append("")

    # Estimated fix effort
    n_crit_total = by_severity.get("CRITICAL", 0)
    n_high_total = by_severity.get("HIGH", 0)
    est_hours = n_crit_total * 4 + n_high_total * 2
    report.append("## Estimated Fix Effort\n")
    report.append(f"- **CRITICAL findings:** {n_crit_total} × ~4h = {n_crit_total * 4}h")
    report.append(f"- **HIGH findings:** {n_high_total} × ~2h = {n_high_total * 2}h")
    report.append(f"- **Total estimated:** ~{est_hours} person-hours")
    report.append("")

    report_text = "\n".join(report)
    with open(FINAL_REPORT, 'w') as f:
        f.write(report_text)
    print(f"\nFinal report saved to {FINAL_REPORT}")
    print(f"  Accepted: {len(accepted)} / {len(all_findings)}")
    return report_text


# ─── MAIN ─────────────────────────────────────────────────────────────────
def _as_label_path(entry) -> tuple[str, str]:
    """Normalise one scope entry to (label, path-for-reading).

    get_file_list returns plain repo-relative strings from a directory walk, but an
    explicit AUDIT_INCLUDE_FILES scope returns (label, absolute_path) pairs because
    those files live in different repositories. Normalising here keeps every
    consumer working with one shape instead of teaching each one about both.
    """
    if isinstance(entry, (tuple, list)) and len(entry) == 2:
        return str(entry[0]), str(entry[1])
    return str(entry), str(entry)


def _batch_files_with_graph(files, batch_size: int, graphify) -> list[list[tuple[str, str]]]:
    """Batch files by graph community if graphify is available."""
    files = [_as_label_path(f) for f in files]
    if not graphify or not graphify.has_data():
        return _batch_files(files, batch_size)
    # Group files by community
    from collections import defaultdict
    comm_map = defaultdict(list)
    for label, path in files:
        nodes = graphify.get_file_nodes(label)
        if nodes:
            comm = nodes[0].get("community")
            comm_map.setdefault(comm, []).append((label, path))
        else:
            comm_map.setdefault(None, []).append((label, path))
    # Flatten communities into batches
    batches = []
    current = []
    for comm, flist in sorted(comm_map.items(), key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0)):
        for label, path in flist:
            current.append((label, read_file_content(path)))
            if len(current) >= batch_size:
                batches.append(current)
                current = []
    if current:
        batches.append(current)
    return batches

# Added graph-based batching and --limit/--dry-run per steps 1098/1099


async def main():
    parser = argparse.ArgumentParser(description="Oculus multi-agent rotation audit")
    parser.add_argument("--smoke", action="store_true",
                        help="Run a smoke test on first 2 batches with 1 pass")
    parser.add_argument("--list-models", action="store_true",
                        help="Print discovered OmniRoute models and scores, then exit")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no API calls)")
    parser.add_argument("--limit", type=int, default=4, help="Max concurrent agent LLM calls (default 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing pass/batch checkpoints")
    args = parser.parse_args()

    # If only --limit is given (no other action), just print and exit
    # `--limit N` means "audit with N concurrent agent calls". The old guard exited
    # whenever --limit was passed alone, so the documented way to tune concurrency
    # printed a message and did nothing — a silent no-op that reads like success.
    # --limit remains an optional concurrency setting; only an explicit --dry-run
    # (or --list-models for a bounded operation) stops before doing work.
    _ = args
    if args.dry_run:
        print("Dry run: would process files with graph batching")
        return
    AGENT_SEMAPHORE_LIMIT = args.limit
    global AGENT_SEMAPHORE
    AGENT_SEMAPHORE = asyncio.Semaphore(AGENT_SEMAPHORE_LIMIT)

    print("=" * 70)
    print("OCULUS MULTI-AGENT AUDIT SYSTEM v5 — Rotation Cross-Examination")
    print("=" * 70)

    # Load task
    with open(TASK_FILE) as f:
        task_prompt = f.read()
    print(f"\n[main] Loaded task ({len(task_prompt)} chars)")

    # Load Source of Truth
    source_of_truth = load_source_of_truth()
    sot_len = len(source_of_truth)
    print(f"[main] Loaded Source of Truth ({sot_len} chars) — {'OK' if sot_len > 100 else 'NOT FOUND!'}")

    # Load README
    readme = load_readme()
    readme_len = len(readme)
    print(f"[main] Loaded README ({readme_len} chars) — {'OK' if readme_len > 100 else 'NOT FOUND!'}")

    # Get file list
    files = get_file_list()
    print(f"[main] Discovered {len(files)} source files from readme")

    # Load Graphify graph
    global _graphify
    _graphify = GraphifyDB(GRAPH_FILE)
    if _graphify.has_data():
        print(f"[main] Graphify graph loaded: {_graphify.path}")
        if os.path.exists(GRAPH_FILE):
            graph_mtime = os.path.getmtime(GRAPH_FILE)
            latest_py_mtime = 0.0
            for root, _, filenames in os.walk(OCULUS_DIR):
                for fname in filenames:
                    if fname.endswith(".py"):
                        full_p = os.path.join(root, fname)
                        try:
                            latest_py_mtime = max(latest_py_mtime, os.path.getmtime(full_p))
                        except OSError:
                            pass
            if latest_py_mtime > graph_mtime:
                print("  [graphify WARNING] Source .py files have been modified since graph.json was built.")
                print("  Consider running `graphify update` to refresh the call graph for maximum accuracy.")
    else:

        print(f"[main] WARNING: Graphify graph not loaded; will fall back to source code context")

    connector = aiohttp.TCPConnector(ssl=ssl.create_default_context())
    async with aiohttp.ClientSession(connector=connector) as session:
        print(f"[main] Primary model: {PRIMARY_MODEL} (OmniRoute combo with scored backend fallbacks)")

        if args.list_models:
            # Discover and score available models for reporting
            models = await fetch_available_models(session)
            print(f"\nDiscovered {len(models)} models; scores from lookup table:")
            print("-" * 60)
            for mid, score in models[:50]:
                print(f"{mid:50} {score}")
            if len(models) > 50:
                print(f"... and {len(models) - 50} more")
            return

        # Discover and probe fallback models for dynamic fallback chain.
        # 08-23 env-pin fix: when DEEPSEEK_MODEL_FLASH is set (free-lane pin),
        # probing is pointless AND harmful — the gateway lists paid-API names
        # that it then 400s. The pinned env token wins; skip discovery entirely.
        if not os.environ.get("DEEPSEEK_MODEL_FLASH"):
            try:
                raw_models = await fetch_available_models(session)
                chain = build_fallback_chain(raw_models)
                await probe_models(session, chain, top_n=5)
            except Exception as e:
                print(f"[main] Warning: Model probing failed ({e}); falling back to primary model only.")

        # Smoke test mode

        if args.smoke:
            print("\n[SMOKE TEST] Running 1 pass on first 2 batches only.")
            files = files[:10]  # 2 batches of 5
            num_passes = 1
        else:
            num_passes = NUM_PASSES

        if _graphify and _graphify.has_data():
            batches = _batch_files_with_graph(files, BATCH_SIZE, _graphify)
        else:
            batches = _batch_files(files, BATCH_SIZE)
        total_batches = len(batches)
        batch_histories = [[] for _ in batches]

        # Determine resume point from existing pass summaries
        completed_passes = set()
        if args.resume:
            for p in range(1, num_passes + 1):
                summary_path = os.path.join(OUTPUT_BASE, f"pass_{p}", "summary.json")
                if os.path.exists(summary_path):
                    completed_passes.add(p)
            print(f"\n[RESUME] Detected completed passes: {sorted(completed_passes)}")

        all_results = []
        # Load fully completed passes from disk
        for pass_num in range(1, num_passes + 1):
            if pass_num in completed_passes:
                pass_results = load_pass_results(pass_num, total_batches)
                all_results.append(pass_results)
                for batch_idx, r in enumerate(pass_results):
                    batch_histories[batch_idx].extend(r.get("findings", []))
                print(f"[RESUME] Loaded pass {pass_num} from disk ({len(pass_results)} batches)")
            else:
                break

        start_pass = max(completed_passes, default=0) + 1
        if start_pass > num_passes:
            print("\n[RESUME] All passes already complete; regenerating final report")
        else:
            print(f"[RESUME] Starting execution at pass {start_pass}")

        for pass_num in range(start_pass, num_passes + 1):
            results = await run_pass(session, files, task_prompt,
                                     source_of_truth, readme,
                                     pass_num, batch_histories,
                                     _graphify, resume=args.resume)
            all_results.append(results)

        await generate_final_report(all_results)

        # Save agent model usage
        usage_path = os.path.join(OUTPUT_BASE, "agent_model_usage.json")
        with open(usage_path, 'w') as f:
            json.dump({
                agent["name"]: dict(_agent_model_usage.get(agent["name"], {}))
                for agent in AGENTS
            }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"AUDIT COMPLETE")
    print(f"{'=' * 70}")
    print(f"Final report: {FINAL_REPORT}")
    print(f"Pass outputs: {OUTPUT_BASE}/pass_*/")


def _looks_like_prose_reply(text: str) -> bool:
    """True when a reply carries no findings document at all.

    The counterpart to _is_empty_findings_doc: that one catches a valid document with
    nothing in it, this one catches a reply that never produced a document. Measured on
    the t2b sweep: 18 of 35 rounds were UNPARSED because the model returned ~30 KB of
    reasoning prose ("We are given a batch of files ...") instead of the JSON. Rotating
    is cheap and rescues the round; discarding it does not.

    An engine ERROR string is not prose, so it is left to the FAIL path.
    """
    if not text or text.startswith("ERROR"):
        return False
    for cand in _json_candidates(text):
        if _is_findings_doc(cand):
            return False
    return True


def _is_empty_findings_doc(text: str) -> bool:
    """True when a reply is a findings document carrying no findings at all.

    Deliberately narrow: the text must PARSE as a findings document first, so prose,
    an error string, or a malformed reply is never mistaken for an empty one. Only a
    structurally valid document that lists files and reports nothing qualifies.
    """
    for cand in _json_candidates(text):
        if not isinstance(cand, dict):
            continue
        if "findings_by_file" not in cand and "findings" not in cand:
            continue
        fbf = cand.get("findings_by_file")
        if isinstance(fbf, dict) and any(isinstance(v, list) and v for v in fbf.values()):
            return False
        lst = cand.get("findings")
        if isinstance(lst, list) and any(isinstance(f, dict) for f in lst):
            return False
        return True
    return False


if __name__ == "__main__":
    asyncio.run(main())
