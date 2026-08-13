#!/usr/bin/env python3
"""
PROJECT MASTER ORCHESTRATOR v1.0
================================
Fully automated, self-adaptive, remotely controllable master orchestrator for your project.

Features:
  - Infinite loop: audit → cross-eval → execute → analyze → graphify rebuild → repeat
  - LLM self-adaptation: reasoning tier analyzes and rewrites workflow scripts as needed
  - Graphify rebuild after each execution step
  - Telegram bot with full remote control + direct LLM chat interface
  - Persistent task board (agent_todo_list.json) for LLM agent
  - State checkpointing and resume capability
  - Error recovery with LLM-based fixes
  - Comprehensive logging and monitoring

Usage:
    python3 run_workflow.py --config config.yaml
    python3 run_workflow.py --resume
    python3 run_workflow.py --status
"""

import os
import sys
import json
import yaml
import asyncio
import subprocess
import logging
import argparse
import shutil
import re
import difflib
import time
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import aiohttp
import subprocess
from telegram import Update, Chat
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters
from telegram.error import TelegramError


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE SUMMON (autonomous fixer)
# ─────────────────────────────────────────────────────────────────────────────
# The bot can invoke the Claude Code CLI (print mode) as a subprocess to fix
# issues it cannot resolve itself. SAFETY RAILS:
#   - single-turn, non-interactive (`claude -p`)
#   - hard timeout + output cap
#   - caller enforces a SUMMON_BUDGET per cycle (see SettingsManager) so a
#     failed fix cannot spiral into an unbounded loop — beyond the budget the
#     pipeline HALTs for a human.
# Note: the `claude` CLI authenticates via the user's Anthropic account. The
# LLM API key is passed through the environment so any LLM-based
# tooling the fix touches works with the right credentials.
SUMMON_LOG_FILE = os.path.join(
    os.getenv("WORK_DIR", os.path.join(os.path.expanduser("~"), "orchestrator_data")),
    "claude_summon.log",
)


def log_summon(text: str):
    try:
        with open(SUMMON_LOG_FILE, "a") as f:
            f.write(f"[{datetime.now().isoformat()}] {text}\n")
    except Exception:
        pass


def call_claude(issue_context: str, workdir: str = None, timeout: int = 600) -> Dict:
    """Invoke Claude Code CLI in print mode to fix an issue.

    Returns {"ok": bool, "rc": int, "output": str}. Single-turn, non-interactive.
    """
    prompt = (
        "You are fixing an issue in the Project quantitative trading system.\n\n"
        "INSTRUCTIONS:\n"
        "1. Diagnose the issue below against the REAL codebase (root modules — "
        "note `your_package.X` imports often shadow root modules; prefer root paths).\n"
        "2. Apply the minimal correct fix to the real code. Respect existing APIs.\n"
        "3. Verify your fix by running the failing command(s).\n"
        "4. Report exactly what you changed and the verification result.\n\n"
        f"ISSUE:\n{issue_context}\n"
    )
    cmd = ["claude", "-p", "--output-format", "json", prompt]
    env = os.environ.copy()
    # ensure LLM key + paths are visible to the fixer subprocess
    env.setdefault("LLM_API_KEY", "")
    log_summon(f"SUMMON workdir={workdir} timeout={timeout}")
    try:
        r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                           timeout=timeout, env=env)
        out = (r.stdout or "") + ("\n" + r.stderr if r.stderr else "")
        log_summon(f"rc={r.returncode} len={len(out)}")
        return {"ok": r.returncode == 0, "rc": r.returncode, "output": out[-20000:]}
    except subprocess.TimeoutExpired:
        log_summon(f"TIMEOUT {timeout}s")
        return {"ok": False, "rc": -1, "output": f"Claude timed out after {timeout}s"}
    except Exception as e:
        log_summon(f"ERROR {e}")
        return {"ok": False, "rc": -2, "output": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG & LOGGER
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {}
LOGGER = logging.getLogger(__name__)
STATE = {
    "cycle": 0,
    "step": 0,
    "phase": "idle",
    "paused": False,
    "running": False,
    "last_audit_file": None,
    "last_plan_file": None,
    "last_error": None,
    "edit_mode": "restricted",
    "start_time": None,
    "phase_start_time": None,
}


class OrchestratorConfig:
    """Load and validate configuration from YAML."""

    @staticmethod
    def _expand(value):
        """Recursively expand ${VAR} references (and ${VAR|default}) from env."""
        if isinstance(value, str):
            def repl(m):
                var, sep, default = m.group(1).partition("|")
                return os.environ.get(var, default if sep else m.group(0))
            return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(\|[^}]*)?\}', repl, value)
        if isinstance(value, dict):
            return {k: OrchestratorConfig._expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [OrchestratorConfig._expand(v) for v in value]
        return value

    @staticmethod
    def load(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            LOGGER.error(f"Config file not found: {config_path}")
            sys.exit(1)

        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        cfg = OrchestratorConfig._expand(cfg)

        required = [
            'telegram_bot_token', 'telegram_chat_id',
            'llm_api_key',
            'repo_dir', 'work_dir',
            'scripts'
        ]
        for key in required:
            if key not in cfg:
                LOGGER.error(f"Missing required config key: {key}")
                sys.exit(1)
        if not cfg['telegram_bot_token'] or cfg['telegram_bot_token'] == "${TELEGRAM_BOT_TOKEN}":
            LOGGER.error("TELEGRAM_BOT_TOKEN is empty or unexpanded. Set it in the environment or config.")
            sys.exit(1)

        # LLM key fallback: if the placeholder wasn't expanded, read the
        # standard tokens file used by the other pipeline scripts.
        if not cfg.get('llm_api_key') or cfg['llm_api_key'] in ("${LLM_API_KEY}", "None"):
            for cand in [
                os.path.expanduser('~/.config/orchestrator/llm_api_key'),
                os.path.expanduser('~/.config/orchestrator/llm_api_key'),
            ]:
                if os.path.exists(cand):
                    try:
                        content = open(cand).read()
                        m = re.search(r'(sk-[A-Za-z0-9]{20,})', content)
                        if m:
                            cfg['llm_api_key'] = m.group(1)
                            break
                    except Exception:
                        pass
        if not cfg.get('llm_api_key') or cfg['llm_api_key'] in ("${LLM_API_KEY}", "None"):
            LOGGER.warning("LLM_API_KEY not set; LLM agent features will be degraded.")

        return cfg


class StateManager:
    """Manage workflow state checkpointing and resumption."""
    
    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state = {
            "cycle": 0,
            "step": 0,
            "phase": "idle",
            "paused": False,
            "running": False,
            "last_audit_file": None,
            "last_plan_file": None,
            "last_error": None,
            "edit_mode": "restricted",
            "start_time": None,
            "phase_start_time": None,
        }
        self.load()
    
    def load(self):
        """Load state from checkpoint file."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    loaded = json.load(f)
                    self.state.update(loaded)
                    LOGGER.info(f"Loaded state from checkpoint: cycle={self.state['cycle']}, step={self.state['step']}, phase={self.state['phase']}")
            except Exception as e:
                LOGGER.warning(f"Could not load state file: {e}. Starting fresh.")
    
    def save(self):
        """Save state to checkpoint file."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            LOGGER.error(f"Could not save state: {e}")
    
    def update(self, **kwargs):
        """Update state and save to disk."""
        self.state.update(kwargs)
        self.save()


class SettingsManager:
    """Persistent, Telegram-editable settings the orchestrator MUST follow.

    Stored as JSON so it can be changed on the fly from Telegram (structured
    /set commands or natural language routed through the LLM). Re-read from
    disk before every phase so edits take effect immediately.
    """

    DEFAULTS = {
        # "full"  -> keep looping audit->cross_eval->execution forever
        # "once"  -> run a single full cycle then stop
        "loop_mode": "full",
        # "none" | "audit" | "cross_eval" | "execution" | "cycle"
        # When set to a phase, the orchestrator pauses AFTER that phase
        # completes and waits for /resume. "cycle" pauses after a full cycle.
        "pause_after": "none",
        # Strict escalation is the default: the engine tries OmniRoute (5) ->
        # fast tier (2) -> reasoning tier (2), then HALTS for human
        # assistance. "continue" is an emergency override only (records the
        # failed step and skips it) and is never used automatically.
        "on_step_failure": "halt",
        # Max cycles before stopping (null/0 = unlimited)
        "max_cycles": 0,
        # Seconds between Telegram progress pings (min 60)
        "notify_interval": 300,
        # "restricted" only lets the LLM edit the 3 workflow scripts;
        # "full" allows editing any file under repo_dir.
        "edit_mode": "restricted",
        # "on" sends phase start / ~50% / finish pings; "off" silences them.
        "notifications": "on",
        # "on" commits any plan changes to the self-run workflow scripts on a
        # dedicated branch and merges back only after the cycle completes.
        "selfmod_branching": "on",
        # The readme is now refreshed by dedicated plan steps appended by the
        # cross-eval (they document the plan's own changes). The orchestrator
        # auto-update is off by default to avoid double work.
        "readme_autoupdate": "off",
        # "on" lets the bot summon Claude Code (claude -p) to fix issues it
        # can't resolve; "off" always HALTs for a human. summon_budget caps
        # how many auto-summons run per cycle so a bad fix can't loop forever.
        "auto_summon_claude": "on",
        "summon_budget": 3,
        # Token-efficiency (8_7): "off" resumes cached passes via
        # audit_state.json checkpoints instead of re-auditing the unchanged
        # codebase fresh every cycle (was burning ~4,635 paid calls/pass).
        "fresh_audit": "off",
    }

    def __init__(self, settings_file: str):
        self.settings_file = settings_file

    def load(self) -> Dict:
        """Load current settings (defaults merged over any persisted file)."""
        settings = dict(self.DEFAULTS)
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        for k, v in loaded.items():
                            if k in settings:
                                settings[k] = v
            except Exception as e:
                LOGGER.warning(f"Could not load settings file {self.settings_file}: {e}")
        return settings

    def get(self, key: str = None):
        """Return one key or the full settings dict."""
        s = self.load()
        return s if key is None else s.get(key)

    def update(self, patch: Dict) -> Dict:
        """Merge a patch into persisted settings atomically and return the new dict."""
        settings = self.load()
        for k, v in (patch or {}).items():
            if k in settings:
                settings[k] = v
            else:
                LOGGER.warning(f"Ignoring unknown settings key: {k}")
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        tmp = self.settings_file + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(settings, f, indent=2)
        os.replace(tmp, self.settings_file)
        LOGGER.info(f"Settings updated: {json.dumps(settings)}")
        return settings

    def describe(self) -> str:
        """Human-readable settings summary for Telegram."""
        s = self.load()
        lines = [
            "⚙️ **Current Settings**",
            f"  loop_mode: `{s['loop_mode']}`",
            f"  pause_after: `{s['pause_after']}`",
            f"  on_step_failure: `{s['on_step_failure']}`",
            f"  max_cycles: `{s['max_cycles'] or 'unlimited'}`",
            f"  notify_interval: `{s['notify_interval']}s`",
            f"  edit_mode: `{s['edit_mode']}`",
            f"  notifications: `{s.get('notifications', 'on')}`",
            f"  selfmod_branching: `{s.get('selfmod_branching', 'on')}`",
            f"  readme_autoupdate: `{s.get('readme_autoupdate', 'off')}`",
            f"  auto_summon_claude: `{s.get('auto_summon_claude', 'on')}`",
            f"  summon_budget: `{s.get('summon_budget', 3)}`",
        ]
        return "\n".join(lines)


class TaskManager:
    """Manage persistent task board for LLM agent."""
    
    def __init__(self, task_file: str):
        self.task_file = task_file
        self.tasks = {}
        self.load()
    
    def load(self):
        """Load tasks from JSON file."""
        if os.path.exists(self.task_file):
            try:
                with open(self.task_file, 'r') as f:
                    data = json.load(f)
                    self.tasks = {task['id']: task for task in data.get('tasks', [])}
                    LOGGER.info(f"Loaded {len(self.tasks)} tasks from {self.task_file}")
            except Exception as e:
                LOGGER.warning(f"Could not load tasks: {e}. Starting fresh.")
        else:
            self._create_default_tasks()
    
    def _create_default_tasks(self):
        """Create default task list."""
        default_tasks = [
            {
                "id": "T001",
                "description": "Check Graphify rebuild succeeded after execution",
                "condition": "graphify_last_run_success == false",
                "action": "verify_graphify_rebuild",
                "status": "pending"
            },
            {
                "id": "T002",
                "description": "Verify no new SoT violations in the latest audit",
                "condition": "audit_findings_count > 0",
                "action": "check_sot_violations",
                "status": "pending"
            },
            {
                "id": "T003",
                "description": "Check Git repo is clean and all commits pushed",
                "condition": "git_status_dirty == true",
                "action": "verify_git_clean",
                "status": "pending"
            },
            {
                "id": "T004",
                "description": "Validate that the latest plan has at least 1 step",
                "condition": "plan_step_count == 0",
                "action": "validate_plan_steps",
                "status": "pending"
            },
            {
                "id": "T005",
                "description": "Check for stale or missing config values in YAML",
                "condition": "config_validation_failed == true",
                "action": "validate_config",
                "status": "pending"
            },
            {
                "id": "T006",
                "description": "Verify the audit and cross-eval scripts are up-to-date",
                "condition": "script_version_mismatch == true",
                "action": "verify_script_versions",
                "status": "pending"
            },
            {
                "id": "T007",
                "description": "Ensure no orphaned or dead code after refactors",
                "condition": "dead_code_detected == true",
                "action": "check_dead_code",
                "status": "pending"
            },
            {
                "id": "T008",
                "description": "Check if Telegram bot is still connected and responsive",
                "condition": "bot_connection_failed == true",
                "action": "verify_telegram_connection",
                "status": "pending"
            },
            {
                "id": "T009",
                "description": "Verify logs folder has enough disk space",
                "condition": "disk_space_low == true",
                "action": "check_disk_space",
                "status": "pending"
            },
            {
                "id": "T010",
                "description": "Validate that all imports resolve in the 3 workflow scripts",
                "condition": "import_errors_detected == true",
                "action": "validate_imports",
                "status": "pending"
            },
            {
                "id": "T011",
                "description": "Check for sudden API credit drops (your LLM provider)",
                "condition": "api_credits_low == true",
                "action": "check_api_credits",
                "status": "pending"
            },
            {
                "id": "T012",
                "description": "Verify the Graphify graph isn't stale before a new execution cycle",
                "condition": "graphify_stale == true",
                "action": "verify_graphify_fresh",
                "status": "pending"
            },
        ]
        
        self.tasks = {task['id']: task for task in default_tasks}
        self.save()
    
    def save(self):
        """Save tasks to JSON file."""
        try:
            os.makedirs(os.path.dirname(self.task_file), exist_ok=True)
            with open(self.task_file, 'w') as f:
                data = {"tasks": list(self.tasks.values())}
                json.dump(data, f, indent=2)
        except Exception as e:
            LOGGER.error(f"Could not save tasks: {e}")
    
    def get_task(self, task_id: str) -> Optional[Dict]:
        """Get a specific task by ID."""
        return self.tasks.get(task_id)
    
    def update_task(self, task_id: str, status: str, result: str = ""):
        """Update task status."""
        if task_id in self.tasks:
            self.tasks[task_id]['status'] = status
            if result:
                self.tasks[task_id]['last_result'] = result
            self.tasks[task_id]['last_updated'] = datetime.now().isoformat()
            self.save()
            LOGGER.info(f"Updated task {task_id}: {status}")
    
    def get_all_tasks(self) -> List[Dict]:
        """Get all tasks."""
        return list(self.tasks.values())
    
    def get_pending_tasks(self) -> List[Dict]:
        """Get all pending tasks."""
        return [t for t in self.tasks.values() if t['status'] == 'pending']


class GitManager:
    """Manage Git operations (commit, rollback)."""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
    
    def commit(self, message: str) -> bool:
        """Commit changes to Git."""
        try:
            subprocess.run(
                ["git", "-C", self.repo_path, "add", "-A"],
                check=True, capture_output=True, timeout=30
            )
            subprocess.run(
                ["git", "-C", self.repo_path, "commit", "-m", message],
                check=True, capture_output=True, timeout=30
            )
            LOGGER.info(f"Git commit: {message}")
            return True
        except subprocess.CalledProcessError as e:
            LOGGER.error(f"Git commit failed: {e.stderr.decode()}")
            return False
        except Exception as e:
            LOGGER.error(f"Git commit error: {e}")
            return False
    
    def rollback(self) -> bool:
        """Rollback last commit."""
        try:
            subprocess.run(
                ["git", "-C", self.repo_path, "reset", "--hard", "HEAD~1"],
                check=True, capture_output=True, timeout=30
            )
            LOGGER.info("Rolled back last commit")
            return True
        except Exception as e:
            LOGGER.error(f"Rollback failed: {e}")
            return False
    
    def checkout_files(self, files: List[str]) -> bool:
        """Checkout specific files from HEAD."""
        try:
            subprocess.run(
                ["git", "-C", self.repo_path, "checkout", "--"] + files,
                check=True, capture_output=True, timeout=30
            )
            LOGGER.info(f"Checked out files: {files}")
            return True
        except Exception as e:
            LOGGER.error(f"Checkout failed: {e}")
            return False

    # ── self-modification branch isolation ──────────────────────────────────
    # When a plan changes the scripts that RUN the pipeline itself, edits and
    # step commits go to a dedicated branch and are merged back to main only
    # AFTER the cycle completes — so the running pipeline is never broken by
    # edits to its own code.
    def current_branch(self) -> str:
        try:
            r = subprocess.run(
                ["git", "-C", self.repo_path, "branch", "--show-current"],
                capture_output=True, text=True, timeout=20)
            return r.stdout.strip()
        except Exception:
            return ""

    def create_branch(self, name: str) -> bool:
        """Create a branch from current HEAD and switch to it."""
        try:
            r = subprocess.run(
                ["git", "-C", self.repo_path, "checkout", "-b", name],
                capture_output=True, text=True, timeout=30)
            ok = r.returncode == 0
            LOGGER.info(f"Created & switched to branch {name}: {r.stderr.strip()[:120]}")
            return ok
        except Exception as e:
            LOGGER.error(f"create_branch failed: {e}")
            return False

    def checkout(self, branch: str) -> bool:
        try:
            r = subprocess.run(
                ["git", "-C", self.repo_path, "checkout", branch],
                capture_output=True, text=True, timeout=30)
            return r.returncode == 0
        except Exception:
            return False

    def merge_branch(self, branch: str) -> bool:
        """Merge `branch` back into main (no-ff merge commit), then push main."""
        try:
            if not self.checkout("main"):
                return False
            r = subprocess.run(
                ["git", "-C", self.repo_path, "merge", "--no-ff", branch,
                 "-m", f"merge {branch}: self-modification from cycle"],
                capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                LOGGER.error(f"merge {branch} into main failed: {r.stderr.strip()[:200]}")
                return False
            r2 = subprocess.run(
                ["git", "-C", self.repo_path, "push", "origin", "HEAD"],
                capture_output=True, text=True, timeout=120)
            if r2.returncode != 0:
                LOGGER.warning(f"merge push failed: {r2.stderr.strip()[:200]}")
            return True
        except Exception as e:
            LOGGER.error(f"merge_branch failed: {e}")
            return False

    def delete_branch(self, branch: str, remote: bool = True) -> bool:
        ok = True
        try:
            r = subprocess.run(
                ["git", "-C", self.repo_path, "branch", "-D", branch],
                capture_output=True, text=True, timeout=30)
            ok = r.returncode == 0
        except Exception:
            ok = False
        if remote:
            try:
                subprocess.run(
                    ["git", "-C", self.repo_path, "push", "origin", "--delete", branch],
                    capture_output=True, text=True, timeout=60)
            except Exception:
                pass
        return ok


# LLM model tiers for the self-tuning LLM agent.
# Default = Flash (cheap, fast); the agent may request an upgrade to Pro when a
# task needs deeper reasoning, and may command a downgrade back when it doesn't.
MODEL_FLASH = os.getenv("LLM_MODEL_FAST", "your-fast-model")
MODEL_PRO = os.getenv("LLM_MODEL_REASONER", "your-reasoning-model")

TIER_INSTRUCTION = (
    "CONTROL: You may request a model-tier change. If this task requires "
    "substantially deeper reasoning (complex debugging, large refactors, "
    "multi-file reasoning), put `[UPGRADE]` alone on a line. If you are on the "
    "pro tier and this task is simple, put `[DOWNGRADE]` alone on a line. "
    "Otherwise put `[KEEP]` alone on a line. Remove the marker from your actual answer."
)

# ── ROLLING CONTEXT WINDOW ──────────────────────────────────────────────────
# Default is a 75k-token rolling conversation window. Fundamentals (rules below)
# are always kept in context; older conversation turns are dropped once the
# window exceeds the budget. The agent may request a larger window via
# [MORE_CONTEXT] (up to MODEL_MAX_CONTEXT_TOKENS) or shrink back via [LESS_CONTEXT].
ROLLING_CONTEXT_TOKENS = int(os.getenv("LLM_CONTEXT_TOKENS", "75000"))
MODEL_MAX_CONTEXT_TOKENS = int(os.getenv("LLM_MAX_CONTEXT_TOKENS", "200000"))
CONTEXT_EXPAND_STEP = int(os.getenv("LLM_CONTEXT_STEP", "25000"))

CONTEXT_INSTRUCTION = (
    "CONTROL: You have a rolling context window (default 75,000 tokens). Older "
    "conversation is dropped automatically once the window is exceeded. If you "
    "need to recall more of the conversation, put `[MORE_CONTEXT]` alone on a "
    "line to request a larger window. If you want to shrink back toward the "
    "default, put `[LESS_CONTEXT]` alone on a line. Remove the marker from your "
    "actual answer."
)

# ── EFFORT LEVELS ────────────────────────────────────────────────────────────
# effort1 (default): leanest possible, zero fluff, direct technical answer.
# effort2: moderate depth, clearer and more detailed.
# effort3: maximum effort — exhaustive, edge cases, trade-offs, full detail.
EFFORT_INSTRUCTIONS = {
    1: "EFFORT LEVEL 1 (lean): Respond with the LEANEST possible output. Zero "
       "fluff, zero preamble, zero filler. If the answer fits in a number, one "
       "sentence, or a short list — output ONLY that. No introductions, no "
       "'here's a summary', no suggestions, no extra sections, no closing "
       "remarks. Just the direct technical answer, as concise as possible.",
    2: "EFFORT LEVEL 2 (balanced): Respond with a moderate level of detail. "
       "Clear, technically precise, reasonably thorough. Cover the essentials "
       "and key reasoning without excessive length.",
    3: "EFFORT LEVEL 3 (maximum): Respond with MAXIMUM thoroughness. Exhaustive "
       "technical analysis: cover edge cases, trade-offs, alternatives, "
       "implementation notes, and all relevant context. No artificial brevity.",
}
EFFORT_TEMPERATURE = {1: 0.2, 2: 0.5, 3: 0.8}
# effort1 must still be LEAN in prose, but 400 tokens is far too tight for
# "explore files with tools then answer" — the model would hit the cap before
# it can respond. 1500 keeps replies terse yet complete for tool-driven work.
EFFORT_MAX_TOKENS = {1: 1500, 2: 2000, 3: 8000}

# ── TELEGRAM FILE-ACCESS SAFETY ──────────────────────────────────────────────
# Which directories the bot may read/write, and what it may never touch.
# Secrets (.env, token files) and build/venv noise are always off-limits.
SECRET_GUARD_SUBSTRINGS = (
    ".env", "tokens_keys", "token", "secret", "credential",
    ".git", ".venv", "venv", "node_modules", "__pycache__",
)
# Directories excluded from any repo walk (fuzzy search, readme update, ...)
SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__",
             ".venv", "env", ".idea", "backups", "tokens_keys",
             "site-packages", "dist-packages", "archive", "legacy"}
MAX_VIEW_BYTES = 64 * 1024        # cap per-file reads fed to LLM/Telegram
MAX_GREP_LINES = 60               # cap /grep /find results
TOOL_RESULT_CAP = 8000            # chars returned per tool call (bounds context use)
PLAN_STEP_RE = re.compile(r'^### STEP\s+(\d+)/\d+:\s*(.*)$', re.IGNORECASE)


def parse_plan_steps(plan_path: Optional[str]) -> List[Dict]:
    """Parse a master plan file into [{num, title, body}] using `### STEP n/661:` headers."""
    if not plan_path or not os.path.exists(plan_path):
        return []
    steps: List[Dict] = []
    try:
        with open(plan_path, "r", errors="replace") as f:
            cur = None
            for line in f:
                m = PLAN_STEP_RE.match(line.rstrip())
                if m:
                    if cur:
                        steps.append(cur)
                    cur = {"num": int(m.group(1)), "title": m.group(2).strip(), "body": []}
                elif cur is not None:
                    cur["body"].append(line)
            if cur:
                steps.append(cur)
    except Exception as e:
        LOGGER.warning(f"Failed to parse plan {plan_path}: {e}")
    return steps


# Always-in-context fundamentals for every LLM call. Never trimmed away.
FUNDAMENTALS = (
    "You are the LLM agent of the Project quantitative trading system orchestrator. "
    "You answer questions and help control the automated pipeline.\n\n"
    "## HARD RULES (always in context)\n"
    "- A plan step is NEVER skipped. The pipeline only advances after a step is "
    "executed AND verified AND committed.\n"
    "- Strict escalation ladder per step: OmniRoute 5-agent team (5 attempts) -> "
    "fast tier (2 retries) -> reasoning tier (2 retries) -> HALT for "
    "human assistance.\n"
    "- Settings (settings.json, re-read before every phase): loop_mode=full|once, "
    "pause_after=none|audit|cross_eval|execution|cycle, on_step_failure=continue|halt, "
    "max_cycles=int, notify_interval=sec>=60, edit_mode=restricted|full.\n"
    "- Live orchestrator state is provided in the CONTEXT block; answer progress "
    "questions from it, never from memory. If the CONTEXT says ALL_STEPS_DONE: YES, "
    "the plan is COMPLETE — never report any specific step number (641, etc.) or "
    "claim work remains; say all 661 steps are done.\n"
    "- The orchestrator notifies on phase start, ~50% done, and finish for audit, "
    "cross-eval+plan, and execution.\n"
    "- You can call file tools (view_file, list_dir, grep_code, read_plan_step, "
    "edit_file, git_diff, run_status) to load context or complete tasks. Anything "
    "you load counts against your rolling context budget (default 75k tokens); "
    "fundamentals are always kept. Larger budgets can be granted via "
    "/expand_context.\n\n"
    + TIER_INSTRUCTION + "\n\n" + CONTEXT_INSTRUCTION
)


class LLMAgent:
    """Self-tuning LLM agent.

    Runs on the Flash tier by default. It may request an upgrade to the Pro
    (reasoner) tier for hard tasks, and may command a downgrade back to Flash
    when its intelligence is no longer needed. The current tier persists to a
    JSON state file so restarts keep the last choice.
    """

    def __init__(self, api_key: str, api_base: str = os.getenv("LLM_API_BASE", "https://api.your-provider.com/v1"),
                 state_file: str = None, history_file: str = None,
                 initial_model: str = MODEL_FLASH):
        self.api_key = api_key
        self.api_base = api_base
        self.model = initial_model
        self.model_tier = "pro" if self.model == MODEL_PRO else "flash"
        self.state_file = state_file
        self.history_file = history_file
        self.usage = {"flash": 0, "pro": 0}
        self.calls_since_switch = 0
        self.calls_since_context_switch = 0
        self.last_tier_event = None
        self.last_context_event = None
        self.history = []                 # rolling conversation [{role, content}]
        self.context_budget = ROLLING_CONTEXT_TOKENS
        self.effort_default = 1           # /effort1 = leanest, default
        self._tool_ctx = None             # TelegramBot instance for file-tool execution
        self._ctx_overhead = 0            # fundamentals+snapshot token overhead (for meter)
        self.last_usage = {}              # accumulated token usage for the last exchange
        self._load_state()
        self._load_history()

    # ── file tools (function-calling for the chat agent) ─────────────────────
    TOOLS = [
        {"type": "function", "function": {"name": "view_file", "description": "Read a file (or a line range of it) into context. Use line ranges for large files. Secrets (.env, tokens) are blocked.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path or repo-relative path"}, "start_line": {"type": "integer", "description": "First line (default 1)"}, "end_line": {"type": "integer", "description": "Last line (default EOF)"}}, "required": ["path"]}}},
        {"type": "function", "function": {"name": "list_dir", "description": "List a directory's contents.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory (default '.')"}}, "required": []}}},
        {"type": "function", "function": {"name": "grep_code", "description": "Grep the codebase for a regex pattern.",
          "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "description": "Optional directory to search"}}, "required": ["pattern"]}}},
        {"type": "function", "function": {"name": "read_plan_step", "description": "Read the full detail of a master-plan step.",
          "parameters": {"type": "object", "properties": {"step": {"type": "integer"}}, "required": ["step"]}}},
        {"type": "function", "function": {"name": "edit_file", "description": "Edit a file at a line/range. Respects edit_mode restrictions (restricted = only the 3 workflow scripts).",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}, "new_text": {"type": "string"}}, "required": ["path", "start_line", "end_line", "new_text"]}}},
        {"type": "function", "function": {"name": "git_diff", "description": "Show uncommitted git changes (stat).",
          "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "run_status", "description": "Show live orchestrator/pipeline status.",
          "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "run_command", "description": "Run a shell command in the project repo (cwd = project dir). General-purpose: run tests, inspect data, git log, check disk, etc. Output capped. Destructive system commands (rm -rf /, mkfs, sudo, shutdown) are blocked.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to run, e.g. 'python -m pytest tests/test_x.py -q' or 'git log --oneline -3'"}}, "required": ["command"]}}},
        {"type": "function", "function": {"name": "git_commit", "description": "Stage and commit all current changes in the repo.",
          "parameters": {"type": "object", "properties": {"message": {"type": "string"}}, "required": ["message"]}}},
    ]

    # Commands a general-purpose agent must never be able to run.
    _BLOCKED_CMD_PATTERNS = ("rm -rf /", "rm -rf ~", "mkfs", "dd if=/dev/zero",
                             "sudo ", "shutdown", "reboot", ":(){", "> /dev/sd", "git push --force")

    TOOL_INSTRUCTION = (
        "TOOLS: You can call tools (view_file, list_dir, grep_code, read_plan_step, "
        "edit_file, git_diff, run_status, run_command, git_commit) to load files, "
        "run commands, and make changes — and you DECIDE which to call yourself "
        "based on the user's request. You are a general-purpose agent: use "
        "run_command for anything the file tools can't do (run tests, inspect data, "
        "git operations, disk/process checks, etc.).\n"
        "HARD RULE — YOU ALWAYS HAVE CONTEXT: This agent is attached to the Project "
        "codebase and has working file tools. Whenever the user asks about the "
        "system/project/codebase — in ANY form, including metaphors, analogies, "
        "summaries, or creative requests — ground your answer in the real code "
        "(call tools; the codebase overview is always in context). Never respond "
        "'I don't have enough context' — that is always false; if you feel you "
        "lack context, call a tool.\n"
        "ANSWER ONLY WHAT IS ASKED: answer the user's exact question and nothing "
        "more. Do NOT add extra examples, do NOT chain a second metaphor after "
        "the one requested, and do NOT keep volunteering alternatives. One "
        "focused answer, then stop."
    )

    # If the agent replies with a "lack of context" refusal, that is always a
    # bug — it is attached to the codebase with file tools. We detect it and
    # force real context in + retry.
    _REFUSAL_RE = re.compile(
        r"(don'?t have (enough |any |the )?(context|info|information|knowledge|"
        r"background|details|access|idea)|"
        r"do not have (enough |any |the )?(context|info|information|knowledge|"
        r"background|details|access|idea)|"
        r"have (no|zero|limited) (context|info|information|knowledge|background|details|access)|"
        r"lack (enough |the )?(context|info|information|knowledge|background|details|access|idea)|"
        r"(can'?t|cannot|unable to) (access|find|locate|provide|make|create|generate|"
        r"build|explain|answer|help|do that|produce)|"
        r"not able to (access|provide|make|create|generate|build|explain|answer|help|do that)|"
        r"not (in|within) my context|"
        r"insufficient (context|info|information|knowledge|data|details)|"
        r"would need (more|additional|some) (context|info|information|details)|"
        r"need (more|additional) (context|info|information|details|data)|"
        r"don'?t know (what|anything|enough|much) (about|of|about the)|"
        r"i('m| am) (not sure|sorry|afraid|unable)|"
        r"could you (provide|give|tell|share|clarify|specify)|"
        r"please (provide|give|share|clarify|specify|elaborate)|"
        r"if you (could|could tell|provide|give) (me|us|more)|"
        r"more (context|info|information|details) (first|before|would help|is needed)|"
        r"clarif|ambiguous|unclear what)",
        re.IGNORECASE)

    def bind_tools(self, tool_ctx):
        """Bind the TelegramBot so the agent can execute file tools."""
        self._tool_ctx = tool_ctx

    async def _run_tool(self, name: str, args: Dict) -> str:
        """Execute a tool call; returns text the model sees next."""
        ctx = self._tool_ctx
        if ctx is None:
            return "ERROR: tools are not bound yet."
        try:
            if name == "view_file":
                path = ctx._resolve_path(args.get("path", ""))
                if not path or not os.path.isfile(path):
                    # fuzzy: the agent may pass a typo'd or short name
                    fuzzy, cands = ctx._resolve_fuzzy_file(args.get("path", ""))
                    if cands:
                        return f"Ambiguous — did you mean one of: {', '.join(c[:40] for c in cands)}? Please pass the full path."
                    if fuzzy and os.path.isfile(fuzzy):
                        path = fuzzy
                    else:
                        return "ERROR: file not found or not allowed."
                start = int(args.get("start_line", 1) or 1)
                end = int(args.get("end_line", 0) or 0) or None
                content, s, e, total = ctx._read_lines(path, start, end)
                return f"FILE {path} (lines {s}-{e} of {total}):\n{content[:TOOL_RESULT_CAP]}"
            if name == "list_dir":
                path = ctx._resolve_path(args.get("path", "."))
                if not path or not os.path.isdir(path):
                    return "ERROR: directory not allowed."
                entries = sorted(os.listdir(path))
                return "\n".join(entries[:120]) or "(empty)"
            if name == "grep_code":
                pat = args.get("pattern", "")
                roots = [ctx._resolve_path(args["path"])] if args.get("path") else None
                out = ctx._grep(pat, roots)
                return out[:TOOL_RESULT_CAP] if not out.startswith("(no matches") else out
            if name == "read_plan_step":
                n = int(args.get("step", 0))
                steps = parse_plan_steps(ctx._plan_path())
                st = next((s for s in steps if s["num"] == n), None)
                if not st:
                    return f"ERROR: step {n} not found in plan."
                status = ctx._step_status(n, ctx._exec_state())
                return f"STEP {n}/{len(steps)}: {st['title']}\nStatus: {status}\n{''.join(st['body'])[:TOOL_RESULT_CAP]}"
            if name == "edit_file":
                path = ctx._resolve_path(args.get("path", ""))
                if not path or not os.path.isfile(path):
                    return "ERROR: file not found or not allowed."
                mode = ctx.state.state.get('edit_mode', 'restricted')
                if mode == 'restricted' and not any(a in path for a in
                        ('parallel_agents.py', 'parallel_agent_cross_eval.py', 'execute_master_plan.py')):
                    return "ERROR: restricted edit_mode — only the 3 workflow scripts are editable."
                if 'run_workflow.py' in path or 'config' in path:
                    return "ERROR: orchestrator/config files are not editable."
                start = int(args.get("start_line", 1) or 1)
                end = int(args.get("end_line", start) or start)
                new_text = args.get("new_text", "")
                with open(path, 'r', errors='replace') as f:
                    lines = f.readlines()
                total = len(lines)
                if not (1 <= start <= end <= total):
                    return f"ERROR: line range {start}-{end} outside 1-{total}."
                lines[start - 1:end] = [ln.rstrip('\n') + '\n' for ln in new_text.split('\n')]
                err = ctx._validate_syntax(path, lines)
                if err:
                    return f"ERROR: validation failed, edit NOT applied:\n{err[:600]}"
                tmp = path + ".tg_tool.tmp"
                with open(tmp, 'w') as f:
                    f.writelines(lines)
                os.replace(tmp, path)
                return f"OK: edited {path} lines {start}-{end}. (uncommitted — run git commit or /save)"
            if name == "git_diff":
                repo = ctx.config.get('git_repo_root', ctx.config.get('repo_dir'))
                r = subprocess.run(["git", "-C", repo, "diff", "--stat"], capture_output=True, text=True, timeout=30)
                return (r.stdout or "(clean working tree)")[:4000]
            if name == "run_command":
                cmd = (args.get("command") or "").strip()
                if not cmd:
                    return "ERROR: empty command."
                low = cmd.lower()
                if any(b in low for b in self._BLOCKED_CMD_PATTERNS):
                    return "ERROR: command blocked (destructive/system command)."
                import shlex as _shlex
                try:
                    argv = _shlex.split(cmd)
                except Exception as e:
                    return f"ERROR: could not parse command: {e}"
                if not argv:
                    return "ERROR: empty command."
                repo = ctx.config.get('git_repo_root', ctx.config.get('repo_dir'))
                proc = await asyncio.create_subprocess_exec(
                    *argv, cwd=repo,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                try:
                    out, _ = await asyncio.wait_for(proc.communicate(), timeout=60)
                except asyncio.TimeoutError:
                    proc.kill()
                    return "ERROR: command timed out after 60s."
                text = (out or b"").decode("utf-8", errors="replace").strip()
                return f"$ {cmd}\nrc={proc.returncode}\n{text[-TOOL_RESULT_CAP:] or '(no output)'}"
            if name == "git_commit":
                repo = ctx.config.get('git_repo_root', ctx.config.get('repo_dir'))
                msg = (args.get("message") or "commit via telegram").strip()
                r1 = subprocess.run(["git", "-C", repo, "add", "-A"], capture_output=True, text=True, timeout=30)
                r2 = subprocess.run(["git", "-C", repo, "commit", "-m", msg],
                                    capture_output=True, text=True, timeout=30)
                if r2.returncode != 0:
                    return f"Commit failed: {(r2.stderr or r2.stdout)[:400]}"
                r3 = subprocess.run(["git", "-C", repo, "push", "origin", "HEAD"],
                                    capture_output=True, text=True, timeout=120)
                return f"Committed: {msg}\n{(r3.stdout or '')[-300:]}"
            if name == "run_status":
                return ctx._build_context()
        except Exception as e:
            return f"ERROR: {e}"
        return f"ERROR: unknown tool '{name}'"

    # ── persistence ────────────────────────────────────────────────────────
    def _load_state(self):
        if not self.state_file or not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file) as f:
                data = json.load(f)
            model = data.get("model", self.model)
            self.model = model if model in (MODEL_FLASH, MODEL_PRO) else self.model
            self.model_tier = "pro" if self.model == MODEL_PRO else "flash"
            self.usage = data.get("usage", self.usage)
            self.context_budget = int(data.get("context_budget", self.context_budget))
            self.context_budget = max(ROLLING_CONTEXT_TOKENS, min(self.context_budget, MODEL_MAX_CONTEXT_TOKENS))
            try:
                self.effort_default = int(data.get("effort_default", self.effort_default))
            except (TypeError, ValueError):
                pass
            if self.effort_default not in (1, 2, 3):
                self.effort_default = 1
        except Exception as e:
            LOGGER.warning(f"Could not load LLM agent state: {e}")

    def _save_state(self):
        if not self.state_file:
            return
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            tmp = self.state_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"model": self.model, "usage": self.usage,
                           "context_budget": self.context_budget,
                           "effort_default": self.effort_default}, f, indent=2)
            os.replace(tmp, self.state_file)
        except Exception as e:
            LOGGER.warning(f"Could not save LLM agent state: {e}")

    def set_effort_default(self, level: int) -> str:
        """Set the default effort level for future messages (persisted)."""
        level = int(level)
        if level not in (1, 2, 3):
            return "❌ Effort must be 1, 2, or 3."
        self.effort_default = level
        self._save_state()
        return f"✅ Default effort set to **{level}** ({'lean' if level == 1 else 'balanced' if level == 2 else 'maximum'})."

    def inject_context(self, label: str, content: str) -> str:
        """Force-load text into the rolling conversation (used by /load, /explain)."""
        if not content:
            return "Empty content; nothing injected."
        self.history.append({"role": "user", "content": f"[INJECTED CONTEXT: {label}]\n{content}"})
        self.history = self._trimmed_history()
        self._save_history()
        return f"Injected ~{self._estimate_tokens(content)} tokens of `{label}` into context."

    def clear_context(self) -> str:
        """Reset the rolling conversation window entirely."""
        n = len(self.history)
        self.history = []
        self._save_history()
        return f"Cleared {n} conversation turns."

    # ── rolling conversation memory ────────────────────────────────────────
    def _load_history(self):
        if not self.history_file or not os.path.exists(self.history_file):
            return
        try:
            with open(self.history_file) as f:
                data = json.load(f)
            self.history = data if isinstance(data, list) else []
        except Exception as e:
            LOGGER.warning(f"Could not load LLM conversation history: {e}")

    def _save_history(self):
        if not self.history_file:
            return
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            tmp = self.history_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.history[-200:], f, indent=2)
            os.replace(tmp, self.history_file)
        except Exception as e:
            LOGGER.warning(f"Could not save LLM conversation history: {e}")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate (~3 chars per token for code/JSON-heavy text)."""
        if not text:
            return 0
        return max(1, len(text) // 3)

    @classmethod
    def _history_tokens(cls, messages: List[Dict]) -> int:
        return sum(cls._estimate_tokens(m.get("content", "")) for m in messages)

    def _trimmed_history(self, reserve_tokens: int = 0) -> List[Dict]:
        """Keep conversation turns within the rolling budget (default 75k).

        Fundamentals are always retained; only the oldest conversation turns are
        dropped. Always keeps the most recent turn. Tool-call sequences are kept
        intact (a tool result without its assistant tool_calls is dropped).
        """
        budget = self.context_budget - self._estimate_tokens(FUNDAMENTALS) - reserve_tokens
        kept = list(self.history)
        while len(kept) > 1 and self._history_tokens(kept) > budget:
            kept.pop(0)  # drop oldest turn
        return self._sanitize_messages(kept)

    @staticmethod
    def _sanitize_messages(messages: List[Dict]) -> List[Dict]:
        """Guarantee the message list is API-valid.

        Every assistant message with `tool_calls` must be immediately followed
        by a `tool` message for EACH tool_call_id it declared — otherwise the
        API rejects the whole request with a 400. Trimming or a mid-sequence
        error can leave a tool_calls message with only some (or none) of its
        results, so we drop any incomplete tool_calls block entirely.
        """
        out = []
        i = 0
        n = len(messages)
        while i < n:
            m = messages[i]
            role = m.get("role")
            if role == "assistant" and m.get("tool_calls"):
                ids = {tc.get("id") for tc in m["tool_calls"] if tc.get("id")}
                # gather the immediately-following tool responses
                j = i + 1
                tool_msgs = []
                while j < n and messages[j].get("role") == "tool":
                    tool_msgs.append(messages[j])
                    j += 1
                got = {tm.get("tool_call_id") for tm in tool_msgs}
                if ids and ids <= got:
                    # complete — keep the assistant + its tool responses
                    out.append(m)
                    out.extend(tool_msgs)
                    i = j
                    continue
                # incomplete (or no ids) — drop the whole block
                i = j
                continue
            if role == "tool":
                # orphaned tool result (its assistant tool_calls was dropped)
                i += 1
                continue
            out.append(m)
            i += 1
        return out

    # ── tier switching ─────────────────────────────────────────────────────
    def _maybe_tier_switch(self, text: str) -> str:
        """Honor [UPGRADE]/[DOWNGRADE] markers with a short anti-thrash cooldown."""
        stripped = re.sub(r'\s*\[(?:UPGRADE|DOWNGRADE|KEEP)\]\s*', '', text).strip()
        request = "keep"
        if "[UPGRADE]" in text:
            request = "upgrade"
        elif "[DOWNGRADE]" in text:
            request = "downgrade"

        self.usage["pro" if self.model_tier == "pro" else "flash"] += 1
        self.calls_since_switch += 1

        if request == "upgrade" and self.model_tier == "flash" and self.calls_since_switch >= 2:
            self.model = MODEL_PRO
            self.model_tier = "pro"
            self.calls_since_switch = 0
            self.last_tier_event = f"🔺 Upgraded LLM agent to reasoning tier (`{MODEL_PRO}`) — flash requested deeper reasoning."
            LOGGER.info(self.last_tier_event)
            self._save_state()
        elif request == "downgrade" and self.model_tier == "pro" and self.calls_since_switch >= 2:
            self.model = MODEL_FLASH
            self.model_tier = "flash"
            self.calls_since_switch = 0
            self.last_tier_event = f"🔻 Downgraded LLM agent to fast tier (`{MODEL_FLASH}`) — pro reasoning no longer needed."
            LOGGER.info(self.last_tier_event)
            self._save_state()
        else:
            self.last_tier_event = None
        return stripped

    def _maybe_context_switch(self, text: str) -> str:
        """Honor [MORE_CONTEXT]/[LESS_CONTEXT] markers with a cooldown."""
        stripped = re.sub(r'\s*\[(?:MORE_CONTEXT|LESS_CONTEXT)\]\s*', '', text).strip()
        request = None
        if "[MORE_CONTEXT]" in text:
            request = "more"
        elif "[LESS_CONTEXT]" in text:
            request = "less"

        self.calls_since_context_switch += 1

        if request == "more" and self.context_budget < MODEL_MAX_CONTEXT_TOKENS and self.calls_since_context_switch >= 2:
            new_budget = min(MODEL_MAX_CONTEXT_TOKENS, self.context_budget + CONTEXT_EXPAND_STEP)
            self.context_budget = new_budget
            self.calls_since_context_switch = 0
            self.last_context_event = f"🧠 Context window expanded to {new_budget:,} tokens (agent requested more context)."
            LOGGER.info(self.last_context_event)
            self._save_state()
        elif request == "less" and self.context_budget > ROLLING_CONTEXT_TOKENS and self.calls_since_context_switch >= 2:
            self.context_budget = ROLLING_CONTEXT_TOKENS
            self.calls_since_context_switch = 0
            self.last_context_event = f"🧠 Context window reset to {ROLLING_CONTEXT_TOKENS:,} tokens."
            LOGGER.info(self.last_context_event)
            self._save_state()
        else:
            if request:
                self.last_context_event = None  # throttled
        return stripped

    def _postprocess(self, text: str) -> str:
        """Strip control markers and apply any requested tier/context switch."""
        if not text:
            return text
        text = self._maybe_tier_switch(text)
        return self._maybe_context_switch(text)

    # ── token usage / cost reporting ────────────────────────────────────────
    # LLM pricing per 1M tokens. Cache hits are ~50x cheaper (hybrid /
    # sparse attention), which is why we always keep fundamentals + snapshot
    # in the prompt — a warm cache is the cheapest path.
    # Prices per 1M tokens for COST REPORTING only (no functional effect).
    # Update these for your provider, keyed by the model names from
    # LLM_MODEL_FAST / LLM_MODEL_REASONER. Cache hits are typically ~50x
    # cheaper — a warm prompt cache is the cheapest path.
    PRICING = {
        "your-fast-model": {"in_hit": 0.0, "in_miss": 0.0, "out": 0.0},
        "your-reasoning-model": {"in_hit": 0.0, "in_miss": 0.0, "out": 0.0},
    }

    def _accumulate_usage(self, usage: Dict):
        """Accumulate token counts across a multi-call exchange (tool loop, recovery)."""
        if not isinstance(usage, dict):
            return
        for k in ("prompt_tokens", "completion_tokens",
                  "prompt_cache_hit_tokens", "prompt_cache_miss_tokens"):
            self.last_usage[k] = self.last_usage.get(k, 0) + int(usage.get(k, 0) or 0)

    def usage_summary(self) -> str:
        """One-line token + cache-rate + estimated-cost summary for the last exchange."""
        u = self.last_usage or {}
        if not u or not u.get("prompt_tokens"):
            return ""
        p_in = u.get("prompt_tokens", 0)
        p_out = u.get("completion_tokens", 0)
        hit = u.get("prompt_cache_hit_tokens", 0)
        miss = u.get("prompt_cache_miss_tokens", max(0, p_in - hit))
        denom = hit + miss
        rate = (100.0 * hit / denom) if denom else 0.0
        price = self.PRICING.get(self.model, self.PRICING[MODEL_FLASH])
        cost = (hit / 1e6) * price["in_hit"] + (miss / 1e6) * price["in_miss"] \
               + (p_out / 1e6) * price["out"]
        return (f"⚡ {p_in:,} in / {p_out:,} out · cache {rate:.0f}% "
                f"· ≈${cost:.5f} ({self.model})")

    async def generate_readme_entries(self, files: List[Dict]) -> str:
        """Generate '#N. /path  Purpose/Mechanism/I-O' readme entries for repo files.

        files: [{"path": repo-relative, "snippet": content excerpt}]
        Returns the markdown entries (with sequential numbers) or '' on failure.
        """
        if not files:
            return ""
        listing = "\n".join(
            f"### {f['path']}\n{f['snippet']}" for f in files)
        prompt = (
            "You document the Project quantitative trading codebase. For EACH file below "
            "produce exactly one numbered entry:\n"
            "#N. {REPO_DIR}/<relative path>\n"
            "Purpose: <one line — what the module does>\n"
            "Mechanism: <1-2 lines — how it works>\n"
            "I/O: <one line — inputs -> outputs>\n\n"
            f"FILES:\n{listing}\n\n"
            "Return ONLY the numbered entries in the same order. No preamble, no code fences."
        )
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": MODEL_FLASH,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 3000,
                }
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers, json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        return ""
                    data = await resp.json()
                    content = self._postprocess(data["choices"][0]["message"].get("content") or "")
        except Exception as e:
            LOGGER.error(f"generate_readme_entries failed: {e}")
            return ""
        # renumber sequentially starting at 1 (the caller offsets to continue)
        lines = content.splitlines()
        num = 0
        out = []
        for ln in lines:
            m = re.match(r'^#\d+\.\s+(.*)$', ln)
            if m:
                num += 1
                out.append(f"#{num}. {m.group(1)}")
            else:
                out.append(ln)
        return "\n".join(out).strip()

    async def analyze_cycle(
        self,
        audit_file: str,
        plan_file: str,
        execution_logs: str,
        tasks: List[Dict],
        prev_cycle_summary: Optional[str] = None
    ) -> Tuple[str, List[Dict]]:
        """
        Analyze the completed cycle and suggest script changes.
        Returns: (analysis_summary, list of suggested edits)
        """
        task_context = "\n".join([
            f"- [{t['status'].upper()}] {t['id']}: {t['description']}"
            for t in tasks
        ])
        
        context = f"""
You are the self-adaptive agent for the Project quantitative trading system orchestrator.

## Current Task Board
{task_context}

## Current Cycle State

**Audit File**: {audit_file}
**Plan File**: {plan_file}
**Execution Logs**: {execution_logs}

**Previous Cycle Summary**: {prev_cycle_summary or "None (first cycle)"}

## Your Task
1. Analyze the audit findings and execution results.
2. Review the task board and identify which tasks are relevant.
3. Compare with the previous cycle (if any) to detect meaningful state changes.
4. If changes are detected, propose specific edits to the 3 workflow scripts:
   - parallel_agents.py (audit)
   - parallel_agent_cross_eval.py (cross-eval + plan)
   - execute_master_plan.py (execution)
5. Only edit if necessary (don't over-optimize).
6. Suggest which tasks should be marked 'done' based on your analysis.
7. Explain your reasoning for each edit.

## Output Format
Return ONLY valid JSON with this structure:
{{
  "analysis": "Brief summary of findings (2-3 sentences)",
  "needs_edits": true/false,
  "edits": [
    {{
      "file": "parallel_agents.py",
      "description": "Why this edit is needed",
      "old_code": "exact code block to find",
      "new_code": "replacement code block"
    }}
  ],
  "tasks_to_mark_done": ["T001", "T003"],
  "summary": "Overall cycle summary for next iteration"
}}

If no edits are needed, set edits to empty list and needs_edits to false.
"""

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": FUNDAMENTALS},
                                 {"role": "user", "content": context}],
                    "temperature": 0.5,
                }
                
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        LOGGER.error(f"LLM API error: {resp.status} {error_text}")
                        return "Error calling LLM API", []
                    
                    data = await resp.json()
                    content = self._postprocess(data["choices"][0]["message"]["content"])

                    try:
                        result = json.loads(content)
                        return result.get("analysis", ""), result.get("edits", [])
                    except json.JSONDecodeError:
                        LOGGER.error(f"LLM returned invalid JSON: {content}")
                        return "Failed to parse LLM response", []
        
        except asyncio.TimeoutError:
            LOGGER.error("LLM analysis timed out")
            return "LLM analysis timed out", []
        except Exception as e:
            LOGGER.error(f"LLM analysis failed: {e}")
            return f"LLM error: {str(e)}", []
    
    async def execute_task(self, task: Dict) -> str:
        """
        Execute a specific task via LLM.
        Returns the result as a string.
        """
        prompt = f"""
You are an agent for the Project system. Execute this task:

**Task ID**: {task['id']}
**Description**: {task['description']}
**Condition**: {task['condition']}
**Action**: {task['action']}

Analyze the condition and execute the action. Provide a detailed response about:
1. Whether the condition is currently true or false
2. What the action accomplished
3. Any issues or observations
4. Recommendations for next steps

Be concise and factual.
"""

        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model,
                    "messages": [{"role": "system", "content": FUNDAMENTALS},
                                 {"role": "user", "content": prompt}],
                    "temperature": 0.7,
                }

                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return f"Error: {resp.status} {error_text}"

                    data = await resp.json()
                    return self._postprocess(data["choices"][0]["message"]["content"])

        except asyncio.TimeoutError:
            return "Task execution timed out (>60s)"
        except Exception as e:
            return f"Error: {str(e)}"

    async def direct_query(self, user_query: str, context_snapshot: str = None,
                           effort: int = None) -> str:
        """
        Handle direct user queries through Telegram using a rolling conversation
        window. Fundamentals (rules) are always in context; the live orchestrator
        snapshot is refreshed each call; older conversation turns are trimmed
        once the window exceeds the current budget (default 75k tokens).

        `effort` selects the answer verbosity: 1=lean (default), 2=balanced,
        3=maximum. If None, uses the persisted default.
        """
        if effort is None:
            effort = self.effort_default
        if effort not in EFFORT_INSTRUCTIONS:
            effort = 1

        # fresh usage tracking for THIS exchange
        self.last_usage = {}

        # Append this turn to the conversation, then trim to the window.
        self.history.append({"role": "user", "content": user_query})

        system = (FUNDAMENTALS
                  + f"\n\n{EFFORT_INSTRUCTIONS[effort]}"
                  + f"\n\n{self.TOOL_INSTRUCTION}")
        if context_snapshot:
            system = f"{system}\n\n=== CONTEXT (fresh) ===\n{context_snapshot}\n=== END CONTEXT ==="

        snapshot_tokens = self._estimate_tokens(context_snapshot or "")
        self._ctx_overhead = self._estimate_tokens(FUNDAMENTALS) + snapshot_tokens

        # Rolling history trimmed to the current budget. Fundamentals are always
        # prepended as system and never trimmed. Tool loads and conversation all
        # share the same rolling window.
        messages = [{"role": "system", "content": system}] + \
                   self._trimmed_history(reserve_tokens=snapshot_tokens)

        # PROACTIVE GROUNDING: when the user asks a creative/grounding question
        # about the codebase (analogy, metaphor, summary, explain), put the real
        # codebase overview right in front of the question so the model has hard
        # material and never has a reason to claim it lacks context.
        _CREATIVE_RE = re.compile(
            r"\b(analogy|metaphor|compare|explain like|eli5|summarize|overview|"
            r"describe|creative|story|pretend|as (a|an)|vibe|personality|theme|mood)\b",
            re.IGNORECASE)
        _CODE_REF_RE = re.compile(
            r"\b(project|codebase|code base|system|project|my code|my repo|"
            r"the repo|what we built|your code)\b",
            re.IGNORECASE)
        if (self._tool_ctx is not None
                and _CREATIVE_RE.search(user_query)
                and _CODE_REF_RE.search(user_query)):
            snap = self._tool_ctx._build_codebase_snapshot()
            if snap:
                messages.append({
                    "role": "user",
                    "content": (
                        "GROUNDING — the real Project codebase is below. Use it to "
                        "answer the user's creative request that follows. Never "
                        "say you lack context — it is all here.\n\n" + snap
                    ),
                })
        base_len = len(messages)   # index where THIS call's new turns start

        session_data = None
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }

                # Tool-capable chat model. your-reasoning-model (Pro tier) does NOT
                # support function calling, so the self-directing chat agent always
                # runs on your-fast-model — otherwise tools silently vanish and the
                # agent answers codebase questions from memory alone. The tier
                # system still drives the pipeline's escalation; for interactive
                # chat, working tools beat a reasoning model that can't call them.
                chat_model = MODEL_FLASH

                def _payload(with_tools: bool):
                    payload = {
                        "model": chat_model,
                        "messages": messages,
                        "temperature": EFFORT_TEMPERATURE[effort],
                        "max_tokens": EFFORT_MAX_TOKENS[effort],
                    }
                    if with_tools:
                        payload["tools"] = self.TOOLS
                        payload["tool_choice"] = "auto"
                    return payload

                reply = None
                # Tool-calling loop: let the model inspect files / plan / state.
                # Each tool round consumes one iteration; the final answer needs
                # one more, so the cap must be comfortably above typical rounds
                # (a "best module" exploration is ~4-6 tool rounds + answer).
                for _ in range(14):
                    resp = await session.post(
                        f"{self.api_base}/chat/completions",
                        headers=headers,
                        json=_payload(True),
                        timeout=aiohttp.ClientTimeout(total=90)
                    )
                    if resp.status == 400:
                        # your-reasoning-model (Pro tier) does not support tools — retry without.
                        resp = await session.post(
                            f"{self.api_base}/chat/completions",
                            headers=headers,
                            json=_payload(False),
                            timeout=aiohttp.ClientTimeout(total=90)
                        )
                    if resp.status != 200:
                        error_text = await resp.text()
                        session_data = f"Error: {resp.status} {error_text}"
                        break

                    data = await resp.json()
                    self._accumulate_usage(data.get("usage") or {})
                    msg = data["choices"][0]["message"]
                    tool_calls = msg.get("tool_calls")

                    if tool_calls:
                        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            try:
                                args = json.loads(fn.get("arguments") or "{}")
                            except json.JSONDecodeError:
                                args = {}
                            result = await self._run_tool(fn.get("name", ""), args)
                            messages.append({"role": "tool", "tool_call_id": tc.get("id"), "content": result})
                        continue

                    reply = self._postprocess(msg.get("content") or "")
                    break

                if reply is None:
                    reply = session_data or "No response from model."

                # REFUSAL RECOVERY: if the model claims it lacks context (it
                # never does — it's attached to the codebase), strip its refusal
                # turns, force the REAL codebase overview in, and retry.
                recoveries = 0
                while reply and self._REFUSAL_RE.search(reply) and recoveries < 2:
                    recoveries += 1
                    snapshot = self._tool_ctx._build_codebase_snapshot() if self._tool_ctx else ""
                    if not snapshot:
                        break
                    # drop the trailing refusal/assistant turns so the model
                    # isn't anchored to its own "I lack context" replies
                    while messages and messages[-1].get("role") in ("assistant", "tool"):
                        messages.pop()
                    messages.append({
                        "role": "user",
                        "content": (
                            "FACT: you ARE attached to the Project codebase, and the "
                            "real overview is below. The user's question was: \""
                            f"{user_query}\"\n\n{snapshot}\n\n"
                            "Answer the user's question grounded in the code above. "
                            "Do NOT refuse, and do NOT say you lack context."
                        ),
                    })
                    resp = await session.post(
                        f"{self.api_base}/chat/completions",
                        headers=headers,
                        json=_payload(False),
                        timeout=aiohttp.ClientTimeout(total=90)
                    )
                    if resp.status != 200:
                        break
                    data = await resp.json()
                    self._accumulate_usage(data.get("usage") or {})
                    reply = self._postprocess(data["choices"][0]["message"].get("content") or "")
        except asyncio.TimeoutError:
            return "Query timed out (>90s)"
        except Exception as e:
            return f"Error: {str(e)}"

        # Persist only THIS call's new turns (tool loads + reply) into the
        # rolling window — the trimmed history already lives in self.history.
        new_turns = messages[base_len:]
        # Keep assistant tool_calls + tool results so replaying history is valid;
        # drop the intermediate assistant turn when it had neither content nor calls.
        self.history.extend([
            m for m in new_turns
            if m["role"] in ("assistant", "tool")
            and (m.get("content") or m.get("tool_calls"))
        ])
        self.history.append({"role": "assistant", "content": reply})
        self.history = self._trimmed_history()
        self._save_history()
        return reply

    async def parse_settings_change(self, user_message: str) -> Dict:
        """Ask the LLM to turn a natural-language settings request into a JSON patch."""
        prompt = f"""You parse settings-change requests for the Project orchestrator.

Valid settings keys and allowed values:
- loop_mode: "full" (loop forever) | "once" (run one cycle then stop)
- pause_after: "none" | "audit" | "cross_eval" | "execution" | "cycle"
  (pause the workflow AFTER that phase completes)
- on_step_failure: "continue" | "halt"  (what to do when a plan step can't be fixed)
- max_cycles: integer (0 = unlimited)
- notify_interval: integer seconds (>= 60) between Telegram progress pings
- edit_mode: "restricted" | "full"
- notifications: "on" | "off"  (whether to send phase-start/50%/finish pings)

User message: "{user_message}"

Return ONLY a JSON object with the keys the user wants to change and their new
values. If the user did not intend a settings change, return {{}}.

Examples:
- "from now on finish execution then pause" -> {{"pause_after": "execution"}}
- "stop looping after one run" -> {{"loop_mode": "once"}}
- "keep going even if steps fail" -> {{"on_step_failure": "continue"}}
- "what is 2+2?" -> {{}}

{TIER_INSTRUCTION}
"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                }
                async with session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status != 200:
                        return {}
                    data = await resp.json()
                    content = self._postprocess(data["choices"][0]["message"]["content"])
                    try:
                        parsed = json.loads(content)
                        return parsed if isinstance(parsed, dict) else {}
                    except json.JSONDecodeError:
                        # Strip code fences if the model wrapped the JSON
                        import re as _re
                        m = _re.search(r'\{.*\}', content, _re.DOTALL)
                        if m:
                            try:
                                return json.loads(m.group(0))
                            except Exception:
                                pass
                        return {}
        except asyncio.TimeoutError:
            return {}
        except Exception as e:
            LOGGER.error(f"parse_settings_change failed: {e}")
            return {}


class ScriptEditor:
    """Safely edit workflow scripts with git rollback support."""
    
    def __init__(self, git_manager: GitManager):
        self.git = git_manager
    
    def apply_edit(self, file_path: str, old_code: str, new_code: str) -> bool:
        """
        Apply a code edit to a file.
        Returns True if successful, False otherwise.
        """
        if not os.path.exists(file_path):
            LOGGER.error(f"File not found: {file_path}")
            return False
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            if old_code not in content:
                LOGGER.error(f"Old code block not found in {file_path}")
                return False
            
            new_content = content.replace(old_code, new_code, 1)
            
            with open(file_path, 'w') as f:
                f.write(new_content)
            
            LOGGER.info(f"Edited {file_path}")
            return True
        
        except Exception as e:
            LOGGER.error(f"Failed to edit {file_path}: {e}")
            return False
    
    def validate_edit(self, file_path: str) -> bool:
        """Validate that a Python file has valid syntax."""
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
            LOGGER.info(f"Syntax validation passed: {file_path}")
            return True
        except SyntaxError as e:
            LOGGER.error(f"Syntax error in {file_path}: {e}")
            return False
        except Exception as e:
            LOGGER.error(f"Validation failed for {file_path}: {e}")
            return False


class WorkflowExecutor:
    """Execute workflow scripts (audit, cross-eval, execution phases)."""
    
    def __init__(self, config: Dict, state_manager: StateManager):
        self.config = config
        self.state = state_manager
        self.logger = LOGGER
    
    async def run_audit(self, settings: Dict = None) -> Tuple[bool, str]:
        """Run audit script (parallel_agents.py)."""
        self.state.update(phase="audit", phase_start_time=datetime.now().isoformat())

        script_path = self.config['scripts']['audit']
        self.logger.info(f"Starting audit phase: {script_path}")

        # Run a FRESH audit by default: --resume would skip passes that already
        # have cached summary.json, "completing" in seconds with no new AI work.
        # Set settings fresh_audit=off to allow cached-resume.
        extra = [] if (settings or {}).get('fresh_audit', 'on') != 'off' else ["--resume"]
        try:
            result = await self._run_script(script_path, timeout=14400,
                                            env_extra=self._base_env(settings),
                                            extra_args=extra)
            
            if result["returncode"] == 0:
                audit_file = self._find_latest_file(
                    self.config['work_dir'],
                    "multi_agent_audit_*.md"
                )
                
                if audit_file:
                    self.state.update(last_audit_file=audit_file)
                    self.logger.info(f"Audit complete: {audit_file}")
                    return True, audit_file
                else:
                    self.logger.warning("Audit completed but no output file found")
                    return True, "unknown"
            else:
                self.logger.error(f"Audit failed: {result['stderr']}")
                return False, result['stderr']
        
        except asyncio.TimeoutError:
            self.logger.error("Audit script timed out (4 hours)")
            return False, "Timeout"
        except Exception as e:
            self.logger.error(f"Audit execution error: {e}")
            return False, str(e)
    
    def _base_env(self, settings: Dict = None) -> Dict:
        """Environment variables passed to every workflow script run."""
        settings = settings or {}
        env = {
            "REPO_DIR": self.config.get('repo_dir', ''),
            "WORK_DIR": self.config.get('work_dir', ''),
            "LLM_API_KEY": self.config.get('llm_api_key', ''),
            "ON_STEP_FAILURE": settings.get('on_step_failure', 'continue'),
        }
        # Allow overrides already present in the parent environment to win.
        for k in list(env):
            if os.environ.get(k):
                env[k] = os.environ[k]
        return env

    async def run_cross_eval(self, settings: Dict = None) -> Tuple[bool, str]:
        """Run cross-eval and plan generation script."""
        self.state.update(phase="cross_eval", phase_start_time=datetime.now().isoformat())

        script_path = self.config['scripts']['cross_eval']
        self.logger.info(f"Starting cross-eval phase: {script_path}")

        try:
            result = await self._run_script(script_path, timeout=14400,
                                            env_extra=self._base_env(settings),
                                            extra_args=["--resume"])
            
            if result["returncode"] == 0:
                plan_file = self._find_latest_file(
                    self.config['work_dir'],
                    "master_plan_*.md"
                )
                
                if plan_file:
                    self.state.update(last_plan_file=plan_file)
                    self.logger.info(f"Cross-eval complete: {plan_file}")
                    return True, plan_file
                else:
                    self.logger.warning("Cross-eval completed but no plan file found")
                    return True, "unknown"
            else:
                self.logger.error(f"Cross-eval failed: {result['stderr']}")
                return False, result['stderr']
        
        except asyncio.TimeoutError:
            self.logger.error("Cross-eval script timed out (1 hour)")
            return False, "Timeout"
        except Exception as e:
            self.logger.error(f"Cross-eval execution error: {e}")
            return False, str(e)
    
    async def run_execution(self, settings: Dict = None) -> Tuple[bool, str]:
        """Run execution script."""
        self.state.update(phase="execute", phase_start_time=datetime.now().isoformat())

        script_path = self.config['scripts']['execution']
        self.logger.info(f"Starting execution phase: {script_path}")

        exec_args = ["--resume"]
        if (settings or {}).get('on_step_failure', 'continue') == 'continue':
            exec_args.append("--continue-on-failure")

        # 2026-08-07: pass the CURRENT plan explicitly. The executor's default
        # MASTER_PLAN_FILE (master_plan_8_3.md) is stale — without this,
        # an unpaused cycle executes the WRONG plan (observed: engine ran
        # "STEP 21/458" of an old plan while the 1545-step 8_7 plan sat unused).
        env_extra = self._base_env(settings)
        plan_path = self._find_latest_file(
            self.config['work_dir'], "master_plan_*.md")
        if plan_path:
            env_extra = dict(env_extra)
            env_extra["MASTER_PLAN_FILE"] = plan_path
            self.logger.info(f"Execution engine will run plan: {plan_path}")

        # Attach guard: if an engine is ALREADY running (e.g. this orchestrator
        # was restarted mid-execution), do NOT spawn a duplicate engine — wait
        # for the live one to finish instead. Makes orchestrator restarts safe
        # at any step without losing the in-flight step.
        running_pid = await self._engine_pid()
        if running_pid:
            self.logger.info(f"Execution engine already running (pid {running_pid}); attaching.")
            return await self._attach_to_engine(running_pid)

        try:
            result = await self._run_script(script_path, timeout=21600,
                                            env_extra=env_extra,
                                            extra_args=exec_args)
            
            if result["returncode"] == 0:
                self.logger.info("Execution phase complete")
                self.state.update(last_error=None)  # clear any stale halt from a prior cycle
                return True, result['stdout']
            else:
                # A non-zero exit means the engine HALTED (escalation ladder
                # exhausted). Give the human the last lines of the execution
                # log so they can see exactly which step stalled and why.
                reason = result['stderr'] or result['stdout']
                log_tail = ""
                try:
                    log_path = os.path.join(self.config.get('work_dir', '/tmp'),
                                            'plan_execution.log')
                    if os.path.exists(log_path):
                        with open(log_path, 'r', errors='replace') as f:
                            lines = f.readlines()
                        log_tail = ''.join(lines[-15:])
                except Exception:
                    pass
                self.logger.error(f"Execution failed: {reason}")
                detail = f"HALTED — escalation ladder exhausted.\n\n{log_tail}\n\n{reason}".strip()
                return False, detail[:2000]
        
        except asyncio.TimeoutError:
            self.logger.error("Execution script timed out (2 hours)")
            return False, "Timeout"
        except Exception as e:
            self.logger.error(f"Execution error: {e}")
            return False, str(e)

    async def _engine_pid(self) -> Optional[int]:
        """Return the PID of a running execution engine, if any."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "pgrep", "-f", "execute_master_plan.py",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
            stdout, _ = await proc.communicate()
            pids = [int(p) for p in stdout.decode().split() if p.strip().isdigit()]
            return pids[0] if pids else None
        except Exception as e:
            self.logger.warning(f"Engine pid lookup failed: {e}")
            return None

    async def _attach_to_engine(self, pid: int) -> Tuple[bool, str]:
        """Wait for a live engine to finish, polling its log + state files.

        Returns (success, detail) inferred from plan_execution_state.json after
        the engine process exits — an engine HALT leaves failed-but-not-completed
        steps; a clean run completes every step.
        """
        audits = self.config.get('work_dir', '/tmp')
        state_path = os.path.join(audits, 'plan_execution_state.json')
        while os.path.exists(f"/proc/{pid}"):
            await asyncio.sleep(15)
        success = True
        try:
            if os.path.exists(state_path):
                with open(state_path) as f:
                    st = json.load(f)
                failed = set(st.get('failed_steps', []))
                completed = set(st.get('completed_steps', []))
                success = not (failed - completed)
        except Exception as e:
            self.logger.warning(f"Attach outcome inference failed: {e}")
        tail = ""
        try:
            log_path = os.path.join(audits, 'plan_execution.log')
            if os.path.exists(log_path):
                with open(log_path, 'r', errors='replace') as f:
                    lines = f.readlines()
                tail = ''.join(lines[-8:])
        except Exception:
            pass
        if success:
            return True, (tail.strip() or "execution finished")
        return False, f"HALTED — engine exited; latest log:\n{tail.strip()[:1500]}"

    async def rebuild_graphify(self) -> Tuple[bool, str]:
        """Rebuild Graphify after execution step."""
        self.state.update(phase="graphify", phase_start_time=datetime.now().isoformat())
        
        repo_dir = self.config['repo_dir']
        self.logger.info(f"Rebuilding Graphify in {repo_dir}")
        
        try:
            result1 = await self._run_command(
                ["graphify", ".", "--code-only"],
                cwd=repo_dir,
                timeout=300
            )
            
            if result1["returncode"] != 0:
                self.logger.error(f"Graphify --code-only failed: {result1['stderr']}")
                return False, result1['stderr']
            
            result2 = await self._run_command(
                ["graphify", "cluster-only", "."],
                cwd=repo_dir,
                timeout=300
            )
            
            if result2["returncode"] != 0:
                self.logger.error(f"Graphify cluster-only failed: {result2['stderr']}")
                return False, result2['stderr']
            
            self.logger.info("Graphify rebuild complete")
            return True, "Graphify rebuilt successfully"
        
        except asyncio.TimeoutError:
            self.logger.error("Graphify rebuild timed out")
            return False, "Timeout"
        except Exception as e:
            self.logger.error(f"Graphify rebuild error: {e}")
            return False, str(e)
    
    async def _run_script(self, script_path: str, timeout: int = 3600, env_extra: Dict = None,
                          extra_args: List[str] = None) -> Dict:
        """Run a Python script, streaming output to disk to bound memory."""
        if not os.path.exists(script_path):
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": f"Script not found: {script_path}"
            }

        temp_dir = tempfile.mkdtemp(prefix="orch_")
        out_path = os.path.join(temp_dir, "stdout.txt")
        err_path = os.path.join(temp_dir, "stderr.txt")

        env = os.environ.copy()
        for k, v in (env_extra or {}).items():
            env[k] = str(v)

        # Use THIS interpreter (the venv) for child scripts — bare "python3"
        # resolves to system python which lacks aiohttp/pytest and caused the
        # audit + STEP 253 verification failures (2026-08-05).
        proc_args = [sys.executable, script_path] + list(extra_args or [])

        try:
            with open(out_path, "wb") as fout, open(err_path, "wb") as ferr:
                result = await asyncio.wait_for(
                    asyncio.create_subprocess_exec(
                        *proc_args,
                        stdout=fout, stderr=ferr,
                        cwd=self.config.get('repo_dir') or os.path.dirname(script_path) or None,
                        env=env,
                    ),
                    timeout=timeout
                )
                # wait without buffering output in RAM; the child writes to files
                rc = await asyncio.wait_for(result.wait(), timeout=timeout)

            def _tail(path: str, limit: int = 200000) -> str:
                try:
                    with open(path, "rb") as f:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        f.seek(max(0, size - limit))
                        return f.read().decode("utf-8", errors="replace")
                except Exception:
                    return ""

            return {
                "returncode": rc,
                "stdout": _tail(out_path),
                "stderr": _tail(err_path)
            }

        except asyncio.TimeoutError:
            raise
        except Exception as e:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": str(e)
            }
        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
    
    async def _run_command(self, cmd: List[str], cwd: str, timeout: int = 300) -> Dict:
        """Run a shell command and capture output."""
        try:
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                ),
                timeout=timeout
            )
            
            stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=60)
            return {
                "returncode": result.returncode,
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace')
            }
        
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            return {
                "returncode": 1,
                "stdout": "",
                "stderr": str(e)
            }
    
    def _find_latest_file(self, directory: str, pattern: str) -> Optional[str]:
        """Find the latest file matching a glob pattern."""
        if not os.path.exists(directory):
            return None
        
        import glob
        files = glob.glob(os.path.join(directory, pattern))
        if not files:
            return None
        
        return max(files, key=os.path.getctime)


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM BOT HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

class TelegramBot:
    """Telegram bot for remote control and monitoring."""
    
    def __init__(
        self,
        token: str,
        chat_id: int,
        config: Dict,
        state_manager: StateManager,
        task_manager: TaskManager,
        llm_agent: LLMAgent,
        script_editor: ScriptEditor,
        workflow_executor: WorkflowExecutor,
        git_manager: GitManager,
        settings_manager: SettingsManager
    ):
        self.token = token
        self.chat_id = chat_id
        self.config = config
        self.state = state_manager
        self.tasks = task_manager
        self.llm = llm_agent
        self.editor = script_editor
        self.executor = workflow_executor
        self.git = git_manager
        self.settings = settings_manager
        self.application = None
        self.orchestrator = None
        self._want_restart = False      # Telegram asked to relaunch the loop
        self._stop_permanent = False    # /shutdown → exit the process for good
        self._file_cache_ts = None      # fuzzy-search index caches (60s TTL)
        self._dir_cache_ts = None
        self._file_names_cache = []
        self._dir_names_cache = []
        self._snap_cache = ""           # codebase-overview snapshot cache (60s)
        self._snap_cache_ts = None
        self._last_usage_ts = None      # usage-line cooldown

    def attach_orchestrator(self, orchestrator: "MasterOrchestrator"):
        """Give the bot a handle so /restart, /continue, /halt can drive the loop."""
        self.orchestrator = orchestrator

    async def initialize(self):
        """Initialize the Telegram bot application (full command set)."""
        self.application = Application.builder().token(self.token).build()

        handlers = [
            # core
            ("start", self.cmd_start), ("help", self.cmd_help), ("commands", self.cmd_commands),
            ("status", self.cmd_status), ("health", self.cmd_health), ("version", self.cmd_version),
            ("summary", self.cmd_summary), ("context", self.cmd_context),
            # workflow control
            ("pause", self.cmd_pause), ("resume", self.cmd_resume), ("stop", self.cmd_stop),
            ("shutdown", self.cmd_shutdown),
            ("halt", self.cmd_halt), ("kill", self.cmd_kill), ("continue", self.cmd_continue),
            ("restart", self.cmd_restart), ("reboot", self.cmd_restart),
            ("cycle", self.cmd_cycle), ("schedule", self.cmd_schedule),
            ("priority", self.cmd_priority), ("batch", self.cmd_batch),
            ("retry", self.cmd_retry), ("skip", self.cmd_skip), ("escalate", self.cmd_escalate),
            # logs & debugging
            ("logs", self.cmd_logs), ("error", self.cmd_error), ("errors", self.cmd_errors),
            ("trace", self.cmd_trace), ("tail", self.cmd_tail), ("debug", self.cmd_debug),
            ("speed", self.cmd_speed), ("eta", self.cmd_eta),
            ("performance", self.cmd_performance), ("stats", self.cmd_stats),
            # file ops
            ("view", self.cmd_view), ("cat", self.cmd_view),
            ("head", self.cmd_head), ("tail", self.cmd_tail),
            ("ls", self.cmd_ls), ("tree", self.cmd_tree),
            ("find", self.cmd_find), ("grep", self.cmd_grep),
            ("diff", self.cmd_diff), ("save", self.cmd_save),
            ("edit", self.cmd_edit), ("edit_mode", self.cmd_edit_mode),
            ("commit", self.cmd_commit), ("rollback", self.cmd_rollback),
            ("load", self.cmd_load), ("clear_context", self.cmd_clear_context), ("clear", self.cmd_clear_context),
            ("expand_context", self.cmd_expand_context), ("decrease_context", self.cmd_decrease_context),
            ("compact_context", self.cmd_compact_context),
            ("summon_claude", self.cmd_summon_claude),
            ("Jarvis", self.cmd_message_claude),       # personal name for talking to Claude
            ("jarvis", self.cmd_message_claude),       # lowercase alias
            ("message_claude", self.cmd_message_claude),  # keep old alias working
            ("mega", self.cmd_message_claude),         # direct line to Claude Code (skips auto-responder)
            # project context
            ("explain", self.cmd_explain), ("whatis", self.cmd_whatis),
            ("step", self.cmd_step), ("plan", self.cmd_plan), ("search", self.cmd_search),
            # settings / tasks
            ("settings", self.cmd_settings), ("set", self.cmd_set),
            ("task", self.cmd_task), ("tasks", self.cmd_tasks),
            # notification toggles
            ("notify_on", self.cmd_notify_on), ("notify_off", self.cmd_notify_off),
            ("quiet", self.cmd_quiet), ("verbose", self.cmd_verbose),
            ("history", self.cmd_history),
            # system / pipeline
            ("validate", self.cmd_validate), ("test", self.cmd_test),
            ("audit", self.cmd_audit), ("deploy", self.cmd_deploy),
            ("update", self.cmd_update),
        ]
        for name, cb in handlers:
            self.application.add_handler(CommandHandler(name, cb))

        # Effort levels: /effort2 explain step 54  OR  /effort2 (set default)
        for _level in (1, 2, 3):
            self.application.add_handler(CommandHandler(
                f"effort{_level}", lambda u, c, lvl=_level: self._cmd_effort(u, c, lvl)
            ))

        # CRITICAL: /purple_apple is the security gate phrase. Messages that
        # START with it are treated as bot commands by python-telegram-bot and
        # would NEVER reach handle_message (silently dropped as unknown
        # commands). Register it as a command that forwards through the same
        # gate logic so '/purple_apple <message>' works.
        self.application.add_handler(CommandHandler(
            "purple_apple", self._cmd_purple_apple
        ))

        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))

        await self.application.initialize()
        await self.application.start()
        # NO updater.start_polling() here: the MCP Telegram plugin (bun
        # server) owns the getUpdates long-poll for this bot token. A second
        # poller trips Telegram's "Conflict: terminated by other getUpdates
        # request", which was crash-looping the orchestrator. This bot is
        # now SEND-ONLY (outbox relay + status pushes); incoming messages
        # arrive via the plugin's watch-inbox monitor instead.

    def _ctx_meter(self) -> str:
        """Live context usage: '🧠 12k/75k' — fundamentals + loads + history over budget."""
        try:
            used = self.llm._ctx_overhead + self.llm._history_tokens(self.llm.history)
            budget = self.llm.context_budget
            return f"🧠 {used // 1000}k/{budget // 1000}k"
        except Exception:
            return ""

    def _flush_usage_line(self) -> str:
        """Return the token/cache/cost line for the LAST LLM exchange.

        Appended to every outgoing message (including /message_claude acks and
        relayed replies). The usage reflects the most recent LLM call this
        process made; a short cooldown keeps notification bursts from spamming
        the same line.
        """
        try:
            if self.llm is None:
                return ""
            u = self.llm.last_usage or {}
            if not u.get("prompt_tokens"):
                # no LLM usage yet — but always show SOMETHING so the user
                # knows the tag is live (zeros until the first real call).
                now = time.time()
                if self._last_usage_ts is not None and now - self._last_usage_ts < 10:
                    return ""
                self._last_usage_ts = now
                return "⚡ 0 in / 0 out · cache 0% · ≈$0.00000 (no calls yet)"
            now = time.time()
            if self._last_usage_ts is not None and now - self._last_usage_ts < 10:
                return ""
            self._last_usage_ts = now
            return self.llm.usage_summary()
        except Exception:
            return ""

    async def send_message(self, text: str, parse_mode: str = "Markdown"):
        """Send a message to the configured chat, chunked and with plain-text fallback.
        Clean — no legacy brain emoji or cost stats. Direct conversation.

        DUPLICATE-SAFE (2026-08-05): the markdown->plain fallback only re-sends
        when the first attempt PROVABLY did not deliver (Telegram 400 "can't
        parse entities"). Timeouts / network errors / 429s are ambiguous —
        Telegram may have delivered the message but the client saw an error —
        and resending delivers the SAME text twice to the user. Log and drop."""
        if not text:
            return
        for chunk in self._chunk_message(text, 4096):
            try:
                await self.application.bot.send_message(
                    chat_id=self.chat_id,
                    text=chunk,
                    parse_mode=parse_mode
                )
            except Exception as e:
                err = str(e)
                if "can't parse entities" not in err:
                    # ambiguous failure (timeout/network/429): may have delivered
                    LOGGER.error(f"Telegram send failed (may have delivered — NOT resending): {err[:120]}")
                    return
                # 400 parse error — provably NOT delivered, safe to resend plain
                try:
                    await self.application.bot.send_message(
                        chat_id=self.chat_id,
                        text=chunk,
                        parse_mode=None
                    )
                except Exception as e2:
                    LOGGER.error(f"Failed to send Telegram message: {e2}")
                    return

    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings — show current settings."""
        await self.send_message(self.settings.describe())
        await self.send_message(
            "Change with `/set <key> <value>` — e.g. `/set pause_after execution`\n"
            "Or just say it in plain words, e.g. \"from now on finish execution then pause\"."
        )

    async def cmd_set(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /set <key> <value> — update a single settings key."""
        if not context.args or len(context.args) < 2:
            await self.send_message(
                "❌ Usage: `/set <key> <value>`\n"
                "Keys: loop_mode (full|once), pause_after (none|audit|cross_eval|execution|cycle), "
                "on_step_failure (continue|halt), max_cycles (int), notify_interval (sec), "
                "edit_mode (restricted|full), notifications (on|off)"
            )
            return

        key = context.args[0].lower()
        value = ' '.join(context.args[1:])

        valid = {
            "loop_mode": lambda v: v in ("full", "once"),
            "pause_after": lambda v: v in ("none", "audit", "cross_eval", "execution", "cycle"),
            "on_step_failure": lambda v: v in ("continue", "halt"),
            "max_cycles": lambda v: str(v).isdigit(),
            "notify_interval": lambda v: str(v).isdigit() and int(v) >= 60,
            "edit_mode": lambda v: v in ("restricted", "full"),
            "notifications": lambda v: v in ("on", "off"),
            "selfmod_branching": lambda v: v in ("on", "off"),
            "readme_autoupdate": lambda v: v in ("on", "off"),
            "auto_summon_claude": lambda v: v in ("on", "off"),
            "summon_budget": lambda v: str(v).isdigit() and 0 <= int(v) <= 10,
            "fresh_audit": lambda v: v in ("on", "off"),
        }
        if key not in valid:
            await self.send_message(f"❌ Unknown key `{key}`. See /set usage.")
            return
        if not valid[key](value):
            await self.send_message(f"❌ Invalid value `{value}` for `{key}`.")
            return

        if key in ("max_cycles", "notify_interval"):
            value = int(value)

        new_settings = self.settings.update({key: value})
        await self.send_message(f"✅ Settings updated:\n```\n{json.dumps(new_settings, indent=2)}\n```")
    
    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        help_text = """
🚀 **Project Orchestrator Started**

Commands:
/status — Current cycle, step, phase
/pause — Pause workflow after current step
/resume — Resume paused workflow
/stop — Stop the orchestrator (graceful shutdown)
/logs — Last 50 lines of execution log
/task <id> — Execute a specific task (e.g., /task T001)
/tasks — Show all tasks and their status
/edit_mode [restricted|full] — Toggle edit permissions
/edit <file> <old> <new> — Edit a file
/commit <message> — Commit changes to Git
/rollback — Rollback last commit
/view <file> [lines] — View file contents
/settings — Show current settings
/set <key> <value> — Change a setting
/effort1|2|3 — Answer at that effort (also prefix any message, e.g. `/effort2 explain step 54`)
/help — Show this help message

**Effort levels**: 1=lean (default), 2=balanced, 3=maximum. `/effort2` alone sets the default.
**Direct LLM Chat**: Send any message to query the LLM agent (default effort 1).

Type `/commands` for the full list of everything you can send.
"""
        await self.send_message(help_text)

    async def cmd_commands(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /commands — list every command type with a short explanation."""
        commands = [
            ("**📂 File ops**", ""),
            ("/view <file> [a-b]  |  /cat", "View file (optional line range, e.g. `:10-30`)"),
            ("/head <file> [n]  |  /tail <file> [n]", "First / last N lines of a file"),
            ("/ls <path>  |  /tree <path> [depth]", "List dir / show structure"),
            ("/find <name> [dir]", "Find files by name"),
            ("/grep <pattern> [dir]", "Grep the codebase"),
            ("/diff [file]", "Show uncommitted git changes"),
            ("/edit <file> <line> <new>", "Edit a line (or `<start>-<end> <new>`)"),
            ("/save", "Commit everything with an auto message"),
            ("**📖 Project context**", ""),
            ("/explain <file|module>", "LLM explains that file"),
            ("/whatis <function>", "Locate + explain a function"),
            ("/step <n>", "Explain step n of the plan + its status"),
            ("/plan", "Plan summary (done/next/fails)"),
            ("/search <query>", "Grep + LLM explanation"),
            ("/context", "LLM context window state"),
            ("/load <file>", "Force-load a file into LLM context"),
            ("/clear_context", "Reset the LLM conversation"),
            ("/expand_context <n>", "Raise context budget (75k-200k tokens)"),
            ("/decrease_context <n>", "Lower context budget (16k-75k)"),
            ("/compact_context", "Reset budget to 75k + clear loaded history"),
            ("/summary", "High-level system summary"),
            ("", "Every reply shows the live context meter, e.g. 🧠 12k/75k."),
            ("", "The agent is general-purpose: ask it anything in plain words and it"),
            ("", "  explores files, runs commands, edits, and commits on its own."),
            ("**⚙️ Execution & control**", ""),
            ("/retry <step>", "Mark step for re-run"),
            ("/skip <step> confirm", "FORCE-mark a step done (requires `confirm` — violates never-skip rule)"),
            ("/escalate <step>", "Re-run step + force conversational agent to Pro"),
            ("/halt", "Emergency stop orchestrator (engine keeps current batch)"),
            ("/kill", "Kill engine + subagents too"),
            ("/continue", "Resume, or relaunch orchestrator if dead"),
            ("/cycle <n>", "Jump to a specific cycle"),
            ("/speed  |  /eta", "Throughput / completion estimate"),
            ("**📋 Workflow mgmt**", ""),
            ("/pause  |  /resume", "Pause / resume the workflow"),
            ("/stop", "Graceful orchestrator stop"),
            ("/schedule <HH:MM|in N min>", "Schedule an automatic pause"),
            ("/priority <high|med|low>", "Set work priority (informational)"),
            ("/batch <a>-<b>", "Record a batch focus range"),
            ("/validate", "Compile-check + git status"),
            ("/test", "Run pytest (can take minutes)"),
            ("/audit", "Mini health assessment via LLM"),
            ("/deploy", "Git push + verify orchestrator"),
            ("**📜 Logs & debugging**", ""),
            ("/logs [n | a-b]", "Orchestrator log tail / range"),
            ("/tail", "Live engine log tail (snapshot)"),
            ("/error", "Last error"),
            ("/errors", "Failed steps summary"),
            ("/trace <step>", "Log lines mentioning a step"),
            ("/performance  |  /stats", "Throughput / system stats"),
            ("/debug [off]", "Toggle debug (notify every 60s)"),
            ("**🔔 Telegram-specific**", ""),
            ("/effort1|2|3 [text]", "Lean/balanced/max answer; alone = set default"),
            ("/notify_on | /notify_off", "Enable/disable progress pings"),
            ("/quiet | /verbose", "Same — silence / restore pings"),
            ("/history", "Recent LLM conversation turns"),
            ("**🖥 System**", ""),
            ("/update", "Fetch + fast-forward pull"),
            ("/version", "Git hash + process status"),
            ("/status  |  /health", "Full status / health check"),
            ("/restart | /reboot", "Restart the orchestrator"),
            ("/shutdown", "Graceful shutdown"),
            ("plain text", "Chat with the LLM — or just say things like \"step 54\", \"show logs\", \"retry step 600\""),
        ]
        lines = ["📚 **All commands**\n"]
        for cmd, desc in commands:
            lines.append(f"`{cmd}` — {desc}" if cmd else f"\n{desc}")
        msg = "\n".join(lines)
        for chunk in self._chunk_message(msg, 4096):
            await self.send_message(chunk)
    
    def _plan_progress(self) -> Dict:
        """Read live step-progress from plan_execution_state.json (fresh each call).

        `failed` = total failure records in history (includes retries that later
        succeeded). `unresolved_fails` = steps that have failed AND are NOT yet
        completed — the number that actually matters right now.
        """
        audits = self.config.get('work_dir', '/tmp')
        state_path = os.path.join(audits, 'plan_execution_state.json')
        progress = {"completed": 0, "current": None, "total": 661, "failed": 0,
                    "unresolved_fails": [], "last_log": ""}
        try:
            if os.path.exists(state_path):
                with open(state_path) as f:
                    st = json.load(f)
                completed = st.get('completed_steps', [])
                failed = st.get('failed_steps', [])
                progress["completed"] = len(completed)
                progress["failed"] = len(failed)
                progress["unresolved_fails"] = sorted({x for x in failed if x not in set(completed)})
                if completed:
                    progress["current"] = max(completed) + 1
            # recent activity from the execution log
            log_path = os.path.join(audits, 'plan_execution.log')
            if os.path.exists(log_path):
                with open(log_path, 'r', errors='replace') as f:
                    lines = f.readlines()
                progress["last_log"] = ''.join(lines[-6:]).strip()
        except Exception as e:
            LOGGER.warning(f"Could not read plan progress: {e}")
        return progress

    def _speed_eta(self) -> Tuple[str, str]:
        """Estimate throughput (steps/h) and ETA from the execution log timestamps."""
        audits = self.config.get('work_dir', '/tmp')
        log_path = os.path.join(audits, 'plan_execution.log')
        times = []
        try:
            if os.path.exists(log_path):
                with open(log_path, 'r', errors='replace') as f:
                    for line in f:
                        m = re.match(r'\[([\d-]+ [\d:]+)\].*?Step (\d+)/661 COMPLETE', line)
                        if m:
                            try:
                                t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                                times.append((int(m.group(2)), t))
                            except ValueError:
                                pass
        except Exception:
            pass
        if len(times) < 2:
            return "n/a", "n/a"
        recent = times[-20:]
        dt = (recent[-1][1] - recent[0][1]).total_seconds()
        steps = recent[-1][0] - recent[0][0]
        if steps <= 0 or dt <= 0:
            return "n/a", "n/a"
        sec_per_step = dt / steps
        speed = 3600.0 / sec_per_step
        remaining = 661 - recent[-1][0]
        eta = recent[-1][1] + timedelta(seconds=max(0, remaining * sec_per_step))
        return f"{speed:.1f} steps/h", eta.strftime("%m-%d %H:%M")

    def _build_context(self) -> str:
        """Assemble a compact snapshot of live orchestrator state for the LLM."""
        st = self.state.state
        prog = self._plan_progress()
        s = self.settings.get()
        speed, eta = self._speed_eta()
        lines = [
            "=== LIVE PROJECT ORCHESTRATOR CONTEXT (fresh) ===",
            f"Context generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — answer progress questions from THESE numbers, never from your conversation memory.",
            f"Cycle: {st.get('cycle', 0)} | Phase: {st.get('phase', 'idle')} | Running: {st.get('running', False)} | Paused: {st.get('paused', False)} | Halted: {st.get('halted', False)}",
            f"Execution progress: {prog['completed']} of {prog['total']} plan steps completed",
            f"Current step (next to run): {prog['current'] if prog['current'] else 'all done'}",
            f"ALL_STEPS_DONE: {'YES' if prog['completed'] >= prog['total'] else 'no'}",
            f"Unresolved failed steps: {prog['unresolved_fails'] or 'none'}",
            f"Speed/ETA: {speed} / {eta}",
            f"Settings: loop_mode={s.get('loop_mode')}, pause_after={s.get('pause_after')}, on_step_failure={s.get('on_step_failure')}, max_cycles={s.get('max_cycles') or 'unlimited'}, notifications={s.get('notifications')}",
            f"Priority: {st.get('current_priority') or 'normal'} | Batch focus: {st.get('batch_range') or 'all steps'}",
            f"Last error: {st.get('last_error') or 'none'}",
            "=== RECENT EXECUTION LOG ===",
            prog['last_log'] or "(no execution activity yet)",
            "=== END CONTEXT ===",
            "",
            self._build_codebase_snapshot(),
        ]
        return "\n".join(lines)

    async def _reply_lean_progress(self):
        """ONE-LINE answer to 'what step are we on' — no dashboard dump."""
        try:
            prog = self._plan_progress()
            done = prog['completed']
            cur = prog['current'] or done
            remaining = prog['total'] - done
            unf = prog['unresolved_fails']
            if unf:
                await self.send_message(f"Step {cur}/{prog['total']} · {done} done · {remaining} left · ❌ fails: {unf}")
            else:
                await self.send_message(f"Step {cur}/{prog['total']} · {done} done · {remaining} left")
        except Exception:
            await self.send_message("Couldn't read progress right now — try /status.")

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command."""
        prog = self._plan_progress()
        st = self.state.state
        speed, eta = self._speed_eta()
        status = f"""
📊 **Orchestrator Status**

Cycle: {st['cycle']} | Phase: {st['phase']}
Running: {'Yes' if st['running'] else 'No'} | Paused: {'Yes' if st['paused'] else 'No'} | Halted: {'Yes' if st.get('halted') else 'No'}

**Execution progress**
Plan steps completed: {prog['completed']} / {prog['total']}
Next step: {prog['current'] if prog['current'] else 'ALL DONE 🎉'}
Unresolved failed steps: {prog['unresolved_fails'] or 'none'}
Speed: {speed} | ETA: {eta}

Edit Mode: {st['edit_mode']}
LLM Agent: {self.llm.model_tier} ({self.llm.model})
Context window: {self.llm.context_budget:,} tokens
Effort: {self.llm.effort_default}
Notifications: {self.settings.get().get('notifications', 'on')}
Last Audit: {st.get('last_audit_file') or 'None'}
Last Plan: {st.get('last_plan_file') or 'None'}
Last Error: {st.get('last_error') or 'None'}
"""
        await self.send_message(status)
    
    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command."""
        self.state.update(paused=True)
        await self.send_message("⏸️ Workflow paused. Will stop after current step completes.")
    
    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command."""
        self.state.update(paused=False)
        await self.send_message("▶️ Workflow resumed.")
    
    async def cmd_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stop — stop the workflow loop but keep the bot alive for /continue."""
        self.state.update(running=False)
        await self.send_message("🛑 Orchestrator loop stopping. Bot stays up — send /continue to resume, /shutdown to exit.")

    async def cmd_shutdown(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /shutdown — stop the loop and exit the process for good."""
        self.state.update(running=False)
        self._stop_permanent = True
        await self.send_message("🛑 Shutting down orchestrator. Goodbye.")
    
    async def cmd_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /logs [n] or /logs <start>-<end>."""
        log_file = self.config.get('log_file', '/tmp/orchestrator.log')

        if not os.path.exists(log_file):
            await self.send_message("❌ No log file found")
            return

        try:
            with open(log_file, 'r', errors='replace') as f:
                lines = f.readlines()
            total = len(lines)

            if context.args:
                m = re.match(r'(\d+)-(\d+)$', context.args[0])
                if m:
                    start, end = int(m.group(1)), int(m.group(2))
                    start = max(1, start)
                    end = min(total, end)
                    content = ''.join(lines[start - 1:end])[-3900:]
                    await self.send_message(f"📋 **Log** lines {start}-{end} of {total}\n```\n{content}\n```")
                    return
                try:
                    n = min(int(context.args[0]), 400)
                except ValueError:
                    n = 50
            else:
                n = 50
            content = ''.join(lines[-n:])[-3900:]
            await self.send_message(f"📋 **Last {n} log lines** (of {total})\n```\n{content}\n```")
        except Exception as e:
            await self.send_message(f"❌ Error reading logs: {e}")
    
    async def cmd_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /task <id> command."""
        if not context.args:
            await self.send_message("❌ Usage: /task <task_id> (e.g., /task T001)")
            return
        
        task_id = context.args[0].upper()
        task = self.tasks.get_task(task_id)
        
        if not task:
            await self.send_message(f"❌ Task {task_id} not found")
            return
        
        await self.send_message(f"🤖 Executing task {task_id}: {task['description']}...")
        
        result = await self.llm.execute_task(task)
        self.tasks.update_task(task_id, "done", result)

        if self.llm.last_tier_event:
            await self.send_message(self.llm.last_tier_event)
        if self.llm.last_context_event:
            await self.send_message(self.llm.last_context_event)

        msg = f"""
✅ **Task {task_id} Complete**

**Task**: {task['description']}
**Result**: {result}
"""
        await self.send_message(msg)
    
    async def cmd_tasks(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /tasks command."""
        all_tasks = self.tasks.get_all_tasks()
        
        pending = [t for t in all_tasks if t['status'] == 'pending']
        done = [t for t in all_tasks if t['status'] == 'done']
        
        msg = f"""
📋 **Task Board**

**Pending** ({len(pending)}):
"""
        for task in pending[:5]:
            msg += f"\n  • {task['id']}: {task['description'][:60]}..."
        
        if len(pending) > 5:
            msg += f"\n  ... and {len(pending) - 5} more"
        
        msg += f"\n\n**Completed** ({len(done)})"
        
        if done:
            for task in done[-3:]:
                msg += f"\n  ✓ {task['id']}: {task['description'][:60]}..."
        
        msg += f"\n\nUse /task <id> to run a specific task"
        
        await self.send_message(msg)
    
    async def cmd_edit_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /edit_mode command."""
        if not context.args:
            await self.send_message(f"Current mode: {self.state.state['edit_mode']}")
            return
        
        mode = context.args[0].lower()
        if mode not in ['restricted', 'full']:
            await self.send_message("❌ Mode must be 'restricted' or 'full'")
            return
        
        self.state.update(edit_mode=mode)
        await self.send_message(f"✅ Edit mode changed to: {mode}")
    
    async def cmd_edit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /edit — line-based: /edit <file> <line> <new_text>  or  /edit <file> <start>-<end> <new_text>."""
        if len(context.args) < 3:
            await self.send_message("❌ Usage: `/edit <file> <line> <new_text>` or `/edit <file> <start>-<end> <new_text>`")
            return

        file_path = self._resolve_path(context.args[0])
        if not file_path:
            await self.send_message("❌ Path not allowed (out of scope or secret file).")
            return
        if not os.path.isfile(file_path):
            await self.send_message(f"❌ File not found: {context.args[0]}")
            return

        mode = self.state.state['edit_mode']
        if mode == 'restricted':
            allowed_files = ('parallel_agents.py', 'parallel_agent_cross_eval.py', 'execute_master_plan.py')
            if not any(allowed in file_path for allowed in allowed_files):
                await self.send_message("❌ Restricted mode: only the 3 workflow scripts are editable. Use `/edit_mode full` for others.")
                return
        if 'run_workflow.py' in file_path or 'config' in file_path:
            await self.send_message("❌ Cannot edit orchestrator or config files.")
            return

        pos = context.args[1]
        new_text = ' '.join(context.args[2:])
        m = re.match(r'^(\d+)(?:-(\d+))?$', pos.strip())
        if not m:
            await self.send_message("❌ Line must be a number or range like `12` or `12-18`.")
            return
        start, end = int(m.group(1)), int(m.group(2)) if m.group(2) else int(m.group(1))

        try:
            with open(file_path, 'r', errors='replace') as f:
                lines = f.readlines()
            total = len(lines)
            if start < 1 or end > total:
                await self.send_message(f"❌ Range {start}-{end} is outside the file (1-{total}).")
                return
            replacement = [ln.rstrip('\n') + '\n' for ln in new_text.split('\n')]
            lines[start - 1:end] = replacement
            err = self._validate_syntax(file_path, lines)
            if err:
                await self.send_message(f"❌ Validation failed — edit NOT applied:\n```\n{err[:800]}\n```")
                return
            tmp = file_path + ".tg_edit.tmp"
            with open(tmp, 'w') as f:
                f.writelines(lines)
            os.replace(tmp, file_path)
            await self.send_message(
                f"✅ Edited `{context.args[0]}` lines {start}-{end} "
                f"({len(replacement)} line(s)). Run `/commit <msg>` to save to git."
            )
        except Exception as e:
            await self.send_message(f"❌ {e}")
    
    async def cmd_commit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /commit command."""
        message = ' '.join(context.args) if context.args else "Manual commit via Telegram"
        
        if self.git.commit(message):
            await self.send_message(f"✅ Changes committed: {message}")
        else:
            await self.send_message("❌ Commit failed")
    
    async def cmd_rollback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /rollback command."""
        if self.git.rollback():
            await self.send_message("✅ Rolled back last commit")
        else:
            await self.send_message("❌ Rollback failed")
    
    async def cmd_view(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /view — /view <file> [start-end] or /view <file>:<start>-<end>."""
        if not context.args:
            await self.send_message("❌ Usage: `/view <file> [start-end]` — e.g. `/view core.py:10-30`")
            return

        spec = context.args[0]
        start_line, end_line = 1, None
        if ':' in spec:
            spec, _, rang = spec.partition(':')
            m = re.match(r'(\d+)(?:-(\d+))?$', rang.strip())
            if m:
                start_line = int(m.group(1))
                end_line = int(m.group(2)) if m.group(2) else None
        elif len(context.args) > 1:
            start_line = int(context.args[1])
            end_line = int(context.args[2]) if len(context.args) > 2 else None

        file_path = self._resolve_path(spec)
        if not file_path:
            await self.send_message("❌ Path not allowed (out of scope or secret file).")
            return
        if not os.path.isfile(file_path):
            await self.send_message(f"❌ File not found: {spec}")
            return

        try:
            content, start_line, end_line, total = self._read_lines(file_path, start_line, end_line)
            cap = 400
            if end_line - start_line + 1 > cap:
                content, start_line, end_line, total = self._read_lines(file_path, start_line, start_line + cap - 1)
            msg = f"📄 **{spec}** (lines {start_line}-{end_line} of {total})\n```\n{content}\n```"
            await self.send_message(msg)
        except Exception as e:
            await self.send_message(f"❌ Error reading file: {e}")
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        await self.cmd_start(update, context)
    
    @staticmethod
    def _looks_like_settings_request(text: str) -> bool:
        t = text.lower()
        keywords = [
            "setting", "from now on", "pause after", "loop mode", "loop_mode",
            "pause_after", "max_cycle", "max cycle", "notify", "stop after",
            "then pause", "and pause", "continue on fail", "on_step_failure",
            "edit mode", "edit_mode", "change your",
        ]
        return any(k in t for k in keywords)

    async def _cmd_effort(self, update: Update, context: ContextTypes.DEFAULT_TYPE, level: int):
        """Handle /effort1|2|3. With text -> answer at that effort; alone -> set default."""
        rest = ' '.join(context.args or []).strip()
        if not rest:
            msg = self.llm.set_effort_default(level)
            await self.send_message(msg)
            return
        await self._process_user_message(rest, effort=level)

    # ── /purple_apple security gate ─────────────────────────────────────────
    # Every message MUST contain the secret phrase /purple_apple. Missing it →
    # "bot unavailable" and a counter increments; 5 consecutive misses LOCK
    # the bot (it ignores everything until the owner sends the unlock phrase
    # with the secret). Protects against unauthorized Telegram users.
    SECRET_PHRASE = "/purple_apple"
    SECRET_UNLOCK = "/purple_apple_unlock"
    MISS_LIMIT = 5

    async def _cmd_purple_apple(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Command handler for '/purple_apple <message>'.

        python-telegram-bot routes any message STARTING with '/' to command
        handlers, so the gate phrase at the start of a message would never
        reach handle_message. This reconstructs the full text (phrase +
        remainder) and runs the SAME gate-pass path.
        """
        try:
            args = " ".join(context.args or [])
            # If the phrase came as '/purple_apple' with the message in args,
            # rebuild the full text. The gate then strips the phrase as usual.
            text = args.strip()
            if text:
                text = f"{self.SECRET_PHRASE} {text}"
            else:
                text = self.SECRET_PHRASE
            await self._handle_gated_message(text)
        except Exception as e:
            LOGGER.error(f"_cmd_purple_apple failed: {e}")
            await self.send_message("⚠️ Error processing message.")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Every plain message is relayed straight to the inbox.
        Sends an instant ack with a unique random ID so the user knows
        it's a fresh receipt (not a cached/replay).

        Security gate: message must contain /purple_apple or it's rejected.
        """
        text = (update.message.text or "").strip()
        if not text:
            return
        await self._handle_gated_message(text)

    async def _handle_gated_message(self, text: str):
        """Shared gate + relay logic used by handle_message AND the
        /purple_apple command handler (messages starting with the phrase
        are routed to the command handler by python-telegram-bot)."""
        # ── Security gate ──
        miss_count = int(self.state.state.get("secret_misses", 0))
        locked = bool(self.state.state.get("secret_locked", False))

        if locked:
            if self.SECRET_UNLOCK in text:
                self.state.update(secret_locked=False, secret_misses=0)
                await self.send_message("🔓 Bot unlocked. Welcome back.")
            else:
                await self.send_message("🔒 Bot locked (security). Use the unlock phrase.")
            return

        if self.SECRET_PHRASE not in text:
            miss_count += 1
            self.state.update(secret_misses=miss_count)
            if miss_count >= self.MISS_LIMIT:
                self.state.update(secret_locked=True, secret_misses=0)
                await self.send_message("🔒 Bot locked: too many failed attempts. Use the unlock phrase.")
            else:
                # Clearer than a bare "bot unavailable" — the user (owner)
                # needs to know WHY the message was dropped.
                await self.send_message(
                    f"🤖 Bot unavailable. ({miss_count}/{self.MISS_LIMIT}) "
                    f"Message not delivered — missing the required key phrase."
                )
            return

        # Gate passed — reset counter
        if miss_count:
            self.state.update(secret_misses=0)
        # Remove the secret phrase from the message before relaying (keep the
        # user's actual content clean).
        text = text.replace(self.SECRET_PHRASE, "").strip()

        try:
            inbox = []
            if os.path.exists(self.CLAUDE_INBOX):
                with open(self.CLAUDE_INBOX) as f:
                    inbox = json.load(f)
            inbox.append({"ts": datetime.now().isoformat(), "from": "telegram", "text": text})
            tmp = self.CLAUDE_INBOX + ".tmp"
            with open(tmp, "w") as f:
                json.dump(inbox, f, indent=2)
            os.replace(tmp, self.CLAUDE_INBOX)
        except Exception as e:
            LOGGER.error(f"relay to claude inbox failed: {e}")
        # Instant ack with unique 30-digit receipt ID + message preview
        import random
        rid = ''.join(str(random.randint(0, 9)) for _ in range(30))
        preview = text[:80] + ("…" if len(text) > 80 else "")
        await self.send_message(f"📥 #{rid} «{preview}»")

    async def _process_user_message(self, user_text: str, effort: int = None):
        """Shared pipeline: settings detection, then the general-purpose LLM.

        Plain messages are NEVER routed to file/status commands — the LLM
        decides what it needs via its own tool calls (view_file, grep_code,
        run_command, ...). Slash commands (/status, /view, /explain, ...) are
        explicit and handled separately.
        """
        user_text = (user_text or "").strip()
        if not user_text:
            return

        # 1) Settings detection from natural language.
        if self._looks_like_settings_request(user_text):
            patch = await self.llm.parse_settings_change(user_text)
            if patch:
                new_settings = self.settings.update(patch)
                await self.send_message(
                    f"✅ Settings updated:\n```\n{json.dumps(new_settings, indent=2)}\n```"
                )
                return

        await self.send_message(f"🤖 Processing: {user_text[:60]}...")

        # Give the LLM the live orchestrator snapshot so it can answer
        # progress questions like "what step are we on" correctly.
        context = self._build_context()
        response = await self.llm.direct_query(
            user_text, context_snapshot=context, effort=effort
        )

        await self._reply_llm(response)

    async def _interpret_command(self, text: str) -> bool:
        """No-op. Plain messages are handled entirely by the LLM agent with its
        own tools. Slash commands are explicit (registered handlers). Kept for
        API stability only — always returns False."""
        return False
    
    # ── LLM reply plumbing ───────────────────────────────────────────────────
    async def _reply_llm(self, response: str):
        """Send an LLM reply + announce any tier/context changes it requested.
        The token/cache/cost line is appended by send_message automatically."""
        if self.llm.last_tier_event:
            await self.send_message(self.llm.last_tier_event)
        if self.llm.last_context_event:
            await self.send_message(self.llm.last_context_event)
        for chunk in self._chunk_message(response, 4096):
            await self.send_message(chunk)

    # ── path safety ──────────────────────────────────────────────────────────
    _SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__",
                  ".venv", "env", ".idea", "backups", "tokens_keys",
                  "site-packages", "dist-packages"}

    def _walkable(self, root: str):
        """Yield (dirpath, dirnames, filenames), pruning venvs & noise dirs."""
        for dirpath, dirnames, filenames in os.walk(root):
            # prune any virtualenv root (marker: pyvenv.cfg) so third-party
            # packages never pollute fuzzy search
            if os.path.isfile(os.path.join(dirpath, "pyvenv.cfg")):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in self._SKIP_DIRS]
            yield dirpath, dirnames, filenames

    def _allowed_roots(self) -> List[str]:
        roots = []
        for key in ("repo_dir", "work_dir", "git_repo_root"):
            p = self.config.get(key)
            if p and os.path.isdir(p):
                roots.append(os.path.abspath(p))
        # the orchestrator's own tree (alt_important_scripts) and its parent workspace
        here = os.path.dirname(os.path.abspath(__file__))
        roots.append(os.path.abspath(here))
        roots.append(os.path.abspath(os.path.join(here, "..")))
        home_ws = "~"
        if os.path.isdir(home_ws):
            roots.append(home_ws)
        return sorted(set(roots))

    def _resolve_path(self, user_path: str) -> Optional[str]:
        """Resolve a user-supplied path against allowed roots; block secrets."""
        if not user_path:
            return None
        p = os.path.expanduser(user_path.strip())
        if not os.path.isabs(p):
            for root in self._allowed_roots():
                cand = os.path.abspath(os.path.join(root, p))
                if os.path.exists(cand):
                    p = cand
                    break
        p = os.path.abspath(p)
        low = p.lower()
        for bad in SECRET_GUARD_SUBSTRINGS:
            if bad in low:
                return None
        for root in self._allowed_roots():
            if p == root or p.startswith(root + os.sep):
                return p
        return None

    def _all_file_names(self) -> List[Tuple[str, str]]:
        """[(normalized_basename, full_path)] for every file under allowed roots.

        Cached for 60s — walking the whole workspace tree on every fuzzy query
        is expensive. On name collisions: files inside the project repo win over
        files elsewhere; otherwise the larger (more substantive) file wins.
        """
        now = time.time()
        if self._file_cache_ts is not None and now - self._file_cache_ts < 60:
            return self._file_names_cache
        project = self.config.get('repo_dir', '')
        names = {}
        for root in self._allowed_roots():
            for dirpath, dirnames, filenames in self._walkable(root):
                for fn in filenames:
                    full = os.path.join(dirpath, fn)
                    base = os.path.splitext(fn)[0].lower()
                    if not base:
                        continue
                    cur = names.get(base)
                    if cur is None:
                        names[base] = full
                        continue
                    # prefer the project-repo path on collisions
                    cur_in = cur.startswith(project + os.sep)
                    new_in = full.startswith(project + os.sep)
                    if new_in and not cur_in:
                        names[base] = full
                    elif new_in == cur_in:
                        try:
                            if os.path.getsize(full) > os.path.getsize(cur):
                                names[base] = full
                        except OSError:
                            pass
        self._file_names_cache = list(names.items())
        self._file_cache_ts = now
        return self._file_names_cache

    def _all_dir_names(self) -> List[Tuple[str, str]]:
        """[(normalized_dirname, path_to_primary_py)] for package dirs under allowed roots."""
        now = time.time()
        if self._dir_cache_ts is not None and now - self._dir_cache_ts < 60:
            return self._dir_names_cache
        dirs = {}
        for root in self._allowed_roots():
            for dirpath, dirnames, _filenames in self._walkable(root):
                for d in dirnames:
                    base = d.lower()
                    if base in ("tests", "test"):
                        continue
                    full = os.path.join(dirpath, d)
                    cands = [os.path.join(full, f) for f in os.listdir(full) if f.endswith('.py')]
                    if not cands:
                        continue
                    # primary: __init__.py if present, else the largest .py (the
                    # substantive module, not a re-export shim)
                    init = [c for c in cands if c.endswith('__init__.py')]
                    if init:
                        primary = init[0]
                    else:
                        primary = max(cands, key=lambda c: os.path.getsize(c))
                    if base not in dirs:
                        dirs[base] = primary
        self._dir_names_cache = list(dirs.items())
        self._dir_cache_ts = now
        return self._dir_names_cache

    def _resolve_fuzzy_file(self, query: str) -> Tuple[Optional[str], List[str]]:
        """Infer the intended file from a fuzzy/typo'd name.

        Returns (best_path, candidates). candidates is a short list of close
        names when the match is AMBIGUOUS — the bot should ask the user instead
        of guessing wrong.
        """
        if not query:
            return None, []
        q = os.path.splitext(query.strip().lower())[0]
        q = re.sub(r'[^a-z0-9_]+', '', q)
        if not q:
            return None, []
        files = self._all_file_names()
        sm = difflib.SequenceMatcher

        # score FILES
        fscored = []
        for base, full in files:
            r = sm(None, q, base).ratio()
            if q in base:
                r += 0.3  # substring is a strong signal
            if r >= 0.5:
                fscored.append((r, base, full))
        fscored.sort(key=lambda t: -t[0])

        # score DIRECTORIES (a dir named 'core' = the package; preferred over a
        # one-line re-export shim but NOT over a substantive file like config.yaml)
        dscored = []
        for base, primary in self._all_dir_names():
            r = sm(None, q, base).ratio()
            if q in base:
                r += 0.3
            if r >= 0.45:
                dscored.append((r + 0.2, base, primary))
        dscored.sort(key=lambda t: -t[0])

        # Decide: prefer the package DIR only when the best FILE is a shim
        # (tiny re-export) and the dir matches comparably.
        if dscored:
            best_f = fscored[0] if fscored else None
            best_d = dscored[0]
            file_is_shim = best_f is not None and os.path.getsize(best_f[2]) < 2048
            if file_is_shim and best_d[0] >= (best_f[0] - 0.1):
                if len(dscored) == 1 or dscored[0][0] - dscored[1][0] >= 0.18:
                    return best_d[2], []
                cands = []
                for r, base, full in dscored[:3]:
                    rel = os.path.relpath(full, self.config.get('repo_dir', '/'))
                    cands.append(f"{base}  ({rel})")
                return None, cands

        # otherwise file-based resolution
        if not fscored:
            return None, []
        if len(fscored) == 1 or fscored[0][0] - fscored[1][0] >= 0.18:
            return fscored[0][2], []
        cands = []
        for r, base, full in fscored[:3]:
            rel = os.path.relpath(full, self.config.get('repo_dir', '/'))
            cands.append(f"{base}  ({rel})")
        return None, cands

    def _build_codebase_snapshot(self) -> str:
        """Compact, REAL overview of the project codebase (cached 60s).

        Used to ground the agent — it should never claim it lacks context, and
        this snapshot is the hard guarantee that real content is in front of it.
        """
        now = time.time()
        if self._snap_cache_ts is not None and now - self._snap_cache_ts < 60:
            return self._snap_cache
        root = self.config.get('repo_dir', '')
        lines = [f"## CODEBASE OVERVIEW ({root})", ""]
        try:
            entries = sorted(os.listdir(root))
            dirs = [e for e in entries
                    if os.path.isdir(os.path.join(root, e)) and not e.startswith('.')]
            files = [e for e in entries
                     if os.path.isfile(os.path.join(root, e)) and not e.startswith('.')]
            lines.append("Directories: " + ", ".join(dirs[:50]))
            lines.append("Top files: " + ", ".join(files[:50]))
            lines.append("")
            for rel in ("README.md", "main.py", "pipeline.py", "execution/core.py"):
                p = os.path.join(root, rel)
                if os.path.isfile(p):
                    try:
                        with open(p, "r", errors="replace") as f:
                            head = "".join(f.readlines()[:15]).strip()
                        lines.append(f"### {rel} (first lines)\n{head[:600]}")
                    except Exception:
                        pass
        except Exception as e:
            lines.append(f"(snapshot error: {e})")
        text = "\n".join(lines)[:6000]
        self._snap_cache = text
        self._snap_cache_ts = now
        return text

    def _read_lines(self, path: str, start: int = 1, end: int = None) -> Tuple[str, int, int, int]:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        total = len(lines)
        start = max(1, start)
        end = min(total, end) if end else total
        if start > end:
            start = end
        return "".join(lines[start - 1:end]), start, end, total

    @staticmethod
    def _validate_syntax(path: str, lines: List[str]) -> Optional[str]:
        """Validate edited content; return an error string or None."""
        ext = os.path.splitext(path)[1].lower()
        text = "".join(lines)
        try:
            if ext == ".py":
                import py_compile
                fd, tmp = tempfile.mkstemp(suffix=".py")
                try:
                    with os.fdopen(fd, "w") as f:
                        f.write(text)
                    py_compile.compile(tmp, doraise=True)
                finally:
                    os.unlink(tmp)
            elif ext == ".json":
                json.loads(text)
            elif ext in (".yaml", ".yml"):
                yaml.safe_load(text)
            else:
                return None
        except Exception as e:
            return str(e)
        return None

    @staticmethod
    def _parse_step_arg(text: str) -> Optional[int]:
        """Extract a step number from a user arg; None if not a clean number."""
        digits = re.sub(r'[^\d]', '', text or '')
        if not digits:
            return None
        n = int(digits)
        return n if 1 <= n <= 661 else None

    def _grep(self, pattern: str, roots: List[str] = None) -> str:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"(invalid pattern: {e})"
        roots = roots or self._allowed_roots()
        hits = []
        seen = set()
        for root in roots:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in self._SKIP_DIRS]
                for fn in filenames:
                    full = os.path.join(dirpath, fn)
                    if any(d in full for d in self._SKIP_DIRS):
                        continue
                    if full in seen:
                        continue
                    seen.add(full)
                    try:
                        if os.path.getsize(full) > 2 * 1024 * 1024:
                            continue
                        with open(full, "r", errors="replace") as f:
                            for lineno, line in enumerate(f, 1):
                                if rx.search(line):
                                    hits.append(f"{full}:{lineno}: {line.rstrip()[:180]}")
                                    if len(hits) >= MAX_GREP_LINES:
                                        return "\n".join(hits)
                    except Exception:
                        continue
        return "\n".join(hits) or "(no matches)"

    def _tree(self, path: str, depth: int = 2) -> str:
        lines = []
        def walk(d, prefix, level):
            if level > depth:
                return
            try:
                entries = sorted(os.listdir(d))
            except Exception:
                return
            for e in entries:
                if e in self._SKIP_DIRS:
                    continue
                full = os.path.join(d, e)
                if os.path.isdir(full):
                    lines.append(f"{prefix}{e}/")
                    walk(full, prefix + "  ", level + 1)
                else:
                    lines.append(f"{prefix}{e}")
        walk(path, "", 0)
        return "\n".join(lines[:200]) or "(empty)"

    def _system_stats(self) -> str:
        out = []
        try:
            meminfo = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    meminfo[k.strip()] = int(v.strip().split()[0]) // 1024
            out.append(f"RAM: {meminfo.get('MemAvailable', 0)}/{meminfo.get('MemTotal', 0)} MB avail")
        except Exception:
            pass
        try:
            for path in ("/", self.config.get('work_dir', '/tmp')):
                du = shutil.disk_usage(path)
                out.append(f"Disk {path}: {du.free / 1024**3:.1f}/{du.total / 1024**3:.1f} GB free")
        except Exception:
            pass
        try:
            load = os.getloadavg()
            out.append(f"Load: {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}")
        except Exception:
            pass
        return "\n".join(out)

    def _proc_running(self, pattern: str) -> bool:
        try:
            r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, timeout=10)
            return bool(r.stdout.strip())
        except Exception:
            return False

    def _port_open(self, port: int) -> bool:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1.0)
        try:
            s.connect(("127.0.0.1", port))
            return True
        except Exception:
            return False
        finally:
            s.close()

    async def _kill_process(self, pattern: str):
        try:
            proc = await asyncio.create_subprocess_exec("pkill", "-TERM", "-f", pattern)
            await proc.wait()
        except Exception:
            pass

    def _plan_path(self) -> Optional[str]:
        audits = self.config.get('work_dir', '/tmp')
        try:
            files = [f for f in os.listdir(audits) if re.match(r'master_plan_.*\.md$', f)]
            if not files:
                return None
            return os.path.join(audits, sorted(files, key=lambda f: os.path.getmtime(os.path.join(audits, f)))[-1])
        except Exception:
            return None

    def _exec_state(self) -> Dict:
        audits = self.config.get('work_dir', '/tmp')
        st = {"completed": set(), "failed": []}
        try:
            with open(os.path.join(audits, 'plan_execution_state.json')) as f:
                d = json.load(f)
            st["completed"] = set(d.get('completed_steps', []))
            st["failed"] = list(d.get('failed_steps', []))
        except Exception:
            pass
        return st

    def _step_status(self, n: int, st: Dict) -> str:
        if n in st["completed"]:
            return "✅ done"
        fails = [x for x in st["failed"] if x == n]
        if fails:
            return f"❌ failed ×{len(fails)}"
        return "⏳ pending"

    # ── system / info commands ───────────────────────────────────────────────
    async def cmd_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        checks = [
            ("Orchestrator", self._proc_running("run_workflow.py")),
            ("Engine", self._proc_running("execute_master_plan.py")),
            ("OmniRoute", self._port_open(20128)),
        ]
        lines = ["💚 **Health**"]
        for name, ok in checks:
            lines.append(f"  {name}: {'✅ up' if ok else '❌ down'}")
        lines.append(self._system_stats())
        try:
            r = subprocess.run(["git", "-C", self.config.get('git_repo_root', self.config.get('repo_dir')), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=20)
            dirty = [l for l in r.stdout.strip().splitlines() if l.strip()]
            lines.append(f"Git: {len(dirty)} uncommitted file(s)")
        except Exception:
            pass
        await self.send_message("\n".join(lines))

    async def cmd_version(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        repo = self.config.get('git_repo_root', self.config.get('repo_dir'))
        gitlog = ""
        try:
            r = subprocess.run(["git", "-C", repo, "log", "-1", "--oneline"], capture_output=True, text=True, timeout=20)
            gitlog = r.stdout.strip()
        except Exception:
            pass
        await self.send_message(
            f"ℹ️ **Version**\nGit: {gitlog or 'n/a'}\n"
            f"Orchestrator: {'running' if self._proc_running('run_workflow.py') else 'down'}\n"
            f"Engine: {'running' if self._proc_running('execute_master_plan.py') else 'down'}\n"
            f"OmniRoute (20128): {'up' if self._port_open(20128) else 'down'}\n"
            f"Python: {sys.version.split()[0]}"
        )

    async def cmd_summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        prog = self._plan_progress()
        st = self.state.state
        s = self.settings.get()
        speed, eta = self._speed_eta()
        await self.send_message(
            f"📊 **System Summary**\n"
            f"Cycle {st['cycle']} | Phase {st['phase']} | Running {st['running']} | Paused {st['paused']} | Halted {st.get('halted')}\n"
            f"Plan: {prog['completed']}/{prog['total']} steps | Next: {prog['current'] or 'DONE 🎉'}\n"
            f"Unresolved fails: {prog['unresolved_fails'] or 'none'}\n"
            f"Speed: {speed} | ETA: {eta}\n"
            f"LLM: {self.llm.model} ({self.llm.model_tier}) | Effort {self.llm.effort_default} | Ctx {self.llm.context_budget:,}\n"
            f"Settings: loop={s.get('loop_mode')} pause_after={s.get('pause_after')} notify={s.get('notifications')}\n"
            f"{self._system_stats()}\n"
            f"Last error: {st.get('last_error') or 'none'}"
        )

    async def cmd_context(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        h = self.llm.history
        toks = self.llm._history_tokens(h)
        await self.send_message(
            f"🧠 **LLM Context**\n"
            f"Budget: {self.llm.context_budget:,} tokens (default {ROLLING_CONTEXT_TOKENS:,})\n"
            f"Current turns: {len(h)} (~{toks:,} tokens)\n"
            f"Model: {self.llm.model} ({self.llm.model_tier})\n"
            f"Effort default: {self.llm.effort_default}\n"
            f"Last tier event: {self.llm.last_tier_event or 'none'}\n"
            f"Last ctx event: {self.llm.last_context_event or 'none'}"
        )

    async def cmd_validate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.send_message("🔍 Running lightweight validation (compile-check + git status)...")
        errs = []
        for s in (self.config.get('scripts') or {}).values():
            if os.path.exists(s):
                r = subprocess.run(["python3", "-m", "py_compile", s], capture_output=True, text=True, timeout=60)
                if r.returncode != 0:
                    errs.append(f"{os.path.basename(s)}: {r.stderr.strip()[-300:]}")
        try:
            r = subprocess.run(["git", "-C", self.config.get('git_repo_root', self.config.get('repo_dir')), "status", "--short"],
                               capture_output=True, text=True, timeout=30)
            dirty = r.stdout.strip()[:1500] or "(clean)"
        except Exception:
            dirty = "(git check failed)"
        status = "OK" if not errs else "FAILED"
        await self.send_message(f"✅ **Validation** ({status})\n```\n{chr(10).join(errs) if errs else 'compileall: ok'}\n\ngit:\n{dirty}\n```")

    async def cmd_test(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        repo = self.config.get('git_repo_root', self.config.get('repo_dir'))
        await self.send_message("🧪 Running pytest (can take minutes — tail streamed when done)...")
        try:
            r = subprocess.run(["python3", "-m", "pytest", "-q", "--timeout=120"], cwd=repo,
                               capture_output=True, text=True, timeout=1800)
            tail = ((r.stdout or "")[-1600:] + "\n" + (r.stderr or "")[-500:]).strip()
            await self.send_message(f"✅ **pytest** rc={r.returncode}\n```\n{tail}\n```")
        except Exception as e:
            await self.send_message(f"❌ pytest error: {e}")

    async def cmd_audit(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.send_message("🕵️ Mini-audit running...")
        resp = await self.llm.direct_query(
            "Quick health assessment of the Project system. Using the live context, list the top 3 risks or issues right now, one line each, then a one-line verdict.",
            context_snapshot=self._build_context()
        )
        await self._reply_llm(resp)

    async def cmd_deploy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        repo = self.config.get('git_repo_root', self.config.get('repo_dir'))
        await self.send_message("🚀 Deploying: git push + verify...")
        try:
            r = subprocess.run(["git", "-C", repo, "push"], capture_output=True, text=True, timeout=120)
            if r.returncode != 0 and "everything up-to-date" not in r.stderr.lower():
                await self.send_message(f"❌ Push failed:\n```\n{r.stderr[-800:]}\n```")
                return
            await self.send_message(
                f"✅ Pushed. Orchestrator: {'RUNNING' if self._proc_running('run_workflow.py') else 'DOWN'}."
            )
        except Exception as e:
            await self.send_message(f"❌ {e}")

    async def cmd_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        repo = self.config.get('git_repo_root', self.config.get('repo_dir'))
        await self.send_message("🔄 Fetching updates...")
        try:
            subprocess.run(["git", "-C", repo, "fetch"], capture_output=True, text=True, timeout=60)
            r2 = subprocess.run(["git", "-C", repo, "status", "-sb"], capture_output=True, text=True, timeout=30)
            if "behind" in r2.stdout:
                r3 = subprocess.run(["git", "-C", repo, "pull", "--ff-only"], capture_output=True, text=True, timeout=120)
                if r3.returncode != 0:
                    await self.send_message(f"❌ Fast-forward pull failed:\n```\n{r3.stderr[-800:]}\n```")
                    return
                await self.send_message(f"✅ Updated.\n```\n{(r3.stdout or '')[-800:]}\n```")
            else:
                await self.send_message(f"✅ Already up to date.\n{r2.stdout.strip()}")
        except Exception as e:
            await self.send_message(f"❌ {e}")

    # ── execution / control commands ─────────────────────────────────────────
    async def cmd_halt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.state.update(running=False, paused=True, halted=True)
        await self.send_message("🛑 **EMERGENCY HALT** — orchestrator stopped. Engine keeps its current step batch. Send /continue to resume.")

    async def cmd_kill(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.send_message("💥 KILL: terminating engine + subagents...")
        self.state.update(running=False, paused=True, halted=True)
        await self._kill_process("execute_master_plan.py")
        await asyncio.sleep(2)
        alive = self._proc_running("execute_master_plan.py")
        await self.send_message(f"Engine: {'still running' if alive else 'terminated'}. Use /continue to relaunch.")

    async def cmd_continue(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.state.update(paused=False, halted=False)
        if self.orchestrator and self.orchestrator.running:
            await self.send_message("▶️ Workflow resumed.")
        else:
            self._want_restart = True
            await self.send_message("▶️ Resuming orchestrator loop...")

    async def cmd_restart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.send_message("♻️ Restarting orchestrator in ~2s...")
        self.state.update(paused=False, halted=False)
        if self.orchestrator:
            self._want_restart = True
            self.orchestrator.running = False
        else:
            await self.send_message("❌ No orchestrator attached.")

    async def cmd_cycle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: /cycle <n>")
            return
        try:
            n = int(context.args[0])
        except ValueError:
            await self.send_message("❌ Cycle must be a number.")
            return
        if n < 1:
            await self.send_message("❌ Cycle must be >= 1.")
            return
        self.state.update(cycle=n - 1)
        await self.send_message(f"✅ Next cycle will be **#{n}**.")

    async def cmd_schedule(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: `/schedule HH:MM` or `/schedule in <n> min|h`")
            return
        text = ' '.join(context.args)
        ts = None
        try:
            m = re.match(r'(\d{1,2}):(\d{2})', text)
            if m:
                now = datetime.now()
                ts = now.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=0, microsecond=0)
                if ts <= now:
                    ts += timedelta(days=1)  # overflow-safe vs ts.replace(day=...)
                ts = ts.timestamp()
            else:
                m = re.match(r'in\s+(\d+)\s*(min|m|hour|h|sec|s)?', text, re.IGNORECASE)
                if m:
                    n = int(m.group(1))
                    unit = (m.group(2) or "min").lower()
                    mult = {"sec": 1, "s": 1, "min": 60, "m": 60, "hour": 3600, "h": 3600}[unit]
                    ts = time.time() + n * mult
        except ValueError:
            ts = None
        if ts is None:
            await self.send_message("❌ Could not parse that time. Try `14:30` or `in 90 min`.")
            return
        self.state.update(scheduled_pause=ts)
        when = datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")
        await self.send_message(f"⏰ Scheduled pause at {when}.")

    async def cmd_priority(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args or context.args[0].lower() not in ("high", "medium", "low"):
            await self.send_message("❌ Usage: /priority high|medium|low")
            return
        self.state.update(current_priority=context.args[0].lower())
        await self.send_message(f"✅ Priority set to `{context.args[0].lower()}` (informational — guides the LLM).")

    async def cmd_batch(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: /batch <start>-<end>, e.g. /batch 597-610")
            return
        m = re.match(r'(\d+)-(\d+)$', context.args[0])
        if not m:
            await self.send_message("❌ Expected a range like `597-610`.")
            return
        a, b = int(m.group(1)), int(m.group(2))
        if not (1 <= a <= b <= 661):
            await self.send_message("❌ Range must be within 1-661.")
            return
        self.state.update(batch_range=[a, b])
        await self.send_message(f"🎯 Batch focus {a}-{b} recorded (engine still runs all remaining steps sequentially).")

    async def cmd_retry(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: /retry <step>")
            return
        n = self._parse_step_arg(context.args[0])
        if n is None:
            await self.send_message(f"❌ `{context.args[0]}` isn't a valid step number (1-661).")
            return
        audits = self.config.get('work_dir', '/tmp')
        path = os.path.join(audits, 'plan_execution_state.json')
        if not os.path.exists(path):
            await self.send_message("❌ No execution state yet.")
            return
        with open(path) as f:
            d = json.load(f)
        comp = list(d.get('completed_steps', []))
        if n in comp:
            comp.remove(n)
        d['completed_steps'] = comp
        d['retry_requested'] = list(d.get('retry_requested', [])) + [n]
        tmp = path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, path)
        await self.send_message(f"🔄 Step {n} un-marked — the engine will re-run it through the full escalation ladder on its next pass.")

    async def cmd_skip(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: `/skip <step> confirm` — note: this violates the never-skip HARD RULE and is only allowed as an explicit human override.")
            return
        n = self._parse_step_arg(context.args[0])
        if n is None:
            await self.send_message(f"❌ `{context.args[0]}` isn't a valid step number (1-661).")
            return
        if 'confirm' not in ' '.join(context.args[1:]).lower():
            await self.send_message(f"⚠️ This will FORCE-mark step {n} as done, violating the never-skip rule. Re-send as `/skip {n} confirm` to override.")
            return
        audits = self.config.get('work_dir', '/tmp')
        path = os.path.join(audits, 'plan_execution_state.json')
        if not os.path.exists(path):
            await self.send_message("❌ No execution state yet.")
            return
        with open(path) as f:
            d = json.load(f)
        comp = list(d.get('completed_steps', []))
        if n not in comp:
            comp.append(n)
        d['completed_steps'] = comp
        d['user_skipped'] = list(d.get('user_skipped', [])) + [n]
        tmp = path + ".tmp"
        with open(tmp, 'w') as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, path)
        await self.send_message(f"⚠️ Step {n} force-marked done at your command. Recorded in `user_skipped`.")

    async def cmd_escalate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: /escalate <step>")
            return
        n = self._parse_step_arg(context.args[0])
        if n is None:
            await self.send_message(f"❌ `{context.args[0]}` isn't a valid step number (1-661).")
            return
        if self.llm.model_tier != 'pro':
            self.llm.model = MODEL_PRO
            self.llm.model_tier = 'pro'
            self.llm._save_state()
            await self.send_message(f"🔺 Conversational agent forced to reasoning tier.")
        await self.cmd_retry(update, context)
        await self.send_message(f"🔺 Step {n} will re-run through the full escalation ladder (OmniRoute → Flash → Pro).")

    async def cmd_error(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        st = self.state.state
        errs = []
        log_file = self.config.get('log_file', '/tmp/orchestrator.log')
        if os.path.exists(log_file):
            with open(log_file, 'r', errors='replace') as f:
                for line in f:
                    if 'ERROR' in line or 'Traceback' in line:
                        errs.append(line.rstrip())
        msg = f"❌ **Last error**: {st.get('last_error') or 'none'}"
        if errs:
            msg += f"\n\nRecent errors in log:\n```\n{chr(10).join(errs[-10:])[-3800:]}\n```"
        await self.send_message(msg)

    async def cmd_errors(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        st = self._exec_state()
        unresolved = sorted({x for x in st["failed"] if x not in st["completed"]})
        await self.send_message(
            f"⚠️ **Failed steps**\n"
            f"Unresolved: {unresolved if unresolved else 'none'}\n"
            f"Total failure records (incl. retries that later succeeded): {len(st['failed'])}"
        )

    async def cmd_trace(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: /trace <step>")
            return
        n = context.args[0]
        audits = self.config.get('work_dir', '/tmp')
        log_path = os.path.join(audits, 'plan_execution.log')
        lines = []
        if os.path.exists(log_path):
            with open(log_path, 'r', errors='replace') as f:
                for line in f:
                    if f'Step {n}/' in line or f'STEP {n}' in line:
                        lines.append(line.rstrip())
        if not lines:
            await self.send_message(f"🔍 No log entries mentioning step {n}.")
            return
        await self.send_message(f"🔍 **Trace step {n}** ({len(lines)} lines)\n```\n{chr(10).join(lines[-25:])[-3800:]}\n```")

    async def cmd_tail(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/tail <file> [n]  →  file tail.   /tail  →  live engine log tail."""
        if context.args:
            path = self._resolve_path(context.args[0])
            if path and os.path.isfile(path):
                n = int(context.args[1]) if len(context.args) > 1 else 20
                n = min(max(1, n), 200)
                content, start, end, total = self._tail_file(path, n)
                await self.send_message(f"📄 **{context.args[0]}** (lines {start}-{end} of {total})\n```\n{content}\n```")
                return
        audits = self.config.get('work_dir', '/tmp')
        log_path = os.path.join(audits, 'plan_execution.log')
        if not os.path.exists(log_path):
            await self.send_message("❌ No execution log yet.")
            return
        with open(log_path, 'r', errors='replace') as f:
            lines = f.readlines()
        await self.send_message(f"📡 **Engine log tail** (last 30 of {len(lines)})\n```\n{chr(10).join(lines[-30:])[-3900:]}\n```")

    def _tail_file(self, path: str, n: int) -> Tuple[str, int, int, int]:
        content, start, end, total = self._read_lines(path, 1, None)
        total = max(1, total)
        start = max(1, total - n + 1)
        content, start, end, total = self._read_lines(path, start, None)
        return content, start, end, total

    async def cmd_debug(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if context.args and context.args[0].lower() in ("off", "0"):
            self.state.update(debug=False)
            self.settings.update({"notify_interval": 300})
            await self.send_message("🐞 Debug off.")
        else:
            self.state.update(debug=True)
            self.settings.update({"notify_interval": 60})
            await self.send_message("🐞 Debug mode ON — progress pings every 60s.")

    async def cmd_speed(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        speed, eta = self._speed_eta()
        await self.send_message(f"⚡ Speed: {speed}\nETA: {eta}")

    async def cmd_eta(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        speed, eta = self._speed_eta()
        await self.send_message(f"⏱ ETA: {eta} (at {speed})")

    async def cmd_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        speed, eta = self._speed_eta()
        st = self._exec_state()
        unresolved = sorted({x for x in st["failed"] if x not in st["completed"]})
        await self.send_message(
            f"⚡ **Performance**\nSpeed: {speed}\nETA: {eta}\n"
            f"Steps done: {len(st['completed'])}/661\nUnresolved fails: {unresolved or 'none'}"
        )

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.send_message(f"📈 **System stats**\n{self._system_stats()}")

    # ── file ops commands ────────────────────────────────────────────────────
    async def cmd_head(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: /head <file> [n]")
            return
        path = self._resolve_path(context.args[0])
        if not path or not os.path.isfile(path):
            await self.send_message("❌ File not found / not allowed.")
            return
        n = min(int(context.args[1]) if len(context.args) > 1 else 20, 200)
        content, start, end, total = self._read_lines(path, 1, n)
        await self.send_message(f"📄 **{context.args[0]}** (lines 1-{end} of {total})\n```\n{content}\n```")

    async def cmd_ls(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        p = context.args[0] if context.args else "."
        path = self._resolve_path(p)
        if not path or not os.path.isdir(path):
            await self.send_message("❌ Not a directory.")
            return
        entries = sorted(os.listdir(path))
        lines = [f"📁 **{path}** ({len(entries)} entries)"]
        for e in entries[:100]:
            full = os.path.join(path, e)
            lines.append(f"  {e}{'/' if os.path.isdir(full) else ''}")
        await self.send_message("\n".join(lines))

    async def cmd_tree(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        p = context.args[0] if context.args else "."
        try:
            depth = min(int(context.args[1]) if len(context.args) > 1 else 2, 4)
        except ValueError:
            depth = 2
        path = self._resolve_path(p)
        if not path or not os.path.isdir(path):
            await self.send_message("❌ Not a directory.")
            return
        await self.send_message(f"🌳 **{path}**\n```\n{self._tree(path, depth)}\n```")

    async def cmd_find(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: /find <name-substring> [dir]")
            return
        needle = context.args[0]
        root = self._resolve_path(context.args[1]) if len(context.args) > 1 else None
        roots = [root] if root else self._allowed_roots()
        hits = []
        for base in roots:
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in self._SKIP_DIRS]
                for fn in filenames:
                    if needle.lower() in fn.lower():
                        hits.append(os.path.join(dirpath, fn))
                        if len(hits) >= 60:
                            break
                if len(hits) >= 60:
                    break
            if len(hits) >= 60:
                break
        if not hits:
            await self.send_message(f"🔍 No files matching `{needle}`.")
            return
        await self.send_message(f"🔍 **{needle}** ({len(hits)}):\n{chr(10).join(hits[:60])}")

    async def cmd_grep(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: /grep <pattern> [dir]")
            return
        pattern = context.args[0]
        root = self._resolve_path(context.args[1]) if len(context.args) > 1 else None
        roots = [root] if root else self._allowed_roots()
        out = self._grep(pattern, roots)
        await self.send_message(f"🔎 `{pattern}`\n```\n{out[:3800]}\n```")

    async def cmd_diff(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        repo = self.config.get('git_repo_root', self.config.get('repo_dir'))
        files = [self._resolve_path(a) for a in context.args]
        cmd = ["git", "-C", repo, "diff"]
        if files:
            cmd += ["--"] + [f for f in files if f]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            out = r.stdout.strip()[:3800] or "(clean working tree)"
            await self.send_message(f"📝 **Git diff**\n```\n{out}\n```")
        except Exception as e:
            await self.send_message(f"❌ {e}")

    async def cmd_save(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = ' '.join(context.args) or "Save via Telegram /save"
        if self.git.commit(message):
            await self.send_message(f"✅ Saved & committed: {message}")
        else:
            await self.send_message("❌ Commit failed (nothing to commit or git error).")

    async def cmd_load(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: /load <file>")
            return
        path = self._resolve_path(context.args[0])
        if not path or not os.path.isfile(path):
            await self.send_message("❌ File not found / not allowed.")
            return
        content, _, _, total = self._read_lines(path, 1, None)
        snippet = content[:MAX_VIEW_BYTES]
        msg = self.llm.inject_context(path, snippet)
        await self.send_message(f"📥 {msg} ({total} lines in file, first {MAX_VIEW_BYTES // 1024}KB loaded).")

    async def cmd_clear_context(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = self.llm.clear_context()
        await self.send_message(f"🧹 {msg}")

    async def cmd_expand_context(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/expand_context <n> — raise the rolling context budget (75k-200k)."""
        if not context.args:
            await self.send_message(f"🧠 Current context budget: **{self.llm.context_budget:,}** tokens. Usage: `/expand_context <tokens>`")
            return
        try:
            n = int(context.args[0])
        except ValueError:
            await self.send_message("❌ Expected a number of tokens, e.g. `/expand_context 120000`.")
            return
        new = max(self.llm.context_budget, min(MODEL_MAX_CONTEXT_TOKENS, n))
        self.llm.context_budget = new
        self.llm._save_state()
        await self.send_message(f"🧠 Context budget expanded to **{new:,}** tokens. (Max {MODEL_MAX_CONTEXT_TOKENS:,}.)")

    async def cmd_decrease_context(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/decrease_context <n> — lower the rolling context budget (16k-75k)."""
        if not context.args:
            await self.send_message(f"🧠 Current context budget: **{self.llm.context_budget:,}** tokens. Usage: `/decrease_context <tokens>`")
            return
        try:
            n = int(context.args[0])
        except ValueError:
            await self.send_message("❌ Expected a number of tokens, e.g. `/decrease_context 40000`.")
            return
        new = min(self.llm.context_budget, max(16000, n))
        self.llm.context_budget = new
        self.llm._save_state()
        await self.send_message(f"🧠 Context budget decreased to **{new:,}** tokens.")

    async def cmd_compact_context(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/compact_context — reset budget to the 75k default and clear loaded history."""
        self.llm.context_budget = ROLLING_CONTEXT_TOKENS
        n = len(self.llm.history)
        self.llm.history = []
        self.llm._save_state()
        self.llm._save_history()
        await self.send_message(f"🧹 Context compacted: budget back to **{ROLLING_CONTEXT_TOKENS:,}** tokens, {n} conversation turns cleared. Fundamentals stay loaded.")

    # ── project context commands ─────────────────────────────────────────────
    async def cmd_explain(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: /explain <file|module> [in <N> words]")
            return
        target = context.args[0]
        rest = ' '.join(context.args[1:])
        wlim = None
        m = re.search(r'in\s+(\d+)\s+words?', rest)
        if m:
            wlim = int(m.group(1))
        path = self._resolve_path(target)
        if not path or not os.path.isfile(path):
            # module-style names: "project.core.ga_controller" -> path under repo_dir
            mod = target.replace('.', '/')
            for cand in (f"{mod}.py", os.path.join("project", mod + ".py")):
                p = self._resolve_path(cand)
                if p and os.path.isfile(p):
                    path = p
                    break
        fuzzy_note = ""
        if not path or not os.path.isfile(path):
            # fuzzy / typo'd names: "borepy" -> core.py, "config" -> config.yaml
            fuzzy, cands = self._resolve_fuzzy_file(target)
            if cands:
                await self.send_message(
                    f"🤔 Not sure which file you mean. Did you mean one of:\n" +
                    "\n".join(f"  • `{c}`" for c in cands) +
                    "\n\nSend the one you want (e.g. `/explain core.py`)."
                )
                return
            if fuzzy and os.path.isfile(fuzzy):
                path = fuzzy
                fuzzy_note = f" (guessed `{os.path.basename(fuzzy)}`)"
            else:
                await self.send_message(f"❌ Could not find a file like `{target}`.")
                return
        content, _, _, total = self._read_lines(path, 1, None)
        snippet = content[:MAX_VIEW_BYTES]
        wlim_txt = f" Reply in at most {wlim} words." if wlim else ""
        await self.send_message(f"🤖 Explaining `{target}`{fuzzy_note} ({total} lines)...")
        # Load the content as a REAL conversation turn (not just the system
        # snapshot) so the model has it in front of it and cannot answer
        # "I don't have access to that file".
        self.llm.inject_context(
            f"FILE: {os.path.basename(path)} ({path})",
            f"```python\n{snippet}\n```" if path.endswith('.py') else snippet
        )
        resp = await self.llm.direct_query(
            f"Explain the file that was just loaded into your context above "
            f"(FILE: {os.path.basename(path)}, path {path}, {total} lines). "
            f"It is the file you resolved from the user's request '{target}'. "
            f"Do NOT say you cannot access it — its content is already in your context."
            f"{wlim_txt}"
        )
        await self._reply_llm(resp)

    async def cmd_whatis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: /whatis <function>")
            return
        name = context.args[0]
        out = self._grep(rf'\bdef\s+{re.escape(name)}\b')
        if not out or out.startswith("(no matches"):
            await self.send_message(f"❌ No definition found for `{name}`.")
            return
        first = out.splitlines()[0]
        filepath, _, lineno = first.rpartition(":")
        path = self._resolve_path(filepath)
        content = ""
        if path:
            try:
                content, _, _, _ = self._read_lines(path, max(1, int(lineno) - 15), int(lineno) + 40)
            except Exception:
                pass
        await self.send_message(f"🔎 Found at `{first}`. Explaining...")
        resp = await self.llm.direct_query(
            f"Explain function `{name}`: what it does, its inputs/outputs, and any issues.",
            context_snapshot=f"DEF LOCATION: {first}\n```python\n{content}\n```"
        )
        await self._reply_llm(resp)

    async def cmd_step(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        plan_path = self._plan_path()
        steps = parse_plan_steps(plan_path) if plan_path else []
        if not steps:
            await self.send_message("❌ No plan file found.")
            return
        total = len(steps)
        n = None
        if context and context.args:
            m = re.search(r'\d{1,3}', context.args[0])
            n = int(m.group(0)) if m else None
        if n is None:
            st = self._exec_state()
            unresolved = sorted({x for x in st["failed"] if x not in st["completed"]})
            cur = max(st["completed"]) + 1 if st["completed"] else 1
            await self.send_message(
                f"📋 **Plan** ({total} steps) — `{os.path.basename(plan_path)}`\n"
                f"Done: {len(st['completed'])}/{total} | Next: {cur}\n"
                f"Unresolved fails: {unresolved if unresolved else 'none'}"
            )
            return
        step = next((s for s in steps if s["num"] == n), None)
        if not step:
            await self.send_message(f"❌ Step {n} not found (plan has 1-{total}).")
            return
        status = self._step_status(n, self._exec_state())
        body = "".join(step["body"]).strip()[:MAX_VIEW_BYTES]
        await self.send_message(f"**STEP {n}/{total}**: {step['title']}\nStatus: {status}")
        resp = await self.llm.direct_query(
            f"Explain plan step {n} ({step['title']}): what to do, why it matters, and how to verify it.",
            context_snapshot=f"PLAN STEP {n}/{total} — STATUS: {status}\n{step['title']}\n{body}"
        )
        await self._reply_llm(resp)

    async def cmd_plan(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.cmd_step(update, context)

    async def cmd_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await self.send_message("❌ Usage: /search <query>")
            return
        q = ' '.join(context.args)
        hits = self._grep(re.escape(q))
        hits_short = "\n".join(hits.splitlines()[:12])
        await self.send_message(f"🔎 Searching `{q}`...")
        resp = await self.llm.direct_query(
            f"Search result for \"{q}\". Summarize where this appears, what the code does, and anything notable.",
            context_snapshot=f"SEARCH RESULTS:\n```\n{hits_short}\n```"
        )
        await self._reply_llm(resp)

    # ── notification toggles / history ───────────────────────────────────────
    async def cmd_notify_on(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.settings.update({"notifications": "on"})
        await self.send_message("🔔 Notifications ON.")

    async def cmd_notify_off(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.settings.update({"notifications": "off"})
        await self.send_message("🔕 Notifications OFF — only direct replies will be sent.")

    async def cmd_quiet(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.cmd_notify_off(update, context)

    async def cmd_verbose(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await self.cmd_notify_on(update, context)

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        h = self.llm.history
        if not h:
            await self.send_message("(no conversation history)")
            return
        lines = []
        for i, m in enumerate(h[-15:], 1):
            c = (m.get('content') or '').replace('\n', ' ')
            lines.append(f"{i}. [{m.get('role')}] {c[:80]}")
        await self.send_message("🧾 **Recent conversation**\n" + "\n".join(lines))

    # ── Claude summon + message relay ────────────────────────────────────────
    # Inbox/outbox files shared with this Claude session (the "mega" relay):
    #   - /message_claude <text>  -> writes to claude_inbox.json for me to read
    #   - this session writes replies to claude_outbox.json; a background
    #     watcher in the orchestrator forwards them to Telegram.
    RELAY_DIR = os.getenv("WORK_DIR", os.path.join(os.path.expanduser("~"), "orchestrator_data"))
    CLAUDE_INBOX = os.path.join(RELAY_DIR, "claude_inbox.json")
    CLAUDE_OUTBOX = os.path.join(RELAY_DIR, "claude_outbox.json")

    async def cmd_summon_claude(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/summon_claude <issue> — run claude -p to fix an issue, then reply."""
        issue = ' '.join(context.args) if context.args else ""
        if not issue:
            await self.send_message("❌ Usage: /summon_claude <describe the issue>")
            return
        if self.settings.get().get('auto_summon_claude', 'on') != 'on':
            await self.send_message("🔒 auto_summon_claude is off. Enable with /set auto_summon_claude on")
            return
        await self.send_message("🤖 Summoning Claude Code to fix it... (this can take a few minutes)")
        res = call_claude(issue, workdir=self.config.get('repo_dir'))
        if res.get("ok"):
            await self.send_message(f"✅ Claude fix complete (rc=0). Result:\n```\n{res['output'][:3000]}\n```")
        else:
            await self.send_message(f"❌ Claude could not fix it (rc={res['rc']}):\n```\n{res['output'][:2000]}\n```")

    async def cmd_message_claude(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """/Jarvis <text> — relay a message to the active Claude session (aka /message_claude)."""
        text = ' '.join(context.args) if context.args else ""
        if not text:
            await self.send_message("❌ Usage: /Jarvis <your message to Claude>")
            return
        try:
            inbox = []
            if os.path.exists(self.CLAUDE_INBOX):
                with open(self.CLAUDE_INBOX) as f:
                    inbox = json.load(f)
            inbox.append({"ts": datetime.now().isoformat(), "from": "telegram", "text": text, "direct": True})
            tmp = self.CLAUDE_INBOX + ".tmp"
            with open(tmp, "w") as f:
                json.dump(inbox, f, indent=2)
            os.replace(tmp, self.CLAUDE_INBOX)
        except Exception as e:
            await self.send_message(f"❌ Could not relay: {e}")
            return
        await self.send_message("📤 Direct message queued for Claude Code. They'll respond personally.")

    async def _relay_outbox(self):
        """Forward any claude->telegram replies from the outbox to chat.

        RENAME-CLAIM (2026-08-11): the old claim-then-send wrote [] to the
        outbox BEFORE sending; a process death between claim and send
        (orchestrator churn at HALT respawns — 2026-08-11 lost 4 messages
        incl. the completion milestone) permanently dropped the items, and
        `except Exception: pass` swallowed send failures the same way. Now
        the outbox is atomically RENAMED to .pending — exactly one relay
        wins the claim, a dead process leaves the file on disk for the next
        relay to recover, and the file is deleted only after every item's
        send attempt completes. Double-send protection (the 08-05 race fix)
        is unchanged: no other relay can see claimed items.
        """
        pending = self.CLAUDE_OUTBOX + ".pending"
        claimed = pending
        if not os.path.exists(claimed):
            # Recover items a previous process claimed but died before sending.
            if os.path.exists(self.CLAUDE_OUTBOX):
                try:
                    os.rename(self.CLAUDE_OUTBOX, pending)  # atomic claim
                except OSError:
                    return  # another relay won the claim; it owns the file
            else:
                return
        try:
            with open(claimed) as f:
                items = json.load(f)
        except Exception:
            # Malformed / unreadable → hand the file back for diagnosis.
            os.replace(claimed, self.CLAUDE_OUTBOX)
            return
        if not items:
            try:
                os.remove(pending)
            except OSError:
                pass
            return
        try:
            # Send the claimed items. Use PLAIN TEXT (parse_mode=None) so
            # markdown-hostile chars in Claude's replies (—, *, _) never
            # trigger a 400 parse error.
            while items:
                item = items[0]
                try:
                    await self.send_message(
                        item.get('text', ''),
                        parse_mode=None,
                    )
                except Exception:
                    # Definitive send failure (send_message already handles
                    # ambiguous/parse cases internally) — keep the UNSENT
                    # items in .pending for the next relay pass instead of
                    # dropping them silently.
                    with open(pending, "w") as f:
                        json.dump(items, f, indent=2)
                    LOGGER.error(
                        f"[RELAY] send failed for {len(items)} outbox item(s); "
                        "kept in .pending for retry")
                    return
                items = items[1:]
        finally:
            try:
                os.remove(pending)  # all items sent → discard the claim
            except OSError:
                pass

    def _chunk_message(self, text: str, max_length: int = 4096) -> List[str]:
        """Split message into chunks for Telegram."""
        chunks = []
        while text:
            chunks.append(text[:max_length])
            text = text[max_length:]
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

PHASES = ["audit", "cross_eval", "execution", "graphify", "analyze"]


class MasterOrchestrator:
    """Main orchestrator loop (settings-driven)."""

    def __init__(
        self,
        config: Dict,
        state_manager: StateManager,
        task_manager: TaskManager,
        executor: WorkflowExecutor,
        llm: LLMAgent,
        editor: ScriptEditor,
        git: GitManager,
        telegram: TelegramBot,
        settings_manager: SettingsManager,
        initial_phase: str = "audit"
    ):
        self.config = config
        self.state = state_manager
        self.tasks = task_manager
        self.executor = executor
        self.llm = llm
        self.editor = editor
        self.git = git
        self.telegram = telegram
        self.settings = settings_manager
        self.initial_phase = initial_phase if initial_phase in PHASES else "audit"
        self.running = False
        self._last_notify = 0.0
        self.logger = LOGGER  # safety: ensure logger is always available

    async def _notify(self, text: str, settings: Dict):
        """Send a Telegram message, throttled to settings['notify_interval']."""
        if settings.get('notifications', 'on') == 'off':
            return
        interval = max(60, int(settings.get('notify_interval', 300) or 300))
        now = time.time()
        if now - self._last_notify >= interval:
            await self.telegram.send_message(text)
            self._last_notify = now

    async def _notify_user(self, text: str, settings: Dict):
        """Phase start / finish ping — silenced entirely by notifications=off."""
        if settings.get('notifications', 'on') == 'off':
            return
        await self.telegram.send_message(text)

    def _is_fresh_output(self, phase: str, result: str) -> bool:
        """True if a phase produced NEW output (not a stale re-run of the same
        file from a prior cycle). Used to suppress redundant 'started/finished'
        pings for audit/cross-eval/execution on every loop iteration."""
        if phase in ("audit", "cross_eval"):
            key = {"audit": "last_audit_file", "cross_eval": "last_plan_file"}[phase]
            last = self.state.state.get(key)
            return bool(result) and result != last
        if phase == "execution":
            # Fresh ONLY if the plan execution state ADVANCED (more steps
            # completed) since the last time we announced. mtime is useless
            # here — the engine touches its state file even on no-op resumes,
            # so a fixed time window re-announces every cycle.
            audits = self.config.get('work_dir', '/tmp')
            path = os.path.join(audits, 'plan_execution_state.json')
            steps = None
            try:
                with open(path) as f:
                    steps = len(json.load(f).get('completed_steps', []))
            except Exception:
                steps = None
            if steps is not None:
                last = self.state.state.get('last_exec_steps_announced')
                if steps != last:
                    # Advance happened (or first run of this process) — announce
                    # and remember, so subsequent no-op cycles stay silent.
                    self.state.update(last_exec_steps_announced=steps)
                    return True
                return False
            # State file unreadable — announce once per process, then stay quiet.
            if not self.state.state.get('last_exec_announced_at'):
                self.state.update(last_exec_announced_at=time.time())
                return True
            return False
        return True

    async def _notify_phase_progress(self, phase: str, settings: Dict):
        """Send one 50%-done message for a work phase, computed from its state file.

        audit:      work_dir/audit_state.json
        cross_eval: work_dir/cross_eval_state.json
        execution:  work_dir/plan_execution_state.json
        """
        audits = self.config.get('work_dir', '/tmp')
        state_files = {
            "audit": os.path.join(audits, 'audit_state.json'),
            "cross_eval": os.path.join(audits, 'cross_eval_state.json'),
            "execution": os.path.join(audits, 'plan_execution_state.json'),
        }
        path = state_files.get(phase)
        done = total = None
        try:
            if path and os.path.exists(path):
                with open(path) as f:
                    st = json.load(f)
                if phase == "execution":
                    if isinstance(st, dict):
                        completed = st.get('completed_steps', [])
                    else:
                        completed = st or []
                    done, total = len(completed), 661
                elif isinstance(st, dict):
                    # audit_state.json has done_count; cross_eval has outputs list
                    done = st.get('done_count') or len(st.get('outputs', [])) or None
                    total = st.get('total_count') or None
                else:
                    # cross_eval_state.json is a LIST of 150 subagent results
                    done = len(st)
                    total = max(done, 1)
        except Exception as e:
            LOGGER.warning(f"Could not read progress for {phase}: {e}")

        if done is None or not total:
            return  # no reliable progress signal; skip the 50% message

        pct = 100.0 * done / total
        if pct >= 50.0 and not self.state.state.get(f'_notified_50_{phase}'):
            label = {
                "audit": "🕵️ Audit",
                "cross_eval": "🧪 Cross-Eval",
                "execution": "⚙️ Execution",
            }[phase]
            if settings.get('notifications', 'on') != 'off':
                await self.telegram.send_message(f"📊 **{label}** is ~{int(pct)}% done ({done}/{total}).")
            self.state.update(**{f'_notified_50_{phase}': True})

    # ── self-modification branch isolation ──────────────────────────────────
    _SELF_RUN_SCRIPTS = (
        "parallel_agents.py", "parallel_agent_cross_eval.py",
        "execute_master_plan.py", "run_workflow.py",
        "start_orchestrator.sh", "config_workflow.yaml",
    )

    def _plan_is_selfmod(self, plan_path: Optional[str]) -> bool:
        """True if the plan targets the scripts that run the pipeline itself."""
        if not plan_path or not os.path.exists(plan_path):
            return False
        try:
            with open(plan_path, "r", errors="replace") as f:
                text = f.read()
        except Exception:
            return False
        low = text.lower()
        return any(s in low for s in self._SELF_RUN_SCRIPTS)

    async def _maybe_start_selfmod(self, settings: Dict, cycle_num: int) -> bool:
        """If selfmod_branching is on and the plan touches self-run scripts,
        ensure a dedicated branch is checked out (idempotent across restarts).
        Returns True when a selfmod branch is active."""
        if settings.get('selfmod_branching', 'on') != 'on':
            return False
        plan = self.state.state.get('last_plan_file')
        if not self._plan_is_selfmod(plan):
            # a selfmod branch left pending from a previously halted cycle
            # (execution failed, plan regenerated non-selfmod): merge it now.
            if self.state.state.get('selfmod_branch'):
                await self._maybe_finish_selfmod(settings)
            return False
        branch = self.state.state.get('selfmod_branch')
        cur = self.git.current_branch()
        if branch and cur == branch:
            return True
        name = branch or f"selfmod/cycle-{cycle_num}"
        if not branch:
            if not self.git.create_branch(name):
                await self.telegram.send_message(f"⚠️ Could not create selfmod branch `{name}` — continuing on current branch.")
                return False
            self.state.update(selfmod_branch=name)
        elif cur != name:
            if not self.git.checkout(name):
                await self.telegram.send_message(f"⚠️ Could not checkout selfmod branch `{name}`.")
                return False
        await self.telegram.send_message(
            f"🌿 Self-modification detected in the plan — executing on branch `{name}`. "
            f"Commits will merge back to main when the cycle completes."
        )
        return True

    async def _maybe_finish_selfmod(self, settings: Dict):
        """After a successful execution, merge the selfmod branch back to main."""
        branch = self.state.state.get('selfmod_branch')
        if not branch:
            return
        self.state.update(selfmod_branch=None)
        ok = self.git.merge_branch(branch)
        self.git.delete_branch(branch)
        if ok:
            await self.telegram.send_message(f"🌿 Merged `{branch}` back into **main** and pushed. Self-modifications live — next cycle runs on the new scripts.")
        else:
            await self.telegram.send_message(f"⚠️ Could not auto-merge `{branch}` into main (conflict?). Review and merge manually.")

    # ── README auto-update (after each successful execution cycle) ──────────
    # The audit + cross-eval use docs/project_context.md as the
    # "documented features" list. As execution creates/changes files, the
    # readme goes stale — so after every successful execution phase we refresh
    # the entries for new/changed files and commit it.
    README_REL = os.path.join("docs", "project_context.md")
    README_EXT = ('.py', '.dart', '.yaml', '.jinja2', '.html', '.json', '.ini', '.md')

    def _readme_path(self) -> str:
        return os.path.join(self.config.get('repo_dir', ''), self.README_REL)

    def _readme_entries(self, content: str) -> Dict[str, str]:
        """path -> full entry block for every '#N. /...' entry."""
        entries = {}
        blocks = re.split(r'(?m)^(?=#\d+\.\s+/)', content)
        for b in blocks:
            m = re.match(r'#\d+\.\s+(/\S+)', b)
            if m:
                entries[m.group(1)] = b
        return entries

    async def _update_readme(self, settings: Dict = None):
        """Refresh readme entries for repo files created/changed since the
        readme was last written. No-op if disabled, nothing changed, or readme
        missing."""
        if (settings or {}).get('readme_autoupdate', 'on') != 'on':
            return
        readme = self._readme_path()
        if not os.path.exists(readme):
            return
        project = self.config.get('repo_dir', '')
        if not project:
            return
        try:
            mtime = os.path.getmtime(readme)
            with open(readme, "r", errors="replace") as f:
                content = f.read()
        except Exception as e:
            LOGGER.error(f"readme read failed: {e}")
            return
        entries = self._readme_entries(content)

        # walk the repo for candidate files
        cands = []
        for root, dirs, files in os.walk(project):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('.')]
            for fn in files:
                if not fn.endswith(self.README_EXT):
                    continue
                full = os.path.join(root, fn)
                if os.path.abspath(full) == os.path.abspath(readme):
                    continue  # never document the readme itself
                if any(x in full for x in ('legacy', 'env', '.venv', 'node_modules', 'site-packages')):
                    continue
                rel = os.path.relpath(full, project)
                key = f"{REPO_DIR}/{rel}"
                cands.append((key, full))

        new_files = [(k, f) for k, f in cands if k not in entries]
        changed = [(k, f) for k, f in cands if k in entries and os.path.getmtime(f) > mtime]
        to_gen = (new_files + changed)[:12]
        if not to_gen:
            return

        # gather snippets
        metas = []
        for k, full in to_gen:
            try:
                with open(full, "r", errors="replace") as f:
                    snippet = f.read()[:700]
            except Exception:
                snippet = ""
            metas.append({"path": os.path.relpath(full, project), "snippet": snippet})

        entries_text = await self.llm.generate_readme_entries(metas)
        if not entries_text:
            LOGGER.warning("readme entry generation returned nothing")
            return

        # renumber to continue after the current max, then insert before the
        # Technical Summary (or append at the end if absent)
        nums = [int(x) for x in re.findall(r'#(\d+)\.', content)]
        base = max(nums) if nums else 0
        lines = entries_text.splitlines()
        num = base
        rebuilt = []
        for ln in lines:
            m = re.match(r'^#\d+\.\s+(.*)$', ln)
            if m:
                num += 1
                rebuilt.append(f"#{num}. {m.group(1)}")
            else:
                rebuilt.append(ln)
        new_section = "\n".join(rebuilt).strip() + "\n\n"

        idx = content.find("## Technical Summary")
        if idx != -1:
            new_content = content[:idx] + new_section + content[idx:]
        else:
            new_content = content.rstrip() + "\n\n" + new_section

        try:
            tmp = readme + ".tmp"
            with open(tmp, "w") as f:
                f.write(new_content)
            os.replace(tmp, readme)
        except Exception as e:
            LOGGER.error(f"readme write failed: {e}")
            return

        self.git.commit(f"[README] Refresh {len(new_files)} new / {len(changed)} changed file entries after execution cycle")
        await self.telegram.send_message(
            f"📚 Readme updated: +{len(new_files)} new entries, {len(changed)} refreshed."
        )

    async def run(self):
        """Main orchestrator loop. Re-reads settings.json before every phase."""
        self.running = True
        self.state.update(running=True, start_time=datetime.now().isoformat())

        # 2026-08-07 (user: "only you or the assistant personally texting
        # me"): the startup ping is automated junk — gate behind
        # notifications=off like every other automated push.
        if self.settings.get('notifications') != 'off':
            await self.telegram.send_message("🚀 Project Orchestrator started.")

        # Background relay: polls the Claude->Telegram outbox every 2s so
        # replies deliver even while a phase (e.g. execution) holds the loop.
        # Does NOT auto-respond — incoming Telegram messages stay in the inbox
        # for Claude Code (the human+AI session) to read and reply to manually.
        async def _relay_task():
            while True:
                try:
                    await self.telegram._relay_outbox()
                except Exception as _e:
                    LOGGER.debug(f"[RELAY] outbox relay error: {_e}")
                await asyncio.sleep(2)

        relay_handle = asyncio.create_task(_relay_task())
        LOGGER.info("[RELAY] background outbox-relay task started (auto-responder disabled).")

        try:
            while self.running and self.state.state.get('running', True):
                while self.state.state['paused']:
                    await asyncio.sleep(1)

                settings = self.settings.get()

                # Honor a scheduled pause (/schedule <time>) once its time arrives.
                sched = self.state.state.get('scheduled_pause')
                if sched:
                    try:
                        sched_t = float(sched)
                    except (TypeError, ValueError):
                        sched_t = None
                        self.state.update(scheduled_pause=None)
                    if sched_t and time.time() >= sched_t:
                        self.state.update(paused=True, scheduled_pause=None)
                        await self.telegram.send_message("⏸️ Scheduled pause reached. Use /resume to continue.")

                cycle_num = self.state.state['cycle'] + 1

                if settings.get('max_cycles'):
                    try:
                        max_c = int(settings['max_cycles'])
                    except (TypeError, ValueError):
                        max_c = 0
                    if max_c and cycle_num > max_c:
                        LOGGER.info(f"max_cycles ({max_c}) reached; stopping.")
                        await self.telegram.send_message(f"🛑 max_cycles ({max_c}) reached. Stopping.")
                        break

                self.state.update(cycle=cycle_num)
                LOGGER.info(f"=== CYCLE {cycle_num} START ===")

                # Start from initial_phase (e.g. resume at execution) until the
                # initial phase pipeline has completed once. Persisted in state so
                # a mid-cycle restart resumes at the right phase instead of
                # restarting the whole loop at audit.
                phases = PHASES
                if self.initial_phase != "audit" and not self.state.state.get('initial_phase_done'):
                    start_idx = PHASES.index(self.initial_phase)
                    phases = PHASES[start_idx:]

                phase_loop_interrupted = False
                for phase in phases:
                    while self.state.state['paused']:
                        await asyncio.sleep(1)

                    settings = self.settings.get()  # pick up Telegram edits mid-cycle

                    # Notify only for the work phases the user cares about:
                    # audit, cross_eval+plan, execution (not graphify/analyze).
                    # Audit/cross-eval 'started' is deferred — we only announce
                    # them when they actually produce NEW output, so the feed
                    # stays meaningful instead of re-announcing the same files
                    # every cycle.
                    WORK_PHASES = ("audit", "cross_eval", "execution")
                    _deferred_start = None
                    if phase in WORK_PHASES:
                        label = {
                            "audit": "🕵️ **Audit**",
                            "cross_eval": "🧪 **Cross-Eval + Plan**",
                            "execution": "⚙️ **Execution**",
                        }[phase]
                        if phase == "execution":
                            # Defer the "started" notification so it only fires
                            # when execution actually produces fresh output —
                            # otherwise every quick cycle spams the user with
                            # "⚙️ Execution started. ✅ Execution finished."
                            _deferred_start = f"{label} started."
                        else:
                            _deferred_start = f"{label} started."

                    if phase == "audit":
                        ok, result = await self.executor.run_audit(settings)
                        await self._notify_phase_progress(phase, settings)
                    elif phase == "cross_eval":
                        ok, result = await self.executor.run_cross_eval(settings)
                        await self._notify_phase_progress(phase, settings)
                    elif phase == "execution":
                        selfmod_active = await self._maybe_start_selfmod(settings, cycle_num)
                        ok, result = await self.executor.run_execution(settings)
                        await self._notify_phase_progress(phase, settings)
                        if ok:
                            await self._update_readme(settings)
                            if selfmod_active:
                                await self._maybe_finish_selfmod(settings)
                    elif phase == "graphify":
                        ok, result = await self.executor.rebuild_graphify()
                    else:  # analyze
                        await self._analyze_and_adapt(cycle_num)
                        ok, result = True, "analysis complete"

                    if not ok:
                        # Retry once, then hand control to error handling (which pauses).
                        retry_count = int(self.config.get('error_handling', {}).get('auto_retry_count', 1) or 0)
                        if retry_count > 0:
                            LOGGER.info(f"Retrying phase '{phase}' once after failure.")
                            if phase == "audit":
                                ok, result = await self.executor.run_audit(settings)
                                await self._notify_phase_progress(phase, settings)
                            elif phase == "cross_eval":
                                ok, result = await self.executor.run_cross_eval(settings)
                                await self._notify_phase_progress(phase, settings)
                            elif phase == "execution":
                                ok, result = await self.executor.run_execution(settings)
                                await self._notify_phase_progress(phase, settings)
                            elif phase == "graphify":
                                ok, result = await self.executor.rebuild_graphify()
                            else:
                                ok = True
                        if not ok:
                            await self._handle_error(f"{phase} failed", result)
                            phase_loop_interrupted = True
                            break  # exit phase loop; outer loop waits while paused
                    else:
                        # Phase finished successfully — tell the user, but ONLY
                        # announce audit/cross-eval when they produced new output.
                        if phase == "audit":
                            if self._is_fresh_output("audit", result):
                                if _deferred_start:
                                    await self._notify_user(_deferred_start, settings)
                                await self._notify_user(f"✅ **Audit** finished: `{result}`", settings)
                        elif phase == "cross_eval":
                            if self._is_fresh_output("cross_eval", result):
                                if _deferred_start:
                                    await self._notify_user(_deferred_start, settings)
                                await self._notify_user(f"✅ **Cross-Eval + Plan** finished: `{result}`", settings)
                        elif phase == "execution":
                            if self._is_fresh_output("execution", result):
                                if _deferred_start:
                                    await self._notify_user(_deferred_start, settings)
                                await self._notify_user(f"✅ **Execution** finished.", settings)

                    # Honor "pause_after" setting after a specific phase completes
                    if settings.get('pause_after') == phase:
                        self.state.update(paused=True)
                        await self.telegram.send_message(
                            f"⏸️ Paused after `{phase}` per settings (`pause_after`). Use /resume to continue."
                        )
                        phase_loop_interrupted = True
                        break

                # Once the initial-phase pipeline has run to completion (no pause/
                # error), later cycles do the full audit->cross_eval->execution loop.
                if not phase_loop_interrupted and not self.state.state.get('initial_phase_done'):
                    self.state.update(initial_phase_done=True)

                if not self.running:
                    break

                if settings.get('loop_mode') == 'once':
                    LOGGER.info("loop_mode=once; stopping after cycle.")
                    await self.telegram.send_message("✅ `loop_mode=once` reached; stopping. Use /resume or edit settings to relaunch.")
                    self.state.update(running=False)
                    self.running = False
                    break

                LOGGER.info(f"=== CYCLE {cycle_num} COMPLETE ===")

                await asyncio.sleep(5)

        except KeyboardInterrupt:
            LOGGER.info("Orchestrator interrupted")
        except Exception as e:
            LOGGER.error(f"Orchestrator error: {e}")
            try:
                await self.telegram.send_message(f"❌ Orchestrator error: {e}")
            except Exception:
                pass
        finally:
            relay_handle.cancel()
            self.state.update(running=False)
            try:
                await self.telegram.send_message("🛑 Orchestrator stopped")
            except Exception:
                pass
    
    async def _analyze_and_adapt(self, cycle_num: int):
        """Analyze cycle results and adapt scripts if needed."""
        self.state.update(phase="analyze", phase_start_time=datetime.now().isoformat())
        
        audit_file = self.state.state.get('last_audit_file')
        plan_file = self.state.state.get('last_plan_file')
        
        if not audit_file or not plan_file:
            LOGGER.warning("Missing audit or plan file for analysis")
            return
        
        try:
            with open(audit_file, 'r') as f:
                audit_content = f.read()[:5000]
            with open(plan_file, 'r') as f:
                plan_content = f.read()[:5000]
            
            prev_summary = None
            
            tasks = self.tasks.get_all_tasks()
            analysis, edits = await self.llm.analyze_cycle(
                audit_file, plan_file, "Execution completed", tasks, prev_summary
            )

            if self.llm.last_tier_event:
                await self.telegram.send_message(self.llm.last_tier_event)
            if self.llm.last_context_event:
                await self.telegram.send_message(self.llm.last_context_event)

            LOGGER.info(f"LLM Analysis: {analysis}")
            
            if edits:
                await self.telegram.send_message(f"🤖 **LLM Analysis**: {analysis}")
                
                for edit in edits:
                    LOGGER.info(f"Applying edit to {edit['file']}: {edit['description']}")
                    
                    file_path = os.path.join(
                        self.config['repo_dir'],
                        edit['file']
                    )
                    
                    if self.editor.apply_edit(file_path, edit['old_code'], edit['new_code']):
                        if self.editor.validate_edit(file_path):
                            self.git.commit(f"[LLM] {edit['description']}")
                            await self.telegram.send_message(
                                f"✅ Edited {edit['file']}: {edit['description']}"
                            )
                        else:
                            self.git.checkout_files([file_path])
                            await self.telegram.send_message(
                                f"❌ Syntax error in {edit['file']}. Reverted."
                            )
                    else:
                        await self.telegram.send_message(
                            f"❌ Could not apply edit to {edit['file']}"
                        )
            else:
                LOGGER.info("No edits needed based on LLM analysis")
        
        except Exception as e:
            LOGGER.error(f"Adaptation phase error: {e}")
    
    async def _handle_error(self, phase: str, error: str):
        """Handle workflow errors."""
        LOGGER.error(f"{phase}: {error}")
        self.state.update(last_error=error)

        # 2026-08-07 (user: "only you or the assistant personally texting me"):
        # gate the error push behind notifications=off. Step halts are handled
        # by the Claude session directly; genuine crashes still reach the user
        # via claude_deadman.sh's pipeline-error alert (direct bot send).
        if self.settings.get('notifications') != 'off':
            await self.telegram.send_message(f"❌ **Error in {phase}**: {error}")
        
        LOGGER.info("Attempting LLM-based error recovery...")
        
        recovery_prompt = f"""
An error occurred in the Project workflow:

**Phase**: {phase}
**Error**: {error}

Analyze this error and suggest how to fix it. Should we:
1. Retry the phase?
2. Modify a workflow script?
3. Manual intervention required?

Respond with a brief action plan.
"""
        
        recovery_suggestion = await self.llm.direct_query(recovery_prompt)

        # 2026-08-07: gate ALL automated pushes (not just the error line) —
        # the recovery suggestion + pause notice went out every halt while
        # notifications=off (user: "only you or the assistant personally
        # texting me"). The Claude session handles halts directly.
        if self.settings.get('notifications') != 'off':
            await self.telegram.send_message(f"🔧 **Recovery Suggestion**:\n{recovery_suggestion}")

        self.state.update(paused=True)
        if self.settings.get('notifications') != 'off':
            await self.telegram.send_message("⏸️ Workflow paused. Use /resume when ready to continue.")


def setup_logging(log_file: str):
    """Configure logging."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Project Master Orchestrator")
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--phase',
        default='audit',
        choices=['audit', 'cross_eval', 'execution', 'graphify', 'analyze'],
        help='Phase to start the first cycle from (default: audit)'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current status and exit'
    )

    args = parser.parse_args()

    config = OrchestratorConfig.load(args.config)

    log_file = config.get('log_file', '/tmp/orchestrator.log')
    setup_logging(log_file)

    LOGGER.info(f"Project Orchestrator starting (config: {args.config})")

    # STEP 1019/1558: record the config fingerprint at every launch so drift
    # across restarts is detectable by comparing digests in the log. Read-only;
    # never mutates live parameters (guarded hot-reload rule).
    try:
        import sys as _sys
        _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _repo_root not in _sys.path:
            _sys.path.insert(0, _repo_root)
        from config.core import compute_config_fingerprint
        LOGGER.info(f"Config fingerprint: {compute_config_fingerprint()}")
    except Exception as _fp_exc:  # pragma: no cover - never block startup
        LOGGER.warning(f"Config fingerprint unavailable: {_fp_exc}")

    state_manager = StateManager(config['state_file'])
    task_manager = TaskManager(config.get('task_file', '/tmp/agent_todo_list.json'))
    git_manager = GitManager(config['repo_dir'])
    llm_agent = LLMAgent(
        config['llm_api_key'],
        state_file=os.path.join(config.get('work_dir', '/tmp'), 'llm_agent_state.json'),
        history_file=os.path.join(config.get('work_dir', '/tmp'), 'llm_conversation.json')
    )
    script_editor = ScriptEditor(git_manager)
    executor = WorkflowExecutor(config, state_manager)
    settings_manager = SettingsManager(config.get('settings_file',
                                                  os.path.join(config.get('work_dir', '/tmp'),
                                                               'workflow_settings.json')))

    telegram_bot = TelegramBot(
        token=config['telegram_bot_token'],
        chat_id=config['telegram_chat_id'],
        config=config,
        state_manager=state_manager,
        task_manager=task_manager,
        llm_agent=llm_agent,
        script_editor=script_editor,
        workflow_executor=executor,
        git_manager=git_manager,
        settings_manager=settings_manager
    )

    # Give the chat agent the file-tool context so it can view/edit files itself.
    llm_agent.bind_tools(telegram_bot)

    await telegram_bot.initialize()

    orchestrator = MasterOrchestrator(
        config=config,
        state_manager=state_manager,
        task_manager=task_manager,
        executor=executor,
        llm=llm_agent,
        editor=script_editor,
        git=git_manager,
        telegram=telegram_bot,
        settings_manager=settings_manager,
        initial_phase=args.phase
    )
    telegram_bot.attach_orchestrator(orchestrator)

    if args.status:
        status = json.dumps(state_manager.state, indent=2)
        print(f"Status: {status}")
        return

    if args.resume and state_manager.state['cycle'] > 0:
        LOGGER.info(f"Resuming orchestrator at cycle {state_manager.state['cycle']}")

    # Keep the process alive across /restart, /continue, /halt and /stop so the
    # whole pipeline can be driven from Telegram without touching the laptop.
    while True:
        await orchestrator.run()
        if telegram_bot._want_restart:
            telegram_bot._want_restart = False
            LOGGER.info("Restart requested via Telegram — relaunching orchestrator loop.")
            state_manager.update(running=True, paused=False, halted=False)
            await asyncio.sleep(1)
            continue
        if telegram_bot._stop_permanent:
            break
        # Stopped for another reason (halt / error / loop_mode=once) — idle so
        # the user can decide from Telegram what to do next.
        await telegram_bot.send_message(
            "⏸️ Orchestrator loop stopped. `/continue` resumes, `/restart` relaunches, `/shutdown` exits."
        )
        while not telegram_bot._want_restart and not telegram_bot._stop_permanent:
            await telegram_bot._relay_outbox()  # forward any claude->telegram replies
            await asyncio.sleep(3)


if __name__ == '__main__':
    asyncio.run(main())