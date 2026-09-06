#!/usr/bin/env python3
"""orch_config.py — single layered-configuration module for the orchestrator.

Precedence (lowest to highest):  code defaults  <  config file  <  environment
<  explicit overrides (CLI flags / per-request opts).

Every knob in the executor, the lane layer and the escalation solver resolves
through here, so the gateway envs and the engine options read the SAME source.

Profiles (``ORCH_PROFILE``) are named presets applied on top of the defaults and
below the config file, so a profile never overrides an explicit user setting.

    ORCH_PROFILE=max-throughput python3 execute_8_27_engine.py --resume

Config file search order (first hit wins) — override with ``ORCH_CONFIG``:
    ./orch.yaml, ./orch.json, ~/.config/orch/orch.yaml,
    <repo>/orch.yaml, <repo>/orch.json
YAML is used when PyYAML is importable; JSON always works.

Paths default to the checkout this file lives in, so the same source runs from
any clone; deployment-specific ids (plan/state filenames) belong in the config
file, never in the code defaults.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# scripts/orch_config.py -> <repo>. Location-derived so a clone needs no edit.
_REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_DEFAULT = str(_REPO_ROOT)


def _base_default() -> str:
    """Plans/state directory: a sibling ``audits_plans`` when one exists (the
    layout the orchestrator was grown in), else ``<repo>/.orch``."""
    sibling = _REPO_ROOT.parent / "audits_plans"
    if sibling.is_dir():
        return str(sibling)
    return str(_REPO_ROOT / ".orch")


BASE_DEFAULT = _base_default()

# ---------------------------------------------------------------- defaults ---
# Every option: name -> (default, type, help). The README table is generated
# from this dict (see `python3 orch_config.py --markdown`).
DEFAULTS: dict[str, tuple[Any, type, str]] = {
    # paths
    "repo_dir":          (REPO_DEFAULT, str, "Repository the fixes are applied to"),
    "base_dir":          (BASE_DEFAULT, str, "Plans/state directory"),
    "plan_json":         ("", str, "Master plan JSON (default: <base>/plan.json)"),
    "exec_state":        ("", str, "Executor state JSON (default: <base>/exec_state.json)"),
    "state_backups":     (True, bool, "Keep a .bak of state before each atomic write"),
    "wake_log":          ("/tmp/main_wake.log", str, "Wake-chain log the escalation signal appends to"),
    "escalation_queue":  ("", str, "Escalation queue JSON (default: <base>/escalation_queue.json)"),
    "log_dir":           ("/tmp", str, "Directory for orchestrator logs"),

    # parallelism / batching
    "group_cap":         (5, int, "Max INDEPENDENT steps packed into one lane call"),
    "parallel":          (3, int, "Batch groups in flight at once"),
    "workers":           (8, int, "Legacy worker hint (batch packing width)"),
    "max_rounds":        (3, int, "Execute/verify rounds before a step escalates"),

    # lane behaviour
    "lane_timeout":      (240, int, "Per-request lane timeout in seconds (was an unbounded 900s total)"),
    "lane_connect_timeout": (15, int, "Connection timeout in seconds"),
    "lane_retries":      (3, int, "Attempts per model before moving to the next model"),
    "lane_max_hops":     (4, int, "Different lanes tried before a call is declared failed"),
    "lane_health_probe": (True, bool, "Probe lanes at startup and park the dead ones"),
    "min_lane_gap_seconds": (0, int, "Minimum seconds between two calls on the same lane"),
    "lanes_extra": ([], list, "Extra lanes: [{name,url,models,cool_base,cool_esc,auth}] appended to the pool"),
    "exclude_lanes": ([], list, "Lane names to leave out of the pool (e.g. a lane waiting for login)"),
    "max_prompt_chars":  (80000, int, "Hard cap on the user prompt sent to a lane"),
    "file_ctx_cap":      (24000, int, "Per-file FILE CONTENTS cap in characters"),
    "files_per_step":    (2, int, "Files inlined per step in a group prompt"),

    # escalation policy
    "escalation_policy": ("auto-fix", str, "wake | auto-fix | queue-only"),
    "escalation_rounds": (3, int, "Escalation-solver attempts per step"),
    "escalation_allow_plan_edit": (True, bool, "Escalation persona may rewrite a broken plan step"),
    "escalation_allow_verify_edit": (True, bool, "Escalation persona may relax/repair the verify command"),
    "escalation_batch":  (5, int, "Escalated steps handled per solver pass"),

    # safety / memory
    # This box idles near 78% used with ~3 GB genuinely available, so a 75%
    # ceiling parked every pass before it started. Absolute headroom is the
    # signal that actually predicts pressure; the percentage stays as a hard
    # backstop well below the OOM band.
    "ram_pct_cap":       (90, int, "Hard ceiling: pause above this system RAM percentage"),
    "min_avail_mb":      (600, int, "Pause when available memory falls below this many MB"),
    "swarm_cap":         (4, int, "Hard concurrency ceiling — never raised at runtime"),
    "dry_run":           (False, bool, "Resolve and print actions without side effects"),
    "verbose":           (False, bool, "Verbose lane/step logging"),
}

# ---------------------------------------------------------------- profiles ---
PROFILES: dict[str, dict[str, Any]] = {
    "default": {},
    "max-throughput": {
        "group_cap": 5, "parallel": 3, "lane_timeout": 180,
        "lane_retries": 2, "escalation_batch": 8,
    },
    "low-memory": {
        "group_cap": 2, "parallel": 1, "workers": 2, "file_ctx_cap": 12000,
        "max_prompt_chars": 40000, "min_avail_mb": 1200, "escalation_batch": 2,
    },
    "conservative": {
        "group_cap": 1, "parallel": 1, "max_rounds": 3, "lane_timeout": 300,
        "min_lane_gap_seconds": 20, "escalation_policy": "wake",
        "escalation_allow_plan_edit": False, "escalation_allow_verify_edit": False,
    },
    "debug": {
        "dry_run": True, "verbose": True, "parallel": 1, "group_cap": 1,
        "state_backups": True,
    },
}

_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}


def _coerce(value: Any, typ: type, key: str) -> Any:
    if isinstance(value, typ) and not (typ is int and isinstance(value, bool)):
        return value
    if typ is bool:
        s = str(value).strip().lower()
        if s in _TRUE:
            return True
        if s in _FALSE:
            return False
        raise ValueError(f"config key '{key}': expected a boolean, got {value!r}")
    if typ is int:
        try:
            return int(str(value).strip())
        except ValueError:
            raise ValueError(f"config key '{key}': expected an integer, got {value!r}")
    return str(value)


def _config_file() -> Path | None:
    explicit = os.environ.get("ORCH_CONFIG")
    candidates = [Path(explicit)] if explicit else [
        Path.cwd() / "orch.yaml",
        Path.cwd() / "orch.json",
        Path.home() / ".config" / "orch" / "orch.yaml",
        Path.home() / ".config" / "orch" / "orch.json",
        Path(REPO_DEFAULT) / "orch.yaml",
        Path(REPO_DEFAULT) / "orch.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _load_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError:
            raise ValueError(
                f"config file {path} is YAML but PyYAML is not installed; "
                f"use a .json config instead")
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text or "{}")
    if not isinstance(data, dict):
        raise ValueError(f"config file {path} must contain a mapping at the top level")
    return data


class Config:
    """Resolved configuration. Attribute and item access both work."""

    def __init__(self, values: dict[str, Any], sources: dict[str, str]):
        self._values = values
        self._sources = sources

    def __getattr__(self, name: str) -> Any:
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name)

    def __getitem__(self, name: str) -> Any:
        return self._values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def source(self, name: str) -> str:
        return self._sources.get(name, "default")

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)

    # derived paths -------------------------------------------------------
    @property
    def plan_path(self) -> Path:
        # A deployment's plan/state filenames are deployment data, not code
        # defaults: set `plan_json`/`exec_state` in the config file. Getting
        # this wrong is silent — a plan that does not cover the state file
        # resolves only the ids it happens to share (a 263-step plan once
        # matched 7 of 52 escalated ids), so the resolver misses the rest.
        return Path(self._values["plan_json"] or
                    Path(self._values["base_dir"]) / "plan.json")

    @property
    def state_path(self) -> Path:
        return Path(self._values["exec_state"] or
                    Path(self._values["base_dir"]) / "exec_state.json")

    @property
    def queue_path(self) -> Path:
        return Path(self._values["escalation_queue"] or
                    Path(self._values["base_dir"]) / "escalation_queue.json")

    def render(self) -> str:
        rows = []
        for key in sorted(self._values):
            src = self.source(key)
            mark = "" if src == "default" else f"   <- {src}"
            rows.append(f"  {key:<28} = {self._values[key]!r}{mark}")
        return "\n".join(rows)


# Legacy env names still honoured so existing systemd units keep working.
ENV_ALIASES = {
    "repo_dir": ["ORCH_REPO_DIR"],
    "plan_json": ["PLAN_JSON"],
    "exec_state": ["EXEC_STATE"],
    "group_cap": ["EXEC_GROUP_CAP"],
    "parallel": ["EXEC_PARALLEL"],
    "workers": ["EXEC_WORKERS"],
    "min_lane_gap_seconds": ["MIN_LANE_GAP_SECONDS"],
}


def load(overrides: dict[str, Any] | None = None, profile: str | None = None) -> Config:
    """Resolve configuration across all layers. Raises ValueError on a bad key."""
    values: dict[str, Any] = {k: v[0] for k, v in DEFAULTS.items()}
    sources: dict[str, str] = {}

    # layer 1: profile
    prof = profile or os.environ.get("ORCH_PROFILE") or "default"
    if prof not in PROFILES:
        raise ValueError(
            f"unknown ORCH_PROFILE '{prof}' — available: {', '.join(sorted(PROFILES))}")
    for k, v in PROFILES[prof].items():
        if k not in DEFAULTS:
            raise ValueError(f"profile '{prof}' sets unknown key '{k}'")
        values[k] = v
        sources[k] = f"profile:{prof}"

    # layer 2: config file
    path = _config_file()
    if path is not None:
        for k, v in _load_file(path).items():
            if k not in DEFAULTS:
                raise ValueError(
                    f"config file {path}: unknown key '{k}' "
                    f"(known keys: {', '.join(sorted(DEFAULTS))})")
            values[k] = _coerce(v, DEFAULTS[k][1], k)
            sources[k] = f"file:{path}"

    # layer 3: environment (ORCH_<KEY> plus legacy aliases)
    for k in DEFAULTS:
        names = [f"ORCH_{k.upper()}"] + ENV_ALIASES.get(k, [])
        for name in names:
            raw = os.environ.get(name)
            if raw is not None and raw != "":
                values[k] = _coerce(raw, DEFAULTS[k][1], k)
                sources[k] = f"env:{name}"
                break

    # layer 4: explicit overrides
    for k, v in (overrides or {}).items():
        if v is None:
            continue
        if k not in DEFAULTS:
            raise ValueError(f"unknown override '{k}'")
        values[k] = _coerce(v, DEFAULTS[k][1], k)
        sources[k] = "override"

    _validate(values)
    return Config(values, sources)


def _validate(v: dict[str, Any]) -> None:
    if v["escalation_policy"] not in ("wake", "auto-fix", "queue-only"):
        raise ValueError(
            f"config key 'escalation_policy': must be wake|auto-fix|queue-only, "
            f"got {v['escalation_policy']!r}")
    if v["swarm_cap"] > 4:
        raise ValueError("config key 'swarm_cap': hard ceiling is 4 — never raise it")
    for key in ("group_cap", "parallel", "max_rounds", "lane_retries",
                "lane_max_hops", "escalation_rounds", "escalation_batch"):
        if v[key] < 1:
            raise ValueError(f"config key '{key}': must be >= 1, got {v[key]}")
    if v["parallel"] > v["swarm_cap"]:
        raise ValueError(
            f"config key 'parallel' ({v['parallel']}) exceeds swarm_cap "
            f"({v['swarm_cap']})")
    if not 10 <= v["ram_pct_cap"] <= 95:
        raise ValueError(f"config key 'ram_pct_cap': must be 10..95, got {v['ram_pct_cap']}")
    if v["lane_timeout"] < 30:
        raise ValueError(f"config key 'lane_timeout': must be >= 30, got {v['lane_timeout']}")
    if not Path(v["repo_dir"]).is_dir():
        raise ValueError(f"config key 'repo_dir': not a directory: {v['repo_dir']}")


def markdown_table() -> str:
    tuned: dict[str, list[str]] = {}
    for name, over in PROFILES.items():
        for k in over:
            tuned.setdefault(k, []).append(name)
    lines = ["| Option | Default | Tuned by profiles | Description |",
             "| --- | --- | --- | --- |"]
    for k in sorted(DEFAULTS):
        d, _t, h = DEFAULTS[k]
        # descriptions carry literal pipes ("wake | auto-fix"), which would
        # otherwise open a new table column
        desc = h.replace("|", "\\|")
        lines.append(f"| `{k}` | `{d!r}` | {', '.join(tuned.get(k, [])) or '—'} | {desc} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    if "--markdown" in sys.argv:
        print(markdown_table())
    else:
        cfg = load()
        print(f"profile: {os.environ.get('ORCH_PROFILE', 'default')}")
        print(cfg.render())
