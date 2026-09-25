"""config.py — the orchestrator's control surface, expressed as DATA.

Everything the panel can change and the MCP can expose lives here as a schema, so there is
one definition of "what is configurable" and both front ends read it. Adding a setting means
adding one entry, not editing a menu and an MCP tool and a help text.

Storage is one JSON file (orch_config.json) written atomically. Resolution order for a value,
most specific first:
    1. an explicit value in orch_config.json
    2. the process environment (so a launcher's env still wins, matching how the engine reads it)
    3. the built-in default shown here

Two shapes of setting:
  * KIND_VALUE   — a scalar, edited in place.
  * KIND_TARGETS — the per-target table (name -> override map). Its KEYS are open-ended because
    a user adds their own repos, so it is edited through its own screen rather than a row.

Nothing here performs an audit. It only records intent; `runs.py` turns intent into an env.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path(os.environ.get("ORCH_CONFIG", ROOT / "orch_config.json"))


def workspace_root() -> Path:
    """Where the audited repositories actually live.

    ⚠️ NOT $HOME/Roni_workspace. On this machine HOME is /home/roni-saguey while the
    workspace is /home/roni/Roni_workspace, so a default built from Path.home() pointed at
    a directory that does not exist — the output dir silently became a missing path and
    `orch doctor` was the only thing that caught it. Discovered from the targets themselves
    first, then the two known layouts, then HOME.
    """
    env = os.environ.get("ORCH_WORKSPACE")
    if env and Path(env).is_dir():
        return Path(env)
    for cand in (Path("/home/roni/Roni_workspace"), Path.home() / "Roni_workspace"):
        if cand.is_dir():
            return cand
    return ROOT


WORKSPACE = workspace_root()

# ── kinds ────────────────────────────────────────────────────────────────────
KIND_VALUE = "value"
KIND_LIST = "list"
KIND_BOOL = "bool"
KIND_INT = "int"
KIND_SECRET = "secret"


def _defaults() -> dict:
    return {
        # --- where the engine lives -------------------------------------------------
        "enginePy": str(ROOT / "engine" / "audit.py"),
        "python": os.environ.get("ORCH_PYTHON", ""),
        "aggregateUrl": "http://127.0.0.1:8090/v1",
        "apiKeyName": "DEEPSEEK_API_KEY",
        "apiKeyValue": "harness",

        # --- how a run behaves ------------------------------------------------------
        "outputDir": str(WORKSPACE / "audits_plans"),
        "passes": 2,
        "batchSize": 5,
        "concurrency": 8,          # engine --limit: concurrent agent LLM calls
        "resume": True,
        "chatTimeout": 300,
        "chatMaxTokens": 8000,
        "requireSubstantive": True,
        "maxEmptyRotations": 2,
        "modelAllowlist": ["ds"],
        "primaryModel": "auto/best-reasoning",

        # --- the targets table -------------------------------------------------------
        # A target is a repo the orchestrator can audit. Keys are open-ended.
        "targets": {
            "helpotron": {
                "dir": str(WORKSPACE / "helpotron"),
                "label": "helpotron",
                "readme": str(WORKSPACE / "helpotron" / "README.md"),
                "sot": "",
                "graph": str(WORKSPACE / "helpotron" / "graphify-out" / "graph.json"),
                "includeFiles": "",
                "extraRoots": "",
                "task": "",
            },
            "t2b": {
                "dir": str(WORKSPACE / "t2b"),
                "label": "t2b",
                "readme": str(WORKSPACE / "t2b" / "README.md"),
                "sot": "",
                "graph": "",
                "includeFiles": "",
                "extraRoots": "",
                "task": "",
            },
            "harness": {
                "dir": str(WORKSPACE / "webchat_worker/harness"),
                "label": "webchat-to-api-harness",
                "readme": str(WORKSPACE / "webchat_worker/harness" / "README.md"),
                "sot": "",
                "graph": "",
                "includeFiles": "",
                "extraRoots": "",
                "task": "",
            },
        },
        "activeTarget": "helpotron",

        # --- UI ---------------------------------------------------------------------
        "dashboard": {
            "refreshSeconds": 5,
            "showPacing": True,
            "showModels": True,
        },
    }


# ── the schema the panel renders ─────────────────────────────────────────────
# Each entry: id, label, kind, help, and optionally group + choices + min/max.
SCHEMA: list[dict] = [
    # connection
    {"id": "enginePy", "group": "Connection", "label": "Engine path", "kind": KIND_VALUE,
     "help": "Path to engine/audit.py. The panel runs this file."},
    {"id": "python", "group": "Connection", "label": "Python interpreter", "kind": KIND_VALUE,
     "help": "Interpreter used to run the engine. Empty = auto-detect one that can import aiohttp."},
    {"id": "aggregateUrl", "group": "Connection", "label": "Model endpoint", "kind": KIND_VALUE,
     "help": "OpenAI-compatible base URL the engine sends to. The aggregate fan-in fronts every lane."},
    {"id": "apiKeyName", "group": "Connection", "label": "API key env name", "kind": KIND_VALUE,
     "help": "Name of the env var the key is passed in (DEEPSEEK_API_KEY for this engine)."},
    {"id": "apiKeyValue", "group": "Connection", "label": "API key value", "kind": KIND_SECRET,
     "help": "Value passed as the key. The aggregate accepts any non-empty value."},

    # run behaviour
    {"id": "outputDir", "group": "Run", "label": "Output directory", "kind": KIND_VALUE,
     "help": "Where batches, findings and the final report are written."},
    {"id": "passes", "group": "Run", "label": "Passes", "kind": KIND_INT, "min": 1, "max": 5,
     "help": "How many audit passes. Pass 1 finds; later passes corroborate. 1 is a first sweep."},
    {"id": "batchSize", "group": "Run", "label": "Batch size (files)", "kind": KIND_INT, "min": 1, "max": 20,
     "help": "Files per batch. A big payload makes a lane time out, so lower it for a slow lane."},
    {"id": "concurrency", "group": "Run", "label": "Concurrent agents", "kind": KIND_INT, "min": 1, "max": 32,
     "help": "The engine's --limit: how many agent calls are in flight at once."},
    {"id": "resume", "group": "Run", "label": "Resume finished batches", "kind": KIND_BOOL,
     "help": "Skip batches already saved on disk. OFF re-runs everything and OVERWRITES results - use with care."},
    {"id": "chatTimeout", "group": "Run", "label": "Per-call timeout (s)", "kind": KIND_INT, "min": 30, "max": 1800,
     "help": "A webchat lane needs minutes, not seconds; a short value bans every lane on its first call."},
    {"id": "chatMaxTokens", "group": "Run", "label": "Max tokens per reply", "kind": KIND_INT, "min": 256, "max": 64000,
     "help": "A reply cut off mid-JSON loses the whole round's findings."},
    {"id": "requireSubstantive", "group": "Run", "label": "Require substantive findings", "kind": KIND_BOOL,
     "help": "Treat an all-empty findings document as inconclusive and rotate models rather than calling it clean."},
    {"id": "maxEmptyRotations", "group": "Run", "label": "Max empty rotations", "kind": KIND_INT, "min": 0, "max": 10,
     "help": "How many alternative models to try before accepting an empty document."},

    # models
    {"id": "modelAllowlist", "group": "Models", "label": "Allowed models", "kind": KIND_LIST,
     "help": "Comma-separated. The engine will ONLY use these. Order matters: the first available one wins each round."},
    {"id": "primaryModel", "group": "Models", "label": "Primary model", "kind": KIND_VALUE,
     "help": "The engine's reported primary. The allowlist decides what actually runs."},

    # ui
    {"id": "dashboard.refreshSeconds", "group": "Dashboard", "label": "Refresh (s)", "kind": KIND_INT, "min": 1, "max": 120,
     "help": "How often the dashboard re-reads run state."},
    {"id": "dashboard.showModels", "group": "Dashboard", "label": "Show model health", "kind": KIND_BOOL,
     "help": "Display the per-model probe results on the dashboard."},
    {"id": "dashboard.showPacing", "group": "Dashboard", "label": "Show lane pacing", "kind": KIND_BOOL,
     "help": "Display per-lane in-flight timing. Useful when a run looks stalled."},
]

BY_ID = {s["id"]: s for s in SCHEMA}


# ── read / write ─────────────────────────────────────────────────────────────

# ── a corrupt config must never be silent, and must never be destroyed ────────
# Two failures, both measured, both the same shape as the ones this project keeps finding:
#
#   1. SILENT FALLBACK. A config with a single typo parsed as an exception, the defaults
#      were returned, and the CLI printed a normal settings list with exit 0. Every setting
#      appeared to have reverted with no indication of why — "I changed it and nothing
#      happened" pointed at the wrong thing entirely.
#   2. DATA LOSS. `setting_save` writes back the dict it loaded, and the loaded dict was the
#      DEFAULTS — so one `config set` replaced the user's entire config with defaults. Measured:
#      a custom value present before the save was gone after it, with no backup taken.
#
# The fix is in two places, deliberately: the error is REMEMBERED so it can be reported, and
# `save_raw` backs up any file it is about to overwrite that does not parse. The second holds
# regardless of which caller does the writing, including a future one written without knowing
# about this class of bug.
_LAST_PARSE_ERROR: dict = {}


def config_problem() -> dict | None:
    """The parse error of the config file, if it has one. None when the file is fine.

    Callers surface this: a settings read that silently substituted defaults is worse than an
    error, because the user is shown a plausible configuration that is not theirs.

    ⚠️ It must CHECK, not just report a remembered error. The first version read the flag that
    load_raw() sets — but a caller that asks about the config before loading anything got None
    and reported "nothing to repair" over a file that plainly does not parse. The check is
    cheap (one read) and is the only thing that makes the answer correct at any call order.
    """
    if not CONFIG_FILE.exists():
        _LAST_PARSE_ERROR.clear()
        return None
    try:
        json.loads(CONFIG_FILE.read_text())
        _LAST_PARSE_ERROR.clear()
        return None
    except Exception as e:
        _LAST_PARSE_ERROR.update({
            "file": str(CONFIG_FILE),
            "error": f"{type(e).__name__}: {e}",
            "bytes": CONFIG_FILE.stat().st_size,
        })
        return dict(_LAST_PARSE_ERROR)


def load_raw() -> tuple[dict, Path]:
    """Raw config, plus the file it came from. Never raises on a missing file."""
    _LAST_PARSE_ERROR.clear()
    if CONFIG_FILE.exists():
        raw = CONFIG_FILE.read_text()
        try:
            return json.loads(raw), CONFIG_FILE
        except Exception as e:
            # Remember WHY, so the CLI/MCP/panel can say so rather than substituting defaults
            # in silence.
            _LAST_PARSE_ERROR.update({
                "file": str(CONFIG_FILE),
                "error": f"{type(e).__name__}: {e}",
                "bytes": len(raw),
            })
            return _defaults(), CONFIG_FILE
    return _defaults(), CONFIG_FILE


def get_path(obj: dict, dotted: str, default=None):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def set_path(obj: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = obj
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def save_raw(data: dict, path: Path | None = None) -> dict:
    """Atomic write. Returns {backedUp: <path>|None}.

    A partially written config silently reverts every setting the next time it is read, so
    the write goes to a temp file in the same directory and is renamed into place.

    ★ If the file being REPLACED does not parse, it is copied aside first. The caller loaded
    defaults in place of it (see load_raw), so writing would turn one typo into the total loss
    of every setting the user had. The backup is unconditional and does not depend on the
    caller knowing this can happen.
    """
    target = path or CONFIG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    backed_up = None
    if target.exists():
        try:
            json.loads(target.read_text())
        except Exception:
            try:
                stamp = time.strftime("%Y%m%d-%H%M%S")
                backed_up = target.with_name(target.name + f".bak-unreadable-{stamp}")
                import shutil as _sh
                _sh.copy2(target, backed_up)
            except Exception:
                backed_up = None
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=".orch_config.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return {"backedUp": str(backed_up) if backed_up else None}


def coerce(spec: dict, raw):
    """Turn a typed-in string into the type the setting expects."""
    kind = spec["kind"]
    if kind == KIND_BOOL:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on", "y")
    if kind == KIND_INT:
        try:
            v = int(str(raw).strip())
        except ValueError:
            raise ValueError(f"{spec['label']} expects a whole number")
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and v < lo:
            raise ValueError(f"{spec['label']} must be at least {lo}")
        if hi is not None and v > hi:
            raise ValueError(f"{spec['label']} must be at most {hi}")
        return v
    if kind == KIND_LIST:
        if isinstance(raw, list):
            items = raw
        else:
            items = [p.strip() for p in str(raw).split(",")]
        return [p for p in items if p]
    return str(raw)


def resolve(path: str):
    """The effective value and where it came from. Env wins over the file, as the engine sees it."""
    data, _ = load_raw()
    spec = BY_ID.get(path)
    env_name = None
    if spec:
        # Map a couple of settings to the env var the engine actually reads, so the panel
        # reports the value that will really be used rather than the one it stored.
        env_name = {
            "outputDir": "AUDIT_OUTPUT_DIR",
            "passes": "AUDIT_NUM_PASSES",
            "batchSize": "AUDIT_BATCH_SIZE",
            "chatTimeout": "AUDIT_CHAT_TIMEOUT",
            "chatMaxTokens": "AUDIT_CHAT_MAX_TOKENS",
            "requireSubstantive": "AUDIT_REQUIRE_SUBSTANTIVE",
            "maxEmptyRotations": "AUDIT_MAX_EMPTY_ROTATIONS",
            "primaryModel": "DEEPSEEK_MODEL_FLASH",
        }.get(path)
    if env_name and os.environ.get(env_name):
        return os.environ[env_name], "env"
    v = get_path(data, path, None)
    if v is None:
        d = get_path(_defaults(), path)
        return d, "default"
    return v, "file"


def resolved_all() -> list[dict]:
    out = []
    for spec in SCHEMA:
        value, source = resolve(spec["id"])
        out.append({**spec, "value": value, "source": source})
    return out


# Settings where a legal-looking value has a consequence the value does not show.
_EMPTY_MEANS: dict = {
    "modelAllowlist": (
        "an empty allowlist means DENY-ALL: the engine will refuse every model and every run "
        "will find nothing to call. Set at least one id (e.g. ds)."
    ),
}


def setting_save(path: str, raw) -> dict:
    """Coerce and persist one setting. Returns a result describing what happened."""
    spec = BY_ID.get(path)
    if not spec:
        return {"ok": False, "reason": f'unknown setting "{path}"'}
    try:
        value = coerce(spec, raw)
    except ValueError as e:
        return {"ok": False, "reason": str(e)}
    # A value can be within range and still mean something the value does not say. Saving it
    # silently is how an empty allowlist looked identical to a configured one while every run
    # found no models at all — the engine's empty chain is deny-all, by design.
    note = None
    if path in _EMPTY_MEANS and value in ([], "", None):
        note = _EMPTY_MEANS[path]
    data, file = load_raw()
    problem = config_problem()
    set_path(data, path, value)
    wr = save_raw(data, file)
    out = {"ok": True, "path": path, "value": value}
    if note:
        out["note"] = note
    if wr.get("backedUp"):
        # Say it plainly: the previous file could not be read, every other setting in it has
        # been replaced by its default, and the original is preserved at this path.
        out["warning"] = ("the config file could not be read, so its other settings were "
                          "replaced by their defaults on save")
        out["backedUp"] = wr["backedUp"]
        out["parseError"] = (problem or {}).get("error")
    return out


def setting_reset(path: str) -> dict:
    spec = BY_ID.get(path)
    if not spec:
        return {"ok": False, "reason": f'unknown setting "{path}"'}
    data, file = load_raw()
    parts = path.split(".")
    cur = data
    for p in parts[:-1]:
        cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None:
            break
    if isinstance(cur, dict):
        cur.pop(parts[-1], None)
    wr = save_raw(data, file)
    env_name = {
        "outputDir": "AUDIT_OUTPUT_DIR", "passes": "AUDIT_NUM_PASSES",
        "batchSize": "AUDIT_BATCH_SIZE", "chatTimeout": "AUDIT_CHAT_TIMEOUT",
    }.get(path)
    cleared_env = False
    if env_name and env_name in os.environ:
        # Report it: a value coming from the environment would keep winning and the reset
        # would look like it did nothing.
        cleared_env = os.environ.pop(env_name) is not None
    value, _ = resolve(path)
    out = {"ok": True, "path": path, "value": value, "clearedEnv": cleared_env}
    if wr.get("backedUp"):
        out["warning"] = ("the config file could not be read, so its other settings were "
                          "replaced by their defaults on save")
        out["backedUp"] = wr["backedUp"]
    return out


# ── targets ──────────────────────────────────────────────────────────────────

TARGET_FIELDS = [
    ("dir", "Repository root the file walk starts from"),
    ("label", "Name the domain profile is keyed by (helpotron / t2b / webchat-to-api-harness)"),
    ("readme", "README the engine reads for documented-feature comparison"),
    ("sot", "Source-of-truth document. Optional - only the SOT specialist gets it."),
    ("graph", "Graphify graph.json. Optional - without it the engine uses source context."),
    ("includeFiles", "Explicit file list (comma-separated). Set this and the walk is skipped entirely."),
    ("extraRoots", "Additional roots to walk (comma-separated) to cover several repos at once."),
    ("task", "Task/audit-protocol prompt file. Optional."),
]


def targets() -> dict:
    data, _ = load_raw()
    return data.get("targets") or {}


def target_get(name: str) -> dict | None:
    return targets().get(name)


def target_save(name: str, overrides: dict) -> dict:
    data, file = load_raw()
    t = data.setdefault("targets", {})
    cur = dict(t.get(name) or {})
    for k, v in overrides.items():
        if k in dict(TARGET_FIELDS):
            cur[k] = "" if v is None else str(v)
    # Sensible defaults for the fields a user will not have.
    cur.setdefault("label", name)
    if not cur.get("dir"):
        d = input(f"Repository root for '{name}': ").strip()
        if not d:
            return {"ok": False, "reason": "a target needs a repository root"}
        cur["dir"] = d
    t[name] = cur
    save_raw(data, file)
    return {"ok": True, "name": name, "target": cur}


def target_remove(name: str) -> dict:
    data, file = load_raw()
    t = data.setdefault("targets", {})
    if name not in t:
        return {"ok": False, "reason": f"no target named {name}"}
    if len(t) <= 1:
        return {"ok": False, "reason": "that is the only target; add another before removing it"}
    t.pop(name)
    if data.get("activeTarget") == name:
        data["activeTarget"] = next(iter(t))
    save_raw(data, file)
    return {"ok": True, "name": name}


def active_target() -> str:
    data, _ = load_raw()
    t = data.get("targets") or {}
    a = data.get("activeTarget")
    if a in t:
        return a
    return next(iter(t)) if t else ""


def set_active_target(name: str) -> dict:
    data, file = load_raw()
    if name not in (data.get("targets") or {}):
        return {"ok": False, "reason": f"no target named {name}"}
    data["activeTarget"] = name
    save_raw(data, file)
    return {"ok": True, "activeTarget": name}


def count_shadowed() -> int:
    """How many settings are overridden by the environment. Surfaced in the panel header."""
    return sum(1 for s in resolved_all() if s["source"] == "env")
