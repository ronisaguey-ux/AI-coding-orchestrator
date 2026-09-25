"""runs.py — start, observe and stop audit runs.

A run is a detached `engine/audit.py` process plus a small JSON record so the panel can find
it again. The record lives in <outputDir>/.orch/run-<id>.json, next to the batches it produced.

Two deliberate choices, both learned from the harness:

  * The process is DETACHED (setsid, own session, output to a log file). A run takes hours;
    it must not die when the terminal that started it closes, and the panel must be able to
    exit without killing it.
  * Liveness is decided by the RECORDED PID, not by a log's mtime. A log that has stopped
    growing might be a finished run, a slow model, or a dead process — only the pid answers
    that, and a run that reports "running" forever is worse than one that reports nothing.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import config

RUNS_SUBDIR = ".orch"


# ── locating things ──────────────────────────────────────────────────────────

def out_dir(target: str | None = None) -> Path:
    """The output directory for a target's runs.

    ★ PER TARGET, not one shared directory. The engine decides a whole pass is complete by
    looking for `pass_N/summary.json` and does not check whose audit produced it, so two
    targets sharing one output dir means the second audit SKIPS its own pass 1 and reports the
    first audit's findings as its own. Measured: a harness run resumed past its entire first
    pass using helpotron batches sitting in the same directory.

    A name is passed through unchanged so an explicit `outputDir` in an override still works
    exactly as written; the target is only appended to the configured base.
    """
    base = Path(str(config.resolve("outputDir")[0]))
    return base / target if target else base


def runs_dir() -> Path:
    """Where run records live. One directory for every target.

    Deliberately NOT under a target's output dir: a run's record has to be findable no matter
    which target it audited, and nesting it per target would mean `orch status` only saw the
    active target's runs.
    """
    d = Path(str(config.resolve("outputDir")[0])) / RUNS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_file(run_id: str) -> Path:
    return runs_dir() / f"run-{run_id}.json"


def pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# ── python discovery ─────────────────────────────────────────────────────────

def find_python() -> str:
    """An interpreter that can import the engine's dependencies.

    The engine needs aiohttp. System python3 usually does not have it; a project venv does.
    Picking the wrong one fails with a bare ModuleNotFoundError at launch, so the candidates
    are tested rather than assumed.
    """
    configured = str(config.resolve("python")[0] or "")
    if configured and Path(configured).exists():
        return configured
    candidates = [
        config.WORKSPACE / "helpotron" / ".venv" / "bin" / "python",
        config.ROOT / ".venv" / "bin" / "python",
        config.WORKSPACE / ".venv" / "bin" / "python",
        Path("/usr/bin/python3"),
        Path(sys.executable),
    ]
    for c in candidates:
        if c and Path(c).exists():
            try:
                r = subprocess.run([str(c), "-c", "import aiohttp"], capture_output=True, timeout=20)
                if r.returncode == 0:
                    return str(c)
            except Exception:
                continue
    return sys.executable


# ── env for a run ────────────────────────────────────────────────────────────

class UnknownTarget(ValueError):
    """Raised when a target is named but does not exist.

    A separate type because `target_get()` returns None for an unknown name and `or {}`
    hid it: the name fell through to root="" -> Path("") -> ".", so `orch scan nosuchtarget`
    walked the ORCHESTRATOR's own source and reported it as the audited repo, and
    `orch agents nosuchtarget` labelled the personas with a target that does not exist.
    Both exited 0. A typo must be an error, not a different answer.
    """


def resolve_target(target: str | None = None):
    """Return (name, record). Raise UnknownTarget if a NAME was given and does not exist.

    An omitted target still falls back to the active one - that is convenience, not a
    silent substitution for a name the user actually typed.
    """
    name = target or config.active_target()
    t = config.target_get(name)
    if t is None:
        raise UnknownTarget(
            "unknown target %r; known: %s" % (name, ", ".join(sorted(config.targets()))))
    return name, t


def build_env(target: str | None = None, overrides: dict | None = None) -> dict:
    """Turn config + target into the environment engine/audit.py reads.

    The engine reads everything from the environment. This is the one place that mapping
    lives, so the panel and the MCP cannot drift apart on what a setting means.
    """
    cfg = {s["id"]: config.resolve(s["id"])[0] for s in config.SCHEMA}
    tname, t = resolve_target(target)
    o = overrides or {}

    env = dict(os.environ)
    key_name = str(cfg.get("apiKeyName") or "DEEPSEEK_API_KEY")
    env[key_name] = str(cfg.get("apiKeyValue") or "")
    env["DEEPSEEK_API_BASE"] = str(o.get("aggregateUrl") or cfg.get("aggregateUrl"))

    # ⚠️ DEEPSEEK_MODEL_FLASH and AUDIT_MODEL_ALLOWLIST are MUTUALLY EXCLUSIVE controls, and
    # setting the first silently disables the second. The engine skips model discovery, the
    # allowlist filter and the responsiveness probe entirely when DEEPSEEK_MODEL_FLASH is set
    # (audit.py: "the pinned env token wins; skip discovery entirely"), and sends that one id
    # whatever the allowlist says. Measured: with the allowlist set to 'ds' AND the pin left
    # at its default, the engine ignored the allowlist and hammered a rate-limited combo
    # model - the exact outcome the allowlist exists to prevent.
    # So: the pin is applied ONLY when no allowlist is configured. An allowlist is the more
    # specific instruction and must win.
    allow = o.get("models") if o.get("models") is not None else cfg.get("modelAllowlist")
    if isinstance(allow, list):
        allow = ",".join(str(x) for x in allow)
    allow = str(allow or "")
    pin = str(o.get("primaryModel") or cfg.get("primaryModel") or "")
    if allow:
        env.pop("DEEPSEEK_MODEL_FLASH", None)   # the allowlist governs; a pin would override it
    elif pin:
        env["DEEPSEEK_MODEL_FLASH"] = pin
    else:
        env.pop("DEEPSEEK_MODEL_FLASH", None)

    # Mapping: setting id -> engine env var.
    # The engine gets the PER-TARGET directory. Passing the shared base here is what let one
    # target's saved pass satisfy another target's resume check.
    if o.get("outputDir"):
        env["AUDIT_OUTPUT_DIR"] = str(o["outputDir"])
    else:
        env["AUDIT_OUTPUT_DIR"] = str(out_dir(tname))
    env["AUDIT_NUM_PASSES"] = str(o.get("passes") or cfg.get("passes"))
    env["AUDIT_BATCH_SIZE"] = str(o.get("batchSize") or cfg.get("batchSize"))
    env["AUDIT_CHAT_TIMEOUT"] = str(cfg.get("chatTimeout"))
    env["AUDIT_PROBE_TIMEOUT"] = str(cfg.get("probeTimeout"))
    env["AUDIT_CHAT_MAX_TOKENS"] = str(cfg.get("chatMaxTokens"))
    env["AUDIT_REQUIRE_SUBSTANTIVE"] = "1" if cfg.get("requireSubstantive") else "0"
    env["AUDIT_MAX_EMPTY_ROTATIONS"] = str(cfg.get("maxEmptyRotations"))
    env["AUDIT_MODEL_ALLOWLIST"] = allow

    env["AUDIT_TARGET_LABEL"] = str(t.get("label") or tname)
    for key, field in (("AUDIT_TARGET_DIR", "dir"), ("AUDIT_README_FILE", "readme"),
                       ("AUDIT_SOT_FILE", "sot"), ("AUDIT_GRAPH_FILE", "graph"),
                       ("AUDIT_INCLUDE_FILES", "includeFiles"), ("AUDIT_EXTRA_ROOTS", "extraRoots"),
                       ("AUDIT_TASK_FILE", "task")):
        v = t.get(field)
        if v:
            env[key] = str(v)
    return env


def _validate(target: str) -> list[str]:
    """Reasons this target cannot be run. Empty means it can."""
    problems = []
    t = config.target_get(target)
    if not t:
        return [f'no target named "{target}"']
    d = t.get("dir") or ""
    if not d:
        problems.append("target has no repository root (dir)")
    elif not Path(d).is_dir():
        problems.append(f"repository root does not exist: {d}")
    inc = t.get("includeFiles") or ""
    if inc:
        for p in [x.strip() for x in inc.split(",") if x.strip()]:
            if not Path(p).exists():
                problems.append(f"listed file does not exist: {p}")
    else:
        g = t.get("readme") or ""
        if g and not Path(g).exists():
            problems.append(f"README does not exist: {g}")
    eng = str(config.resolve("enginePy")[0])
    if not Path(eng).exists():
        problems.append(f"engine not found: {eng}")
    return problems


# ── start / stop ─────────────────────────────────────────────────────────────

def start(target: str | None = None, overrides: dict | None = None, resume: bool | None = None) -> dict:
    tname = target or config.active_target()
    problems = _validate(tname)
    if problems:
        return {"ok": False, "reason": "; ".join(problems), "problems": problems}

    cfg = {s["id"]: config.resolve(s["id"])[0] for s in config.SCHEMA}
    env = build_env(tname, overrides)
    py = find_python()
    engine = str(cfg["enginePy"])
    run_id = time.strftime("%Y%m%d-%H%M%S")
    # One log per run. A shared log mixes runs, so the progress parser reads another run's
    # batches and rounds and reports a run's state as someone else's.
    log = runs_dir() / f"run-{run_id}.log"
    latest = runs_dir() / "last.log"

    # -u keeps the engine's stdout UNBUFFERED. Redirected to a log file, Python block-buffers
    # stdout (4-8KB), so every progress line that does not pass flush=True sits invisible in the
    # buffer - exactly during the long waits you launched the run to watch. Measured 2026-09-25:
    # a run sat silent for 7 minutes at the startup probe with 1047 bytes in the log, and
    # `orch watch` could show nothing at all. Fixing it here covers every print in the engine,
    # not just the ones somebody remembered to flush.
    argv = [py, "-u", engine]
    use_resume = cfg.get("resume") if resume is None else resume
    if use_resume:
        argv.append("--resume")
    argv += ["--limit", str(cfg.get("concurrency") or 8)]

    t = config.target_get(tname) or {}
    label = str(t.get("label") or tname)
    o_dir = Path(env["AUDIT_OUTPUT_DIR"])
    o_dir.mkdir(parents=True, exist_ok=True)

    logfh = open(log, "ab", buffering=0)
    logfh.write(f"\n\n===== orch run: {tname} at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n".encode())
    try:
        # last.log stays as a convenience pointer for anything reading the newest log; the
        # per-run file is the one the record names.
        latest.write_bytes(Path(log).read_bytes())
    except Exception:
        pass
    try:
        proc = subprocess.Popen(
            argv, cwd=str(config.ROOT), env=env,
            stdin=subprocess.DEVNULL, stdout=logfh, stderr=subprocess.STDOUT,
            start_new_session=True,   # detach: survives this CLI and the shell that opened it
        )
    except Exception as e:
        return {"ok": False, "reason": f"could not start the engine: {e}"}
    finally:
        logfh.close()

    rec = {
        "id": run_id,
        "target": tname,
        "label": label,
        "pid": proc.pid,
        "startedAt": time.time(),
        "outputDir": str(o_dir),
        "log": str(log),
        "argv": argv,
        "passes": env.get("AUDIT_NUM_PASSES"),
        "batchSize": env.get("AUDIT_BATCH_SIZE"),
        "resume": bool(use_resume),
        "allowlist": env.get("AUDIT_MODEL_ALLOWLIST"),
        "concurrency": cfg.get("concurrency"),
    }
    _run_file(rec["id"]).write_text(json.dumps(rec, indent=2))
    return {"ok": True, "run": rec}


def stop(run_id: str) -> dict:
    rec = get(run_id)
    if not rec:
        return {"ok": False, "reason": f"no run {run_id}"}
    pid = int(rec.get("pid") or 0)
    if not pid_alive(pid):
        return {"ok": True, "stopped": False, "reason": "already finished"}
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception as e:
            return {"ok": False, "reason": f"could not stop pid {pid}: {e}"}
    for _ in range(20):
        if not pid_alive(pid):
            return {"ok": True, "stopped": True}
        time.sleep(0.25)
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
    return {"ok": True, "stopped": True, "killed": True}


# ── reading state ────────────────────────────────────────────────────────────

def get(run_id: str) -> dict | None:
    f = _run_file(run_id)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def list_runs() -> list[dict]:
    out = []
    for f in sorted(runs_dir().glob("run-*.json")):
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        rec["alive"] = pid_alive(int(rec.get("pid") or 0))
        out.append(rec)
    out.sort(key=lambda r: r.get("startedAt") or 0, reverse=True)
    return out


def current_run() -> dict | None:
    for r in list_runs():
        if r.get("alive"):
            return r
    runs = list_runs()
    return runs[0] if runs else None


BATCH_RE = re.compile(r"\[Pass\s+(\d+)\s+Batch\s+(\d+)/(\d+)\]\s+(\S+)")
ROUND_RE = re.compile(r"Round\s+(\d+)\s+(.+?)\s+->\s+(\w+)\s+\((\d+(?:\.\d+)?)s,\s*(\d+)\s+findings")


def progress(run: dict | None = None) -> dict:
    """Parse the run's log into progress the panel can show.

    The engine writes a line per batch and per round; reading the log is the only way to see
    where a run is without asking it. Numbers the log does not state come back as None rather
    than a guess.
    """
    run = run or current_run()
    if not run:
        return {"running": False, "batches": [], "rounds": [], "findings": 0}
    log = Path(run.get("log") or "")
    text = ""
    if log.exists():
        try:
            text = log.read_text(errors="replace")
        except Exception:
            text = ""
    batches, rounds = [], []
    for m in BATCH_RE.finditer(text):
        batches.append({"pass": int(m.group(1)), "index": int(m.group(2)),
                        "total": int(m.group(3)), "file": m.group(4)})
    for m in ROUND_RE.finditer(text):
        rounds.append({"round": int(m.group(1)), "agent": m.group(2).strip(),
                       "verdict": m.group(3), "seconds": float(m.group(4)),
                       "findings": int(m.group(5))})
    ok = [r for r in rounds if r["verdict"].upper() in ("OK", "PASS")]
    # `get(run_id)` reads the record without the liveness flag that list_runs adds, so a run
    # asked for BY ID reported running=false while its process was healthy. Derive it.
    alive = run["alive"] if "alive" in run else pid_alive(int(run.get("pid") or 0))
    return {
        "running": bool(alive),
        "run": run,
        "batches": batches,
        "currentBatch": batches[-1] if batches else None,
        "rounds": rounds[-12:],
        "roundsTotal": len(rounds),
        "roundsOk": len(ok),
        "findings": sum(r["findings"] for r in ok),
        "lastLine": (text.strip().splitlines() or [""])[-1][:160],
        "tail": (text.strip().splitlines() or [])[-14:],
    }


def batches_on_disk(run: dict | None = None) -> dict:
    run = run or current_run()
    if not run:
        return {}
    base = Path(run.get("outputDir") or "")
    out = {}
    if not base.exists():
        return out
    for pf in sorted(base.glob("pass_*")):
        files = sorted(pf.glob("*.json"))
        total = 0
        for f in files:
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            fbf = d.get("findings_by_file") or {}
            if isinstance(fbf, dict):
                total += sum(len(v) for v in fbf.values() if isinstance(v, list))
            elif isinstance(d.get("findings"), list):
                total += len(d["findings"])
        out[pf.name] = {"files": len(files), "findings": total}
    return out


def findings(run: dict | None = None, limit: int = 400) -> list[dict]:
    """Every finding written to disk for this run, flattened."""
    run = run or current_run()
    if not run:
        return []
    base = Path(run.get("outputDir") or "")
    out = []
    if not base.exists():
        return out
    for pf in sorted(base.glob("pass_*")):
        for f in sorted(pf.glob("*.json")):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            fbf = d.get("findings_by_file")
            if isinstance(fbf, dict):
                for path, lst in fbf.items():
                    for x in (lst or []):
                        if isinstance(x, dict):
                            out.append({**x, "file": path, "pass": pf.name, "batch": f.stem})
            lst = d.get("findings")
            if isinstance(lst, list):
                for x in lst:
                    if isinstance(x, dict):
                        out.append({**x, "pass": pf.name, "batch": f.stem})
            if len(out) >= limit:
                return out[:limit]
    return out


def lane_health() -> list[dict]:
    """Health of each webchat lane the aggregate fronts. Read-only and best-effort."""
    import urllib.request

    ports = {"ds": 8081, "gm": 8085, "cg": 8087}
    out = []
    for name, port in ports.items():
        entry = {"lane": name, "port": port, "ok": False}
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                entry["ok"] = True
                entry["health"] = json.loads(r.read().decode())
        except Exception as e:
            entry["error"] = str(e)[:80]
        out.append(entry)
    return out


def aggregate_health() -> dict:
    import urllib.request

    url = str(config.resolve("aggregateUrl")[0]).rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=4) as r:
            d = json.loads(r.read().decode())
        return {"ok": True, "models": [m.get("id") for m in d.get("data", [])]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:100], "url": url}


_scan_cache: dict = {}

# ── planning: what a target WOULD audit, before starting ──────────────────────
# Knowing the scope before a run is the difference between "8 batches, ~40 minutes" and
# discovering the walk picked up 4,000 files an hour in. This mirrors the engine's include
# rule (an explicit file list beats the walk) so the number it reports is the number the
# engine will use.
# ★ MUST MIRROR engine/audit.py's EXCLUDE_DIRS / EXCLUDE_FILES.
# A scan that disagrees with the engine is worse than no scan: it reported 4,109 files for a
# repo the engine audits about a quarter of, which turned a usable estimate into an alarming
# one (848 hours) and would have made a user think the scope was the whole tree. The lists are
# duplicated rather than imported because importing the engine needs aiohttp; a test compares
# them, so drift is caught instead of trusted.
DEFAULT_EXCLUDES = {
    "archive", "__pycache__", ".git", "node_modules", "build", "dist", "coverage",
    ".next", ".nuxt", ".cache", ".turbo", ".pytest_cache", ".mypy_cache",
    ".dart_tool", "graphify-out", "backups", "vendor", "third_party",
    "oculus_env", ".venv", "venv", "env", "target", "uploads", "logs", ".orch",
    "sandbox", "data", "tmp",
}
DEFAULT_EXCLUDE_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "uv.lock",
    "Cargo.lock", "composer.lock", "graph.json", "tsconfig.json", "jsconfig.json",
    ".eslintcache",
}
CODE_EXT = {".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".dart", ".go", ".rs",
            ".java", ".rb", ".php", ".cs", ".c", ".cc", ".cpp", ".h", ".hpp", ".swift",
            ".kt", ".scala", ".sql", ".sh", ".yaml", ".yml", ".toml", ".json", ".html",
            ".css", ".scss"}


def scan_target(target: str | None = None, on_progress=None, use_cache: bool = False) -> dict:
    """The files a run of this target would audit, and how they batch up.

    `on_progress(n)` is called as files are found, so a caller can show movement instead of a
    still frame: the walk takes seconds on a real repo and a user cannot tell a slow scan from
    a hung one without it. `use_cache` reuses a result from the last 60 seconds, for a UI that
    re-renders on every keypress.
    """
    tname, t = resolve_target(target)
    if use_cache:
        c = _scan_cache.get(tname)
        if c and (time.time() - c["at"]) < 60:
            return c["result"]
    inc = str(t.get("includeFiles") or "")
    if inc:
        files = [f.strip() for f in inc.replace("\n", ",").split(",") if f.strip()]
        missing = [f for f in files if not Path(f).exists()]
        return {"target": tname, "source": "explicit list", "files": len(files),
                "missing": missing, "list": files}
    root = Path(str(t.get("dir") or ""))
    if not root.is_dir():
        return {"target": tname, "source": "walk", "files": 0,
                "error": f"repository root does not exist: {root}"}
    extra = [Path(p.strip()) for p in str(t.get("extraRoots") or "").split(",") if p.strip()]
    found: list[str] = []
    for base in [root] + extra:
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            try:
                if p.is_symlink() or not p.is_file():
                    continue
                rel = p.relative_to(base)
                if any(part in DEFAULT_EXCLUDES for part in rel.parts[:-1]):
                    continue
                if p.name in DEFAULT_EXCLUDE_FILES:
                    continue
                if p.suffix.lower() not in CODE_EXT:
                    continue
                found.append(str(p))
                if on_progress and len(found) % 100 == 0:
                    on_progress(len(found))
            except (OSError, ValueError):
                continue
    if on_progress:
        on_progress(len(found))
    result = {"target": tname, "source": "walk", "files": len(found),
              "roots": [str(root)] + [str(x) for x in extra], "sample": found[:12]}
    _scan_cache[tname] = {"at": time.time(), "result": result}
    return result


def estimate(target: str | None = None, overrides: dict | None = None) -> dict:
    """A rough wall-clock estimate, built from MEASURED round times when a run exists.

    Deliberately approximate and says so: it is a planning aid, not a promise. The number
    that matters is which of its inputs is a guess.
    """
    o = overrides or {}
    batch_size = int(o.get("batchSize") or config.resolve("batchSize")[0] or 5)
    passes = int(o.get("passes") or config.resolve("passes")[0] or 2)
    # Reuse the scan the caller just did: without this the screen walked the whole tree twice
    # (measured 11.6s each), so opening Scope cost ~23s of a frozen screen for one number.
    scan = scan_target(target, use_cache=True)
    n = int(scan.get("files") or 0)
    if not n:
        return {"target": target or config.active_target(), "files": 0, "error": scan.get("error") or "no files found"}

    per_batch_rounds = 8        # 8 specialists + the confirm pass
    batches = max(1, (n + batch_size - 1) // batch_size)

    # Prefer a measured average round time from the most recent run's log; fall back to a
    # labelled assumption rather than presenting a guess as a measurement.
    basis, avg_round_s = "assumed", 60.0
    run = current_run()
    if run:
        rounds = [r["seconds"] for r in progress(run).get("rounds") or [] if r.get("seconds")]
        if rounds:
            avg_round_s = sum(rounds) / len(rounds)
            basis = f"measured from run {run.get('id')} ({len(rounds)} rounds)"

    total_s = batches * passes * per_batch_rounds * avg_round_s
    return {
        "target": target or config.active_target(),
        "files": n, "batchSize": batch_size, "batches": batches, "passes": passes,
        "roundsPerBatch": per_batch_rounds,
        "totalRounds": batches * passes * per_batch_rounds,
        "avgRoundSeconds": round(avg_round_s, 1),
        "basis": basis,
        "estimatedHours": round(total_s / 3600, 2),
        "note": "an estimate, not a promise: it assumes every round succeeds at the average pace "
                "and that no model times out",
    }


# ── process control ──────────────────────────────────────────────────────────

def pause(run_id: str) -> dict:
    """SIGSTOP a run: it keeps its place and its memory, and uses no CPU."""
    rec = get(run_id) or current_run()
    if not rec:
        return {"ok": False, "reason": "no run"}
    pid = int(rec.get("pid") or 0)
    if not pid_alive(pid):
        return {"ok": False, "reason": "the run is not alive"}
    try:
        os.killpg(os.getpgid(pid), signal.SIGSTOP)
    except Exception as e:
        return {"ok": False, "reason": f"could not pause pid {pid}: {e}"}
    return {"ok": True, "paused": True, "run": rec["id"], "pid": pid}


def resume(run_id: str) -> dict:
    rec = get(run_id) or current_run()
    if not rec:
        return {"ok": False, "reason": "no run"}
    pid = int(rec.get("pid") or 0)
    if not pid_alive(pid):
        return {"ok": False, "reason": "the run is not alive"}
    try:
        os.killpg(os.getpgid(pid), signal.SIGCONT)
    except Exception as e:
        return {"ok": False, "reason": f"could not resume pid {pid}: {e}"}
    return {"ok": True, "resumed": True, "run": rec["id"], "pid": pid}


# ── batch surgery ────────────────────────────────────────────────────────────

def batch_files(run: dict | None = None) -> list[dict]:
    run = run or current_run()
    if not run:
        return []
    base = Path(run.get("outputDir") or "")
    out = []
    if not base.exists():
        return out
    for pf in sorted(base.glob("pass_*")):
        for f in sorted(pf.glob("*.json")):
            fbf = {}
            try:
                d = json.loads(f.read_text())
                fbf = d.get("findings_by_file") or {}
            except Exception:
                pass
            n = sum(len(v) for v in fbf.values() if isinstance(v, list)) if isinstance(fbf, dict) else 0
            out.append({"pass": pf.name, "batch": f.stem, "path": str(f),
                        "bytes": f.stat().st_size, "findings": n,
                        "files": sorted(fbf.keys()) if isinstance(fbf, dict) else []})
    return out


def delete_batch(run: dict | None, pass_name: str, batch: str) -> dict:
    """Delete one saved batch so the next --resume run re-audits it.

    This is the documented way to force a re-run: the engine treats a saved batch as done
    and never redoes it, so deleting the file is the only way to re-ask a question. The
    file is MOVED to a .discarded sibling rather than removed, because the findings in it
    are the only copy and the owner may still want them.
    """
    run = run or current_run()
    if not run:
        return {"ok": False, "reason": "no run to operate on"}
    base = Path(run.get("outputDir") or "")
    f = base / pass_name / f"{batch}.json"
    if not f.exists():
        return {"ok": False, "reason": f"no such batch: {pass_name}/{batch}.json"}
    if not str(f.resolve()).startswith(str(base.resolve())):
        return {"ok": False, "reason": "refusing to touch a path outside the run output"}
    dest = f.with_suffix(".json.discarded")
    try:
        if dest.exists():
            dest = f.with_suffix(f".json.discarded-{int(time.time())}")
        f.rename(dest)
    except Exception as e:
        return {"ok": False, "reason": f"could not discard the batch: {e}"}
    return {"ok": True, "discarded": str(dest), "reRun": "start the next run with resume on"}


def rerun(target: str | None = None, overrides: dict | None = None) -> dict:
    """Start a fresh run, stopping whatever is live first."""
    cur = current_run()
    stopped = None
    if cur and cur.get("alive"):
        stopped = stop(cur["id"])
    return {**start(target, overrides), "stoppedPrevious": stopped}


# ── results: the verified set and the report ─────────────────────────────────

def verify_status(run: dict | None = None) -> dict:
    """Read verified_findings.json if the verifier has been run for this output dir."""
    run = run or current_run()
    if not run:
        return {"present": False}
    base = Path(run.get("outputDir") or "")
    for cand in (base / "verified_findings.json", base / "pass_1" / "verified_findings.json"):
        if cand.exists():
            try:
                d = json.loads(cand.read_text())
            except Exception as e:
                return {"present": True, "path": str(cand), "error": f"unparseable: {e}"}
            items = d if isinstance(d, list) else (d.get("findings") or d.get("verified") or [])
            counts: dict[str, int] = {}
            for x in items if isinstance(items, list) else []:
                v = str((x or {}).get("verdict") or "?").upper()
                counts[v] = counts.get(v, 0) + 1
            return {"present": True, "path": str(cand), "total": len(items) if isinstance(items, list) else None,
                    "byVerdict": counts}
    return {"present": False, "lookedIn": str(base)}


def report(run: dict | None = None, max_chars: int = 20000) -> dict:
    """The engine's final report, once it has been written."""
    run = run or current_run()
    if not run:
        return {"present": False}
    base = Path(run.get("outputDir") or "")
    cands = sorted(base.glob("code_audit_*.md")) + sorted(base.glob("multi_agent_oculus_audit_*.md"))
    if not cands:
        return {"present": False, "lookedIn": str(base),
                "hint": "the report is written after the final pass completes"}
    f = cands[-1]
    try:
        txt = f.read_text(errors="replace")
    except Exception as e:
        return {"present": True, "path": str(f), "error": str(e)}
    return {"present": True, "path": str(f), "chars": len(txt), "truncated": len(txt) > max_chars,
            "text": txt[:max_chars]}


def findings_export(run: dict | None = None, fmt: str = "json") -> dict:
    """Write the run's findings to one file in json, markdown or csv."""
    run = run or current_run()
    if not run:
        return {"ok": False, "reason": "no run"}
    fs = findings(run, limit=100000)
    if not fs:
        return {"ok": False, "reason": "no findings on disk yet (a batch saves only when all of its rounds finish)"}
    base = Path(run.get("outputDir") or ".")
    fmt = (fmt or "json").lower()
    if fmt == "md":
        lines = [f"# Findings — {run.get('target')}", f"_{len(fs)} findings_", ""]
        for f in fs:
            lines.append(f"## {f.get('severity') or f.get('criticality') or ''} {f.get('title') or f.get('issue') or '(untitled)'}")
            for k in ("file", "line_range", "category", "confidence"):
                if f.get(k):
                    lines.append(f"- **{k}**: {f[k]}")
            for k in ("description", "detail", "issue", "impact", "recommendation", "fix", "prove"):
                if f.get(k):
                    lines += ["", str(f[k]), ""]
            lines.append("")
        out = base / "findings.md"
        out.write_text("\n".join(lines))
    elif fmt == "csv":
        import csv as _csv
        cols = ["severity", "criticality", "title", "issue", "file", "line_range", "category", "confidence"]
        out = base / "findings.csv"
        with out.open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for f in fs:
                w.writerow({k: str(f.get(k, "")).replace("\n", " ")[:2000] for k in cols})
    else:
        out = base / "findings.json"
        out.write_text(json.dumps(fs, indent=2, default=str))
    return {"ok": True, "path": str(out), "findings": len(fs), "format": fmt}


# ── connectivity ─────────────────────────────────────────────────────────────

def endpoint_probe(timeout: int = 240) -> dict:
    """Time one tiny request through the configured endpoint.

    This is the measurement that decides whether a "slow lane" is slow or broken: a lane
    that answers a trivial prompt is working, and a timeout is the CALLER's problem, not the
    lane's.

    ⚠️ The default is 240s, not 60s, because the gateway injects a deliberate random 20-80s
    wait before EVERY send (an anti-bot measure, `SEND_GAP_MIN_MS`/`MAX`), and generation is
    on top of that. Measured: the same trivial prompt answered in 3.6s once and timed out at
    60s the next time, because the 60s window ended inside the pacing gap. A probe that
    reports "failed" over a lane that is merely inside its pacing window is worse than no
    probe: it sends you hunting a fault that is not there.
    """
    import urllib.request

    url = str(config.resolve("aggregateUrl")[0]).rstrip("/") + "/chat/completions"
    key_name = str(config.resolve("apiKeyName")[0])
    key = str(config.resolve("apiKeyValue")[0])
    allow = config.resolve("modelAllowlist")[0]
    if isinstance(allow, list):
        model = next((str(x).strip() for x in allow if str(x).strip()), "")
    else:
        model = next((x.strip() for x in str(allow or "").split(",") if x.strip()), "")
    if not model:
        # Say why rather than sending a request with an empty model and reporting whatever
        # the endpoint says about it: an empty allowlist is deny-all by design, so a 404 here
        # would send the reader looking at the endpoint instead of at the setting.
        # Every key a caller reads must be present on this path too. Returning a shorter dict
        # threw KeyError on elapsedMs in the CLI, so a clear message turned into a traceback —
        # an early return is still part of the contract.
        return {"ok": False, "model": "", "elapsedMs": 0,
                "error": "no model is configured: the model allowlist is empty, which means "
                         "deny-all. Set one with `orch config set modelAllowlist ds`.",
                "url": url}
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
                       "max_tokens": 10}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"content-type": "application/json", key_name: key})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
        ms = int((time.time() - t0) * 1000)
        content = (((d.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
        return {"ok": True, "model": model, "elapsedMs": ms, "reply": content[:80], "url": url}
    except Exception as e:
        return {"ok": False, "model": model, "elapsedMs": int((time.time() - t0) * 1000),
                "error": str(e)[:200], "url": url}


def logs_list() -> list[dict]:
    d = runs_dir()
    out = []
    for f in sorted(d.glob("*.log")):
        try:
            st = f.stat()
            out.append({"file": str(f), "name": f.name, "bytes": st.st_size,
                        "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime))})
        except OSError:
            continue
    out.sort(key=lambda x: x["modified"], reverse=True)
    return out
