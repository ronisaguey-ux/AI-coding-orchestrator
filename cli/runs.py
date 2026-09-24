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

def out_dir(target_label: str | None = None) -> Path:
    base = str(config.resolve("outputDir")[0])
    return Path(base)


def runs_dir() -> Path:
    d = out_dir() / RUNS_SUBDIR
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

def build_env(target: str | None = None, overrides: dict | None = None) -> dict:
    """Turn config + target into the environment engine/audit.py reads.

    The engine reads everything from the environment. This is the one place that mapping
    lives, so the panel and the MCP cannot drift apart on what a setting means.
    """
    cfg = {s["id"]: config.resolve(s["id"])[0] for s in config.SCHEMA}
    tname = target or config.active_target()
    t = config.target_get(tname) or {}
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
    env["AUDIT_OUTPUT_DIR"] = str(o.get("outputDir") or cfg.get("outputDir"))
    env["AUDIT_NUM_PASSES"] = str(o.get("passes") or cfg.get("passes"))
    env["AUDIT_BATCH_SIZE"] = str(o.get("batchSize") or cfg.get("batchSize"))
    env["AUDIT_CHAT_TIMEOUT"] = str(cfg.get("chatTimeout"))
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

    argv = [py, engine]
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
