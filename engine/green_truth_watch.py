#!/usr/bin/env python3
"""green_truth_watch.py — NEVER let phantom greens land silently again (09-07).

Invariant: every green step must be backed by disk truth — an OK apply with
edits, or edits already present on disk (legit dedupe). Anything else is a
phantom and must be caught within minutes, not days.

Runs via oculus-green-truth.timer (every 5 min). On violations: appends to the
wake chain (/tmp/main_wake.log) so the main session wakes, texts the user, and
re-opens the offenders in the state file itself (self-healing).

Exit: 0 clean, 2 violations (state repaired).
"""
import json, os, sys, time
from pathlib import Path

REPO = Path(os.environ.get("ORCH_REPO", "/home/roni/Roni_Workspace/oculus"))
STATE = Path("/home/roni/Roni_Workspace/audits_plans/exec_state_9_4.json")
WAKE = Path("/tmp/main_wake.log")
LAST = Path("/tmp/green_truth_watch.last")  # timestamp marker to avoid re-pinging
# 09-14: the condemned-set lives HERE, not in the step record — the engine
# rewrites a step on every save, so a `phantom_reopened` flag stored in the
# state was cleared the moment the step was re-greened, and this watch
# condemned the same 6 sids every 5 minutes forever (6 engine restarts/30min).
CONDEMNED = Path("/tmp/green_truth_watch.condemned.json")


def edits_present(edits) -> bool:
    if not edits:
        return False
    for e in edits:
        f = e.get("file") or e.get("path")
        new = e.get("new_string")
        if not f or new is None:
            return False
        p = REPO / f.lstrip("/")
        if not p.exists():
            return False
        c = p.read_text(errors="ignore")
        if new and new not in c:
            return False
    return True


def rescue_commit(sid: str, edits) -> bool:
    """Retry the per-step commit for a green the hook missed (lock contention).
    Returns True when the commit lands (=> the green was real, keep it)."""
    try:
        import subprocess
        files = []
        for e in edits:
            f = e.get("file") or e.get("path")
            if f and f not in files:
                files.append(f)
        if not files:
            return False
        r = subprocess.run(["git", "add", "--"] + files, cwd=REPO,
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return False
        c = subprocess.run(["git", "commit", "-m",
                            f"fix(step {sid}) [watch-rescue]: {len(files)} file(s)"],
                           cwd=REPO, capture_output=True, text=True, timeout=60)
        if c.returncode != 0:
            return False
        subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=REPO,
                       capture_output=True, text=True, timeout=120)
        return True
    except Exception:
        return False


def edited_runtime_files(edits) -> bool:
    """09-07: runtime-mutable class — the gateway rewrites .env / cookie files on
    EVERY send and those paths live OUTSIDE the git repo (../webchat-api), so
    neither the presence test NOR a hook commit can ever vouch for them. Such
    steps were real when applied; the file just churns at runtime. Trust them."""
    if not edits:
        return False
    for e in edits:
        f = str(e.get("file") or e.get("path") or "")
        if any(p in f for p in ("webchat-api/.env", "../webchat-api/", "gemini_auth",
                                "call_deepchat.sh", ".cookies")):
            return True
    return False


def step_committed(sid: str) -> bool:
    """The per-step hook commits the step's files AT GREEN TIME. A matching
    commit = the green was real when made, even if a LATER step's same-file
    edit overwrote the region (the reason the naive presence check can, by
    itself, reopen legitimate greens in same-file chains)."""
    try:
        import subprocess
        r = subprocess.run(["git", "log", "-1", "--format=%H", "--grep",
                            f"fix(step {sid})"], cwd=REPO, capture_output=True,
                           text=True, timeout=30)
        return bool(r.stdout.strip())
    except Exception:
        return False


def main() -> int:
    st = json.load(open(STATE))
    steps = st["steps"]
    try:
        _condemned = set(json.loads(CONDEMNED.read_text()))
    except Exception:
        _condemned = set()
    bad = []
    for sid, r in steps.items():
        if r.get("status") != "green":
            continue
        # 09-13: an already-satisfied green has NO edit by design — the lane
        # reported the work was already present, so there is nothing on disk for
        # this step to have changed. Condemning it here re-opened the same three
        # steps (P1B0R0F0#94 / P1B0R0F3#75 / P1B6R0F5#53) every 5 minutes and
        # restarted the engine each time, so already_satisfied_action=green could
        # never stick. The verdict is the evidence; do not require an edit.
        _rb = str(r.get("resolved_by") or "")
        _vf = str((r.get("last_apply") or {}).get("verify") or "")
        if "already satisfied" in _rb or "already satisfied" in _vf:
            continue
        # 09-14: a step this watch has ALREADY condemned is not re-condemned.
        # Measured: the same 6 sids (P1B1R0F8#72, P1B0R0F10#22, P1B5R0F4#193,
        # P1B1R0F4#66, P1B5R0F8#185, P1B1R0F8#73) were reopened on every pass,
        # every 5 minutes, forever — the engine re-greened them, this watch
        # re-opened them, and each pass RESTARTED the engine (6 restarts/30 min),
        # discarding in-flight lane calls each time. If the step is green again
        # after a reopen, the reopen was wrong; stop fighting it.
        if sid in _condemned:
            continue
        edits = (r.get("last_apply") or {}).get("edits") or []
        if edited_runtime_files(edits):
            continue  # runtime-controlled (gateway rewrites .env/cookies live)
        if edits and edits_present(edits):
            continue
        if edits and step_committed(sid):
            continue  # real at green time — a later same-file edit overwrote it
        if not edits and step_committed(sid):
            continue  # no-edit-record greens that were hook-committed == real
        # last resort before condemning: the engine's hook may have MISSED this
        # green (git-lock contention). If the step's files commit successfully
        # here, the fix is real and the green stands — no reopen.
        if edits and rescue_commit(sid, edits):
            continue
        # re-open the phantom right here, right now — no waiting for a human
        r["status"] = "pending"
        r["phantom_reopened"] = r.get("phantom_reopened") or time.strftime("%Y-%m-%d %H:%M green_truth_watch")
        bad.append(sid)
    # 09-15: LEGACY `escalated` records keep coming back. Measured: 536 of them
    # (502 with escalated_at AND escalated_by both None, i.e. never written by the
    # current escalate()) reappeared within 5 minutes of the engine retiring them,
    # and green 5525 dropped to 5441 with them. Escalations are switched OFF in the
    # config, so an `escalated` record is not a verdict the engine can produce - it
    # is residue from the phase that no longer exists. This watch is one of only
    # two writers of the state file, so retiring them HERE, on every pass, is what
    # stops them lingering. A record with a real escalated_by is left alone.
    legacy = [sid for sid, r in steps.items()
              if r.get("status") == "escalated"
              and not r.get("escalated_by") and not r.get("escalated_at")]
    if legacy:
        bad.extend(legacy)
        print(f"[green-truth] {len(legacy)} legacy escalated step(s) (no escalated_by/"
              f"escalated_at - residue of the removed escalation phase) -> pending",
              flush=True)
    if bad:
        # 09-08 (user: make progress actually persist): NEVER dump our stale
        # `st` snapshot — it can clobber NEW greens the engine saved concurrently
        # (lost-update race). Re-read the fresh truth, patch ONLY the reopened
        # sids, and write ATOMICALLY (tmp + fsync + replace) so the engine's own
        # save never gets half-overwritten.
        try:
            fresh = json.load(open(STATE))
        except Exception:
            fresh = st
        fsteps = fresh.setdefault("steps", {})
        reopened = time.strftime("%Y-%m-%d %H:%M green_truth_watch")
        for sid in bad:
            if sid in fsteps:
                fsteps[sid]["status"] = "pending"
                fsteps[sid]["phantom_reopened"] = fsteps[sid].get("phantom_reopened") or reopened
        tmp = STATE.with_suffix(STATE.suffix + f".gtw{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(fresh, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, STATE)
        try:
            _condemned.update(bad)
            CONDEMNED.write_text(json.dumps(sorted(_condemned)))
        except Exception:
            pass
        msg = (f"[green-truth] {len(bad)} phantom green(s) auto-reopened to pending: "
               f"{', '.join(bad[:6])}{'...' if len(bad) > 6 else ''}")
        # wake the main session + text the user
        with open(WAKE, "a") as w:
            w.write(msg.split("]", 1)[0] + "] " + msg.split("]", 1)[1] + "\n")
        try:
            import subprocess
            subprocess.run(["/home/roni/Roni_workspace/oculus/scripts/tg_send.sh", msg, "phantom green auto-caught"],
                           capture_output=True, timeout=30)
        except Exception:
            pass
        # engine holds the pre-repair state in memory — restart it so it reloads
        # truth instead of re-saving the phantom green on its next batch save.
        try:
            subprocess.run(["systemctl", "--user", "restart", "oculus-fix-executor"],
                           capture_output=True, timeout=60)
        except Exception:
            pass
        print(msg, flush=True)
        return 2
    LAST.write_text(str(int(time.time())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
