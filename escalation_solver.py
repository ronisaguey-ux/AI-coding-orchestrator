#!/usr/bin/env python3
"""escalation_solver.py — autonomous solver for escalated executor steps.

Before this existed, an escalated step wrote a line to ``/tmp/main_wake.log``
and waited for a human.  The 2026-09-05 audit showed that was mostly wasted:
37 of 52 escalations carried **zero edits and no verifier reason**, because the
lane layer returned a bare ``""`` for transport failures and the executor read
that as "the model produced no edits".  Those steps were never broken — the
pipeline just could not tell a dead gateway from a considered answer.

The solver retries escalations through the repaired lane layer in three tiers
of increasing leniency, stopping at the first tier that produces a verified
green:

  tier 1  RETRY    Re-run the original step against healthy lanes with fresh
                   file context.  Clears the transport-failure backlog.
  tier 2  REPAIR   Re-run with the previous failure quoted back (what was
                   applied, what the verifier said, why the anchor missed) so
                   the model can pick a different anchor or create the file.
  tier 3  PERSONA  The escalation persona.  Granted leniency the executor
                   never has: it may rewrite a stale plan step, supply its own
                   verification command, or declare the finding obsolete with
                   evidence — because by tier 3 the likely fault is the STEP,
                   not the code.

Persona leniency is bounded.  ``escalation_allow_plan_edit`` and
``escalation_allow_verify_edit`` gate the two rewrite powers, custom verify
commands pass the :mod:`orch_verify` allowlist, and an ``obsolete`` verdict
must cite evidence that is checked against the filesystem — a step is never
closed on the model's say-so alone.

Runtime traffic uses the FREE lanes only; the paid key is never read here.

    python3 escalation_solver.py --limit 5          # solve five escalations
    python3 escalation_solver.py --dry-run          # plan only, no writes
    python3 escalation_solver.py --step P1B2R0F5    # one specific step
    python3 escalation_solver.py --watch            # drain continuously
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import fnmatch
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import orch_config
import orch_lanes
import orch_verify

LOG_PREFIX = "[esc]"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"{LOG_PREFIX} {ts} {msg}", flush=True)


# ------------------------------------------------------------- personas -----
RETRY_SYSTEM = (
    "You are a code FIX EXECUTOR. Fix the single step described below. "
    "Answer ONE JSON object only: {\"edits\":[{\"file\":\"<repo-relative path>\","
    "\"old_string\":\"<exact existing text>\",\"new_string\":\"<replacement>\"}],"
    "\"notes\":\"<short>\"}. Each old_string must be UNIQUE and byte-exact from "
    "the file contents shown. To create an absent file use old_string \"\" and "
    "put the full content in new_string. No fences, no prose."
)

REPAIR_SYSTEM = (
    "You are a code FIX EXECUTOR retrying a step that FAILED. You are shown "
    "what was attempted and why it failed. Common causes: the old_string was "
    "not byte-exact, the file moved, or the anchor text has changed. Choose a "
    "DIFFERENT, verifiable anchor this time, or create the file if it is "
    "genuinely absent. Answer ONE JSON object only: {\"edits\":[{\"file\":\"...\","
    "\"old_string\":\"...\",\"new_string\":\"...\"}],\"notes\":\"<what you changed "
    "about your approach>\"}. No fences, no prose."
)

PERSONA_SYSTEM = (
    "You are the ESCALATION PERSONA — the last resort for a step that has "
    "already failed repeated normal fix attempts. Assume the PLAN STEP may "
    "itself be wrong: written against a file that has since moved or been "
    "deleted, describing a finding already fixed, or demanding a check that "
    "cannot pass. You have powers the normal executor does not.\n\n"
    "Answer ONE JSON object only, choosing exactly one action:\n\n"
    "1. Normal fix — the code really is broken:\n"
    "   {\"action\":\"fix\",\"edits\":[{\"file\":\"...\",\"old_string\":\"...\","
    "\"new_string\":\"...\"}],\"verify_cmd\":\"<optional check command>\","
    "\"justification\":\"<why this is the right fix>\"}\n\n"
    "2. Rewrite the step — the step targets the wrong file or misstates the "
    "work; correct it and fix the real target:\n"
    "   {\"action\":\"rewrite_step\",\"step_correction\":{\"files\":[\"<correct "
    "paths>\"],\"fix\":\"<corrected instruction>\"},\"edits\":[...],"
    "\"verify_cmd\":\"<optional>\",\"justification\":\"<what the step got wrong>\"}\n\n"
    "3. Obsolete — the finding no longer applies (already fixed, file "
    "intentionally removed, duplicate):\n"
    "   {\"action\":\"obsolete\",\"evidence\":{\"file\":\"<path you checked>\","
    "\"expect\":\"absent\"|\"contains\",\"text\":\"<text that must be present "
    "when expect is contains>\"},\"justification\":\"<why this is moot>\"}\n\n"
    "The evidence you cite for 'obsolete' is CHECKED against the real "
    "filesystem — a claim that does not hold is rejected and the step stays "
    "open, so never guess. verify_cmd must be a single read-only check "
    "(pytest/node --check/python3/grep/git status); shell operators, "
    "redirection and installers are rejected. No fences, no prose."
)


# --------------------------------------------------------------- helpers ----
def is_executor_active() -> bool:
    """Return True if an execution engine process is running."""
    try:
        res = subprocess.run(["pgrep", "-f", "orchestrato[r]|execute_8_27_engin[e]"],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except OSError:
        return False


def read_state(path: Path) -> dict:
    if not path.exists():
        return {"steps": {}}
    lock_path = path.with_suffix(path.suffix + ".lock")
    try:
        with open(lock_path, "a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_SH)
            try:
                if not path.exists():
                    return {"steps": {}}
                data = json.loads(path.read_text(encoding="utf-8"))
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{LOG_PREFIX} state file {path} is corrupt: {exc}")
    except OSError as exc:
        raise SystemExit(f"{LOG_PREFIX} could not read state {path}: {exc}")
    data.setdefault("steps", {})
    return data


def write_state(path: Path, state: dict, backups: bool) -> None:
    """Atomic replace with an optional single rolling backup.

    The executor wrote state with a plain ``write_text``; a crash mid-write
    truncated the file and lost every step result. Never delete the original —
    the backup is copied before the replace.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            if backups and path.exists():
                shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
            tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(state, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def load_plan_index(path: Path, wanted: set[str]) -> dict[str, dict]:
    """Index only the steps we need — the plan is ~12 MB / 14k steps."""
    if not path.exists():
        raise SystemExit(f"{LOG_PREFIX} plan file not found: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    steps = plan.get("steps") or []
    index = {s["finding_id"]: s for s in steps
             if s.get("finding_id") in wanted}
    log(f"plan {path.name}: {len(steps)} steps, matched {len(index)}/{len(wanted)}")
    return index


def file_context(repo: Path, files: list[str], cap: int,
                 max_files: int) -> str:
    blocks = []
    for rel in files[:max_files]:
        p = repo / rel.lstrip("/")
        if not p.exists():
            blocks.append(f"--- {rel} ---\n[FILE DOES NOT EXIST]")
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            blocks.append(f"--- {rel} ---\n[UNREADABLE: {exc}]")
            continue
        if len(text) > cap:
            text = text[:cap] + "\n...[TRUNCATED]"
        blocks.append(f"--- {rel} ---\n{text}")
    return "\n\n".join(blocks) if blocks else "[no files declared]"


# Credential material the solver must never rewrite. An exact-string replace
# proposed by a lane could corrupt live auth for the gateways this box runs, and
# the safety rules forbid touching these regardless. Matched on the basename.
SECRET_GLOBS = (".cookies*", ".env", ".env.*", "*.env", "*.token", "*.key",
                "*.pem", "*credentials*.json", "*secret*")


def unreachable_reason(repo: Path, rel: str) -> str:
    """Why this target can never be edited, or "" if it is a legitimate target.

    Both a gate before spending lane calls and a guard at write time. A step
    whose every target is unreachable is unsolvable by any model: burning three
    tiers on it (as P1B4R0F1 did, ~2 minutes of lane time per pass, forever)
    is pure waste.
    """
    rel = str(rel or "").strip()
    if not rel:
        return "empty path"
    # An absolute target must be refused, not silently reinterpreted: the old
    # lstrip("/") turned a request for /etc/passwd into repo/etc/passwd, which
    # passed confinement while editing a file nobody asked for.
    if PurePath(rel).is_absolute():
        return "absolute path — outside repo, refused"
    try:
        (repo / rel.lstrip("/")).resolve().relative_to(repo.resolve())
    except ValueError:
        return "outside repo — refused"
    name = PurePath(rel).name
    for pattern in SECRET_GLOBS:
        if fnmatch.fnmatch(name, pattern):
            return "credential file — refused"
    return ""


def apply_edits(repo: Path, edits: list, dry_run: bool = False) -> tuple[bool, str, int]:
    """Exact-string replace with an AST guard. Returns (ok, message, applied).

    Unlike the executor's version this reports an EMPTY edit list as a failure.
    ``all([])`` is vacuously true, so an empty list used to be recorded as a
    successful apply — the single largest source of false escalations.
    """
    if not edits:
        return False, "no edits proposed", 0

    results: list[str] = []
    applied = 0
    ok_all = True

    for edit in edits:
        if not isinstance(edit, dict):
            ok_all = False
            results.append("malformed edit entry")
            continue
        rel = str(edit.get("file") or "").lstrip("/")
        old = str(edit.get("old_string") or "")
        new = str(edit.get("new_string") or "")
        if not rel:
            ok_all = False
            results.append("edit missing file")
            continue
        path = repo / rel
        blocked = unreachable_reason(repo, rel)
        if blocked:
            ok_all = False
            results.append(f"{rel}: {blocked}")
            continue

        if not path.exists():
            if old == "" and new:
                if dry_run:
                    results.append(f"{rel}: would create")
                    applied += 1
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(new, encoding="utf-8")
                ok, msg = orch_verify.check_syntax(path)
                if not ok:
                    path.unlink()
                    ok_all = False
                    results.append(f"{rel}: created then reverted — {msg}")
                    continue
                applied += 1
                results.append(f"{rel}: created")
                continue
            ok_all = False
            results.append(f"{rel}: file absent and no content supplied")
            continue

        try:
            cur = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            ok_all = False
            results.append(f"{rel}: read failed: {exc}")
            continue

        if old == "":
            ok_all = False
            results.append(f"{rel}: empty old_string but file exists")
            continue
        if old not in cur:
            ok_all = False
            results.append(f"{rel}: old_string not found (context changed)")
            continue
        if cur.count(old) > 1:
            ok_all = False
            results.append(f"{rel}: old_string matches {cur.count(old)}x — not unique")
            continue

        updated = cur.replace(old, new, 1)
        if dry_run:
            applied += 1
            results.append(f"{rel}: would apply")
            continue

        path.write_text(updated, encoding="utf-8")
        ok, msg = orch_verify.check_syntax(path)
        if not ok:
            path.write_text(cur, encoding="utf-8")  # revert, keep the tree green
            ok_all = False
            results.append(f"{rel}: reverted — {msg}")
            continue
        applied += 1
        results.append(f"{rel}: applied")

    return ok_all and applied > 0, "; ".join(results[:6]), applied


def check_evidence(repo: Path, evidence: dict) -> tuple[bool, str]:
    """Validate an 'obsolete' claim against the filesystem."""
    if not isinstance(evidence, dict):
        return False, "no evidence supplied"
    rel = str(evidence.get("file") or "").lstrip("/")
    expect = str(evidence.get("expect") or "").lower()
    if not rel:
        return False, "evidence names no file"
    path = repo / rel
    if expect == "absent":
        if path.exists():
            return False, f"claimed {rel} is absent, but it exists"
        return True, f"confirmed {rel} is absent"
    if expect == "contains":
        text = str(evidence.get("text") or "")
        if not text:
            return False, "evidence 'contains' with no text"
        if not path.exists():
            return False, f"claimed {rel} contains text, but the file is missing"
        try:
            body = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            return False, f"cannot read {rel}: {exc}"
        if text not in body:
            return False, f"{rel} does not contain the cited text"
        return True, f"confirmed {rel} contains the cited text"
    return False, f"unsupported expect value {expect!r}"


def memory_state() -> tuple[float, float]:
    """Return (percent_used, available_mb) from /proc/meminfo."""
    try:
        fields = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, rest = line.partition(":")
            fields[key] = float(rest.strip().split()[0])
        total = fields.get("MemTotal", 0.0)
        avail = fields.get("MemAvailable", 0.0)
        if total <= 0:
            return 0.0, 0.0
        return (total - avail) / total * 100.0, avail / 1024.0
    except (OSError, ValueError, IndexError):
        return 0.0, 0.0


# ---------------------------------------------------------------- solver ----
class EscalationSolver:
    def __init__(self, cfg, dry_run: bool = False):
        self.cfg = cfg
        self.dry_run = dry_run or bool(cfg.dry_run)
        self.repo = Path(cfg.repo_dir)
        self.state_path = cfg.state_path
        self.queue_path = cfg.queue_path
        self.pool = orch_lanes.LanePool(cfg, log=lambda m: log(m))
        self.audit: list[dict] = []
        self._audit_flushed = 0

    # -- prompt builders ------------------------------------------------
    def _step_block(self, sid: str, step: dict) -> str:
        files = [f for f in (step.get("files") or []) if f]
        ctx = file_context(self.repo, files, int(self.cfg.file_ctx_cap),
                           int(self.cfg.files_per_step))
        return (
            f"STEP {sid}: {step.get('title')}\n"
            f"CATEGORY: {step.get('category')}\n"
            f"FINDING: {step.get('finding')}\n"
            f"FIX GUIDANCE: {step.get('fix')}\n"
            f"MECHANISM: {step.get('mechanism')}\n"
            f"FILES: {', '.join(files) or '(none declared)'}\n\n"
            f"FILE CONTENTS:\n{ctx}\n"
        )

    def _failure_block(self, rec: dict) -> str:
        last = rec.get("last_apply") or {}
        edits = last.get("edits") or []
        tried = (json.dumps(edits)[:1500] if edits else
                 "(none — the lane never answered, so the code is untouched)")
        return (
            f"PREVIOUS ATTEMPTS: {rec.get('rounds', 0)} rounds, all failed.\n"
            f"LAST EDITS TRIED: {tried}\n"
            f"APPLY RESULT: {last.get('apply_msg') or '(none)'}\n"
            f"VERIFIER SAID: {last.get('verify') or '(verifier never answered)'}\n"
            f"LOCAL CHECKS: {str(last.get('checks') or '(none)')[:600]}\n"
        )

    # -- tiers ----------------------------------------------------------
    async def _tier_retry(self, session, sid, step, rec) -> dict:
        res = await self.pool.call(
            session, RETRY_SYSTEM,
            self._step_block(sid, step) + "\nOUTPUT THE EDITS NOW.")
        if not res.ok:
            return {"outcome": "lane_failure", "detail": res.error}
        obj = orch_lanes.parse_json_object(res.content)
        return self._apply_and_verify(sid, step, obj.get("edits") or [],
                                      lane=res.lane, tier="retry")

    async def _tier_repair(self, session, sid, step, rec) -> dict:
        prompt = (self._step_block(sid, step) + "\n" + self._failure_block(rec)
                  + "\nPICK A DIFFERENT ANCHOR AND OUTPUT THE EDITS NOW.")
        res = await self.pool.call(session, REPAIR_SYSTEM, prompt)
        if not res.ok:
            return {"outcome": "lane_failure", "detail": res.error}
        obj = orch_lanes.parse_json_object(res.content)
        return self._apply_and_verify(sid, step, obj.get("edits") or [],
                                      lane=res.lane, tier="repair")

    async def _tier_persona(self, session, sid, step, rec) -> dict:
        prompt = (self._step_block(sid, step) + "\n" + self._failure_block(rec)
                  + "\nDECIDE: fix, rewrite_step, or obsolete. OUTPUT THE JSON NOW.")
        res = await self.pool.call(session, PERSONA_SYSTEM, prompt)
        if not res.ok:
            return {"outcome": "lane_failure", "detail": res.error}
        obj = orch_lanes.parse_json_object(res.content)
        action = str(obj.get("action") or "fix").lower()
        justification = str(obj.get("justification") or "")[:400]

        if action == "obsolete":
            ok, detail = check_evidence(self.repo, obj.get("evidence") or {})
            if not ok:
                return {"outcome": "persona_rejected", "tier": "persona",
                        "detail": f"obsolete claim rejected: {detail}",
                        "justification": justification, "lane": res.lane}
            return {"outcome": "obsolete", "tier": "persona", "detail": detail,
                    "justification": justification, "lane": res.lane}

        step_correction = None
        if action == "rewrite_step":
            if not bool(self.cfg.escalation_allow_plan_edit):
                return {"outcome": "persona_rejected", "tier": "persona",
                        "detail": "plan edits disabled by escalation_allow_plan_edit",
                        "lane": res.lane}
            correction = obj.get("step_correction") or {}
            if isinstance(correction, dict) and correction:
                step_correction = {
                    "files": [str(f) for f in (correction.get("files") or []) if f],
                    "fix": str(correction.get("fix") or "")[:600],
                }
                if step_correction["files"]:
                    step = {**step, "files": step_correction["files"]}

        verify_cmd = obj.get("verify_cmd") or None
        if verify_cmd and not bool(self.cfg.escalation_allow_verify_edit):
            verify_cmd = None

        out = self._apply_and_verify(sid, step, obj.get("edits") or [],
                                     lane=res.lane, tier="persona",
                                     verify_cmd=verify_cmd)
        out["justification"] = justification
        if step_correction:
            out["step_correction"] = step_correction
        return out

    # -- apply + verify -------------------------------------------------
    def _apply_and_verify(self, sid, step, edits, lane, tier,
                          verify_cmd: str | None = None) -> dict:
        files = [f for f in (step.get("files") or []) if f]
        ok, msg, applied = apply_edits(self.repo, edits, self.dry_run)
        if not ok:
            return {"outcome": "apply_failed", "tier": tier, "lane": lane,
                    "detail": msg, "edits": edits}
        if self.dry_run:
            return {"outcome": "dry_run", "tier": tier, "lane": lane,
                    "detail": msg, "edits": edits}
        touched = sorted({str(e.get("file", "")).lstrip("/")
                          for e in edits if isinstance(e, dict) and e.get("file")})
        result = orch_verify.verify_files(self.repo, touched or files,
                                          custom_cmd=verify_cmd)
        return {"outcome": "green" if result.ok else "verify_failed",
                "tier": tier, "lane": lane, "detail": msg,
                "verify": result.summary(), "edits": edits,
                "applied": applied, "verify_cmd": verify_cmd}

    # -- driver ---------------------------------------------------------
    async def solve_step(self, session, sid: str, step: dict,
                         rec: dict) -> dict:
        tiers = [("retry", self._tier_retry),
                 ("repair", self._tier_repair),
                 ("persona", self._tier_persona)]
        attempts: list[dict] = []
        for name, fn in tiers[: max(1, int(self.cfg.escalation_rounds))]:
            log(f"  {sid} tier={name}")
            try:
                out = await fn(session, sid, step, rec)
            except Exception as exc:  # a solver bug must not kill the drain
                out = {"outcome": "solver_error", "tier": name,
                       "detail": f"{type(exc).__name__}: {exc}"}
            attempts.append(out)
            log(f"  {sid} tier={name} -> {out['outcome']} "
                f"({str(out.get('detail'))[:110]})")
            if out["outcome"] in ("green", "obsolete", "dry_run"):
                return {"final": out["outcome"], "attempts": attempts,
                        "resolved_by": name, "last": out}
        return {"final": "unresolved", "attempts": attempts,
                "resolved_by": None, "last": attempts[-1] if attempts else {}}

    async def run(self, limit: int | None, only: str | None,
                  watch: bool = False) -> int:
        import aiohttp

        if is_executor_active():
            log("executor running — skipping pass to avoid collision")
            return 0

        state = read_state(self.state_path)
        steps = state["steps"]
        escalated = [k for k, v in steps.items()
                     if v.get("status") == "escalated"]
        if only:
            escalated = [s for s in escalated if s == only]
            if not escalated:
                log(f"step {only} is not in escalated status — nothing to do")
                return 0
        if not escalated:
            log("no escalated steps — backlog is clear")
            return 0

        batch = escalated[:limit] if limit else escalated[
            : int(self.cfg.escalation_batch)]
        log(f"backlog {len(escalated)} escalated; this pass handles {len(batch)}")
        log(f"mode: {'DRY-RUN (no writes)' if self.dry_run else 'LIVE'} | "
            f"policy={self.cfg.escalation_policy} | repo={self.repo}")

        index = load_plan_index(self.cfg.plan_path, set(batch))
        resolved = 0

        timeout = aiohttp.ClientTimeout(total=int(self.cfg.lane_timeout) + 60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if bool(self.cfg.lane_health_probe):
                health = await self.pool.probe(session)
                if not any(health.values()):
                    log("ABORT: no lane is answering — refusing to burn the backlog")
                    return 2

            for sid in batch:
                pct, avail_mb = memory_state()
                min_avail = float(self.cfg.get("min_avail_mb", 600))
                if avail_mb < min_avail or pct > float(self.cfg.ram_pct_cap):
                    log(f"memory pressure ({pct:.0f}% used, {avail_mb:.0f}MB "
                        f"available; floor {min_avail:.0f}MB / ceiling "
                        f"{self.cfg.ram_pct_cap}%) — stopping this pass early")
                    break

                step = index.get(sid)
                if step is None:
                    log(f"{sid}: NOT IN PLAN — cannot resolve, leaving escalated")
                    self.audit.append({"step": sid, "final": "not_in_plan"})
                    continue

                targets = [f for f in (step.get("files") or []) if f]
                blocks = {f: unreachable_reason(self.repo, f) for f in targets}
                if targets and all(blocks.values()):
                    why = "; ".join(f"{f}: {r}" for f, r in blocks.items())
                    log(f"{sid}: UNREACHABLE TARGETS — {why[:160]}")
                    self.audit.append({
                        "step": sid,
                        "title": str(step.get("title"))[:160],
                        "final": "blocked",
                        "reason": why[:400],
                        "ts": datetime.now(timezone.utc).isoformat(),
                    })
                    if not self.dry_run:
                        rec = steps[sid]
                        rec["status"] = "blocked"
                        rec["resolved_by"] = "escalation_solver:preflight"
                        rec["escalation_result"] = {"tier": "preflight",
                                                    "reason": why[:400]}
                        write_state(self.state_path, state,
                                    bool(self.cfg.state_backups))
                        self._write_queue()
                    continue

                log(f"solving {sid}: {str(step.get('title'))[:90]}")
                rec = steps[sid]
                outcome = await self.solve_step(session, sid, step, rec)
                entry = {
                    "step": sid,
                    "title": str(step.get("title"))[:160],
                    "final": outcome["final"],
                    "resolved_by": outcome["resolved_by"],
                    "attempts": outcome["attempts"],
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                self.audit.append(entry)

                if self.dry_run:
                    continue

                last = outcome["last"]
                if outcome["final"] == "green":
                    rec["status"] = "green"
                    rec["resolved_by"] = f"escalation_solver:{outcome['resolved_by']}"
                    rec["escalation_result"] = {
                        "tier": last.get("tier"), "lane": last.get("lane"),
                        "verify": str(last.get("verify"))[:600],
                        "verify_cmd": last.get("verify_cmd"),
                        "justification": last.get("justification"),
                        "step_correction": last.get("step_correction"),
                    }
                    resolved += 1
                elif outcome["final"] == "obsolete":
                    rec["status"] = "obsolete"
                    rec["resolved_by"] = "escalation_solver:persona"
                    rec["escalation_result"] = {
                        "tier": "persona", "evidence": last.get("detail"),
                        "justification": last.get("justification"),
                    }
                    resolved += 1
                else:
                    rec["escalation_attempts"] = rec.get("escalation_attempts", 0) + 1
                    rec["last_escalation_try"] = {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "final": outcome["final"],
                        "detail": str(last.get("detail"))[:300],
                    }
                write_state(self.state_path, state, bool(self.cfg.state_backups))
                # Flush the audit trail per step, not at exit: a systemd stop or
                # a timeout mid-pass used to discard every record but the ones
                # already on disk, leaving solved steps with no explanation.
                self._write_queue()

        self._write_queue()
        counts: dict[str, int] = {}
        for e in self.audit:
            counts[e["final"]] = counts.get(e["final"], 0) + 1
        log(f"pass complete: {counts}")
        log(f"lane health: {self.pool.health_report()}")
        log(f"resolved {resolved}/{len(batch)} this pass")
        return 0

    def _write_queue(self) -> None:
        """Append audit entries not yet on disk. Safe to call after every step."""
        if self.dry_run:
            return
        pending = self.audit[self._audit_flushed:]
        if not pending:
            return
        try:
            existing = []
            if self.queue_path.exists():
                existing = json.loads(self.queue_path.read_text(encoding="utf-8"))
                if not isinstance(existing, list):
                    existing = []
            existing.extend(pending)
            tmp = self.queue_path.with_suffix(f".tmp{os.getpid()}")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(existing[-500:], fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.queue_path)
            self._audit_flushed = len(self.audit)
            log(f"audit trail -> {self.queue_path} ({len(pending)} new)")
        except (OSError, json.JSONDecodeError) as exc:
            log(f"could not write audit trail: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="escalated steps to handle this pass")
    ap.add_argument("--step", default=None, help="solve one specific step id")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print without writing files or state")
    ap.add_argument("--watch", action="store_true",
                    help="keep draining the backlog until it is empty")
    ap.add_argument("--interval", type=int, default=300,
                    help="seconds between passes in --watch mode")
    ap.add_argument("--profile", default=None, help="ORCH_PROFILE preset")
    ap.add_argument("--print-config", action="store_true",
                    help="print the resolved config and exit")
    args = ap.parse_args()

    try:
        cfg = orch_config.load(
            overrides={"dry_run": True} if args.dry_run else None,
            profile=args.profile)
    except ValueError as exc:
        print(f"{LOG_PREFIX} config error: {exc}", file=sys.stderr)
        return 2

    if args.print_config:
        print(cfg.render())
        return 0

    solver = EscalationSolver(cfg, dry_run=args.dry_run)
    if not args.watch:
        return asyncio.run(solver.run(args.limit, args.step))

    while True:
        rc = asyncio.run(EscalationSolver(cfg, dry_run=args.dry_run)
                         .run(args.limit, args.step))
        if rc == 2:
            log(f"lanes down — sleeping {args.interval}s before retrying")
        state = read_state(cfg.state_path)
        remaining = sum(1 for v in state["steps"].values()
                        if v.get("status") == "escalated")
        log(f"remaining escalated: {remaining}")
        if remaining == 0:
            log("backlog drained")
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
