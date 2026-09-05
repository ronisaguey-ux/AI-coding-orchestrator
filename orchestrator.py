#!/usr/bin/env python3
"""orchestrator.py — multi-lane fix-executor for audit and remediation plans.

Executes a master plan JSON against a repository using multi-lane LLM
routing (local gateways, free models, or API endpoints), with per-step
verification (AST/pytest/syntax), atomic state tracking, and automatic
escalation of difficult steps.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orch_config
import orch_lanes
import orch_verify

CFG = orch_config.load()
BASE = Path(CFG.base_dir)
PLAN_FILE = CFG.plan_path
STATE_FILE = CFG.state_path
REPO = Path(CFG.repo_dir)

WORKERS = int(CFG.workers)
MAX_ROUNDS = int(CFG.max_rounds)
GROUP_CAP = int(CFG.group_cap)
PARALLEL = int(CFG.parallel)
SAVE_LOCK = asyncio.Semaphore(1)

EXEC_SYSTEM = (
    "You are a code FIX EXECUTOR for the repository. The plan below "
    "contains 1-5 INDEPENDENT steps (disjoint files — no edits reference another step's file). "
    "Fix EVERY step in ONE reply. "
    "Answer ONE JSON object only: {\"edits\":[{\"step\":\"<step id, e.g. STEP 1 or P1B1R0F12>\","
    "\"file\":\"<repo-relative path>\",\"old_string\":\"<exact existing text>\",\"new_string\":"
    "\"<replacement text>\"}],\"notes\":\"<short>\"}. EVERY edit must carry the step id it "
    "belongs to; leave the step id off only when only ONE step is present. "
    "Each old_string must be UNIQUE and byte-exact from the file contents. Use full function "
    "bodies when replacing functions. If the target FILE IS ABSENT and the fix requires creating "
    "it, emit {\"file\":\"...\",\"old_string\":\"\",\"new_string\":\"<full file content>\"}. "
    "Batch as much real work into this ONE turn as the steps allow — no prose about it, just do it. "
    "No fences, no prose, no submit_answer. If the plan conflicts with reality, "
    "put empty edits and explain in notes."
)
VERIFY_SYSTEM = (
    "You are the SELF-VERIFIER for the step you just fixed. You receive your edits + the test/check "
    "output. Answer ONE JSON object only: {\"verdict\":\"green\"} when your change is correct and "
    "no NEW failure is attributable to it; otherwise {\"verdict\":\"red\",\"reason\":\"<terse>\","
    "\"edits\":[{\"file\":\"...\",\"old_string\":\"...\",\"new_string\":\"...\"}]} with corrected "
    "edits. PREEXISTING/ignorable output lines are not your fault. No fences."
)


def load_state() -> dict:
    if STATE_FILE.exists():
        lock_path = STATE_FILE.with_suffix(STATE_FILE.suffix + ".lock")
        try:
            with open(lock_path, "a+", encoding="utf-8") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_SH)
                try:
                    if STATE_FILE.exists():
                        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
    return {"steps": {}}


def save_state(st: dict) -> None:
    """Atomic write: temp file + fsync + replace, with a rolling backup."""
    if getattr(CFG, "dry_run", False):
        return
    lock_path = STATE_FILE.with_suffix(STATE_FILE.suffix + ".lock")
    try:
        with open(lock_path, "a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                if CFG.state_backups and STATE_FILE.exists():
                    try:
                        STATE_FILE.with_suffix(STATE_FILE.suffix + ".bak").write_bytes(
                            STATE_FILE.read_bytes())
                    except OSError:
                        pass
                tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + f".tmp{os.getpid()}")
                with open(tmp, "w", encoding="utf-8") as fh:
                    json.dump(st, fh)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, STATE_FILE)
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


async def save_state_serialized(st: dict) -> None:
    """save_state under SAVE_LOCK to protect against concurrent writes."""
    async with SAVE_LOCK:
        save_state(st)


def signal_escalation(sid: str, rec: dict) -> None:
    """Signal an escalation to the wake log for external monitor notification."""
    if getattr(CFG, "dry_run", False):
        return
    try:
        la = rec.get("last_apply") or {}
        with open(CFG.wake_log, "a", encoding="utf-8") as f:
            f.write(f"[escalation] EXEC step {sid} escalated — "
                    f"edits: {json.dumps(la.get('edits'))[:240]} | "
                    f"verify: {str(la.get('verify', ''))[:240]} | "
                    f"checks: {str(la.get('checks', ''))[:200]} | "
                    f"state: {STATE_FILE}\n")
    except Exception:
        pass


def file_text(rel: str, size_cap: int = 24000) -> str:
    p = REPO / rel.lstrip("/")
    try:
        t = p.read_text(encoding="utf-8", errors="ignore")
        if len(t) > size_cap:
            t = t[:size_cap] + "\n...[TRUNCATED]"
        return t
    except Exception:
        return ""


POOL = orch_lanes.LanePool(CFG)


async def lane_call_result(session, system: str, user: str):
    """Return a LaneResult. Check .ok before trusting .content."""
    return await POOL.call(session, system, user)


async def lane_call(session, system: str, user: str, retries: int = 4) -> str:
    """Backwards-compatible string form. Prefer lane_call_result."""
    res = await POOL.call(session, system, user)
    return res.content if res.ok else ""


def parse_json(text) -> dict:
    """Balanced-object JSON extraction."""
    return orch_lanes.parse_json_object(text)


def apply_edits(edits: list) -> tuple:
    """Exact string replace with py syntax guard."""
    import ast
    if not edits:
        return False, "no edits proposed (lane returned nothing usable)"
    results = []
    for e in edits or []:
        f = str(e.get("file") or "")
        old = str(e.get("old_string") or "")
        new = str(e.get("new_string") or "")
        path = REPO / f.lstrip("/")
        # CREATE: absent file + empty old_string -> write new_string as full content
        if not path.exists() and old == "" and new:
            if f.endswith(".py") and not f.endswith(".py.in"):
                try:
                    ast.parse(new)
                except SyntaxError as se:
                    results.append({"file": f, "ok": False,
                                    "msg": f"new file failed syntax check: {se}"})
                    continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new, encoding="utf-8", errors="ignore")
            results.append({"file": f, "ok": True, "msg": "created"})
            continue
        if old == "" and new:
            results.append({"file": f, "ok": False,
                            "msg": "empty old_string but file exists (context changed)"})
            continue
        try:
            cur = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as ex:
            results.append({"file": f, "ok": False, "msg": f"read fail: {ex}"})
            continue
        if old and old in cur:
            if f.endswith(".py"):
                try:
                    ast.parse(cur.replace(old, new, 1))
                except SyntaxError as se:
                    results.append({"file": f, "ok": False,
                                    "msg": f"syntax break (edit rejected): {se}"})
                    continue
            path.write_text(cur.replace(old, new, 1), encoding="utf-8", errors="ignore")
            results.append({"file": f, "ok": True, "msg": "applied"})
        else:
            results.append({"file": f, "ok": False,
                            "msg": "old_string not found (context changed)"})
    return all(r["ok"] for r in results), "; ".join(
        f"{r['file']}:{r['msg']}" for r in results[:4])


def run_checks(files: list) -> str:
    """Local verification per step: syntax check each touched file + git state."""
    result = orch_verify.verify_files(REPO, files or [])
    outs = [result.report]
    r = subprocess.run(["git", "status", "--short"], capture_output=True, text=True,
                       timeout=60, check=False, cwd=str(REPO))
    outs.append(f"git status:\n{r.stdout[:800]}")
    outs.append(f"VERDICT: {'checks passed' if result.ok else 'CHECKS FAILED'}")
    return "\n".join(outs)


async def run_step(session, step: dict, st: dict) -> dict:
    sid = step["finding_id"]
    rec = st["steps"].get(sid)
    if rec is None:
        rec = {"rounds": 0, "status": "pending"}
        st["steps"][sid] = rec
    files = [x for x in (step.get("files") or []) if x]
    ctx = "\n\n".join(file_text(f) for f in files[:2])
    plan_user = (
        f"STEP {sid}: {step.get('title')}\nSTATE: {step.get('finding')}\n"
        f"FIX GUIDANCE: {step.get('fix')}\nMECHANISM: {step.get('mechanism')}\n"
        f"FILES: {', '.join(files)}\n\nFILE CONTENTS:\n{ctx}\n"
    )
    for rnd in range(rec["rounds"], MAX_ROUNDS):
        rec["rounds"] = rnd + 1
        rec["status"] = "executing"
        await save_state_serialized(st)
        res = await lane_call_result(session, EXEC_SYSTEM,
                                     plan_user + "\nOUTPUT THE EDITS NOW.")
        if not res.ok:
            rec["status"] = "pending"
            rec["last_lane_error"] = res.error[:300]
            await save_state_serialized(st)
            print(f"[step {sid}] lanes unavailable ({res.error[:90]}) — left pending", flush=True)
            return rec
        ex = parse_json(res.content)
        edits = ex.get("edits") or []
        ok, msg = apply_edits(edits)
        rec["last_apply"] = {"rnd": rnd + 1, "ok": ok, "edits": edits,
                             "apply_msg": msg, "lane": res.lane}
        checks = run_checks(files)
        vres = await lane_call_result(
            session, VERIFY_SYSTEM,
            f"STEP {sid}: your edits {json.dumps(edits)[:3000]}\n"
            f"CHECK OUTPUT:\n{checks}\n\nRUN VERIFICATION NOW.")
        if not vres.ok:
            rec["status"] = "pending"
            rec["last_lane_error"] = f"verify lane: {vres.error[:280]}"
            await save_state_serialized(st)
            print(f"[step {sid}] verify lane unavailable — left pending", flush=True)
            return rec
        v = parse_json(vres.content)
        if v.get("verdict") == "green":
            rec["status"] = "green"
            rec["last_apply"]["checks"] = checks[-600:]
            await save_state_serialized(st)
            return rec
        rec["last_apply"]["verify"] = v.get("reason", "")
        new_edits = v.get("edits") or edits
        okay2, msg2 = apply_edits(new_edits)
        plan_user = (f"STEP {sid}: corrected attempt\nFILES: {', '.join(files)}\n"
                     f"VERIFY SAID: {v.get('reason','')}\n\nFILE CONTENTS:\n"
                     f"{ctx}\n")
        rec["last_apply"] = {"rnd": rnd + 1, "ok": okay2, "edits": new_edits,
                             "apply_msg": msg2, "lane": res.lane}
    rec["status"] = "escalated"
    signal_escalation(sid, rec)
    await save_state_serialized(st)
    return rec


async def run_group(session, steps: list, st: dict) -> dict:
    """Run up to GROUP_CAP independent steps in a single model turn."""
    if len(steps) == 1:
        return await run_step(session, steps[0], st)
    plan, step_ids = [], []
    for i, s in enumerate(steps, 1):
        sid = s["finding_id"]
        rec = st["steps"].get(sid)
        if rec is None:
            rec = {"rounds": 0, "status": "pending"}
            st["steps"][sid] = rec
        rec["rounds"] = rec.get("rounds", 0) + 1
        rec["status"] = "executing"
        files = [x for x in (s.get("files") or []) if x]
        ctx = "\n\n".join(file_text(f) for f in files[:2])
        plan.append(
            f"STEP {i} ({sid}): {s.get('title')}\nSTATE: {s.get('finding')}\n"
            f"FIX GUIDANCE: {s.get('fix')}\nMECHANISM: {s.get('mechanism')}\n"
            f"FILES: {', '.join(files)}\n\nFILE CONTENTS:\n{ctx}\n"
        )
        step_ids.append((str(i), sid, files))
    plan_user = "\n\n".join(plan) + "\nOUTPUT THE EDITS NOW — tag every edit with STEP N."
    gres = await lane_call_result(session, EXEC_SYSTEM, plan_user)
    if not gres.ok:
        for _idx, sid, _files in step_ids:
            rec = st["steps"][sid]
            rec["status"] = "pending"
            rec["rounds"] = max(0, rec.get("rounds", 1) - 1)
            rec["last_lane_error"] = gres.error[:300]
        await save_state_serialized(st)
        print(f"[group] lanes unavailable ({gres.error[:90]}) — {len(step_ids)} steps left pending", flush=True)
        return {sid: "pending" for _i, sid, _f in step_ids}
    ex = parse_json(gres.content)
    edits = [e for e in (ex.get("edits") or []) if isinstance(e, dict)]
    by_step = {sid: [] for _, sid, _ in step_ids}
    for e in edits:
        tag = str(e.get("step") or "").replace("STEP", "").strip().strip("()").strip()
        matched = None
        for idx, sid, _ in step_ids:
            if tag in (idx, sid, sid.split("#")[0]):
                matched = sid
                break
        if matched is None:
            matched = step_ids[0][1] if len(step_ids) == 1 else None
        if matched:
            by_step[matched].append(e)
    for idx, sid, files in step_ids:
        rec = st["steps"][sid]
        mine = by_step[sid]
        ok, msg = apply_edits(mine)
        rec["last_apply"] = {"rnd": rec["rounds"], "ok": ok, "edits": mine, "apply_msg": msg}
        checks = run_checks(files)
        rec["last_apply"]["checks"] = checks[-600:]
        vres = await lane_call_result(
            session, VERIFY_SYSTEM,
            f"STEP {sid}: your edits {json.dumps(mine)[:3000]}\n"
            f"CHECK OUTPUT:\n{checks}\n\nRUN VERIFICATION NOW.")
        if not vres.ok:
            rec["status"] = "pending"
            rec["last_lane_error"] = f"verify lane: {vres.error[:280]}"
            continue
        v = parse_json(vres.content)
        if v.get("verdict") == "green":
            rec["status"] = "green"
        else:
            new_edits = v.get("edits") or mine
            _, msg2 = apply_edits(new_edits)
            rec["last_apply"] = {"rnd": rec["rounds"], "ok": ok,
                                 "edits": new_edits, "apply_msg": msg2}
            if rec["rounds"] >= MAX_ROUNDS:
                rec["status"] = "escalated"
                signal_escalation(sid, rec)
            else:
                rec["status"] = "pending"
    await save_state_serialized(st)
    return {s["finding_id"]: st["steps"].get(s["finding_id"], {}).get("status") for s in steps}


def pack_steps(steps: list) -> list:
    """Density pack with disjoint file targets."""
    packed, group, group_files = [], [], set()
    for s in steps:
        f = frozenset(x for x in (s.get("files") or []) if x)
        if group and (len(group) >= GROUP_CAP or (f & group_files)):
            packed.append(group)
            group, group_files = [], set()
        group.append(s)
        group_files |= f
    if group:
        packed.append(group)
    return packed


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--resume", action="store_true", help="Resume from current state")
    ap.add_argument("--batch", type=int, default=None, help="1-based pack batch")
    ap.add_argument("--only-step", default=None, help="Process one specific finding id")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of steps to process")
    ap.add_argument("--dry-run", action="store_true", help="Resolve config and plan without side effects")
    ap.add_argument("--print-config", action="store_true", help="Print resolved config and exit")
    ap.add_argument("--profile", default=None, help="ORCH_PROFILE preset")
    args = ap.parse_args()

    global CFG, BASE, PLAN_FILE, STATE_FILE, REPO
    if args.dry_run or args.profile:
        overrides = {"dry_run": True} if args.dry_run else None
        CFG = orch_config.load(overrides=overrides, profile=args.profile)
        BASE = Path(CFG.base_dir)
        PLAN_FILE = CFG.plan_path
        STATE_FILE = CFG.state_path
        REPO = Path(CFG.repo_dir)

    if args.print_config or args.dry_run:
        print("[eng] === DRY RUN (Zero side effects, zero state writes) ===")
        print(f"[eng] Config file: {orch_config._config_file()}")
        print(f"[eng] Repo dir:    {CFG.repo_dir}")
        print(f"[eng] Base dir:    {CFG.base_dir}")
        print(f"[eng] Plan file:   {PLAN_FILE}")
        print(f"[eng] State file:  {STATE_FILE}")
        print(f"[eng] Queue file:  {CFG.queue_path}")
        print(f"[eng] Wake log:    {CFG.wake_log}")
        if not PLAN_FILE.exists():
            print(f"[eng] WARNING: Plan file does not exist: {PLAN_FILE}")
            return
        plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
        steps = plan.get("steps") or []
        st = load_state()
        print(f"[eng] Total plan steps: {len(steps)} | Recorded state steps: {len(st.get('steps', {}))}")
        todo = steps
        if args.only_step:
            todo = [s for s in todo if s.get("finding_id") == args.only_step]
        if args.limit is not None:
            todo = todo[: args.limit]
        batches = pack_steps(todo)
        print(f"[eng] Target steps: {len(todo)} | Disjoint batches: {len(batches)}")
        print(CFG.render())
        print("[eng] === END DRY RUN ===")
        return

    if not PLAN_FILE.exists():
        print(f"[eng] Error: plan file does not exist: {PLAN_FILE}", file=sys.stderr)
        return 1

    plan = json.loads(PLAN_FILE.read_text(encoding="utf-8"))
    steps = plan.get("steps") or []
    print(f"[eng] plan {PLAN_FILE.name}: {len(steps)} steps | workers={WORKERS}")
    st = load_state()
    todo = steps
    if args.only_step:
        todo = [s for s in todo if s.get("finding_id") == args.only_step]
    if args.limit is not None:
        todo = todo[: args.limit]
    batches = pack_steps(todo)
    print(f"[eng] packed {len(batches)} batches (disjoint-file groups)")
    sel = batches
    if args.batch is not None:
        sel = [batches[args.batch - 1]]
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=int(CFG.lane_timeout) + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        gpb = asyncio.Semaphore(PARALLEL)
        skip_index = args.batch if args.batch is not None else 0

        async def run_one(idx_batch: tuple) -> None:
            idx, batch = idx_batch
            async with gpb:
                bnum = idx if skip_index == 0 else skip_index
                pending = [s for s in batch
                           if (st["steps"].get(s["finding_id"]) or {}).get("status")
                           in (None, "pending", "executing")]
                print(f"[BATCH {bnum}] {len(batch)} steps, {len(pending)} pending", flush=True)
                if not pending:
                    return
                await run_group(session, pending, st)
                statuses = {s["finding_id"]: st["steps"].get(s["finding_id"], {}).get("status")
                            for s in batch}
                print(f"[BATCH {bnum}] done: {statuses}", flush=True)

        await asyncio.gather(*(run_one(ib) for ib in enumerate(sel, 1)))
    await save_state_serialized(st)
    aggr = dict((k, v.get("status")) for k, v in st["steps"].items())
    print(f"EXEC_DONE: total={len(aggr)} green={sum(1 for x in aggr.values() if x == 'green')} "
          f"escalated={sum(1 for x in aggr.values() if x == 'escalated')}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
