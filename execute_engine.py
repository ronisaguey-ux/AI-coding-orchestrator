#!/usr/bin/env python3
"""execute_8_27_engine.py — master-plan executor for the 8_27 cycle (09-03).

Executes the per-finding master plan JSON (cross_eval_plan_9_3*.json,
17,345 per-finding steps) on the FREE webchat lanes ONLY (deepseek 8080 /
gemini 8085 round-robin, model anymodel, no paid API — user 09-02 policy).
The pattern is the proven 8_26 OXA engine: per step EXECUTE call -> apply
edits (exact string replace + py syntax guard) -> VERIFY call over test
output (<=3 rounds) -> green/escalated. Steps are density-packed into batches
of WORKERS with DISJOINT file targets (never co-editing a file in one batch).

State: audits_plans/exec_state_8_27.json (per step_id: pending -> applied ->
green|escalated). Usage: python3 execute_8_27_engine.py [--resume] [--batch N]
[--only-step ID] [--limit N]
"""
import argparse, asyncio, json, os, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orch_config, orch_lanes, orch_verify

# 09-05: the whole engine now resolves through orch_config (defaults < file <
# env < CLI), so the gateway envs and the engine options read ONE source.
CFG = orch_config.load()

BASE = Path(CFG.base_dir)
# The former default (cross_eval_plan_9_3.json, 263 steps) matched only
# 7 of the 52 escalated ids in exec_state_9_4; the 14,356-step 9_4_fixed plan
# covers 199/199. A wrong default here silently starved every step lookup.
PLAN_FILE = CFG.plan_path
STATE_FILE = CFG.state_path
REPO = Path(CFG.repo_dir)

WORKERS = int(os.environ.get("EXEC_WORKERS", "8"))
MAX_ROUNDS = 3
# OpenRouter free pool (user key 09-05) — API-speed FREE lane; model "openrouter/free"
# auto-routes to whatever :free model is available; (url, model, cool_base, cool_esc, auth)
OPENROUTER_KEY = ""
try:
    OPENROUTER_KEY = Path("/home/roni/.claude/openrouter.token").read_text().strip()
except Exception:
    pass

LANES = [
    ("http://127.0.0.1:8080/v1/chat/completions", "anymodel", 90, 270),  # deepseek 8080 (base, escalated)
    ("http://127.0.0.1:8085/v1/chat/completions", "gemini 3.7 flash webchat", 300, 900),  # gemini 8085 — long settle window (user 09-04)
    ("http://127.0.0.1:20128/v1/chat/completions",
     ["auto/best-coding", "auto/coding", "auto/best-chat", "auto/best-reasoning"],
     120, 360),  # omniroute — multi-alias, per-model fallback (user 09-05; probes: coding/best-chat 200, best-coding 429)
    ("https://openrouter.ai/api/v1/chat/completions", "openrouter/free", 45, 120, OPENROUTER_KEY),  # openrouter free pool (user 09-05)
]
_lane_idx = 0
# dead-lane cooldown: consecutive failures (5xx/conn) mark lane dead for N sec
_lane_dead_until = {}
_lane_fail_streak = {}
_model_dead_until = {}   # (lane, model) -> ts — per-model cooldown (user 09-05: model, not lane)
_model_fail_streak = {}

GROUP_CAP = int(CFG.group_cap)  # orch.yaml group_cap (user: >=15 per batch)
PARALLEL = min(3, max(1, int(os.environ.get("EXEC_PARALLEL", "3"))))  # 09-05 (user): parallel BATCH groups, memory-safe cap (webchat lanes each drive ONE tab anyway)
SAVE_LOCK = asyncio.Semaphore(1)  # serialize state dumps (dict-change-during-iteration guard for concurrent groups)
EXEC_SYSTEM = (
    "You are a code FIX EXECUTOR for the target repo (Python/FastAPI/React). The plan below "
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
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"steps": {}}


def save_state(st: dict) -> None:
    """Atomic write: temp file + fsync + replace, with a rolling backup.

    A plain write_text truncated the state file if the process died mid-dump,
    losing every step result recorded so far.
    """
    if getattr(CFG, "dry_run", False):
        return
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


async def save_state_serialized(st: dict) -> None:
    """09-05: save_state under the SAVE_LOCK — parallel groups share one state
    dict; a plain dump during another task's mutation can raise
    'dictionary changed size during iteration' and lose the last writer."""
    async with SAVE_LOCK:
        save_state(st)


def signal_escalation(sid: str, rec: dict) -> None:
    """09-05 (user): an escalated step must wake the MAIN operator, who then
    FIXES IT personally. Append to the wake chain (same file the session
    watches) so main notices without polling.
    """
    if getattr(CFG, "dry_run", False):
        return
    try:
        la = rec.get("last_apply") or {}
        with open("/tmp/main_wake.log", "a", encoding="utf-8") as f:
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
        t = p.read_text(errors="ignore")
        if len(t) > size_cap:
            t = t[:size_cap] + "\n...[TRUNCATED]"
        return t
    except Exception:
        return ""


LANE_COOLDOWN = 90  # sec a failed lane stays out of rotation

# 09-05: the pool replaces the hand-rolled rotation below. It parses SSE bodies
# (OmniRoute was silently dark), bounds each request (a wedged Gemini call used
# to stall a worker for the full 900s client timeout), hops to another lane on
# failure, and — critically — reports transport failure as ok=False instead of
# collapsing it to "" where the caller read it as "the model produced no edits".
POOL = orch_lanes.LanePool(CFG)


async def lane_call_result(session, system: str, user: str):
    """Return a LaneResult. Check .ok before trusting .content."""
    return await POOL.call(session, system, user)


async def lane_call(session, system: str, user: str, retries: int = 4) -> str:
    """Backwards-compatible string form. Prefer lane_call_result."""
    res = await POOL.call(session, system, user)
    return res.content if res.ok else ""


async def _legacy_lane_call(session, system: str, user: str, retries: int = 4) -> str:
    import aiohttp
    global _lane_idx
    # pick next lane that is not in cooldown; if all cooldown, force the round-robin one
    n = len(LANES)
    picks = []
    for i in range(n):
        picks.append((_lane_idx + i) % n)
    chosen = None
    for i in picks:
        if time.time() >= _lane_dead_until.get(i, 0):
            chosen = i
            break
    if chosen is None:
        chosen = picks[0]
    _lane_idx = chosen + 1
    url, model, cool_base, cool_esc, auth = (LANES[chosen] + (None,))[:5]
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    # per-model fallback (user 09-05): a rate-limited MODEL is cooled + skipped,
    # the lane keeps serving its next model. Lane dead only when ALL its models cooled.
    for m in models:
        if time.time() < _model_dead_until.get((chosen, m), 0):
            continue
        payload = {"model": m,
                   "messages": [{"role": "system", "content": system[:60000]},
                                {"role": "user", "content": user[:80000]}],
                   "max_tokens": 16000, "temperature": 0.2}
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = f"Bearer {auth}"
        for attempt in range(1, retries + 1):
            try:
                async with session.post(url, json=payload, headers=headers) as r:
                    body = await r.text()
                    if r.status == 200:
                        _lane_fail_streak[chosen] = 0
                        _model_fail_streak[(chosen, m)] = 0
                        try:
                            c = json.loads(body)["choices"][0]["message"]["content"]
                            return c if isinstance(c, str) else ""
                        except Exception:
                            return ""
                    # hard-fail: cooldown THIS model, fall through to the next one
                    if r.status in (404, 422, 500, 502, 503):
                        _model_fail_streak[(chosen, m)] = _model_fail_streak.get((chosen, m), 0) + 1
                        if _model_fail_streak[(chosen, m)] >= 2:
                            cool = cool_esc if _model_fail_streak[(chosen, m)] >= 3 else cool_base
                            _model_dead_until[(chosen, m)] = time.time() + cool
                            print(f"[lanes] lane {chosen} model {m} marked dead "
                                  f"{cool}s (status {r.status})", flush=True)
                        break
                    if r.status == 429:
                        _model_fail_streak[(chosen, m)] = _model_fail_streak.get((chosen, m), 0) + 1
                        _model_dead_until[(chosen, m)] = time.time() + cool_base
                        print(f"[lanes] lane {chosen} model {m} rate-limited "
                              f"{cool_base}s — skipping (in-lane fallback)", flush=True)
                        break
                    await asyncio.sleep(4)
            except Exception:
                _model_fail_streak[(chosen, m)] = _model_fail_streak.get((chosen, m), 0) + 1
                if _model_fail_streak[(chosen, m)] >= 2:
                    _model_dead_until[(chosen, m)] = time.time() + cool_base
                    print(f"[lanes] lane {chosen} model {m} conn error "
                          f"-> {cool_base}s (in-lane fallback)", flush=True)
                await asyncio.sleep(6 * attempt)
    # every model of this lane is cooled — brief lane pause keeps rotation fair
    if models and all(time.time() < _model_dead_until.get((chosen, mm), 0) for mm in models):
        _lane_dead_until[chosen] = time.time() + cool_base
        print(f"[lanes] lane {chosen} {model} all models cooled {cool_base}s", flush=True)
    return ""


def parse_json(text) -> dict:
    """Balanced-object scan. The old greedy r'\\{.*\\}' spanned from the first
    brace to the last, so any prose after the object, a second object, or a
    brace inside an old_string value broke the parse and looked like a refusal.
    """
    return orch_lanes.parse_json_object(text)


def apply_edits(edits: list) -> tuple:
    """Exact string replace with py syntax guard (proven 8_26 logic).

    09-05: an EMPTY edit list is now an explicit failure. ``all([])`` is
    vacuously True, so "the lane returned nothing" was recorded as a successful
    apply — 43 of 52 escalated steps carried ok=true over zero edits.
    """
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
            path.write_text(new, errors="ignore")
            results.append({"file": f, "ok": True, "msg": "created"})
            continue
        if old == "" and new:
            results.append({"file": f, "ok": False,
                            "msg": "empty old_string but file exists (context changed)"})
            continue
        try:
            cur = path.read_text(errors="ignore")
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
            path.write_text(cur.replace(old, new, 1), errors="ignore")
            results.append({"file": f, "ok": True, "msg": "applied"})
        else:
            results.append({"file": f, "ok": False,
                            "msg": "old_string not found (context changed)"})
    return all(r["ok"] for r in results), "; ".join(
        f"{r['file']}:{r['msg']}" for r in results[:4])


def run_checks(files: list) -> str:
    """Local verification per step: syntax check each touched file + git state.

    09-05: the previous implementation ran ``py_compile`` and then DISCARDED the
    return code, appending "py_compile <f>: ok" unconditionally — a file with a
    syntax error reported success, so the verifier model was judging fixes
    against output that could never say "broken". Checking is now ast.parse
    in-process (no bytecode written, no module code executed, no subprocess).
    """
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
        # A transport failure is NOT a failed fix. Leaving the step pending
        # keeps it eligible for the next pass instead of burning an escalation
        # on a dead gateway — the defect behind 37 of 52 escalations on 09-05.
        if not res.ok:
            rec["status"] = "pending"
            rec["last_lane_error"] = res.error[:300]
            await save_state_serialized(st)
            print(f"[step {sid}] lanes unavailable ({res.error[:90]}) — "
                  f"left pending", flush=True)
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
    """09-05 (user): up to GROUP_CAP INDEPENDENT steps in ONE exec lane call.

    The group arrives from pack_steps with disjoint file targets. The EXEC_SYSTEM
    contract now allows up to 5 steps per reply: build one user prompt with STEP 1..N
    blocks, take ONE lane_call, split the tagged edits per step, then verify every
    step (per-step verify keeps the red/corrected-attempt semantics of run_step).
    """
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
        # Whole-group transport failure: return every step to pending rather
        # than escalating a batch of untouched steps.
        for _idx, sid, _files in step_ids:
            rec = st["steps"][sid]
            rec["status"] = "pending"
            rec["rounds"] = max(0, rec.get("rounds", 1) - 1)
            rec["last_lane_error"] = gres.error[:300]
        await save_state_serialized(st)
        print(f"[group] lanes unavailable ({gres.error[:90]}) — "
              f"{len(step_ids)} steps left pending", flush=True)
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
            # Only escalate once the step has actually spent its rounds. The
            # group path used to escalate after a SINGLE attempt while
            # run_step allowed MAX_ROUNDS, so a step's retry budget depended
            # on whether packing happened to place it alone in a batch.
            if rec["rounds"] >= MAX_ROUNDS:
                rec["status"] = "escalated"
                signal_escalation(sid, rec)
            else:
                rec["status"] = "pending"
    await save_state_serialized(st)
    return {s["finding_id"]: st["steps"].get(s["finding_id"], {}).get("status") for s in steps}


def pack_steps(steps: list) -> list:
    """Density pack up to GROUP_CAP regardless of file overlap.
    09-05 (user): "make each batch have at least 15 steps" — the disjoint-file
    rule capped most batches at 1-2 and starved throughput. Same-file steps in
    ONE call are contract-legal (each edit anchors on its own unique old_string)
    and per-step VERIFY catches any mis-application; packing is by count only."""
    packed, group = [], []
    for s in steps:
        if group and len(group) >= GROUP_CAP:
            packed.append(group)
            group = []
        group.append(s)
    if group:
        packed.append(group)
    return packed


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch", type=int, default=None, help="1-based pack batch")
    ap.add_argument("--only-step", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="Resolve config and plan without side effects")
    ap.add_argument("--print-config", action="store_true", help="Print resolved config and exit")
    args = ap.parse_args()

    global CFG, BASE, PLAN_FILE, STATE_FILE, REPO
    if args.dry_run:
        CFG = orch_config.load(overrides={"dry_run": True})
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

    plan = json.loads(PLAN_FILE.read_text())
    steps = plan.get("steps") or []
    print(f"[eng] plan {PLAN_FILE.name}: {len(steps)} steps | lanes={len(POOL.lanes)} | workers={WORKERS}")
    st = load_state()
    todo = steps
    if args.only_step:
        todo = [s for s in todo if s["finding_id"] == args.only_step]
    if args.limit is not None:
        todo = todo[: args.limit]
    batches = pack_steps(todo)
    print(f"[eng] packed {len(batches)} batches (disjoint-file groups)")
    sel = batches
    if args.batch is not None:
        sel = [batches[args.batch - 1]]
    tok = None
    import aiohttp
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
        num = args.batch if args.batch is not None else 0
        # 09-05 (user): PARALLELITY — up to EXEC_PARALLEL batch groups in
        # flight at once. Lanes round-robin per lane_call, so groups spread
        # across DS / gemini / openrouter naturally; state saves are
        # SAVE_LOCK-serialized (see save_state_serialized). Memory-safe cap:
        # each webchat lane drives ONE tab anyway, so extra groups only add
        # small HTTP payloads, not browsers.
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
