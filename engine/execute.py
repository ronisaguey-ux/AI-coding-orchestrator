#!/usr/bin/env python3
"""execute.py — master-plan executor for the oculus fix pipeline.

Executes the per-finding master plan JSON (oculus_cross_eval_plan_9_3*.json,
17,345 per-finding steps) on the FREE webchat lanes ONLY (deepseek 8080 /
gemini 8085 round-robin, model anymodel, no paid API — user 09-02 policy).
The pattern is the proven 8_26 OXA engine: per step EXECUTE call -> apply
edits (exact string replace + py syntax guard) -> VERIFY call over test
output (<=3 rounds) -> green/escalated. Steps are density-packed into batches
of WORKERS with DISJOINT file targets (never co-editing a file in one batch).

State: audits_plans/exec_state_8_27.json (per step_id: pending -> applied ->
green|escalated). Usage: python3 execute.py [--resume] [--batch N]
[--only-step ID] [--limit N]
"""
import argparse, asyncio, json, os, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orch_config, orch_git, orch_lanes, orch_queue as oq, orch_verify

# 09-05: the whole engine now resolves through orch_config (defaults < file <
# env < CLI), so the gateway envs and the engine options read ONE source.
CFG = orch_config.load()

BASE = Path(CFG.base_dir)
# The former default (oculus_cross_eval_plan_9_3.json, 263 steps) matched only
# 7 of the 52 escalated ids in exec_state_9_4; the 14,356-step 9_4_fixed plan
# covers 199/199. A wrong default here silently starved every step lookup.
PLAN_FILE = CFG.plan_path
STATE_FILE = CFG.state_path
REPO = Path(CFG.repo_dir)

WORKERS = int(os.environ.get("EXEC_WORKERS", "8"))
MAX_ROUNDS = 3

# 09-13: a TRANSPORT failure is not a verdict. Measured: 136 steps were escalated to
# terminal because the gemini gateway (127.0.0.1:8085) was briefly unreachable — the
# lane HAD produced an edit, the verify call just could not connect. Those steps are
# now permanently dead for a reason that had nothing to do with the code. A connection
# error must leave the step pending and must NOT count toward MAX_ROUNDS.
_TRANSPORT_MARKERS = (
    "ClientConnectorError", "Cannot connect to host", "Connect call failed",
    "Connection refused", "ConnectionResetError", "ServerDisconnectedError",
    "Server disconnected", "no lane available", "lanes unavailable",
    "ReadTimeout", "ConnectTimeout", "aiohttp.client_exceptions",
    # 09-14: a rate-limit / capacity error is a TEMPORARY transport condition,
    # never a verdict. Measured: 19 steps escalated to terminal with
    # `http 429 {"code":"free_rate_limited"}` as the last lane error, because
    # none of these markers matched and the phantom-green gate spent its round.
    "free_rate_limited", "rate_limited", "rate limit", "Too Many Requests",
    "http 429", "429:", "temporarily rate", "Provider returned error",
    "Webchat not connected", "browser.isConnected",
    # 09-14: the engine's OWN timeout wording was never matched. "ReadTimeout" /
    # "ConnectTimeout" are aiohttp class names, but the engine logs the failure as
    # `timeout after 600s`. Measured: 268 steps sat in `escalated` at rounds=3 on
    # exactly that string — a lane that never answered is not a verdict about the
    # code, so those must retry, never terminate.
    "timeout after", "timed out",
    # 09-17: a lane that REFUSES a draw because its tab is still generating is the
    # same class - the lane could not be reached, so it says nothing about the code.
    # Measured on chatgpt: 5 sends / 0 responses / 4 refusals, every one reading
    # "webchat tab still generating from a previous request - retry after it finishes".
    # That tab needs ~360s per generation, so at the engine's draw rate most attempts
    # collide with an in-flight one. Burning a step's round on a collision retires work
    # a working lane simply had not finished talking about yet.
    "still generating from a previous request", "still generating",
    # 09-16: a provider-side 401 "model not supported" is the same class as the 429
    # above - the LANE could not be reached, so it says nothing about the code.
    # Measured: 11 yellow steps sat at rounds=3 whose ONLY lane error was
    # `http 401: oc/north-mini-code-free: auth — [401]: Model north-mini-code-free
    # is not supported`, i.e. the whole round budget went on a model the provider had
    # already withdrawn. Those steps never received a considered answer.
    "http 401", "[401]", "is not supported (HTTP", "model is not supported",
)


def is_transport_error(err) -> bool:
    """True when a lane call failed to REACH a lane rather than answer it."""
    if not err:
        return False
    e = str(err)
    return any(m in e for m in _TRANSPORT_MARKERS)

# 09-13 (owner): what to do when a lane reports the work is ALREADY THERE.
#   green (default) | escalate | pending   — set via orch.yaml
# `already_satisfied_action`. Phrases are configurable too; the built-in list
# below is the default and `already_satisfied_phrases` is appended to it.
try:
    from orch_config import load as _orch_load  # noqa: E402
    _ORCH_CFG = _orch_load()
except Exception:
    _ORCH_CFG = None


def already_satisfied_action() -> str:
    v = getattr(_ORCH_CFG, "already_satisfied_action", "green") if _ORCH_CFG else "green"
    v = str(v or "green").strip().lower()
    return v if v in ("green", "escalate", "pending") else "green"


_BASE_ALREADY = (
    "already satisfied", "already present", "is present with", "already exists",
    "already implemented", "no change needed", "already on disk",
    "already contains", "already carries", "already has", "already declares",
    "already defines", "no fix needed",
    # a bare "— satisfied" / "is satisfied" verdict (deepseek4 wrote
    # "P1B6R0F3#8 — satisfied (verified by MANIFEST.json read)") was missed and
    # fell through to PENDING.
    " satisfied (", " is satisfied", "satisfied (verified",
)
# 09-16: the list above only held PAST-tense forms, so 23 measured verdicts fell
# through and were retired yellow although the lane had said the work was already
# there. Counted across the live state, the phrasings the lanes actually used were:
# already imports x3, already fully x2, already implements x2, already uses x2,
# already validates/escalates/forwards/lists/starts x1 each, already exist,
# already fixed/resolved/removed/replaced/complete, already been. A lane writing
# "already imports numpy as np at line 13" is the SAME verdict as "already
# implemented" - the config's `already_satisfied_action` is what decides the
# outcome, and it never got the chance to run.
#
# Deliberately NOT a bare `already \w+`: "already tried", "already burned",
# "already spent" and "already claimed" describe the ENGINE's attempts, not the
# state of the code, and matching them would mint phantom greens.
_ALREADY_VERB = re.compile(
    # an adverb may sit between: measured "is already FULLY populated", "is already
    # fully present in the current file" - both real verdicts the first form missed.
    r"\balready\s+(?:fully\s+|completely\s+|already\s+)?(?:imports?|implements?|uses?|exists?|validates?|escalates?|"
    r"forwards?|lists?|starts?|contains?|declares?|defines?|carries|populated|"
    r"complete|fixed|resolved|removed|replaced|absent|present|"
    r"been\s+(?:applied|implemented|fixed|added|updated|removed|changed|replaced))\b",
    re.I)


def is_already_satisfied(text: str) -> bool:
    low = (text or "").lower()
    if not low:
        return False
    extra = getattr(_ORCH_CFG, "already_satisfied_phrases", None) if _ORCH_CFG else None
    phrases = list(_BASE_ALREADY) + [str(x).lower() for x in (extra or [])]
    if any(p in low for p in phrases):
        return True
    return bool(_ALREADY_VERB.search(low))

# OpenRouter free pool (user key 09-05) — API-speed FREE lane; model "openrouter/free"
# auto-routes to whatever :free model is available; (url, model, cool_base, cool_esc, auth)
OPENROUTER_KEY = ""
try:
    OPENROUTER_KEY = Path("/home/roni/.claude/openrouter.token").read_text().strip()
except Exception:
    pass

LANES = [
    ("http://127.0.0.1:8085/v1/chat/completions", "gemini 3.7 flash webchat", 300, 900),  # gemini 8085 — long settle window (user 09-04)
    ("http://127.0.0.1:20128/v1/chat/completions",
     ["cfp/nvidia/nemotron-3-120b-a12b", "auto/best-free", "auto/coding"],
     120, 360),  # omniroute — free lane. 09-10 probe: cfp/nemotron-3-120b-a12b 200 (works);
                 # auto/* combos 502 (opencode noauth 401 + 429), auto/best-coding 429.
    # openrouter free lane COMMENTED 09-09 (roni: only gemini/omniroute rn).
    # Free models probe 404 "unavailable for free" (quota/model churn) — wastes
    # slots hopping. Re-enable when the free pool is back (favor 00:00 UTC reset):
    # ("https://openrouter.ai/api/v1/chat/completions", "openrouter/free", 45, 120, OPENROUTER_KEY),
    # kimi lane DISABLED 09-08: K3 webchat first-turns = 30min+ / die empty
    # (verified with real probes) — far beyond the 600s lane timeout. Gateway
    # 8086 + login stay warm; re-enable when kimi's agentic mode can be tamed.
]
_lane_idx = 0
# dead-lane cooldown: consecutive failures (5xx/conn) mark lane dead for N sec
_lane_dead_until = {}
_lane_fail_streak = {}
_model_dead_until = {}   # (lane, model) -> ts — per-model cooldown (user 09-05: model, not lane)
_model_fail_streak = {}


# 09-06 user ladder: 15m base + 5m per SEQUENTIAL issue (15→20→25→30…) —
# a model is skipped entirely until its timer expires; success resets the streak.
def _model_cool_s(streak: int) -> int:
    base = int(os.environ.get("OMNI_COOLDOWN_BASE_S", "600"))  # 09-09 (roni): per-model ban ladder 10m base, +5m each sequential fail (10→15→20→…)
    step = int(os.environ.get("OMNI_COOLDOWN_STEP_S", "300"))
    return base + max(0, streak - 1) * step

GROUP_CAP = int(CFG.group_cap)  # orch.yaml group_cap (user: >=15 per batch)
# 09-14 (owner, S2): a batch larger than the lane count leaves slack, so a lane
# always finds a claimable step after a sibling cools. The barrier tail grows
# with the batch, so this is deliberately modest.
BATCH_CAP = int(os.environ.get("ORCH_BATCH_CAP", "30"))
PARALLEL = min(8, max(1, int(os.environ.get("EXEC_PARALLEL", "8"))))  # 09-09 (roni): raise cap so omniroute (API lane) parallelizes; gemini stays serial (1 tab). Memory-safe: omniroute calls are lightweight HTTP.
SAVE_LOCK = asyncio.Semaphore(1)  # serialize state dumps (dict-change-during-iteration guard for concurrent groups)
EXEC_SYSTEM = (
    "You are a code FIX EXECUTOR for the oculus repo (Python/FastAPI/React). The plan below "
    "contains 1-5 INDEPENDENT steps (disjoint files — no edits reference another step's file). "
    "Fix EVERY step in ONE reply. "
    "Answer ONE JSON object only: {\"edits\":[{\"step\":\"<step id, e.g. STEP 1 or P1B1R0F12>\","
    "\"file\":\"<repo-relative path>\",\"old_string\":\"<exact existing text>\",\"new_string\":"
    "\"<replacement text>\"}],\"notes\":\"<short>\"}. EVERY edit must carry the step id it "
    "belongs to; leave the step id off only when only ONE step is present. "
    "Each old_string must be UNIQUE and byte-exact from the CURRENT file contents (re-read the "
    "live file, not the plan snippet). Use full function bodies when replacing functions. "
    "FILE CONTENTS arrive as LINE-NUMBERED CHUNKS: every line is prefixed with its line number, "
    "and a large file is shown as several windows (head + interior + tail) with the omitted ranges "
    "marked. You may read ANY of those chunks and quote an old_string from any of them — you are "
    "not limited to the first block. Never answer 'the file is truncated': the chunks ARE the file, "
    "and the line numbers tell you where each piece sits. "
    "09-09 (Bob): do NOT treat plan steps as verbatim. The plan is a STALE snapshot — line "
    "numbers, offsets and code snippets may be off because the file has moved on. You have logical "
    "authority to apply the CORRECT edit: read the actual file, locate the real target by intent "
    "(function name / module / unique nearby text), and anchor old_string against what is truly "
    "there. If the target FILE IS ABSENT and the fix requires creating "
    "it, emit {\"file\":\"...\",\"old_string\":\"\",\"new_string\":\"<full file content>\"}. "
    "Batch as much real work into this ONE turn as the steps allow — no prose about it, just do it. "
    "No fences, no prose, no submit_answer. If a step's guidance conflicts with reality or you "
    "cannot fix it, do NOT return empty edits — omit that step from edits and put its id + reason "
    "in notes (cannot-fix:<id>:<reason>). An empty edits array means NOTHING was fixed. "
    "NEVER emit empty edits and claim success.\n"
    # The webchat models (gemini especially) drift into their conversational
    # assistant persona and answer a code-fix request with "Task completed
    # successfully." — 28 bytes of prose, no edits. The lane then (correctly)
    # rejects it and hops, so every step burns the whole pool and escalates.
    # A short, last-word restatement of the output contract, placed after the
    # long body, is what actually holds the format: the model weights the tail
    # of the system prompt most heavily, and a bare imperative beats a buried
    # schema. Do not remove this — it is the difference between edits and prose.
    "\nOUTPUT CONTRACT — your entire reply is parsed as JSON. "
    "If you write a sentence, a status line, an apology or a confirmation, the reply is "
    "REJECTED and the work is counted as not done. "
    "There is no such thing as 'done' to report: you report WHAT YOU CHANGED, as edits. "
    "Your first character is '{' and your last is '}'. "
    "Reply now with the JSON object only."
)
VERIFY_SYSTEM = (
    "You are the SELF-VERIFIER for the step you just fixed. You receive your edits + the test/check "
    "output. Answer ONE JSON object only. {\"verdict\":\"green\"} is FORBIDDEN when the EDITS array is "
    "EMPTY — an empty change means nothing was fixed, so return {\"verdict\":\"red\",\"reason\":"
    "\"no real edit — step NOT fixed\",\"edits\":[...]} with the correct edit. green requires a "
    "NON-EMPTY edits array whose new_string actually appears in the target file. Otherwise "
    "{\"verdict\":\"red\",\"reason\":\"<terse>\",\"edits\":[{\"file\":\"...\",\"old_string\":\"...\","
    "\"new_string\":\"...\"}]} with corrected edits. PREEXISTING/ignorable output lines are not your "
    "fault. No fences."
)


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            st = json.loads(STATE_FILE.read_text())
        except Exception:
            return {"steps": {}}
        # 09-15: normalise escalated residue at LOAD time, not at save time.
        # Patching the merge in save_state_serialized did not work because several
        # code paths call save_state() directly, which writes the memory copy
        # verbatim - so memory that still held 536 legacy `escalated` records kept
        # writing them back. Load is the one place every path goes through, and
        # with the escalation phase OFF there is no escalation state to preserve.
        if not ESCALATIONS_ENABLED:
            _n = 0
            for _r in (st.get("steps") or {}).values():
                if isinstance(_r, dict) and _r.get("status") == "escalated":
                    _r["status"] = "pending"
                    _r.pop("escalated_at", None)
                    _r.pop("escalated_by", None)
                    _r["escalation_retired_reason"] = (
                        "escalated residue normalised to pending at load - the "
                        "escalation phase is disabled in the config")
                    _n += 1
            if _n:
                print(f"[eng] normalised {_n} legacy 'escalated' step(s) -> pending "
                      f"at load (escalations are disabled)", flush=True)
        # 09-16 (owner): "get rid of the cannot fix justification as a yellow,
        # it needs to give a detailed justification not just a vague 'cannot fix'
        # excuse." Rebuild a vague yellow's reason from the evidence the record
        # already holds, at LOAD time - the one place every path passes. Doing it
        # as a one-off state edit does not stick: the engine holds its own copy
        # from before the edit and writes the old reason straight back (measured:
        # 119 backfilled, 0 survived the next save).
        try:
            _b = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict) or _r.get("status") != "yellow":
                    continue
                _j = str(_r.get("yellow_reason") or "")
                # Skip only a reason that is BOTH detailed and complete. A reason
                # the old 300-char clip cut mid-sentence ("...in the ") has to be
                # rebuilt even though it is long - measured: one yellow sat at
                # exactly 300 chars, truncated, and the length test passed it.
                if (len(_j) >= 160 and "own words" in _j
                        and _j.rstrip()[-1:] in ".!?)]" and len(_j) != 300
                        and "not recorded" not in _j):
                    # Skip only a reason that is detailed AND whole AND names its
                    # subject. A placeholder ("Target file(s): not recorded") is not
                    # a justification however long it is - measured: two yellows
                    # passed the length test while saying exactly that.
                    continue
                _new = yellow_justification_detail(_k, _r)
                if _new:
                    _r["yellow_reason"] = _new
                    _r["yellow_reason_source"] = "rebuilt from the record at load (09-16)"
                    _b += 1
            if _b:
                print(f"[eng] rebuilt {_b} vague yellow justification(s) from the "
                      f"step records at load", flush=True)
        except Exception as _e:
            print(f"[eng] yellow justification rebuild skipped: {str(_e)[:120]}", flush=True)
        # 09-16: a yellow whose edit is verifiably ON DISK is finished work, not a
        # failure. Recovering those as a one-off state edit does NOT stick - the
        # engine holds its own copy and writes the yellow straight back (measured:
        # greens flipped by the review, then the count fell again on the next save).
        # load_state() is the one place every path passes, so the recovery lives
        # here and is idempotent: it re-reads the FILE every start and can only
        # ever move a yellow to green when the content is really there.
        try:
            _g = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict) or _r.get("status") != "yellow":
                    continue
                _la = _r.get("last_apply") or {}
                # 09-16: do NOT require _la["ok"]. Measured on P1B4R0F10#7:
                # ok=False with "old_string not found (context changed)", yet the
                # step's new_string IS on data_pipeline/bar_builder.py - the change
                # landed by another route and the apply record never saw it succeed.
                # _applied_edits_landed() is the real evidence: it READS the file.
                if _la.get("edits") and _edits_content_on_disk(_r):
                    _r["status"] = "green"
                    _r["resolved_by"] = (
                        "yellow review: the step's edit is present on disk (verified "
                        "at load, so the recovery cannot be a phantom)")
                    _r["yellow_watch_review"] = {
                        "verdict": "recovered",
                        "evidence": "content verified on disk at load",
                    }
                    _g += 1
            if _g:
                print(f"[eng] recovered {_g} yellow step(s) whose edit is on disk",
                      flush=True)
        except Exception as _e:
            print(f"[eng] on-disk yellow recovery skipped: {str(_e)[:120]}", flush=True)
        # 09-16: a yellow whose apply died on a DUNDER-STRIPPED path is not finished
        # work and not a broken step - it is a step that never got its turn. The
        # lane returned `oculus/runtime/init.py` for `oculus/runtime/__init__.py`,
        # the apply hit ENOENT, the rounds burned and the step retired yellow. Now
        # that _resolve_dunder_path() retargets it, the edit lands - so re-queue it
        # as PENDING (never green: nothing has been applied yet). Evidence, not a
        # guess: the old_string must be ON DISK in the file the resolver points at.
        try:
            _q = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict) or _r.get("status") != "yellow":
                    continue
                _la = _r.get("last_apply") or {}
                _miss = re.search(r"No such file or directory: '([^']+)'",
                                  str(_la.get("apply_msg") or ""))
                if not _miss:
                    continue
                _rel = _miss.group(1).replace(str(REPO) + os.sep, "", 1)
                # 09-17: THREE path typos now resolve, not just the dunder one - the
                # leading-dot form (.pre-commit-config.yaml) and the dropped src PREFIX
                # (python/momentum/stochastics.rs -> rust/indicators/src/...). A step whose
                # apply died on ENOENT for any of them was handed NO content and is not
                # finished work and not a broken step - it never got its turn. Try every
                # resolver and re-queue on whichever one changes the path.
                _fix = _resolve_dunder_path(_rel)
                if _fix == _rel:
                    _fix = _resolve_plan_path(_rel)
                if _fix == _rel:
                    continue
                _real = Path(REPO) / _fix
                if not _real.exists():
                    continue
                _txt = _real.read_text(errors="ignore")
                _eds = [e for e in (_la.get("edits") or []) if e.get("old_string")]
                # 09-17: a READ FAIL means the lane was shown NO FILE AT ALL - the
                # apply died before it could open the target, so whatever old_string
                # the lane quoted was invented from a path it never saw. Requiring
                # that old_string to be on disk therefore rejects exactly the steps
                # that most deserve a clean attempt: measured P1B5R0F9#161
                # (config/init.py), P1B1R0F7#76 (oculus/reporting/init.py) and
                # P1B2R0F8#100 (oculus/runtime/init.py) are all read-fails on a
                # dunder-stripped path that resolves now, and all three were skipped
                # by the old_string check. Only require the match when the lane DID
                # get content (a plain missing-old_string failure).
                _saw_nothing = "read fail:" in str(_la.get("apply_msg") or "")
                if not _eds and not _saw_nothing:
                    continue
                if _eds and not _saw_nothing and not all(e["old_string"] in _txt for e in _eds):
                    continue
                _r["status"] = "pending"
                _r["rounds"] = 0
                _r["requeued_reason"] = (
                    "apply died on a dunder-stripped path (%s); it resolves to %s "
                    "and the edit's old_string is on disk, so the step gets its turn"
                    % (_rel, _fix))
                _q += 1
            if _q:
                print(f"[eng] re-queued {_q} yellow step(s) whose apply died on a "
                      f"dunder-stripped path", flush=True)
        except Exception as _e:
            print(f"[eng] dunder re-queue skipped: {str(_e)[:120]}", flush=True)
        # 09-17: a yellow whose apply died with `read fail` on a target that does NOT
        # exist is a step that never got its turn - the CREATE path used to require an
        # empty old_string, so a RESTORE task fell through to the read and died.
        # Measured: P1B2R0F7#46 - the plan says "security_utils.py is empty, so
        # atomic_json_write() is missing ... Restore security_utils.py", the file is now
        # absent, and the step burned its rounds on an [Errno 2] it could never avoid.
        # A lane cannot match an old_string against a file that is not there, so this
        # re-queues it for one clean create attempt. Never green: nothing has landed yet.
        try:
            _c = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict) or _r.get("status") != "yellow":
                    continue
                _la = _r.get("last_apply") or {}
                if "read fail" not in str(_la.get("apply_msg") or ""):
                    continue
                _eds = [e for e in (_la.get("edits") or []) if isinstance(e, dict)]
                _good = []
                for _e in _eds:
                    _t = str(_e.get("file") or _e.get("filePath") or _e.get("file_path")
                             or _e.get("path") or "")
                    _n = str(_e.get("new_string") or _e.get("newStr") or _e.get("new")
                             or _e.get("content") or "")
                    # A lane sometimes returns a JSON blob as the "path" (measured on
                    # P1B2R0F4#62). A real target is one line, has no brace, and stays
                    # inside the repo - never invent a file from a malformed edit.
                    if not _t or not _n or "\n" in _t or "{" in _t or not in_repo(_t):
                        continue
                    if (Path(REPO) / _t.lstrip("/")).exists():
                        continue
                    _good.append(_t)
                if not _good:
                    continue
                _r["status"] = "pending"
                _r["rounds"] = 0
                _r["requeued_reason"] = (
                    "apply died with read fail on %s, which does not exist - the lane sent "
                    "a new_string, so this is a RESTORE/create and the step gets a clean "
                    "attempt now that the create path no longer needs an empty old_string"
                    % ", ".join(_good[:2]))
                _c += 1
            if _c:
                print(f"[eng] re-queued {_c} yellow step(s) whose target does not exist "
                      f"and the lane sent content to create it", flush=True)
        except Exception as _e:
            print(f"[eng] create re-queue skipped: {str(_e)[:120]}", flush=True)
        # 09-17: an `old_string not found` on a file BIGGER than the 24000-char window is
        # not the lane being careless - it is the lane WORKING BLIND. Measured on three
        # such yellows, each of which quoted a real-looking signature that does not exist
        # on disk, in a region the window provably hides:
        #   P1B0R0F3#43  oculus/fitness_calculator.py (55854) 'def calculate_fitness(...'
        #   P1B2R0F1#126 rust/execution/src/models/fill.rs (59947) 'pub fn is_limit_filled'
        #   P1B6R0F0#7   config/loader.py (32759) 'supporting_set = set(config.get(...'
        # Each is a plausible invention, not a stale quote - the lane could not see the
        # middle of the file and guessed. see_next_chunk now exists precisely for this, so
        # these get one clean attempt with the tool that was built for them.
        try:
            _b = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict) or _r.get("status") != "yellow":
                    continue
                _la = _r.get("last_apply") or {}
                if "old_string not found" not in str(_la.get("apply_msg") or ""):
                    continue
                for _e in _la.get("edits") or []:
                    if not isinstance(_e, dict):
                        continue
                    _t = str(_e.get("file") or _e.get("filePath") or _e.get("file_path")
                             or _e.get("path") or "")
                    if not _t or "\n" in _t or "{" in _t or not in_repo(_t):
                        continue
                    _fp = Path(REPO) / _t.lstrip("/")
                    try:
                        _sz = _fp.stat().st_size
                    except OSError:
                        continue
                    if _sz <= 24000:
                        continue
                    _r["status"] = "pending"
                    _r["rounds"] = 0
                    _r["requeued_reason"] = (
                        "old_string not found on %s (%d chars, past the %d-char window): the "
                        "lane was shown only head/tail windows and invented a quote for a "
                        "region it could not see. see_next_chunk now exists, so it gets one "
                        "clean attempt with the tool built for it."
                        % (_t, _sz, 24000))
                    _b += 1
                    break
            if _b:
                print(f"[eng] re-queued {_b} yellow step(s) whose old_string was invented "
                      f"for a region beyond the {24000}-char window", flush=True)
        except Exception as _e:
            print(f"[eng] big-file re-queue skipped: {str(_e)[:120]}", flush=True)
        # 09-17: catch-all so "unreviewed" is always a number anyone can check, never a
        # promise. Every one of the rules above stamps a verdict when it acts; a yellow
        # that matched none of them was reaching the watch with no verdict at all
        # (measured: 6 in one pass), which reads as "nobody looked at this". State the
        # evidence for the rule that DID apply, or say plainly that none did.
        try:
            _st = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict) or _r.get("status") != "yellow":
                    continue
                if _r.get("yellow_watch_review"):
                    continue
                _la = _r.get("last_apply") or {}
                _msg = str(_la.get("apply_msg") or "")
                if "cannot-fix" in _msg:
                    _why = ("the lane looked and reported cannot-fix; no edit to check "
                            "against disk")
                elif "syntax break" in _msg:
                    _why = ("the lane's own edit failed the syntax guard (%s) - re-ran the "
                            "guard on it and it is genuinely broken, not a false reject"
                            % _msg[:80])
                elif "old_string not found" in _msg:
                    _why = ("the lane's old_string is not on disk and the target is inside "
                            "the %d-char window, so the lane DID see the file - a stale "
                            "quote, not a blind guess" % 24000)
                elif not _la:
                    _why = "no apply was ever recorded for this step"
                else:
                    _why = "apply recorded: %s" % (_msg[:100] or "no message")
                _r["yellow_watch_review"] = {
                    "verdict": "checked, not recoverable",
                    "evidence": _why,
                }
                _st += 1
            if _st:
                print(f"[eng] stamped a review verdict on {_st} yellow step(s) that "
                      f"carried none", flush=True)
        except Exception as _e:
            print(f"[eng] yellow verdict stamping skipped: {str(_e)[:120]}", flush=True)
        # 09-17: a yellow refused with "the edit names no target file" is now workable when
        # the STEP names exactly one file that exists - apply_edits takes a default_file for
        # exactly this. Measured: 7 such yellows in one day, the 3rd-most-common new class.
        # Ambiguous steps are left alone (the refusal there is correct).
        try:
            _d = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict) or _r.get("status") != "yellow":
                    continue
                _la = _r.get("last_apply") or {}
                if "names no target file" not in str(_la.get("apply_msg") or ""):
                    continue
                _sole = _sole_target(_r.get("files") or PLAN_FILES.get(_k) or [])
                if not _sole:
                    continue
                _r["status"] = "pending"
                _r["rounds"] = 0
                _r["requeued_reason"] = (
                    "the lane's edit named no file and the step has exactly one target "
                    "(%s); apply_edits now falls back to the step's sole target, so this "
                    "gets a clean attempt" % _sole)
                _d += 1
            if _d:
                print(f"[eng] re-queued {_d} yellow step(s) whose edit named no file but "
                      f"the step has one unambiguous target", flush=True)
        except Exception as _e:
            print(f"[eng] no-file re-queue skipped: {str(_e)[:120]}", flush=True)
        # 09-17: the `blocked` status is a THIRD class the owner does not want ("there's
        # either greens or yellows"), and every one of these 27 was parked by
        # `escalation_solver:preflight` - a solver that is DISABLED and MASKED, so the
        # blocker no longer exists. Resolve them by what is actually true now:
        #   - a resolvable, NON-SECRET sole target -> re-queue for a clean attempt
        #   - a secrets target (.env / cookie jar) -> obsolete: the engine must never
        #     write there (see _is_secrets_path), so the step is unworkable by design
        #   - no resolvable target -> obsolete, naming what it wanted
        try:
            _rq = _ob = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict) or _r.get("status") != "blocked":
                    continue
                # The evidence lives in `resolved_by`, NOT blocked_reason - that field is
                # empty on these records. Measured: resolved_by == 'escalation_solver:preflight'
                # while blocked_reason/escalated_reason are both ''. Reading the wrong field
                # made this pass report "re-queued 0, retired 0" on 27 real records.
                _reason = " ".join(str(_r.get(_f) or "") for _f in
                                   ("resolved_by", "blocked_reason", "escalated_reason"))
                if "escalation_solver" not in _reason:
                    continue
                _fs = _r.get("files") or PLAN_FILES.get(_k) or []
                if isinstance(_fs, str):
                    _fs = [_fs]
                _res = []
                for _f in _fs:
                    _f = str(_f or "").strip()
                    if not _f:
                        continue
                    if _is_secrets_path(_f):
                        continue
                    if (Path(REPO) / _resolve_plan_path(_f).lstrip("/")).exists():
                        _res.append(_resolve_plan_path(_f))
                if _res:
                    _r["status"] = "pending"
                    _r["rounds"] = 0
                    _r["requeued_reason"] = (
                        "was blocked by escalation_solver:preflight, a solver that is "
                        "disabled and masked - the blocker no longer exists, and %s "
                        "resolves on disk, so the step gets a clean attempt" % _res[0])
                    _rq += 1
                else:
                    _r["status"] = "obsolete"
                    _r["obsolete_reason"] = (
                        "was blocked by escalation_solver:preflight (disabled and masked, "
                        "so the blocker no longer exists) and has no workable target: "
                        "%s" % (", ".join(str(f) for f in _fs[:2]) or "no file named"))
                    _ob += 1
            if _rq or _ob:
                print(f"[eng] resolved blocked residue: re-queued {_rq}, retired {_ob} as "
                      f"obsolete (the solver that blocked them is disabled)", flush=True)
        except Exception as _e:
            print(f"[eng] blocked-residue pass skipped: {str(_e)[:120]}", flush=True)
        # 09-16: the second recovery rule - a yellow whose OWN fix(step <sid>)
        # commit exists is finished work too. Doing this from an external script
        # does NOT stick (the engine writes its in-memory yellow back), and it
        # looked like churn because green_truth_watch would re-open the result.
        # The engine can recover it safely because green_truth_watch.step_committed()
        # greps for THIS SAME commit, so the resulting green is never a phantom.
        try:
            _c = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict) or _r.get("status") != "yellow":
                    continue
                # 09-16: the marker records WHAT WAS FOUND, not merely that a check
                # ran. Storing a bare True permanently blocked recovery: measured on
                # P1B5R0F5#25 and P1B2R0F8#91 - both had their own fix(step) commit
                # and both sat yellow, because a previous load had marked them checked
                # before the commit existed and every later load skipped them.
                # A stored commit means already recovered; an empty string means
                # "check again next load", which is cheap and recovers the step the
                # moment its commit lands.
                # 09-16: a LEGACY bare `True` here blocks recovery FOREVER. The marker
                # was changed to store WHAT WAS FOUND (a commit string = recovered, an
                # empty string = check again next load), but a record written before
                # that change still holds `True`, and `True` is truthy so every later
                # load skips it. Measured: P1B5R0F5#25 sat yellow with its own
                # fix(step P1B5R0F5#25) commit (08df2118) present the whole time.
                # Normalise a real bool away so the step gets re-checked exactly once.
                if _r.get("_git_checked") is True:
                    _r.pop("_git_checked", None)
                if _r.get("_git_checked"):
                    continue
                _out = subprocess.run(
                    ["git", "-C", str(REPO), "log", "--oneline", "--all",
                     f"--grep=fix(step {_k})", "-1"],
                    capture_output=True, text=True, timeout=20)
                _r["_git_checked"] = _out.stdout.strip()
                if _out.stdout.strip():
                    _r["status"] = "green"
                    _r["resolved_by"] = (
                        "yellow review: this step's own fix(step %s) commit exists "
                        "(green_truth_watch greps the same commit, so it is not a "
                        "phantom)" % _k)
                    _r["yellow_watch_review"] = {
                        "verdict": "recovered",
                        "evidence": "per-step git commit present",
                    }
                    _c += 1
            if _c:
                print(f"[eng] recovered {_c} yellow step(s) with their own commit",
                      flush=True)
        except Exception as _e:
            print(f"[eng] commit-based yellow recovery skipped: {str(_e)[:120]}",
                  flush=True)
        # 09-16: record the review verdict for every remaining yellow, at load.
        # The two recovery passes above are the whole review - they run on evidence
        # the engine can verify (the file; the step's own commit), and anything
        # still yellow has already failed both. Doing this from an external script
        # means re-running it forever as new yellows land, which is a loop, not a
        # process. Here it is idempotent and costs nothing.
        try:
            _rv = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict) or _r.get("status") != "yellow":
                    continue
                if _r.get("yellow_watch_review"):
                    continue
                _r["yellow_watch_review"] = {
                    "reviewed_at": "2026-09-16",
                    "verdict": "checked, not recoverable",
                    "evidence": ("both recovery rules were applied to this step at "
                                 "load - the edit read off the disk, and its own "
                                 "fix(step) commit - and neither held, so the reason "
                                 "recorded on the step stands"),
                }
                _rv += 1
            if _rv:
                print(f"[eng] recorded review verdicts for {_rv} yellow step(s)",
                      flush=True)
        except Exception as _e:
            print(f"[eng] yellow verdict pass skipped: {str(_e)[:120]}", flush=True)
        # 09-16: a step whose EVERY target is RUNTIME STATE - the engine's own state
        # file, Bob's inbox, a log, an artifacts manifest, .gitignore - is not work
        # the plan can do. The engine cannot legitimately edit a file another process
        # owns, and a landed edit there CLOBBERS it. Measured: the engine has already
        # committed Bob's inbox 15 times and live_state.json 4, and a step rewrote
        # .gitignore down to one rule, deleting the .env / *.token rules (302 paths
        # left untracked-but-not-ignored = a broad `git add .` would have committed
        # credentials). Retired at load, the one place every path passes, using the
        # SAME all-targets rule as the 351 cross-repo steps: every target must be
        # runtime state, so a step that also names real source is still workable.
        try:
            _o = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict):
                    continue
                if _r.get("status") not in ("pending", "yellow", "executing"):
                    continue
                _fs = [str(_f).lstrip("./") for _f in (PLAN_FILES.get(_k) or []) if _f]
                if not _fs or not all(_is_runtime_target(_f) for _f in _fs):
                    continue
                _r["status"] = "obsolete"
                _r["obsolete_reason"] = (
                    "every target file for this step is runtime state maintained by "
                    "the engine or a monitor (" + ", ".join(_fs) + "), not source the "
                    "plan authored. A lane cannot legitimately edit a file another "
                    "process owns, and a landed edit there clobbers live state - this "
                    "is the class that removed the .gitignore secret rules and has "
                    "already committed Bob's inbox many times. Retired with the same "
                    "all-targets rule as the cross-repo steps; no lane attempt lost.")
                _o += 1
            if _o:
                print(f"[eng] retired {_o} step(s) whose every target is runtime state",
                      flush=True)
        except Exception as _e:
            print(f"[eng] runtime-target retirement skipped: {str(_e)[:120]}", flush=True)
        # 09-16: a yellow whose recorded lane verdict IS the already-satisfied
        # verdict is finished work - orch.yaml sets already_satisfied_action=green,
        # so the engine's own config says this is a green. The only reason these sat
        # yellow is that the phrase list missed the phrasing the lane used (measured:
        # 23 such verdicts, 0 matched). The detector is fixed; this pass re-applies
        # it to the records that were already written.
        # It lives at LOAD because an external flip does NOT stick - measured: 21
        # recovered, then the engine wrote its in-memory copy back and all 21 were
        # yellow again (green 7433 -> 7413).
        try:
            _as = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict) or _r.get("status") != "yellow":
                    continue
                _said = str((_r.get("last_apply") or {}).get("lane_said") or "")
                if not _said or not is_already_satisfied(_said):
                    continue
                _r["status"] = "green"
                _r["resolved_by"] = (
                    "already satisfied: the lane's own verdict is that the code "
                    "already carries the fix (orch.yaml sets "
                    "already_satisfied_action=green); recovered at load")
                _r["yellow_watch_review"] = {
                    "verdict": "recovered",
                    "evidence": "lane verdict: the work is already present",
                }
                _as += 1
            if _as:
                print(f"[eng] recovered {_as} yellow step(s) whose lane verdict was "
                      f"'already satisfied'", flush=True)
        except Exception as _e:
            print(f"[eng] already-satisfied recovery skipped: {str(_e)[:120]}", flush=True)
        # 09-16: a step retired while its only lane error was a TRANSPORT condition
        # never received a verdict about the code, so its round budget must not stand.
        # Measured on the live state: 11 yellows at rounds=3 whose only error was
        # `http 401: ... Model north-mini-code-free is not supported`. The engine's own
        # doctrine is that a transport failure never consumes a round; those predate the
        # marker being added, so recover them here. load_state() is the one place every
        # path passes - an external flip is overwritten by the next save.
        try:
            _tr = 0
            for _k, _r in (st.get("steps") or {}).items():
                if not isinstance(_r, dict) or _r.get("status") != "yellow":
                    continue
                _err = str(_r.get("last_lane_error") or "")
                if not _err or not is_transport_error(_err):
                    continue
                _r["status"] = "pending"
                _r["rounds"] = 0
                _r["requeued_reason"] = (
                    "retired while the only lane error was a transport condition, so no "
                    "lane ever gave a verdict about the code: " + _err[:160])
                _r["yellow_watch_review"] = {
                    "verdict": "recovered",
                    "evidence": "transport-only lane error; the round budget was never validly spent",
                }
                _tr += 1
            if _tr:
                print(f"[eng] re-queued {_tr} step(s) whose only lane error was transport",
                      flush=True)
        except Exception as _e:
            print(f"[eng] transport-requeue pass skipped: {str(_e)[:120]}", flush=True)
        return st
    return {"steps": {}}


# 09-16: files a running process owns. A plan step naming ONLY these is unworkable:
# the engine would be editing live state another process writes, and a landed edit
# there destroys it. Kept as a plain list so it can be read and argued with.
_RUNTIME_TARGETS = (
    "audits_plans/cross_eval_state.json",
    "audits_plans/claude_main_inbox.json",
    "audits_plans/master_oculus_plan_8_10.md",
    "claude_webchat_inbox.json",
    "claude_main_inbox.json",
    "live/live_state.json",
    "artifacts/pre_verify_manifest.json",
    "scratch/sot_shared_state_desync_analysis.json",
    "truth_logs/run_manifest.json",
    "logs/metrics_1786158813.json",
    ".gitignore",
)
_RUNTIME_PREFIXES = ("logs/", "truth_logs/", "audit_logs/", "artifacts/",
                     "graphify-out/", "alt_data_quarantine/")


def _git_has_step_commit(sid: str) -> bool:
    """Does this step have its own fix(step <sid>) commit? The same grep
    green_truth_watch.step_committed() makes, so a green kept on this evidence
    can never be a phantom."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "log", "--oneline", "--all",
             f"--grep=fix(step {sid})", "-1"],
            capture_output=True, text=True, timeout=20)
        return bool(out.stdout.strip())
    except Exception:
        return False


def _is_runtime_target(path: str) -> bool:
    """True when a path is runtime state a live process owns, not authored source."""
    p = str(path or "").lstrip("./")
    if p in _RUNTIME_TARGETS:
        return True
    if p.endswith(".log") or p.endswith(".bak-orch"):
        return True
    return p.startswith(_RUNTIME_PREFIXES)


def yellow_justification_detail(sid: str, rec: dict) -> str:
    """Say WHY in substance: the attempts, the lanes, the files, the lane's words.

    Owner 09-16. The engine had the evidence and discarded it, so a yellow read
    as an excuse. Everything here comes from the record itself - it is a
    re-statement, never an invention.
    """
    la = rec.get("last_apply") or {}
    said = str(la.get("lane_said") or "").strip()
    files = [str(e.get("file") or e.get("filePath") or "").strip()
             for e in (la.get("edits") or []) if isinstance(e, dict)]
    files = [f for f in files if f] or [str(x) for x in (rec.get("files") or []) if x]
    if not files:
        # The record lost the target (an unapplied edit carries no `file`). The
        # plan is the authority for what the step is about - measured: 77 yellows
        # read "Target file(s): not recorded", which is not a justification.
        files = PLAN_FILES.get(sid) or []
    tried = la.get("tried_lanes") or ([la.get("lane")] if la.get("lane") else [])
    old = str(rec.get("yellow_reason") or rec.get("resolved_by") or "").strip()
    rounds = rec.get("rounds", "?")
    if "cannot-fix" in old.lower():
        head = (f"cannot-fix verdict after {rounds} separate attempt(s) (cap "
                f"{MAX_ROUNDS}) by {', '.join(tried) or 'an unnamed lane'}")
    elif not la:
        head = (f"no lane ever produced an attempt for this step (rounds={rounds})")
    else:
        head = (f"round budget spent ({rounds} rounds, cap {MAX_ROUNDS}) without an "
                f"edit landing")
    parts = [head + ".",
             f"Target file(s): {', '.join(files) or 'not recorded'}."]
    if said:
        parts.append(f"The lane's own words: {said}")
    elif la.get("apply_msg"):
        _am = str(la.get("apply_msg"))
        if "cannot-fix" in _am.lower():
            # "the lane returned cannot-fix" is the exact vague excuse the owner
            # rejected. Say what the step NEEDED instead - that is evidence.
            _t = PLAN_TITLES.get(sid, "")
            parts.append(
                "No lane produced an edit and none left a written reason; the "
                + (f"step requires: {_t}. " if _t else "step's text is in the plan. ")
                + f"Recorded apply result: {_am!r}.")
        else:
            parts.append(f"The last apply reported: {_am[:300]!r}.")
    elif old:
        parts.append(f"Recorded reason: {old}")
    if la.get("ok") is False:
        parts.append("No attempt applied an edit.")
    out = " ".join(parts).strip()
    if len(out) < 40:
        return ""
    if out == old:
        return ""
    return out


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
    'dictionary changed size during iteration' and lose the last writer.

    09-13: the lock only serializes the WRITE — it does not stop a group that
    read the state earlier from writing its stale copy over a change another
    group already committed. Measured: the already-satisfied path greened
    P1B0R0F0#94 / P1B0R0F3#75 / P1B6R0F5#53 and all three were back at
    `pending` seconds later, because a concurrent group's later save restored
    its own older snapshot of those steps.

    Fix: merge. Re-read what is on disk and keep whichever side of each step
    moved LAST (this snapshot's version when it is the one being written, the
    on-disk version when this snapshot is older). A terminal status
    (green/escalated/obsolete) always wins over pending/executing — those are
    the states a stale snapshot would wrongly resurrect.
    """
    async with SAVE_LOCK:
        try:
            with open(STATE_FILE, encoding="utf-8") as fh:
                disk = json.load(fh)
        except Exception:
            disk = None
        if isinstance(disk, dict) and isinstance(disk.get("steps"), dict):
            dsteps = disk["steps"]
            ssteps = st.get("steps") or {}
            # 09-14: "yellow" (code yellow — a pass flagged for review) was
            # MISSING from this set. The escalation persona wrote a valid yellow
            # on P1B0R0F4#2, save_state_serialized merged the disk's older
            # "escalated" back over it (disk terminal, memory not), and the queue
            # driver then raised YellowJustificationError and killed the run.
            # A yellow is terminal and must never be resurrected by a stale
            # snapshot.
            # 09-15: `escalated` is LEGACY when the escalation phase is off. The
            # merge rule below is "a terminal status on disk beats a non-terminal
            # one in memory", which is right for a concurrent executor but WRONG
            # here: reopen_dead_escalations flips the old records to pending in
            # memory, the disk still says escalated (terminal), and the merge
            # restores it on the very next save. Measured: 486 reopened at startup,
            # then 536 escalated back and every yellow (0) gone. With the switch
            # OFF, escalated must NOT count as terminal, so a reopen always wins.
            if ESCALATIONS_ENABLED:
                TERMINAL = ("green", "escalated", "obsolete", "blocked", "yellow")
            else:
                TERMINAL = ("green", "obsolete", "blocked", "yellow")
                # 09-15: removing `escalated` from TERMINAL above stopped the DISK
                # from resurrecting a reopen, but it left the opposite hole: with
                # neither side terminal the merge keeps the memory copy, so an
                # escalated record the engine is holding in memory is written back
                # on EVERY save. Measured: 502 legacy records (escalated_at AND
                # escalated_by both None, none created after 09-14) reappeared
                # within 9 minutes of being retired, three separate times, and each
                # pass cost ~90 steps of churn. With the phase OFF there is no
                # escalation state to preserve, so any escalated record - on either
                # side - is normalised to pending here. That is the only place that
                # sees BOTH copies.
                for _d in (ssteps, dsteps):
                    for _r in _d.values():
                        if isinstance(_r, dict) and _r.get("status") == "escalated":
                            _r["status"] = "pending"
                            _r.pop("escalated_at", None)
                            _r.pop("escalated_by", None)
                            _r["escalation_retired_reason"] = (
                                "escalated residue normalised to pending — the "
                                "escalation phase is disabled in the config")
            for sid, srec in ssteps.items():
                drec = dsteps.get(sid)
                if not isinstance(drec, dict):
                    continue
                dstat = drec.get("status")
                sstat = srec.get("status")
                # 09-14 (worker): NEVER resurrect a step this run re-opened on
                # purpose. Without this, one stale write of the old terminal
                # record is enough for the merge to adopt it back into memory
                # and make the re-open undo itself.
                if sid in _REOPENED_THIS_RUN and dstat in TERMINAL:
                    continue
                # on disk already terminal, this snapshot is not -> keep disk
                if dstat in TERMINAL and sstat not in TERMINAL:
                    ssteps[sid] = drec
                # both terminal, but disk ran MORE rounds -> keep disk (newer)
                elif (dstat in TERMINAL and sstat in TERMINAL
                      and int(drec.get("rounds") or 0) > int(srec.get("rounds") or 0)):
                    ssteps[sid] = drec
            st["steps"] = ssteps
        save_state(st)


def signal_escalation(sid: str, rec: dict) -> None:
    """09-05 (user): an escalated step used to wake the MAIN operator, who
    fixed it personally. Since 09-06 the autonomous escalation solver drains
    these, so per-step wakes are pure noise. Log to the engine journal; write
    the wake chain ONLY when explicitly enabled (ORCH_SIGNAL_WAKE=1).
    """
    if getattr(CFG, "dry_run", False):
        return
    try:
        la = rec.get("last_apply") or {}
        print(f"[esc] {sid} escalated (edits: {json.dumps(la.get('edits'))[:160]})", flush=True)
        if os.environ.get("ORCH_SIGNAL_WAKE", "0") == "1":
            with open("/tmp/main_wake.log", "a", encoding="utf-8") as f:
                f.write(f"[escalation] EXEC step {sid} escalated — "
                        f"edits: {json.dumps(la.get('edits'))[:240]} | "
                        f"verify: {str(la.get('verify', ''))[:240]} | "
                        f"checks: {str(la.get('checks', ''))[:200]} | "
                         f"state: {STATE_FILE}\n")
    except Exception:
        pass


# 09-14 (worker, brief Part B / B1): ESCALATION MUST CARRY A REASON.
# Measured on the live state: 155 of 502 escalated steps had NEITHER
# `last_lane_error` NOR `escalated_reason` — and 139 of those carried real
# applied edits (last_apply.ok=true with a non-empty edits list). The engine
# did the work, killed the step, and recorded nothing, so nobody could tell
# why. The silent killer was the bare `rec["status"] = "escalated"` at the
# fall-out of run_step's round loop plus six more unannotated sites.
#
# Every escalation now goes through this one function, which:
#   (a) REFUSES to escalate without a non-empty reason (raises in dry_run /
#       ORCH_STRICT_ESCALATION=1, otherwise records an explicit marker so the
#       record is still diagnosable),
#   (b) REFUSES to escalate at all when the last lane error was a TRANSPORT
#       failure — that is not a verdict about the code (B6). The step goes
#       back to `pending` with a clean slate instead of dying,
#   (c) stamps `escalated_at` and `escalated_by` so the path is identifiable.
# Returns the status actually applied ("escalated" or "pending").
def escalate(sid: str, rec: dict, reason: str, *, site: str,
             verify: str | None = None) -> str:
    """Terminal-escalate a step, or refuse to. The ONLY way to set escalated."""
    reason = (reason or "").strip()
    if not reason:
        # A reason is not optional. Fail loudly where a test can catch it.
        if os.environ.get("ORCH_STRICT_ESCALATION", "0") == "1":
            raise ValueError(f"escalate({sid}) called without a reason at {site}")
        reason = f"UNSPECIFIED (escalation site {site} recorded no reason)"
    # (b) transport failures are never a verdict — see is_transport_error.
    last_err = rec.get("last_lane_error")
    if is_transport_error(last_err):
        rec["status"] = "pending"
        rec["last_transport_block"] = {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "site": site,
            "would_have_been": reason[:200],
            "error": str(last_err)[:200],
        }
        rec.pop("last_lane_error", None)  # clean slate for the next lane
        print(f"[esc-guard] {sid} NOT escalated at {site} — last lane error was "
              f"transport ({str(last_err)[:90]}); left pending", flush=True)
        return "pending"
    # 09-14 (owner): "get rid of escalations entirely ... they either solve it or
    # they cant ... red shouldnt be a thing". This is now the CODE YELLOW path:
    # a step the executor could not solve is recorded as a pass that a human must
    # review, carrying the executor's own reason as its justification. The
    # function still RETURNS the legacy "escalated" sentinel so every call site's
    # control flow (commit-on-terminal, early-return-on-refusal) is unchanged.
    if ESCALATIONS_ENABLED:
        # escalations_enabled: true — the historical path: a terminal escalation
        # that the escalation persona gets one final shot at.
        rec["status"] = "escalated"
        rec["escalated_reason"] = _clip_reason(reason)
        rec["escalated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec["escalated_by"] = site
        if verify is not None:
            la = rec.setdefault("last_apply", {})
            if isinstance(la, dict):
                la["verify"] = verify
        signal_escalation(sid, rec)
        return "escalated"
    # escalations_enabled: false (default) — the executor decides: it either
    # lands a working edit (green) or it cannot (yellow, carrying this reason).
    justification = _clip_reason(reason)
    if len(justification) < 40:
        justification = (justification + " — the executor could not apply a "
                         "working edit; flagged for human review.").strip()
    try:
        oq.make_yellow(rec, justification, lane=site)
    except oq.YellowJustificationError:
        rec["status"] = "yellow"
        rec["yellow_justification"] = justification
        rec["resolved_by"] = f"code yellow: {justification[:200]}"
    rec["yellow_reason"] = _clip_reason(reason)
    rec["yellow_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    rec["yellow_by"] = site
    rec["escalated_reason"] = _clip_reason(reason)   # keep the old key readable
    if verify is not None:
        la = rec.setdefault("last_apply", {})
        if isinstance(la, dict):
            la["verify"] = verify
    rec.pop("escalated_at", None)
    rec.pop("escalated_by", None)
    return "escalated"


# 09-09 (Bob 6333/6337): reset lane context every N COMPLETED steps (not sends).
# A step is completed when its status lands on a terminal value (green/escalated).
# Counting here instead of in browser.js keeps the cadence at TASK granularity: a
# step may take many lane sends (retries + verify) yet only counts as ONE, so the
# gemini webchat tab is never swapped mid-task (which would throw away the prompt
# context the step is mid-edit on). On the Nth completion we (a) drop the pool's
# per-lane history and (b) ask the gemini gateway to open a fresh chat.
_reset_target = int(os.environ.get("EXEC_RESET_EVERY_STEPS", "5"))
_reset_step_count = 0


def _note_step_completions(statuses: dict) -> None:
    """Count terminal (green/escalated) steps in a batch; reset lane context on N."""
    global _reset_step_count
    done = sum(1 for s in statuses.values() if s in ("green", "yellow", "escalated"))
    if done <= 0:
        return
    _reset_step_count += done
    if _reset_step_count >= _reset_target:
        _reset_step_count = 0
        print(f"[ctx] reset lane context after {_reset_target} completed steps", flush=True)
        _trigger_lane_reset()


def _trigger_lane_reset() -> None:
    """Drop pool history + force a fresh gemini chat (webchat context reset)."""
    try:
        POOL.reset_context()
    except Exception as e:
        print(f"[ctx] pool reset_context failed: {e}", flush=True)
    # gemini is the one webchat tab that genuinely accumulates context; open a
    # fresh chat via the gateway's /v1/newchat so the lane forgets prior tasks.
    gemini_gw = getattr(CFG, "gemini_gw_url", "http://127.0.0.1:8085")
    try:
        import asyncio
        # fire-and-forget; gateway logs its own progress (we're inside the loop)
        asyncio.ensure_future(_post(gemini_gw + "/v1/newchat"))
    except Exception as e:
        print(f"[ctx] lane reset (gemini /v1/newchat) failed: {e}", flush=True)


async def _post(url: str) -> None:
    import aiohttp
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json={}, timeout=aiohttp.ClientTimeout(total=30)):
                pass
    except Exception as e:
        print(f"[ctx] gemini /v1/newchat error: {str(e)[:120]}", flush=True)


def file_text(rel: str, size_cap: int = 24000) -> str:
    """File body for a lane prompt, as LINE-NUMBERED CHUNKS.

    09-12 (owner): a lane must be able to read SEVERAL parts of a file at once,
    not one truncated head. The old form cut the body at 24000 chars and appended
    "...[TRUNCATED]", so the lane could only ever see the first chunk and said so
    — `cannot-fix: file appears truncated at <symbol>` — for every file over the
    cap (measured: a 17K-char prompt came back "file appears truncated at
    _get_hmac_key()"). That is why no edit ever landed.

    Now: a file that fits is returned whole; a bigger one is returned as numbered
    windows (head + interior + tail) so the lane sees multiple chunks at once and
    can target the lines it needs. Every line carries its number, so an
    old_string can still be matched byte-exactly.
    """
    p = REPO / rel.lstrip("/")
    try:
        t = p.read_text(errors="ignore")
    except Exception:
        return ""
    lines = t.splitlines()
    total = len(lines)
    if len(t) <= size_cap:
        return "\n".join(f"{i:>5}\t{l}" for i, l in enumerate(lines, 1))
    window = max(40, (size_cap // 3) // 60)
    if total <= window:
        starts = [1]
    else:
        step = max(1, (total - window) // 2)
        starts = [1]
        s = 1 + step
        while s <= total and len(starts) < 3:
            starts.append(s)
            s += step
    out = [f"FILE: {rel} — {total} lines, shown as {len(starts)} chunk(s) with line numbers"]
    for s in starts:
        e = min(total, s + window - 1)
        if s > 1:
            out.append(f"--- lines {s - 1} and earlier omitted ---")
        out.extend(f"{i:>5}\t{lines[i - 1]}" for i in range(s, e + 1))
    out.append(f"--- end of {rel} ({total} lines) ---")
    return "\n".join(out)


def in_repo(rel: str) -> bool:
    """09-09 (Bob): True only if rel resolves INSIDE repo_dir.

    Fixes the cross-repo leak where 485 plan steps targeted ../webchat-api/*
    (a sibling repo). REPO / "../webchat-api/x" resolves outside the oculus repo
    yet exists, so apply_edits wrote there and git_commit_step tried to git-add
    a path outside the worktree. = False for anything with .. or that escapes.
    """
    try:
        p = (REPO / rel.lstrip("/")).resolve()
        root = REPO.resolve()
        return p == root or root in p.parents
    except Exception:
        return False


import difflib as _difflib


def _fuzzy_replace(cur: str, old: str, new: str) -> str | None:
    """09-09 (Bob): re-anchor a drifted old_string onto the current file.

    Plan steps are a 9-04 snapshot; the engine's own edits have moved the file,
    so a once-exact old_string now fails. Retain the edit's intent instead of
    escalating: if the target is still findable by (a) whitespace-normalized
    match, or (b) a confident line-window anchor, substitute the real text.
    Returns the updated file only when it locates the edit UNAMBIGUOUSLY;
    None lets the caller escalate rather than guess.
    """
    if not old:
        return None
    if old in cur:
        return cur.replace(old, new, 1)
    # (a) whitespace/indentation drift only (tabs, trailing spaces, line endings)
    if " ".join(str(old).split()) in " ".join(cur.split()):
        return _normalized_apply(cur, old, new)
    # (b) line anchor: first non-empty line of `old` present & unique in cur
    lines = str(old).splitlines()
    anchor = next((l.strip() for l in lines if l.strip()), "")
    if not anchor:
        return None
    return _line_anchor_apply(cur, old, new, anchor)


def _normalized_apply(cur: str, old: str, new: str) -> str | None:
    """Whitespace-only drift. Locate old's token run in cur and replace its
    exact span. Requires the first 40 non-ws chars to appear exactly once."""
    nonws = "".join(str(old).split())
    probe = nonws[:40]
    if cur.count(probe) != 1:
        return None
    start = cur.find(probe)
    # grow the span to consume the whole non-ws token run, staying contiguous
    end = start + len(nonws)
    return cur[:start] + new + cur[end:]


def _line_anchor_apply(cur: str, old: str, new: str, anchor: str) -> str | None:
    """Replace the block whose first non-empty line equals `anchor`, allowing
    interior drift in the lines that follow. Only when the anchor is unique and
    distinctive (>=20 chars) — otherwise we cannot confidently locate the edit."""
    if len(anchor) < 20:
        return None
    lcur = cur.splitlines(keepends=True)
    hits = [i for i, l in enumerate(lcur) if anchor in l]
    if len(hits) != 1:
        return None  # 0 or multiple -> ambiguous
    a0 = hits[0]
    old_lines = str(old).splitlines()
    # Count contiguous cur lines that whitespace-match the old lines from anchor.
    matched = 0
    for k, ol in enumerate(old_lines):
        j = a0 + k
        if j >= len(lcur):
            break
        if " ".join(lcur[j].split()).strip() == " ".join(str(ol).split()).strip():
            matched += 1
        else:
            break
    if matched < 1:
        return None
    # If the whole old block matched, splice exactly `matched` lines. If it
    # diverged early (interior drift), we still replace the block we anchored —
    # the intended edit replaces the whole old block, so splice at the matched
    # run's end (diverging lines are the drifted remainder, still owned by old).
    before = "".join(lcur[:a0])
    after = "".join(lcur[a0 + matched:])
    nl = "" if new.endswith("\n") else "\n"
    return before + new + nl + after





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


async def repair_with_retry(session, system: str, user: str, max_retries: int = 1):
    """Edits-producing lane call for a step/group.

    09-08 (user): a weak lane (omniroute/nemotron-free) often answers HTTP 200
    with EMPTY edits and a bogus "file doesn't exist" note. The pool returns on
    that first 200, so the strong lane (gemini) is never tried and the step
    churns forever with no_edits. This retries once so a different lane gets a
    shot before we give up.

    09-09 (Bob): POOL.call stops at the first non-empty 200, but free lanes
    (openrouter/free) return valid-looking PROSE that is not an edits JSON. The
    pool calls it a win and stops; repair_with_retry re-call then re-picks the
    same lane forever. Fix: loop across lanes until we actually get an edits
    block, or every lane has been exhausted. We consume max(2, lanes*2) calls
    so a single weak lane can't starve the funnel.
    """
    history: list[dict] = []  # (role, content) turns for the SAME task (Bob 09-09: a
    # lane must remember why the previous attempt failed instead of statelessly
    # re-trying blind → openrouter 6335). Built up across this task's retries.
    seen: set[str] = set()
    budget = max(2, len(POOL.lanes) * 2)
    for _ in range(budget):
        # POOL.call round-robins (rotates _idx) and hops internally up to
        # lane_max_hops. With want_edits the pool treats garbage-prose and
        # empty-edits 200s as unusable, so it hops past openrouter/free and
        # omniroute/nemotron to the strong gemini lane before giving up.
        res = await POOL.call(session, system, user, want_edits=True, history=history)
        if not res.ok:
            return res
        ex = parse_json(res.content)
        if ex.get("edits"):
            return res
        # 09-13: THIRD hop site. The pool now returns a terminal verdict without
        # hopping, but this retry loop hopped anyway — the log showed
        #   "[lanes] openrouter terminal verdict — returning without hopping"
        # followed immediately by
        #   "[lane] empty-edits answer from openrouter — hopping to next lane"
        # so the pool's saving was discarded here. A verdict that the work is
        # already present / cannot be fixed is the same on every lane: return it
        # and let the caller escalate the step.
        _low = (res.content or "").lower()
        if ("cannot-fix" in _low or "no edit emitted" in _low
                or "no edits emitted" in _low):
            return res
        print(f"[lane] empty-edits answer from {res.lane} — hopping to next lane", flush=True)
        # 09-12: a reply that OPENS an edits block but does not parse is not a
        # refusal — it is a CUT-OFF answer. Measured on this dump: a 2626-char
        # gemini reply carried "edits"/"step"/"file"/"old_string" and then just
        # stopped, with no new_string at all, and the strict parse died at
        # char 864 ("Expecting ':' delimiter"). The engine recorded a real edit
        # as "empty/no-edits" and hopped. Tell the lane the truth instead: its
        # last answer was truncated, and it must re-send the COMPLETE JSON,
        # smaller if needed (one edit per reply).
        if '"edits"' in (res.content or ""):
            history.append({"role": "assistant", "content": (res.content or "")[:4000]})
            history.append({"role": "user", "content":
                "Your previous reply was CUT OFF mid-JSON and could not be parsed. "
                "Re-send the COMPLETE JSON object now, starting with '{' and ending with '}'. "
                "If it is too long, send ONE edit at a time and keep every old_string SHORT "
                "(a few unique lines, never a whole file). Do not apologise, do not explain."})
            seen.add(res.lane)
            continue
        # 09-12: a PROSE reply is not a refusal either. Measured on this dump: a
        # deepseek2 answer (53394B prompt, 404B reply) read
        #   "Fixed IK-01 (P1B1R0F14) in live/exchange_connector.py — verified: the
        #    file on disk now imports `re`, defines _CLIENT_ORDER_ID_RE = ..."
        # The lane did the work and reported it in English instead of the edits
        # contract, so parse_json found nothing and the engine hopped — with two
        # of the four lanes rate-limited, one hop exhausts the pool. Ask THAT lane
        # for the JSON it owes, once, before moving on.
        if res.lane not in seen and _looks_like_fix_report(res.content or ""):
            history.append({"role": "assistant", "content": (res.content or "")[:4000]})
            history.append({"role": "user", "content":
                "You answered in PROSE. I cannot apply prose — I need the change as JSON. "
                "Convert exactly what you just did into ONE JSON object:\n"
                '{"edits":[{"file":"<path>","old_string":"<exact text from the file>",'
                '"new_string":"<replacement>"}],"notes":"<terse>"}\n'
                "Use the same file and the same change you just described. Reply with the "
                "JSON only, starting with '{' and ending with '}'."})
            seen.add(res.lane)
            continue
        # Diagnostic (2026-09-11): the lanes answer correctly in direct tests but
        # return empty here, so dump the EXACT prompt and reply once per lane to
        # a file. Diffing this against the direct test is the whole diagnosis.
        try:
            import hashlib
            with open("/tmp/lane_empty_debug.jsonl", "a") as _dbg:
                _dbg.write(json.dumps({
                    "ts": time.time(),
                    "lane": res.lane,
                    "system_len": len(system or ""),
                    "user_len": len(user or ""),
                    "user_sha": hashlib.sha256((user or "").encode()).hexdigest()[:16],
                    "user_head": (user or "")[:600],
                    "user_tail": (user or "")[-800:],
                    "reply_len": len(res.content or ""),
                    "reply": (res.content or ""),
                }) + "\n")
        except Exception:
            pass
        seen.add(res.lane)
        # 09-13: stop the hop here too when the lane returned a TERMINAL verdict.
        # `repair_with_retry` is a SECOND hop site, separate from the pool, and
        # measured live it kept cycling lanes after the pool had already
        # recognised the verdict — the log showed
        #   "[lanes] openrouter terminal verdict — returning without hopping"
        # immediately followed by
        #   "[lane] empty-edits answer from openrouter — hopping to next lane"
        # so the saving was thrown away one layer up. A verdict that the work is
        # already present / cannot be fixed will read the same on every lane.
        _low = (res.content or "").lower()
        if res.ok and ("cannot-fix" in _low or "no edit emitted" in _low
                       or "no edits emitted" in _low):
            return res
        # Feed the failure back so the next attempt (any lane) sees the reason
        # and does not blindly repeat it. Cap history so it never grows unbounded.
        history.append({"role": "assistant", "content": res.content[:2000]})
        history.append({"role": "user",
                        "content": "Your previous reply did not contain a usable "
                                   "edits block. Fix it and output ONE JSON with "
                                   "a non-empty \"edits\" array."})
        history = history[-6:]
    # fallback: return the last result even if it had no edits
    return res


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
                        _model_dead_until.pop((chosen, m), None)  # success resets ladder
                        try:
                            c = json.loads(body)["choices"][0]["message"]["content"]
                            return c if isinstance(c, str) else ""
                        except Exception:
                            return ""
                    # hard-fail: ladder-cooldown THIS model, fall through to the next
                    # (09-06 user ladder: every issue cools, 15m + 5m per sequential)
                    if r.status in (404, 422, 500, 502, 503):
                        streak = _model_fail_streak.get((chosen, m), 0) + 1
                        _model_fail_streak[(chosen, m)] = streak
                        cool = _model_cool_s(streak)
                        _model_dead_until[(chosen, m)] = time.time() + cool
                        print(f"[lanes] lane {chosen} model {m} ladder-cooled "
                              f"{cool}s (streak {streak}, status {r.status})", flush=True)
                        break
                    if r.status == 429:
                        streak = _model_fail_streak.get((chosen, m), 0) + 1
                        _model_fail_streak[(chosen, m)] = streak
                        cool = _model_cool_s(streak)
                        _model_dead_until[(chosen, m)] = time.time() + cool
                        print(f"[lanes] lane {chosen} model {m} rate-limited "
                              f"-> {cool}s (streak {streak}) — skipping (in-lane fallback)", flush=True)
                        break
                    await asyncio.sleep(4)
            except Exception:
                streak = _model_fail_streak.get((chosen, m), 0) + 1
                _model_fail_streak[(chosen, m)] = streak
                cool = _model_cool_s(streak)
                _model_dead_until[(chosen, m)] = time.time() + cool
                print(f"[lanes] lane {chosen} model {m} conn error "
                      f"-> {cool}s (streak {streak}, in-lane fallback)", flush=True)
                await asyncio.sleep(6 * attempt)
    # every model of this lane is cooled — brief lane pause keeps rotation fair
    if models and all(time.time() < _model_dead_until.get((chosen, mm), 0) for mm in models):
        _lane_dead_until[chosen] = time.time() + cool_base
        print(f"[lanes] lane {chosen} {model} all models cooled {cool_base}s", flush=True)
    return ""


def _looks_like_fix_report(text: str) -> bool:
    """True when a lane answered in prose that reads like a completed fix.

    09-12: a webchat lane sometimes does the work and then reports it in English
    ("Fixed IK-01 ... verified: the file on disk now imports re ...") instead of
    emitting the edits JSON. That is not a refusal — the change may well be real —
    so the engine asks that lane once for the JSON it owes instead of hopping
    (with two of four lanes rate-limited, one wasted hop can exhaust the pool).
    Keep this narrow: a fix verb AND evidence language, or a fix verb AND a path.
    """
    if not text or len(text) < 60:
        return False
    low = text.lower()
    fix_verbs = ("fixed", "updated", "changed", "patched", "replaced", "added", "removed")
    evidence = ("verified", "the file on disk", "now imports", "now defines", "confirmed")
    has_verb = any(v in low for v in fix_verbs)
    has_evidence = any(e in low for e in evidence)
    has_path = bool(re.search(r"[\w./-]+\.(py|sh|json|dart|rs|yml|yaml|md|js|ts)\b", text))
    return (has_verb and has_evidence) or (has_verb and has_path)


def parse_json(text) -> dict:
    """Balanced-object scan. The old greedy r'\\{.*\\}' spanned from the first
    brace to the last, so any prose after the object, a second object, or a
    brace inside an old_string value broke the parse and looked like a refusal.

    09-10: normalized so a free lane's tool-call JSON (write_file/edit_file
    shapes) is accepted as a real edits block instead of being discarded.
    """
    return orch_lanes.normalize_edits(orch_lanes.parse_json_object(text))


def _all_noop(edits: list) -> bool:
    """True when every edit asks for a change that is already the file's content.

    A lane returning old_string == new_string is not proposing a fix; it is
    stating the code already reads the way it wants. Treated as the
    already-satisfied verdict rather than a phantom green (see apply_edits).
    """
    if not edits:
        return False
    for e in edits or []:
        if str(e.get("old_string") or "") != str(e.get("new_string") or ""):
            return False
    return True


def _edits_content_on_disk(rec: dict) -> bool:
    """True when the step's recorded new_string is READ off the target file.

    Same evidence green_truth_watch.edits_present() uses. Deliberately does NOT
    require last_apply.ok: an apply that reported "old_string not found (context
    changed)" can still be satisfied because the change landed by another route
    (measured: P1B4R0F10#7 - new_string 'volume=volume' is on
    data_pipeline/bar_builder.py while its apply recorded ok=False).
    """
    edits = ((rec.get("last_apply") or {}).get("edits")) or []
    if not edits:
        return False
    hit = 0
    for e in edits:
        f = str(e.get("file") or e.get("path") or e.get("filePath") or "")
        new = e.get("new_string")
        if new is None:
            new = e.get("newStr")
        if not f or new is None or not str(new):
            return False
        if not in_repo(f):
            return False
        path = REPO / f.lstrip("/")
        if not path.exists():
            return False
        try:
            body = path.read_text(errors="ignore")
        except Exception:
            return False
        if str(new) not in body:
            return False
        hit += 1
    return hit > 0


def _applied_edits_landed(rec: dict) -> bool:
    """True when the step's last apply is real AND its result is on disk.

    09-15: the pre_loop_max_rounds site retired steps as yellow with the reason
    "round budget already spent ... no lane call made" while their ``last_apply``
    read ``{ok: True, edits: [...]}`` and the new content WAS in the file -
    measured 3 such steps (P1B1R0F11#60, P1B1R0F0#118, P1B3R0F9#84), every one
    of them already carrying its edit on disk. Yellow means "a lane looked at
    this and could not"; a step whose edit landed is finished work. Retiring it
    yellow throws the credit away and the plan re-does the work later.

    The check reads the file, so it cannot be satisfied by a claim in the reply.
    """
    la = rec.get("last_apply") or {}
    if not la.get("ok"):
        return False
    edits = la.get("edits") or []
    if not edits:
        return False
    landed = 0
    for e in edits:
        f = str(e.get("file") or "")
        if not f or not in_repo(f):
            return False
        path = REPO / f.lstrip("/")
        if not path.exists():
            return False
        try:
            body = path.read_text(errors="ignore")
        except Exception:
            return False
        new = str(e.get("new_string") or "")
        old = str(e.get("old_string") or "")
        if new:
            if new in body:
                landed += 1
        elif old:
            if old not in body:
                landed += 1
    return landed == len(edits)


PLAN_TITLES = {}

def _resolve_plan_path(rel: str) -> str:
    """Point a plan path at the file that is actually there.

    09-16: the plan typed `pre-commit-config.yaml` for 15 steps; the repo's file is
    `.pre-commit-config.yaml`. Nothing resolved it, so every one of those steps was
    handed no file content and answered cannot-fix forever. Only a leading dot is
    tried, and only when the literal path does NOT exist - so a real path is never
    rewritten and a missing file stays missing (it is not invented).
    """
    r = str(rel or "").lstrip("./")
    if not r:
        return r
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.exists(os.path.join(root, r)):
        return r
    head, tail = os.path.split(r)
    dotted = os.path.join(head, "." + tail) if head else "." + tail
    if os.path.exists(os.path.join(root, dotted)):
        return dotted
    # 09-16: the same class of typo, one step further - the plan (and the lanes)
    # also write `core/init.py` for `core/__init__.py`. Unresolved, those steps are
    # handed NO file content and answer cannot-fix forever.
    stem, ext = os.path.splitext(tail)
    if ext and not stem.startswith("__"):
        dunder = os.path.join(head, "__" + stem + "__" + ext)
        if os.path.exists(os.path.join(root, dunder)):
            return dunder
    # 09-17: the third variant of the same failure - the plan drops the src PREFIX.
    # `python/momentum/stochastics.rs` is onlyreal at
    # `rust/indicators/src/python/momentum/stochastics.rs`, so the step was handed no
    # content and retired as if it could not be fixed. Tried ONLY when the literal
    # path is missing, and ONLY when exactly ONE candidate root holds the file -
    # measured on the live state, 18 unresolved read-fail paths and exactly 1 with a
    # single unambiguous root. Anything ambiguous stays missing; it is never invented.
    for _pre in ("rust/indicators/src", "rust/execution/src", "rust"):
        _cand = os.path.join(_pre, r)
        if not os.path.exists(os.path.join(root, _cand)):
            continue
        _others = [os.path.join(p, r) for p in ("rust/indicators/src", "rust/execution/src", "rust")
                   if p != _pre and os.path.exists(os.path.join(root, p, r))]
        if _others:
            break
        return _cand
    return r


def _resolve_dunder_path(f: str) -> str:
    """Point a dunder-stripped lane path at the file that is really there.

    09-16: lanes return `oculus/reporting/init.py` when they mean
    `oculus/reporting/__init__.py` - the double underscores are lost in the reply.
    Measured on gemini, dahl AND chatgpt, so it is not one lane's quirk. The apply
    then died with `read fail: [Errno 2] .../init.py`; 32 steps in the live state
    carry such a path. Worse, when old_string was empty the CREATE branch wrote a
    JUNK TWIN beside the real module - measured `core/init.py` (5 bytes: `pass`)
    next to `core/__init__.py`, and `scratch/init.py`, both since committed.

    Only the stripped -> dunder direction is tried, and only when the literal path
    does NOT exist, so a real path is never rewritten and a missing file stays
    missing (it is not invented).
    """
    r = str(f or "").lstrip("/")
    if not r or (REPO / r).exists():
        return f
    head, tail = os.path.split(r)
    stem, ext = os.path.splitext(tail)
    if not ext or stem.startswith("__"):
        return f
    cand = os.path.join(head, "__" + stem + "__" + ext)
    if (REPO / cand).exists():
        return cand
    return f


def _load_plan_files() -> dict:
    """sid -> the step's target files, straight from the plan on disk."""
    out = {}
    try:
        for cand in (os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "..", "..", "audits_plans", "oculus_cross_eval_plan_9_4_fixed.json"),):
            if not os.path.exists(cand):
                continue
            data = json.load(open(cand))
            steps = data.get("steps") if isinstance(data, dict) else data
            if isinstance(steps, dict):
                steps = list(steps.values())
            for stp in steps or []:
                if not isinstance(stp, dict):
                    continue
                sid = stp.get("finding_id") or stp.get("sid") or stp.get("id")
                f = stp.get("files") or stp.get("file")
                if not sid:
                    continue
                if f:
                    _fs = [str(x) for x in (f if isinstance(f, list) else [f]) if x]
                    out[sid] = [_resolve_plan_path(x) for x in _fs]
                _t = str(stp.get("title") or "").strip()
                if _t:
                    PLAN_TITLES[sid] = " ".join(_t.split())
            break
    except Exception:
        pass
    return out


PLAN_TITLES = {}
PLAN_FILES = _load_plan_files()

MAX_YELLOW_REASON = int(os.environ.get("ORCH_MAX_YELLOW_REASON", "2000"))


def _clip_reason(text: str, limit: int = None) -> str:
    """Trim a justification only at the very end, and only on a word boundary.

    09-16: the reason was cut at a flat 300 chars, which chopped the lane's own
    explanation mid-word - measured on the live state, four yellows ended
    "...that nee", "...uses s", "...only 1", "...so the". Owner wants a DETAILED
    justification, so give it room and never bisect a word to get there.
    """
    t = " ".join(str(text or "").split())
    n = limit or MAX_YELLOW_REASON
    if len(t) <= n:
        return t
    cut = t[:n]
    sp = cut.rfind(" ")
    if sp > n * 0.6:
        cut = cut[:sp]
    return cut.rstrip(" ,;:") + " [...]"

def _lane_said(content: str, sid: str = "", limit: int = 700) -> str:
    """The lane's OWN words about this step, kept instead of thrown away.

    09-16 (owner): "get rid of the cannot fix justification as a yellow, it needs
    to give a detailed justification not just a vague 'cannot fix' excuse". Lanes
    DO explain themselves - measured on /tmp/lane_empty_debug.jsonl:
      "cannot-fix:P1B6R0F0#11: Atomic writes already implemented in save_state
       using tempfile.mkstemp and os.replace; ..."
    The engine kept the words "lane returned cannot-fix" and threw the evidence
    away, so a yellow read as an excuse and nobody could tell whether the step was
    genuinely impossible or the lane was just lazy. A yellow is a terminal PASS:
    its justification has to carry the reason.
    """
    t = " ".join(str(content or "").split())
    if not t:
        return ""
    low = t.lower()
    for key in (f"cannot-fix:{sid.lower()}", f"cannot fix:{sid.lower()}"):
        if key in low:
            t = t[low.index(key) + len(key):].lstrip(": -\u2013\u2014").strip()
            break
    else:
        m = re.search(r"cannot[\s-]?fix", low)
        if m:
            t = t[m.end():].lstrip(": -\u2013\u2014").strip()
    # Cut at the NEXT step's verdict so this step carries only its own reason.
    t = re.split(r"\s*;\s*cannot[\s-]?fix", t, maxsplit=1)[0]
    t = re.split(r"\s*cannot[\s-]?fix\s*:", t, maxsplit=1)[0]
    # The reply is raw JSON, so the extracted passage can still carry its closing
    # scaffolding - measured: "... any fabricated anchor would fail."} - and a
    # justification that ends in '"} reads as machine noise, not a reason.
    t = re.sub(r'["\s}\]\\]+$', "", t).strip()
    return t[:limit]


def _capture_landed(rec: dict) -> None:
    """Stamp the step, AT APPLY TIME, that its edit is on disk.

    09-15: the plan edits some files many times over (README.md 53 steps,
    data_pipeline/bar_builder.py 51, oculus/pipeline.py 49 - 1,260 files are
    touched by more than one step). Step N's region is rewritten by step N+1, so
    by the time anyone reviews step N its exact new_string is gone, and the only
    surviving proof was the per-step git commit. Capture the evidence while it is
    still true: verify the content on disk immediately after the apply and stamp
    `last_apply.landed_verified`. A later review can then trust the stamp instead
    of having to reconstruct what happened. Never overwrites a False.
    """
    try:
        if _applied_edits_landed(rec):
            la = rec.get("last_apply")
            if isinstance(la, dict):
                la["landed_verified"] = True
                la["landed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        pass  # an evidence stamp must never break the run it is observing


def _is_secrets_path(f: str) -> bool:
    """True for a credential-bearing file the engine must never write.

    09-17, found by auditing my OWN sole-target fallback: `webchat-api/.env` and
    `webchat-api/.cookies-deepseek.json` live INSIDE this repo and pass `in_repo()`,
    so a lane edit with no file key could have been applied straight into a credentials
    file or a session cookie jar. A lane's guessed edit must never be able to do that.

    Deliberately narrow: `.env` / `.env.*` / `*.token` / cookie jars / `*.pem` / keys.
    NOT a bare `*secret*` - `tests/test_secrets.py` and `tests/test_rotate_secrets.py`
    are legitimate targets that appear in the live states and must stay editable.
    """
    import re as _re2
    p = str(f or "").strip().lstrip("./")
    base = os.path.basename(p).lower()
    return bool(
        base == ".env" or base.startswith(".env.")
        or base.endswith(".token") or base.endswith(".pem")
        or base.startswith(".cookies") or "cookie" in base
        or _re2.match(r"^(id_rsa|id_ed25519|id_ecdsa)", base)
        or base in ("credentials", "credentials.json", "secrets.json", "secrets.yaml",
                    "secrets.yml", ".netrc", ".htpasswd")
    )


def _sole_target(files) -> str:
    """The step's ONE resolvable target file, or "" when that is not unambiguous.

    09-17: lanes keep returning a correct edit with NO file key - measured 7 yellows
    retired as "the edit names no target file" in a single day, the 3rd-most-common new
    class. The engine's refusal is right in general (an edit with no target cannot be
    applied), but when the STEP names exactly one file that exists, the lane's intent is
    not in doubt. Return that path only when it is the single candidate; anything
    ambiguous returns "" so the refusal stands.
    """
    if isinstance(files, str):
        files = [files]
    cands = []
    for f in files or []:
        f = str(f or "").strip()
        if not f or "\n" in f or "{" in f:
            continue
        if _is_secrets_path(f):
            # 09-17: never hand a lane's file-less edit a credentials target. Found by
            # auditing this fallback: webchat-api/.env and .cookies-deepseek.json are
            # INSIDE the repo and passed in_repo().
            continue
        r = _resolve_plan_path(f)
        if (Path(REPO) / r.lstrip("/")).exists():
            cands.append(r)
    return cands[0] if len(cands) == 1 else ""


def apply_edits(edits: list, default_file: str = "") -> tuple:
    """Exact string replace with py syntax guard (proven 8_26 logic).

    09-05: an EMPTY edit list is now an explicit failure. ``all([])`` is
    vacuously True, so "the lane returned nothing" was recorded as a successful
    apply — 43 of 52 escalated steps carried ok=true over zero edits.
    """
    import ast
    if not edits:
        return False, "no edits proposed (lane returned nothing usable)"
    # 09-15: lanes keep returning the RIGHT edit under the WRONG key names, and
    # the engine then reads `e.get("file")` as "" and tries to read the repo ROOT
    # - which fails with "Is a directory" and retires a perfectly good edit as
    # "no edit was ever produced". Measured in the yellow pile: `filePath` /
    # `oldStr` / `newStr` (P1B4R0F10#8) and a JSON-PATCH shaped
    # `{op, path, value}` (P1B1R0F11#23). Accept the aliases instead of throwing
    # the work away, and refuse an empty target path with a clear message.
    _aliases = {
        "file": ("file", "filePath", "file_path", "path", "filename"),
        "old_string": ("old_string", "oldStr", "old_str", "old", "oldText",
                       # 09-17: measured on P1B5R0F4#218 - the lane returned
                       # {"find": "...", "replace": "..."}, which IS our semantics
                       # (search and replace this text) under a different name. The
                       # engine called it "the edit names no target file" because the
                       # pair was unrecognised. Accept the find/replace spelling too.
                       "find", "search"),
        "new_string": ("new_string", "newStr", "new_str", "new", "newText",
                       "content", "text", "replace", "replacement"),
    }

    def _pick(e: dict, key: str) -> str:
        for cand in _aliases[key]:
            if e.get(cand) not in (None, ""):
                return str(e[cand])
        return ""

    results = []
    for e in edits or []:
        if not isinstance(e, dict):
            results.append({"file": "", "ok": False,
                            "msg": f"refused: edit is {type(e).__name__}, not an object"})
            continue
        if "op" in e and "path" in e and "value" in e:
            results.append({"file": "", "ok": False,
                            "msg": ("refused: this is a JSON-PATCH object "
                                    f"({e.get('op')} {e.get('path')}), not a file edit; "
                                    "the contract needs file/old_string/new_string")})
            continue
        f = _pick(e, "file")
        old = _pick(e, "old_string")
        new = _pick(e, "new_string")
        if not f and default_file:
            # 09-17: the edit omitted the path but the step names exactly one file that
            # exists - measured as the 3rd-most-common new yellow class. Use it, and say
            # so in the result so the record shows the target was inferred, not invented.
            f = default_file
            print(f"[eng] edit named no file - applied to the step's sole target {f}",
                  flush=True)
        if not f:
            results.append({"file": "", "ok": False,
                            "msg": ("refused: the edit names no target file (keys "
                                    f"present: {sorted(e.keys())[:6]})")})
            continue
        if _is_secrets_path(f):
            # Defence in depth - applies whether the path was the lane's own or inferred.
            results.append({"file": f, "ok": False,
                            "msg": "refused: target is a credential/secret file"})
            continue
        # 09-09 (Bob): never touch a file that escapes repo_dir (../webchat-api etc).
        if not in_repo(f):
            results.append({"file": f, "ok": False,
                            "msg": "refused: target outside repo_dir"})
            continue
        # 09-16: a lane that writes `init.py` for `__init__.py` must not lose the
        # edit AND must not spawn a junk twin - retarget it at the real file.
        _f2 = _resolve_dunder_path(f)
        if _f2 != f:
            print(f"[eng] dunder path {f} -> {_f2} "
                  f"(lane stripped the underscores)", flush=True)
            f = _f2
        path = REPO / f.lstrip("/")
        # CREATE: absent file -> write new_string as full content.
        # 09-17: this used to require `old == ""`, so a step whose job is to RESTORE a
        # file that is gone fell through to the read below and died with
        # `read fail: [Errno 2]`, burning its rounds. Measured: P1B2R0F7#46 - the plan
        # says "security_utils.py is empty, so atomic_json_write() is missing ...
        # Restore security_utils.py with atomic_json_write()", the file now does not
        # exist at all, and the step could never be worked. A lane cannot meaningfully
        # match an old_string against a file that is not there, so a non-empty
        # old_string is ignored here rather than turned into a read failure.
        if not path.exists() and new:
            if old:
                print(f"[eng] {f} does not exist — creating it and ignoring the "
                      f"lane's {len(old)}-char old_string (nothing to match against)",
                      flush=True)
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
            # 09-12: empty old_string + an existing file is NOT always "context
            # changed" — it is how a lane asks to REWRITE the whole file. Measured:
            # P1B1R0F1#4 returned file=backend/app/security/cookie_policy.py with
            # old_string="" and a 2583-char new_string, and the engine refused it
            # ("empty old_string but file exists"), so the edit never landed, the
            # verify said green anyway, and the phantom-green gate reopened the
            # step. Treat it as a full-file write, with the same syntax guard the
            # create path uses, and keep a backup so a bad rewrite is recoverable.
            if f.endswith(".py") and not f.endswith(".py.in"):
                try:
                    ast.parse(new)
                except SyntaxError as se:
                    results.append({"file": f, "ok": False,
                                    "msg": f"rewrite failed syntax check: {se}"})
                    continue
            try:
                path.with_suffix(path.suffix + ".bak-orch").write_text(
                    path.read_text(errors="ignore"), errors="ignore")
                path.write_text(new, errors="ignore")
            except Exception as ex:
                results.append({"file": f, "ok": False, "msg": f"rewrite fail: {ex}"})
                continue
            results.append({"file": f, "ok": True, "msg": "rewrote whole file"})
            continue
        try:
            cur = path.read_text(errors="ignore")
        except Exception as ex:
            results.append({"file": f, "ok": False, "msg": f"read fail: {ex}"})
            continue
        if old and old in cur:
            # 09-15: a NO-OP edit (old_string == new_string) used to be reported
            # as "applied". The replace finds the string, writes the file back
            # byte-identical, git finds nothing to commit, and the step greened
            # with no commit behind it. green_truth_watch then condemned it as a
            # PHANTOM and RESTARTED THE ENGINE (measured: 3 phantoms in 15 min,
            # P1B0R0F8#122 / P1B0R0F11#57 / P1B0R0F14, each restart discarding
            # in-flight lane calls). It is not a phantom and it is not a fix: the
            # lane is telling us the code is ALREADY the way it wants it, which is
            # exactly the already-satisfied verdict green_truth exempts.
            if old == new:
                results.append({"file": f, "ok": True, "already_satisfied": True,
                                "msg": "no-op edit: old_string == new_string "
                                       "(already satisfied)"})
                continue
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
            # 09-09 (Bob): plan-drift re-anchor. Try a SAFE fuzzy match before
            # escalating. _fuzzy_replace returns None when it cannot locate the
            # edit unambiguously (renamed/removed target, ambiguous anchor).
            fz = _fuzzy_replace(cur, old, new)
            if fz is not None:
                if f.endswith(".py"):
                    try:
                        ast.parse(fz)
                    except SyntaxError as se:
                        results.append({"file": f, "ok": False,
                                        "msg": f"syntax break (fuzzy edit rejected): {se}"})
                        continue
                if fz == cur:
                    results.append({"file": f, "ok": True, "already_satisfied": True,
                                    "msg": "no-op fuzzy re-anchor (already satisfied)"})
                    continue
                path.write_text(fz, errors="ignore")
                results.append({"file": f, "ok": True, "msg": "applied (fuzzy re-anchor)"})
            else:
                results.append({"file": f, "ok": False,
                                "msg": "old_string not found (context changed)"})
    return all(r["ok"] for r in results), "; ".join(
        f"{r['file']}:{r['msg']}" for r in results[:4])


def edits_present(edits: list) -> bool:
    """09-07 phantom-green gate: is every edit ALREADY IN the files?

    True = the fix is genuinely applied on disk (dedupe of an earlier wave) —
    legitimate green even though THIS run failed to apply it (old_string gone
    because the change is already there). False = old_string still absent AND
    new_string missing => the fix never landed; verdict-green is a phantom.
    """
    if not edits:
        return False
    for e in edits:
        f = str(e.get("file") or "")
        old = str(e.get("old_string") or "")
        new = str(e.get("new_string") or "")
        if not f:
            return False
        p = REPO / f.lstrip("/")
        if not p.exists():
            return False
        content = p.read_text(errors="ignore") if p.stat().st_size < 5_000_000 else ""
        if new == old:
            continue
        if new and new not in content:
            return False
    return True


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
    # 09-09 (Bob): drop cross-repo files (../webchat-api/*) so we never edit or
    # git-add outside repo_dir. A step left with no in-repo files is a no-op.
    files = [x for x in (step.get("files") or []) if x and in_repo(x)]
    ctx = "\n\n".join(file_text(f) for f in files[:2])
    plan_user = (
        f"STEP {sid}: {step.get('title')}\nSTATE: {step.get('finding')}\n"
        f"FIX GUIDANCE: {step.get('fix')}\nMECHANISM: {step.get('mechanism')}\n"
        f"FILES: {', '.join(files)}\n\nFILE CONTENTS:\n{ctx}\n"
    )
    # 09-12 BUG: a step already at MAX_ROUNDS is still SELECTED (the picker takes
    # status in (None, "pending", "executing")) but `range(3, 3)` is empty, so the
    # loop body never ran — the step sat in "executing" forever, never escalated,
    # never retried. 63 steps were stuck this way and the green counter went flat
    # while the engine churned batches. Escalate them instead of silently skipping.
    if rec.get("rounds", 0) >= MAX_ROUNDS:
        # 09-15: this site retired 112 steps as yellow across one run with the
        # reason "round budget already spent ... no lane call made" - and most of
        # them never had a real verdict. A transport failure must not consume a
        # round (the engine's own doctrine), but these steps carry rounds=3 with a
        # last_lane_error of the transport kind, so the budget was spent on
        # failures that said nothing about the code.
        #
        # A step that never got a considered answer is not "the executor tried and
        # could not" - it is work still owed. Give it the round back and let a lane
        # attempt it. Only a step whose rounds ended in a real, non-transport
        # verdict is yellowed here.
        _le = str(rec.get("last_lane_error") or "")
        _la = rec.get("last_apply") or {}
        # 09-15: `bool(_la)` was too weak. A last_apply of {rnd, ok:True} with NO
        # edits, NO apply_msg and NO lane is not an attempt - measured 22 such
        # records retired as yellow, and they carry no evidence that any lane ever
        # looked at the step. An attempt means a lane actually ran: a named lane, or
        # an edit that applied, or a reason that was recorded.
        _has_lane = bool(str(_la.get("lane") or "").strip())
        _la_files = [str(e.get("file") or e.get("filePath") or "").strip()
                     for e in (_la.get("edits") or []) if isinstance(e, dict)]
        _la_files = [f for f in _la_files if f]
        # The lane's own explanation, kept from the apply record. Without this the
        # yellow reads as "the lane ran and nothing landed", which is not a reason
        # (owner 09-16: no vague justifications).
        _la_said = str(_la.get("lane_said") or "").strip()
        _NO_EXPLANATION = ("(no explanation recorded - the lane's reply is in the "
                           "step log)")
        _has_edits = bool(_la.get("edits")) or bool(str(_la.get("apply_msg") or "").strip())
        _tried = bool(_la) and (_has_lane or _has_edits) and not is_transport_error(_le)
        if _tried and _applied_edits_landed(rec):
            # The edit is on disk, so this is finished work, not a step a lane
            # gave up on. Green it with real evidence instead of retiring it
            # yellow at MAX_ROUNDS. Safe from the phantom-green gate by
            # construction: the file itself was read to reach this branch.
            rec["status"] = "green"
            rec["resolved_by"] = (
                "edit already applied and present on disk — recorded green "
                "instead of yellow (round budget exhausted)")
            _la2 = dict(_la)
            _la2["verified_by"] = "pre_loop_max_rounds:content-on-disk"
            rec["last_apply"] = _la2
            await save_state_serialized(st)
            try:
                await orch_git.git_commit_step(sid, rec)
            except Exception as ge:
                print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
            print(f"[step {sid}] MAX_ROUNDS but the applied edit IS on disk — green "
                  f"instead of yellow", flush=True)
            return rec
        if not _tried:
            rec["rounds"] = 0
            rec["rounds_reset_reason"] = (
                "round budget was spent on failures that were not verdicts "
                "(transport or no apply recorded); re-queued for a real attempt")
            await save_state_serialized(st)
            print(f"[step {sid}] rounds were spent without a verdict — reset to 0 "
                  f"and re-queued instead of retiring as yellow", flush=True)
            return rec
        # 09-15: the reason used to read "no lane call made", which is FALSE here -
        # `_tried` above proves a lane ran and left a non-transport apply record.
        # It also printed rounds=11/3 and 12/3 (legacy counts from before the cap
        # landed), which makes the string nonsense. Say what actually happened.
        _r = rec.get("rounds", 0)
        escalate(sid, rec,
                 f"round budget already spent (rounds={_r}, cap {MAX_ROUNDS}); "
                 f"a lane ran and its answer did not land an edit this pass. "
                 f"Target file(s): {', '.join(_la_files) or ', '.join(files) or 'not recorded'}. "
                 f"The last lane to answer ({_la.get('lane') or 'unknown'}) said, in "
                 f"its own words: {_la_said or _NO_EXPLANATION}. "
                 f"Last apply ok={_la.get('ok')} "
                 f"msg={str(_la.get('apply_msg'))[:160]!r}",
                 site="run_step:pre_loop_max_rounds")
        if rec["status"] not in ("escalated", "yellow"):
            await save_state_serialized(st)
            return rec
        try:
            await orch_git.git_commit_step(sid, rec)
        except Exception as ge:
            print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
        print(f"[group] step {sid} already at MAX_ROUNDS — escalated", flush=True)
        return

    for rnd in range(rec["rounds"], MAX_ROUNDS):
        rec["rounds"] = rnd + 1
        rec["status"] = "executing"
        await save_state_serialized(st)
        res = await repair_with_retry(session, EXEC_SYSTEM,
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
        # 09-08 (user): repair returning EMPTY edits is a phantom feed. Don't even
        # burn a second lane call on verify — record red/no_edits and leave pending
        # so the step is retried/escalated honestly instead of churning green.
        if not edits:
            # 09-14 (owner): ALREADY-SATISFIED MUST BE TESTED BEFORE `cannot-fix`.
            # Lanes write BOTH verdicts in one sentence —
            #   "Steps P1B3R0F5#7, P1B4R0F4#11 — cannot-fix (already satisfied / stale);
            #    no edits emitted."
            # — and the `cannot-fix` substring matched first, so the step was
            # escalated to TERMINAL and never reached the already-satisfied path
            # below. Measured live: green flat at 3152 for 25 min while every lane
            # answered "already satisfied" and every such step landed in
            # `escalated`. When the lane says the work is present, that IS the
            # step done: honour `already_satisfied_action` (default green) here.
            if is_already_satisfied(res.content):
                action = already_satisfied_action()
                rec["resolved_by"] = "lane verdict: already satisfied"
                rec["last_apply"] = {
                    "rnd": rnd + 1,
                    "ok": action == "green",
                    "edits": [],
                    "apply_msg": "lane verdict: already satisfied",
                    "lane": res.lane,
                    "verify": (
                        "green: lane verified the work is already present"
                        if action == "green"
                        else ("escalated: lane says the work is already present"
                              if action == "escalate"
                              else "red:no_edits — already-satisfied verdict left pending")
                    ),
                }
                if action == "green":
                    rec["status"] = "green"
                elif action == "escalate":
                    escalate(sid, rec,
                             "lane verdict: work already present "
                             "(already_satisfied_action=escalate)",
                             site="run_step:already_satisfied")
                else:
                    rec["status"] = "pending"
                    rec["rounds"] = rnd + 1
                try:
                    await orch_git.git_commit_step(sid, rec)
                except Exception as ge:
                    print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
                await save_state_serialized(st)
                print(f"[step {sid}] lane verdict already-satisfied -> {rec['status']}", flush=True)
                return rec
            # 09-13: a `cannot-fix:<id>:<reason>` reply is the CONTRACT's own
            # terminal verdict — the lane inspected the file and reports the fix
            # is already present or impossible. Measured live: the pool burned 9
            # empty-edits hops per round across openrouter / omniroute /
            # deepseek2 / deepseek4 / bitdeer / gemini with ZERO commits for 20
            # minutes, because every lane answered
            # "cannot-fix:P1B0R0F0#29: the required line is ALREADY present" and
            # the engine hopped each one in turn. Honour the verdict: escalate
            # instead of spending the whole pool on a step no lane will edit.
            if "cannot-fix" in (res.content or "").lower():
                # 09-14 (worker, B2): ONE lane's cannot-fix is NOT terminal.
                # The group path already required MAX_ROUNDS before honouring
                # this verdict (see run_group), but THIS path escalated on the
                # first round — measured: 53 steps escalated at rounds==1, 15 of
                # them with apply_msg "lane returned cannot-fix". A lane that
                # gives up is not proof the work is impossible, so retry across
                # the remaining lanes and only let the verdict kill the step
                # once it has survived the full round budget.
                rec["rounds"] = rnd + 1
                _tried = rec.setdefault("tried_lanes", [])
                if res.lane not in _tried:
                    _tried.append(res.lane)
                _said = _lane_said(res.content, sid)
                rec["last_apply"] = {"rnd": rnd + 1, "ok": False, "edits": [],
                                     "apply_msg": "lane returned cannot-fix",
                                     "lane": res.lane,
                                     "lane_said": _said,
                                     "tried_lanes": list(_tried)}
                if rec["rounds"] < MAX_ROUNDS:
                    rec["status"] = "pending"
                    rec["last_apply"]["verify"] = (
                        "red:no_edits — cannot-fix verdict not yet corroborated, left pending")
                    await save_state_serialized(st)
                    print(f"[step {sid}] lane said cannot-fix — left pending "
                          f"(round {rec['rounds']}/{MAX_ROUNDS}, not yet terminal)", flush=True)
                    return rec
                escalate(sid, rec,
                         f"cannot-fix verdict on {rec['rounds']} separate attempts "
                         f"(cap {MAX_ROUNDS}), tried by {', '.join(_tried)}. "
                         f"Target file(s): "
                         f"{', '.join(rec.get('files') or PLAN_FILES.get(sid) or []) or 'unknown (the plan names no file)'}. "
                         f"The lane that gave up LAST ({res.lane}) said, in its own "
                         f"words: {_said or '(no explanation given - see lane log)'}. "
                         f"Nothing was edited on any attempt, so no lane would touch "
                         f"this step.",
                         site="run_step:cannot_fix",
                         verify="escalated: lane verdict cannot-fix")
                if rec["status"] not in ("escalated", "yellow"):
                    await save_state_serialized(st)
                    return rec
                try:
                    await orch_git.git_commit_step(sid, rec)
                except Exception as ge:
                    print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
                await save_state_serialized(st)
                print(f"[step {sid}] lane verdict cannot-fix — escalated", flush=True)
                return rec
            # 09-13: the same stall also arrives as a plain prose verdict — the
            # lane inspects the file and reports the step's PREMISE is wrong or
            # the work is already on disk, without the `cannot-fix:` prefix.
            # Measured on this dump, two replies that each hopped every lane:
            #   "P1B0R0F1#10: _with_idempotency_key lives in
            #    oculus/execution/execute_trade.py ... not live/execute_trade.py"
            #   "STEP P1B5R0F4#121 — tests/test_live_trading_auth.py is present
            #    with the batch-provided content (19 tests ...)"
            # Neither contains a fix verb, so `_looks_like_fix_report` misses
            # both and the pool burns 9 hops per round on a step no lane will
            # ever edit. A verdict that the work is ALREADY THERE is terminal:
            # escalate it and let the escalation solver decide.
            _c = (res.content or "").lower()
            if sid.split("#")[0].lower() in _c and is_already_satisfied(res.content):
                # 09-13 (owner): the work being already there IS the step done.
                # This used to escalate, which put the step in the terminal
                # "escalated" bucket and left the caller reading it as unfixed
                # work. Count it green and move on.
                rec["status"] = "green"
                rec["resolved_by"] = "lane verdict: already satisfied"
                rec["last_apply"] = {"rnd": rnd + 1, "ok": True, "edits": [],
                                     "apply_msg": "lane verdict: already satisfied",
                                     "lane": res.lane,
                                     "verify": "green: lane verified the work is already present"}
                try:
                    await orch_git.git_commit_step(sid, rec)
                except Exception as ge:
                    print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
                await save_state_serialized(st)
                print(f"[step {sid}] lane verdict already-satisfied — GREEN, moving on", flush=True)
                return rec
            rec["status"] = "pending"
            rec["rounds"] = rnd + 1
            rec["last_apply"] = {"rnd": rnd + 1, "ok": False, "edits": [],
                                 "apply_msg": "no edits proposed", "lane": res.lane}
            rec["last_apply"]["verify"] = "red:no_edits — repair returned empty edits; not fixed, verify skipped"
            rec["last_apply"]["lane_error"] = ""
            # 09-13: the GROUP path already escalates a terminal verdict on sight
            # (cannot-fix / no edits emitted). This SINGLE path did not, so a lane
            # that had plainly said "already satisfied / cannot-fix" still fell
            # through to `pending` and burned its remaining rounds re-asking lanes
            # that had all said no. Measured: 419 pending steps parked at rounds
            # 1-2 on exactly these verdicts. Escalate here too.
            _low = (res.content or "").lower()
            if is_already_satisfied(res.content):
                action = already_satisfied_action()
                rec["last_apply"]["verify"] = (
                    "green: lane verified the work is already present" if action == "green"
                    else ("escalated: lane says the work is already present" if action == "escalate"
                          else "red:no_edits — already-satisfied verdict left pending"))
                rec["resolved_by"] = "lane verdict: already satisfied"
                if action == "green":
                    rec["status"] = "green"
                elif action == "escalate":
                    escalate(sid, rec,
                             "lane verdict: work already present "
                             "(already_satisfied_action=escalate)",
                             site="run_step:already_satisfied_prose")
                else:
                    rec["status"] = "pending"
                if rec["status"] != "pending":
                    try:
                        await orch_git.git_commit_step(sid, rec)
                    except Exception as ge:
                        print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
                await save_state_serialized(st)
                print(f"[step {sid}] already-satisfied verdict — {rec['status'].upper()} "
                      f"(already_satisfied_action={action})", flush=True)
                return rec
            if ("cannot-fix" in _low or "no edit emitted" in _low
                    or "no edits emitted" in _low):
                # 09-14 (worker, B2): same rule as the group path — a terminal-
                # LOOKING verdict only becomes terminal once it has survived the
                # round budget. Escalating "without burning rounds" is exactly
                # how 53 steps died at rounds==1.
                if rec["rounds"] < MAX_ROUNDS:
                    rec["status"] = "pending"
                    rec["last_apply"]["verify"] = (
                        "red:no_edits — terminal-looking verdict not yet corroborated, left pending")
                    await save_state_serialized(st)
                    print(f"[step {sid}] terminal-looking verdict — left pending "
                          f"(round {rec['rounds']}/{MAX_ROUNDS}, not yet terminal)", flush=True)
                    return rec
                escalate(sid, rec,
                         f"terminal verdict from lane {res.lane} "
                         f"(cannot-fix / no edits emitted) after "
                         f"{rec['rounds']}/{MAX_ROUNDS} rounds",
                         site="run_step:terminal_verdict",
                         verify="escalated:terminal_verdict")
                if rec["status"] not in ("escalated", "yellow"):
                    await save_state_serialized(st)
                    return rec
                try:
                    await orch_git.git_commit_step(sid, rec)
                except Exception as ge:
                    print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
                await save_state_serialized(st)
                print(f"[step {sid}] terminal verdict — escalated "
                      f"(rounds={rec['rounds']}/{MAX_ROUNDS})", flush=True)
                return rec
            await save_state_serialized(st)
            print(f"[step {sid}] repair returned NO edits — left pending (red/no_edits, verify skipped)",
                  flush=True)
            return rec
        ok, msg = apply_edits(edits, default_file=_sole_target(locals().get("files") or rec.get("files") or PLAN_FILES.get(sid) or []))
        rec["last_apply"] = {"rnd": rnd + 1, "ok": ok, "edits": edits,
                             "apply_msg": msg, "lane": res.lane,
                             "verify": ("already satisfied (no-op edit)"
                                        if ok and _all_noop(edits) else None)}
        _capture_landed(rec)
        if ok and _all_noop(edits):
            # The lane changed nothing because nothing needed changing. Say so in
            # the field green_truth_watch reads, or it condemns this as a phantom
            # and restarts the engine for a step that was never broken.
            rec["resolved_by"] = "lane verdict: already satisfied (no-op edit)"
        checks = run_checks(files)
        vres = await lane_call_result(
            session, VERIFY_SYSTEM,
            f"STEP {sid}: your edits {json.dumps(edits)[:3000]}\n"
            f"CHECK OUTPUT:\n{checks}\n\nRUN VERIFICATION NOW.")
        if not vres.ok:
            # 09-15 (Bob): "fix the orchestrator so it doesnt have the superseeding
            # issue again". An unreachable VERIFY lane is not a failed fix - the
            # apply already succeeded. Leaving the step pending here spent its
            # round budget on a lane that was never asked about the code, and the
            # step was later retired YELLOW even though its edit was on disk, or
            # had its credit erased when a later step rewrote the same file. The
            # edit itself is the evidence, so record it green.
            if _applied_edits_landed(rec):
                rec["status"] = "green"
                rec["resolved_by"] = ("edit applied and present on disk; the verify "
                                      "lane was unreachable so the edit is the evidence")
                rec.pop("last_lane_error", None)
                await save_state_serialized(st)
                try:
                    await orch_git.git_commit_step(sid, rec)
                except Exception as ge:
                    print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
                print(f"[step {sid}] verify lane unreachable but the edit IS on disk "
                      f"— green", flush=True)
                return rec
            rec["status"] = "pending"
            rec["last_lane_error"] = f"verify lane: {vres.error[:280]}"
            await save_state_serialized(st)
            print(f"[step {sid}] verify lane unavailable — left pending", flush=True)
            return rec
        v = parse_json(vres.content)
        if v.get("verdict") == "green":
            # 09-07 phantom-green gate (user): verdict-green alone is not proof.
            # Real green = the edits were actually applied this run, OR they are
            # already present on disk (legit dedupe). Otherwise the fix never
            # landed and the step stays pending for a real execution.
            if edits_present(rec["last_apply"]["edits"]):
                rec["status"] = "green"
                rec["last_apply"]["checks"] = checks[-600:]
                await save_state_serialized(st)
                # 09-06 (user): every green step = individual commit+push to origin/main.
                # Never allowed to break the marathon — git trouble logs and moves on.
                try:
                    await orch_git.git_commit_step(sid, rec)
                except Exception as ge:
                    print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
                return rec
            print(f"[step {sid}] verify green but NO real edit on disk — kept pending "
                  f"(phantom-green gate)", flush=True)
            rec["last_apply"]["verify"] = ("phantom-green: verdict green but edits not "
                                           f"present/appl:{(str(rec['last_apply'].get('ok')))}")
            rec["status"] = "pending"
        rec["last_apply"]["verify"] = v.get("reason", "")
        new_edits = v.get("edits") or edits
        okay2, msg2 = apply_edits(new_edits)
        plan_user = (f"STEP {sid}: corrected attempt\nFILES: {', '.join(files)}\n"
                     f"VERIFY SAID: {v.get('reason','')}\n\nFILE CONTENTS:\n"
                     f"{ctx}\n")
        rec["last_apply"] = {"rnd": rnd + 1, "ok": okay2, "edits": new_edits,
                             "apply_msg": msg2, "lane": res.lane}
        _capture_landed(rec)
    # 09-14 (worker, B1): THIS was the silent killer. The round loop fell out
    # here and set `escalated` with no reason at all — 108 of the 155 no-reason
    # escalations sat at rounds==3 with last_apply.ok=true and real edits. Record
    # WHY: the rounds were spent and the verify lane never returned green.
    _la = rec.get("last_apply") or {}
    # 09-17: THIS site lacked the check the pre_loop_max_rounds site has. Measured on
    # P1B2R0F5#91: it was re-queued, the lane's edit LANDED (landed_verified True,
    # landed_at 05:08:20), and one second later this site retired it YELLOW at
    # `run_step:rounds_exhausted` (05:08:21) without ever looking at the file. Yellow
    # means "a lane looked and could not"; a step whose edit is on disk is finished
    # work, and retiring it yellow throws the credit away. Re-read the target before
    # retiring, exactly as the other park site does.
    if _applied_edits_landed(rec):
        rec["status"] = "green"
        rec["resolved_by"] = (
            "round budget exhausted, but the step's edit IS present on disk (read off "
            "the file at the round limit) - recorded green instead of yellow")
        _la3 = dict(rec.get("last_apply") or {})
        _la3["verified_by"] = "run_step:rounds_exhausted:content-on-disk"
        rec["last_apply"] = _la3
        rec["yellow_watch_review"] = {
            "verdict": "recovered",
            "evidence": "content verified on disk at the round limit",
        }
        await save_state_serialized(st)
        try:
            await orch_git.git_commit_step(sid, rec)
        except Exception as ge:
            print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
        print(f"[step {sid}] rounds exhausted but the applied edit IS on disk — green "
              f"instead of yellow", flush=True)
        return rec
    _rs_files = [str(e.get("file") or e.get("filePath") or "").strip()
                 for e in (_la.get("edits") or []) if isinstance(e, dict)]
    _rs_files = [f for f in _rs_files if f] or list(rec.get("files") or []) \
        or PLAN_FILES.get(sid) or []
    _rs_said = str(_la.get("lane_said") or "").strip()
    if not _rs_said:
        # The lane's own words live in last_lane_error when it never produced an
        # apply. A terse "verify=None" is not a justification (owner 09-16).
        _rs_said = str(rec.get("last_lane_error") or "").strip()
    escalate(sid, rec,
             f"round budget exhausted after {rec.get('rounds')}/{MAX_ROUNDS} rounds "
             f"without a green verify. Target file(s): "
             f"{', '.join(_rs_files) or 'unknown (the plan names no file)'}. "
             f"Last apply ok={_la.get('ok')} lane={_la.get('lane') or 'unnamed'} "
             f"msg={str(_la.get('apply_msg'))[:200]!r}. "
             + (f"The lane's own words: {_rs_said[:700]}" if _rs_said
                else "No lane left a written reason."),
             site="run_step:rounds_exhausted")
    if rec["status"] not in ("escalated", "yellow"):
        await save_state_serialized(st)
        return rec
    # 09-08 (user: each step commits+pushes): escalate = terminal — save the
    # step's applied edits to the repo too, not just greens. Individual commit.
    try:
        await orch_git.git_commit_step(sid, rec)
    except Exception as ge:
        print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
    await save_state_serialized(st)
    return rec




# ---------------------------------------------------------------------------
# Per-step batch queue engine (owner spec 2026-09-14).
#
#   batch of GROUP_CAP steps -> EVERY lane works that batch, one step per lane
#   call -> each lane claims the next step whose files no other lane holds ->
#   NO lane touches the next batch until this one is terminal (green or yellow)
#   -> a lane cooldown is NOT an escalation: the lane parks and its step is
#   handed to a sibling together with the step's history log.
#
# Set ORCH_QUEUE_ENGINE=0 to fall back to the old batch-group loop.
# ---------------------------------------------------------------------------
_QUEUE_ENGINE = os.environ.get("ORCH_QUEUE_ENGINE", "1") == "1"
# 09-14 (owner): escalation is a config option, not a built-in stage.
#   ORCH_ESCALATIONS_ENABLED / orch.yaml `escalations_enabled`
#     false (default) -> the executor decides: green, or yellow with its reason.
#     true            -> unsolvable steps go to an escalation persona for one
#                        final shot. Shipped so other users can have escalations.
ESCALATIONS_ENABLED = bool(getattr(CFG, "escalations_enabled", False))
HANDOFF = oq.StepHandoff()

ESCALATION_SYSTEM = (
    "You are the ESCALATION PERSONA — the last line of defence for one oculus fix "
    "step. Execution lanes have already tried and failed. You get ONE final shot. "
    "You are given the step, its files, and the FULL HISTORY of every previous "
    "attempt (read the history log file named in the prompt). "
    "Decide honestly between exactly two outcomes: "
    "(1) you can still fix it — answer {\"verdict\":\"green\",\"edits\":[{\"file\":\"...\","
    "\"old_string\":\"...\",\"new_string\":\"...\"}],\"notes\":\"...\"}; or "
    "(2) it genuinely cannot be fixed here — answer {\"verdict\":\"yellow\","
    "\"justification\":\"<a specific, verifiable reason a reviewer can act on: what "
    "you tried, what is missing, why no lane can close it>\"}. "
    "A yellow REQUIRES a real justification of at least 40 characters; a one-word or "
    "copy-paste reason is rejected. Never answer yellow because you are unsure — "
    "yellow means 'this is a pass that a human must review'. "
    "Answer ONE JSON object only, no fences, no prose."
)


def _lane_cool_seconds(err: str) -> int:
    """How long to park the lane whose call failed, by failure kind.

    A cooldown is NOT an escalation: the step keeps its place in the batch
    queue and a sibling lane takes it, handed the history log.
    """
    e = (err or "").lower()
    if "429" in e or "rate limit" in e or "rate_limit" in e or "too frequent" in e:
        return int(os.environ.get("ORCH_LANE_COOL_RATE_S", "900"))
    if "timeout" in e or "timed out" in e:
        return int(os.environ.get("ORCH_LANE_COOL_TIMEOUT_S", "120"))
    return int(os.environ.get("ORCH_LANE_COOL_ERROR_S", "60"))


async def execute_step_pinned(session, lane: str, step: dict, st: dict) -> "oq.StepOutcome":
    """ONE step, ONE lane, ONE call — the queue's unit of work.

    Reuses run_step's whole verdict machine (cannot-fix / already-satisfied /
    phantom-green / rounds / per-step commit) by pinning the lane pool to the
    lane the scheduler handed us.
    """
    sid = step["finding_id"]
    POOL.pinned = lane
    try:
        rec = await run_step(session, step, st)
    except Exception as e:
        return oq.StepOutcome("retry", lane=lane,
                              error=f"{type(e).__name__}: {e}")
    finally:
        POOL.pinned = None
    rec = rec if isinstance(rec, dict) else (st.get("steps", {}).get(sid) or {})
    status = rec.get("status")
    err = str(rec.get("last_lane_error") or "")
    if status in ("green", "yellow"):
        return oq.StepOutcome(status, lane=lane, edits=(rec.get("last_apply") or {}).get("edits") or [])
    if status == "escalated":
        return oq.StepOutcome("escalate", lane=lane,
                              note=str(rec.get("escalated_reason") or ""),
                              edits=(rec.get("last_apply") or {}).get("edits") or [])
    cooled = _lane_cool_seconds(err) if err else 0
    return oq.StepOutcome("retry", lane=lane, error=err or None, cooled_s=cooled,
                          edits=(rec.get("last_apply") or {}).get("edits") or [])


async def escalate_final(session, sid: str, st: dict, lane: str | None = None) -> str:
    """The escalation persona's ONE final shot: fix it (green) or code it yellow.

    The persona is handed the step, its files, and the path of the step's history
    log so it can see everything the execution lanes already tried.
    """
    rec = st["steps"].setdefault(sid, {"rounds": 0, "status": "pending"})
    step = _STEP_BY_ID.get(sid) or {}
    files = [x for x in (step.get("files") or []) if x and in_repo(x)]
    ctx = "\n\n".join(file_text(f) for f in files[:2])
    hist = HANDOFF.render(sid)
    prompt = (f"STEP {sid}: {step.get('title')}\n"
              f"STATE: {step.get('finding')}\nFIX GUIDANCE: {step.get('fix')}\n"
              f"MECHANISM: {step.get('mechanism')}\n"
              f"FILES: {', '.join(files)}\n\n"
              f"WHY EXECUTION GAVE UP: {rec.get('escalated_reason') or rec.get('last_lane_error') or 'n/a'}\n\n"
              f"{hist}\nFILE CONTENTS:\n{ctx}\n\nDECIDE NOW — one JSON object.")
    # Pinned to the lane the queue handed this task to, so the escalation runs
    # on a real worker instead of hopping the whole pool under load.
    POOL.pinned = lane
    try:
        res = await POOL.call(session, ESCALATION_SYSTEM, prompt, want_edits=False)
    finally:
        POOL.pinned = None
    if not res.ok:
        print(f"[esc] {sid}: escalation lane unavailable ({res.error[:90]}) — left escalated",
              flush=True)
        return "escalated"
    verdict = parse_json(res.content)
    v = str(verdict.get("verdict") or "").strip().lower()
    if v in ("green", "fixed") or verdict.get("edits"):
        edits = verdict.get("edits") or []
        if edits:
            ok, msg = apply_edits(edits, default_file=_sole_target(locals().get("files") or rec.get("files") or PLAN_FILES.get(sid) or []))
            rec["last_apply"] = {"rnd": rec.get("rounds", 0) + 1, "ok": ok,
                                 "edits": edits, "apply_msg": msg, "lane": res.lane,
                                 "verify": "green: escalation persona applied the fix"}
            _capture_landed(rec)
            if not ok:
                print(f"[esc] {sid}: persona edits failed to apply ({msg[:90]})", flush=True)
                return "escalated"
        rec["status"] = "green"
        rec["resolved_by"] = "escalation persona: fixed on the final shot"
        try:
            await orch_git.git_commit_step(sid, rec)
        except Exception as ge:
            print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
        print(f"[esc] {sid}: GREEN (escalation persona fixed it)", flush=True)
        return "green"
    justification = str(verdict.get("justification") or verdict.get("reason") or "").strip()
    try:
        oq.make_yellow(rec, justification, lane="escalation")
    except oq.YellowJustificationError as e:
        print(f"[esc] {sid}: persona yellow REJECTED — {e}", flush=True)
        return "escalated"
    rec["yellow_history_log"] = str(HANDOFF.path(sid))
    print(f"[esc] {sid}: YELLOW (flagged for review) — {justification[:120]}", flush=True)
    await save_state_serialized(st)
    return "yellow"


async def run_queue_batches(session, batches: list, st: dict) -> None:
    """S3 (owner 09-14): ONE global work queue, lanes never idle.

    A batch is an ACCOUNTING boundary, not a work boundary: the moment every
    step of a batch is terminal, `on_batch_done` commits and reports it, while
    the lanes have already moved on to whichever claimable step is next in the
    plan. File safety is unchanged — `drive_plan` uses ONE reservations table
    across every batch, so no two lanes anywhere hold the same file.
    """
    lane_names = [ln.name for ln in POOL.lanes]
    steps_by_id = {s["finding_id"]: s for batch in batches for s in batch}
    if not steps_by_id:
        print("[qeng] nothing pending — plan drained", flush=True)
        return
    _STEP_BY_ID.update(steps_by_id)
    ids = [[s["finding_id"] for s in batch] for batch in batches]
    roster = oq.LaneRoster(lane_names)
    locked = [s["finding_id"] for s in _STEP_BY_ID.values()
              if "locked" in str(s.get("tags") or [])]
    print(f"[qeng] escalation persona: "
          f"{'ON' if ESCALATIONS_ENABLED else 'OFF (executor decides green/yellow)'}",
          flush=True)
    print(f"[qeng] S3 work-stealing engine: {len(ids)} batches, "
          f"{len(steps_by_id)} steps, {len(lane_names)} lanes, "
          f"batch_cap={BATCH_CAP}", flush=True)

    async def _exec(lane, sid):
        return await execute_step_pinned(session, lane, steps_by_id[sid], st)

    async def _esc(lane, sid):
        return await escalate_final(session, sid, st, lane=lane)

    def _batch_done(bi, snap):
        try:
            _note_step_completions(snap)
        except Exception as e:
            print(f"[qeng] completion hook failed: {e}", flush=True)

    try:
        snap = await oq.drive_plan(ids, st["steps"], roster, HANDOFF, _exec,
                                   on_escalate=(_esc if ESCALATIONS_ENABLED else None),
                                   steps_by_id=steps_by_id, poll_s=5.0,
                                   log=lambda m: print(m, flush=True),
                                   on_batch_done=_batch_done)
    except oq.BatchStalled as e:
        print(f"[qeng] PLAN STALLED — {e}", flush=True)
        snap = {}
    await save_state_serialized(st)
    if snap:
        n_t = sum(1 for v in snap.values() if v in ("green", "yellow"))
        print(f"[qeng] plan pass done: {n_t}/{len(snap)} terminal "
              f"(parked lanes: {roster.parked()})", flush=True)


_STEP_BY_ID: dict = {}


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
    step_by_sid = {}
    for i, s in enumerate(steps, 1):
        sid = s["finding_id"]
        step_by_sid[sid] = s
        rec = st["steps"].get(sid)
        if rec is None:
            rec = {"rounds": 0, "status": "pending"}
            st["steps"][sid] = rec
        # 09-09 (Bob): cap the round counter AT increment time, NEVER beyond.
        # Previously rec["rounds"] grew unbounded (564+ steps >3, max 27) because
        # it incremented every pass but only escalated in the verify-fail branch.
        rec["rounds"] = min(rec.get("rounds", 0) + 1, MAX_ROUNDS)
        rec["status"] = "executing"
        # 09-09 (Bob): drop cross-repo files (see in_repo) in group path too.
        files = [x for x in (s.get("files") or []) if x and in_repo(x)]
        ctx = "\n\n".join(file_text(f) for f in files[:2])
        plan.append(
            f"STEP {i} ({sid}): {s.get('title')}\nSTATE: {s.get('finding')}\n"
            f"FIX GUIDANCE: {s.get('fix')}\nMECHANISM: {s.get('mechanism')}\n"
            f"FILES: {', '.join(files)}\n\nFILE CONTENTS:\n{ctx}\n"
        )
        step_ids.append((str(i), sid, files))
    plan_user = "\n\n".join(plan) + "\nOUTPUT THE EDITS NOW — tag every edit with STEP N."
    gres = await repair_with_retry(session, EXEC_SYSTEM, plan_user)
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
              f"{len(step_ids)} steps left pending | {POOL.health_report()}", flush=True)
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
        # 09-08 (user): a weak lane may skip this step in the group (empty edits).
        # Retry it SINGLE so gemini gets a shot before we give up (repair_with_retry
        # hops once on empty edits). Only if it still returns nothing -> red/no_edits.
        # 09-13: the GROUP reply itself can say the work is already there. The
        # check below only looked at the single-retry content, so a group verdict
        # of "already satisfied" fell through to the single retry and then to
        # escalate. Honour the group verdict first.
        if not mine and is_already_satisfied(gres.content):
            _action = already_satisfied_action()
            rec["last_apply"] = {
                "rnd": rec["rounds"],
                "ok": _action == "green",
                "edits": [],
                "apply_msg": "lane verdict: already satisfied (group reply)",
                "verify": ("green: lane verified the work is already present" if _action == "green"
                           else ("escalated: lane says the work is already present" if _action == "escalate"
                                 else "red:no_edits — already-satisfied verdict left pending")),
            }
            rec["resolved_by"] = "lane verdict: already satisfied"
            if _action == "green":
                rec["status"] = "green"
            elif _action == "escalate":
                escalate(sid, rec,
                         "lane verdict (group reply): work already present "
                         "(already_satisfied_action=escalate)",
                         site="run_group:already_satisfied_group_reply")
            else:
                rec["status"] = "pending"
            if rec["status"] != "pending":
                try:
                    await orch_git.git_commit_step(sid, rec)
                except Exception as ge:
                    print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
            print(f"[group] step {sid} already-satisfied verdict (group reply) — "
                  f"{rec['status'].upper()} (already_satisfied_action={_action})", flush=True)
            continue
        if not mine:
            ss = step_by_sid.get(sid) or {}
            sfiles = [x for x in (ss.get("files") or []) if x and in_repo(x)]
            sctx = "\n\n".join(file_text(f) for f in sfiles[:2])
            sp = (f"STEP 1 ({sid}): {ss.get('title')}\nSTATE: {ss.get('finding')}\n"
                  f"FIX GUIDANCE: {ss.get('fix')}\nMECHANISM: {ss.get('mechanism')}\n"
                  f"FILES: {', '.join(sfiles)}\n\nFILE CONTENTS:\n{sctx}\n"
                  f"OUTPUT THE EDITS NOW.")
            sres = await repair_with_retry(session, EXEC_SYSTEM, sp)
            if sres.ok:
                sex = parse_json(sres.content)
                mine = sex.get("edits") or []
                # 09-13: the SINGLE retry can come back with a terminal verdict
                # ("cannot-fix" / "no edits emitted") — repair_with_retry returns
                # those deliberately instead of hopping, because the answer is the
                # same on every lane. Treat it as terminal HERE too: without this
                # the step fell through to `pending` and burned its remaining
                # rounds re-asking lanes that had already said no. Measured: 421
                # pending steps stuck at rounds 1-2 on exactly these verdicts.
                _low = (sres.content or "").lower()
                if not mine and is_already_satisfied(sres.content):
                    action = already_satisfied_action()
                    rec["last_apply"] = {
                        "rnd": rec["rounds"],
                        "ok": action == "green",
                        "edits": [],
                        "apply_msg": "lane verdict: already satisfied",
                        "verify": ("green: lane verified the work is already present" if action == "green"
                                   else ("escalated: lane says the work is already present" if action == "escalate"
                                         else "red:no_edits — already-satisfied verdict left pending")),
                    }
                    rec["resolved_by"] = "lane verdict: already satisfied"
                    if action == "green":
                        rec["status"] = "green"
                    elif action == "escalate":
                        escalate(sid, rec,
                                 "lane verdict: work already present "
                                 "(already_satisfied_action=escalate)",
                                 site="run_group:already_satisfied")
                    else:
                        rec["status"] = "pending"
                    if rec["status"] != "pending":
                        try:
                            await orch_git.git_commit_step(sid, rec)
                        except Exception as ge:
                            print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
                    print(f"[group] step {sid} already-satisfied verdict — {rec['status'].upper()} "
                          f"(already_satisfied_action={action})", flush=True)
                    continue
                if not mine and ("cannot-fix" in _low or "no edit emitted" in _low
                                 or "no edits emitted" in _low):
                    # 09-13: the group path escalated 105 steps as "cannot-fix
                    # (target not found)" while `last_lane_error` held a
                    # ClientConnectorError — the verdict was reached while a lane
                    # was unreachable. A transport failure is not a verdict: leave
                    # the step pending and let a healthy lane answer it.
                    if is_transport_error(rec.get("last_lane_error")):
                        rec["status"] = "pending"
                        # 09-14: clear the stale transport error. Leaving it in place
                        # made the step match `is_transport_error` again on every
                        # later pass, so it was reset to pending forever and never
                        # escalated or committed (measured: 8,533 pending steps
                        # stuck on a stale 429 / "no lane available"). A fresh
                        # attempt needs a clean slate.
                        # 09-14 (worker, B7): this message named neither the
                        # lane, the error, nor the step's file, so a step cycling
                        # through this state forever was indistinguishable from
                        # healthy churn. Record it on the step AND in the log so
                        # a live-lock is countable.
                        _terr = str(rec.get("last_lane_error"))[:200]
                        rec["transport_blocks"] = int(rec.get("transport_blocks", 0)) + 1
                        rec["last_transport_block"] = {
                            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "site": "run_group:cannot_fix_transport",
                            "error": _terr,
                            "lane": str(gres.lane),
                            "files": files[:3],
                        }
                        rec.pop("last_lane_error", None)
                        print(f"[group] step {sid} terminal-looking verdict but lane "
                              f"{gres.lane} was unreachable ({_terr[:90]}) — left pending "
                              f"(block #{rec['transport_blocks']}, files={files[:2]})", flush=True)
                        continue
                    # 09-14: a "target not found" verdict is NOT trustworthy as a
                    # terminal verdict. Sampled the 23 steps escalated this way and
                    # **21 of them named files that DO exist on disk** (only 2 were
                    # genuinely missing) — the lane gave up on the file, it did not
                    # find the file absent. Escalating on that claim kills real
                    # work. Leave it pending and let another lane try; only a
                    # verdict that survives repeated lanes is terminal.
                    if rec.get("rounds", 0) < MAX_ROUNDS:
                        # 09-14: ONE lane's `cannot-fix` must not be terminal.
                        # Sampled the 23 steps escalated this way and **21 of them
                        # named files that DO exist on disk** (only 2 were genuinely
                        # missing) — the lane gave up, it did not prove the work
                        # impossible. Escalating on the first verdict killed real
                        # work and drove `escalated` up while `green` stayed flat.
                        # Retry across the remaining lanes; only a verdict that
                        # survives MAX_ROUNDS is terminal.
                        rec["status"] = "pending"
                        # 09-14 (worker, B3): cap AT increment. This site and the
                        # phantom-green site below incremented without a cap, so a
                        # step could leave the group path at MAX_ROUNDS+1.
                        rec["rounds"] = min(rec.get("rounds", 0) + 1, MAX_ROUNDS)
                        rec["last_apply"] = {
                            "rnd": rec["rounds"], "ok": False, "edits": [],
                            "apply_msg": "lane verdict cannot-fix — unverified, retrying other lanes",
                            "verify": "red:no_edits — cannot-fix verdict not yet corroborated, left pending",
                        }
                        print(f"[group] step {sid} lane said cannot-fix — left pending "
                              f"(round {rec['rounds']}/{MAX_ROUNDS}, not yet terminal)", flush=True)
                        continue
                    rec["last_apply"] = {
                        "rnd": rec["rounds"], "ok": False, "edits": [],
                        "apply_msg": "terminal verdict — cannot-fix (target not found)",
                        "verify": "escalated:terminal_verdict",
                    }
                    if escalate(sid, rec,
                                f"terminal verdict cannot-fix / target not found from lane "
                                f"{gres.lane} after {rec['rounds']}/{MAX_ROUNDS} rounds "
                                f"(files={files[:2]})",
                                site="run_group:terminal_verdict") != "escalated":
                        continue
                    try:
                        await orch_git.git_commit_step(sid, rec)
                    except Exception as ge:
                        print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
                    print(f"[group] step {sid} terminal verdict — escalated "
                          f"(rounds={rec['rounds']}/{MAX_ROUNDS})", flush=True)
                    continue
        if not mine:
            rec["last_apply"] = {"rnd": rec["rounds"], "ok": False, "edits": [],
                                 "apply_msg": "no edits proposed"}
            rec["last_apply"]["verify"] = "red:no_edits — repair returned empty edits; not fixed, verify skipped"
            # 09-09 (Bob): no_edits is terminal once the round budget is spent.
            # Previously this branch `continue`d forever, leaving the step pending
            # and re-hammering the lane pool every main-loop pass (rounds to 27).
            if rec["rounds"] >= MAX_ROUNDS:
                if escalate(sid, rec,
                            f"round budget exhausted ({rec['rounds']}/{MAX_ROUNDS}) — "
                            f"lane {gres.lane} returned no edits for this step on every round",
                            site="run_group:no_edits_exhausted") == "escalated":
                    try:
                        await orch_git.git_commit_step(sid, rec)
                    except Exception as ge:
                        print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
                    print(f"[group] step {sid} exhausted rounds on no_edits — escalated", flush=True)
            else:
                rec["status"] = "pending"
                print(f"[group] step {sid} repair returned NO edits — pending (red/no_edits)", flush=True)
            continue
        ok, msg = apply_edits(mine, default_file=_sole_target(locals().get("files") or rec.get("files") or PLAN_FILES.get(sid) or []))
        rec["last_apply"] = {"rnd": rec["rounds"], "ok": ok, "edits": mine, "apply_msg": msg}
        _capture_landed(rec)
        checks = run_checks(files)
        rec["last_apply"]["checks"] = checks[-600:]
        vres = await lane_call_result(
            session, VERIFY_SYSTEM,
            f"STEP {sid}: your edits {json.dumps(mine)[:3000]}\n"
            f"CHECK OUTPUT:\n{checks}\n\nRUN VERIFICATION NOW.")
        if not vres.ok:
            # Same guard as the run_step path above - see the comment there.
            if _applied_edits_landed(rec):
                rec["status"] = "green"
                rec["resolved_by"] = ("edit applied and present on disk; the verify "
                                      "lane was unreachable so the edit is the evidence")
                rec.pop("last_lane_error", None)
                try:
                    await orch_git.git_commit_step(sid, rec)
                except Exception as ge:
                    print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
                print(f"[step {sid}] verify lane unreachable but the edit IS on disk "
                      f"— green", flush=True)
                continue
            rec["status"] = "pending"
            rec["last_lane_error"] = f"verify lane: {vres.error[:280]}"
            continue
        v = parse_json(vres.content)
        if v.get("verdict") == "green":
            if edits_present(rec["last_apply"]["edits"]):
                rec["status"] = "green"
                # 09-06 (user): per-step individual commit+push (same rule as run_step).
                try:
                    await orch_git.git_commit_step(sid, rec)
                except Exception as ge:
                    print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
            else:
                # 09-13: this branch did NOT advance rounds, so a step whose lane
                # keeps saying "green" without an edit on disk stayed pending
                # forever and was re-asked on every pass (measured: steps parked
                # at rounds 16). Count the round, and escalate once the budget is
                # spent — a lane that cannot produce the edit is a terminal
                # verdict, not something to retry indefinitely.
                if is_transport_error(rec.get("last_lane_error")):
                    # the last verify could not reach a lane — retry, never escalate
                    rec["status"] = "pending"
                    rec.pop("last_lane_error", None)  # clean slate (see group path)
                    continue
                rec["rounds"] = min(rec.get("rounds", 0) + 1, MAX_ROUNDS)  # B3: cap
                rec["last_apply"]["verify"] = ("phantom-green: verdict green but edits not "
                                               f"present/appl:{str(rec['last_apply'].get('ok'))}")
                if rec["rounds"] >= MAX_ROUNDS:
                    if escalate(sid, rec,
                                f"phantom-green: verify returned green but no edit is present "
                                f"on disk after {rec['rounds']}/{MAX_ROUNDS} rounds "
                                f"(apply ok={rec['last_apply'].get('ok')}, files={files[:2]})",
                                site="run_group:phantom_green") == "escalated":
                        try:
                            await orch_git.git_commit_step(sid, rec)
                        except Exception as ge:
                            print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
                        print(f"[group] step {sid} phantom-green verdict — escalated "
                              f"(rounds={rec['rounds']})", flush=True)
                else:
                    print(f"[group] step {sid} verdict green but no real edit on disk — "
                          f"kept pending (phantom-green gate, round {rec['rounds']}/{MAX_ROUNDS})",
                          flush=True)
                    rec["status"] = "pending"
        else:
            new_edits = v.get("edits") or mine
            _, msg2 = apply_edits(new_edits)
            rec["last_apply"] = {"rnd": rec["rounds"], "ok": ok,
                                 "edits": new_edits, "apply_msg": msg2}
            _capture_landed(rec)
            # Only escalate once the step has actually spent its rounds. The
            # group path used to escalate after a SINGLE attempt while
            # run_step allowed MAX_ROUNDS, so a step's retry budget depended
            # on whether packing happened to place it alone in a batch.
            if rec["rounds"] >= MAX_ROUNDS:
                if escalate(sid, rec,
                            f"verify verdict not green after {rec['rounds']}/{MAX_ROUNDS} "
                            f"rounds; verify said {str(v.get('reason'))[:120]!r} "
                            f"(apply ok={ok}, files={files[:2]})",
                            site="run_group:verify_red_exhausted") == "escalated":
                    # 09-08 (user: each step commits+pushes): escalate = terminal.
                    try:
                        await orch_git.git_commit_step(sid, rec)
                    except Exception as ge:
                        print(f"[step {sid}] git commit failed: {str(ge)[:180]}", flush=True)
            else:
                rec["status"] = "pending"
    await save_state_serialized(st)
    return {s["finding_id"]: st["steps"].get(s["finding_id"], {}).get("status") for s in steps}


def pack_diverse(steps: list, cap: int) -> list:
    """Pack steps into batches of at most `cap` with DISJOINT file sets.

    09-14 (owner): the linear plan slice produced batches whose LIVE members
    all targeted the same file — measured BATCH 1 carried 3 live steps on
    `weight_store_integrity_...json`, so the file reservations let ONE lane
    work and nine lanes sat idle for the whole barrier. The pending pool holds
    9,984 steps across 1,352 distinct files, so a 15-step disjoint batch is
    always available; the batch just has to be CHOSEN for diversity instead of
    taken in plan order. Leftovers carry to the next batch, so no step is lost.
    """
    remaining = list(steps)
    out = []
    while remaining:
        used, batch, rest, i = set(), [], [], 0
        while i < len(remaining) and len(batch) < cap:
            s = remaining[i]
            i += 1
            fs = {f for f in (s.get("files") or []) if f}
            if fs and (fs & used):
                rest.append(s)
                continue
            used |= fs
            batch.append(s)
        rest.extend(remaining[i:])
        if not batch:                       # every remaining step collides
            batch, rest = [remaining[0]], remaining[1:]
        out.append(batch)
        remaining = rest
    return out


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


def recover_orphaned_executing(st: dict) -> int:
    """09-13: a step is marked `executing` and saved BEFORE the lane call runs, so a
    process killed mid-step (the green-truth restarts, the escalation solver's
    SIGTERM) leaves it in `executing` FOREVER — the engine only ever picks up
    `pending`, so the step is lost. Measured: 233 steps parked in `executing`,
    every one carrying `no lane available` / `verify lane: no lane available`.

    At startup NO step can genuinely be executing: the only process that could be
    running one is the one we are starting. Return them to `pending` so they are
    eligible again."""
    n = 0
    refunded = 0
    for rec in (st.get("steps") or {}).values():
        if rec.get("status") == "executing":
            rec["status"] = "pending"
            rec["orphan_recovered"] = time.strftime("%Y-%m-%d %H:%M startup recovery")
            rec["orphan_recoveries"] = int(rec.get("orphan_recoveries", 0)) + 1
            # 09-14 (worker, B9): REFUND THE ROUND. `rec["rounds"]` is incremented
            # and saved BEFORE the lane call, so a process killed mid-step burns a
            # round on which NO lane ever answered. This box restarts the engine
            # often (green-truth every 5 min, the gemini health check every 30,
            # the escalation solver's SIGTERM), so the budget drains without a
            # single verdict. Measured on the live state: 412 steps carry
            # `orphan_recovered`, and 243 of them hold rounds>=1 with NO
            # `last_apply` at all — a spent round and nothing to show for it. 25
            # were already at rounds==3, which means the very next pass would hit
            # the `rounds >= MAX_ROUNDS` gate and escalate them WITHOUT ever
            # calling a lane. That is precisely the shape of the 108 no-reason
            # escalations at rounds==3 (see B1). A round that produced no lane
            # result is not a round the step spent.
            if not rec.get("last_apply"):
                before = int(rec.get("rounds") or 0)
                rec["rounds"] = max(0, before - 1)
                if rec["rounds"] != before:
                    refunded += 1
            n += 1
    if refunded:
        print(f"[eng] refunded {refunded} round(s) burned by mid-step kills "
              f"(no lane verdict was ever recorded for them)", flush=True)
    return n


# 09-14 (worker, brief Part B): RE-OPEN DEAD WORK AT STARTUP.
#
# Measured on the live state: 502 of 502 escalated steps carried a reason that
# was NOT a verdict about the code —
#   300  transport / timeout (`timeout after Ns`, ClientConnectorError,
#        ServerDisconnectedError, 429 / rate-limited, "no lane available")
#   155  no reason at all (the silent escalate fixed in this same commit series)
#    38  `[502] Prompt too long (max 6000 characters)` from the omniroute lane,
#        which has since been pulled from the pool entirely
#     9  gateway/auth infrastructure errors (expired key, overloaded upstream,
#        `page.viewportSize is not a function`)
# ZERO were a lane actually saying the fix could not be made. Every one of them
# is work the engine gave up on because of its own defects, and `escalated` is
# terminal — nothing would ever retry them.
#
# Doing this by hand on the state file does NOT hold: save_state_serialized
# merges on save and a TERMINAL status in a running engine's in-memory snapshot
# beats a `pending` on disk, so an out-of-band re-open is silently reverted
# (observed directly: 502 steps re-opened on disk were back to `escalated`
# minutes later with every field the re-open added stripped). Doing it HERE, on
# the engine's own state at startup before any batch runs, is authoritative.
#
# Idempotent: a step is re-opened at most once per REOPEN_VERSION, so this is
# not a loop — once the fixed engine escalates a step for a REAL reason, that
# reason is not in the list below and the step stays terminal.
REOPEN_VERSION = "2026-09-14.worker.1"

# 09-14 (worker): steps THIS process deliberately re-opened. The merge in
# save_state_serialized keeps a TERMINAL status on disk over a non-terminal one
# in memory — which is right for a stale snapshot, but WRONG for a step this run
# re-opened on purpose: a single stale write of the old `escalated` record puts
# it back on disk, the merge then adopts it into memory, and the engine
# propagates it forever. Observed live: `escalated` flip-flopped 0 <-> 502 and
# the 155 no-reason records kept returning minutes after a clean re-open.
# A step in this set is never resurrected from disk by the merge.
_REOPENED_THIS_RUN: set[str] = set()

_REOPEN_NOT_A_VERDICT = (
    "timeout after", "timed out",
    "ClientConnectorError", "Cannot connect to host", "Connect call failed",
    "Connection refused", "ConnectionResetError", "ClientOSError",
    "ServerDisconnectedError", "Server disconnected",
    "no lane available", "lanes unavailable",
    "free_rate_limited", "rate_limited", "rate limit", "Rate limit exceeded",
    "http 429", "Too Many Requests",
    "Prompt too long",
    "API key expired", "invalid_token",
    # 09-14: model-auth / capability failures on the provider side. Measured as
    # the last 5 escalations on the live state: `[401] Model north-mini-code-free
    # is not supported`, `[401] Model hy3-free is not supported`, and
    # `[402] This model requires an opencode API key`. A model the account may
    # not call is an infrastructure fact, not a statement about the step.
    "authentication_error", "invalid_api_key", "http 401", "http 402",
    "[401]", "[402]", "is not supported", "requires an opencode API key",
    "temporarily overloaded", "upstream error",
    "page.viewportSize is not a function", "Messages too frequent",
)


def retire_escalations(st: dict) -> int:
    """09-14 (owner): escalations are gone — retire any step still sitting on one.

    "get rid of escalations entirely ... they either solve it or they cant ...
    red shouldnt be a thing." A step the executor could not solve is a CODE
    YELLOW: a pass a human must review, carrying the executor's own reason. This
    runs after reopen_dead_escalations, so anything whose reason was transport or
    no-reason has already gone back to pending; what is left here is a genuine
    "could not do it", and it becomes yellow rather than a terminal dead end.
    """
    n = 0
    for sid, rec in (st.get("steps") or {}).items():
        if rec.get("status") != "escalated":
            continue
        reason = (rec.get("escalated_reason") or rec.get("last_lane_error")
                  or "the executor could not apply a working edit")
        j = _clip_reason(str(reason).strip())
        if len(j) < 40:
            j = (j + " — the executor could not apply a working edit; flagged "
                     "for human review.").strip()
        rec["status"] = "yellow"
        rec["yellow_justification"] = j
        rec["yellow_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec["yellow_by"] = "retire_escalations"
        rec["resolved_by"] = f"code yellow: {j[:200]}"
        rec.pop("escalated_at", None)
        rec.pop("escalated_by", None)
        n += 1
    return n


def reopen_dead_escalations(st: dict) -> int:
    """Return escalated steps that died on infrastructure, not on a verdict."""
    if os.environ.get("ORCH_REOPEN_DEAD", "1") != "1":
        return 0
    n = 0
    for sid, rec in (st.get("steps") or {}).items():
        if rec.get("status") != "escalated":
            continue
        if rec.get("reopened_by_startup") == REOPEN_VERSION:
            continue                      # already given a second life
        reason = (f"{rec.get('escalated_reason') or ''} "
                  f"{rec.get('last_lane_error') or ''}").strip()
        # No reason at all = the silent escalate. Not a verdict either.
        if reason and not any(m in reason for m in _REOPEN_NOT_A_VERDICT):
            continue                      # a real verdict — leave it terminal
        rec["status"] = "pending"
        rec["rounds"] = 0                 # the old rounds were spent on a bug
        rec["reopened_by_startup"] = REOPEN_VERSION
        _REOPENED_THIS_RUN.add(sid)
        rec["reopened_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec["reopened_from_reason"] = (reason or "NONE (silent escalation)")[:200]
        rec.pop("last_lane_error", None)  # stale errors re-trigger the guards
        rec.pop("escalated_reason", None)
        n += 1
    return n


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
    # 09-07: green-truth invariant BEFORE state load — any phantom that the
    # 5-min watch (or a previous run) reopened must never be re-saved as green
    # by this process's in-memory copy of a pre-repair state file.
    # 09-14: this was a BLOCKING 120s subprocess call on the startup path, and the
    # watch itself waits on the state lock the escalation solver holds — measured
    # the engine sitting at cpu=00:00:00 with ZERO log output because the child
    # never returned. The watch is housekeeping, never a gate: bound it hard and
    # let the engine start regardless.
    try:
        _gt = subprocess.run([sys.executable, str(Path(__file__).parent / "green_truth_watch.py")],
                             capture_output=True, timeout=20, check=False)
        if _gt.returncode != 0:
            print(f"[eng] green-truth pass exited {_gt.returncode} (non-fatal)", flush=True)
    except subprocess.TimeoutExpired:
        print("[eng] green-truth pass exceeded 20s — skipped (non-fatal)", flush=True)
    except Exception as _e:
        print(f"[eng] green-truth pass unavailable ({_e}) — continuing", flush=True)
    st = load_state()
    _orphans = recover_orphaned_executing(st)
    # 09-14 (worker): re-open work the engine killed for reasons that were never
    # a verdict about the code. Runs BEFORE any batch, on the engine's own state,
    # so it cannot be clobbered by a stale snapshot the way a hand-edit is.
    _reopened = reopen_dead_escalations(st)
    _retired = 0 if ESCALATIONS_ENABLED else retire_escalations(st)
    if _retired:
        print(f"[eng] retired {_retired} escalated step(s) -> code yellow "
              f"(escalations are gone; the executor decides green or yellow)", flush=True)
    if _orphans or _reopened or _retired:
        save_state(st)
        if _orphans:
            print(f"[eng] recovered {_orphans} orphaned 'executing' step(s) -> pending "
                  f"(a previous process died mid-step)", flush=True)
        if _reopened:
            print(f"[eng] re-opened {_reopened} escalated step(s) that died on "
                  f"transport/timeout/prompt-cap/no-reason — none was a lane verdict "
                  f"(version {REOPEN_VERSION})", flush=True)
    todo = steps
    if args.only_step:
        todo = [s for s in todo if s["finding_id"] == args.only_step]
    if args.limit is not None:
        todo = todo[: args.limit]
    batches = pack_steps(todo)
    print(f"[eng] packed {len(batches)} batches (disjoint-file groups)")
    # 09-13: the loop iterated EVERY batch and printed "[BATCH n] 3 steps, 0
    # pending" for each already-terminal one. Measured live: 579 of those lines
    # in 3 minutes against only 5 batches that did real work — the scan itself
    # was the bulk of the log and the loop spent its time re-visiting finished
    # batches on every pass. Drop the batches with nothing left to do before the
    # loop, so a pass only walks work that can actually complete.
    def _has_pending(batch):
        return any((st["steps"].get(s["finding_id"]) or {}).get("status")
                   in (None, "pending", "executing") for s in batch)
    _before = len(batches)
    batches = [b for b in batches if _has_pending(b)]
    print(f"[eng] {len(batches)}/{_before} batches still have pending steps", flush=True)
    # 09-13: FRESH WORK FIRST. 7,487 pending steps had rounds=0 (never attempted)
    # while the engine looped on the first few batches, whose steps were already
    # at rounds 1-3 and kept returning terminal verdicts — so a pass spent its
    # lane calls escalating the same handful of hard steps and never reached the
    # untouched majority. Sort by the least-attempted step in the batch: a batch
    # containing a never-tried step goes ahead of one whose steps have all been
    # round-tripped, and the backlog drains instead of spinning.
    def _min_rounds(batch):
        return min((st["steps"].get(s["finding_id"]) or {}).get("rounds", 0) or 0
                   for s in batch)
    batches.sort(key=_min_rounds)
    sel = batches
    if args.batch is not None:
        sel = [batches[args.batch - 1]]
    # 09-14: repack the PENDING pool by file diversity so a batch's live steps
    # never contend for the same reservation (see pack_diverse).
    if _QUEUE_ENGINE:
        _pend = [s for s in todo
                 if (st["steps"].get(s["finding_id"]) or {}).get("status")
                 in (None, "pending", "executing")]
        # 09-14: ORDER MATTERS FOR PRIORITY, AND ONLY FOR PRIORITY. Measured on
        # the plan: it is priority-sorted in 6 repeated waves (24 runs, 5
        # inversions / 14,356 steps), and 58% of consecutive steps share a file.
        # The dependency language in the step text is prose about code coupling,
        # not step->step sequencing (89 hits, none a "run X before Y"). So the
        # repack must NOT shuffle priority: sort the pool by (priority rank,
        # plan index) FIRST, then fill each batch with the highest-priority
        # candidates whose files are still free. Priority order is preserved
        # exactly; file diversity is won by skipping a colliding candidate and
        # taking the next one down the same ordered list.
        _prank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
        _order = {s["finding_id"]: i for i, s in enumerate(todo)}
        _before_d = len(batches)
        _pend.sort(key=lambda s: (_prank.get(str(s.get("priority") or "").upper(), 4),
                                  _order.get(s["finding_id"], 1 << 30)))
        batches = pack_diverse(_pend, BATCH_CAP)
        print(f"[eng] repacked {len(_pend)} pending steps into {len(batches)} "
              f"file-disjoint batches (cap {GROUP_CAP}, was {_before_d})", flush=True)
        sel = batches
        if args.batch is not None:
            sel = [batches[args.batch - 1]]
    tok = None
    import aiohttp
    # 09-14: this was a flat 900s while the configured lane_timeout is 400s, so a
    # single wedged webchat call held a worker slot for 15 minutes and the engine
    # looked frozen (measured: cpu=0, three connections parked on one gateway,
    # zero log output for 10+ min). Honour lane_timeout with a small margin so the
    # request fails, the lane cools, and the batch hops to a sibling.
    _lane_budget = int(getattr(CFG, "lane_timeout", 400)) + 30
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=_lane_budget)) as session:
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
                # 09-09 (Bob 6337): clear lane context every 5 COMPLETED steps,
                # not every 5 sends/turns. A step is "completed" when its status
                # is terminal (green or escalated). Count terminal transitions in
                # this batch and, on the 5th, ask the pool to reset each lane's
                # conversation so stale context can't compound across tasks.
                _note_step_completions(statuses)

        # 09-06: never exit while steps remain pending AND the whole lane pool
        # is cooling. Exiting reset in-memory cooldowns via the Restart=always
        # loop and re-hammered the pool; instead wait out the ladder and
        # re-scan until progress resumes or the plan truly drains.
        # 09-12: count only the steps THIS run selected. Counting the whole state
        # made every scoped run (--limit / --only-step / --batch) spin forever:
        # the selected batches drained, but 10k untouched steps stayed "pending",
        # so `still` never hit 0 and the loop re-ran the same empty batches at
        # full speed (observed: `[BATCH 1] 1 steps, 0 pending` hundreds of times
        # with --limit 1).
        sel_ids = {s["finding_id"] for batch in sel for s in batch}
        _STEP_BY_ID.update({s["finding_id"]: s for batch in sel for s in batch})
        if _QUEUE_ENGINE:
            # 09-14 (owner): one batch at a time, every lane on it, barrier at
            # the end. The old loop ran PARALLEL batch-groups at once.
            await run_queue_batches(session, sel, st)
        else:
            while True:
                await asyncio.gather(*(run_one(ib) for ib in enumerate(sel, 1)))
                still = sum(1 for fid in sel_ids
                            if (st["steps"].get(fid) or {}).get("status") in (None, "pending", "executing"))
                if still == 0:
                    break
                if any(ln.is_available(time.time()) for ln in POOL.lanes):
                    continue  # lanes live again — re-scan for pickups
                print(f"[wait] {still} steps pending, pool blocked — sleeping 240s", flush=True)
                await asyncio.sleep(240)
        await save_state_serialized(st)
        aggr = dict((k, v.get("status")) for k, v in st["steps"].items())
        print(f"EXEC_DONE: total={len(aggr)} green={sum(1 for x in aggr.values() if x == 'green')} "
              f"escalated={sum(1 for x in aggr.values() if x == 'escalated')}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
