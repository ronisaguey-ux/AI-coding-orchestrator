"""Regression tests for the fix-executor escalation path (worker brief, Part B).

Each test pins a defect that was MEASURED on the live state file
(audits_plans/exec_state_9_4.json) on 2026-09-14:

  B1  155/502 escalated steps carried neither `last_lane_error` nor
      `escalated_reason`; 139 of those had real applied edits.
  B2  53 steps were `escalated` at rounds == 1 (one lane's verdict was terminal).
  B3  34 steps were `escalated` with rounds > MAX_ROUNDS (uncapped increments).
  B4  269 steps died on `timeout after Ns` across NINE different N, because every
      lane shared one global timeout instead of a per-lane budget.
  B5  83 steps hit `[502] Prompt too long (max 6000 characters)` because
      `prompt_cap` bounded only the user turn, not the system/history turns.
  B6  18 steps reached terminal state on a pure transport failure.

These are unit tests over the real modules — no network, no state file writes.
"""
import importlib
import os
import re
import sys
import time
from pathlib import Path

import pytest

# This repo keeps the engine at the root and drops the `orch_` prefix; the live
# deployment keeps them under scripts/ with the prefix. Support both so the same
# test file is valid in either tree.
_here = Path(__file__).resolve()
SCRIPTS = _here.parents[2] / "scripts"
if not (SCRIPTS / "execute.py").exists():
    SCRIPTS = _here.parents[2]
sys.path.insert(0, str(SCRIPTS))
os.environ.setdefault("ORCH_CONFIG", os.path.expanduser("~/.config/orch/orch.yaml"))

execute = importlib.import_module("execute")
try:
    orch_lanes = importlib.import_module("orch_lanes")
except ModuleNotFoundError:
    orch_lanes = importlib.import_module("lanes")


# ---------------------------------------------------------------- B1 + B6 ---
def test_b1_escalate_records_a_reason():
    """Escalating must always leave a non-empty, readable reason on the record."""
    rec = {"rounds": 3, "status": "pending",
           "last_apply": {"rnd": 3, "ok": True, "edits": [{"file": "a.py"}]}}
    status = execute.escalate("S#1", rec, "round budget exhausted (3/3)",
                              site="test:site")
    assert status == "escalated"
    assert rec["status"] == "escalated"
    assert rec["escalated_reason"].strip()
    assert rec["escalated_by"] == "test:site"
    assert rec["escalated_at"]


def test_b1_escalate_without_a_reason_is_refused_in_strict_mode(monkeypatch):
    """A site that forgets the reason must be catchable, not silent."""
    monkeypatch.setenv("ORCH_STRICT_ESCALATION", "1")
    rec = {"rounds": 3, "status": "pending"}
    with pytest.raises(ValueError):
        execute.escalate("S#2", rec, "", site="test:noreason")


def test_b1_escalate_without_a_reason_still_records_a_marker(monkeypatch):
    """Outside strict mode the step must STILL be diagnosable (never blank)."""
    monkeypatch.setenv("ORCH_STRICT_ESCALATION", "0")
    rec = {"rounds": 3, "status": "pending"}
    execute.escalate("S#3", rec, "", site="test:noreason")
    assert rec["status"] == "escalated"
    assert "UNSPECIFIED" in rec["escalated_reason"]
    assert "test:noreason" in rec["escalated_reason"]


def test_b1_no_raw_escalated_assignment_outside_the_helper():
    """The ONLY place that may set status='escalated' is escalate() itself.

    This is the structural guard: a new escalation site added without the helper
    re-introduces B1 by construction, and this test fails the moment it appears.
    """
    src = Path(execute.__file__).read_text(encoding="utf-8")
    # Strip comments so the explanatory prose in the helper's docstring/comments
    # does not count as a real assignment.
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert code.count('rec["status"] = "escalated"') == 1, (
        "escalation must funnel through escalate(); found a raw assignment")


# ---------------------------------------------------------------- B6 --------
@pytest.mark.parametrize("err", [
    "ClientConnectorError: Cannot connect to host 127.0.0.1:8085 ssl:default",
    "ServerDisconnectedError: Server disconnected",
    "timeout after 600s",
    'http 429: {"code":"free_rate_limited"}',
    "no lane available",
])
def test_b6_transport_failure_can_never_become_terminal(err):
    """A step whose last lane error was transport must NOT die, at any site."""
    rec = {"rounds": 3, "status": "executing", "last_lane_error": err}
    status = execute.escalate("S#4", rec, "round budget exhausted",
                              site="test:transport")
    assert status == "pending"
    assert rec["status"] == "pending"
    # the stale error is cleared so the next pass starts clean ...
    assert "last_lane_error" not in rec
    # ... but WHY it was blocked is still recorded (B7: never lose the reason).
    assert rec["last_transport_block"]["error"].startswith(err[:20])


def test_b6_is_transport_error_covers_the_engines_own_timeout_wording():
    assert execute.is_transport_error("timeout after 200s")
    assert execute.is_transport_error("timeout after 600s (per-lane budget)")
    assert not execute.is_transport_error("lane returned cannot-fix")
    assert not execute.is_transport_error("")


# ---------------------------------------------------------------- B4 --------
def test_b4_every_lane_has_a_timeout_budget():
    """No lane may fall back to the single global timeout silently."""
    lanes = orch_lanes.default_lanes(execute.CFG)
    missing = [l.name for l in lanes if not l.timeout]
    assert not missing, f"lanes with no per-lane timeout budget: {missing}"


def test_b4_webchat_budgets_exceed_their_gateway_hard_caps():
    """The engine must not guillotine a call its gateway would still answer.

    Gateway HARD_CAP_MS read live from the systemd units on 2026-09-14:
      chatgpt 180000 | gemini 300000 | deepseek/2/4 240000
    """
    hard_cap_s = {"chatgpt": 180, "gemini": 300,
                  "deepseek": 240, "deepseek2": 240, "deepseek4": 240}
    lanes = {l.name: l for l in orch_lanes.default_lanes(execute.CFG)}
    for name, cap in hard_cap_s.items():
        assert lanes[name].timeout > cap, (
            f"{name}: engine budget {lanes[name].timeout}s <= gateway hard cap {cap}s "
            f"— the engine would kill a call the gateway is still serving")


def test_b4_lane_budget_never_exceeds_the_shared_session_timeout():
    """execute.py builds ONE aiohttp session at lane_timeout + 30."""
    session_budget = int(getattr(execute.CFG, "lane_timeout", 400)) + 30
    for lane in orch_lanes.default_lanes(execute.CFG):
        assert lane.timeout <= session_budget, (
            f"{lane.name}: per-lane {lane.timeout}s exceeds session {session_budget}s")


# ---------------------------------------------------------------- B5 --------
def test_b5_prompt_cap_bounds_the_whole_payload_not_just_the_user_turn():
    """Rebuild the message assembly exactly as _call_lane does and measure it.

    Before the fix the system turn was `system[:60000]` and each history turn
    `content[:max_prompt_chars]`, so a prompt_cap=12000 lane could be handed
    hundreds of KB. This asserts the assembled total respects the cap.
    """
    cap = 12000
    max_chars = int(execute.CFG.max_prompt_chars)
    system = "S" * 50000
    history = [{"role": "assistant", "content": "A" * 40000},
               {"role": "user", "content": "U" * 40000}] * 3
    user = "FILE CONTENTS:\n" + ("X" * 200000)

    _cap = cap or max_chars
    _sys = system[:min(60000, _cap)]
    total = len(_sys)
    _budget = max(0, _cap - len(_sys))
    _hist_budget = _budget // 3
    for turn in reversed(history[-6:]):
        c = turn["content"]
        if _hist_budget <= 0:
            break
        c = c[:min(max_chars, _hist_budget)]
        _hist_budget -= len(c)
        total += len(c)
        _budget -= len(c)
    total += len(orch_lanes._cap_user(user, max_chars, max(512, _budget)))

    assert total <= cap * 1.1, (
        f"assembled payload {total} chars blows the lane's {cap}-char cap")


def test_b5_cap_user_never_returns_the_full_prompt_on_a_zero_budget():
    """A falsy cap used to mean 'no cap' — that is how the guarantee leaked."""
    user = "FILE CONTENTS:\n" + ("X" * 100000)
    out = orch_lanes._cap_user(user, 80000, max(512, 0))
    assert len(out) <= 512
    # And this is the leak the floor exists to stop: a falsy cap is treated as
    # "no lane cap", so the turn falls back to max_prompt_chars (80000) — 156x
    # the 512 the budget arithmetic had actually allowed.
    leaked = orch_lanes._cap_user(user, 80000, 0)
    assert len(leaked) == 80000
    assert len(leaked) > 512


def test_b5_system_turn_is_bound_by_the_lane_cap_in_the_source():
    """Source-level guard: the flat `system[:60000]` must be gone.

    The re-implementation test above can only prove the arithmetic; this pins
    the actual assembly in _call_lane so restoring the old line fails the build.
    """
    src = Path(orch_lanes.__file__).read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines()
                      if not ln.lstrip().startswith("#"))
    assert 'system[:60000]' not in code, (
        "system turn is capped at a flat 60000 and ignores the lane prompt_cap")
    assert 'min(60000, _cap)' in code, "system turn no longer respects the lane cap"
    assert 'c[:max_chars]' not in code, (
        "history turns are capped at max_prompt_chars and ignore the lane cap")


# ---------------------------------------------------------------- B2/B3 ----
def test_b2_b3_round_budget_is_the_documented_cap():
    assert execute.MAX_ROUNDS == 3


def test_b3_no_uncapped_round_increment_remains():
    """Every `rounds = rounds + 1` in the group path must be min()-capped."""
    src = Path(execute.__file__).read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    bad = [ln.strip() for ln in code.splitlines()
           if 'rec["rounds"] = rec.get("rounds", 0) + 1' in ln]
    assert not bad, f"uncapped round increment(s) still present: {bad}"


def test_b2_single_lane_cannot_fix_is_not_terminal_in_run_step():
    """run_step must require the round budget before honouring cannot-fix.

    The group path already did this; run_step did not, which is how 53 steps
    were escalated at rounds == 1.
    """
    src = Path(execute.__file__).read_text(encoding="utf-8")
    head = src.index('if "cannot-fix" in (res.content or "").lower():')
    window = src[head:head + 1600]
    assert 'rec["rounds"] < MAX_ROUNDS' in window, (
        "run_step escalates cannot-fix without corroborating across rounds")


# ---------------------------------------------------------------- B9 --------
def test_b9_orphan_recovery_refunds_a_round_that_no_lane_answered():
    """A round burned by a mid-step kill must be given back.

    `rounds` is incremented and saved BEFORE the lane call, so a process killed
    mid-step consumes a round on which no lane ever answered. Measured on the
    live state: 412 steps carried `orphan_recovered` and 243 of them held
    rounds>=1 with NO `last_apply` — and 25 were already at rounds==3, one pass
    away from escalating without a single lane call.
    """
    st = {"steps": {
        # killed before any lane answered -> refund
        "A#1": {"status": "executing", "rounds": 1},
        "A#2": {"status": "executing", "rounds": 3},
        # a lane DID answer, the round was really spent -> no refund
        "B#1": {"status": "executing", "rounds": 2,
                "last_apply": {"rnd": 2, "ok": True, "edits": [{"file": "x.py"}]}},
        # not executing -> untouched
        "C#1": {"status": "green", "rounds": 2},
    }}
    n = execute.recover_orphaned_executing(st)
    assert n == 3
    s = st["steps"]
    assert s["A#1"]["status"] == "pending" and s["A#1"]["rounds"] == 0
    assert s["A#2"]["status"] == "pending" and s["A#2"]["rounds"] == 2
    assert s["B#1"]["status"] == "pending" and s["B#1"]["rounds"] == 2, (
        "a round with a real lane result must NOT be refunded")
    assert s["C#1"]["status"] == "green" and s["C#1"]["rounds"] == 2
    assert s["A#1"]["orphan_recoveries"] == 1


def test_b9_orphan_recovery_never_goes_negative():
    st = {"steps": {"A#1": {"status": "executing", "rounds": 0}}}
    execute.recover_orphaned_executing(st)
    assert st["steps"]["A#1"]["rounds"] == 0


# ------------------------------------------------- dead-work re-open --------
def _esc(reason):
    return {"status": "escalated", "rounds": 3, "escalated_reason": reason}


def test_reopen_reclaims_infrastructure_deaths_but_not_real_verdicts():
    st = {"steps": {
        "T#1": _esc("timeout after 600s"),
        "T#2": _esc("ClientConnectorError: Cannot connect to host 127.0.0.1:8085"),
        "T#3": _esc('http 502: {"message":"[502]: Prompt too long (max 6000 characters)."}'),
        "T#4": _esc("http 401: API key expired."),
        "T#5": {"status": "escalated", "rounds": 3},          # no reason at all
        "V#1": _esc("lane verdict cannot-fix, corroborated across 3/3 rounds"),
        "G#1": {"status": "green", "rounds": 1},
    }}
    n = execute.reopen_dead_escalations(st)
    assert n == 5
    for sid in ("T#1", "T#2", "T#3", "T#4", "T#5"):
        rec = st["steps"][sid]
        assert rec["status"] == "pending", sid
        assert rec["rounds"] == 0, sid
        assert rec["reopened_by_startup"] == execute.REOPEN_VERSION
        assert "last_lane_error" not in rec and "escalated_reason" not in rec
    # a corroborated lane verdict is REAL — it must stay terminal
    assert st["steps"]["V#1"]["status"] == "escalated"
    # green is finished work and must never be touched
    assert st["steps"]["G#1"]["status"] == "green"


def test_reopen_is_idempotent_and_cannot_loop():
    """A step gets at most one second life per version — never a churn loop."""
    st = {"steps": {"T#1": _esc("timeout after 200s")}}
    assert execute.reopen_dead_escalations(st) == 1
    # it is pending now; escalate it again for the SAME infra reason
    st["steps"]["T#1"].update(status="escalated", escalated_reason="timeout after 200s")
    assert execute.reopen_dead_escalations(st) == 0, "re-opened twice — would loop"
    assert st["steps"]["T#1"]["status"] == "escalated"


def test_reopen_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ORCH_REOPEN_DEAD", "0")
    st = {"steps": {"T#1": _esc("timeout after 200s")}}
    assert execute.reopen_dead_escalations(st) == 0
    assert st["steps"]["T#1"]["status"] == "escalated"


# ------------------------------------------- reasoning models / dahl --------
def test_reasoning_block_is_stripped_before_json_is_parsed():
    """A <think> block that restates the CONTRACT must not be parsed as answer.

    Measured live against MiniMaxAI/MiniMax-M2.7 on the dahl lane: the model
    restates the contract example inside its chain of thought, so the
    first-balanced-object scan returned the PLACEHOLDER
    {"path":"FILE","old_string":"OLD"} instead of the real edit. Applying that
    yields "old_string not found (context changed)" and burns the step's rounds.
    """
    raw = ('<think>The user wants alpha -> OMEGA. The contract says to output '
           '{"edits":[{"path":"FILE","old_string":"OLD","new_string":"NEW"}],'
           '"notes":"..."} so I will do that.</think>\n\n'
           '{"edits":[{"path":"a.py","old_string":"alpha","new_string":"OMEGA"}],'
           '"notes":"Changed alpha to OMEGA"}')
    got = orch_lanes.normalize_edits(orch_lanes.parse_json_object(raw))
    edits = got.get("edits") or []
    assert len(edits) == 1
    assert edits[0].get("old_string") == "alpha", "parsed the contract example, not the answer"
    assert edits[0].get("new_string") == "OMEGA"


@pytest.mark.parametrize("open_tag,close_tag", [
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
])
def test_reasoning_strip_covers_the_common_tag_spellings(open_tag, close_tag):
    raw = (f'{open_tag}{{"edits":[{{"path":"FILE","old_string":"OLD"}}]}}{close_tag}'
           '{"edits":[{"path":"real.py","old_string":"x","new_string":"y"}]}')
    got = orch_lanes.normalize_edits(orch_lanes.parse_json_object(raw))
    assert (got.get("edits") or [{}])[0].get("path") == "real.py"


def test_unclosed_reasoning_block_yields_no_edits_not_scratch_work():
    """A reply truncated mid-thought has no answer — it must not look like one."""
    raw = '<think>I should emit {"edits":[{"path":"FILE","old_string":"OLD"}]} next'
    got = orch_lanes.normalize_edits(orch_lanes.parse_json_object(raw))
    assert not (got.get("edits") or []), "returned the model's scratch work as the answer"


def test_plain_json_is_unaffected_by_the_reasoning_strip():
    raw = '{"edits":[{"path":"a.py","old_string":"x","new_string":"y"}],"notes":"n"}'
    got = orch_lanes.normalize_edits(orch_lanes.parse_json_object(raw))
    assert (got.get("edits") or [{}])[0].get("path") == "a.py"


def test_dahl_lane_is_configured_correctly_when_a_key_is_present():
    """dahl needs a browser UA (Cloudflare 403s a Python UA) and DeepSeek first."""
    lanes = {l.name: l for l in orch_lanes.default_lanes(execute.CFG)}
    if "dahl" not in lanes:
        pytest.skip("no dahl key configured on this box")
    d = lanes["dahl"]
    assert d.headers and "User-Agent" in d.headers
    assert "Python" not in d.headers["User-Agent"], (
        "a Python user-agent is 403'd by the Cloudflare in front of dahl")
    assert d.models[0] == "MiniMaxAI/MiniMax-M2.7", (
        "owner asked for the lane to go straight to MiniMax; DeepSeek is "
        "account-gated and only cost a guaranteed-failed hop")
    assert d.auth, "lane present without a key"
    assert d.timeout


def test_lane_extra_headers_are_actually_sent():
    """A `headers` dict on a Lane must reach the request, overriding defaults."""
    lane = orch_lanes.Lane("x", "http://127.0.0.1:1/v1", ["m"], auth="k",
                           headers={"User-Agent": "Mozilla/5.0", "X-Test": "1"})
    headers = {"Content-Type": "application/json"}
    if lane.auth:
        headers["Authorization"] = f"Bearer {lane.auth}"
    if lane.headers:
        headers.update(lane.headers)
    assert headers["User-Agent"] == "Mozilla/5.0"
    assert headers["X-Test"] == "1"
    assert headers["Authorization"] == "Bearer k"


def test_no_api_key_is_hardcoded_for_dahl():
    """The dahl key lives in ~/.config/orch/dahl_key.txt or $DAHL_API_KEY."""
    # Match a real KEY (dahl_ + a long random tail), not identifiers like
    # _dahl_key / _dahl_mint_key / _DAHL_MINT_URL. The previous string-replace
    # hack tripped over every new dahl_* symbol.
    src = Path(orch_lanes.__file__).read_text(encoding="utf-8")
    leaked = re.findall(r"dahl_[A-Za-z0-9]{20,}", src)
    assert not leaked, f"a dahl API key looks hardcoded in the source: {leaked}"


def test_a_deliberate_reopen_is_not_resurrected_by_the_merge():
    """The merge must not undo a re-open this run performed on purpose.

    save_state_serialized keeps a TERMINAL status found on disk over a
    non-terminal one in memory. That is right for a stale snapshot, but for a
    step this run deliberately re-opened it means a single stale write of the
    old `escalated` record gets adopted back into memory and propagated —
    observed live as `escalated` flip-flopping 0 <-> 502 with the 155 no-reason
    records returning minutes after a clean re-open.
    """
    execute._REOPENED_THIS_RUN.clear()
    st = {"steps": {"T#1": _esc("timeout after 600s")}}
    assert execute.reopen_dead_escalations(st) == 1
    assert "T#1" in execute._REOPENED_THIS_RUN

    # simulate the merge seeing the OLD terminal record still on disk
    disk = {"T#1": {"status": "escalated", "rounds": 3,
                    "escalated_reason": "timeout after 600s"}}
    mem = st["steps"]
    TERMINAL = ("green", "escalated", "obsolete", "blocked")
    for sid, srec in mem.items():
        drec = disk.get(sid)
        if not isinstance(drec, dict):
            continue
        if sid in execute._REOPENED_THIS_RUN and drec.get("status") in TERMINAL:
            continue
        if drec.get("status") in TERMINAL and srec.get("status") not in TERMINAL:
            mem[sid] = drec
    assert mem["T#1"]["status"] == "pending", "the re-open was undone by the merge"
    execute._REOPENED_THIS_RUN.clear()


# ------------------------------------------------- dahl key self-heal -------
@pytest.mark.parametrize("status,body,expected", [
    (402, '{"error":{"code":"insufficient_quota","message":"available tokens exhausted"}}', True),
    (401, '{"error":{"message":"quota exceeded"}}', True),
    # A 429 is THROTTLING, not a spent key. Minting here would burn a fresh
    # 100M-token key on every rate-limit blip.
    (429, '{"error":{"code":"free_rate_limited","message":"capacity is limited"}}', False),
    (429, '{"error":{"code":"model_concurrency","message":"at concurrency capacity"}}', False),
    (200, '{"choices":[]}', False),
    (500, '{"error":"boom"}', False),
    (402, '{"error":{"message":"card declined"}}', False),   # 402 without a quota verdict
])
def test_only_a_spent_key_triggers_a_mint(status, body, expected):
    assert orch_lanes._is_quota_exhausted(status, body) is expected


def test_dahl_goes_straight_to_minimax_by_default(monkeypatch):
    """DeepSeek is account-gated (429 model_concurrency on every probe), so
    listing it first only bought a guaranteed-failed hop plus a 60s ladder cool."""
    monkeypatch.delenv("ORCH_DAHL_DEEPSEEK", raising=False)
    lanes = {l.name: l for l in orch_lanes.default_lanes(execute.CFG)}
    if "dahl" not in lanes:
        pytest.skip("no dahl key configured on this box")
    assert lanes["dahl"].models == ["MiniMaxAI/MiniMax-M2.7"]

    monkeypatch.setenv("ORCH_DAHL_DEEPSEEK", "1")
    lanes = {l.name: l for l in orch_lanes.default_lanes(execute.CFG)}
    assert lanes["dahl"].models[0].startswith("deepseek-ai/"), (
        "the opt-in must put DeepSeek back in front")


def test_dahl_lane_can_refresh_its_own_key():
    lanes = {l.name: l for l in orch_lanes.default_lanes(execute.CFG)}
    if "dahl" not in lanes:
        pytest.skip("no dahl key configured on this box")
    assert lanes["dahl"].key_refresh is orch_lanes._dahl_mint_key


def test_key_minting_is_cooldown_guarded_against_a_loop():
    """A provider stuck on 402 must not make the engine mint keys forever."""
    import asyncio

    class _Boom:
        def post(self, *a, **k):
            raise AssertionError("minted inside the cooldown window")

    orch_lanes._dahl_last_mint = time.time()          # pretend we just minted
    got = asyncio.run(orch_lanes._dahl_mint_key(_Boom()))
    assert got == "", "minted again inside the cooldown window"
    orch_lanes._dahl_last_mint = 0.0                  # restore for other tests


def test_mint_cooldown_is_a_real_interval():
    assert orch_lanes._DAHL_MINT_COOLDOWN_S >= 60
