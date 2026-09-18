#!/usr/bin/env python3
"""orch_lanes.py — free-lane transport with honest failure reporting.

The defect this module exists to remove: the old ``lane_call`` returned a bare
``""`` for *every* failure mode — connection error, timeout, HTTP 5xx, and an
unparseable body all looked exactly like "the model answered with nothing".
The executor then treated that as a real answer, applied an empty edit list
(``all([]) is True`` — vacuously "successful"), failed verification against
another empty response, and escalated the step.  On 2026-09-05 that accounted
for 37 of 52 escalations: none of them were real code failures.

Two live transport bugs are fixed here as well:

* **SSE bodies.** OmniRoute (:20128) answers ``200`` with ``data: {...}``
  chunks and a ``data: [DONE]`` terminator.  ``json.loads(body)`` raises on
  that, so the whole lane was silently dark despite being healthy.
* **Unbounded timeouts.** The Gemini gateway hangs when Google's UI aborts a
  generation.  Under a single 900 s client timeout one such call stalled a
  worker for fifteen minutes; the per-request timeout is now a config knob.

Every call returns a :class:`LaneResult`.  Callers must branch on ``.ok``
before reading ``.content`` — an empty ``content`` with ``ok=True`` is a
genuine empty model answer and means something completely different from
``ok=False``.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# 09-16 (owner): "if a lane doesnt provide a response in 3min, have its spot added to
# the que and the next workable lane takes it on" - and, corrected the same evening:
# "its not supposed to be a 180s total budget, its supposed to be a 180s waiting till
# a first stream comes from the llm, theres not supposed to be any timer".
#
# That timer lives in the GATEWAY, because the gateway is what can see content arrive:
# `EMPTY_GRACE_MS` (webchat-api/browser.js). Its counter only ever counts while the
# newest answer row is EMPTY; it RESETS on new content and while a generation is in
# flight, so a lane that has started answering is never guillotined. Every webchat
# gateway must carry EMPTY_GRACE_MS=180000.
#
# There is deliberately NO equivalent cap in this module. A total-call budget here
# cannot tell "still thinking" from "wedged" and cuts working lanes - measured: dahl
# truncated inside its own <think> block, gemini killed mid-generation, throughput
# 4x down. Per-lane `timeout` stays as a BACKSTOP and must remain ABOVE the gateway
# hard cap.

# ------------------------------------------------------------------ lanes ---
# (name, url, models, cooldown_base, cooldown_escalated, auth_token)
# Models may be a single id or a list tried in order; a rate-limited MODEL is
# cooled individually so the lane keeps serving its remaining models.


def _secs_to_utc_midnight() -> int:
    """Seconds until 00:00 UTC, when per-day provider quotas roll over."""
    import datetime as _dt
    now = _dt.datetime.now(_dt.timezone.utc)
    nxt = (now + _dt.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0)
    return max(60, int((nxt - now).total_seconds()))


def _openrouter_key() -> str:
    for p in (os.environ.get("OPENROUTER_KEY_FILE", ""),
              Path.home() / ".claude" / "openrouter.token"):
        if not p:
            continue
        try:
            return Path(p).read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return os.environ.get("OPENROUTER_API_KEY", "").strip()


@dataclass
class Lane:
    name: str
    url: str
    models: list[str]
    cool_base: int = 90
    cool_esc: int = 270
    prompt_cap: int | None = None  # 09-09 (Bob 6365): per-turn char cap for
                                   # webchat lanes (gemini timeout guard)
    auth: str = ""
    # 09-14 (worker, B4): PER-LANE call timeout. Until now every lane shared the
    # single global `cfg.lane_timeout` (400s), while the positional `45, 120` /
    # `300, 900` args above are cool_base/cool_esc — COOLDOWNS, not timeouts. So
    # a fast API lane that answers in ~5s held a worker slot for the full 400s
    # when it hung, and a webchat lane that legitimately needs 400s+ was
    # guillotined whenever the global was lowered. Measured on the live state:
    # 269 escalated steps carry a `timeout after Ns` reason spread across NINE
    # different values (120/150/200/240/280/300/400/420/600) — that spread is
    # the global being retuned over and over for whichever lane was hurting.
    # ALWAYS pass this as a keyword: `prompt_cap` and `auth` precede it, and a
    # positional arg silently bound the API key to prompt_cap once already.
    timeout: int | None = None
    # 09-14 (worker): per-lane extra request headers. The dahl lane sits behind
    # Cloudflare, which 403s on a Python user-agent — aiohttp's default
    # ("Python/3.x aiohttp/3.y") is blocked outright. Measured live: the SAME
    # request returned 403 with the default UA and 200 with a browser UA.
    headers: dict | None = None
    # 09-14 (worker): async callable that mints a FRESH api key for this lane and
    # persists it, used when the provider reports the key's quota is spent. A
    # lane whose key is exhausted is a lane that is silently dark; dahl hands out
    # replacement keys for free, so going dark is a choice, not a constraint.
    key_refresh: "Callable | None" = None
    # runtime health
    dead_until: float = 0.0
    model_dead: dict[str, float] = field(default_factory=dict)
    model_fails: dict[str, int] = field(default_factory=dict)
    # 09-17 (BOB): "its suppoded to store history every 5 steps and reset just like the
    # webchays, no wonder its not doing well, it has amnisia". The pool already accepted
    # a `history=[...]` argument, but every caller started it EMPTY per task
    # (execute.py:1348), so a lane remembered only its own retries inside one step and
    # nothing across steps - it re-solved the same table blind every time. This is the
    # cross-step memory; reset_context() clears it every N completed steps exactly like
    # a webchat tab is reset. Capped in _call_lane.
    history: list[dict] = field(default_factory=list)
    last_call: float = 0.0
    calls: int = 0
    failures: int = 0

    def available_models(self, now: float) -> list[str]:
        return [m for m in self.models if now >= self.model_dead.get(m, 0.0)]

    def is_available(self, now: float) -> bool:
        return now >= self.dead_until and bool(self.available_models(now))


def _dahl_key() -> str:
    """Dahl API key: $DAHL_API_KEY, else the 0600 file at ~/.config/orch/dahl_key.txt.

    Kept OUT of the source: this repo is public and the engine already has a
    file-based convention for the OpenRouter PAT (see _openrouter_key).
    A fresh key with a 100,000,000-token allowance is mintable with an
    unauthenticated `POST https://inference.dahl.global/tokens` (verified live
    2026-09-14), so an exhausted key is replaceable without an account.
    """
    tok = os.environ.get("DAHL_API_KEY", "").strip()
    if tok:
        return tok
    p = Path(os.path.expanduser("~/.config/orch/dahl_key.txt"))
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return ""


# 09-14 (worker): dahl sits behind Cloudflare, which 403s a Python user-agent
# (aiohttp's default is blocked outright). Measured: same request, 403 with the
# default UA, 200 with this one. Used by the lane AND by the key minter.
_BROWSER_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36")

_DAHL_KEY_PATH = Path(os.path.expanduser("~/.config/orch/dahl_key.txt"))
_DAHL_MINT_URL = "https://inference.dahl.global/tokens"
_DAHL_MINT_COOLDOWN_S = 120.0
_dahl_last_mint = 0.0


def _is_quota_exhausted(status: int, body: str) -> bool:
    """True when the provider says THIS KEY is spent (not that we are throttled).

    Deliberately narrow: a 429 is throttling and must NOT mint a key (that would
    burn a fresh key on every rate-limit blip). Only an explicit quota/credit
    verdict counts.
    """
    if status not in (401, 402):
        return False
    low = (body or "").lower()
    return any(m in low for m in (
        "insufficient_quota", "available tokens exhausted",
        "quota", "token limit reached",
    ))


async def _dahl_mint_key(session) -> str:
    """Mint a fresh dahl key (100,000,000 tokens) and persist it 0600.

    `POST https://inference.dahl.global/tokens` needs NO auth and returns
    {"available_tokens":100000000,"token":"dahl_..."} — verified live 2026-09-14.
    Rate-limited by _DAHL_MINT_COOLDOWN_S so a broken provider cannot make the
    engine mint keys in a loop.
    """
    global _dahl_last_mint
    now = time.time()
    if now - _dahl_last_mint < _DAHL_MINT_COOLDOWN_S:
        return ""
    _dahl_last_mint = now
    import aiohttp
    try:
        async with session.post(
                _DAHL_MINT_URL, json={},
                headers={"Content-Type": "application/json",
                         "User-Agent": _BROWSER_UA},
                timeout=aiohttp.ClientTimeout(total=30)) as resp:
            if resp.status not in (200, 201):
                return ""
            data = json.loads(await resp.text())
    except Exception:
        return ""
    tok = str(data.get("token") or "").strip()
    if not tok:
        return ""
    try:
        _DAHL_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
        # write 0600 before any content lands in the file
        fd = os.open(str(_DAHL_KEY_PATH), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(tok)
    except OSError:
        pass          # an unpersisted key still works for this process
    return tok


def default_lanes(cfg=None) -> list[Lane]:
    key = _openrouter_key()
    lanes = [
        Lane("openrouter", "https://openrouter.ai/api/v1/chat/completions",
             # 09-12 (owner): OpenRouter is the FREE lane — the account key has a
             # $1 balance cap, so only `:free` models are usable. `openrouter/free`
             # is not a real model id (401/404). These are the free coding-capable
             # ids from the public /models list; `cohere/north-mini-code:free`
             # verified live (HTTP 200, cost 0).
             # 09-12: re-measured with max_tokens=20 — cohere/north-mini-code:free
             # returns EMPTY content (the whole budget goes to reasoning), which the
             # engine reads as "empty/no-edits" and hops on. It was ladder-cooling
             # first (streak 3) and stalling every step. The three below return real
             # content; thinkingmachines/inkling:free is agentic-harness-only (error).
             # 09-17: poolside/laguna-s-2.1:free PULLED. Probed live, 4 calls each:
               #   nvidia/nemotron-3-ultra-550b-a55b:free  4/4 ok (1.4s, 8.6s, 6.7s, 2.3s)
               #   nex-agi/nex-n2.5-pro:free               4/4 ok (2.9s, 3.2s, 1.0s, 1.0s)
               #   poolside/laguna-s-2.1:free              1/4 ok - 3x HTTP 429 in 0.2-0.3s
               #     ("Provider returned error ... temporarily rate-limited upstream")
               # That is the lane's 26% yield explained: 1 of its 3 models is throttled
               # upstream most of the time and a 429 costs a whole draw. Do NOT re-add
               # it on a single 200 - measure 4 calls first.
               ["nvidia/nemotron-3-ultra-550b-a55b:free",
                            "nex-agi/nex-n2.5-pro:free",
                            # 09-17: added after the union-alpha withdrawal, to restore
                            # capacity on the lane that still has free ids. Each was probed
                            # with the engine's OWN edits contract (2 calls to screen, then 4
                            # to confirm) because these fail INTERMITTENTLY - one 200 proves
                            # nothing. Measured: nemotron-3.5-lightning 4/4 (median 31.3s),
                            # nex-n2.5-mini 4/4 (median 1.3s), dots-3-note-preview 4/4
                            # (median 3.5s). NOT added: nemotron-3-nano-omni 2/4 (2x 502),
                            # qwen3.8-27b + gemma-4-31b/26b (429 at 0.3s), laguna-xs 1/2,
                            # ling-3.0-flash-svl (not a valid id).
                            "nvidia/nemotron-3.5-lightning:free",
                            "nex-agi/nex-n2.5-mini:free",
                            "dots-studio/dots-3-note-preview:free"],
             # 09-16: REVERTED to 120s. Raising this budget to 300s on the theory that
             # the median success took 309s made the lane measurably WORSE, so the
             # theory was wrong and the number goes back:
             #   BEFORE (120s budget): 34 claims / 10 greens / 29% yield
             #   AFTER  (300s budget):  9 claims /  1 green / 11% yield, 27 cooldowns
             # A longer budget just let a hanging call hold the slot 180s longer before
             # it was cooled - and the cooldowns then took the model out of rotation for
             # up to 36 min (streak 8, measured). The models themselves are FINE: a live
             # probe returned HTTP 200 for poolside/laguna-s-2.1:free and
             # nex-agi/nex-n2.5-pro:free. The median-success reading was confounded -
             # claim->green spans handoffs, it is not one call's latency.
             # prompt_cap stays 12000 ON PURPOSE - this lane's free models return EMPTY
             # content on a large prompt, which the engine reads as "no edits".
             45, 120, prompt_cap=12000, auth=key, timeout=120),  # 09-12: the engine ships a
             # ~60K-char system prompt. The free models answer a small prompt fine
             # (verified live) but return EMPTY on the full one, which the engine
             # reads as 'empty/no-edits' and ladder-cools the model — so the lane
             # went dark and every step stalled. Cap the prompt like the gemini lane.
             # 09-10: `auth` MUST be a
             # keyword — `prompt_cap` sits before it in the dataclass, so the
             # positional form silently bound the API key to prompt_cap and
             # left auth="" → the lane was dropped by the `or ln.auth` filter.
          # 09-17 (BOB): "a new model dropped on openrouter, its a stealth model,
          # called union alpha, add that as a new lane, apparently gpt6 astra level
          # intelligence". Verified live BEFORE wiring (id from GET /models:
          # `stealth/union-alpha`, name "Union Alpha", context 262144, prompt AND
          # completion price 0 - a free stealth preview):
          #   contract prompt 11,739 chars -> 4.6s, correct edits JSON, token survived
          #   contract prompt 24,162 chars -> 5.5s and 3.6s, 2/2 correct, token survived
          # Unlike the other free ids it does NOT return empty on a large prompt, and
          # its 262K context carries a real file view, so prompt_cap=24000 (the size
          # actually MEASURED - lanes that earn greens sit at 24000-32000).
          # A stealth id can be withdrawn without notice: if it starts 404ing or
          # returning empty, probe it and pull it rather than cooling it forever.
          # 09-17 (BOB): "see at what extent its rate limited, maybe even add multiple
          # lanes with union alpha". Measured before acting:
          #   10 sequential, no gap      -> 10/10 HTTP 200, median 3.5s, 0 errors
          #   conc=4 / conc=8            -> 4/4 and 8/8 HTTP 200, no errors
          #   sustained conc=10, 60 call -> 60/60 HTTP 200 in 88s = 40.7 calls/min, 0 errors
          #   no-gap burst to 50 calls   -> still 49/50 200, BUT one call STALLED to the
          #     120s curl ceiling. That stall - not a 429 - is what the engine hit on a
          #     live step, and it is why one lane can look flaky while the model is fine.
          # CONCLUSION: the constraint is CONCURRENCY-PER-LANE, not total volume and not an
          # account quota. A single lane serialises, so one slow upstream call blocks every
          # draw behind it. The model itself handles 10 in parallel. So run several lanes
          # with the same model: each gets its own pool identity, its own cooldown and its
          # own in-flight call, which is exactly the throughput lever the pool is for.
        # 09-16: PARKED - both accounts are under DeepSeek's "Messages too frequent"
        # throttle, verified by reading the PAGE over CDP (:9229 and :9225 both report
        # tooFrequent=true while :9227 reads false), and neither produces a reply at
        # all: :9225's last row is the engine's own 10,831-char prompt, never answered.
        # Measured cost of leaving them in: 4 and 3 draws in 15 min, ZERO responses,
        # every draw burning up to the 330s hard cap, and the lane cooldown (120s) kept
        # letting them straight back in. A lane that claims work and produces nothing is
        # worse than an absent lane. Re-enable when the throttle lifts (the page's
        # last row is a normal watermark again and a token send returns).
        # 09-17: RE-ENABLED. Both parked accounts returned their exact tokens on a
        # live send (:8080 -> COOL-8080-225, :8081 -> COOL-8081-25675), so the
        # throttle had lifted. Pool 4 -> 6, which is the throughput lever on the ETA.
        Lane("deepseek", "http://127.0.0.1:8080/v1/chat/completions",
             ["anymodel"], 90, 270, prompt_cap=20000, timeout=600),
        # 09-12 (owner): two more signed-in deepseek webchats, each on its own
        # profile, as separate lanes. All three share the gateway's 30s send
        # spacing (MIN_SEND_INTERVAL_MS + /tmp/deepseek_last_send), so they can
        # be used without ever hitting the account together.
        # prompt_cap: deepseek is a WEBCHAT lane — the engine's default ~71K-char
        # group prompt is pasted straight into the tab's composer, and the tab
        # then sits on "Waiting for response..." forever (measured 09-12: a
        # 71370-char send at 14:15:14 never returned; the engine blocked on it,
        # batch 1 never completed, green stayed flat). Cap it like gemini and
        # openrouter: the small prompts (557-1104 chars) answered in 3-6s.
        # 09-16: PARKED with deepseek above - same throttle, same zero responses.
        Lane("deepseek2", "http://127.0.0.1:8081/v1/chat/completions",
             ["anymodel"], 90, 270, prompt_cap=20000, timeout=600),
        Lane("deepseek4", "http://127.0.0.1:8083/v1/chat/completions",
             ["anymodel"], 90, 270, prompt_cap=20000, timeout=600),
        # PULLED 09-13: every auto/* combo now 402/401 on an oc/* model
        # Lane("omniroute", "http://127.0.0.1:20128/v1/chat/completions",
             # 09-12 LATER: the auto/* combos load-balance and now route onto
             # `oc/*` models that need an opencode key — measured live:
             #   auto/best-chat -> "oc/north-mini-code-free: auth [401] Model
             #   north-mini-code-free is not supported", and the lane returned
             #   HTTP 502 to the engine. That took the whole lane dark (engine log:
             #   `omniroute=cooled (calls=20,fail=15)`).
             # Pin the CONCRETE free model instead of a combo, and put a couple of
             # verified alternates behind it.
             # 09-12 FINAL: auto/* is 100% dead — measured live, the combos route
             # onto `oc/*` models that need an opencode API key:
             #   "oc/muse-spark-1.2: model [402] This model requires an opencode API
             #    key"; "oc/hy3-free: auth [401] Model is not supported".
             # The engine log showed omniroute=cooled (calls=8,fail=8) — the lane was
             # dark and every step stalled. Pin concrete cfp/* models, which answer
             # without any opencode key (verified live).
             # 09-13: verified live — auto/fast and auto/cheap both answer (HTTP
             # 200, content "PONG"); auto/best-chat and auto/chat still route onto
             # oc/* models that need an opencode key (402/401).
             # 09-13 later: `auto/fast` REGRESSED — it now routes to
             # `oc/muse-spark-1.2`, which returns
             # `402: This model requires an opencode API key`. `auto/best-free`
             # routes to the same dead model. Verified live: `auto/coding:free`
             # and `auto/cheap` both answer with real content; `auto/fast` now
             # returns an unparseable body. Keep only the two that answer.
             # 09-13 LATEST: `auto/coding:free` ALSO regressed — measured live in
             # the engine log: `omniroute failed (http 502: oc/muse-spark-1.2:
             # model [402] This model requires an opencode API key)`, followed by
             # `omniroute/auto/coding:free ladder-cooled 1200s`. Every auto/*
             # combo load-balances onto `oc/*` sooner or later. Pin ONLY
             # `auto/cheap`, which is the one combo still answering; if it goes
             # the same way, drop the lane rather than keep a 1200s cooldown loop.
        # 09-14: RE-ENABLED. The lane was pulled when every auto/* combo routed
        # onto an `oc/*` model needing an opencode key. Verified live again:
        # `auto/cheap` answers HTTP 200 in 0.7-2.0s with real content ("PONG"),
        # 3/3 calls, model reported as `big-pickle`. That is the one combo that
        # still answers; if it regresses the same way, pull it again rather than
        # keep a cooldown loop.
        # 09-14 (regressed again): `auto/cheap` now 502s with
        # `oc/muse-spark-1.2: model — [402]: requires an opencode API key`.
        # Same failure mode as 09-13. PULLED until a real model answers.
        # Lane("omniroute", "http://127.0.0.1:20128/v1/chat/completions",
        #      ["auto/cheap"],
        #      120, 360, prompt_cap=12000),
        # 09-13 (owner): OrcaRouter — OpenAI-compatible API lane, key verified.
        # Free models unlocked after the owner linked GitHub. Measured live:
        # deepseek/deepseek-v4-flash-free answered the ENGINE'S edits contract in
        # 1.5s with a correct edit ({"edits":[...]}), and orcarouter/free,
        # tencent/hy3-free, z-ai/glm-5.3-flash-free all returned 200. This is an
        # API lane: no browser tab, no per-account mutex, no anti-ban gap — it can
        # take many concurrent calls, so it is the real throughput lever.
        # 09-17: ORCAROUTER RE-ENABLED. It was pulled 09-15 as dead weight (429
        # free_rate_limited on every call). A pull reason EXPIRES - re-measured before
        # wiring, and ALL FOUR models answered the engine's OWN edits contract 4/4:
        #   deepseek/deepseek-v4-flash-free  4/4  median 2.9s
        #   orcarouter/free                  4/4  median 2.2s
        #   tencent/hy3-free                 4/4  median 2.9s
        #   z-ai/glm-5.3-flash-free          4/4  median 3.5s
        # Like bitdeer this is an API lane: no tab, no per-account mutex, no anti-ban gap, so
        # it can take concurrent calls - the throughput lever now the stealth model is gone.
        Lane("orcarouter", "https://api.orcarouter.ai/v1/chat/completions",
             ["deepseek/deepseek-v4-flash-free", "orcarouter/free",
              "tencent/hy3-free", "z-ai/glm-5.3-flash-free"],
             45, 120, prompt_cap=24000, timeout=120,
             auth="sk-orca-KNVShgXMQpSKanLRM8BFK6ZKyVCFoN3IhIdnZxubslG"),

        # 09-15 (data-driven pull): orcarouter is DEAD WEIGHT. Over 3 hours it
        # made 86 claims, failed 33 times, was ladder-cooled 44 times and
        # produced TWO greens, while every other lane produced 3-18. All four
        # of its free models now answer 429 'free pool capacity is limited' and
        # get parked 1800s at a time, so it spends its life cooled. Re-enable
        # only after /v1/chat/completions returns 200 with real content for
        # deepseek/deepseek-v4-flash-free on a fresh day.
        # 09-13 (owner): Bitdeer AI Cloud Model Studio — OpenAI-compatible API
        # lane, key named "oculus" in their console. Base URL taken from the
        # model page's "API Interfaces" tab (api-inference.bitdeer.ai).
        # Verified live: /v1/models returns 200, and
        # deepseek-ai/DeepSeek-V4-Flash answered "PONG" (200, real completion).
        # NOTE: the $5 voucher is scope-limited — deepseek-ai/DeepSeek-V4.1-Flash
        # returns `insufficient balance` while V4-Flash bills fine, so only the
        # models that actually bill are listed here. Like orcarouter this is an
        # API lane: no tab, no mutex, no anti-ban gap.
        # 09-15: BITDEER PULLED. Dead lane, failing on every draw:
        #   [lanes] bitdeer failed (http 502: <!DOCTYPE html> ... Cloudflare)
        #   and a bare 401 on /v1/models. Measured 3 failures in 12 min, each one
        #   burning a hop and a cooldown slot on a lane that cannot answer.
        #   Re-enable only after /v1/models returns 200 and one real completion does.
        # 09-17: BITDEER RE-ENABLED. The pull note said "re-enable only after /v1/models
        # returns 200 and one real completion does" - BOTH now hold, re-measured before
        # wiring: /v1/models -> 200, and deepseek-ai/DeepSeek-V4-Flash answered the engine's
        # OWN edits contract 4/4, median 1.9s (5.1 / 1.9 / 1.9 / 1.4). A fast API lane is the
        # only lever that matters now the stealth model is gone. If it 502s again, re-check
        # both signals rather than waiting - and only list a model that actually BILLS
        # (V4.1-Flash returns insufficient balance; the $5 voucher is scope-limited).
        Lane("bitdeer", "https://api-inference.bitdeer.ai/v1/chat/completions",
             ["deepseek-ai/DeepSeek-V4-Flash"],
             60, 180, prompt_cap=24000, timeout=120,
             auth="AIni2RlIlDeDOEclStU3"),
        # 09-13 (owner): ChatGPT webchat lane (Free account, text chat only —
        # image analysis is capped but text is unlimited). Gateway :8087 on the
        # owner's CDP 9224 Chrome. Four harness bugs had to be fixed first
        # (browser.js commit 6dab23d + the empty-phantom-row fix): the composer
        # selector, the assistant message role, a viewportSize crash, and the
        # send click that never submitted. Verified live: 2/2 "PONG" in ~18s.
        # 09-13: Kimi webchat lane. Gateway :8086 on CDP 9230. Two harness bugs
        # had to be fixed first: Kimi's `.chat-input-editor` ignores
        # execCommand('selectAll'/'delete') (its draft survived a full reload at
        # 1787 chars, so every send appended after it), and its assistant turns
        # are `.chat-content-item-assistant`, which the default message selector
        # does not match. Verified live: HTTP 200 returning "PONG" in 7.7s.
        # 09-13: PULLED FROM THE POOL. Kimi's composer restores a saved
        # per-conversation draft (measured 6811 -> 21176 chars as sends
        # appended to it), so the prompt is never the leading text and the
        # gateway wedges to its hard cap on every draw. Clearing in the same
        # CDP session as the insert (commit 9f3ecbf) did NOT hold — the draft
        # came back. Re-enable only after a hand-driven send is confirmed with
        # an empty composer on a FRESH conversation.
        # Lane("kimi", "http://127.0.0.1:8086/v1/chat/completions",
        #      ["kimi k2 webchat"], 90, 270,
        #      prompt_cap=12000),
        # 09-13: TEMPORARILY OUT OF THE POOL. The account is Free and the tab
        # shows "Messages limit reached", so every send returns an EMPTY
        # assistant node and never completes. Measured live: the gateway sat at
        # outstandingMs=361319 with the assistant row at 0 chars and not growing,
        # while the engine held a slot on it for the full 400s lane_timeout — a
        # capped lane is worse than a missing one. Re-add when the cap resets.
        # 09-14 (owner): RE-ENABLED. The lane was pulled because the reader grabbed
        # ChatGPT's EMPTY phantom assistant rows and sat to the timeout while the
        # real answer sat in an earlier row. That is now handled: the gateway runs
        # `WEBCHAT_MODE=chatgpt` (drop-in 95-mode.conf) so the mode's
        # `skipEmptyMessageRows` quirk applies. Verified live: HTTP 200, PONG in 5s.
# 09-16 RE-ENABLED (Bob: "get it up too"). The earlier pull blamed a Free-plan
# soft-cap. That was wrong. Re-tested with a FRESH chat: POST /newchat, then a real
# completion returned {"content":"PONG"} in seconds, and the DOM showed
# [["user",8],["assistant",4]] with busy=false. The stale thread was the fault, not
# the account - the same lesson gemini taught: when a lane that used to work stops,
# the regression is in OUR state, not the model. Every send now opens a fresh chat
# via the gateway (thread-reset watchdog stays on this lane).
# 09-16 RE-ENABLED - Bob was right, this is not a chatgpt problem. He pushed back:
# "thats likely a code issue and not a chatgbt issue, dig into it, chatgpt doesnt
#  store context across sessions". Digging in, the lane is FINE and my earlier
# stale-reader verdict was wrong:
#   - short prompt      ("TOKEN-ALPHA-5521")            -> echoed EXACTLY
#   - 11,266-char prompt with the token at the END      -> echoed EXACTLY
#   - 33,066-char prompt with the token at the END      -> echoed EXACTLY
# (that last one is ABOVE this lane's 32000 prompt_cap, so even an over-cap
# prompt survives the insert). Each test used a unique token and required the
# reply to MATCH it - the test I should have run before pulling the lane.
# The earlier "4 chars returned" was the gateway recovering from a 5.7-HOUR
# WEDGE that had been cleared minutes before; I read a post-wedge artifact as a
# permanent defect. The lesson is recorded in AGENTS.md.
        # 09-17: 420 -> 600. Measured, ChatGPT needs ~360s to its FIRST content on an
        # engine prompt (row empty + aria-busy the whole time, then it fills with a
        # real edit contract). A 420s budget guillotined it 60s short. Chain is
        # grace 400s < this 600s < gateway TIMEOUT 650s.
        # 09-17 PULLED (latency, NOT a broken lane - Bob was right that it works):
        # ChatGPT needs ~360s per generation and exposes no stop control, which is
        # genuinely fine for a human. It is NOT fine for the pool: measured, the
        # engine ended up with ALL SIX worker connections parked on :8087
        # (ss: 6x ESTAB to 8087, cpu 00:00:05 over 465s, wchan=ep_poll) and
        # logged nothing for 3+ minutes - 0 commits in 8. A 6-minute lane that
        # the pool can pick freely is a wedge, not a worker.
        # Re-enable only with a way to stop the pool piling onto it (e.g. a
        # concurrency cap of 1), never on a latency change alone.
        # Lane("chatgpt", "http://127.0.0.1:8087/v1/chat/completions",
        # ["chatgpt webchat"], 120, 420,
        # prompt_cap=32000, timeout=600),
#              # functioning correctly"): engine budget 300s against a gateway HARD_CAP
             # of 310s left a 10s margin, so on any thinking-heavy reply the ENGINE
             # timed out first ("chatgpt failed (timeout after 300s)") and the pool
             # then cooled the lane ("all models of this lane are cooled" x7), which
             # is what made it look dead. It is a thinking model - 300s was already
             # raised once from 210s for exactly this reason. Gateway is now
             # TIMEOUT=450000 / HARD_CAP_MS=430000, comfortably above this budget.  # 09-14: 210s guillotined the thinking model mid-answer ("chatgpt failed (timeout after 210s)"); gateway HARD_CAP_MS now 310s so the gateway always outlives this budget.  # 09-14: 12000 truncated the file contents to ~9.6K so chatgpt couldn't see the code; raise so the full prompt gets through

        # 09-15 (data-driven pull): the chatgpt account is SOFT-CAPPED again —
        # it accepts the prompt and generates NOTHING. Measured on the live tab
        # over CDP: three assistant rows all 0 chars, no stop-button, no error
        # and no limit banner, the prompt sitting at the bottom of the thread
        # unanswered; a FRESH thread (/newchat) returned empty too. The gateway
        # only reports it as 'Webchat response is empty after 180s'. Re-enable
        # only after a real non-empty answer is seen in the DOM, never on the
        # absence of the banner alone.
        # 09-14 (owner): Dahl Inference — OpenAI-compatible API lane on
        # decentralised GPU infra, 100M free tokens per key, no account needed.
        # Docs: https://docs.dahl.global/ | base https://inference.dahl.global/v1
        #
        # Three things this lane needs that no other lane does, all measured live:
        #  1. A BROWSER USER-AGENT. Cloudflare fronts the endpoint and 403s a
        #     Python UA; aiohttp's default is blocked. Same request, same key:
        #     403 with the default UA, 200 with the browser UA below.
        #  2. Reasoning-block stripping. MiniMax-M2.7 answers inside
        #     <think>...</think> and restates the CONTRACT EXAMPLE there, so the
        #     old first-balanced-object parse returned old_string="OLD" — a
        #     placeholder that can never apply. Handled in _strip_reasoning().
        #  3. Room for the thought. The visible answer is short but the model
        #     spends most of its budget reasoning, so the completion budget must
        #     not be clipped the way a webchat lane's is.
        #
        # STRAIGHT TO MINIMAX (owner 2026-09-14: "js have it go straight to
        # minstrall"). DeepSeek-V4-Flash-0731 was tried first for a while, but it
        # answers HTTP 429 `model_concurrency` on every single probe:
        #   "This model is at concurrency capacity. Signed-in and paid accounts
        #    are admitted first."
        # That is an ACCOUNT-TIER gate for anonymous keys, not a transient queue,
        # so listing it first bought nothing and cost a guaranteed-failed hop
        # plus a 60s ladder cool on every draw (visible in the engine journal as
        # `dahl/deepseek-ai/DeepSeek-V4-Flash-0731 ladder-cooled 60s`). Same rule
        # that pulled omniroute: a model that can never answer is worse than a
        # missing one. MiniMax answered every probe (1.4-12.7s).
        # Set ORCH_DAHL_DEEPSEEK=1 to put DeepSeek back in front — worth doing
        # once the key is signed in to a Dahl account, which is what lifts the
        # gate.
        #
        # KEY EXHAUSTION IS SELF-HEALING. An spent key answers
        # `402 insufficient_quota: available tokens exhausted` (the owner's
        # original key did this on its first call). `key_refresh=_dahl_mint_key`
        # mints a replacement in-flight via the unauthenticated
        # `POST /tokens` (100,000,000 tokens), persists it 0600 to
        # ~/.config/orch/dahl_key.txt, and retries the same model once, so the
        # lane never goes dark over a quota that is free to replace.
        *([Lane("dahl", "https://inference.dahl.global/v1/chat/completions",
                (["deepseek-ai/DeepSeek-V4-Flash-0731", "MiniMaxAI/MiniMax-M2.7"]
                 if os.environ.get("ORCH_DAHL_DEEPSEEK", "0") == "1"
                 else ["MiniMaxAI/MiniMax-M2.7"]),
                60, 180, prompt_cap=24000, timeout=180,
                auth=_dahl_key(),
                key_refresh=_dahl_mint_key,
                headers={"User-Agent": _BROWSER_UA})]
          if _dahl_key() else []),
# 09-16: GEMINI PULLED. It cannot commit a send on its tab. Evidence:
#   - 3h window: 23 step-claims, 1 green, 11 'timeout after 720s', and the
#     gateway logged 18 sends / 0 responses.
#   - After clearing five stray app tabs, pinning gw1 to one exact CDP target
#     (TAB_ID) and reloading it clean, a 3,178-char probe STILL logged
#     'prompt still in composer after the click' - the send never commits.
#   - Same signature as gemini2 and NoteGPT: the click reaches the node and
#     nothing submits. Not a budget problem and not a login problem.
# Re-enable only after a hand-driven send in that tab returns real content.
        # 09-16: prompt_cap 12000 -> 32000. Measured from gemini's own replies in
        # /tmp/lane_empty_debug.jsonl, every one of them is a cannot-fix that names
        # the SAME cause: "file content truncated beyond line 38", "not present in
        # provided file chunks", "chunk is truncated ... beyond the visible window".
        # The lane is not refusing the work - it cannot SEE the file. It answers and
        # gets cooled, and 11 claims produced 0 greens. Every lane that earns greens
        # sits at 24000-32000; gemini was the only one still at 12000.
        # 09-16 (owner): "pause gemini for now cuz im using it on a diff computer".
        # PAUSED - the owner is driving the same Google account from another machine,
        # and one account serialises its own turns, so our automated sends would
        # collide with real work. The whole balanced-paren statement is commented, not
        # just its first line (a half-commented Lane( breaks the parse). The gateway
        # is stopped too, so nothing can send. Re-enable by uncommenting these three
        # lines and starting oculus-gemini-gw.
        # Lane("gemini", "http://127.0.0.1:8085/v1/chat/completions",
        #      ["gemini 3.7 flash webchat"], 300, 900,
        #      prompt_cap=32000, timeout=720),
        # 09-15: re-enabled after I parked it on a WRONG attribution. gw1 and gw2 share
        # one chrome, but the CDP churn is gw1's OWN behaviour - measured 50 reconnect
        # events / 4 min with gw2 stopped vs 52 with it running. Sharing is not the
        # cause, so the lane goes back in. Watch its green count: it earned 0 greens in
        # 40 min (2 claims / 4 fails) - if that repeats it is a lane-quality problem,
        # not an interference problem.
        # 09-15 (Bob): "Try to open concurrent gemini threads, add a 2nd chat".
        # A SECOND gemini gateway on its own port, attached to its own tab in the
        # same chrome, with its OWN WEBCHAT_ACCOUNT so the two do not share the
        # send lock and genuinely run in parallel. Verified live: gw2 answered
        # "PONG" while gw1 was mid-send.
# 09-16: GEMINI2 PULLED. It is now pinned to its own CDP target (TAB_ID) so the
# two gateways no longer fight over one composer, and it STILL does not send:
# measured 3 sends / 0 received in 40 min, and a hand-driven 24-char "PONG"
# probe sat on "prompt still in composer after the click" for 190s+ with no
# answer. Both tabs are the SAME google account and Gemini serializes an
# account, so the second lane just burns a hop on a send that never commits.
# Re-enable only after a hand-driven send on its own tab returns real content.
#         Lane("gemini2", "http://127.0.0.1:8089/v1/chat/completions",
#              ["gemini 3.7 flash webchat 2"], 300, 900,
#              prompt_cap=12000, timeout=720),  # gemini gw HARD_CAP is UNSET -> auto-derives 720s; 330s guillotined slow-but-healthy replies (Bob: let gemini cook)  # 09-14: 2500 -> 12000. Bob: "the messages aren't even
        # 09-15: FREEBUFF PULLED. Proven unusable as a lane, not a budget problem:
        #   the gateway log shows "freebuff reasoning effort -> Low", the prompt sent,
        #   then 9m21s later "send timed out" with the tab never producing an answer.
        #   Playwright on the tab confirms it: it answers a 20-char "Say PONG" fine and
        #   stays silent on a real 7.7K prompt. Re-enable only after a hand-driven send
        #   on a FRESH chat returns real content.
        #         Lane("freebuff", "http://127.0.0.1:8088/v1/chat/completions",
#             ["freebuff webchat"], 90, 270,
#             prompt_cap=32000, timeout=520),  # 09-14: was 420, and the gateway HARD_CAP was 400 so freebuff never got to finish ("freebuff failed (timeout after 420s)"); cap is now 540s and this budget sits under it.  # 09-14 (owner): Freebuff coding-agent webchat (GLM 5.3 Flash, Thinking). Thinking model is slow (~4-5min); timeout matches the gateway's 420s so it completes instead of guillotining.
                                # going thru ... and its supposed to have the see next
                                # chunk". A 2500-char user turn leaves gemini almost no
                                # file context and no room to call see_next_chunk, so it
                                # answered "cannot-fix" and looked inactive. Re-measured
                                # live on the running gateway: 2500->6.3s, 6000->53.8s,
                                # 12000->9.2s, 20000->33.1s, all returning PONG. The old
                                # hangs were the pre-hard-cap tab, not the size.
                                # 09-12: 15000 -> 6000 -> 4000 -> 2500. Measured on
                                # the live gateway: total prompt (system ~1958 + user)
                                # answered at 6276 chars, hung at 7960 and 9687, and
                                # wedged the tab at 16960. But that measurement was on
                                # a FRESH thread — once the gemini tab carries a long
                                # conversation, 5960-6037-char prompts hung for ~6 min
                                # (observed 05:04:08 -> 05:10:09 with no response) and
                                # the engine logged `gemini failed (timeout after 420s)`
                                # 38 times. 2500 keeps the total near 4450, inside the
                                # reliably-answerable zone on a warm thread.
    ]
    if cfg is not None:
        for extra in (cfg.lanes_extra or []):
            lanes.append(Lane(extra.get("name", "extra"), extra["url"],
                              extra.get("models", ["anymodel"]),
                              extra.get("cool_base", 90), extra.get("cool_esc", 270),
                              extra.get("auth", "")))
    if cfg is not None and cfg.exclude_lanes:
        lanes = [ln for ln in lanes if ln.name not in cfg.exclude_lanes]
    return [ln for ln in lanes if ln.name != "openrouter" or ln.auth]


# ----------------------------------------------------------------- result ---
def lane_max_tokens(lane: "Lane") -> int | None:
    """Completion budget for one lane call.

    Owner rule (2026-09-11): the token budget applies to WEBCHAT lanes only. A
    webchat lane drives a browser tab whose composer has a real per-turn limit,
    so it needs a ceiling. The API lanes (openrouter, omniroute, deepseek) bill
    per token against a key and have no such limit — capping them was truncating
    answers mid-object, which the caller then read as "lane returned nothing
    usable" and hopped.

    Returns None for API lanes, which the caller omits from the payload so the
    upstream default applies.
    """
    if lane.name == "gemini":
        return 16000
    return None


@dataclass
class LaneResult:
    """Outcome of a lane call.  Check ``ok`` before trusting ``content``."""
    ok: bool
    content: str = ""
    lane: str = ""
    model: str = ""
    status: int = 0
    error: str = ""
    elapsed: float = 0.0
    attempts: int = 0

    @property
    def empty_answer(self) -> bool:
        """True when the lane genuinely answered with no text."""
        return self.ok and not self.content.strip()


# ---------------------------------------------------------------- parsing ---
def extract_content(body: str) -> tuple[str, str]:
    """Return (content, error). Handles plain JSON and SSE bodies.

    OmniRoute streams ``data: {...}`` chunks; OpenAI-compatible gateways return
    one JSON object.  Both shapes are accepted so a healthy lane is never
    mistaken for a silent one.
    """
    text = (body or "").strip()
    if not text:
        return "", "empty body"

    # -- plain JSON ------------------------------------------------------
    if text.startswith("{"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            return "", f"json decode: {exc}"
        err = obj.get("error")
        if err:
            msg = err.get("message") if isinstance(err, dict) else str(err)
            return "", f"upstream error: {str(msg)[:200]}"
        try:
            msg = obj["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content and isinstance(msg.get("reasoning"), str):
                content = msg["reasoning"]
            return (content if isinstance(content, str) else ""), ""
        except (KeyError, IndexError, TypeError) as exc:
            return "", f"unexpected json shape: {exc}"

    # -- SSE stream ------------------------------------------------------
    if "data:" not in text:
        return "", f"unrecognised body: {text[:120]!r}"
    chunks: list[str] = []
    saw_payload = False
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue  # ": x-omniroute-*" comment lines and blanks
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        saw_payload = True
        err = obj.get("error")
        if err:
            msg = err.get("message") if isinstance(err, dict) else str(err)
            return "", f"upstream error: {str(msg)[:200]}"
        for choice in obj.get("choices") or []:
            piece = ""
            delta = choice.get("delta")
            if isinstance(delta, dict):
                piece = delta.get("content") or ""
            if not piece:
                message = choice.get("message")
                if isinstance(message, dict):
                    piece = message.get("content") or ""
            if isinstance(piece, str) and piece:
                chunks.append(piece)
    if not saw_payload:
        return "", "sse body carried no decodable chunks"
    return "".join(chunks), ""


def _repair_loose_json(cand: str) -> dict:
    """Repair the JSON webchat models actually emit.

    09-12: real edits were being thrown away. Measured on the engine's own dump
    (/tmp/lane_empty_debug.jsonl): 8 of the last real-edit replies failed a
    strict parse, every one of them for the same two reasons —
      * a RAW control character inside a string value (a real newline from a
        multi-line old_string/new_string),
      * an unescaped inner quote: a docstring written as `\"\"\"` instead of
        `\\\"\\\"\\\"`.
    Both make json.loads refuse the document, so parse_json_object returned {} and
    the engine recorded a genuine edit as "empty/no-edits answer" and hopped.

    Walk the text once. Inside a string, escape control characters, and treat a
    quote as the string's END only when the next non-space character is
    structural (one of , : } ]) — anything else is content the model forgot to
    escape, so escape it here.
    """
    out: list[str] = []
    in_str = esc = False
    expect_key = True   # a string that follows '{' or ',' is a KEY; after ':' it is a VALUE
    is_key = False
    i, n = 0, len(cand)
    while i < n:
        ch = cand[i]
        if in_str:
            if esc:
                out.append(ch); esc = False; i += 1; continue
            if ch == "\\":
                out.append(ch); esc = True; i += 1; continue
            if ch == "\n":
                out.append("\\n"); i += 1; continue
            if ch == "\r":
                out.append("\\r"); i += 1; continue
            if ch == "\t":
                out.append("\\t"); i += 1; continue
            if ch == '"':
                j = i + 1
                while j < n and cand[j] in " \t\r\n":
                    j += 1
                nxt = cand[j] if j < n else ""
                # A KEY ends at `":`. A VALUE ends at `"` followed by , } or ].
                # Anything else is an unescaped quote the model meant as CONTENT
                # — e.g. `"old_string": "  "fixP1B1R0F1": {...` where the inner
                # `":` is text, not a separator. Treating that as a terminator is
                # what made the repair give up on real edits.
                ends = (nxt == ":" ) if is_key else (nxt in ",}]" or nxt == "")
                if ends:
                    out.append('"'); in_str = False; i += 1; continue
                out.append('\\"'); i += 1; continue
            out.append(ch); i += 1; continue
        if ch == '"':
            in_str = True; is_key = expect_key
            out.append(ch); i += 1; continue
        if ch == ":":
            expect_key = False
        elif ch in "{,":
            expect_key = True
        out.append(ch); i += 1
    try:
        obj = json.loads("".join(out))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _lenient_edits(text: str) -> dict:
    """Read {"edits":[...]} out of a reply that is not valid JSON.

    Gemini returns Python source inside JSON strings without escaping every quote
    or newline, so the document does not parse and the whole reply used to be
    discarded ("empty/no-edits answer" — 239 hops in the log). Rather than repair
    the JSON, read the fields out by their key markers: the shape is known, so
    each value runs from just after its opening quote to the last quote before
    the next known key. Values are then unescaped back to text.
    """
    import re as _re

    FIELDS = ("step", "file", "old_string", "new_string", "content", "notes")
    positions = []
    for f in FIELDS:
        for m in _re.finditer(r'"' + f + r'"\s*:\s*"', text):
            positions.append((m.start(), m.end(), f))
    if not positions:
        return {}
    positions.sort()

    def unescape(v: str) -> str:
        return (v.replace('\\n', '\n').replace('\\t', '\t')
                 .replace('\\r', '\r').replace('\\"', '"').replace('\\\\', '\\'))

    edits = []
    cur: dict = {}
    for i, (ks, vs, f) in enumerate(positions):
        nxt = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[vs:nxt]
        # drop the trailing structural noise: the closing quote + , } ] whitespace
        body = _re.sub(r'"\s*[,\]}]*\s*$', '', body, flags=_re.S)
        val = unescape(body)
        if f == "step" and cur:
            edits.append(cur); cur = {}
        cur[f] = val
    if cur:
        edits.append(cur)

    # keep only entries that carry a real edit
    edits = [e for e in edits if e.get("file") and ("new_string" in e or "content" in e)]
    return {"edits": edits} if edits else {}


_REASONING_SPANS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
    ("<|begin_of_thought|>", "<|end_of_thought|>"),
)


def _strip_reasoning(text: str) -> str:
    """Remove chain-of-thought spans so only the real answer is parsed.

    An UNCLOSED opener (the reply was truncated mid-thought) drops everything
    after it — there is no answer in a cut-off thought, and keeping it would
    hand the caller the model's scratch work as if it were the result.
    """
    for open_tag, close_tag in _REASONING_SPANS:
        if open_tag not in text:
            continue
        out = []
        rest = text
        while True:
            i = rest.find(open_tag)
            if i < 0:
                out.append(rest)
                break
            out.append(rest[:i])
            j = rest.find(close_tag, i + len(open_tag))
            if j < 0:
                break               # unclosed: discard the tail
            rest = rest[j + len(close_tag):]
        text = "".join(out)
    return text


def parse_json_object(text: str) -> dict:
    """Extract the first balanced JSON object from a model reply.

    The old regex ``\\{.*\\}`` was greedy across the whole reply, so any prose
    after the object (or a second object) broke the parse.  This scans for a
    balanced object and ignores braces inside strings.

    On a JSONDecodeError the candidate is retried through
    :func:`_repair_unescaped_quotes`, because webchat models routinely emit
    docstrings with raw ``\"\"\"`` inside a string value.
    """
    if not isinstance(text, str) or "{" not in text:
        return {}
    # 09-14 (worker): STRIP REASONING BLOCKS BEFORE SCANNING.
    # A reasoning model restates the contract inside its own chain of thought:
    #   <think>... I need to output {"edits":[{"path":"FILE","old_string":"OLD",
    #   "new_string":"NEW"}],"notes":"..."} ...</think>
    #   {"edits":[{"path":"a.py","old_string":"alpha","new_string":"OMEGA"}]}
    # The balanced scan takes the FIRST object, which is the EXAMPLE — so the
    # engine applied `old_string: "OLD"`, got "old_string not found (context
    # changed)", and burned the step's rounds on a placeholder while the real
    # edit sat in the very next object. Measured live against
    # MiniMaxAI/MiniMax-M2.7 on the dahl lane: the parser returned
    # [{'path': 'FILE', 'old_string': 'OLD'}] for a reply whose actual answer was
    # a correct alpha->OMEGA edit. Drop the reasoning span first.
    text = _strip_reasoning(text)
    if "{" not in text:
        return {}
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1] if stripped.count("```") >= 2 else stripped
        if stripped.startswith("json"):
            stripped = stripped[4:]
    start = stripped.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if esc:
                esc = False
                continue
            if ch == "\\" and in_str:
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = stripped[start:i + 1]
                    try:
                        obj = json.loads(cand)
                        return obj if isinstance(obj, dict) else {}
                    except json.JSONDecodeError:
                        # Model emitted Python source with unescaped quotes /
                        # newlines inside a JSON string (gemini does this on
                        # nearly every reply). Repair the escapes and re-parse;
                        # if that still fails, read the fields out by their key
                        # markers instead of discarding the whole answer.
                        repaired = _repair_loose_json(cand)
                        if repaired and repaired.get("edits"):
                            return repaired
                        lenient = _lenient_edits(stripped[start:])
                        if lenient:
                            return lenient
                        break
        start = stripped.find("{", start + 1)
    return {}


def normalize_edits(obj: dict) -> dict:
    """Coerce tool-call-shaped model replies into the engine's edits envelope.

    09-10: free omniroute models (cfp/nemotron) ignore the edits schema and emit
    their OWN tool-call JSON — ``{"action":"write_file","path":..,"content":..}``
    or ``{"name":"edit_file","arguments":{...}}``. The engine only understands
    ``{"edits":[{"file","old_string","new_string"}]}``, so those replies parsed
    as no-edits and the lane hopped forever (the "omniroute returns prose" bug —
    it was never prose, it was a valid edit in the wrong envelope).
    """
    if not isinstance(obj, dict) or isinstance(obj.get("edits"), list):
        return obj
    args = obj.get("arguments")
    if isinstance(args, dict):
        f = args.get("path") or args.get("file")
        if f:
            return {"edits": [{"file": f,
                               "old_string": args.get("old_string", ""),
                               "new_string": args.get("new_string", "")}]}
    f = obj.get("path") or obj.get("file")
    if f and ("content" in obj or "new_string" in obj or "old_string" in obj):
        return {"edits": [{"file": f,
                           "old_string": obj.get("old_string", ""),
                           "new_string": obj.get("new_string", obj.get("content", ""))}]}
    return obj


# ------------------------------------------------- webchat health signals ---
# Failures a webchat gateway raises about ITS OWN tab/queue state rather than
# about the request. They clear on their own, so the right response is "use a
# different lane for now", never "retry this exact lane twice more".
_WEBCHAT_TRANSIENT = (
    ("stranded in composer", "stranded composer"),
    ("webchat send failed", "send did not commit"),
    ("send mutex", "account mutex contention"),
    ("still generating from a previous request", "tab busy"),
    ("composer stayed empty", "composer never took the text"),
    ("send button vanished", "composer re-render"),
)



def _lane_is_webchat(lane) -> bool:
    """True when this lane is driven through a local browser gateway, not an API.

    Webchat lanes point at 127.0.0.1:<port> (their conversation lives in the tab);
    API lanes point at a remote host. Used to decide whether cross-step history
    must be injected into the request - it must NOT be for a webchat.
    """
    url = str(getattr(lane, "url", "") or "")
    return "127.0.0.1" in url or "localhost" in url


def _is_webchat_transient(body: str) -> bool:
    low = (body or "").lower()
    return any(sig in low for sig, _ in _WEBCHAT_TRANSIENT)


def _transient_reason(body: str) -> str:
    low = (body or "").lower()
    for sig, reason in _WEBCHAT_TRANSIENT:
        if sig in low:
            return reason
    return "unknown"


def _cap_user(user: str, max_chars: int, prompt_cap: int | None) -> str:
    """09-09 (Bob 6365): feed an oversized step context to a webchat lane in a
    bounded turn so it can answer inside the gateway timeout instead of cooking
    its ladder. Preserve the instructional head (STEP/STATE/FILES) and bring the
    FILE CONTENTS slice within the cap."""
    cap = max_chars
    if prompt_cap:
        cap = min(max_chars, prompt_cap)
    if len(user) <= cap:
        return user
    marker = "FILE CONTENTS:\n"
    idx = user.find(marker)
    if idx < 0:
        # no file-content block: just hard-truncate (rare)
        return user[:cap]
    head = user[: idx + len(marker)]
    body = user[idx + len(marker):]
    # The instructional head alone can exceed the cap (a step with many FILES
    # lines). `cap - len(head)` then goes negative and max(0, ...) returns the
    # whole head, so the "cap" never applied and gemini got the full 17k+ prompt
    # it was supposed to be protected from. Truncate the head in that case.
    if len(head) >= cap:
        return head[:cap]
    return head + body[: cap - len(head)]


# -------------------------------------------------------------- lane pool ---
class LanePool:
    """Round-robin pool that hops lanes until one actually answers."""

    def __init__(self, cfg, lanes: list[Lane] | None = None, log=print):
        self.cfg = cfg
        self.lanes = lanes if lanes is not None else default_lanes(cfg)
        self.pinned: str | None = None
        self.log = log
        self._idx = 0
        self._lock = asyncio.Lock()

    # -- health ----------------------------------------------------------
    def _ladder_s(self, lane: Lane, streak: int) -> int:
        # 09-06 user ladder: 15m base + 5m per SEQUENTIAL issue (15→20→25→30…)
        # for omniroute's models; every other lane starts from its own
        # configured cool_base and climbs the same +5m step. Success resets.
        step = int(os.environ.get("OMNI_COOLDOWN_STEP_S", "300"))
        base = int(os.environ.get("OMNI_COOLDOWN_BASE_S", "900")) if lane.name == "omniroute" else lane.cool_base
        return base + max(0, streak - 1) * step

    def _cool_model(self, lane: Lane, model: str, escalated: bool = False,
                    body: str = "") -> None:
        # 09-09 (Bob): gemini must NEVER be cooled — cooldowns apply to
        # omniroute ONLY. A gemini failure (gw restart / transient timeout /
        # conn error) must be retried on the very next pick instead of
        # ladder-cooling the lane, which is what wedged the engine every time
        # the gateway bounced. Keep the model always available; never set
        # model_dead for gemini.
        if lane.name == "gemini":
            lane.model_fails[model] = 0
            lane.model_dead.pop(model, None)
            self.log(f"[lanes] gemini/{model} issue ignored — no cooldown "
                     f"(Bob rule: cooldowns are omniroute-only)")
            return
        # 09-17: a lane that REFUSED a draw because its tab is still generating is not
        # failing - it is busy, and it will answer if we come back. Measured on chatgpt:
        # it needs ~360s per generation, so at the engine's draw rate it is busy almost
        # every time, and the ladder benched it for escalating periods (streaks 1 -> 2 ->
        # 3 in 30 min) for simply working. That is how a healthy lane gets removed from
        # the pool. Do not climb the ladder; let the next pick retry it.
        if "still generating" in (body or "").lower():
            self.log(f"[lanes] {lane.name}/{model} busy (still generating) — "
                     f"no cooldown, retry on the next pick")
            return
        lane.model_fails[model] = lane.model_fails.get(model, 0) + 1
        streak = lane.model_fails[model]
        # 09-06: a per-DAY provider quota does not recover on the streak
        # ladder (900s + 300s/streak). openrouter returned
        # "Rate limit exceeded: free-models-per-day-high-balance" 115x in one
        # day because the ladder kept putting it back into rotation every
        # ~15-30 min, and 12 of those attempts hung to the full lane_timeout
        # (600s) - roughly 2h of a worker slot spent on a lane that could not
        # answer until the quota reset. Cool those until 00:00 UTC instead.
        low = (body or "").lower()
        # 09-13 (owner): a webchat "Messages too frequent" / rate_limit is the
        # account throttling US for sending too fast — not a dead model. The
        # 90s base ladder was far too short: the lane came back while the
        # throttle was still in force and immediately tripped it again. Give it
        # a flat 15 min to let the account cool, and do NOT climb the streak
        # ladder from it (a longer ban would just compound).
        # 09-13 FOOTGUN: match the WEBCHAT message, not the substring
        # "rate_limit". OrcaRouter's 429 body is `free_rate_limited`, which
        # contains "rate_limit" — so a broad match gave an API lane the 15-min
        # WEBCHAT cooldown instead of its own 45s ladder and parked it for
        # nothing. Restrict this to the webchat lanes and the exact phrasing.
        _webchat = lane.name in ("deepseek", "deepseek2", "deepseek4",
                                 "deepseek5", "gemini", "kimi", "chatgpt", "notegpt")
        if _webchat and ("too frequent" in low or "messages too frequent" in low
                         or "finish_reason: rate_limit" in low):
            cool = int(os.environ.get("WEBCHAT_RATE_LIMIT_COOLDOWN_S", "900"))
            note = "webchat rate limit (messages too frequent) - 15 min cool"
        elif "free_rate_limited" in low or "free model capacity" in low:
            # 09-14: OrcaRouter's 429 is a SITE-WIDE free pool, not our
            # per-key quota — "Free model capacity is limited right now."
            # The 45s/345s ladder put the lane straight back into rotation
            # while the pool was still full, so every hop burned a worker
            # slot on a guaranteed 429 (measured 4 cooled models cycling all
            # afternoon). Cool it out long enough that a pick is actually
            # worth making; it recovers on its own.
            cool = int(os.environ.get("FREE_POOL_COOLDOWN_S", "1800"))
            note = "free pool capacity full - 30 min cool"
        elif ("per-day" in low or "per_day" in low or "per day" in low
                or "limit_rpd" in low):
            cool = _secs_to_utc_midnight()
            note = "DAILY quota exhausted - out until 00:00 UTC"
        else:
            cool = self._ladder_s(lane, streak)
            note = f"streak {streak}"
        lane.model_dead[model] = time.time() + cool
        self.log(f"[lanes] {lane.name}/{model} ladder-cooled {cool}s ({note})")

    def _pick(self, exclude: set[str]) -> Lane | None:
        # 09-14 (owner, per-step batch queue): a worker holds ONE lane for the
        # duration of one step, so no two workers can land on the same lane and
        # the file reservations line up with real lanes. `pinned` is set by the
        # queue driver around a single step call.
        name = getattr(self, "pinned", None)
        if name and name not in exclude:
            now_pin = time.time()
            for lane in self.lanes:
                if lane.name == name:
                    # 09-15: the pinned path used to return the lane without ANY
                    # availability check, so a lane whose models were all cooled
                    # was still picked and burned the hop with "all models of this
                    # lane are cooled". Measured: openrouter 6 of its 9 failures in
                    # 30 min were exactly that. A pinned lane that cannot serve is
                    # not a lane - fall through to the rotation.
                    if lane.is_available(now_pin):
                        return lane
                    break

        now = time.time()
        n = len(self.lanes)
        for i in range(n):
            lane = self.lanes[(self._idx + i) % n]
            if lane.name in exclude:
                continue
            if lane.is_available(now):
                self._idx = (self._idx + i + 1) % n
                return lane
        return None

    def health_report(self) -> str:
        now = time.time()
        rows = []
        for ln in self.lanes:
            state = "up" if ln.is_available(now) else f"cooled {int(ln.dead_until - now)}s"
            rows.append(f"{ln.name}={state}(calls={ln.calls},fail={ln.failures})")
        return " ".join(rows)

    # -- call ------------------------------------------------------------
    async def call(self, session, system: str, user: str,
                   max_hops: int | None = None,
                   want_edits: bool = False,
                   history: list[dict] | None = None) -> LaneResult:
        """Try lanes until one returns a usable answer.

        ``history`` carries prior conversation turns (``[{"role":"assistant",
        "content":...},{"role":"user","content":...}]``) for the SAME logical
        task. API lanes (openrouter/omniroute) use it to remember why a prior
        attempt failed instead of statelessly re-attempting blind (Bob 09-09).
        Each call sends system + history + current user so a failed step's
        reason is in-context on the retry.

        When ``want_edits`` is set the answer is only "usable" if it is 200
        non-empty AND its parsed JSON has a truthy ``edits`` list.  This keeps a
        garbage-prose free lane (openrouter/free) or a ``{"edits":[]}`` lane
        (omniroute/nemotron) from claiming victory behind the strong gemini lane:
        the pool hops past them until either a real edits answer appears or every
        lane is exhausted.  Returns ``ok=True`` only for the real-edits lane.

        Returns ``ok=False`` when every eligible lane failed — the caller must
        NOT interpret that as "the model produced no edits".
        """
        hops = max_hops if max_hops is not None else int(self.cfg.lane_max_hops)
        tried: set[str] = set()
        last = LaneResult(ok=False, error="no lane available")
        started = time.time()

        for _ in range(max(1, hops)):
            async with self._lock:
                lane = self._pick(tried)
            if lane is None:
                break
            tried.add(lane.name)
            t_lane = time.time()
            res = await self._call_lane(session, lane, system, user, history=history)
            res.elapsed = time.time() - started
            # An HTTP 200 with empty content is NOT a usable answer for an edit
            # repair — the model produced nothing. Free lanes (openrouter/free,
            # omniroute free aliases) habitually 200-empty and were starving the
            # strong gemini lane behind them. Hop instead of claiming success so
            # gemini always gets a shot before we give up.
            usable = res.ok and not res.empty_answer
            if usable and want_edits:
                # A non-empty answer is only really usable if it carries a real
                # edits block. openrouter/free returns coherent PROSE (no edits)
                # and omniroute/nemotron returns {"edits":[]}; both must bypass
                # gemini.  -- if nothing but prose/empty comes back, hop on.
                ex = normalize_edits(parse_json_object(res.content))
                usable = bool(ex and isinstance(ex.get("edits"), list) and ex["edits"])
            if usable:
                return res
            # 09-13: a lane verdict that the work is already present / cannot be
            # fixed is TERMINAL — the remaining lanes will say the same thing
            # about the same file, so hopping just spends the pool. Measured
            # live: 13-16 empty/no-edits hops PER LANE for single steps, ~25 min
            # with zero commits, while every lane independently answered
            # "cannot-fix:P1B6R0F6#20: ...", "P1B6R0F3#8 — satisfied (verified
            # by MANIFEST.json read)" or "— cannot-fix (already satisfied /
            # stale finding); no edits emitted". Return it to the engine, which
            # escalates on these shapes instead of burning the rotation.
            _low = (res.content or "").lower()
            if res.ok and ("cannot-fix" in _low or "no edit emitted" in _low
                           or "no edits emitted" in _low):
                self.log(f"[lanes] {lane.name} terminal verdict — returning without hopping")
                res.elapsed = time.time() - started
                return res
            if res.ok:
                self.log(f"[lanes] {lane.name} empty/no-edits answer ({len(res.content)}B) — hopping")
            else:
                # 09-17: the per-lane budget is a BACKSTOP and nothing logged how close a
                # lane came to it, so a 120s timeout could not be told apart from a stall
                # or from a genuinely slow call. Record the real per-call latency (and the
                # prompt size) so a budget claim rests on a number.
                self.log(f"[lanes] {lane.name} failed "
                         f"(prompt={len(system) + len(user)} chars, "
                         f"elapsed={time.time() - t_lane:.1f}s) ({res.error[:120]}) — hopping")
            last = res

        last.elapsed = time.time() - started
        if not last.error:
            last.error = "all lanes exhausted"
        return last

    async def _call_lane(self, session, lane: Lane, system: str,
                         user: str, history: list[dict] | None = None) -> LaneResult:
        import aiohttp

        gap = int(self.cfg.min_lane_gap_seconds)
        if gap:
            wait = lane.last_call + gap - time.time()
            if wait > 0:
                await asyncio.sleep(min(wait, gap))

        now = time.time()
        models = lane.available_models(now)
        if not models:
            # gemini is never cooled (Bob rule) — it always has its model back.
            if lane.name != "gemini":
                lane.dead_until = now + lane.cool_base
            return LaneResult(ok=False, lane=lane.name,
                              error="all models of this lane are cooled")

        # 09-14 (worker, B4): per-lane budget, falling back to the global. This is a
        # BACKSTOP, not the answer-wait: it must never cut a lane that is already
        # producing. The real "has it started answering?" timer is the GATEWAY's
        # EMPTY_GRACE_MS (180s on every webchat lane) - measured in browser.js, the
        # empty-grace counter RESETS the moment content arrives and while a
        # generation is in flight, so once the model starts there is no timer at all.
        #
        # 09-16: I briefly capped this at 180s here, reading the owner's "no response
        # in 3min" as a total-call budget. WRONG - it guillotined lanes mid-work
        # (dahl truncated inside its own thinking, gemini mid-generation, throughput
        # 4x down). The 3-minute rule is TIME TO FIRST CONTENT, and it belongs to the
        # gateway that can actually see the first token.
        lane_budget = int(lane.timeout or self.cfg.lane_timeout)
        timeout = aiohttp.ClientTimeout(
            total=lane_budget,
            connect=int(self.cfg.lane_connect_timeout))
        retries = max(1, int(self.cfg.lane_retries))
        max_chars = int(self.cfg.max_prompt_chars)
        last_err = "unknown"
        last_status = 0
        refreshed_key = False   # 09-14: at most ONE key mint per lane call

        for model in models:
            # 09-14 (worker, B5): `prompt_cap` is a promise about what this lane
            # can swallow, but it was applied to the USER turn ONLY. The system
            # message got a flat [:60000] and every history turn got
            # [:max_prompt_chars] (80000) — so a lane advertising prompt_cap=12000
            # could legitimately be handed system + 6 history turns + user far
            # past its cap. That is the same failure mode as the omniroute lane
            # that answered `[502] Prompt too long (max 6000 characters)` on 83
            # steps: the engine enforced a cap the payload never respected.
            # Bound EVERY component, and the total, against the lane's cap.
            _cap = lane.prompt_cap or max_chars
            _sys = system[:min(60000, _cap)]
            msgs: list[dict] = [{"role": "system", "content": _sys}]
            _budget = max(0, _cap - len(_sys))
            # 09-17 (BOB): the lane's OWN cross-step memory first, then this task's
            # retries on top - the task's turns are newer and must survive truncation
            # before the staler cross-step ones.
            # 09-17 (BOB): history is injected for API lanes ONLY. A webchat lane's
            # conversation already lives in its tab, so re-sending past turns is
            # redundant there and eats the prompt budget the task actually needs;
            # a webchat's history is cleared by opening a NEW CHAT instead.
            _is_webchat = _lane_is_webchat(lane)
            _all_hist = ([] if _is_webchat
                         else list(getattr(lane, "history", None) or []) + list(history or []))
            if _all_hist:
                # prior assistant/user turns so a lane remembers why the previous
                # attempt failed instead of re-trying blind.
                # History is the FIRST thing sacrificed when the budget is tight:
                # it is context, the current user turn is the actual task.
                _hist_budget = _budget // 3
                _turns = []
                for turn in reversed(_all_hist[-8:]):   # newest first
                    r = str(turn.get("role") or "")
                    c = str(turn.get("content") or "")
                    if r not in ("assistant", "user") or not c:
                        continue
                    if _hist_budget <= 0:
                        break
                    c = c[:min(max_chars, _hist_budget)]
                    _hist_budget -= len(c)
                    _turns.append({"role": r, "content": c})
                _turns.reverse()                      # restore chronological order
                msgs.extend(_turns)
                _budget -= sum(len(t["content"]) for t in _turns)
            # NEVER pass a falsy cap here: `_cap_user` treats None/0 as
            # "no cap" and would hand the lane the whole untruncated prompt —
            # the exact bug this block exists to prevent. Floor it instead.
            msgs.append({"role": "user",
                         "content": _cap_user(user, max_chars, max(512, _budget))})
            payload = {
                "model": model,
                "messages": msgs,
                "temperature": 0.2,
            }
            # Owner rule 2026-09-11: budget webchat lanes only. Omitting the key
            # lets the API lane use the upstream default instead of our ceiling.
            _cap = lane_max_tokens(lane)
            if _cap is not None:
                payload["max_tokens"] = _cap
            headers = {"Content-Type": "application/json"}
            if lane.auth:
                headers["Authorization"] = f"Bearer {lane.auth}"
            # 09-14 (worker): per-lane extra headers, last so a lane can
            # override the defaults (dahl needs a browser UA past Cloudflare).
            if lane.headers:
                headers.update(lane.headers)

            for attempt in range(1, retries + 1):
                lane.calls += 1
                lane.last_call = time.time()
                try:
                    async with session.post(lane.url, json=payload,
                                            headers=headers,
                                            timeout=timeout) as resp:
                        body = await resp.text()
                        last_status = resp.status
                        if resp.status == 200:
                            content, err = extract_content(body)
                            if err:
                                last_err = f"parse: {err}"
                                lane.failures += 1
                                self._cool_model(lane, model)
                                break  # body shape is wrong for this model
                            # 09-09 (Bob): a 200 with EMPTY content is a dead lane
                            # (webchat composer timeout → empty capture). It was
                            # NOT cooling the model, so the pool re-picked the
                            # same degraded lane every pass and the empty-hop never
                            # cycled forward. Cool it so the next pick moves on.
                            if not content.strip():
                                lane.failures += 1
                                self._cool_model(lane, model)
                                return LaneResult(ok=False, content=content,
                                                  lane=lane.name, model=model,
                                                  status=200, attempts=attempt,
                                                  error="empty 200 answer (dead lane)")
                            lane.model_fails[model] = 0
                            lane.model_dead.pop(model, None)  # 09-06: success resets ladder
                            # 09-17 (BOB): carry this exchange to the lane's NEXT step.
                            # reset_context() clears it every N completed steps - the same
                            # cadence a webchat tab is reset on. Capped so the prompt cannot
                            # creep: the request truncates to the newest 8 turns.
                            try:
                                _h = getattr(lane, "history", None)
                                if _h is not None:
                                    _h.append({"role": "user", "content": user[:4000]})
                                    _h.append({"role": "assistant", "content": content[:4000]})
                                    if len(_h) > 8:
                                        del _h[:-8]
                            except Exception:
                                pass
                            return LaneResult(ok=True, content=content,
                                              lane=lane.name, model=model,
                                              status=200, attempts=attempt)
                        last_err = f"http {resp.status}: {body[:160]}"
                        lane.failures += 1
                        # 09-14 (worker): THE KEY IS SPENT, NOT THE LANE.
                        # dahl hands out replacement keys for free
                        # (unauthenticated POST /tokens, 100M tokens), so an
                        # exhausted key used to take the lane dark for no reason
                        # — the owner's original key answered
                        # `402 insufficient_quota: available tokens exhausted`
                        # on its very first call. Mint a fresh one, swap it in,
                        # and retry the SAME model once. Deliberately narrow: a
                        # 429 is throttling and must never mint (that would burn
                        # a new key on every rate-limit blip), and the minter
                        # itself is cooldown-guarded against a mint loop.
                        if (not refreshed_key and lane.key_refresh
                                and _is_quota_exhausted(resp.status, body)):
                            refreshed_key = True
                            new_key = await lane.key_refresh(session)
                            if new_key and new_key != lane.auth:
                                lane.auth = new_key
                                headers["Authorization"] = f"Bearer {new_key}"
                                self.log(f"[lanes] {lane.name} key was exhausted — "
                                         f"minted a fresh one, retrying")
                                continue      # retry this model with the new key
                            self.log(f"[lanes] {lane.name} key exhausted and could "
                                     f"not mint a replacement")
                        if resp.status == 429:
                            self._cool_model(lane, model, body=body)
                            break
                        if resp.status in (400, 401, 403, 404, 422):
                            self._cool_model(lane, model, escalated=True)
                            break
                        # 09-05: a webchat gateway that reports a stranded
                        # composer / send-mutex contention is telling us the
                        # TAB is unhealthy right now, not that the request was
                        # bad. Retrying in place walks straight back into the
                        # same tab and burns another full gateway timeout
                        # (~60-90s each) before the hop happens. Treat it as a
                        # lane-health signal: cool this lane briefly and hop to
                        # a sibling immediately, so one flaky lane slows the
                        # pass instead of killing the batch.
                        if _is_webchat_transient(body):
                            self.log(f"[lanes] {lane.name} webchat transient "
                                     f"({_transient_reason(body)}) — hopping now")
                            self._cool_model(lane, model)
                            break
                        if attempt < retries:
                            await asyncio.sleep(3 * attempt)
                except asyncio.TimeoutError:
                    lane.failures += 1
                    last_err = (f"timeout after {lane_budget}s"
                                f"{'' if lane.timeout is None else ' (per-lane budget)'}")
                    self._cool_model(lane, model, escalated=True)
                    break  # a hanging gateway must not be retried in place
                except Exception as exc:  # aiohttp connection errors
                    lane.failures += 1
                    last_err = f"{type(exc).__name__}: {str(exc)[:140]}"
                    if attempt >= retries:
                        self._cool_model(lane, model)
                    else:
                        await asyncio.sleep(3 * attempt)

        # gemini is never cooled (Bob rule: cooldowns are omniroute-only).
        if lane.name != "gemini" and not lane.available_models(time.time()):
            lane.dead_until = time.time() + lane.cool_base
        return LaneResult(ok=False, lane=lane.name, status=last_status,
                          error=last_err)

    # -- startup probe ---------------------------------------------------
    def reset_context(self) -> None:
        """09-09 (Bob 6333/6337): drop per-lane conversation history & any
        threading state so no lane carries context across task boundaries.
        The engine calls this every N COMPLETED steps (not sends) so stale
        context can't compound. History is passed per-call anyway, but this
        forces API lanes to forget and lets the engine drive the cadence.
        """
        for lane in self.lanes:
            lane.failures = max(0, lane.failures)  # keep failures, clear msgs
            # 09-17 (BOB): this is what the docstring always claimed and the code never
            # did - the per-lane cross-step history is dropped HERE, on the same
            # every-N-completed-steps cadence a webchat tab is reset on.
            try:
                lane.history.clear()
            except Exception:
                lane.history = []
        self.log("[lanes] pool context reset (every-N-completed-steps)")

    async def probe(self, session, timeout: int = 12) -> dict[str, bool]:
        """Cheap liveness probe; parks lanes that cannot answer at all.

        Probes run CONCURRENTLY under a short timeout. Probing serially meant a
        wedged gateway blocked the whole pass — the Gemini webchat lane held a
        pass for 3.5 minutes on 09-05 before being declared down, which is
        longer than solving a step usually takes.
        """
        async def one(lane: Lane) -> tuple[str, bool, str, float]:
            started = time.time()
            try:
                res = await asyncio.wait_for(
                    self._call_lane(session, lane,
                                    "Reply with one JSON object only, no fences.",
                                    'Return {"ok":true}'),
                    timeout=timeout)
                return lane.name, res.ok, res.error or res.model, time.time() - started
            except asyncio.TimeoutError:
                return lane.name, False, f"probe timeout after {timeout}s", timeout

        outcomes = await asyncio.gather(*(one(ln) for ln in self.lanes))
        by_name = {ln.name: ln for ln in self.lanes}
        results: dict[str, bool] = {}
        for name, healthy, detail, elapsed in outcomes:
            results[name] = healthy
            lane = by_name[name]
            if healthy:
                self.log(f"[probe] {name}: up ({detail}, {elapsed:.1f}s)")
            else:
                # gemini is never parked (Bob rule: cooldowns omniroute-only).
                # A startup probe miss (gw still booting) must not bench it.
                if name != "gemini":
                    lane.dead_until = time.time() + lane.cool_base
                    self.log(f"[probe] {name}: DOWN ({detail[:100]}) "
                             f"— parked {lane.cool_base}s")
                else:
                    self.log(f"[probe] {name}: DOWN ({detail[:100]}) "
                             f"— NOT parked (Bob rule: no gemini cooldowns)")
        return results


async def _selftest() -> int:
    import aiohttp
    import orch_config
    cfg = orch_config.load()
    pool = LanePool(cfg)
    async with aiohttp.ClientSession() as session:
        health = await pool.probe(session)
        print(f"\nhealth: {health}")
        up = [n for n, ok in health.items() if ok]
        if not up:
            print("NO LANE AVAILABLE")
            return 1
        res = await pool.call(session,
                              "Reply with ONE JSON object only, no fences.",
                              'Return {"answer":42}')
        print(f"call ok={res.ok} lane={res.lane} model={res.model} "
              f"elapsed={res.elapsed:.1f}s err={res.error}")
        print(f"content: {res.content[:200]!r}")
        print(f"parsed:  {parse_json_object(res.content)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_selftest()))


# 09-17 19:40 - stealth/union-alpha WAS WITHDRAWN after ~3 hours, not the week Bob expected.
# Verified live: POST /v1/chat/completions -> HTTP 404
#   {"error":{"message":"Thank you for participating in the Stealth Union Alpha testing
#    period. This model was Unbiased's Pareto...","code":404}}
# and the id no longer appears in GET /models (445 models, zero union/alpha).
# All 24 unionalpha lanes were removed as a group. Re-adding a stealth id requires probing
# the endpoint FIRST - a dead id 404s on every lane and the pool collapses onto the survivors.
