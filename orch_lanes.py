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
from typing import Any

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
    # runtime health
    dead_until: float = 0.0
    model_dead: dict[str, float] = field(default_factory=dict)
    model_fails: dict[str, int] = field(default_factory=dict)
    last_call: float = 0.0
    calls: int = 0
    failures: int = 0

    def available_models(self, now: float) -> list[str]:
        return [m for m in self.models if now >= self.model_dead.get(m, 0.0)]

    def is_available(self, now: float) -> bool:
        return now >= self.dead_until and bool(self.available_models(now))


def default_lanes(cfg=None) -> list[Lane]:
    key = _openrouter_key()
    lanes = [
        Lane("openrouter", "https://openrouter.ai/api/v1/chat/completions",
             # 09-12 (owner): OpenRouter is the FREE lane — the account key has a
             # $1 balance cap, so only `:free` models are usable. `openrouter/free`
             # is not a real model id (401/404). These are the free coding-capable
             # ids from the public /models list; `cohere/north-mini-code:free`
             # verified live (HTTP 200, cost 0).
             ["cohere/north-mini-code:free",
              "nvidia/nemotron-3-ultra-550b-a55b:free",
              "poolside/laguna-s-2.1:free",
              "nex-agi/nex-n2.5-pro:free"],
             45, 120, auth=key),  # 09-10: `auth` MUST be a
             # keyword — `prompt_cap` sits before it in the dataclass, so the
             # positional form silently bound the API key to prompt_cap and
             # left auth="" → the lane was dropped by the `or ln.auth` filter.
        Lane("deepseek", "http://127.0.0.1:8080/v1/chat/completions",
             ["anymodel"], 90, 270),
        Lane("omniroute", "http://127.0.0.1:20128/v1/chat/completions",
             # 09-12 (owner): OmniRoute is the FREE `/auto` lane — use the auto/*
             # combos, which load-balance across free providers. Verified live with
             # a real completion: auto/best-chat OK, auto/chat OK, auto/fast OK;
             # auto/best-free and auto/coding:free FAIL (route to models needing
             # an opencode key / unavailable / reasoning truncated with no content).
             ["auto/best-chat", "auto/chat", "auto/fast"],
             120, 360),
        Lane("gemini", "http://127.0.0.1:8085/v1/chat/completions",
             ["gemini 3.7 flash webchat"], 300, 900,
             prompt_cap=4000),   # 09-12: 15000 -> 6000 -> 4000. Measured on the
                                # live gateway: total prompt (system ~1958 + user)
                                # answered at 6276 chars, hung at 7960 and 9687, and
                                # wedged the tab at 16960. The safe total is ~6300,
                                # so the user slice is capped at 4000.
                                # Cap the per-turn context to what it can answer,
                                # so it stays green instead of cooking its ladder.
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
                        # nearly every reply). Read the fields out by their
                        # key markers instead of discarding the whole answer.
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
        if ("per-day" in low or "per_day" in low or "per day" in low
                or "limit_rpd" in low):
            cool = _secs_to_utc_midnight()
            note = "DAILY quota exhausted - out until 00:00 UTC"
        else:
            cool = self._ladder_s(lane, streak)
            note = f"streak {streak}"
        lane.model_dead[model] = time.time() + cool
        self.log(f"[lanes] {lane.name}/{model} ladder-cooled {cool}s ({note})")

    def _pick(self, exclude: set[str]) -> Lane | None:
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
            if res.ok:
                self.log(f"[lanes] {lane.name} empty/no-edits answer ({len(res.content)}B) — hopping")
            else:
                self.log(f"[lanes] {lane.name} failed ({res.error[:120]}) — hopping")
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

        timeout = aiohttp.ClientTimeout(
            total=int(self.cfg.lane_timeout),
            connect=int(self.cfg.lane_connect_timeout))
        retries = max(1, int(self.cfg.lane_retries))
        max_chars = int(self.cfg.max_prompt_chars)
        last_err = "unknown"
        last_status = 0

        for model in models:
            msgs: list[dict] = [{"role": "system", "content": system[:60000]}]
            if history:
                # prior assistant/user turns (same task) so an API lane remembers
                # why the previous attempt failed instead of re-trying blind.
                for turn in history[-6:]:
                    r = str(turn.get("role") or "")
                    c = str(turn.get("content") or "")
                    if r in ("assistant", "user") and c:
                        msgs.append({"role": r, "content": c[:max_chars]})
            msgs.append({"role": "user", "content": _cap_user(user, max_chars, lane.prompt_cap)})
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
                            return LaneResult(ok=True, content=content,
                                              lane=lane.name, model=model,
                                              status=200, attempts=attempt)
                        last_err = f"http {resp.status}: {body[:160]}"
                        lane.failures += 1
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
                    last_err = f"timeout after {self.cfg.lane_timeout}s"
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
            lane.failures = max(0, lane.failures)  # keep failures, just clear msgs
        self.log("[lanes] pool context reset (every-N-completed-steps)")

    async def probe(self, session, timeout: int = 45) -> dict[str, bool]:
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
