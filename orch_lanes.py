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
             ["openrouter/free"], 45, 120, key),
        Lane("deepseek", "http://127.0.0.1:8080/v1/chat/completions",
             ["anymodel"], 90, 270),
        Lane("omniroute", "http://127.0.0.1:20128/v1/chat/completions",
             ["auto/best-coding", "auto/coding", "auto/best-chat",
              "auto/best-reasoning"], 120, 360),
        Lane("gemini", "http://127.0.0.1:8085/v1/chat/completions",
             ["gemini 3.7 flash webchat"], 300, 900),
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


def parse_json_object(text: str) -> dict:
    """Extract the first balanced JSON object from a model reply.

    The old regex ``\\{.*\\}`` was greedy across the whole reply, so any prose
    after the object (or a second object) broke the parse.  This scans for a
    balanced object and ignores braces inside strings.
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
                    try:
                        obj = json.loads(stripped[start:i + 1])
                        return obj if isinstance(obj, dict) else {}
                    except json.JSONDecodeError:
                        break
        start = stripped.find("{", start + 1)
    return {}


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
    def _cool_model(self, lane: Lane, model: str, escalated: bool = False) -> None:
        lane.model_fails[model] = lane.model_fails.get(model, 0) + 1
        cool = lane.cool_esc if (escalated or lane.model_fails[model] >= 3) else lane.cool_base
        lane.model_dead[model] = time.time() + cool
        self.log(f"[lanes] {lane.name}/{model} cooled {cool}s "
                 f"(streak {lane.model_fails[model]})")

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
                   max_hops: int | None = None) -> LaneResult:
        """Try lanes until one returns a usable answer.

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
            res = await self._call_lane(session, lane, system, user)
            res.elapsed = time.time() - started
            if res.ok:
                return res
            last = res
            self.log(f"[lanes] {lane.name} failed ({res.error[:120]}) — hopping")

        last.elapsed = time.time() - started
        if not last.error:
            last.error = "all lanes exhausted"
        return last

    async def _call_lane(self, session, lane: Lane, system: str,
                         user: str) -> LaneResult:
        import aiohttp

        gap = int(self.cfg.min_lane_gap_seconds)
        if gap:
            wait = lane.last_call + gap - time.time()
            if wait > 0:
                await asyncio.sleep(min(wait, gap))

        now = time.time()
        models = lane.available_models(now)
        if not models:
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
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system[:60000]},
                             {"role": "user", "content": user[:max_chars]}],
                "max_tokens": 16000,
                "temperature": 0.2,
            }
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
                            lane.model_fails[model] = 0
                            return LaneResult(ok=True, content=content,
                                              lane=lane.name, model=model,
                                              status=200, attempts=attempt)
                        last_err = f"http {resp.status}: {body[:160]}"
                        lane.failures += 1
                        if resp.status == 429:
                            self._cool_model(lane, model)
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

        if not lane.available_models(time.time()):
            lane.dead_until = time.time() + lane.cool_base
        return LaneResult(ok=False, lane=lane.name, status=last_status,
                          error=last_err)

    # -- startup probe ---------------------------------------------------
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
                lane.dead_until = time.time() + lane.cool_base
                self.log(f"[probe] {name}: DOWN ({detail[:100]}) "
                         f"— parked {lane.cool_base}s")
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
