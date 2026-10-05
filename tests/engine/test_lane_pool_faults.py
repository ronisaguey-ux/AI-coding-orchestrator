#!/usr/bin/env python3
"""Fault-classification and pin-isolation tests for the LIVE lane pool.

Each test pins a defect read out of engine/orch_lanes.py and traced to the
gateway that produces the error. None of these are hypotheticals: each test's
docstring names the production response that triggers it.

  1. A BUSY webchat lane must NOT climb the cooldown ladder.
  2. `POOL.pinned` must be per-worker, not one shared slot on the pool.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import orch_config  # noqa: E402
import orch_lanes as ol  # noqa: E402


# ------------------------------------------------------------------ harness --
class FakeResp:
    """The slice of aiohttp's ClientResponse that _call_lane actually touches."""

    def __init__(self, status, body):
        self.status = status
        self._body = body

    async def text(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Replays a canned response per POST and counts the calls made."""

    def __init__(self, status, body):
        self.status = status
        self.body = body
        self.calls = 0

    def post(self, url, **kw):
        self.calls += 1
        return FakeResp(self.status, self.body)


def make_cfg(**over):
    """The REAL engine config, with only the knobs these tests care about forced.

    Using the real loader keeps the budgets under test the ones that actually
    run - a hand-rolled Config would let a test pass against numbers the engine
    never uses.
    """
    cfg = orch_config.load()
    for k, v in (dict(lane_timeout=240, lane_connect_timeout=15, lane_retries=1,
                      lane_max_hops=2, min_lane_gap_seconds=0, **over)).items():
        setattr(cfg, k, v)
    return cfg


def run(coro):
    """Run a coroutine on its own loop (py3.14 has no implicit current loop)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# The exact body webchat-api's server.js produces for a tab that is still
# generating: the thrown Error from browser.js:1156 is caught by the
# /v1/chat/completions catch-all and answered 500 (server.js:3002).
BUSY_500 = ('{"error":{"message":"webchat tab still generating from a previous '
            'request \\u2014 retry after it finishes","type":"api_error"}}')


def test_busy_tab_does_not_climb_the_cooldown_ladder():
    """A lane that is merely BUSY must not be cooled, let alone ladder-cooled.

    The gateway answers a busy tab with HTTP 500 and the message
    "webchat tab still generating from a previous request". _call_lane routes
    that body through the webchat-transient branch, which is exactly where the
    09-17 BUSY exemption is supposed to fire - but that call site does not hand
    the body to _cool_model, so _cool_model cannot see it and falls through to
    the streak ladder. Repeating the BUSY 3x benches a healthy lane for
    90s + 180s + 270s, which is the "healthy lane gets removed from the pool"
    failure the 09-17 note set out to prevent.
    """
    cfg = make_cfg()
    lane = ol.Lane("chatgpt", "http://127.0.0.1:8087/v1/chat/completions",
                   ["chatgpt webchat"], 90, 270, prompt_cap=32000, timeout=720)
    pool = ol.LanePool(cfg, lanes=[lane], log=lambda *a, **k: None)

    async def busy_three_times():
        for _ in range(3):
            await pool._call_lane(FakeSession(500, BUSY_500), lane, "sys", "user")

    run(busy_three_times())

    assert lane.model_fails.get("chatgpt webchat", 0) == 0, (
        f"a BUSY lane climbed the ladder: model_fails="
        f"{dict(lane.model_fails)} (streak must stay 0)")
    assert "chatgpt webchat" not in lane.model_dead, (
        f"a BUSY lane was cooled for {lane.model_dead.get('chatgpt webchat')}s "
        f"- being busy is not being broken")


def test_a_genuinely_unhealthy_tab_still_cools():
    """The BUSY exemption must not neuter the OTHER webchat transients.

    _WEBCHAT_TRANSIENT carries five signatures. Only one of them - "still
    generating" - means the tab is healthy and merely occupied. The rest
    (stranded composer, send did not commit, account mutex contention, composer
    never took the text, send button vanished) mean the tab is actually wedged,
    and the 09-05 doctrine is to cool the lane and hop so one flaky lane slows
    the pass instead of killing the batch. If passing the body through had
    accidentally exempted all five, every wedged tab would be retried forever.
    """
    cfg = make_cfg()
    lane = ol.Lane("chatgpt", "http://127.0.0.1:8087/v1/chat/completions",
                   ["chatgpt webchat"], 90, 270, prompt_cap=32000, timeout=720)
    pool = ol.LanePool(cfg, lanes=[lane], log=lambda *a, **k: None)

    for sig, _reason in ol._WEBCHAT_TRANSIENT:
        if "still generating" in sig:
            continue
        lane.model_dead.clear()
        lane.model_fails.clear()
        lane.dead_until = 0.0
        body = '{"error":{"message":"%s","type":"api_error"}}' % sig

        async def one():
            await pool._call_lane(FakeSession(500, body), lane, "sys", "user")

        run(one())
        assert "chatgpt webchat" in lane.model_dead, (
            f"transient {sig!r} no longer cools the lane - a wedged tab would be "
            f"retried forever and hold worker slots")
        assert lane.model_fails.get("chatgpt webchat", 0) >= 1, (
            f"transient {sig!r} did not register a failure streak")


def test_pinned_lane_is_per_worker_not_a_shared_slot():
    """Two concurrent lane workers must each get THEIR OWN pinned lane.

    orch_queue.drive_plan starts one asyncio task per lane
    (orch_queue.py: `tasks = [asyncio.create_task(worker(l)) ...]`), and each
    worker runs execute_step_pinned, which does `POOL.pinned = lane` and clears
    it in a `finally`. `pinned` is a plain attribute on the shared pool object,
    so the last writer wins for every in-flight worker: two workers land on one
    lane at once, and whichever finishes first clears the pin for everybody
    else, so the survivors free-roam the whole pool mid-step. That breaks the
    scheduler's "one lane, one step" contract - which is also what keeps the
    file reservations aligned with real lanes.
    """
    cfg = make_cfg()
    a = ol.Lane("laneA", "http://127.0.0.1:9001/v1/chat/completions", ["m"], 90, 270)
    b = ol.Lane("laneB", "http://127.0.0.1:9002/v1/chat/completions", ["m"], 90, 270)
    pool = ol.LanePool(cfg, lanes=[a, b], log=lambda *x, **k: None)

    a_ready, b_ready = asyncio.Event(), asyncio.Event()
    a_go, b_go = asyncio.Event(), asyncio.Event()
    got = {}

    async def worker(name, lane, ready, go):
        pool.pinned = lane
        ready.set()
        await go.wait()          # let the sibling overwrite the pin first
        got[name] = pool._pick(set())
        pool.pinned = None       # execute_step_pinned's `finally`

    async def scenario():
        ta = asyncio.ensure_future(worker("A", "laneA", a_ready, a_go))
        await a_ready.wait()
        tb = asyncio.ensure_future(worker("B", "laneB", b_ready, b_go))
        await b_ready.wait()
        a_go.set()
        b_go.set()
        await asyncio.gather(ta, tb)

    run(scenario())

    assert got["A"] is not None and got["B"] is not None, "both workers got a lane"
    assert got["A"] is not got["B"], (
        f"two workers landed on the same lane {got['A'] and got['A'].name!r} - "
        f"A picked {got['A'].name!r} and B picked {got['B'].name!r}; the pin is "
        f"one shared slot on the pool, not per-worker")
    assert got["A"].name == "laneA" and got["B"].name == "laneB", (
        f"each worker must get its own pin: A got {got['A'].name!r}, "
        f"B got {got['B'].name!r}")


def test_clearing_one_workers_pin_does_not_unpin_the_other():
    """`finally: POOL.pinned = None` must not strip a sibling worker's pin.

    This is the other half of the same shared-slot bug and the one that hurts
    throughput hardest: a worker that finishes early wipes the pin for every
    step still in flight, so those steps stop being lane-bound and start
    hopping the entire pool - colliding head-on with lanes other workers are
    already on.
    """
    cfg = make_cfg()
    # laneA is listed SECOND and the rotation cursor starts at 0, so an UNPINNED
    # pick deterministically returns laneB. A that gets laneA can only have got
    # it through its own pin - no accidental pass on rotation order.
    a = ol.Lane("laneA", "http://127.0.0.1:9001/v1/chat/completions", ["m"], 90, 270)
    b = ol.Lane("laneB", "http://127.0.0.1:9002/v1/chat/completions", ["m"], 90, 270)
    pool = ol.LanePool(cfg, lanes=[b, a], log=lambda *x, **k: None)

    got = {}

    async def long_worker():
        pool.pinned = "laneA"
        await asyncio.sleep(0.05)      # still mid-step when B finishes
        got["A"] = pool._pick(set())
        pool.pinned = None

    async def short_worker():
        await asyncio.sleep(0.01)
        pool.pinned = "laneB"
        got["B"] = pool._pick(set())
        pool.pinned = None             # B finishes first and clears the pin

    async def scenario():
        await asyncio.gather(asyncio.ensure_future(long_worker()),
                             asyncio.ensure_future(short_worker()))

    run(scenario())

    assert got["A"] is not None and got["A"].name == "laneA", (
        f"A lost its pin when B finished - A would have hopped the whole pool "
        f"and got {got['A'] and got['A'].name!r} instead of laneA")