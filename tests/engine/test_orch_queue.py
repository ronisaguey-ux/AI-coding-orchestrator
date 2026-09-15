#!/usr/bin/env python3
"""Test suite for orch_queue.py — the per-step batch work queue.

Covers the owner's five rules (09-14):
  1. one step per lane call, every lane works the same batch
  2. file reservations: no two lanes edit the same file, ever
  3. batch barrier: no step leaves the batch non-terminal
  4. cooldown is not an escalation: park the lane, hand the step to a sibling
  5. idle when nothing is claimable; never skip ahead
plus the yellow class (mandatory justification) and the handoff log.
"""
import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import orch_queue as oq  # noqa: E402


def make_queue(pairs, cap=15, res=None):
    ids = [p[0] for p in pairs]
    steps = {i: {"finding_id": i, "files": f} for i, f in pairs}
    recs = {i: {"rounds": 0, "status": "pending"} for i in ids}
    return oq.BatchQueue(ids, recs, reservations=res, steps_by_id=steps, group_cap=cap)


async def no_escalate(sid):
    return "yellow"


class TestYellow(unittest.TestCase):
    def test_short_justification_rejected(self):
        with self.assertRaises(oq.YellowJustificationError):
            oq.make_yellow({}, "nope")

    def test_empty_justification_rejected(self):
        with self.assertRaises(oq.YellowJustificationError):
            oq.make_yellow({}, "   ")

    def test_valid_justification_accepted_and_stamped(self):
        rec = {}
        oq.make_yellow(rec, "The fix is blocked on an external vendor API that is "
                            "not reachable from this sandbox; flagged for review.",
                       lane="escalation")
        self.assertEqual(rec["status"], "yellow")
        self.assertTrue(rec["yellow_justification"])
        self.assertEqual(rec["yellow_by"], "escalation")
        self.assertTrue(oq.is_terminal(rec))

    def test_yellow_is_terminal_green_is_terminal_escalated_is_not(self):
        self.assertTrue(oq.is_terminal({"status": "yellow"}))
        self.assertTrue(oq.is_terminal({"status": "green"}))
        self.assertFalse(oq.is_terminal({"status": "escalated"}))
        self.assertFalse(oq.is_terminal({"status": "pending"}))
        self.assertFalse(oq.is_terminal(None))


class TestFileReservations(unittest.TestCase):
    def test_all_or_nothing(self):
        r = oq.FileReservations()
        self.assertTrue(r.reserve(["a.py", "b.py"], "L1"))
        self.assertFalse(r.reserve(["b.py", "c.py"], "L2"))
        self.assertNotIn("c.py", r.holders())

    def test_conflict_detected(self):
        r = oq.FileReservations()
        r.reserve(["a.py"], "L1")
        self.assertEqual(r.conflicts(["a.py"], "L2"), ["a.py"])
        self.assertEqual(r.conflicts(["a.py"], "L1"), [])

    def test_release_frees_only_own(self):
        r = oq.FileReservations()
        r.reserve(["a.py"], "L1")
        r.reserve(["b.py"], "L2")
        r.release("L1")
        self.assertEqual(r.holders(), {"b.py": "L2"})

    def test_release_all_held(self):
        r = oq.FileReservations()
        r.reserve(["a.py", "b.py"], "L1")
        self.assertEqual(r.release("L1"), 2)
        self.assertEqual(r.holders(), {})

    def test_reserve_no_files_always_succeeds(self):
        r = oq.FileReservations()
        self.assertTrue(r.reserve([], "L1"))


class TestLaneRoster(unittest.TestCase):
    def test_cooled_lane_not_available(self):
        r = oq.LaneRoster(["a", "b"])
        r.cool("a", 60)
        self.assertEqual(r.available(), ["b"])
        self.assertEqual(r.parked(), ["a"])

    def test_cool_expires(self):
        r = oq.LaneRoster(["a"])
        now = time.time()
        r.cool("a", 10)
        self.assertTrue(r.is_cooled("a", now))
        self.assertFalse(r.is_cooled("a", now + 11))

    def test_cool_remaining_and_soonest_wake(self):
        r = oq.LaneRoster(["a", "b"])
        now = time.time()
        r.cool("a", 100)
        r.cool("b", 50)
        self.assertAlmostEqual(r.cool_remaining("a", now), 100, delta=2)
        self.assertAlmostEqual(r.soonest_wake(now), 50, delta=2)

    def test_cooldown_counts(self):
        r = oq.LaneRoster(["a"])
        r.cool("a", 1)
        r.cool("a", 1)
        self.assertEqual(r.cooldown_counts(), {"a": 2})


class TestStepHandoff(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.h = oq.StepHandoff(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_record_and_entries(self):
        self.h.record("P1#1", "lane-a", "retry", edits=[{"file": "a.py"}],
                      reply="could not apply", error="timeout after 400s")
        entries = self.h.entries("P1#1")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["lane"], "lane-a")
        self.assertEqual(entries[0]["reply"], "could not apply")
        self.assertEqual(entries[0]["error"], "timeout after 400s")

    def test_path_is_safe_for_slash_and_hash_ids(self):
        p = self.h.path("P1B/../../etc#3")
        self.assertEqual(p.parent, Path(self.tmp.name))

    def test_render_gives_path_and_history(self):
        self.h.record("S1", "lane-a", "retry", reply="first try failed")
        text = self.h.render("S1")
        self.assertIn(str(self.h.path("S1")), text)
        self.assertIn("first try failed", text)
        self.assertIn("Read that file", text)

    def test_render_chunks_long_history_keeping_the_tail(self):
        for i in range(200):
            self.h.record("S2", f"lane-{i}", "retry", reply=f"attempt {i} " + "x" * 200)
        text = self.h.render("S2", max_chars=500)
        self.assertIn("read the file for the rest", text.lower())
        self.assertLessEqual(len(text), 500 + 400)
        self.assertIn("attempt 199", text)

    def test_render_empty_when_no_history(self):
        self.assertEqual(self.h.render("nope"), "")


class TestBatchQueue(unittest.TestCase):
    def test_claim_skips_inflight_and_terminal(self):
        q = make_queue([("S1", ["a.py"]), ("S2", ["b.py"]), ("S3", ["c.py"])])
        q.records["S3"]["status"] = "green"
        self.assertEqual(q.claim("L1"), "S1")
        self.assertEqual(q.claim("L1"), "S2")
        self.assertIsNone(q.claim("L1"))
        self.assertEqual(q.status("S3"), "green")

    def test_claim_refuses_locked_files(self):
        q = make_queue([("S1", ["shared.py"]), ("S2", ["shared.py"])])
        self.assertEqual(q.claim("L1"), "S1")
        self.assertIsNone(q.claim("L2"))
        self.assertEqual(q.blocked_by_files(), ["S2"])
        q.finish("S1")
        # finish() only releases the slot — S1 is NOT done, so it is claimable again
        self.assertEqual(q.claim("L2"), "S1")
        q.records["S1"]["status"] = "green"
        q.finish("S1")
        self.assertEqual(q.claim("L2"), "S2")

    def test_barrier_holds_until_all_terminal(self):
        q = make_queue([("S1", ["a.py"]), ("S2", ["b.py"])])
        self.assertFalse(q.is_done())
        q.claim("L1")
        q.records["S1"]["status"] = "green"
        self.assertFalse(q.is_done())
        self.assertEqual(q.non_terminal(), ["S2"])
        q.records["S2"]["status"] = "yellow"
        self.assertTrue(q.is_done())

    def test_handoff_releases_files(self):
        q = make_queue([("S1", ["shared.py"])])
        q.claim("L1")
        self.assertIn("shared.py", q.res.holders())
        q.handoff("S1")
        self.assertEqual(q.res.holders(), {})
        self.assertNotIn("S1", q.inflight)
        self.assertEqual(q.claim("L2"), "S1")

    def test_awaiting_escalation(self):
        q = make_queue([("S1", ["a.py"])])
        q.claim("L1")
        q.records["S1"]["status"] = "escalated"
        q.finish("S1")
        self.assertEqual(q.awaiting_escalation(), ["S1"])
        self.assertEqual(q.next_escalation(), "S1")
        self.assertIsNone(q.next_escalation())

    def test_cap_enforced(self):
        pairs = [(f"S{i}", [f"f{i}.py"]) for i in range(16)]
        with self.assertRaises(ValueError):
            make_queue(pairs, cap=15)

    def test_pack_batches_respects_cap(self):
        batches = oq.pack_batches([f"S{i}" for i in range(32)], group_cap=15)
        self.assertEqual([len(b) for b in batches], [15, 15, 2])


class TestDriveBatch(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.h = oq.StepHandoff(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    async def test_lanes_work_the_same_batch_in_parallel(self):
        pairs = [(f"S{i}", [f"f{i}.py"]) for i in range(10)]
        q = make_queue(pairs)
        roster = oq.LaneRoster([f"L{i}" for i in range(5)])
        live = {"n": 0, "max": 0}
        seen_lanes = set()

        async def exec_step(lane, sid):
            seen_lanes.add(lane)
            live["n"] += 1
            live["max"] = max(live["max"], live["n"])
            await asyncio.sleep(0.02)
            live["n"] -= 1
            q.records[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        snap = await oq.drive_batch(q, roster, self.h, exec_step, no_escalate, poll_s=0.005)
        self.assertGreater(live["max"], 1, "lanes did not run concurrently")
        self.assertTrue(q.is_done())
        self.assertEqual(set(snap.values()), {"green"})
        self.assertGreaterEqual(len(seen_lanes), 2)

    async def test_only_one_step_per_lane_call(self):
        pairs = [(f"S{i}", [f"f{i}.py"]) for i in range(9)]
        q = make_queue(pairs)
        roster = oq.LaneRoster(["L0", "L1", "L2"])
        calls = []

        async def exec_step(lane, sid):
            calls.append((lane, sid))
            await asyncio.sleep(0.005)
            q.records[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        await oq.drive_batch(q, roster, self.h, exec_step, no_escalate, poll_s=0.005)
        self.assertEqual(len(calls), 9)
        self.assertEqual(len({sid for _l, sid in calls}), 9)

    async def test_no_two_lanes_ever_hold_the_same_file(self):
        pairs = [(f"S{i}", ["shared.py"]) for i in range(15)]
        q = make_queue(pairs)
        roster = oq.LaneRoster([f"L{i}" for i in range(8)])
        state = {"n": 0, "max": 0}

        async def exec_step(lane, sid):
            state["n"] += 1
            state["max"] = max(state["max"], state["n"])
            await asyncio.sleep(0.01)
            state["n"] -= 1
            q.records[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        await oq.drive_batch(q, roster, self.h, exec_step, no_escalate, poll_s=0.005)
        self.assertEqual(state["max"], 1, "two lanes co-edited shared.py")
        self.assertTrue(q.is_done())

    async def test_cooldown_hands_step_to_a_sibling_with_history(self):
        pairs = [("S1", ["a.py"])]
        q = make_queue(pairs)
        roster = oq.LaneRoster(["L1", "L2"])
        attempts = []

        async def exec_step(lane, sid):
            attempts.append(lane)
            if lane == "L1":
                return oq.StepOutcome("retry", lane=lane, cooled_s=60,
                                      error="429 rate limited",
                                      reply="lane was throttled", note="throttled")
            q.records[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        await oq.drive_batch(q, roster, self.h, exec_step, no_escalate, poll_s=0.005)
        self.assertEqual(attempts, ["L1", "L2"])
        self.assertEqual(q.status("S1"), "green")
        self.assertIn("L1", roster.parked())
        entries = self.h.entries("S1")
        self.assertEqual([e["lane"] for e in entries], ["L1", "L2"])
        self.assertEqual(entries[0]["error"], "429 rate limited")

    async def test_lane_sits_idle_while_files_are_locked(self):
        pairs = [("S1", ["a.py"]), ("S2", ["a.py"])]
        q = make_queue(pairs)
        roster = oq.LaneRoster(["L1", "L2"])
        gate = asyncio.Event()
        state = {"n": 0, "max": 0}

        async def exec_step(lane, sid):
            state["n"] += 1
            state["max"] = max(state["max"], state["n"])
            if sid == "S1":
                await gate.wait()
            state["n"] -= 1
            q.records[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        task = asyncio.create_task(
            oq.drive_batch(q, roster, self.h, exec_step, no_escalate, poll_s=0.005))
        await asyncio.sleep(0.05)
        self.assertEqual(state["max"], 1, "second lane entered the locked file")
        gate.set()
        await asyncio.wait_for(task, timeout=5)
        self.assertTrue(q.is_done())

    async def test_execution_failure_routes_to_escalation_persona(self):
        pairs = [("S1", ["a.py"])]
        q = make_queue(pairs)
        roster = oq.LaneRoster(["L1"])
        seen = []

        async def exec_step(lane, sid):
            return oq.StepOutcome("escalate", lane=lane, note="cannot-fix: target absent")

        async def on_escalate(sid):
            seen.append(sid)
            oq.make_yellow(q.records[sid],
                           "Target symbol does not exist anywhere in the repo and the "
                           "plan snippet is stale; needs human review before retry.",
                           lane="escalation")
            return "yellow"

        await oq.drive_batch(q, roster, self.h, exec_step, on_escalate, poll_s=0.005)
        self.assertEqual(seen, ["S1"])
        self.assertEqual(q.status("S1"), "yellow")
        self.assertTrue(q.is_done())

    async def test_escalation_persona_can_fix_it_green(self):
        pairs = [("S1", ["a.py"])]
        q = make_queue(pairs)
        roster = oq.LaneRoster(["L1"])

        async def exec_step(lane, sid):
            return oq.StepOutcome("escalate", lane=lane)

        async def on_escalate(sid):
            q.records[sid]["status"] = "green"
            return "green"

        await oq.drive_batch(q, roster, self.h, exec_step, on_escalate, poll_s=0.005)
        self.assertEqual(q.status("S1"), "green")
        self.assertTrue(q.is_done())

    async def test_escalation_without_justification_stalls_loudly(self):
        pairs = [("S1", ["a.py"])]
        q = make_queue(pairs)
        roster = oq.LaneRoster(["L1"])

        async def exec_step(lane, sid):
            return oq.StepOutcome("escalate", lane=lane)

        async def on_escalate(sid):
            try:
                oq.make_yellow(q.records[sid], "no")
            except oq.YellowJustificationError:
                return "escalated"
            return "yellow"

        with self.assertRaises(oq.BatchStalled):
            await oq.drive_batch(q, roster, self.h, exec_step, on_escalate,
                                 poll_s=0.005, max_stall_rounds=4)

    async def test_all_lanes_cooled_waits_then_resumes(self):
        pairs = [("S1", ["a.py"]), ("S2", ["b.py"])]
        q = make_queue(pairs)
        roster = oq.LaneRoster(["L1"])
        calls = []

        async def exec_step(lane, sid):
            calls.append(sid)
            if len(calls) == 1:
                return oq.StepOutcome("retry", lane=lane, cooled_s=0.05)
            q.records[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        await asyncio.wait_for(
            oq.drive_batch(q, roster, self.h, exec_step, no_escalate, poll_s=0.005),
            timeout=5)
        self.assertTrue(q.is_done())
        self.assertGreaterEqual(len(calls), 3)

    async def test_barrier_never_lets_a_lane_touch_another_batch(self):
        pairs = [(f"S{i}", [f"f{i}.py"]) for i in range(4)]
        q = make_queue(pairs)
        other = make_queue([("X1", ["x.py"])])
        roster = oq.LaneRoster(["L1"])
        seen = set()

        async def exec_step(lane, sid):
            seen.add(sid)
            await asyncio.sleep(0.005)
            q.records[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        await oq.drive_batch(q, roster, self.h, exec_step, no_escalate, poll_s=0.005)
        self.assertEqual(seen, {"S0", "S1", "S2", "S3"})
        self.assertEqual(other.snapshot(), {"X1": "pending"})

    async def test_stress_15_steps_8_lanes_mixed_outcomes(self):
        pairs = [(f"S{i}", [f"f{i}.py"]) for i in range(15)]
        q = make_queue(pairs, cap=15)
        roster = oq.LaneRoster([f"L{i}" for i in range(8)])
        attempts = {}

        async def exec_step(lane, sid):
            n = attempts.get(sid, 0) + 1
            attempts[sid] = n
            if n == 1 and int(sid[1:]) % 3 == 0:
                return oq.StepOutcome("retry", lane=lane, cooled_s=0.02)
            if n == 2 and int(sid[1:]) % 3 == 0:
                return oq.StepOutcome("escalate", lane=lane)
            q.records[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        async def on_escalate(sid):
            oq.make_yellow(q.records[sid],
                           f"Escalation persona exhausted its one final shot on {sid}; "
                           "blocked on a missing upstream fixture, flagged for review.",
                           lane="escalation")
            return "yellow"

        await asyncio.wait_for(
            oq.drive_batch(q, roster, self.h, exec_step, on_escalate, poll_s=0.005),
            timeout=10)
        self.assertTrue(q.is_done())
        self.assertEqual(len(q.terminal()), 15)
        self.assertEqual(q.non_terminal(), [])
        for sid in q.order:
            self.assertIn(q.status(sid), ("green", "yellow"))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestDrivePlan(unittest.IsolatedAsyncioTestCase):
    """S3: work stealing across batches — lanes never idle."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.h = oq.StepHandoff(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    async def test_lane_steals_work_from_a_later_batch(self):
        # batch A is blocked on one shared file; batch B has free steps.
        # Under the barrier engine the lane idled; under S3 it takes B's work.
        recs = {}
        steps_by_id = {}
        for sid, fs in [("A1", ["shared.py"]), ("A2", ["shared.py"]),
                        ("B1", ["b1.py"]), ("B2", ["b2.py"])]:
            recs[sid] = {"rounds": 0, "status": "pending"}
            steps_by_id[sid] = {"finding_id": sid, "files": fs}
        roster = oq.LaneRoster(["L1", "L2"])
        order = []

        async def exec_step(lane, sid):
            order.append(sid)
            await asyncio.sleep(0.01)
            recs[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        async def esc(sid):
            return "yellow"

        await asyncio.wait_for(
            oq.drive_plan([["A1", "A2"], ["B1", "B2"]], recs, roster, self.h,
                          exec_step, esc, steps_by_id=steps_by_id, poll_s=0.005),
            timeout=5)
        self.assertEqual(set(order), {"A1", "A2", "B1", "B2"})
        # A2 cannot be claimed while A1 holds shared.py, so B1/B2 ran first
        self.assertLess(order.index("B1"), order.index("A2"))

    async def test_no_file_is_ever_held_by_two_lanes_across_batches(self):
        recs, steps_by_id = {}, {}
        ids = []
        for bi in range(4):
            b = []
            for j in range(6):
                sid = f"B{bi}S{j}"
                # every step in every batch shares one file
                recs[sid] = {"rounds": 0, "status": "pending"}
                steps_by_id[sid] = {"finding_id": sid, "files": ["one.py"]}
                b.append(sid)
            ids.append(b)
        roster = oq.LaneRoster([f"L{i}" for i in range(8)])
        state = {"n": 0, "max": 0}

        async def exec_step(lane, sid):
            state["n"] += 1
            state["max"] = max(state["max"], state["n"])
            await asyncio.sleep(0.005)
            state["n"] -= 1
            recs[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        await asyncio.wait_for(
            oq.drive_plan(ids, recs, roster, self.h, exec_step,
                          lambda sid: "yellow", steps_by_id=steps_by_id,
                          poll_s=0.005),
            timeout=10)
        self.assertEqual(state["max"], 1, "two lanes co-edited one.py")
        self.assertTrue(all(r["status"] == "green" for r in recs.values()))

    async def test_batch_boundary_reports_when_all_its_steps_terminal(self):
        recs, steps_by_id = {}, {}
        for sid in ("A1", "A2", "B1"):
            recs[sid] = {"rounds": 0, "status": "pending"}
            steps_by_id[sid] = {"finding_id": sid, "files": [sid + ".py"]}
        roster = oq.LaneRoster(["L1"])
        done_batches = []

        async def exec_step(lane, sid):
            recs[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        await asyncio.wait_for(
            oq.drive_plan([["A1", "A2"], ["B1"]], recs, roster, self.h,
                          exec_step, lambda sid: "yellow",
                          steps_by_id=steps_by_id, poll_s=0.005,
                          on_batch_done=lambda bi, snap: done_batches.append(bi)),
            timeout=5)
        self.assertEqual(done_batches, [0, 1])

    async def test_retry_and_escalation_still_close_every_step(self):
        recs, steps_by_id = {}, {}
        for i in range(12):
            sid = f"S{i}"
            recs[sid] = {"rounds": 0, "status": "pending"}
            steps_by_id[sid] = {"finding_id": sid, "files": [f"f{i}.py"]}
        roster = oq.LaneRoster([f"L{i}" for i in range(6)])
        tries = {}

        async def exec_step(lane, sid):
            tries[sid] = tries.get(sid, 0) + 1
            if tries[sid] == 1 and int(sid[1:]) % 4 == 0:
                return oq.StepOutcome("retry", lane=lane, cooled_s=0.01)
            if tries[sid] == 2 and int(sid[1:]) % 4 == 0:
                return oq.StepOutcome("escalate", lane=lane)
            recs[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        async def esc(lane, sid):
            oq.make_yellow(recs[sid],
                           f"Escalation persona could not close {sid}: the target "
                           "symbol is absent from the repo and the plan snippet is "
                           "stale; flagged for human review.", lane=lane)
            return "yellow"

        await asyncio.wait_for(
            oq.drive_plan([[f"S{i}" for i in range(0, 6)],
                           [f"S{i}" for i in range(6, 12)]],
                          recs, roster, self.h, exec_step, esc,
                          steps_by_id=steps_by_id, poll_s=0.005),
            timeout=10)
        self.assertTrue(all(r["status"] in ("green", "yellow") for r in recs.values()))


class TestEscalationsInQueue(unittest.IsolatedAsyncioTestCase):
    """Escalations are work in the per-step queue, resolved as soon as they occur."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.h = oq.StepHandoff(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _q(self, pairs):
        recs = {sid: {"rounds": 0, "status": "pending"} for sid, _ in pairs}
        steps = {sid: {"finding_id": sid, "files": fs} for sid, fs in pairs}
        return oq.BatchQueue([sid for sid, _ in pairs], recs,
                             steps_by_id=steps, group_cap=15), recs

    def test_claim_task_offers_escalation_before_fresh_work(self):
        q, recs = self._q([("S1", ["a.py"]), ("S2", ["b.py"])])
        recs["S2"]["status"] = "escalated"
        sid, kind = q.claim_task("L1")
        self.assertEqual((sid, kind), ("S2", "escalate"))

    def test_claim_task_returns_execute_when_no_escalation(self):
        q, _ = self._q([("S1", ["a.py"])])
        sid, kind = q.claim_task("L1")
        self.assertEqual((sid, kind), ("S1", "execute"))

    def test_claim_task_reserves_files_for_escalation_too(self):
        q, recs = self._q([("S1", ["shared.py"]), ("S2", ["shared.py"])])
        recs["S1"]["status"] = "escalated"
        recs["S2"]["status"] = "escalated"
        self.assertIsNotNone(q.claim_task("L1"))
        self.assertIsNone(q.claim_task("L2"), "two lanes got the same locked file")

    async def test_escalation_resolved_inline_without_waiting(self):
        recs = {}
        steps = {}
        for i in range(6):
            sid = f"S{i}"
            recs[sid] = {"rounds": 0, "status": "pending"}
            steps[sid] = {"finding_id": sid, "files": [f"f{i}.py"]}
        roster = oq.LaneRoster([f"L{i}" for i in range(4)])
        order = []
        escalated_to_lane = []

        async def exec_step(lane, sid):
            order.append(("exec", sid, lane))
            if int(sid[1:]) % 3 == 0:
                return oq.StepOutcome("escalate", lane=lane)
            recs[sid]["status"] = "green"
            return oq.StepOutcome("green", lane=lane)

        async def on_escalate(lane, sid):
            escalated_to_lane.append((sid, lane))
            order.append(("esc", sid, lane))
            oq.make_yellow(recs[sid],
                           f"Escalation persona on {lane} closed {sid} as code yellow: "
                           "the target symbol is absent and the plan snippet is stale.",
                           lane=lane)
            return "yellow"

        await asyncio.wait_for(
            oq.drive_plan([[f"S{i}" for i in range(6)]], recs, roster, self.h,
                          exec_step, on_escalate, steps_by_id=steps, poll_s=0.005),
            timeout=10)
        self.assertTrue(all(r["status"] in ("green", "yellow") for r in recs.values()))
        self.assertEqual(len(escalated_to_lane), 2)
        # each escalation ran on a lane directly, not through a side escalator
        self.assertTrue(all(lane.startswith("L") for _s, lane in escalated_to_lane))
        # and it happened in the same pass as the step, not after all exec work
        for sid, lane in escalated_to_lane:
            idx_esc = order.index(("esc", sid, lane))
            self.assertLess(idx_esc, len(order), "escalation deferred to the end")

    async def test_escalation_gives_up_after_bounded_attempts(self):
        recs = {"S1": {"rounds": 0, "status": "pending"}}
        steps = {"S1": {"finding_id": "S1", "files": ["a.py"]}}
        roster = oq.LaneRoster(["L1", "L2", "L3"])
        calls = []

        async def exec_step(lane, sid):
            return oq.StepOutcome("escalate", lane=lane)

        async def on_escalate(lane, sid):
            calls.append(lane)
            return "escalated"          # lane could not give a verdict

        await asyncio.wait_for(
            oq.drive_plan([["S1"]], recs, roster, self.h, exec_step, on_escalate,
                          steps_by_id=steps, poll_s=0.005,
                          max_escalation_attempts=3),
            timeout=10)
        self.assertLessEqual(len(calls), 4, "escalation retried without bound")
        self.assertEqual(recs["S1"]["status"], "escalated")
        self.assertIn("escalation_gave_up", recs["S1"])

    async def test_lane_never_idles_while_escalations_remain(self):
        recs = {}
        steps = {}
        for i in range(3):
            sid = f"E{i}"
            recs[sid] = {"rounds": 0, "status": "escalated"}
            steps[sid] = {"finding_id": sid, "files": [f"e{i}.py"]}
        roster = oq.LaneRoster(["L1", "L2", "L3"])
        seen = []

        async def exec_step(lane, sid):
            return oq.StepOutcome("green", lane=lane)

        async def on_escalate(lane, sid):
            seen.append(sid)
            recs[sid]["status"] = "green"
            return "green"

        await asyncio.wait_for(
            oq.drive_plan([["E0", "E1", "E2"]], recs, roster, self.h,
                          exec_step, on_escalate, steps_by_id=steps, poll_s=0.005),
            timeout=10)
        self.assertEqual(sorted(seen), ["E0", "E1", "E2"])
        self.assertTrue(all(r["status"] == "green" for r in recs.values()))
