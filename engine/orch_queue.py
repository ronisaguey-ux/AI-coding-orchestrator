#!/usr/bin/env python3
"""orch_queue.py — per-step batch work queue for the oculus fix engine.

Owner spec (2026-09-14), which this module implements:

  1. BATCH = up to GROUP_CAP steps (default 15). Every lane works the SAME
     batch, in parallel, and every lane call handles exactly ONE step
     ("one step per lane call to prevent conflicts").
  2. A lane claims the next step that is (a) not done, (b) not in flight, and
     (c) file-compatible — none of the files it will edit is currently reserved
     by another lane ("reserved slots so files aren't being edited at the same
     time no matter what").
  3. NO lane advances to the next batch until the current batch is DONE.
     DONE = every step is terminal: `green`, or `yellow` — a pass carrying a
     MANDATORY written justification, flagged for later review. A step the
     execution lanes cannot fix goes to the escalation persona for ONE final
     shot, which either fixes it (green) or files it as code yellow.
  4. A lane COOLDOWN is NOT an escalation and does not close a step. The cooled
     lane is parked on a temporary timeout, its step returns to the queue, and
     the next available lane picks it up — handed the step, the task, and the
     path of the step's HISTORY LOG, which holds everything the previous lanes
     tried (edits AND reply text), injected in chunks when it is long.
  5. If nothing is claimable — every pending step has a reserved file, or none
     are pending — the lane SITS IDLE. It waits; it never jumps to a batch.

Pure logic: no network, no engine globals. The async driver takes
`execute_step` and `on_escalate` by injection, so the whole scheduler is
unit-testable with fakes (see tests/test_orch_queue.py).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

TERMINAL = frozenset({"green", "yellow"})
DEFAULT_GROUP_CAP = 15
MIN_YELLOW_JUSTIFICATION = int(os.environ.get("ORCH_MIN_YELLOW_JUSTIFICATION", "40"))
HANDOFF_DIR = Path(os.environ.get("ORCH_HANDOFF_DIR", "/tmp/orch_step_handoffs"))
HANDOFF_INJECT_CHARS = int(os.environ.get("ORCH_HANDOFF_INJECT_CHARS", "6000"))
DEFAULT_POLL_S = float(os.environ.get("ORCH_QUEUE_POLL_S", "2.0"))
_SAFE_SID = re.compile(r"[^A-Za-z0-9_.+#-]+")


class YellowJustificationError(ValueError):
    """A code yellow without a real, written justification is not a verdict."""


class BatchStalled(RuntimeError):
    """The batch barrier would never lift: fail loudly instead of hanging."""


def validate_justification(text: str) -> str:
    j = (text or "").strip()
    if len(j) < MIN_YELLOW_JUSTIFICATION:
        raise YellowJustificationError(
            "code yellow needs a justification of at least "
            f"{MIN_YELLOW_JUSTIFICATION} chars, got {len(j)}: {j[:80]!r}")
    return j


def make_yellow(rec: dict, justification: str, *, lane: str | None = None) -> dict:
    """Terminal PASS with a mandatory written justification (owner 09-14)."""
    j = validate_justification(justification)
    rec["status"] = "yellow"
    rec["yellow_justification"] = j
    rec["yellow_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    if lane:
        rec["yellow_by"] = lane
    rec["resolved_by"] = f"code yellow: {j[:200]}"
    return rec


def is_terminal(rec) -> bool:
    return (rec or {}).get("status") in TERMINAL


def step_files(step) -> list:
    """The files a step will touch, deduped, stable order."""
    if not step:
        return []
    raw = step.get("files")
    if raw is None:
        raw = step.get("targets")
    if isinstance(raw, str):
        raw = [raw]
    seen, out = set(), []
    for f in (raw or []):
        f = str(f or "").strip()
        if f and f not in seen:
            seen.add(f)
            out.append(f)
    return out


class FileReservations:
    """Reserved slots: exactly one holder per file path, ever.

    A reservation covers the whole step's file set and is released the moment
    the step leaves its lane — finished, retried, or handed off after a
    cooldown. Reserve is all-or-nothing: a partial reservation would let two
    lanes co-edit, which is the thing this class exists to prevent.
    """

    def __init__(self):
        self._by_file: dict = {}
        self._by_holder: dict = {}

    def conflicts(self, files, holder=None) -> list:
        return sorted({f for f in (files or [])
                       if f in self._by_file and self._by_file[f] != holder})

    def can_reserve(self, files, holder=None) -> bool:
        return not self.conflicts(files, holder)

    def reserve(self, files, holder) -> bool:
        files = [f for f in (files or []) if f]
        if not self.can_reserve(files, holder):
            return False
        for f in files:
            self._by_file[f] = holder
            self._by_holder.setdefault(holder, set()).add(f)
        return True

    def release(self, holder) -> int:
        freed = 0
        for f in self._by_holder.pop(holder, set()):
            if self._by_file.get(f) == holder:
                self._by_file.pop(f, None)
                freed += 1
        return freed

    def held_by(self, holder) -> set:
        return set(self._by_holder.get(holder, set()))

    def holders(self) -> dict:
        return dict(self._by_file)

    def locked_files(self) -> set:
        return set(self._by_file)


class LaneRoster:
    """Cooldown-aware view of the lane pool.

    A cooled lane is parked for a bounded time. It is not failed, not
    escalated, and not removed: an available sibling takes its step, and the
    parked lane rejoins the workforce when its timer expires.
    """

    def __init__(self, lanes):
        self._lanes = list(lanes)
        self._cool_until: dict = {}
        self._cool_count: dict = {}

    def lane_names(self) -> list:
        return list(self._lanes)

    def cool(self, lane, seconds) -> None:
        self._cool_until[lane] = time.time() + max(0.0, float(seconds))
        self._cool_count[lane] = self._cool_count.get(lane, 0) + 1

    def is_cooled(self, lane, now=None) -> bool:
        now = time.time() if now is None else now
        return self._cool_until.get(lane, 0) > now

    def cool_remaining(self, lane, now=None) -> float:
        now = time.time() if now is None else now
        return max(0.0, self._cool_until.get(lane, 0) - now)

    def available(self, now=None) -> list:
        return [l for l in self._lanes if not self.is_cooled(l, now)]

    def parked(self, now=None) -> list:
        return [l for l in self._lanes if self.is_cooled(l, now)]

    def soonest_wake(self, now=None) -> float:
        now = time.time() if now is None else now
        waits = [self._cool_until[l] - now for l in self._lanes
                 if self._cool_until.get(l, 0) > now]
        return min(waits) if waits else 0.0

    def cooldown_counts(self) -> dict:
        return dict(self._cool_count)


class StepHandoff:
    """Append-only per-step history log, written by every lane that touches a
    step and handed to its replacement.

    Owner 09-14: "both, and inject in chunks if necessary ... save its history
    to a log file, and tell the new replacement lane the step+task+etc and to
    read the log file for extra context of what occurred". So the log holds the
    attempted EDITS and the raw REPLY text, and the prompt carries the path.
    """

    def __init__(self, directory=None):
        self.dir = Path(directory or HANDOFF_DIR)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, sid) -> Path:
        safe = _SAFE_SID.sub("_", str(sid))[:120] or "step"
        return self.dir / f"{safe}.log"

    def record(self, sid, lane, outcome, *, note="", error=None,
               edits=None, reply=None, extra=None) -> Path:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "lane": lane,
            "outcome": outcome,
        }
        if note:
            entry["note"] = str(note)[:4000]
        if error:
            entry["error"] = str(error)[:4000]
        if edits:
            entry["edits"] = edits
        if reply:
            entry["reply"] = str(reply)[:20000]
        if extra:
            entry["extra"] = extra
        p = self.path(sid)
        with open(p, "a") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
        return p

    def entries(self, sid) -> list:
        p = self.path(sid)
        if not p.exists():
            return []
        out = []
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    out.append({"raw": line[:4000]})
        return out

    def exists(self, sid) -> bool:
        return self.path(sid).exists()

    def render(self, sid, max_chars=None) -> str:
        """The handoff prompt block: path first, then the tail of the history.

        Long logs are injected in chunks rather than dropped, so a replacement
        lane always gets the most recent context and is told where the rest is.
        """
        max_chars = HANDOFF_INJECT_CHARS if max_chars is None else max_chars
        p = self.path(sid)
        if not p.exists():
            return ""
        text = p.read_text()
        head = (f"PREVIOUS ATTEMPTS ON THIS STEP ARE LOGGED AT: {p}\n"
                f"Read that file for the full history (every lane, every attempt, "
                f"every edit and reply).\n")
        if len(text) <= max_chars:
            return head + "HISTORY SO FAR:\n" + text
        return (head + f"HISTORY SO FAR (last {max_chars} chars; read the file for the rest):\n"
                + text[-max_chars:])

    def clear(self, sid) -> None:
        p = self.path(sid)
        if p.exists():
            p.unlink()


class BatchQueue:
    """One batch: a step set, a shared claim queue, and a completion barrier.

    Status lives in the engine's own step records, which are the single source
    of truth; the queue reads them live and never keeps a second copy.
    """

    def __init__(self, step_ids, records, *, reservations=None, steps_by_id=None,
                 group_cap=DEFAULT_GROUP_CAP):
        ids = [str(s) for s in step_ids]
        if len(ids) > group_cap:
            raise ValueError(f"batch of {len(ids)} exceeds group_cap {group_cap}")
        self.order = ids
        self.records = records
        self.res = reservations if reservations is not None else FileReservations()
        self.steps_by_id = steps_by_id or {}
        self.inflight: dict = {}
        self._clip = group_cap

    def files(self, sid) -> list:
        return step_files(self.steps_by_id.get(sid))

    def record(self, sid) -> dict:
        return self.records.setdefault(sid, {"rounds": 0, "status": "pending"})

    def status(self, sid) -> str:
        return (self.records.get(sid) or {}).get("status") or "pending"

    def non_terminal(self) -> list:
        return [s for s in self.order if not is_terminal(self.records.get(s))]

    def terminal(self) -> list:
        return [s for s in self.order if is_terminal(self.records.get(s))]

    def is_done(self) -> bool:
        """The barrier: every step in the batch is green or yellow."""
        return not self.non_terminal()

    def pending(self) -> list:
        return [s for s in self.order
                if self.status(s) == "pending" and s not in self.inflight]

    def awaiting_escalation(self) -> list:
        return [s for s in self.order if self.status(s) == "escalated"]

    def has_inflight(self) -> bool:
        return bool(self.inflight)

    def claim(self, holder):
        """Next step this holder may legally take, or None (the lane idles).

        Skips steps already done, already in flight, or whose files another
        lane holds a reservation on.
        """
        for sid in self.order:
            if sid in self.inflight:
                continue
            if self.status(sid) != "pending":
                continue
            files = self.files(sid)
            if self.res.can_reserve(files, holder):
                self.res.reserve(files, holder)
                self.inflight[sid] = holder
                return sid
        return None

    def claimable_exists(self) -> bool:
        for sid in self.order:
            if sid in self.inflight or self.status(sid) != "pending":
                continue
            if self.res.can_reserve(self.files(sid)):
                return True
        return False

    def blocked_by_files(self) -> list:
        """Pending steps that exist but whose files are all reserved."""
        out = []
        for sid in self.order:
            if sid in self.inflight or self.status(sid) != "pending":
                continue
            if not self.res.can_reserve(self.files(sid), None):
                out.append(sid)
        return out

    def finish(self, sid, holder=None) -> None:
        h = self.inflight.pop(sid, holder)
        if h is not None:
            self.res.release(h)

    def handoff(self, sid) -> None:
        """Return a step to the queue so a replacement lane can take it.

        Releases the file reservations first — the step is not being edited
        while it sits in the queue.
        """
        h = self.inflight.pop(sid, None)
        if h is not None:
            self.res.release(h)

    def next_escalation(self):
        for sid in self.order:
            if sid in self.inflight:
                continue
            if self.status(sid) == "escalated":
                self.inflight[sid] = "__escalation__"
                return sid
        return None

    def claim_task(self, holder, prefer=("escalate", "execute")):
        """Next unit of work for this holder, as (sid, kind).

        ESCALATION WORK IS OFFERED FIRST. An escalated step is one whose card is
        already on the table, and closing it is what lets a batch finish, so it
        must not wait behind fresh work. `kind` is "escalate" (the escalation
        persona's final shot) or "execute" (a normal lane attempt). Both kinds
        take the step's file reservations, because the escalation persona may
        still apply edits.
        """
        for kind in prefer:
            for sid in self.order:
                if sid in self.inflight:
                    continue
                st = self.status(sid)
                if kind == "execute" and st != "pending":
                    continue
                if kind == "escalate" and st != "escalated":
                    continue
                if self.res.can_reserve(self.files(sid), holder):
                    self.res.reserve(self.files(sid), holder)
                    self.inflight[sid] = holder
                    return sid, kind
        return None

    def snapshot(self) -> dict:
        return {sid: self.status(sid) for sid in self.order}


@dataclass
class StepOutcome:
    """What a lane call did to a step.

    green    — fixed and verified
    yellow   — pass with justification (terminal)
    retry    — nothing terminal; the step stays claimable
    escalate — the execution lanes could not fix it; escalation persona's turn
    """
    outcome: str
    lane: str | None = None
    error: str | None = None
    cooled_s: int = 0
    note: str = ""
    edits: list = field(default_factory=list)
    reply: str = ""


def apply_outcome(records: dict, sid, res) -> str:
    """Translate a lane's StepOutcome into the step record's status."""
    rec = records.setdefault(sid, {"rounds": 0, "status": "pending"})
    o = str(res.outcome or "").strip().lower()
    if o == "green":
        rec["status"] = "green"
    elif o == "yellow":
        if not rec.get("yellow_justification"):
            raise YellowJustificationError(
                f"lane {res.lane} returned yellow for {sid} with no recorded "
                f"justification")
        rec["status"] = "yellow"
    elif o == "escalate":
        rec["status"] = "escalated"
    elif o == "retry":
        if rec.get("status") not in TERMINAL:
            rec["status"] = "pending"
    else:
        raise ValueError(f"unknown step outcome {o!r} for {sid}")
    return rec["status"]


def _apply_outcome(queue, sid, res) -> str:
    """Translate a lane's StepOutcome into the step's own status field.

    The queue reads status from the engine records, so the driver — not the
    caller — owns this transition. Without it a lane returning "escalate"
    without touching the record left the step "pending", the worker re-claimed
    it forever, and the batch barrier never lifted.
    """
    rec = queue.record(sid)
    o = str(res.outcome or "").strip().lower()
    if o == "green":
        rec["status"] = "green"
    elif o == "yellow":
        if not rec.get("yellow_justification"):
            raise YellowJustificationError(
                f"lane {res.lane} returned yellow for {sid} with no recorded "
                f"justification")
        rec["status"] = "yellow"
    elif o == "escalate":
        rec["status"] = "escalated"
    elif o == "retry":
        if rec.get("status") not in TERMINAL:
            rec["status"] = "pending"
    else:
        raise ValueError(f"unknown step outcome {o!r} for {sid}")
    return rec["status"]


async def drive_batch(queue: BatchQueue, roster: LaneRoster, handoff: StepHandoff,
                      execute_step, on_escalate, *, poll_s=None, log=None,
                      max_stall_rounds=120):
    """Drive ONE batch to completion. Returns when every step is terminal.

    Every available lane pulls steps from the shared queue; a lane that cools
    parks and its step is handed to a sibling with the history log; the
    escalation persona gets the last word on anything the lanes could not fix.
    Lanes sit idle when nothing is claimable — they never touch another batch.
    """
    poll_s = DEFAULT_POLL_S if poll_s is None else poll_s
    log = log or (lambda *a, **k: None)
    idle_logged: dict = {}
    stall = {"rounds": 0, "last_progress": queue.snapshot()}

    def _note(msg):
        log(msg)

    async def _idle(lane, why):
        key = f"{lane}:{why}"
        if idle_logged.get(lane) != why:
            _note(f"[queue] {lane} idle — {why}")
            idle_logged[lane] = why
        await asyncio.sleep(poll_s)

    async def worker(lane):
        while not queue.is_done():
            wait = roster.cool_remaining(lane)
            if wait > 0:
                await _idle(lane, f"parked on cooldown for {wait:.0f}s")
                continue
            sid = queue.claim(lane)
            if sid is None:
                if queue.claimable_exists():
                    await _idle(lane, "no file-compatible step free")
                elif queue.has_inflight():
                    await _idle(lane, "waiting on in-flight siblings")
                elif queue.awaiting_escalation():
                    await _idle(lane, "only escalation work left")
                else:
                    await asyncio.sleep(poll_s)
                continue
            idle_logged.pop(lane, None)
            _note(f"[queue] {lane} claimed {sid} (files={queue.files(sid)[:2]})")
            try:
                res = await execute_step(lane, sid)
            except Exception as e:                      # a lane crash is transport, not a verdict
                res = StepOutcome("retry", lane=lane,
                                  error=f"{type(e).__name__}: {e}")
            if not isinstance(res, StepOutcome):
                res = StepOutcome(str(res), lane=lane)
            res.lane = res.lane or lane
            handoff.record(sid, res.lane, res.outcome, note=res.note,
                           error=res.error, edits=res.edits, reply=res.reply)
            if res.cooled_s and res.cooled_s > 0:
                roster.cool(lane, res.cooled_s)
                queue.handoff(sid)
                _note(f"[queue] {lane} cooled {res.cooled_s}s — {sid} handed off "
                      f"with history {handoff.path(sid).name}")
                continue
            _apply_outcome(queue, sid, res)
            queue.finish(sid)
            _note(f"[queue] {sid} -> {queue.status(sid)} by {res.lane}")

    async def escalator():
        attempts: dict = {}
        while not queue.is_done():
            sid = queue.next_escalation()
            if sid is None:
                if queue.non_terminal():
                    await asyncio.sleep(poll_s)
                    continue
                break
            n = attempts.get(sid, 0) + 1
            attempts[sid] = n
            _note(f"[queue] escalation persona final shot on {sid} (attempt {n})")
            try:
                verdict = await on_escalate(sid)
            except Exception as e:
                _note(f"[queue] escalation of {sid} errored: {e}")
                queue.finish(sid)
                await asyncio.sleep(poll_s)
                continue
            rec = queue.record(sid)
            if verdict == "green":
                rec["status"] = "green"
            elif verdict == "yellow" and rec.get("status") != "yellow":
                raise YellowJustificationError(
                    f"escalation persona declared {sid} yellow with no recorded "
                    f"justification")
            queue.finish(sid)
            _note(f"[queue] {sid} -> {queue.status(sid)} (escalation persona)")
            if verdict not in TERMINAL:
                await asyncio.sleep(poll_s)

    # A batch that stops making progress is a bug, not patience: fail loudly.
    async def barrier():
        while not queue.is_done():
            await asyncio.sleep(poll_s)
            snap = queue.snapshot()
            if snap != stall["last_progress"]:
                stall["last_progress"] = snap
                stall["rounds"] = 0
                continue
            stall["rounds"] += 1
            if stall["rounds"] >= max_stall_rounds:
                raise BatchStalled(
                    f"batch made no progress for {max_stall_rounds} polls; "
                    f"stuck={queue.non_terminal()[:8]} "
                    f"parked={roster.parked()} locked={sorted(queue.res.locked_files())[:8]}")

    tasks = [asyncio.create_task(worker(l)) for l in roster.lane_names()]
    tasks.append(asyncio.create_task(escalator()))
    tasks.append(asyncio.create_task(barrier()))
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return queue.snapshot()


def pack_batches(step_ids, group_cap=DEFAULT_GROUP_CAP):
    """Split a step-id list into sequential batches of at most group_cap."""
    ids = [str(s) for s in step_ids]
    return [ids[i:i + group_cap] for i in range(0, len(ids), group_cap)] or [[]]


async def drive_plan(batches, records, roster, handoff, execute_step, on_escalate, *,
                     steps_by_id=None, poll_s=None, log=None, max_stall_rounds=600,
                     on_batch_done=None, max_escalation_attempts=3):
    """ONE global work queue that carries BOTH kinds of work.

    The owner's rule (09-14): "escalations are supposed to be resolved as soon as
    they occur, also add escalations to per step queues". So there is no separate
    escalator sitting on the side. A lane that finishes a step pulls the next
    task from the same queue, and the queue offers ESCALATION work first: the
    moment a step lands in `escalated` it becomes claimable by whichever lane is
    next free, and that lane runs the escalation persona's one final shot.

    Safety is unchanged — one reservations table spans the whole plan, and both
    kinds of task take the step's file reservations, because the escalation
    persona may apply edits of its own.

    A lane is never idle while any task is claimable, and the batch stays an
    accounting boundary: `on_batch_done` fires when every step of a batch is
    terminal.
    """
    poll_s = DEFAULT_POLL_S if poll_s is None else poll_s
    log = log or (lambda *a, **k: None)
    steps_by_id = steps_by_id or {}
    res = FileReservations()

    order, batch_of = [], {}
    members, live_batches = {}, []
    for bi, ids in enumerate(batches):
        ids = [str(x) for x in ids]
        if not any(not is_terminal(records.get(s)) for s in ids):
            continue
        live_batches.append(bi)
        members[bi] = list(ids)
        for sid in ids:
            order.append(sid)
            batch_of[sid] = bi
    if not order:
        return {}

    def status(sid):
        return (records.get(sid) or {}).get("status") or "pending"

    inflight: dict = {}
    esc_attempts: dict = {}

    def claim(holder):
        for kind in ("escalate", "execute"):
            for sid in order:
                if sid in inflight or parked(sid):
                    continue
                st = status(sid)
                if kind == "execute" and st != "pending":
                    continue
                if kind == "escalate" and st != "escalated":
                    continue
                fs = step_files(steps_by_id.get(sid))
                if res.can_reserve(fs, holder):
                    res.reserve(fs, holder)
                    inflight[sid] = holder
                    return sid, kind
        return None

    def release(sid):
        h = inflight.pop(sid, None)
        if h:
            res.release(h)

    def parked(sid):
        """A step whose escalation spent its attempt budget is NOT re-claimable.

        Without this the queue re-offered it on every pass: the escalation
        persona answered nothing, the step stayed non-terminal, the worker
        claimed it again forever and the batch barrier never lifted (measured:
        the give-up test hung until its timeout). Parked steps are excluded from
        claiming and from the barrier, but they keep their `escalated` status so
        the solver and a human can still see them.
        """
        return bool((records.get(sid) or {}).get("escalation_gave_up"))

    def all_done():
        return all(is_terminal(records.get(s)) or parked(s) for s in order)

    reported: set = set()

    def report_batches():
        for bi in live_batches:
            if bi in reported:
                continue
            if all(is_terminal(records.get(s)) for s in members[bi]):
                reported.add(bi)
                log(f"[plan] batch {bi} complete ({len(members[bi])} steps)")
                if on_batch_done:
                    try:
                        on_batch_done(bi, {s: status(s) for s in members[bi]})
                    except Exception as e:
                        log(f"[plan] on_batch_done({bi}) raised: {e}")

    async def run_escalation(lane, sid):
        """The escalation persona's one final shot, ON this lane, right now."""
        n = esc_attempts.get(sid, 0) + 1
        esc_attempts[sid] = n
        log(f"[plan] {lane} -> ESCALATION final shot on {sid} "
            f"(attempt {n}/{max_escalation_attempts})")
        try:
            verdict = await on_escalate(lane, sid)
        except Exception as e:
            verdict = "escalated"
            log(f"[plan] escalation of {sid} on {lane} errored: {e}")
        rec = records.setdefault(sid, {"rounds": 0, "status": "escalated"})
        if verdict == "green":
            rec["status"] = "green"
            rec["resolved_by"] = "escalation persona: fixed on the final shot"
        elif verdict == "yellow":
            if rec.get("status") != "yellow":
                raise YellowJustificationError(
                    f"escalation persona declared {sid} yellow with no recorded "
                    f"justification")
        else:
            # No verdict — usually the lane itself was unreachable. Keep the step
            # claimable so the NEXT free lane can try, and only give up once the
            # attempt budget is spent, saying so out loud.
            rec["escalation_attempts"] = n
            if n >= max_escalation_attempts:
                rec["status"] = "escalated"
                rec["escalation_gave_up"] = (
                    f"no verdict from {n} lane attempt(s); last lane {lane}")
                log(f"[plan] {sid} escalation gave up after {n} attempt(s) — "
                    f"left escalated (needs the solver/human)")
            else:
                rec["status"] = "escalated"

    async def worker(lane):
        while not all_done():
            wait = roster.cool_remaining(lane)
            if wait > 0:
                await asyncio.sleep(min(wait, poll_s))
                continue
            task = claim(lane)
            if task is None:
                await asyncio.sleep(poll_s)
                continue
            sid, kind = task
            if kind == "escalate":
                await run_escalation(lane, sid)
                release(sid)
                report_batches()
                log(f"[plan] {sid} -> {status(sid)} (escalation, {lane})")
                continue
            log(f"[plan] {lane} claimed {sid} (batch {batch_of[sid]})")
            try:
                out = await execute_step(lane, sid)
            except Exception as e:
                out = StepOutcome("retry", lane=lane, error=f"{type(e).__name__}: {e}")
            if not isinstance(out, StepOutcome):
                out = StepOutcome(str(out), lane=lane)
            out.lane = out.lane or lane
            handoff.record(sid, out.lane, out.outcome, note=out.note,
                           error=out.error, edits=out.edits, reply=out.reply)
            if out.cooled_s and out.cooled_s > 0:
                roster.cool(lane, out.cooled_s)
                release(sid)
                log(f"[plan] {lane} cooled {out.cooled_s}s — {sid} handed off "
                    f"with history {handoff.path(sid).name}")
                continue
            apply_outcome(records, sid, out)
            if status(sid) == "escalated":
                # In the queue as escalation work from the very next claim, so a
                # free lane resolves it immediately instead of at a restart.
                esc_attempts.pop(sid, None)
            release(sid)
            report_batches()
            log(f"[plan] {sid} -> {status(sid)} by {out.lane}")

    def snapshot():
        return {s: status(s) for s in order}

    stall = {"snap": snapshot(), "rounds": 0}

    async def watch():
        while not all_done():
            await asyncio.sleep(poll_s)
            snap = snapshot()
            if snap != stall["snap"]:
                stall["snap"] = snap
                stall["rounds"] = 0
                continue
            stall["rounds"] += 1
            if stall["rounds"] >= max_stall_rounds:
                stuck = [s for s in order
                         if not is_terminal(records.get(s)) and not parked(s)]
                raise BatchStalled(
                    f"plan made no progress for {max_stall_rounds} polls; "
                    f"stuck={stuck[:8]} parked={roster.parked()} "
                    f"locked={sorted(res.locked_files())[:8]}")

    tasks = [asyncio.create_task(worker(l)) for l in roster.lane_names()]
    tasks.append(asyncio.create_task(watch()))
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    report_batches()
    return snapshot()
