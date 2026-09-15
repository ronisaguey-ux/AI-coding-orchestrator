#!/usr/bin/env python3
"""
plan_postprocess.py — deterministic rebuild of the OXA cross-eval plan
(from plan_state.json cached per-finding plans) with:
  - correct severity ranking (UPPER/lower case-insensitive)
  - dependency-ish ordering: area (server/web/scripts/deploy/tests) +
    severity priority, server-HIGH first
  - batches of 15 with NO two steps in the same batch touching the same file
    (parallel-apply safety) — duplicates are deferred to later batches
  - full coverage check (no step dropped)

Writes webchat_audit_8_26/oculus_cross_eval_plan.json.
"""
import json
import os
import sys
import time
from pathlib import Path

BASE = Path("/home/roni/Roni_Workspace/audits_plans/webchat_audit_8_26")
PSTATE = BASE / "plan_state.json"
PLAN = BASE / "oculus_cross_eval_plan.json"

# ── Event bus (P1B0R0F1#196) ─────────────────────────────────────────────────
# Batch postprocessing used to require a timer that re-read the plan artifact.
# We now publish plan lifecycle events over Redis pub/sub so consumers
# (webchat server, orchestrator, agent supervisor) react the moment a plan is
# rebuilt instead of polling PLAN on an interval. The batch artifact remains
# the source of truth; the bus is a best-effort acceleration layer.
BROKER_URL_ENV = "OCULUS_PLAN_BROKER_URL"
BROKER_CHANNEL_ENV = "OCULUS_PLAN_BROKER_CHANNEL"
DEFAULT_BROKER_CHANNEL = "oculus:plan:events"


def _broker_url():
    return os.environ.get(BROKER_URL_ENV) or os.environ.get("REDIS_URL")


def _broker_channel():
    return os.environ.get(BROKER_CHANNEL_ENV, DEFAULT_BROKER_CHANNEL)


class EventBus:
    """Redis pub/sub event bus for plan lifecycle events.

    Producers (this postprocessor, agent step runners) call ``publish``;
    consumers call ``subscribe``. If ``redis`` is not installed or the broker
    is unreachable, ``available`` stays False and ``publish`` is a no-op so
    the caller silently keeps the batch artifact flow.
    """

    def __init__(self, url=None, channel=None):
        self.url = url or _broker_url()
        self.channel = channel or _broker_channel()
        self._client = None
        self.available = False
        if self.url:
            self._connect()

    def _connect(self):
        try:
            import redis  # type: ignore
        except ImportError:
            return
        try:
            self._client = redis.Redis.from_url(self.url, decode_responses=True)
            self._client.ping()
            self.available = True
        except Exception:
            self._client = None
            self.available = False

    def publish(self, event_type, payload=None):
        if not self.available or self._client is None:
            return False
        event = {"type": event_type, "ts": time.time(), "payload": payload or {}}
        try:
            self._client.publish(self.channel, json.dumps(event))
            return True
        except Exception:
            self.available = False
            return False


def watch_plan_state(bus=None):
    """Watch plan_state.json and publish an event on every change.

    Requires the optional ``watchdog`` package; returns the observer so the
    caller can ``join()``, or None if unavailable (caller should fall back to
    a one-shot ``main()`` run).
    """
    bus = bus or EventBus()
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print("watchdog not installed; event-driven watch disabled", flush=True)
        return None

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if str(event.src_path).endswith("plan_state.json"):
                bus.publish("plan_state_changed", {"file": str(PSTATE)})

    observer = Observer()
    observer.schedule(_Handler(), str(BASE), recursive=False)
    observer.start()
    return observer


def subscribe(bus=None, handler=None):
    """Blocking pub/sub consumer; ``handler(event_dict)`` runs per event.

    When ``handler`` is None each event is printed as a JSON line so the
    process can be piped into another orchestrator.
    """
    bus = bus or EventBus()
    if not bus.available:
        print("broker unavailable; cannot subscribe", flush=True)
        return
    pubsub = bus._client.pubsub(ignore_subscribe_messages=True)
    pubsub.subscribe(bus.channel)
    for msg in pubsub.listen():
        if msg.get("type") != "message":
            continue
        try:
            event = json.loads(msg["data"])
        except (ValueError, TypeError):
            continue
        if handler is not None:
            handler(event)
        else:
            print(json.dumps(event), flush=True)

PRIORITY = [
    ("server", "high"), ("server", "medium"), ("server", "low"),
    ("web", "high"), ("web", "medium"), ("web", "low"),
    ("scripts", "high"), ("scripts", "medium"),
    ("deploy", "high"), ("deploy", "medium"), ("deploy", "low"),
    ("tests", "high"), ("tests", "medium"), ("tests", "low"),
    ("scripts", "low"),
]

SEVERITY_MAP = {
    "high": "high", "medium": "medium", "low": "low",
    "HIGH": "high", "MEDIUM": "medium", "LOW": "low",
    "High": "high", "Medium": "medium", "Low": "low",
}


def normalize_severity(sev) -> str:
    """Return a canonical severity key ('high'/'medium'/'low') from any input.

    Uses an explicit mapping instead of string slicing so 'HIGH'/'High' are
    never corrupted to 'hig' and silently downgraded to the default priority.
    """
    sev = str(sev or "MEDIUM").strip()
    return SEVERITY_MAP.get(sev, SEVERITY_MAP.get(sev.lower(), "medium"))


def area_of(rel: str) -> str:
    rel = str(rel or "").lower()
    if any(x in rel for x in ("server/", "app/", "api", "main.py", "security", "routes")):
        return "server" if "test" not in rel else "tests"
    if any(x in rel for x in ("web/", "src/", ".jsx", ".tsx", ".vue")):
        return "web"
    if "test" in rel:
        return "tests"
    if "script" in rel:
        return "scripts"
    if any(x in rel for x in ("deploy", "docker", "compose", "ci", "k8s")):
        return "deploy"
    return "server"


def _validate_plan_state(pstate) -> None:
    """Fail loudly on an unexpected plan_state.json shape (P1B1R0F1#79).

    Without this a malformed plan_state silently falls back to defaults,
    dropping or reordering SoT restoration steps while the coverage check
    (which only sums batch sizes) still reports full coverage.
    """
    if not isinstance(pstate, dict):
        raise ValueError(f"plan_state.json root must be an object, got {type(pstate).__name__}")
    if "done" not in pstate:
        raise ValueError("plan_state.json missing required 'done' mapping")
    done = pstate["done"]
    if not isinstance(done, dict):
        raise ValueError(f"plan_state.json 'done' must be an object, got {type(done).__name__}")
    for iid, rec in done.items():
        if not isinstance(rec, dict):
            raise ValueError(f"plan_state 'done[{iid}]' must be an object, got {type(rec).__name__}")
        for key in ("item", "plan"):
            if key in rec and rec[key] is not None and not isinstance(rec[key], dict):
                raise ValueError(f"plan_state 'done[{iid}].{key}' must be object or null")


def main():
    pstate = json.loads(PSTATE.read_text())
    _validate_plan_state(pstate)
    steps = []
    for iid, rec in pstate["done"].items():
        it = rec.get("item") or {}
        plan = rec.get("plan") or {}
        sev = normalize_severity(it.get("severity"))
        steps.append({
            "step_id": f"S-{iid}",
            "title": plan.get("step_title") or str(it.get("finding", ""))[:120],
            "file": str(it.get("file") or ""),
            "area": area_of(it.get("file")),
            "severity": sev,
            "test_hint": plan.get("test_hint") or "pytest / web build",
            "changes": plan.get("changes") or "",
            "depends_on": plan.get("depends_on") or "none",
        })
    rank = {k: i for i, k in enumerate(PRIORITY)}
    steps.sort(key=lambda s: (rank.get((s["area"], s["severity"]), 99), s["step_id"]))

    # Greedy distinct-file batching: fill each batch with up to 15 steps whose
    # files are all distinct; unique-file steps that can't fit are deferred.
    batches, pool = [], steps[:]
    while pool:
        batch, used_files = [], set()
        remaining = []
        for st in pool:
            f = st["file"]
            if len(batch) < 15 and f not in used_files:
                batch.append(st)
                if f:
                    used_files.add(f)
            else:
                remaining.append(st)
        if not batch:  # all remaining share one file — put one per batch
            batch = [remaining.pop(0)]
        batches.append({"batch_id": f"BATCH-{len(batches) + 1:02d}",
                        "purpose": f"{len(batch)} parallel steps (area={batch[0]['area']})",
                        "steps": [{k: st[k] for k in ("step_id", "title", "file", "area",
                                                      "severity", "test_hint", "depends_on")}
                                  for st in batch]})
        pool = remaining

    out = {
        "meta": {"model": "stealth/ox-alpha", "source": "postprocess-deterministic",
                 "steps_needed": len(steps), "batches": len(batches),
                 "generated_at_epoch": int(__import__("time").time())},
        "batches": batches,
    }
    PLAN.write_text(json.dumps(out, indent=1))
    covered = sum(len(b["steps"]) for b in batches)
    sizes = [len(b["steps"]) for b in batches]
    conflicted = sum(1 for b in batches
                     if len({s["file"] for s in b["steps"]}) < len(b["steps"]))
    print(f"PLAN_FINAL batches={len(batches)} covered={covered}/{len(steps)} "
          f"conflicts={conflicted} sizes={sizes[:10]}...", flush=True)

    # Event-driven notification: fan the rebuild out to any subscribed
    # consumer so downstream postprocessing starts immediately instead of
    # waiting for the next batch poll of PLAN.
    EventBus().publish("plan_rebuilt", {
        "batches": len(batches),
        "covered": covered,
        "steps_needed": len(steps),
        "conflicts": conflicted,
        "plan": str(PLAN),
    })


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--subscribe":
        subscribe()
    elif len(sys.argv) > 1 and sys.argv[1] == "--watch":
        observer = watch_plan_state()
        if observer is not None:
            try:
                observer.join()
            except KeyboardInterrupt:
                observer.stop()
                observer.join()
        else:
            main()
    else:
        main()
