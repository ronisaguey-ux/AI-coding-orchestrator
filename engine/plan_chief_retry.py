#!/usr/bin/env python3
"""
plan_chief_retry.py — re-run the chief batching with a bigger max_tokens budget.
The runner's chief call truncated at MAX_TOKENS (12000) for 533 steps; this
rebuilds steps_brief from the cached plan_state.json, queries OXA with
max_tokens 60000, and writes oculus_cross_eval_plan.json only if the result
has >= 2 batches covering ALL steps; else deterministic fallback batching.
"""
import json, re, sys
from pathlib import Path

BASE = Path("/home/roni/Roni_Workspace/audits_plans/webchat_audit_8_26")
PSTATE = BASE / "plan_state.json"
PLAN = BASE / "oculus_cross_eval_plan.json"
TOKEN_FILE = Path("/home/roni/.claude/openrouter.token")
MODEL = "stealth/ox-alpha"

CHIEF_SYSTEM = (
    "You are the PLAN CHIEF for an Oculus code-remediation run. You are given ALL fix steps "
    "(each with title/file/area/severity/test_hint/depends_on). Emit ONE JSON object only: "
    "{\"batches\":[{\"batch_id\":\"BATCH-01\",\"purpose\":\"<one line>\",\"steps\":[{\"step_id\":\"S-xx\","
    "\"title\":\"...\",\"files\":[\"...\"],\"area\":\"server|web|scripts|deploy|tests\","
    "\"severity\":\"high|medium|low\",\"test_hint\":\"...\",\"depends_on\":\"S-xx|none\"}]}],"
    "\"order\":\"BATCH-01,...\"} Rules: 8-15 steps per batch; order batches by dependency+area "
    "(security/server core first, then web, then scripts/deploy, then tests); HIGH severity first "
    "within area; a step may only depend on steps in the SAME or an EARLIER batch; no step in batch "
    "N may write a file also written by a step in the same batch unless they are the same step. "
    "COVER EVERY STEP — none may be dropped. Output ONLY the JSON object."
)


def area_of(rel: str) -> str:
    rel = str(rel or "").lower()
    if any(x in rel for x in ("server/", "app/", "api", "main.py", "security", "routes")):
        return "server"
    if any(x in rel for x in ("web/", "src/", ".jsx", ".tsx", ".vue")):
        return "web"
    if "script" in rel:
        return "scripts"
    if any(x in rel for x in ("deploy", "docker", "compose", "ci", "k8s")):
        return "deploy"
    if "test" in rel:
        return "tests"
    return "server"


def build_steps_brief() -> list:
    pstate = json.loads(PSTATE.read_text())
    items = []
    for iid, rec in pstate["done"].items():
        it = rec.get("item") or {}
        plan = rec.get("plan") or {}
        items.append({"step_id": f"S-{iid}", "title": plan.get("step_title") or it.get("finding", "")[:120],
                      "file": it.get("file"), "area": area_of(it.get("file")),
                      "severity": it.get("severity", "medium"),
                      "test_hint": plan.get("test_hint") or "pytest / web build",
                      "changes": plan.get("changes") or "",
                      "depends_on": plan.get("depends_on") or "none"})
    return sorted(items, key=lambda x: (x["step_id"],))


def chief_call(steps_brief: list) -> dict | None:
    import asyncio, urllib.request

    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "system", "content": CHIEF_SYSTEM},
                     {"role": "user", "content": "## STEPS\n" + json.dumps(steps_brief)}],
        "max_tokens": 60000,
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {TOKEN_FILE.read_text().strip()}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            text = json.loads(r.read())["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"chief call failed: {e}", flush=True)
        return None
    try:
        d = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
        return d if isinstance(d.get("batches"), list) else None
    except Exception:
        return None


def deterministic(steps_brief: list) -> dict:
    order = [("server", "high"), ("server", "medium"), ("server", "low"),
             ("web", "high"), ("web", "medium"), ("web", "low"),
             ("scripts", "high"), ("scripts", "medium"),
             ("deploy", "high"), ("deploy", "medium"), ("deploy", "low"),
             ("tests", "high"), ("tests", "medium"), ("tests", "low"),
             ("scripts", "low"), ("tests", "low")]
    srank = {k: i for i, k in enumerate(order)}
    steps = sorted(steps_brief, key=lambda x: (srank.get((x["area"], x["severity"]), 99),
                                               x["step_id"]))
    batches, cur = [], []
    for st in steps:
        cur.append({k: st[k] for k in ("step_id", "title", "file", "area", "severity",
                                       "test_hint", "depends_on")})
        if len(cur) == 15:
            batches.append({"batch_id": f"BATCH-{len(batches)+1:02d}", "steps": cur})
            cur = []
    if cur:
        batches.append({"batch_id": f"BATCH-{len(batches)+1:02d}", "steps": cur})
    return {"batches": batches, "order": ",".join(b["batch_id"] for b in batches)}


def main():
    steps_brief = build_steps_brief()
    total = len(steps_brief)
    print(f"steps: {total}", flush=True)
    expected = {st["step_id"] for st in steps_brief}
    plan = chief_call(steps_brief)
    source = "chief"
    plan_ids = set()
    if plan is not None:
        for b in plan.get("batches") or []:
            for st in b.get("steps") or []:
                plan_ids.add(st.get("step_id"))
    ok = (
        plan is not None
        and len(plan.get("batches") or []) >= 2
        and plan_ids == expected
    )
    if not ok:
        print("chief parse failed or incomplete — deterministic fallback", flush=True)
        plan = deterministic(steps_brief)
        source = "fallback"
    n_batches = len(plan["batches"])
    sizes = [len(b.get("steps") or []) for b in plan["batches"]]
    PLAN.write_text(json.dumps({
        "meta": {"model": MODEL, "steps_needed": total, "source": source,
                 "batch_sizes": sizes, "batches": n_batches,
                 "generated_at_epoch": int(__import__("time").time())},
        **plan}, indent=1))
    print(f"PLAN_READY {n_batches} batches / {total} steps (sizes {sizes})", flush=True)


if __name__ == "__main__":
    main()
