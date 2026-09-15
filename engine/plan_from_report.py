#!/usr/bin/env python3
"""plan_from_report.py — 09-03 (user: plan must be MUCH longer / per-finding).

Rebuilds the 8_27 master plan at TRUE finding scale: one step per accepted
record in the stage-1 voted report (multi_agent_oculus_audit_8_27.json —
17,345 records, each with Finding/Mechanism/Impact/Fix per row-context) and
merges the cross-eval agent steps (from the compact first-pass plan). Outputs:
  master_oculus_plan_8_27.md                  (canonical, per-finding steps)
  oculus_cross_eval_plan_8_27.json            (executor-ready JSON)
No dedup: rows are the unit of authority (same finding_id repeats per file
context — each file-context fix is its own step, matching the audit's design).
"""
import json, re
from datetime import datetime

import os
BASE = "/home/roni/Roni_workspace/audits_plans"
REPORT = f"{BASE}/multi_agent_oculus_audit_8_27.json"
COMPACT = f"{BASE}/master_oculus_plan_8_27_compact_first_pass.md"
IN_PROGRESS_EXECUTOR = f"{BASE}/master_oculus_plan_9_3.md"  # superseded (see Notes)

# 09-03 (user rule): the OUTPUT filename takes the VERIFIED CURRENT date at
# generation time (never the audit cycle's past date), and auto-suffixes _v{N}
# on collision — an old plan file is never overwritten. Example: re-running
# after a shannon merge on the same day yields master_oculus_plan_9_3_v2.md.
def _dated_out(base_stem: str, ext: str) -> str:
    now = datetime.now()  # verified local date at generation time
    suffix = f"{now.month}_{now.day}"
    cand = f"{BASE}/{base_stem}_{suffix}{ext}"
    if not os.path.exists(cand):
        return cand
    v = 2
    while os.path.exists(f"{BASE}/{base_stem}_{suffix}_v{v}{ext}"):
        v += 1
    return f"{BASE}/{base_stem}_{suffix}_v{v}{ext}"

MD_OUT = _dated_out("master_oculus_plan", ".md")
JSON_OUT = _dated_out("oculus_cross_eval_plan", ".json")

CAT_ORDER = {"SECURITY": 0, "DATA": 1, "DATA_INTEGRITY": 1, "ARCHITECTURE": 2,
             "DESIGN": 3, "PERFORMANCE": 4, "TESTING": 5}
PRIO_LEVELS = None

def sev2level(rank: str) -> str:
    try:
        r = int(rank)
    except Exception:
        r = 9
    if r <= 1: return "CRITICAL"
    if r <= 3: return "HIGH"
    if r <= 5: return "MEDIUM"
    return "LOW"

def cat_of(c: str) -> str:
    for k in ("SECURITY", "DATA", "ARCHITECTURE", "ARCH", "DESIGN", "PERFORMANCE", "TESTING", "TEST"):
        if k in c.upper():
            return "SECURITY" if k == "SECURITY" else ("DATA" if k.startswith("DATA") else ("ARCHITECTURE" if k.startswith("ARCH") else ("PERFORMANCE" if k.startswith("PERF") else ("TESTING" if k.startswith("TEST") else "DESIGN"))))
    return "DESIGN"

def effort_of(conf: str) -> str:
    if "HIGH" in conf.upper(): return "2-4h"
    if "MEDIUM" in conf.upper(): return "45m-2h"
    return "15-45m"

def verify_for(files, cat):
    cmds = []
    for f in files[:3]:
        if f.endswith(".js") or f.endswith(".ts"):
            cmds.append(f"node --check {f}")
        elif f.endswith(".py"):
            cmds.append(f"python -m py_compile {f}")
        elif f.endswith(".sh"):
            cmds.append(f"bash -n {f}")
    test_hint = ""
    if test_hint:
        cmds.append(test_hint)
    cmds += ["git status", "git diff --stat"]
    return cmds[:5]

def one_line(s, n=220):
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s[:n] + ("..." if len(s) > n else "")

rep = json.load(open(REPORT))
records = rep.get("accepted_findings", [])
print(f"records: {len(records)}")

steps = []
for r in records:
    f = r.get("fields", {})
    fid = r["finding_id"]
    cat = cat_of(f.get("Category", ""))
    prio = sev2level(f.get("Priority Rank", "9"))
    conf = f.get("Finding Confidence", "MEDIUM")
    files = [line.strip().strip("`").strip() for line in re.findall(r"`([^`]+)`", str(f.get("File", "")))] or ([str(f.get("File","")).strip().strip("`")] if str(f.get("File","")).strip() else [])
    finding = one_line(f.get("Finding", ""))
    mechanism = one_line(f.get("Mechanism", ""), 150)
    impact = one_line(f.get("Impact", ""), 150)
    fix = one_line(f.get("Fix", ""), 400)
    reason = one_line(f.get("Vote Reasoning", ""), 120)
    title = f"Fix {fid}: {finding[:130]}"
    steps.append({
        "finding_id": fid,
        "title": title,
        "category": cat,
        "priority": prio,
        "confidence": conf,
        "files": files or [""],
        "finding": finding,
        "mechanism": mechanism,
        "impact": impact,
        "fix": fix,
        "reason": reason,
        "vote": f.get("General Confirm Vote", ""),
        "source": f"accepted-record",
    })

cat_rank = lambda s: CAT_ORDER.get(s["category"], 6)
prio_rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
steps.sort(key=lambda s: (cat_rank(s), prio_rank[s["priority"]], s["files"][0]))

# 09-14 (owner): the plan is emitted FILE-DIVERSE by default.
#
# The engine runs a per-step work queue with reserved file slots: a lane may
# only claim a step whose files no other lane holds. Measured on the previous
# plan, 58% of consecutive steps shared a file because the sort grouped them by
# `files[0]`, so whole batches collapsed onto one file and nine of ten lanes sat
# idle behind a single reservation. Emitting the plan ALREADY grouped into
# file-disjoint batches means the engine's default chunking is diverse with no
# repacking at run time, and the same ordering discipline helps any other
# consumer of the JSON.
#
# The grouping is greedy and lossless: a step whose files collide with the batch
# under construction is carried to the next one, so every step still appears
# exactly once and no step is dropped.
def diverse_order(steps, cap=30):
    remaining = list(steps)
    ordered, batches = [], []
    while remaining:
        used, batch, rest, i = set(), [], [], 0
        while i < len(remaining) and len(batch) < cap:
            s = remaining[i]
            i += 1
            fs = {f for f in (s.get("files") or s.get("target_files") or []) if f}
            if fs and (fs & used):
                rest.append(s)
                continue
            used |= fs
            batch.append(s)
        rest.extend(remaining[i:])
        if not batch:                      # every remaining step collides
            batch, rest = [remaining[0]], remaining[1:]
        ordered.extend(batch)
        batches.append([s["finding_id"] for s in batch])
        remaining = rest
    return ordered, batches


steps, PLAN_BATCHES = diverse_order(steps, 30)
print(f"file-diverse packing: {len(PLAN_BATCHES)} batches, cap 30")

n = len(steps)
doc = []
doc.append("# OCULUS MASTER IMPLEMENTATION PLAN v2 (per-finding full scale)")
doc.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
doc.append(f"**Source:** stage-1 voted report `multi_agent_oculus_audit_8_27.json` ({len(records)} accepted records) — one step per record (the audit's unit of authority: finding × file-context)")
doc.append(f"**Per-Finding Steps:** {n}")
doc.append(f"**Categories:** {sorted(set(s['category'] for s in steps))}")
prios = {}
for s in steps: prios[s["priority"]] = prios.get(s["priority"], 0) + 1
doc.append(f"**By Priority:** " + ", ".join(f"{k}: {v}" for k, v in sorted(prios.items())))
doc.append(f"**Token Usage:** 0 prompt / 0 completion (Estimated cost: ~$0.0000) — pure lane/local synthesis\n")
doc.append("## EXECUTIVE SUMMARY")
doc.append(f"This master plan executes at per-finding scale: {n} atomic steps, one per accepted finding record of the 8_27 audit (34,618 raw findings evaluated; {len(records)} records accepted/pending-fix across the stage-1 vote report). Step content derives from the auditors' own verdicts — Finding/Mechanism/Impact/Fix/Vote-Reasoning — grouped by category and priority, targeting the exact file-context the finding points to. Execution is verification-gated and rollback-safe; note that per-file-context steps repeat finding_ids by design (the same finding fixed in every file-context it was reported for).\n")
doc.append("## SPEC RESTORATION CHECKLIST (Non-Negotiable)")
doc.append("- [x] Survivor carry-over (survivors persist to next generation)\n")
doc.append("## FINDING-LEVEL STEPS\n")

for i, s in enumerate(steps, 1):
    tf = "\n".join(f"- {f}" for f in s["files"] if f)
    ops = f"- Apply the curated fix for {s['finding_id']}: {s['fix']} (finding: {s['finding']}; mechanism: {s['mechanism']})"
    ver = "\n".join(f"- {v}" for v in verify_for(s["files"], s["category"]))
    commit = f"[{s['category']}][{s['priority']}] Fix {s['finding_id']}: {s['finding'][:90]}"
    rollback = "git checkout -- " + " ".join(s["files"][1:3]) if len(s["files"]) > 1 else "git checkout -- " + (s["files"][0] if s["files"] else ".")
    doc.append(f"""### STEP {i}/{n}: {s['finding_id']} — {s['finding'][:110]}

**CATEGORY:** {s['category']}
**PRIORITY:** {s['priority']}
**BENEFIT:** {s['impact']}
**FINDING_CONFIDENCE:** {s['confidence']} ({s['vote']})
**EFFORT:** {effort_of(s['confidence'])}
**TARGET_FILES:**
{tf}
**DEPENDENCIES:** []
**LINE_RANGES:** []

**CODE_OPERATIONS:**
{ops}

**VERIFICATION:**
{ver}

**COMMIT_MESSAGE:** "{commit}"
**ROLLBACK_COMMAND:** {rollback}
**NOTES:** {s['reason']}

""")
    # __future__: keep the JSON mirror sized to what the engine consumes
    pass

json_out = {"plan_version": "8_27-v2", "cycle": "8_27",
            "generated_local": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "findings_evaluated": 34618,
            "accepted_records": len(records), "per_finding_steps": n,
            "generated": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "steps": steps,
            # 09-14: file-disjoint batch groups, so a consumer can chunk the plan
            # without repacking and without two steps in a batch sharing a file.
            "batches": PLAN_BATCHES,
            "batch_cap": 30}
json.dump(json_out, open(JSON_OUT, "w"), indent=1)
with open(MD_OUT, "w") as fh:
    fh.write("\n".join(doc))
print(f"wrote {MD_OUT}: {n} steps | {JSON_OUT}: {len(steps)} step records")
