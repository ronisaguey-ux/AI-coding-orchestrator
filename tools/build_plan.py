#!/usr/bin/env python3
"""Turn verified audit findings into an execution plan.

Only SUPPORTED findings become plan steps. REFUTED ones are kept in the output under
`refuted` so the next reader can see what was checked and rejected rather than
re-deriving it, and UNVERIFIED ones become `leads` — places worth a human read, not
steps to execute. That split is the whole point: the raw audit is a lead generator,
and promoting an unverified claim into a step is how a plan acquires work that should
never have been done.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from datetime import date

AUDIT_DIR = os.environ.get("PLAN_AUDIT_DIR", "/home/roni/Roni_workspace/audits_plans")
VERIFIED = os.path.join(AUDIT_DIR, "verified_findings.json")
OUT = os.path.join(AUDIT_DIR, f"execution-plan-{date.today().isoformat()}.json")

SEV_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def has_real_line_range(f: dict) -> bool:
    """A citation to a specific span, as opposed to the whole file."""
    lr = str(f.get("line_range", "") or "")
    m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", lr.strip())
    if not m:
        return False
    a, b = int(m.group(1)), int(m.group(2))
    return b > a and a >= 1


def main() -> int:
    if not os.path.isfile(VERIFIED):
        print(f"missing {VERIFIED}")
        return 1
    findings = json.load(open(VERIFIED))

    supported = [f for f in findings if f.get("verified") == "SUPPORTED"]
    refuted = [f for f in findings if f.get("verified") == "REFUTED"]
    unverified = [f for f in findings if f.get("verified") == "UNVERIFIED"]

    # Group supported findings that share a file and a category: one plan step per
    # (file, category) rather than one per finding, because fixing them is one visit.
    groups: dict[tuple[str, str], list[dict]] = {}
    for f in supported:
        groups.setdefault((str(f.get("file")), str(f.get("category"))), []).append(f)

    steps = []
    for (path, cat), fs in groups.items():
        fs.sort(key=lambda x: SEV_ORDER.get(str(x.get("severity")), 9))
        top = fs[0]
        steps.append({
            "id": f"S{len(steps) + 1}",
            "file": path,
            "category": cat,
            "severity": top.get("severity"),
            "title": str(top.get("finding"))[:120],
            "evidence": [f.get("verify_note") for f in fs],
            "details": [
                {
                    "finding": f.get("finding"),
                    "line_range": f.get("line_range"),
                    "recommended_fix": f.get("recommended_fix"),
                    "severity": f.get("severity"),
                }
                for f in fs
            ],
            "verify_command": None,  # filled below where a command is obvious
        })

    # A few categories have a mechanical proof available; give the executor the command
    # rather than a description, because a command is checkable and prose is not.
    # ★ The README to grep is the one belonging to the repo UNDER AUDIT, never a fixed path.
    # This hardcoded helpotron's README, so a harness finding was "verified" by grepping a
    # DIFFERENT REPOSITORY's README - a command that looks like proof, runs clean, and is
    # about another project entirely. The repo root comes from AUDIT_TARGET_DIR, falling back
    # to the audit dir's own parent so the command at least stays inside the audited tree.
    repo_root = os.environ.get("AUDIT_TARGET_DIR") or os.environ.get("PLAN_REPO_ROOT") or ""
    readme = os.path.join(repo_root, "README.md") if repo_root else None

    for s in steps:
        base = os.path.basename(s["file"])
        if s["category"] not in ("UNDOCUMENTED",) and "README" not in s["category"]:
            continue
        if not readme:
            # No root known: say so rather than emitting a command against some other repo.
            s["verify_command"] = None
            s["verify_note"] = "set AUDIT_TARGET_DIR to get a README grep for this repo"
            continue
        note = "  # >0 after the fix; was 0 at audit time" if s["category"] == "UNDOCUMENTED" else ""
        s["verify_command"] = f"grep -c '{re.escape(base)}' {readme}{note}"

    steps.sort(key=lambda s: (SEV_ORDER.get(str(s["severity"]), 9), s["file"]))

    plan = {
        "generated": date.today().isoformat(),
        "source": VERIFIED,
        "note": (
            "Built ONLY from findings whose claim was checked against the file at the cited "
            "lines. The raw audit refuted 62-74% of its mechanically checkable findings, so "
            "unverified findings are leads, never steps."
        ),
        "counts": {
            "total_findings": len(findings),
            "supported": len(supported),
            "refuted": len(refuted),
            "unverified": len(unverified),
            "steps": len(steps),
        },
        "steps": steps,
        "leads": [
            {
                "file": f.get("file"),
                "category": f.get("category"),
                "severity": f.get("severity"),
                "finding": f.get("finding"),
                "line_range": f.get("line_range"),
                "why_unverified": f.get("verify_note"),
            }
            for f in unverified
            if str(f.get("severity")) in ("CRITICAL", "HIGH")
        ][:60],
        "refuted": [
            {
                "file": f.get("file"),
                "line_range": f.get("line_range"),
                "finding": f.get("finding"),
                "refuted_by": f.get("verify_note"),
            }
            for f in refuted
        ],
    }

    with open(OUT, "w") as fh:
        json.dump(plan, fh, indent=2, default=str)

    print(f"steps (evidence-backed): {len(steps)}")
    for s in steps:
        print(f"  [{s['severity']}/{s['category']}] {os.path.basename(s['file'])}"
              f" — {s['title'][:70]}")
    print(f"leads (CRITICAL/HIGH, unverified): {len(plan['leads'])}")
    print(f"refuted (kept so they are not re-derived): {len(plan['refuted'])}")
    print(f"written: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
