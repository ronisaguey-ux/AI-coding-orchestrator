#!/usr/bin/env python3
"""Cross-evaluate the three audits and produce one consolidated, verified picture.

Run after the three sweeps settle. For each repo it verifies every finding against
the real source, then reports the comparison that actually matters: how many claims
survived, how many were invented, and which probes did the refuting. It deliberately
prints nothing about UNVERIFIED findings except their count — an unchecked claim is
not a result, and listing them invites someone to act on one.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
# ★ These paths must match where a run ACTUALLY writes (cli/runs.py out_dir() is per target:
# audits_plans/<target>). They did not: harness pointed at audits_plans_harness and t2b at a
# t2b/ dir that does not exist, so cross_eval read stale or missing passes and compared the
# wrong things. Kept in one table so a repo move is one edit.
_W = "/home/roni/Roni_workspace"
REPOS = [
    ("helpotron", f"{_W}/audits_plans/helpotron", f"{_W}/helpotron"),
    ("harness", f"{_W}/audits_plans/harness", f"{_W}/webchat_worker/harness"),
    ("t2b", f"{_W}/audits_plans/t2b", f"{_W}/t2b"),
]


def run_verifier(out_dir: str, root: str) -> list[dict]:
    env = dict(os.environ, VERIFY_DIR=os.path.join(out_dir, "pass_1"), VERIFY_ROOT=root)
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "verify_findings.py")],
        capture_output=True, text=True, env=env, timeout=900,
    )
    verified = os.path.join(out_dir, "verified_findings.json")
    if not os.path.isfile(verified):
        return []
    return json.load(open(verified))


def main() -> int:
    overall = Counter()
    summary = []

    for name, out_dir, root in REPOS:
        findings = run_verifier(out_dir, root)
        counts = Counter(f.get("verified") for f in findings)
        overall.update(counts)

        checkable = counts.get("SUPPORTED", 0) + counts.get("REFUTED", 0)
        refuted_pct = (100 * counts.get("REFUTED", 0) / checkable) if checkable else 0

        summary.append({
            "repo": name,
            "findings": len(findings),
            "supported": counts.get("SUPPORTED", 0),
            "refuted": counts.get("REFUTED", 0),
            "unverified": counts.get("UNVERIFIED", 0),
            "refuted_pct_of_checkable": round(refuted_pct),
        })

        print(f"\n=== {name} ===")
        print(f"  findings   {len(findings)}")
        print(f"  supported  {counts.get('SUPPORTED', 0)}")
        print(f"  refuted    {counts.get('REFUTED', 0)}"
              + (f"  ({refuted_pct:.0f}% of checkable)" if checkable else ""))
        print(f"  unverified {counts.get('UNVERIFIED', 0)}")
        if counts.get("SUPPORTED", 0):
            print("  supported findings:")
            for f in findings:
                if f.get("verified") == "SUPPORTED":
                    print(f"    [{f.get('severity')}/{f.get('category')}] "
                          f"{os.path.basename(str(f.get('file')))}:{f.get('line_range')}")
                    print(f"      {str(f.get('finding'))[:110]}")

    total_checkable = overall.get("SUPPORTED", 0) + overall.get("REFUTED", 0)
    print("\n" + "=" * 62)
    print(f"  TOTAL findings     {sum(s['findings'] for s in summary)}")
    print(f"  TOTAL supported    {overall.get('SUPPORTED', 0)}")
    print(f"  TOTAL refuted      {overall.get('REFUTED', 0)}")
    print(f"  TOTAL unverified   {overall.get('UNVERIFIED', 0)}")
    if total_checkable:
        print(f"  refuted share of checkable: "
              f"{100 * overall.get('REFUTED', 0) / total_checkable:.0f}%")

    # ★ Write where the summary belongs, and make the directory if the run has not created it.
    # This crashed with FileNotFoundError whenever the first repo in the table had no pass yet:
    # the whole cross-eval ran, printed every number, and then died on the last line, so the
    # comparison existed only in scrollback. A report you cannot open later is not a report.
    out_dir = os.environ.get("CROSS_OUT_DIR") or os.path.join(_W, "audits_plans")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "cross-eval.json")
    with open(out, "w") as fh:
        json.dump({"repos": summary,
                   "totals": dict(overall)}, fh, indent=2)
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
