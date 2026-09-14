#!/usr/bin/env python3
"""audit.py — convert the multi-agent MD report into a
structure-preserving JSON (LLM-parsable) sibling: multi_agent_oculus_audit_8_27.json.
Parses metadata, agent usage, summary stats tables, spec matrix, and every
Accepted Findings block into records. 09-02 user: "make it a json report instead
of md report so it's more llm parsable".
"""
import json
import re
import sys

MD = "/home/roni/Roni_workspace/audits_plans/multi_agent_oculus_audit_8_27.md"
OUT = MD.replace(".md", ".json")

text = open(MD, encoding="utf-8").read()
lines = text.split("\n")

def meta(key):
    m = re.search(rf"\*\*{key}:\*\*\s*(.+)", text)
    return m.group(1).strip() if m else None

out = {
    "report_type": "multi_agent_oculus_audit",
    "generated_at": meta("Generated"),
    "passes": int(meta("Passes") or 0),
    "total_findings": int(meta("Total Findings") or 0),
    "accepted_findings": int(meta("Accepted Findings") or 0),
    "rejected_findings": int(meta("Rejected Findings") or 0),
    "acceptance_threshold": meta("Acceptance Threshold"),
    "acceptance_rate": round(100.0 * float(meta("Accepted Findings")) / float(meta("Total Findings") or 1), 1),
    "agent_model_usage": [],
    "summary_statistics": {},
    "spec_compliance_matrix": [],
    "accepted_findings": [],
}

# ── agent model usage ─────────────────────────────────────────────────────
for m in re.finditer(r"- \*\*(.+?)\*\* — composite model score: \*\*\s*([\d.]+)\s*\*\*, calls: (\d+), top models: (.+)",
                     text):
    out["agent_model_usage"].append({
        "agent": m.group(1), "composite_score": float(m.group(2)),
        "calls": int(m.group(3)),
        "top_models": dict(re.findall(r"(\w+)\((\d+)\)", m.group(4))),
    })

# ── summary statistics tables (By Severity / By Category) ──────────────────
def parse_bullets(header):
    """Collect '- **K:** V' bullets under a '### header' until the next heading."""
    idx = lines.index(header) if header in lines else -1
    if idx < 0:
        return {}
    entries = {}
    for ln in lines[idx + 1:]:
        s = ln.strip()
        if s.startswith("#"):
            break
        m = re.match(r"-\s*\*\*(.+?):\*\*\s*([\d.,]+)", s)
        if m:
            entries[m.group(1).strip()] = int(m.group(2).replace(",", ""))
    return entries

for table in ("### By Severity (Accepted)", "### By Category (Accepted)",
              "### Critical Classification Totals (Accepted)"):
    out["summary_statistics"][table] = parse_bullets(table)

# ── spec compliance matrix ─────────────────────────────────────────────────
idx = text.find("## Spec Compliance Matrix")
if idx >= 0:
    tail = text[idx: text.find("## Accepted Findings", idx) if "## Accepted Findings" in text[idx:] else len(text)]
    for m in re.finditer(r"-\s*\[( |x)\]\s*(.+)", tail):
        out["spec_compliance_matrix"].append({"checked": m.group(1) == "x", "item": m.group(2).strip()})

# ── accepted findings blocks ───────────────────────────────────────────────
# Blocks: ### <FID> — <SEVERITY> [ (📖README|📖SoT|...)]  then "- **Field:** value"
FID_RE = re.compile(r"^### (P\d+B\d+R\d+F\d+)\s*—\s*([A-Z]+)(.*)$")
cur = None
for ln in lines:
    m = FID_RE.match(ln.strip())
    if m:
        if cur:
            out["accepted_findings"].append(cur)
        cur = {"finding_id": m.group(1), "severity": m.group(2).strip(),
               "tags": [t.strip() for t in re.findall(r"\((📖[^)]+)\)", m.group(3))], "fields": {}}
        continue
    if cur is not None and (m2 := re.match(r"^-\s*\*\*(.+?):\*\*\s*(.*)$", ln.strip())):
        cur["fields"][m2.group(1).strip()] = m2.group(2)
if cur:
    out["accepted_findings"].append(cur)

json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
print(f"wrote {OUT}: {len(out['accepted_findings'])} finding records, "
      f"{len(out['agent_model_usage'])} agents, tables={list(out['summary_statistics'].keys())}")
