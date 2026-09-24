# FULL-FIELD AUDIT — scoring, UI/UX, security, correctness, performance

Audit the target repositories across every field that matters, not just bugs.

## Fields to cover (each finding must name which field it is)
1. SECURITY — authn/authz, injection, secrets, SSRF, path traversal, IDOR, unsafe deserialisation
2. CORRECTNESS — logic errors, race conditions, wrong defaults, silent failures, error handling
3. UI/UX — unclear states, missing feedback, unreachable controls, empty/error states, accessibility
4. PERFORMANCE — hot paths, N+1, unbounded growth, missing indexes, blocking IO
5. DATA INTEGRITY — migrations, backups, partial writes, idempotency
6. RELIABILITY — retries, timeouts, crash recovery, monitoring gaps
7. MAINTAINABILITY — dead code, duplicated logic, stale comments that contradict code
8. TESTING — untested critical paths, tests that pass vacuously, tests that cannot run

## Rules
- Every finding needs file:line, what is wrong, why it matters, and how to prove it.
- Prefer a reproducible failure over a theory. If you cannot reproduce it, say so.
- Rank by exploitability/impact, not by how easy it is to describe.
- State explicitly when a suspected issue is NOT real, and why — a refutation is a result.

## EVIDENCE REQUIREMENT — added 2026-09-24 after measuring this audit's own output

**Measured: of 165 findings from a full pass, 33 cited a line range and 8 of those named a
specific construct. Checked against the cited lines, 8 of 8 were ABSENT — the finding was
wrong.** Two verbatim examples, both of which would have entered an execution plan as real
defects:

- `adminctl.py:150-180` — *"accepts raw SQL commands without parameterization"*. The cited
  lines are HTTP calls to the API (`f"/api/admin/user/{id}/plan"`). The file contains **no
  SQL at all**.
- `serve.py:15-25` — *"uses deprecated asyncio event loop patterns"*. The cited lines are
  `os.path.join`, a `PORT` read, and a `class` definition. There is no asyncio in the file.

**A confident, specific, wrong finding is worse than no finding**, because it costs a
verifier's time and it teaches the reader to distrust the whole report.

So, per finding:

1. **Read the line range you are about to cite before you cite it.** If you have not seen
   the construct in those exact lines, you may not claim it is there.
2. **`line_range` must be the lines that contain the defect, not the whole file.** `1-200`,
   `1-50`, `1-100` are not ranges, they are a file. A finding whose range is the entire file
   is treated as unsupported.
3. **If you cannot cite exact lines, write `line_range: "N/A"` and lower the confidence to
   LOW.** An honest "I could not locate this precisely" is a useful result. An invented
   line number is not.
4. **Name the construct you are claiming.** "Contains hardcoded credentials" is checkable —
   the credential literal must be visible in the cited lines. "Potentially insecure" is not
   a finding.
5. **Do not report the absence of your own context as a defect.** A finding that a file has
   "no graph data available" or was "not provided in batch" describes this audit's inputs,
   not the code. Report that in a separate `analysis_gaps` list, never as a finding.
6. **A file being undocumented is only a finding if you checked.** Before claiming "X is not
   documented in the README", search the README for X. Several such findings were false
   because the file IS named there.
