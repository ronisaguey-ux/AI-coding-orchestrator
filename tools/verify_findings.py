#!/usr/bin/env python3
"""Verify audit findings against the actual source before they reach a plan.

The problem this exists for, measured on the first pass: of 165 findings, 33 cited a
line range and 8 of those named a specific construct (raw SQL, a deprecated event
loop, a hardcoded key). Checking each against the cited lines, **8 of 8 were absent** —
the audit produced confident, specific, wrong claims. Two examples:

  adminctl.py:150-180  "accepts raw SQL commands without parameterization"
      -> the cited lines are HTTP calls to the API; the file contains no SQL at all.
  serve.py:15-25       "uses deprecated asyncio event loop patterns"
      -> the cited lines are a PATH join, a PORT read, and a class definition.

A finding that cannot be checked is not evidence. This script checks the checkable ones
and marks the rest UNVERIFIED rather than promoting them, so an execution plan is built
from claims that survived contact with the code.
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.environ.get("VERIFY_ROOT", "/home/roni/Roni_workspace/helpotron")
FINDINGS_DIR = os.environ.get("VERIFY_DIR", "/home/roni/Roni_workspace/audits_plans/pass_1")

# Each probe: the construct a finding claims, and the regex that would have to appear
# in the cited lines for the claim to be supportable. Deliberately GENEROUS — a probe
# that fires on any plausible spelling keeps false "unsupported" verdicts rare, because
# the cost of wrongly rejecting a real finding is as bad as accepting a fake one.
PROBES = [
    ("sql",        r"\b(execute|executemany|cursor)\s*\(|\bSELECT\b.{0,40}[+%]|\bsql\b"),
    ("sympify",    r"\bsympify\b|\bparse_expr\b|\blambdify\b"),
    ("eval",       r"\beval\s*\(|\bexec\s*\("),
    ("shell",      r"shell\s*=\s*True|os\.system\s*\(|subprocess\.|Popen"),
    ("eventloop",  r"get_event_loop|asyncio\.coroutine|run_until_complete|loop\."),
    ("innerhtml",  r"innerHTML|outerHTML|insertAdjacentHTML|document\.write"),
    ("except",     r"except\s*:"),
    ("pickle",     r"\bpickle\.loads?\b|\bmarshal\.loads?\b"),
    ("secret",     r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*[\"'][A-Za-z0-9_\-]{8,}"),
    ("xss",        r"innerHTML|dangerouslySetInnerHTML|eval\(|document\.write"),
    ("ssrf",       r"requests\.(get|post)|httpx\.|aiohttp|urlopen|fetch\("),
    ("traversal",  r"os\.path\.join|open\(|\.\./|send_file|FileResponse"),
    ("race",       r"threading\.|Lock\(|asyncio\.|await |concurrent"),
    ("nplusone",   r"for .* in .*:\s*$[\s\S]{0,200}?\.(query|filter|get)\("),
    ("logging",    r"(?i)(print|logger?\.(info|debug|error)).{0,80}(key|token|secret|password)"),
]

# Claim keywords that route a finding to a probe.
ROUTES = [
    ("sql",       ("sql", "injection", "parameteriz", "query string")),
    ("sympify",   ("sympify", "sympy")),
    # "rce" as a bare substring matches "source", "resource", "forced" — measured:
    # it refuted a finding about Promise parallelism and one about httpx timeouts,
    # neither of which mentions code execution. Word-bounded, and "eval(" is a
    # literal so comparison is a separate branch.
    ("eval",      ("eval(", "arbitrary code", "code execution", "rce")),
    ("shell",     ("shell", "command injection", "subprocess", "os.system")),
    ("eventloop", ("event loop", "asyncio", "coroutine", "deprecat")),
    ("innerhtml", ("innerhtml", "dom injection", "xss")),
    ("except",    ("bare except", "swallow", "silent failure")),
    ("pickle",    ("pickle", "deserializ")),
    ("secret",    ("hardcoded", "hard-coded", "credential", "api key", "secret")),
    ("ssrf",      ("ssrf",)),
    ("traversal", ("path traversal", "directory traversal")),
    ("nplusone",  ("n+1", "n + 1", "per-row", "in a loop")),
]


def code_lines_only(fp: str) -> str:
    """The file with comments and docstrings removed.

    A credential probe that reads prose finds placeholders. Measured: create_admin.py
    was reported as having "hardcoded default password for admin account" because line 4
    of its module docstring reads:
        Usage: ADMIN_PASSWORD="your-secure-pass" python scripts/create_admin.py
    The actual code reads os.getenv("ADMIN_PASSWORD") and falls back to
    secrets.token_urlsafe(16). Matching documentation is not matching code.
    """
    try:
        src = open(fp, errors="replace").read()
    except OSError:
        return ""
    if fp.endswith(".py"):
        import ast
        try:
            tree = ast.parse(src)
        except SyntaxError:
            return _strip_comments(src)
        # Blank out every docstring span, then every comment line.
        lines = src.split("\n")
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                body = getattr(node, "body", None) or []
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    d = body[0].value
                    for i in range(getattr(d, "lineno", 1) - 1, getattr(d, "end_lineno", 1)):
                        if 0 <= i < len(lines):
                            lines[i] = ""
        return _strip_comments("\n".join(lines))
    return _strip_comments(src)


def _strip_comments(src: str) -> str:
    out = []
    for line in src.split("\n"):
        t = line.strip()
        if t.startswith(("#", "//", "*", "/*")):
            continue
        out.append(line)
    return "\n".join(out)


def resolve(fp: str) -> str | None:
    """Find the file the finding names. Labels are repo-relative; some are bare names."""
    if os.path.isfile(fp):
        return fp
    cand = os.path.join(ROOT, fp)
    if os.path.isfile(cand):
        return cand
    base = os.path.basename(fp)
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in (".venv", "node_modules", ".git", "dist", "__pycache__")]
        if base in files:
            return os.path.join(root, base)
    return None


# Short acronyms need word boundaries: "rce" inside "source"/"resource"/"forced", and
# "xss" inside a word, produced false refutations. Every other key is a phrase or a
# literal, where a substring test is correct.
_WORD_BOUNDED = {"rce", "xss", "ssrf", "sql", "orm"}


def _route_matches(low: str, keys) -> bool:
    for k in keys:
        if k in _WORD_BOUNDED:
            if re.search(r"\b" + re.escape(k) + r"\b", low):
                return True
        elif k in low:
            return True
    return False


def pick_probe(text: str):
    low = text.lower()
    for name, keys in ROUTES:
        if _route_matches(low, keys):
            return name
    return None


def verify_one(f: dict) -> dict:
    """Return the finding with a verification verdict attached."""
    fp = resolve(str(f.get("file", "") or ""))
    lr = str(f.get("line_range", "") or "").strip()
    text = f"{f.get('finding','')} {f.get('mechanism','')} {f.get('impact','')}"

    out = dict(f)
    out["verified"] = "UNVERIFIED"
    out["verify_note"] = ""

    if not fp:
        out["verify_note"] = "file not found"
        return out

    # Claims that a file CAPTURES credentials/session data are checkable: the file must
    # actually touch document.cookie, storage, or a credential-shaped field. Measured:
    # claimed of extension/recorder.js — "captures DOM snapshots including CSRF tokens
    # and session cookies" — which is a MICROPHONE recorder (getUserMedia +
    # MediaRecorder) with zero matches for cookie/csrf/session/password/token.
    low3 = text.lower()
    if any(k in low3 for k in ("captures", "collects", "stores", "snapshots", "exposes"))             and any(k in low3 for k in ("credential", "cookie", "csrf", "session token",
                                        "api key", "password", "auth token")):
        body = open(fp, errors="replace").read()
        touches = re.search(
            r"document\.cookie|localStorage|sessionStorage|chrome\.cookies|"
            r"getCookie\(|csrf|apiKey|api_key|authToken|access_token", body, re.I)
        if not touches:
            out["verified"] = "REFUTED"
            out["verify_note"] = "file has no cookie/storage/credential access"
            return out
        out["verify_note"] = "file does touch credential/storage APIs; severity needs reading"
        return out


    if not re.fullmatch(r"\d+\s*-\s*\d+", lr):
        out["verify_note"] = "no line range cited"
        return out

    a, b = (int(x) for x in re.split(r"\s*-\s*", lr))
    if a < 1 or b <= a:
        out["verify_note"] = "degenerate line range"
        return out

    try:
        lines = open(fp, errors="replace").read().split("\n")
    except OSError as e:
        out["verify_note"] = f"unreadable: {e}"
        return out

    # A range past EOF means the auditor invented the location, so the claim is false
    # rather than merely unchecked. Measured: a CRITICAL found "arbitrary code execution
    # via browser console" at cf_computer.py:180-220 in a 155-line file, and returned
    # UNVERIFIED — which puts an invented location in the human-review queue instead of
    # the refuted pile. A location that does not exist cannot hold the defect.
    if a > len(lines) or b > len(lines) + 20:
        out["verified"] = "REFUTED"
        out["verify_note"] = f"cites lines {a}-{b}; file has {len(lines)} lines"
        return out
    seg = "\n".join(lines[a - 1:b])

    # "Lacks authentication / no authz / no rate limiting / no validation / no audit log"
    # is refutable by finding the guard. These are the most common SECURITY claims and
    # they are mechanically checkable: the guard is a named call, or it is not there.
    # Measured: claimed "admin endpoint lacks authentication" of admin_advanced.py, which
    # calls require_admin on every route.
    GUARDS = {
        "authn": (("lacks authentication", "no authentication", "without authentication",
                   "unauthenticated", "no auth"), r"require_admin|get_current_user|"
                   r"Depends\(|authenticate|require_session_auth|verify_jwt"),
        "authz": (("no authorization", "lacks authorization", "no access control",
                   "missing ownership", "no permission"), r"require_admin|user_id\s*==|"
                   r"is_admin|ownership|\.filter\(.*user_id"),
        "ratelimit": (("no rate limit", "without rate limit", "no throttl"),
                      r"limiter|ratelimit|rate_limit|throttl"),
        "validation": (("no validation", "without validation", "unvalidated",
                        "no input validation", "arbitrary input"),
                       r"validate|pydantic|BaseModel|Field\(|re\.match|check_|sanitiz"),
        "auditlog": (("no audit", "without audit", "no logging", "not logged"),
                     r"AdminAction|_audit|audit_|logger\.|log\.(info|warning)|record\("),
        "sandbox": (("no sandbox", "without sandbox", "unsandboxed", "arbitrary code"),
                    r"sandbox|_is_safe|whitelist|allowlist|DISALLOWED|rlimit|seccomp|ast\."),
    }
    body_full = open(fp, errors="replace").read()
    seg_check = seg if "seg" in dir() else body_full
    # Before judging a "no guard for X" claim, check that X EXISTS. Measured:
    # "create_export_job() lacks an idempotency key" was marked SUPPORTED against
    # server/export.py, which defines no such function at all — the file holds
    # markdown_to_latex and a set of export_* formatters. A missing guard on a missing
    # function is not a finding, it is an invention.
    _m = re.search(r"\b([a-z_][a-z0-9_]{3,})\s*\(\s*\)", text)
    if _m:
        symbol = _m.group(1)
        body_syms = open(fp, errors="replace").read()
        if not re.search(r"\b(def|function|class|const|let|var)\s+" + re.escape(symbol) + r"\b",
                         body_syms):
            out["verified"] = "REFUTED"
            out["verify_note"] = f"{symbol}() does not exist in this file"
            return out

    # "Inverted/unclear logic without justification" is checkable: the justification is
    # either in the file or it is not. Measured: flagged of SANDBOX_LOG ("inverted logic
    # without clear justification") in a repo where settings.js:613 carries a comment
    # explaining the inversion AND sandbox.js:15 documents "default: true — log every
    # denial" in its env-var block.
    if any(k in low3 for k in ("inverted logic", "without clear justification",
                               "without justification", "unclear justification",
                               "unjustified", "no justification")):
        body = open(fp, errors="replace").read()
        # A nearby comment or doc line that mentions the same setting and a reason word.
        sym = re.search(r"\bSANDBOX_[A-Z_]+\b|\b[A-Z][A-Z0-9_]{4,}\b", text)
        if sym:
            name = sym.group(0)
            pat = (r"(?:#|//|/\*).*" + re.escape(name) + r".*(invert|default|because|so that|unless|intentional)")
            if re.search(pat, body, re.I | re.M):
                out["verified"] = "REFUTED"
                out["verify_note"] = f"the {name} behaviour is explained in a comment here"
                return out
            # The justification is often in the file that IMPLEMENTS the setting rather
            # than the one that tests it. Measured: "SANDBOX_LOG uses inverted logic
            # without clear justification" cited harness_tests/cli_settings.test.js, while
            # the explanation sits in cli/settings.js:613 and sandbox.js:15. Searching
            # only the cited file produced a SUPPORTED verdict on a claim that is false.
            for root, dirs, files in os.walk(ROOT):
                dirs[:] = [d for d in dirs
                           if d not in (".venv", "node_modules", ".git", "dist", "__pycache__")]
                hit = False
                for fn in files:
                    if not fn.endswith((".py", ".js", ".jsx", ".ts", ".tsx")):
                        continue
                    fp2 = os.path.join(root, fn)
                    try:
                        other = open(fp2, errors="replace").read()
                    except OSError:
                        continue
                    if re.search(pat, other, re.I | re.M):
                        out["verified"] = "REFUTED"
                        out["verify_note"] = (f"the {name} behaviour is explained in a "
                                              f"comment in {os.path.basename(fp2)}")
                        hit = True
                        break
                if hit:
                    return out

    # A "no guard for SUBJECT" claim needs the SUBJECT to exist too. Measured: "DNS
    # record creation not idempotent" came back SUPPORTED against server/cf_computer.py,
    # which is a Cloudflare Computer-Use workspace client with ZERO matches for
    # dns/record/subdomain anywhere in the file. The subject was invented, so its
    # missing guard is not a defect.
    SUBJECTS = {
        "dns": r"\bdns\b|zone_id|record_id|\bcname\b|\ba_record\b",
        "email": r"smtp|sendgrid|mailgun|send_mail|smtplib",
        "payment": r"stripe|checkout|invoice|payment_intent|webhook",
        "upload": r"upload|multipart|File\(|UploadFile",
        "cache": r"\bcache\b|redis|memcach",
        "queue": r"queue|celery|task_id|enqueue",
        # Measured: "OAuth state parameter not validated in callback" twice against
        # server/canvas_sync.py, which is a REST sync engine (RFC 5988 pagination,
        # bearer-token fetches) with ZERO matches for oauth/callback/authorize/
        # redirect_uri/pkce. There is no OAuth flow in the file to have a flaw.
        "oauth": r"\boauth\b|authorize|redirect_uri|\bpkce\b|code_verifier|client_id",
    }
    for subj, pat in SUBJECTS.items():
        if subj not in low3:
            continue
        if not re.search(pat, body_full, re.I):
            out["verified"] = "REFUTED"
            out["verify_note"] = f"file contains no {subj}-related code; subject does not exist here"
            return out

    # A guard claim is only meaningful for something that EXPOSES a surface. A CLI
    # script has no auth guard because it is not an endpoint — flagging that as a
    # supported security gap is a false positive of mine, and it was one: this probe
    # called server/admin_advanced.py "lacks authentication", but that file is a
    # service CLASS with zero routes, and every route in the module that serves it
    # (routes_admin_advanced.py) calls require_admin. Same for the repro_*/probe_*
    # scripts, which are only ever run by hand.
    EXPOSES = re.compile(r"@(?:router|app)\.(?:get|post|put|patch|delete)|"
                         r"APIRouter\(|add_api_route|@app\.(get|post)|@router\.websocket")
    for name, (phrases, pat) in GUARDS.items():
        if not any(k in low3 for k in phrases):
            continue
        if not EXPOSES.search(body_full):
            out["verified"] = "UNVERIFIED"
            out["verify_note"] = ("not an endpoint surface (no route decorators); "
                                  f"a {name} guard is not applicable here")
            return out
        # Look in the cited lines first; fall back to the file, because a guard often
        # sits on a decorator or a signature just outside the cited range.
        found = re.search(pat, seg_check, re.M) or re.search(pat, body_full, re.M)
        out["verified"] = "REFUTED" if found else "SUPPORTED"
        out["verify_note"] = (f"{name} guard present" if found
                              else f"no {name} guard found in a file that exposes routes")
        return out

    # "Dead file / backup artifact / stale / legacy, should be deleted" is refutable by
    # asking whether the live code imports it. Measured: claimed of server/database.py —
    # "backup artifact, stale SQLAlchemy models from August 2026" — which is the module
    # main.py imports 38 models from.
    low2 = text.lower()
    if any(k in low2 for k in ("backup artifact", "dead file", "stale ", "should be deleted",
                               "no longer used", "unused file", "orphaned file")):
        base = os.path.basename(fp)
        stem = base.rsplit(".", 1)[0]
        imported = False
        for root, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs
                       if d not in (".venv", "node_modules", ".git", "dist", "__pycache__")]
            for fn in files:
                if not fn.endswith((".py", ".js", ".jsx", ".ts", ".tsx")):
                    continue
                fp2 = os.path.join(root, fn)
                if fp2 == fp:
                    continue
                try:
                    body = open(fp2, errors="replace").read()
                except OSError:
                    continue
                if f"import {stem}" in body or f"from .{stem}" in body \
                        or f"from {stem}" in body or f'"{stem}' in body or f"'{stem}" in body:
                    imported = True
                    break
            if imported:
                break
        out["verified"] = "REFUTED" if imported else "UNVERIFIED"
        out["verify_note"] = ("module is imported by other source files"
                              if imported else f"no importer of {stem} found")
        return out

    # "Hardcoded credentials" is the single most common false claim in this audit.
    # Measured: claimed against create_admin.py, whose ONLY password sources are
    # os.getenv("ADMIN_PASSWORD") and secrets.token_urlsafe(16) — there is no literal.
    # The claim is checkable: a hardcoded credential needs a string literal assigned to
    # a credential-shaped name. An env read or a random generator refutes it.
    low1 = text.lower()
    if any(k in low1 for k in ("hardcoded", "hard-coded", "hard coded", "default credential",
                               "default admin password", "credentials in source")):
        body = code_lines_only(fp)
        literal = None
        for ln in body.split("\n"):
            m = re.match(r"\s*(?:const |let |var |export )?"
                         r"(password|passwd|secret|api[_-]?key|token)\w*\s*[:=]\s*[\"'][^\"']{6,}[\"']",
                         ln, re.I)
            if m:
                literal = m
                break
        env_or_random = re.search(
            r"os\.getenv\(|os\.environ|secrets\.|token_urlsafe|getpass|input\(", body)
        if literal:
            out["verified"] = "SUPPORTED"
            out["verify_note"] = f"credential literal present: {literal.group(0)[:40]}"
        elif env_or_random:
            out["verified"] = "REFUTED"
            out["verify_note"] = "no literal; value comes from env or a random generator"
        else:
            out["verify_note"] = "no credential literal and no env/random source found"
        return out

    # "No idempotency / no rollback / partial failure / float precision" are all
    # checkable against the file, and all four were claimed against a migration script
    # that has an idempotency marker, a backup, ONE commit for the whole batch, and
    # integer-only arithmetic (balances come from `ceil`, so x100 is exact).
    low0 = text.lower()
    if any(k in low0 for k in ("checkpoint", "resume", "re-executes all stages",
                               "starts from the beginning", "not resumable")):
        body = open(fp, errors="replace").read()
        has_state = re.search(r"load_state|save_state|checkpoint|resume", body, re.I)
        out["verified"] = "SUPPORTED" if not has_state else "REFUTED"
        out["verify_note"] = ("no state/resume machinery in the file" if not has_state
                              else "state/resume machinery present")
        return out
    if any(k in low0 for k in ("idempot", "re-run", "run again", "multiply twice")):
        has = re.search(r"MARKER|already_migrat|already migrated|marker|_done\b", 
                        open(fp, errors="replace").read(), re.I)
        out["verified"] = "SUPPORTED" if not has else "REFUTED"
        out["verify_note"] = ("no idempotency guard found" if not has
                              else "idempotency guard present (marker/already-migrated)")
        return out
    if any(k in low0 for k in ("no rollback", "without rollback", "no transaction",
                               "partial failure", "inconsistent state")):
        body = open(fp, errors="replace").read()
        commits = len(re.findall(r"\.commit\(\)", body))
        has_backup = bool(re.search(r"shutil\.copy|backup|\.bak", body, re.I))
        # A single commit for the whole batch IS the transaction boundary.
        if commits == 1 and has_backup:
            out["verified"] = "REFUTED"
            out["verify_note"] = f"one commit for the batch + backup ({commits} commit, backup present)"
        elif commits == 1:
            out["verified"] = "REFUTED"
            out["verify_note"] = "single commit for the batch = atomic"
        else:
            out["verify_note"] = f"{commits} commits; transaction claim not mechanically decidable"
        return out
    if "float" in low0 or "floating" in low0 or "rounding error" in low0:
        body = open(fp, errors="replace").read()
        # If the values are produced by ceil/int they are integers, and int*100 is exact.
        integer_source = bool(re.search(r"ceil|int\(|round\(", body))
        if integer_source:
            out["verified"] = "REFUTED"
            out["verify_note"] = "values are integer-producing (ceil/int), so x100 is exact"
            return out

    # A stated size is a hard number and therefore trivially checkable: a finding that
    # calls a file "a 600-line monolithic script" is refuted by counting. Measured: one
    # such finding claimed 600 lines for a 254-line file with 11 named functions.
    m_size = re.search(r"\b(\d{2,5})[\s-]*line\b", text, re.I)
    if m_size and not re.search(r"\bline_range\b", text[:m_size.start()], re.I):
        claimed = int(m_size.group(1))
        actual = len(lines)
        # Generous band: prose rounds, and a claim of "600-line" against 590 is fine.
        if claimed >= 50 and not (actual * 0.8 <= claimed <= actual * 1.25):
            out["verified"] = "REFUTED"
            out["verify_note"] = (f"claims {claimed} lines; {os.path.basename(fp)} is {actual}")
            return out

    # The documentation class is the largest category of finding and it is checkable
    # without any construct guesswork: the claim is "X is not documented", so grep the
    # README for X. A claim of "undocumented" against a file the README names is false.
    low = text.lower()
    if any(k in low for k in ("not documented", "undocumented", "not mentioned in",
                              "not in the readme", "no documentation", "documented in readme")):
        base = os.path.basename(fp)
        try:
            readme = open(os.path.join(ROOT, "README.md"), errors="replace").read()
        except OSError:
            readme = ""
        named = base in readme
        out["verified"] = "SUPPORTED" if not named else "REFUTED"
        out["verify_note"] = (f"{base} {'is' if named else 'is not'} named in README.md")
        return out

    probe = pick_probe(text)
    if not probe:
        out["verify_note"] = "no mechanical probe for this claim"
        return out

    # SSRF needs a REQUEST-CONTROLLED url. If the script builds its url from a constant
    # and takes no arguments, there is no input to steer and the claim is refuted.
    # Measured: claimed against probe_vision_models.py, which uses a hardcoded
    # `IMAGE = ROOT / "verify-artifacts" / ...` and has no argv/argparse at all.
    if probe == "ssrf":
        body = open(fp, errors="replace").read()
        takes_input = re.search(
            r"sys\.argv|argparse|input\(|request\.(args|form|json|query)|"
            r"@app\.(get|post)|@router\.(get|post)", body)
        if not takes_input:
            out["verified"] = "REFUTED"
            out["verify_note"] = "no request/argv input in the file; url is not attacker-controlled"
            return out

    pat = dict(PROBES)[probe]
    if re.search(pat, seg, re.M):
        out["verified"] = "SUPPORTED"
        out["verify_note"] = f"{probe}: construct present in {os.path.basename(fp)}:{lr}"
    else:
        out["verified"] = "REFUTED"
        out["verify_note"] = f"{probe}: construct ABSENT from {os.path.basename(fp)}:{lr}"
    return out


def main() -> int:
    files = sorted(
        os.path.join(FINDINGS_DIR, n)
        for n in os.listdir(FINDINGS_DIR)
        if n.startswith("batch_") and n.endswith(".json")
    ) if os.path.isdir(FINDINGS_DIR) else []
    if not files:
        print(f"no batch files in {FINDINGS_DIR}", file=sys.stderr)
        return 1

    findings = []
    for path in files:
        try:
            findings += json.load(open(path)).get("findings", [])
        except (OSError, ValueError):
            continue

    verified = [verify_one(f) for f in findings]
    counts: dict[str, int] = {}
    for f in verified:
        counts[f["verified"]] = counts.get(f["verified"], 0) + 1

    print(f"findings: {len(verified)}")
    for k in ("SUPPORTED", "REFUTED", "UNVERIFIED"):
        print(f"  {k:12} {counts.get(k, 0)}")

    checkable = counts.get("SUPPORTED", 0) + counts.get("REFUTED", 0)
    if checkable:
        pct = 100 * counts.get("REFUTED", 0) / checkable
        print(f"  of the {checkable} mechanically checkable, {pct:.0f}% were refuted")

    out_path = os.path.join(os.path.dirname(FINDINGS_DIR.rstrip("/")), "verified_findings.json")
    with open(out_path, "w") as fh:
        json.dump(verified, fh, indent=2, default=str)
    print(f"written: {out_path}")

    print("\n-- refuted (these would have entered the plan as real) --")
    for f in verified:
        if f["verified"] == "REFUTED":
            print(f"  {os.path.basename(str(f.get('file')))}:{f.get('line_range')}"
                  f"  {str(f.get('finding'))[:78]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
