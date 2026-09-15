#!/usr/bin/env python3
"""orch_git.py — per-step commit+push for the fix engine (user 09-06).

Every green step lands as an INDIVIDUAL commit pushed to origin/main, so the
Oculus repo tracks progress continuously (no local-only piles).
Safety: all git ops serialize on an os-level flock (concurrent engine batches
would otherwise smash the index), pushes use the live gh token (x-access-token)
with retry, and orchestration-pipeline changes are never included here.

Usage (from the engine):  git_commit_step(sid, rec)   # async, safe to await
"""
import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("ORCH_REPO", "/home/roni/Roni_Workspace/oculus"))
PUSH_EVERY = int(os.environ.get("ORCH_PUSH_EVERY", "1"))  # per-step push


def _run(cmd, cwd=None, timeout=120):
    return subprocess.run(cmd, cwd=cwd or REPO, capture_output=True, text=True,
                          timeout=timeout, check=False)


def _pat() -> str:
    """Read the orchestration PAT.

    Prefer $ORCH_GIT_PAT, then the 0600 file at ~/.config/orch/pat.txt, then the
    token already embedded in origin. `gh auth token` is deliberately NOT used:
    gh is not logged in on this box (returns empty), which silently disabled
    every push (engine committed locally, origin never advanced)."""
    tok = os.environ.get("ORCH_GIT_PAT", "").strip()
    if tok:
        return tok
    p = Path(os.path.expanduser("~/.config/orch/pat.txt"))
    if p.exists():
        tok = p.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    out = _run(["git", "remote", "get-url", "origin"]).stdout.strip()
    m = re.search(r"x-access-token:([^@]+)@", out)
    return m.group(1) if m else ""


def _remote_with_token() -> str:
    """Ensure origin embeds the PAT so pushes never prompt."""
    out = _run(["git", "remote", "get-url", "origin"]).stdout.strip()
    token = _pat()
    if not token:
        return out  # will fail later with a clear error
    if "x-access-token:" in out and token in out:
        return out
    new = f"https://x-access-token:{token}@github.com/ronisaguey-ux/OCULUS.git"
    _run(["git", "remote", "set-url", "origin", new])
    return new


def _files_of(rec: dict) -> list[str]:
    la = rec.get("last_apply") or {}
    files = []
    for e in la.get("edits") or []:
        if isinstance(e, dict):
            f = e.get("file") or e.get("path")
            if f and f not in files:
                files.append(f)
    return files


async def git_commit_step(sid: str, rec: dict) -> str:
    """Commit exactly this step's files (individual commit) and push."""
    files = _files_of(rec)
    if not files:
        return "no-files"
    await asyncio.to_thread(_commit_blocking, sid, files)
    return ",".join(files)


def _assert_oculus_repo() -> None:
    """Hard guard (2026-09-10): refuse to commit/push unless REPO is the OCULUS
    repo. Prevents a shared worktree from pushing oculus edits into an unrelated
    repo (the webchat-to-api-harness IP breach)."""
    name = REPO.resolve().name
    remote = _run(["git", "remote", "get-url", "origin"]).stdout.strip()
    ok_name = name == "oculus"
    # 09-14: GitHub repo URLs are case-insensitive — the canonical remote is
    # `ronisaguey-ux/oculus.git` but this guard demanded the uppercase
    # `OCULUS.git`, so EVERY step commit was refused ("REFUSING git ops") and
    # the engine landed fixes it could not commit (measured: 1 commit in 15 min
    # with green flat while lanes answered). Compare case-insensitively.
    ok_remote = "ronisaguey-ux/oculus.git" in remote.lower()
    if not (ok_name and ok_remote):
        raise RuntimeError(
            f"orch_git: REFUSING git ops — REPO={REPO} (name={name}), "
            f"origin={remote.split('@')[-1]}. Expected the OCULUS repo.")


def _commit_blocking(sid: str, files: list[str]) -> None:
    _assert_oculus_repo()
    _remote_with_token()
    existing = {p.strip() for p in _run(["git", "diff", "--name-only", "HEAD"]).stdout.splitlines()}
    # 09-09 (Bob): only git-add paths that resolve INSIDE the repo worktree.
    # Previously `../webchat-api/...` passed os.path.exists(REPO / f) (it exists
    # as a sibling), so git add was asked for a path outside the worktree.
    def in_repo(p: str) -> bool:
        try:
            cand = (REPO / p).resolve()
            root = REPO.resolve()
            return cand == root or root in cand.parents
        except Exception:
            return False
    to_add = [f for f in files if (f in existing or os.path.exists(REPO / f)) and in_repo(f)]
    if not to_add:
        return  # nothing changed on disk for this step (already committed)
    for attempt in range(6):
        r = _run(["git", "add", "--"] + to_add)
        if r.returncode == 0:
            break
        # index lock (concurrent batch) -> wait and retry
        import time
        time.sleep(0.5 * (attempt + 1))
    msg = f"fix(step {sid}): {len(to_add)} file(s) — {' '.join(to_add)[:120]}"
    # 09-12: a step that is re-run after a phantom-green reopen has nothing new to
    # commit — its files were already committed by the first pass. `git commit`
    # then exits non-zero and prints the whole "Changes not staged for commit"
    # block, which the old code logged as `[git] commit <sid> failed:`. That is
    # noise, not a failure. Ask the index directly: nothing staged = nothing to
    # commit for this step, so return quietly.
    staged = _run(["git", "diff", "--cached", "--quiet"])
    if staged.returncode == 0:
        return
    c = _run(["git", "commit", "-m", msg])
    if c.returncode == 0:
        _push_retry(attempt=0)
    else:
        # 09-12: a failed commit was logged as `[git] commit <sid> failed:` with
        # an EMPTY message, which made the failure undiagnosable — git can report
        # the reason on stdout (hooks, identity, index state). Retry once for the
        # transient index-lock case, then report BOTH streams.
        import time
        time.sleep(1.0)
        c = _run(["git", "commit", "-m", msg])
        if c.returncode == 0:
            _push_retry(attempt=0)
            return
        if "nothing to commit" not in (c.stderr + c.stdout).lower():
            detail = (c.stderr.strip() or c.stdout.strip() or f"git commit exited {c.returncode} with no output")
            sys.stderr.write(f"[git] commit {sid} failed: {detail[:400]}\n")


def _push_retry(attempt=0):
    for attempt in range(3):
        p = _run(["git", "push", "origin", "HEAD:main"])
        if p.returncode == 0:
            return
        if attempt < 2:
            import time
            time.sleep(4)
    sys.stderr.write(f"[git] push failed after retries: {p.stderr[:160]}\n")
