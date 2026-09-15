#!/usr/bin/env python3
"""Verify the plan gate: run tests and enforce rollback_command in every plan step.

Experimental plan-state daemon: verifies plans against a compiled JSON AST
cache so repeated gate runs do not rescan the markdown with string regexes.
"""

import sys
import os
import json
import hashlib
from pathlib import Path

# Ensure the scripts directory is on the path so we can import plan_runner and plan_validator
sys.path.insert(0, os.path.dirname(__file__))

from plan_runner import run_tests
from plan_validator import validate_rollback_commands


def _read_plan(path: Path) -> str:
    """Read plan file contents with simple in-memory caching."""
    if not hasattr(_read_plan, "_cache") or _read_plan._path != path:
        _read_plan._cache = path.read_text()
        _read_plan._path = path
    return _read_plan._cache


_DEFAULT_PLANS_DIR = Path(os.environ.get("OCULUS_PLANS_DIR", "/home/roni/Roni_workspace/audits_plans"))
_PLAN_PATH = Path(os.environ.get("OCULUS_PLAN_PATH", str(_DEFAULT_PLANS_DIR / "master_oculus_plan_8_14.md")))
_CACHE_PATH = Path(os.environ.get("OCULUS_PLAN_CACHE_PATH", str(_DEFAULT_PLANS_DIR / ".master_oculus_plan_8_14.cache.json")))


def _plan_digest(path: Path) -> str:
    """Return a content hash for the plan file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step_has_rollback(lines):
    """Return True if any line in the step contains a rollback command marker."""
    for line in lines:
        if "rollback_command" in line or "rollback-command" in line:
            return True
    return False


def _parse_plan_ast(text: str) -> dict:
    """Build a lightweight compiled-AST representation of the markdown plan.

    Instead of repeatedly regex-scanning the whole file on every gate run, the
    parsed step headings and rollback presence are cached as JSON. A fresh cache
    hit lets verify_plan_gate skip the expensive string scanning entirely.
    """
    steps = []
    current_heading = None
    current_lines = []
    for line in text.splitlines():
        if line.startswith("## ") or line.startswith("# "):
            if current_heading is not None:
                steps.append({
                    "heading": current_heading,
                    "has_rollback": _step_has_rollback(current_lines),
                })
            current_heading = line.lstrip("# ").strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_heading is not None:
        steps.append({
            "heading": current_heading,
            "has_rollback": _step_has_rollback(current_lines),
        })
    return {"steps": steps}


def _load_plan_cache(path: Path):
    """Return cached validation data if the cache is fresh for the plan file."""
    try:
        cache = json.loads(_CACHE_PATH.read_text())
        if isinstance(cache, dict) and cache.get("path") == str(path) and cache.get("digest") == _plan_digest(path):
            return cache
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _store_plan_cache(path: Path, errors):
    """Persist the compiled plan AST and its validation errors to JSON."""
    try:
        cache = {
            "path": str(path),
            "digest": _plan_digest(path),
            "mtime": path.stat().st_mtime,
            "compiled_ast": _parse_plan_ast(path.read_text()),
            "rollback_errors": errors,
        }
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(cache, indent=2, sort_keys=True)
        tmp_path = _CACHE_PATH.with_name(_CACHE_PATH.name + ".tmp")
        fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        os.replace(tmp_path, _CACHE_PATH)
    except OSError:
        pass


def main():
    """Run the plan gate verification."""
    # 1. Validate rollback_command in every plan step
    plan_path = _PLAN_PATH
    cache = _load_plan_cache(plan_path)
    if cache is not None:
        errors = list(cache.get("rollback_errors") or [])
    else:
        errors = validate_rollback_commands(plan_path)
        errors = [str(error) for error in (errors or [])]
        _store_plan_cache(plan_path, errors)
    if errors:
        print("Plan gate verification failed: missing rollback_command in steps:", file=sys.stderr)
        for err in errors:
            print(f"  {err}", file=sys.stderr)
        return 1
    print("All plan steps have rollback_command.")

    # 2. Run the test suite
    try:
        result = run_tests()
        if result:
            print("Plan gate verification passed.")
            return 0
        else:
            print("Plan gate verification failed (tests).", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Error running plan gate: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
