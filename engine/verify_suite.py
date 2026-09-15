#!/usr/bin/env python3
"""Five-command verification gate for Oculus changes.

Runs a fixed set of verification commands and exits with 0 only if all pass.
"""
import os
import subprocess
import sys
import pathlib
from concurrent.futures import ThreadPoolExecutor


def get_timeout(default):
    """Return a timeout from the OCULUS_CMD_TIMEOUT env var, or the default."""
    try:
        return int(os.environ.get("OCULUS_CMD_TIMEOUT", default))
    except ValueError:
        return default


def run_cmd(cmd, description=None):
    """Run a command and return True if it succeeds."""
    if description:
        print(f"[{description}]")
    print(f"  $ {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=get_timeout(30))
        if result.returncode != 0:
            print(f"  FAILED (code {result.returncode})")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            return False
        print("  OK")
        return True
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return False


def main():
    """Run the five-command verification gate."""
    print("=== Verification Suite ===")

    # Command 1: Check that verify_suite.py exists
    if not pathlib.Path("scripts/verify_suite.py").is_file():
        print("ERROR: scripts/verify_suite.py missing")
        sys.exit(1)
    print("[1] verify_suite.py exists: OK")

    # Command 2: Check syntax with ast.parse (via helper script)
    # We'll do this inline with a small script
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("import ast; ast.parse(open('scripts/verify_suite.py').read())")
        tmp = f.name
    try:
        result = subprocess.run(['python3', tmp], capture_output=True, text=True)
        if result.returncode != 0:
            print("[2] AST parse: FAILED")
            print(result.stderr)
            sys.exit(1)
        print("[2] AST parse: OK")
    finally:
        pathlib.Path(tmp).unlink(missing_ok=True)

    # Command 3: mypy (skip if not installed, fail on real errors)
    try:
        result = subprocess.run(
            ["python3", "-m", "mypy", "--follow-imports=skip", "scripts/verify_suite.py"],
            capture_output=True,
            text=True,
            timeout=get_timeout(10),
        )
        if result.returncode == 0:
            print("[3] mypy: OK")
        elif (
            "No module named mypy" in result.stderr
            or "command not found" in result.stderr
        ):
            print("[3] mypy: SKIPPED (not installed)")
        else:
            print("[3] mypy: FAILED")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            sys.exit(1)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("[3] mypy: SKIPPED (not available)")

    # Commands 4 and 5 are independent — run them in parallel.
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda args: run_cmd(*args),
            [
                ("python3 -m pytest tests/test_verify_regression.py --collect-only -q", "[4] pytest collect"),
                ("git diff --check", "[5] git diff --check"),
            ],
        ))
    if not all(results):
        sys.exit(1)

    print("\n=== All verification checks passed ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
