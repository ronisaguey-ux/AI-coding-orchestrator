"""Runner for the plan gate: executes the full pytest suite."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# P1B0R0F4#180: test-suite timeout is configurable instead of hardcoded.
# Override with OCULUS_PLAN_RUNNER_TIMEOUT (seconds, int or float).
DEFAULT_TEST_TIMEOUT_SECONDS = 1200


def _test_timeout() -> float:
    """Return the pytest subprocess timeout in seconds from the environment."""
    raw = os.environ.get("OCULUS_PLAN_RUNNER_TIMEOUT")
    if raw is None or raw == "":
        return float(DEFAULT_TEST_TIMEOUT_SECONDS)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise SystemExit(
            f"invalid OCULUS_PLAN_RUNNER_TIMEOUT={raw!r}: must be a number of seconds"
        )
    if value <= 0:
        raise SystemExit(
            f"invalid OCULUS_PLAN_RUNNER_TIMEOUT={raw!r}: must be > 0"
        )
    return value


def run_tests(extra_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run the full pytest suite against the tests/ directory.

    Returns the raw CompletedProcess so callers can inspect stdout and returncode.
    On timeout, returns a CompletedProcess with returncode=124 and a timeout message.
    """
    args = [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no", "-p", "no:cacheprovider"]
    if extra_args:
        if not isinstance(extra_args, list):
            raise TypeError("extra_args must be a list of strings")
        args.extend(extra_args)
    timeout = _test_timeout()
    # shell=False is mandatory: args are passed as an explicit list to prevent
    # shell command injection via extra_args or any other user-controlled input.
    try:
        return subprocess.run(
            args,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as e:
        # Return a controlled failure result with timeout info
        stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr = e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        timeout_msg = f"\n=== TEST TIMEOUT after {timeout:g} seconds ==="
        return subprocess.CompletedProcess(
            args=args,
            returncode=124,
            stdout=stdout + timeout_msg,
            stderr=stderr + timeout_msg,
        )
    except FileNotFoundError as e:
        # pytest (or the Python interpreter) is missing; fail controlled.
        return subprocess.CompletedProcess(
            args=args,
            returncode=127,
            stdout="",
            stderr=f"=== TEST EXECUTABLE NOT FOUND: {e} ===",
        )
    except OSError as e:
        # Permission errors, fork failures, etc. — surface as a controlled result.
        return subprocess.CompletedProcess(
            args=args,
            returncode=126,
            stdout="",
            stderr=f"=== TEST EXECUTION FAILED: {e} ===",
        )


def _run_sampler() -> int:
    """Produce a sample summary line from the last test run (used by --sample)."""
    proc = run_tests()
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    print(tail)
    print(f"SAMPLE RETURNCODE: {proc.returncode}")
    return 0


if __name__ == "__main__":
    if "--sample" in sys.argv:
        raise SystemExit(_run_sampler())
    raise SystemExit("Usage: python3 -m scripts.plan_runner --sample")
