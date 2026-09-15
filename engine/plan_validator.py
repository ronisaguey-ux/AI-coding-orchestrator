"""Validators for the plan gate: compile-sweep, pre-existing failure tracking, and rollback_command enforcement."""
from pathlib import Path
import py_compile
import re

ROOT = Path(__file__).resolve().parent.parent
PRE_EXISTING_FAILURES: set[str] = set()  # Empty now that Step 28 fixed test_jit_probability_execution_units

# Bounds (P1B0R0F7#37): regex work on plan files is bounded so a pathological
# or hostile plan cannot cause runaway backtracking / memory use. Files larger
# than MAX_PLAN_BYTES are refused outright; individual lines longer than
# MAX_LINE_LEN are rejected before being fed to any regex.
MAX_PLAN_BYTES = 4 * 1024 * 1024  # 4 MiB
MAX_LINE_LEN = 100_000


def get_pre_existing_failures() -> set[str]:
    """Return the set of known pre-existing test failures."""
    return PRE_EXISTING_FAILURES


def compile_sweep(root: Path | None = None) -> list[str]:
    """Compile all Python files under root (excluding venv/cache/archive/git).

    Returns a list of failure strings (empty if all compile cleanly).
    """
    root = root or ROOT
    failures: list[str] = []
    for py in sorted(root.rglob("*.py")):
        if any(part in {".venv", "__pycache__", "archive", ".git"} for part in py.parts):
            continue
        try:
            py_compile.compile(str(py), doraise=True)
        except py_compile.PyCompileError as exc:
            failures.append(f"COMPILE: {py}: {exc}")
    return failures


def validate_rollback_commands(plan_path: Path) -> list[str]:
    """Check that every plan step contains a non-empty rollback_command field
    with a safe command shape (no shell metacharacters).

    Returns a list of error strings (empty if all steps pass).
    """
    if not plan_path.exists():
        return [f"Plan file not found: {plan_path}"]
    size = plan_path.stat().st_size
    if size > MAX_PLAN_BYTES:
        return [
            f"Plan file too large: {size} bytes exceeds limit {MAX_PLAN_BYTES} "
            f"({plan_path})"
        ]
    text = plan_path.read_text(encoding="utf-8", errors="replace")
    # Reject over-long lines before any regex runs (bounded work per line).
    for lineno, line in enumerate(text.splitlines(), 1):
        if len(line) > MAX_LINE_LEN:
            return [
                f"Plan line {lineno} too long: {len(line)} chars exceeds "
                f"limit {MAX_LINE_LEN}"
            ]
    step_header_re = re.compile(
        r"^[ \t]{0,3}(?:#{1,6}[ \t]+)?(?:step[ \t]+)?(\d+)"
        r"(?:[ \t]*/[ \t]*\d+|[ \t]+of[ \t]+\d+)?[ \t]*"
        r"(?:[:–—.)-]\s*)?.*$",
        re.MULTILINE | re.IGNORECASE,
    )
    step_headers = list(step_header_re.finditer(text))
    if not step_headers:
        return [
            "No plan steps found; expected a header such as "
            "'### STEP 1/10: ...', '## Step 1: ...', 'Step 1 ...', "
            "or '1. ...'"
        ]

    errors = []
    safe_re = re.compile(r'^[A-Za-z0-9_./-]+(?: [A-Za-z0-9_./:=-]+)*$')
    allowed_starts = ("git", "python", "python3", "bash", "sh", "rm", "cp", "mv",
                      "mkdir", "touch", "chmod", "chown", "curl", "wget",
                      "./", "scripts/", "/usr/bin/", "/bin/", "/usr/local/bin/")
    for i, match in enumerate(step_headers):
        step_num = int(match.group(1))
        start = match.end()
        end = step_headers[i+1].start() if i+1 < len(step_headers) else len(text)
        step_body = text[start:end]
        in_code_block = False
        cmd = None
        for line in step_body.splitlines():
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            m = re.match(r'^rollback_command\s*[:=]\s*(.+?)\s*$', stripped)
            if m:
                cmd = m.group(1).strip().strip('`').strip()
                break
        if cmd is None:
            errors.append(f"Step {step_num} missing rollback_command")
            continue
        if len(cmd) >= 2 and cmd[0] == cmd[-1] and cmd[0] in ('"', "'"):
            cmd = cmd[1:-1].strip()
        first = cmd.split()[0] if cmd.split() else ""
        if not cmd:
            errors.append(f"Step {step_num} rollback_command is empty")
        elif not safe_re.match(cmd) or not first.startswith(allowed_starts):
            errors.append(f"Step {step_num} rollback_command is not a valid command: {cmd[:80]}")
    return errors


def _run_self_test() -> int:
    """Internal self-check used by `python3 -m scripts.plan_validator --test`."""
    errors = compile_sweep()
    if errors:
        print("VALIDATOR SELF-TEST: FAIL")
        for err in errors:
            print(" ", err)
        return 1
    print("VALIDATOR SELF-TEST: PASS")
    return 0


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        raise SystemExit(_run_self_test())
    raise SystemExit("Usage: python3 -m scripts.plan_validator --test")
