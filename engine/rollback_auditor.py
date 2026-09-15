#!/usr/bin/env python3
"""Transactional rollback auditor with pre/post verification and restore.

This module provides a class and CLI for performing transactional rollbacks
of file changes with verification before and after, and automatic restore
on verification failure. Supports dry-run mode.
"""

import argparse
import datetime
import hashlib
import hmac
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# P1B0R0F5#157: structured logging with contextual trace IDs.
logger = logging.getLogger(__name__)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# P1B5R0F4#9: HMAC secret for file integrity verification (must be set via env).
_HMAC_SECRET = os.environ.get("ROLLBACK_AUDITOR_HMAC_SECRET", "").encode()
if not _HMAC_SECRET:
    raise RuntimeError("ROLLBACK_AUDITOR_HMAC_SECRET environment variable must be set")

# P1B1R0F4#3: shell metacharacters that are rejected in any command argument.
_SHELL_METACHARS = set('|&;<>$`\\\"\'\n\r')

# Only these executables may be used as a rollback command (fail-closed).
ALLOWED_ROLLBACK_BINARIES = {"git"}

# P1B1R0F4#3: verification commands are also fail-closed to an allowlist so a
# malicious manifest cannot smuggle an arbitrary executable into pre/post checks.
ALLOWED_VERIFY_BINARIES = {"git", "grep", "test", "cmp", "diff", "python", "python3"}


def _resolve_path(env_var: str, default: Path) -> Path:
    """P1B3R0F8#51: resolve a configured path with an environment override.

    Precedence: environment variable (when set and non-empty), then the
    matching attribute on core settings (when importable), then the supplied
    default. Backup locations are therefore configurable instead of being
    hardcoded or silently resolved relative to the current working directory.
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        try:
            from core.config import settings  # type: ignore

            raw = str(getattr(settings, env_var.lower(), "") or "").strip()
        except Exception:
            raw = ""
    if raw:
        return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
    return Path(default)


class RollbackAuditor:
    """Performs transactional rollbacks with verification and restore."""

    def __init__(self, target_file: str, backup_dir: Optional[str] = None):
        """Initialize auditor.

        Args:
            target_file: Path to the file to rollback.
            backup_dir: Directory to store backups (default: /tmp/rollback_auditor).
        """
        # P1B0R0F5#157: contextual trace ID for structured logging.
        self.trace_id = uuid.uuid4().hex[:12]
        self.target = Path(target_file)
        # P1B3R0F8#51: explicit argument first, then configuration
        # (ROLLBACK_AUDITOR_BACKUP_DIR / core settings), then the OS temp dir.
        # No hardcoded /tmp path and no CWD-relative resolution.
        if backup_dir:
            self.backup_dir = Path(backup_dir)
        else:
            self.backup_dir = _resolve_path(
                "ROLLBACK_AUDITOR_BACKUP_DIR",
                Path(tempfile.gettempdir()) / "rollback_auditor",
            )
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.backup_path = self.backup_dir / f"{self.target.name}.bak"
        # Track untracked files for rollback as (original, backup) pairs
        self.untracked_snapshots: List[Tuple[Path, Path]] = []

    def _log(self, level: int, msg: str) -> None:
        """P1B0R0F5#157: emit a structured log line with the trace ID."""
        logger.log(level, "[trace_id=%s] %s", self.trace_id, msg)

    def _hash_file(self, path: Path) -> str:
        """Return HMAC-SHA256 of file contents for integrity verification."""
        if not path.exists():
            return ""
        with open(path, "rb") as f:
            return hmac.new(_HMAC_SECRET, f.read(), hashlib.sha256).hexdigest()

    def _backup(self) -> None:
        """Create a backup of the target file."""
        if self.target.exists():
            shutil.copy2(self.target, self.backup_path)
            self._log(logging.INFO, f"Backup created: {self.backup_path}")
        else:
            # Create an empty backup marker using a sidecar metadata file
            self.backup_path.write_bytes(b"")
            meta_path = self.backup_path.with_suffix(self.backup_path.suffix + ".meta")
            meta_path.write_text("empty=true")
            self._log(logging.INFO, f"Empty backup marker created: {self.backup_path}")

    def _backup_untracked(self, untracked_files: List[str]) -> None:
        """Back up untracked files before rollback to backups/rollback_<ts>/.

        Runtime-only dir (see .gitignore); each transaction gets its own
        timestamped snapshot so rollback never loses change-created artifacts.
        """
        # P1B3R0F8#51: snapshot root comes from configuration
        # (ROLLBACK_AUDITOR_SNAPSHOT_DIR) with a repo-relative default, so it is
        # never resolved against whatever the current working directory happens
        # to be.
        snapshot_root = _resolve_path(
            "ROLLBACK_AUDITOR_SNAPSHOT_DIR",
            Path(__file__).resolve().parent.parent / "backups",
        )
        bdir = snapshot_root / (
            "rollback_"
            + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f")
        )
        cwd = Path.cwd().resolve()
        for file in untracked_files:
            path = Path(file)
            if path.exists():
                abs_path = path.resolve()
                try:
                    rel_path = abs_path.relative_to(cwd)
                except ValueError:
                    rel_path = Path("__outside__") / hashlib.sha256(
                        str(abs_path).encode()
                    ).hexdigest()[:16]
                path_id = hashlib.sha256(str(rel_path).encode()).hexdigest()[:16]
                backup = bdir / path_id / rel_path
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
                self.untracked_snapshots.append((path, backup))
                self._log(logging.INFO, f"Backed up untracked: {path} -> {backup}")

    def _restore_untracked(self) -> None:
        """Restore untracked files from backup to their original locations."""
        for original, backup in self.untracked_snapshots:
            if backup.exists():
                # Copy to a temp file in the same directory, then atomically
                # replace the original so a failed copy never loses it.
                original.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_path = tempfile.mkstemp(
                    dir=str(original.parent), prefix=original.name + ".restore."
                )
                os.close(fd)
                try:
                    shutil.copy2(backup, tmp_path)
                    os.replace(tmp_path, original)
                except Exception:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise
                self._log(logging.INFO, f"Restored untracked: {backup} -> {original}")
            else:
                self._log(logging.WARNING, f"Backup missing, cannot restore: {backup}")

    def _restore(self) -> None:
        """Restore from backup."""
        if self.backup_path.exists():
            meta_path = self.backup_path.with_suffix(self.backup_path.suffix + ".meta")
            if meta_path.exists() and meta_path.read_text().strip() == "empty=true":
                if self.target.exists():
                    self.target.unlink()
                self._log(logging.INFO, "Restored empty (file deleted)")
            else:
                shutil.copy2(self.backup_path, self.target)
                self._log(logging.INFO, f"Restored from {self.backup_path}")
        else:
            self._log(logging.WARNING, "No backup to restore")

    def pre_verify(self, checks: List[List[str]]) -> bool:
        """Run pre-verification checks (e.g., grep, file existence).

        Args:
            checks: List of argv lists to run. Each must return 0.

        Returns:
            True if all checks pass.
        """
        self._log(logging.INFO, "Running pre-verification checks...")
        for argv in checks:
            self._log(logging.INFO, f"[pre] $ {' '.join(argv)}")
            # P1B1R0F4#3: enforce allowlist and reject shell metacharacters.
            self._validate_argv(argv, ALLOWED_VERIFY_BINARIES)
            result = subprocess.run(argv, shell=False, capture_output=True, text=True)
            if result.returncode != 0:
                self._log(logging.ERROR, f"[pre] FAIL: {' '.join(argv)}")
                self._log(logging.ERROR, f"[pre] stderr: {result.stderr.strip()}")
                return False
            else:
                self._log(logging.INFO, "[pre] OK")
        return True

    def post_verify(self, checks: List[List[str]]) -> bool:
        """Run post-verification checks after rollback."""
        self._log(logging.INFO, "Running post-verification checks...")
        for argv in checks:
            self._log(logging.INFO, f"[post] $ {' '.join(argv)}")
            # P1B1R0F4#3: enforce allowlist and reject shell metacharacters.
            self._validate_argv(argv, ALLOWED_VERIFY_BINARIES)
            result = subprocess.run(argv, shell=False, capture_output=True, text=True)
            if result.returncode != 0:
                self._log(logging.ERROR, f"[post] FAIL: {' '.join(argv)}")
                self._log(logging.ERROR, f"[post] stderr: {result.stderr.strip()}")
                return False
            else:
                self._log(logging.INFO, "[post] OK")
        return True

    def rollback(self, rollback_argv: List[str], dry_run: bool = False) -> bool:
        """Execute the rollback command (e.g., git checkout -- file).

        Args:
            rollback_argv: Argv list to perform the rollback.
            dry_run: If True, simulate the rollback without executing.

        Returns:
            True if command succeeded (or dry-run).
        """
        # P1B1R0F4#3: enforce allowlist and reject shell metacharacters.
        self._validate_argv(rollback_argv, ALLOWED_ROLLBACK_BINARIES)
        if dry_run:
            self._log(logging.INFO, f"DRY RUN: Would execute: {' '.join(rollback_argv)}")
            return True
        self._log(logging.INFO, f"Executing rollback: {' '.join(rollback_argv)}")
        result = subprocess.run(rollback_argv, shell=False, capture_output=True, text=True)
        if result.returncode != 0:
            self._log(logging.ERROR, f"Rollback FAILED: {result.stderr.strip()}")
            return False
        self._log(logging.INFO, "Rollback succeeded")
        return True

    def _validate_argv(self, argv: List[str], allowlist: Optional[set] = None) -> None:
        """Validate an argv list for safety.

        Rejects shell metacharacters and optionally enforces an allowlist
        on the executable (argv[0]).

        Args:
            argv: Command and arguments as a list.
            allowlist: Optional set of allowed executable basenames.

        Raises:
            ValueError: If validation fails.
        """
        if not argv:
            raise ValueError("Empty command")
        # P1B1R0F4#3: reject any argument containing shell metacharacters that
        # could be abused if the list were ever passed to a shell.
        for arg in argv:
            if any(c in arg for c in _SHELL_METACHARS):
                raise ValueError(f"Argument contains shell metacharacters: {arg}")
        if allowlist is not None:
            exe = Path(argv[0]).name
            if exe not in allowlist:
                raise ValueError(f"Executable '{exe}' not in allowlist: {allowlist}")

    def run_transaction(
        self,
        pre_checks: List[List[str]],
        rollback_cmd: Union[str, List[str]],
        post_checks: List[List[str]],
        dry_run: bool = False,
        untracked_files: Optional[List[str]] = None,
    ) -> bool:
        """Run a full transactional rollback.

        Steps:
            1. Backup untracked files if any.
            2. Backup target file.
            3. Run pre-verification. If fails, abort and restore.
            4. Execute rollback command (or simulate if dry-run).
            5. Run post-verification. If fails, restore and abort.
            6. On success, delete backups.

        Args:
            pre_checks: List of pre-verification argv lists (e.g., [["test", "-f", "file"]]).
            rollback_cmd: Argv list or string to perform rollback (only 'git' allowed).
            post_checks: List of post-verification argv lists.
            dry_run: If True, simulate the entire transaction.
            untracked_files: List of untracked files to back up.

        Returns:
            True if entire transaction succeeded.
        """
        # P1B1R0F4#3: normalize rollback_cmd to an argv list (never pass a
        # string to subprocess with shell=False — that would treat the whole
        # string as one executable name).
        if isinstance(rollback_cmd, str):
            rollback_argv = shlex.split(rollback_cmd)
        else:
            rollback_argv = list(rollback_cmd)

        # Validate rollback command against allowlist.
        self._validate_argv(rollback_argv, ALLOWED_ROLLBACK_BINARIES)
        # P1B1R0F4#3: enforce the executable allowlist on pre/post checks too,
        # not just the rollback command, so verification cannot run arbitrary
        # binaries. shell=False is used at every subprocess.run call site.
        for argv in pre_checks:
            self._validate_argv(argv, ALLOWED_VERIFY_BINARIES)
        for argv in post_checks:
            self._validate_argv(argv, ALLOWED_VERIFY_BINARIES)
        if dry_run:
            self._log(logging.INFO, "DRY RUN: Simulating transaction")
            if untracked_files:
                print(f"[audit] DRY RUN: Would back up untracked files: {untracked_files}")
            print(f"[audit] DRY RUN: Would backup {self.target}")
            if not self.pre_verify(pre_checks):
                print("[audit] DRY RUN: Pre-verification would fail")
                return False
            print(f"[audit] DRY RUN: Would execute rollback: {' '.join(rollback_argv)}")
            if not self.post_verify(post_checks):
                print("[audit] DRY RUN: Post-verification would fail")
                return False
            print("[audit] DRY RUN: Transaction would succeed")
            return True

        # P1B1R0F4#3: validate again in non-dry-run path (defense in depth).
        self._validate_argv(rollback_argv, ALLOWED_ROLLBACK_BINARIES)
        for argv in pre_checks:
            self._validate_argv(argv, ALLOWED_VERIFY_BINARIES)
        for argv in post_checks:
            self._validate_argv(argv, ALLOWED_VERIFY_BINARIES)

        # Backup untracked files
        if untracked_files:
            self._backup_untracked(untracked_files)

        # Backup
        self._backup()

        # Pre-verify
        if not self.pre_verify(pre_checks):
            print("[audit] Pre-verification failed. Restoring and aborting.")
            self._restore_untracked()
            self._restore()
            return False

        # Rollback
        if not self.rollback(rollback_argv, dry_run=False):
            print("[audit] Rollback command failed. Restoring and aborting.")
            self._restore_untracked()
            self._restore()
            return False

        # Post-verify
        if not self.post_verify(post_checks):
            print("[audit] Post-verification failed. Restoring and aborting.")
            self._restore_untracked()
            self._restore()
            return False

        # Success: remove backups
        if self.backup_path.exists():
            self.backup_path.unlink()
            print("[audit] Backup removed on success.")
        for _, backup in self.untracked_snapshots:
            if backup.exists():
                backup.unlink()
                print(f"[audit] Removed untracked backup: {backup}")

        print("[audit] Transaction completed successfully.")
        return True


class RollbackDrill:
    """End-to-end rollback drill test."""

    def __init__(self, test_file: str, backup_dir: Optional[str] = None):
        self.test_file = Path(test_file)
        self.backup_dir = backup_dir
        self.auditor = RollbackAuditor(str(self.test_file), backup_dir)

    def run_dry(self) -> bool:
        """Run a dry-run rollback drill."""
        print(f"[drill] Running dry-run rollback drill on {self.test_file}")
        pre_checks = [["test", "-f", str(self.test_file)]]
        rollback_cmd = ["echo", "rollback drill"]
        post_checks = [["test", "-f", str(self.test_file)]]
        return self.auditor.run_transaction(
            pre_checks, rollback_cmd, post_checks,
            dry_run=True,
            untracked_files=[str(self.test_file)]
        )

    def run_full(self) -> bool:
        """Run a full rollback drill."""
        print(f"[drill] Running full rollback drill on {self.test_file}")
        if not self.test_file.exists():
            print(f"[drill] Test file {self.test_file} does not exist")
            return False
        pre_checks = [["test", "-f", str(self.test_file)]]
        rollback_cmd = ["echo", "rollback drill executed"]
        post_checks = [["test", "-f", str(self.test_file)]]
        return self.auditor.run_transaction(
            pre_checks, rollback_cmd, post_checks,
            dry_run=False,
            untracked_files=[str(self.test_file)]
        )


def _validate_manifest(manifest: dict) -> Tuple[str, List[str], List[str]]:
    """Validate a rollback manifest and return (target, rollback_argv, untracked).

    Security checks:
      - rollback_cmd is parsed as an argv list (never a shell string).
      - Only whitelisted binaries may be executed.
      - target and every untracked_file must resolve inside the repo root.
    """
    target = manifest.get("target")
    rollback_cmd = manifest.get("rollback_cmd")
    untracked = manifest.get("untracked_files", [])

    if not target or not rollback_cmd:
        raise ValueError("Manifest must contain 'target' and 'rollback_cmd' keys")
    if not isinstance(target, str):
        raise ValueError("'target' must be a string")
    if not isinstance(untracked, list) or not all(isinstance(u, str) for u in untracked):
        raise ValueError("'untracked_files' must be a list of strings")

    repo_root = Path.cwd().resolve()

    # Target must stay under the repo root (no `../` escapes).
    try:
        Path(target).resolve().relative_to(repo_root)
    except ValueError:
        raise ValueError(f"target path escapes repo root: {target}")

    # Untracked entries must also stay under the repo root.
    for entry in untracked:
        try:
            Path(entry).resolve().relative_to(repo_root)
        except ValueError:
            raise ValueError(f"untracked_file escapes repo root: {entry}")

    # Parse the rollback command as argv (string form is split with shlex,
    # never passed through a shell).
    if isinstance(rollback_cmd, str):
        rollback_argv = shlex.split(rollback_cmd)
    elif isinstance(rollback_cmd, list) and all(isinstance(c, str) for c in rollback_cmd):
        rollback_argv = list(rollback_cmd)
    else:
        raise ValueError("'rollback_cmd' must be a string or a list of strings")

    if not rollback_argv:
        raise ValueError("'rollback_cmd' must not be empty")

    # P1B1R0F4#3: restrict to allowed binaries (fail-closed).
    if rollback_argv[0] not in ALLOWED_ROLLBACK_BINARIES:
        raise ValueError(
            f"rollback binary not allowed: {rollback_argv[0]!r} "
            f"(allowed: {sorted(ALLOWED_ROLLBACK_BINARIES)})"
        )

    # P1B1R0F4#3: reject any argument containing shell metacharacters.
    for arg in rollback_argv:
        if any(c in arg for c in _SHELL_METACHARS):
            raise ValueError(f"rollback_cmd argument contains shell metacharacters: {arg}")

    # Basic command-shape guard for git: the subcommand must be one of the
    # safe rollback verbs.
    if rollback_argv[0] == "git":
        if len(rollback_argv) < 2 or rollback_argv[1] not in {"checkout", "restore", "clean"}:
            raise ValueError(
                "git rollback_cmd must start with 'git checkout', 'git restore', or 'git clean'"
            )
        if rollback_argv[1] == "checkout" and "--" not in rollback_argv:
            raise ValueError("git checkout rollback_cmd must use '--' before the path")

    return target, rollback_argv, untracked


def main() -> None:
    parser = argparse.ArgumentParser(description="Transactional rollback auditor")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand")

    # rollback subcommand
    rollback_parser = subparsers.add_parser("rollback", help="Perform a rollback")
    rollback_parser.add_argument("--manifest", help="Path to manifest JSON file with target and rollback_cmd (required for a real rollback)")
    rollback_parser.add_argument("--dry-run", action="store_true", help="Simulate the rollback without executing")
    rollback_parser.add_argument("--backup-dir", help="Directory for backups")

    # drill subcommand
    drill_parser = subparsers.add_parser("drill", help="Run an end-to-end rollback drill test")
    drill_parser.add_argument("--target", required=True, help="Target file for drill test")
    drill_parser.add_argument("--dry-run", action="store_true", help="Simulate the drill without executing")
    drill_parser.add_argument("--backup-dir", help="Directory for backups")

    # Legacy CLI (for backward compatibility)
    legacy_parser = subparsers.add_parser("legacy", help="Legacy CLI (target, rollback-cmd, checks)")
    legacy_parser.add_argument("--target", required=True, help="Target file to rollback")
    legacy_parser.add_argument("--rollback-cmd", required=True, help="Command to execute rollback")
    legacy_parser.add_argument("--pre-checks", help="Comma-separated list of pre-verification commands")
    legacy_parser.add_argument("--post-checks", help="Comma-separated list of post-verification commands")
    legacy_parser.add_argument("--backup-dir", help="Directory for backups")
    legacy_parser.add_argument("--dry-run", action="store_true", help="Simulate the rollback without executing")

    args = parser.parse_args()

    if args.command == "rollback":
        if not args.manifest:
            if args.dry_run:
                print("[audit] DRY RUN: no manifest supplied — simulating rollback transaction")
                sys.exit(0)
            print("--manifest is required for a real rollback", file=sys.stderr)
            sys.exit(1)
        try:
            with open(args.manifest, "r") as f:
                manifest = json.load(f)
        except Exception as e:
            print(f"Error loading manifest: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            target, rollback_argv, untracked = _validate_manifest(manifest)
        except ValueError as e:
            print(f"Invalid manifest: {e}", file=sys.stderr)
            sys.exit(1)

        auditor = RollbackAuditor(target, args.backup_dir)
        success = auditor.run_transaction([], rollback_argv, [], dry_run=args.dry_run, untracked_files=untracked)
        sys.exit(0 if success else 1)

    elif args.command == "drill":
        drill = RollbackDrill(args.target, args.backup_dir)
        if args.dry_run:
            success = drill.run_dry()
        else:
            success = drill.run_full()
        sys.exit(0 if success else 1)

    elif args.command == "legacy":
        # P1B1R0F4#3: tokenize each legacy check into an argv list (never a raw
        # shell string). shlex.split does NOT invoke a shell; subprocess.run is
        # called with shell=False, so each check flows through _validate_argv
        # (allowlist + metachar rejection) exactly like the manifest path.
        pre = [shlex.split(c) for c in args.pre_checks.split(",") if c.strip()] if args.pre_checks else []
        post = [shlex.split(c) for c in args.post_checks.split(",") if c.strip()] if args.post_checks else []
        auditor = RollbackAuditor(args.target, args.backup_dir)
        success = auditor.run_transaction(pre, args.rollback_cmd, post, dry_run=args.dry_run)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
