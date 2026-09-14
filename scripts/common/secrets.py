"""Unified secret loading and redaction (STEP 286/1566).

Single source of truth for API-credential handling: every loader in the repo
behaves identically and no code path can accidentally print a raw secret.
Fail-closed by design — a missing secret raises instead of degrading to an
empty string, so a misconfigured environment never silently ships an empty
Authorization header.
"""
import fcntl
import json
import logging
import os
import stat
from pathlib import Path
from typing import Optional, Sequence, Union

KeyFile = Union[str, Path, None]

# Candidate JSON fields tried, in order, when a key file parses as an object.
_DEFAULT_JSON_FIELDS = ('api_key', 'key', 'DEEPSEEK_API_KEY')


def _check_key_file_permissions(path: Path) -> None:
    """Refuse to read a secret file that any other user could access.

    STEP 367/1566: the file must be owned by the current user and carry no
    group/other permission bits (mode & 0o077 == 0). A world- or group-readable
    key file is the same as a leaked key, so this fails closed instead of
    proceeding with a weak file.
    Uses atomic fd-based check to avoid TOCTOU races.
    """
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        mode = stat.S_IMODE(st.st_mode)
        if st.st_uid != os.getuid() or mode & 0o077:
            raise RuntimeError(
                f'refusing insecure key file permissions on {path}: '
                f'owner uid {st.st_uid} (expected {os.getuid()}), '
                f'mode {oct(mode)} (expected no group/other bits)')
    finally:
        os.close(fd)


def load_secret(env_name: str,
                key_file: Optional[KeyFile] = None,
                json_fields: Sequence[str] = _DEFAULT_JSON_FIELDS) -> str:
    """Return a secret from the environment or a key file; raise when missing.

    The environment variable wins (stripped); a whitespace-only variable
    counts as unset. Otherwise the key file is read: a JSON object yields the
    first present field from ``json_fields``, a JSON array yields its first
    element, and any other content is treated as a plaintext key. Missing
    everywhere raises RuntimeError — never an empty or None fallback.
    """
    value = os.environ.get(env_name)
    if value and value.strip():
        return value.strip()
    if key_file is not None:
        path = Path(key_file)
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
        try:
            st = os.fstat(fd)
            mode = stat.S_IMODE(st.st_mode)
            if st.st_uid != os.getuid() or mode & 0o077:
                raise RuntimeError(
                    f'refusing insecure key file permissions on {path}: '
                    f'owner uid {st.st_uid} (expected {os.getuid()}), '
                    f'mode {oct(mode)} (expected no group/other bits)')
            content = os.read(fd, 65536).decode('utf-8').strip()
            if not content:
                raise RuntimeError(f'empty key file: {path}')
            try:
                data = json.loads(content)
            except ValueError:
                data = None
            if isinstance(data, dict):
                # Explicit candidate fields first (preserves the cross-eval
                # loader's api_key → key → DEEPSEEK_API_KEY precedence),
                # then the env name itself as a fallback field name.
                for field in (*json_fields, env_name):
                    candidate = data.get(field)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
                raise RuntimeError(
                    f'no recognized key field in {env_name} key file: {path}')
            if isinstance(data, list) and data and isinstance(data[0], str):
                stripped = data[0].strip()
                if not stripped:
                    raise RuntimeError(f'empty key in {env_name} key file array: {path}')
                return stripped
            return content
        finally:
            os.close(fd)
    raise RuntimeError('missing required secret: ' + env_name)


def redact(value) -> str:
    """Return a redacted display form of a secret; empty values stay empty."""
    if not value:
        return ''
    return '***'


def log_redacted(logger: logging.Logger, message: str, secret: str) -> None:
    """Log ``message`` with the secret substituted already redacted.

    ``message`` must contain a ``%s`` placeholder for the redacted secret.
    """
    logger.info(message, redact(secret))
