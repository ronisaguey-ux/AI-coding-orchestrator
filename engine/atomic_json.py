"""scripts/atomic_json.py — durable atomic JSON persistence (STEP 362/1566).

Shared by the Telegram daemons (telegram_auto_responder, telegram_ack_daemon).
Writes to a unique temp file in the destination directory, flushes + fsyncs the
data, atomically replaces the target, then fsyncs the directory so the rename
itself is durable. On any failure the temp file is removed and the exception
re-raised (fail-closed) — daemon callers already run inside a try/except loop
that logs and continues, so a torn write can never silently masquerade as a
completed state write.
"""
import errno
import json
import os
import shutil
import stat
import tempfile


def save_json(path, data):
    directory = os.path.dirname(os.path.abspath(path)) or '.'
    fd, tmp = tempfile.mkstemp(dir=directory, prefix='.atomic-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        # Set restrictive permissions (0600) on the temp file before rename
        os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        try:
            os.replace(tmp, path)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            # Cross-device link: fall back to shutil.move (copy + unlink).
            shutil.move(tmp, path)
            with open(path, 'rb') as fh:
                os.fsync(fh.fileno())
        # Ensure the final file has 0600 permissions as well
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        dfd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
