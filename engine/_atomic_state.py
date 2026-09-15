#!/usr/bin/env python3
"""Atomic state management for inbox monitor and wake scripts.

Provides atomic read/write operations for seen-state and offset persistence
with file locking to prevent concurrent corruption.
"""

import json
import os
import tempfile
import fcntl
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


class AtomicState:
    """Atomic state manager with file locking."""

    def __init__(self, state_file: str):
        self.state_file = Path(state_file)
        self.lock_file = Path(f"{state_file}.lock")
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create parent directories if they don't exist."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

    def _acquire_lock(self) -> None:
        """Acquire exclusive lock on the state file."""
        # Use a separate lock file for cross-process safety
        lock_fd = open(self.lock_file, 'w')
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # Keep the lock file descriptor open to hold the lock
        self._lock_fd = lock_fd

    def _release_lock(self) -> None:
        """Release the lock."""
        if hasattr(self, '_lock_fd'):
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            self._lock_fd.close()
            del self._lock_fd

    def _fsync_dir(self) -> None:
        """fsync the state file's directory so a completed os.replace is durable.

        Without this a crash immediately after the rename can leave the new
        contents non-durable even though the rename returned success, which is
        the remaining durability gap in the atomic-write path. Best-effort:
        platforms that cannot open a directory for fsync are ignored.
        """
        dir_fd = None
        try:
            dir_fd = os.open(str(self.state_file.parent), os.O_RDONLY)
            os.fsync(dir_fd)
        except OSError:
            pass
        finally:
            if dir_fd is not None:
                os.close(dir_fd)

    def read(self) -> Dict[str, Any]:
        """Read state atomically with lock.

        Returns:
            Dict containing state data. Returns empty dict only if the file
            does not exist. Corrupt or unreadable files raise RuntimeError.
        """
        self._acquire_lock()
        try:
            if not self.state_file.exists():
                return {}
            with open(self.state_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError as exc:
            backup_fd, backup_name = tempfile.mkstemp(
                prefix=f"{self.state_file.name}.corrupt.",
                suffix=".bak",
                dir=self.state_file.parent,
            )
            try:
                os.close(backup_fd)
                backup_path = Path(backup_name)
                shutil.copy2(self.state_file, backup_path)
            except Exception:
                try:
                    os.close(backup_fd)
                except OSError:
                    pass
                try:
                    os.unlink(backup_name)
                except OSError:
                    pass
                raise
            raise RuntimeError(
                f"Corrupt state file {self.state_file}; backed up to {backup_path}"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"Unable to read state file {self.state_file}") from exc
        finally:
            self._release_lock()

    def write(self, data: Dict[str, Any]) -> bool:
        """Write state atomically with lock and os.replace.
        
        Args:
            data: State data to write.
            
        Returns:
            True if write succeeded, False otherwise.
        """
        self._acquire_lock()
        tmp_path = None
        try:
            # Create a temporary file in the same directory as state_file for atomic replace
            with tempfile.NamedTemporaryFile(mode='w', dir=self.state_file.parent, delete=False) as tmp_file:
                json.dump(data, tmp_file, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                tmp_path = tmp_file.name
            # Atomic replace using os.replace (renames atomically, replaces existing)
            os.replace(tmp_path, str(self.state_file))
            # fsync the containing directory so the rename itself is durable
            self._fsync_dir()
            return True
        except (IOError, OSError, ValueError):
            # Clean up temporary file if it exists and an error occurred
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return False
        finally:
            self._release_lock()

    def _locked_update(self, modifier: callable) -> bool:
        """Perform read-modify-write atomically under a single lock.
        
        Args:
            modifier: Callable that takes the current state dict and modifies it in place.
            Should return True to write, False to abort.
            
        Returns:
            True if update succeeded, False otherwise.
        """
        self._acquire_lock()
        tmp_path = None
        try:
            if not self.state_file.exists():
                data = {}
            else:
                with open(self.state_file, 'r') as f:
                    try:
                        data = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        data = {}
            
            if not modifier(data):
                return False
            
            # Create a temporary file in the same directory as state_file for atomic replace
            with tempfile.NamedTemporaryFile(mode='w', dir=self.state_file.parent, delete=False) as tmp_file:
                json.dump(data, tmp_file, indent=2)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                tmp_path = tmp_file.name
            # Atomic replace using os.replace (renames atomically, replaces existing)
            os.replace(tmp_path, str(self.state_file))
            # fsync the containing directory so the rename itself is durable
            self._fsync_dir()
            return True
        except (IOError, OSError, ValueError):
            # Clean up temporary file if it exists and an error occurred
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return False
        finally:
            self._release_lock()

    def update(self, key: str, value: Any) -> bool:
        """Update a single key atomically.
        
        Args:
            key: Key to update.
            value: Value to set.
            
        Returns:
            True if update succeeded, False otherwise.
        """
        return self._locked_update(lambda data: data.__setitem__(key, value) or True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from state.
        
        Args:
            key: Key to retrieve.
            default: Default value if key not found.
            
        Returns:
            Value or default.
        """
        data = self.read()
        return data.get(key, default)


class AtomicOffsetState(AtomicState):
    """Specialized atomic state for tracking offsets."""

    def __init__(self, state_file: str = ".inbox_state.json"):
        super().__init__(state_file)

    def get_seen_state(self) -> Dict[str, int]:
        """Get seen state dictionary.
        
        Returns:
            Dict mapping message IDs to processed offsets.
        """
        return self.read().get("seen", {})

    def update_seen_state(self, message_id: str, offset: int) -> bool:
        """Update seen state for a message.
        
        Args:
            message_id: Message identifier.
            offset: Processed offset.
            
        Returns:
            True if update succeeded, False otherwise.
        """
        return self._locked_update(lambda data: data.setdefault("seen", {}).__setitem__(message_id, offset) or True)

    def get_last_offset(self, channel: str) -> int:
        """Get last processed offset for a channel.
        
        Args:
            channel: Channel name.
            
        Returns:
            Last processed offset (0 if not found).
        """
        data = self.read()
        return data.get("offsets", {}).get(channel, 0)

    def update_last_offset(self, channel: str, offset: int) -> bool:
        """Update last processed offset for a channel.
        
        Args:
            channel: Channel name.
            offset: New offset value.
            
        Returns:
            True if update succeeded, False otherwise.
        """
        return self._locked_update(lambda data: data.setdefault("offsets", {}).__setitem__(channel, offset) or True)


    def get_seen_state_atomic(self) -> Dict[str, int]:
        """Get seen state dictionary atomically.
        
        Returns:
            Dict mapping message IDs to processed offsets.
        """
        return self.read().get("seen", {})

    def get_last_offset_atomic(self, channel: str) -> int:
        """Get last processed offset for a channel atomically.
        
        Args:
            channel: Channel name.
            
        Returns:
            Last processed offset (0 if not found).
        """
        data = self.read()
        return data.get("offsets", {}).get(channel, 0)

    def get_seen_state_atomic(self) -> Dict[str, int]:
        """Get seen state dictionary atomically.
        
        Returns:
            Dict mapping message IDs to processed offsets.
        """
        return self.read().get("seen", {})

    def get_last_offset_atomic(self, channel: str) -> int:
        """Get last processed offset for a channel atomically.
        
        Args:
            channel: Channel name.
            
        Returns:
            Last processed offset (0 if not found).
        """
        data = self.read()
        return data.get("offsets", {}).get(channel, 0)

    def get_seen_state_atomic(self) -> Dict[str, int]:
        """Get seen state dictionary atomically.
        
        Returns:
            Dict mapping message IDs to processed offsets.
        """
        return self.read().get("seen", {})

    def get_last_offset_atomic(self, channel: str) -> int:
        """Get last processed offset for a channel atomically.
        
        Args:
            channel: Channel name.
            
        Returns:
            Last processed offset (0 if not found).
        """
        data = self.read()
        return data.get("offsets", {}).get(channel, 0)
