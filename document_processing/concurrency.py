from __future__ import annotations

import errno
import fcntl
import json
import os
import re
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from hermes_constants import get_default_hermes_root


_PROFILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class DocumentConcurrencyError(RuntimeError):
    pass


def _lock_root() -> Path:
    root = Path(get_default_hermes_root())
    cache = root / "cache"
    runtime = cache / "document-processing"
    locks = runtime / "locks"
    try:
        for directory in (root, cache, runtime, locks):
            if directory.is_symlink():
                raise DocumentConcurrencyError("Document lock directory must not be a link")
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        runtime.chmod(0o700)
        locks.chmod(0o700)
        resolved_root = root.resolve(strict=True)
        resolved_cache = cache.resolve(strict=True)
        resolved_runtime = runtime.resolve(strict=True)
        resolved_locks = locks.resolve(strict=True)
    except DocumentConcurrencyError:
        raise
    except (OSError, RuntimeError):
        raise DocumentConcurrencyError("Document lock root could not be prepared") from None
    if (
        resolved_cache.parent != resolved_root
        or resolved_runtime.parent != resolved_cache
        or resolved_locks.parent != resolved_runtime
    ):
        raise DocumentConcurrencyError("Document lock root failed path validation")
    return resolved_locks


def _paddle_lock_path() -> Path:
    return _lock_root() / "paddle-0.lock"


def _open_lock_file(path: Path) -> int:
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise DocumentConcurrencyError("Document lock file failed validation")
        os.fchmod(descriptor, 0o600)
        return descriptor
    except DocumentConcurrencyError:
        try:
            os.close(descriptor)
        except (OSError, UnboundLocalError):
            pass
        raise
    except OSError:
        raise DocumentConcurrencyError("Document lock file could not be opened") from None


def _write_owner(descriptor: int, profile_name: str, waited_seconds: float) -> None:
    payload = json.dumps(
        {
            "pid": os.getpid(),
            "profile": profile_name,
            "acquired_at": time.time(),
            "waited_seconds": round(waited_seconds, 3),
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    os.lseek(descriptor, 0, os.SEEK_SET)
    os.ftruncate(descriptor, 0)
    os.write(descriptor, payload)
    os.fsync(descriptor)


@contextmanager
def paddle_slot(profile_name: str, *, timeout: float = 180.0) -> Iterator[float]:
    """Acquire the single host-wide heavy PaddleOCR slot.

    ``flock`` ownership is released automatically on process exit, so stale
    metadata cannot hold the queue after a crash. The JSON body is diagnostic
    only; the kernel lock is the source of truth.
    """
    if not isinstance(profile_name, str) or not _PROFILE_RE.fullmatch(profile_name):
        raise DocumentConcurrencyError("Document lock profile failed validation")
    try:
        timeout_value = float(timeout)
    except (TypeError, ValueError):
        raise DocumentConcurrencyError("Document lock timeout is invalid") from None
    if not 0.05 <= timeout_value <= 600.0:
        raise DocumentConcurrencyError("Document lock timeout must be within 0.05-600 seconds")

    descriptor = _open_lock_file(_paddle_lock_path())
    started = time.monotonic()
    acquired = False
    try:
        deadline = started + timeout_value
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DocumentConcurrencyError(
                        "Timed out waiting for the host PaddleOCR slot"
                    ) from None
                time.sleep(min(0.1, remaining))
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise DocumentConcurrencyError(
                        "Document PaddleOCR slot could not be acquired"
                    ) from None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise DocumentConcurrencyError(
                        "Timed out waiting for the host PaddleOCR slot"
                    ) from None
                time.sleep(min(0.1, remaining))
        waited = max(0.0, time.monotonic() - started)
        _write_owner(descriptor, profile_name, waited)
        yield waited
    finally:
        if acquired:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(descriptor)
