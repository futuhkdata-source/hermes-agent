#!/usr/bin/env python3
"""Deterministic, aggregate-only replay for P1-B execution advice."""
from __future__ import annotations

import argparse
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from advisory import AdvisoryInputError, summarize_snapshots


_MAX_SNAPSHOT_BYTES = 1_000_000


@dataclass
class LoadStats:
    seen: int = 0
    invalid: int = 0
    input_errors: int = 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay execution-shadow snapshots into aggregate advisory counts."
    )
    parser.add_argument(
        "--telemetry-root",
        required=True,
        type=Path,
        help="Path to the execution-shadow telemetry root containing turns/.",
    )
    parser.add_argument(
        "--as-of",
        required=True,
        help="Explicit timezone-aware ISO-8601 replay clock.",
    )
    parser.add_argument(
        "--stale-after-seconds",
        type=int,
        default=900,
        help="Age after which a non-terminal snapshot is advisory-stale.",
    )
    return parser


def _read_regular_json(path: str | Path, *, dir_fd: int | None = None) -> dict[str, Any]:
    flags = os.O_RDONLY
    for name in ("O_BINARY", "O_CLOEXEC", "O_NONBLOCK", "O_NOFOLLOW"):
        flags |= getattr(os, name, 0)
    open_kwargs = {"dir_fd": dir_fd} if dir_fd is not None else {}
    fd = os.open(path, flags, **open_kwargs)
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > _MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot is not a bounded regular file")

        path_stat = os.stat(path, dir_fd=dir_fd, follow_symlinks=False)
        if (
            not stat.S_ISREG(path_stat.st_mode)
            or (opened.st_dev, opened.st_ino) != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise ValueError("snapshot changed during open")

        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_SNAPSHOT_BYTES:
            chunk = os.read(fd, min(65_536, _MAX_SNAPSHOT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > _MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot exceeds the bounded read limit")
        parsed = json.loads(b"".join(chunks).decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("snapshot must be an object")
        return parsed
    finally:
        os.close(fd)


def _iter_snapshots(turns_dir: Path, stats: LoadStats) -> Iterator[dict[str, Any]]:
    """Yield bounded regular JSON rows without following directory/file links."""
    use_dir_fd = os.name != "nt" and hasattr(os, "O_DIRECTORY")
    directory_fd: int | None = None
    try:
        if use_dir_fd:
            directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_flags |= getattr(os, "O_CLOEXEC", 0)
            directory_flags |= getattr(os, "O_NOFOLLOW", 0)
            directory_fd = os.open(turns_dir, directory_flags)
            entries = os.scandir(directory_fd)
        else:
            entries = os.scandir(turns_dir)

        with entries:
            for entry in entries:
                if not entry.name.endswith(".json"):
                    continue
                stats.seen += 1
                try:
                    candidate: str | Path
                    candidate = entry.name if directory_fd is not None else Path(entry.path)
                    yield _read_regular_json(candidate, dir_fd=directory_fd)
                except Exception:
                    # Per-record failures are counted only. Paths, row data, and
                    # dynamic exception strings never reach stdout/stderr.
                    stats.invalid += 1
    except Exception:
        stats.input_errors += 1
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _is_private_directory_candidate(path: Path) -> bool:
    try:
        return stat.S_ISDIR(path.lstat().st_mode) and not path.is_symlink()
    except OSError:
        return False


def _empty_summary() -> dict[str, Any]:
    return summarize_snapshots([], as_of=None, stale_after_seconds=900)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    turns_dir = args.telemetry_root / "turns"
    input_ready = _is_private_directory_candidate(turns_dir)
    stats = LoadStats()
    options_valid = True

    try:
        summary = summarize_snapshots(
            _iter_snapshots(turns_dir, stats) if input_ready else (),
            as_of=args.as_of,
            stale_after_seconds=args.stale_after_seconds,
        )
    except AdvisoryInputError:
        options_valid = False
        summary = _empty_summary()

    summary["snapshots_seen"] = stats.seen
    summary["snapshots_invalid"] += stats.invalid
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    failed = (
        not input_ready
        or not options_valid
        or stats.input_errors > 0
        or stats.seen == 0
        or summary["snapshots_invalid"] > 0
    )
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
