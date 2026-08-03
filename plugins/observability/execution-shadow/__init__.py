"""Local, profile-scoped shadow telemetry for execution lifecycle quality.

This plugin is deliberately observational:
- it never returns a control-flow value from a hook;
- it never changes messages, tools, routing, or task state;
- every hook is fail-open;
- raw prompts, assistant text, child goals, tool arguments, and tool results are
  classified in memory and never persisted.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import stat as stat_module
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback uses process lock only
    fcntl = None  # type: ignore[assignment]

import yaml

from hermes_constants import get_hermes_home

from .advisory import evaluate_snapshot


logger = logging.getLogger(__name__)
_SCHEMA_VERSION = "hermes.execution-shadow.v1"
_PHASE_ORDER = {
    "discovery": 0,
    "implementation": 1,
    "verification": 2,
    "closure": 3,
    "finalization": 4,
    "terminal": 5,
}
_STATE_LOCK = threading.RLock()
_STATES: dict[tuple[str, str, str], "TurnState"] = {}
_CONFIG_CACHE: dict[str, tuple[tuple[int, int], "Settings"]] = {}
_EVENT_FILES: dict[str, Path] = {}


@dataclass(frozen=True)
class Settings:
    enabled: bool = False
    mode: str = "shadow"
    scope_freeze_ratio: float = 0.60
    closure_ratio: float = 0.75
    finalization_ratio: float = 0.85
    max_reviewers: int = 1
    max_remediations: int = 1
    max_event_file_bytes: int = 5_000_000
    advisory_enabled: bool = False
    max_advisory_file_bytes: int = 65_536
    advisory_transition_log_enabled: bool = False
    max_advisory_transition_record_bytes: int = 65_536
    max_advisory_transition_total_bytes: int = 8_000_000
    max_advisory_transition_records: int = 1_024


@dataclass
class TurnState:
    home: str
    session_id: str
    task_id: str
    turn_id: str
    platform: str = ""
    phase: str = "discovery"
    phase_history: list[str] = field(default_factory=lambda: ["discovery"])
    budget_used: int = 0
    budget_max: int = 0
    budget_ratio: float = 0.0
    scope_frozen: bool = False
    closure_reserved: bool = False
    finalization_only: bool = False
    api_calls: int = 0
    api_errors: int = 0
    tool_calls_attempted: int = 0
    tool_errors: int = 0
    assistant_tool_calls: int = 0
    helpers_started: int = 0
    helpers_completed: int = 0
    reviewers_started: int = 0
    remediations_started: int = 0
    artifact_created: bool = False
    tests_passed: bool = False
    commit_created: bool = False
    delivery_verified: bool = False
    rollback_ready: bool = False
    final_response_present: bool = False
    terminal_verdict: str | None = None
    durable_execution_proven: bool = False
    completed: bool | None = None
    failed: bool | None = None
    interrupted: bool | None = None
    exit_reason: str = ""
    warnings: set[str] = field(default_factory=set)
    created_at: str = field(default_factory=lambda: _utc_now())
    updated_at: str = field(default_factory=lambda: _utc_now())
    last_advisory_signature: tuple[
        str, tuple[str, ...], tuple[str, ...], str
    ] | None = field(default=None, repr=False)
    last_advisory_primary_action: str | None = field(default=None, repr=False)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_nonnegative_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _load_settings(home: Path) -> Settings:
    """Read the active profile's config with a cheap mtime/size cache."""
    config_path = home / "config.yaml"
    try:
        stat = config_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        return Settings()

    cache_key = str(config_path)
    cached = _CONFIG_CACHE.get(cache_key)
    if cached and cached[0] == signature:
        return cached[1]

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        section = raw.get("execution_shadow") if isinstance(raw, dict) else None
        section = section if isinstance(section, dict) else {}
        settings = Settings(
            enabled=_as_bool(section.get("enabled"), False),
            mode=str(section.get("mode") or "shadow").strip().lower(),
            scope_freeze_ratio=_as_float(section.get("scope_freeze_ratio"), 0.60),
            closure_ratio=_as_float(section.get("closure_ratio"), 0.75),
            finalization_ratio=_as_float(section.get("finalization_ratio"), 0.85),
            max_reviewers=_as_nonnegative_int(section.get("max_reviewers"), 1),
            max_remediations=_as_nonnegative_int(section.get("max_remediations"), 1),
            max_event_file_bytes=max(
                64_000,
                _as_nonnegative_int(section.get("max_event_file_bytes"), 5_000_000),
            ),
            advisory_enabled=_as_bool(section.get("advisory_enabled"), False),
            max_advisory_file_bytes=min(
                1_000_000,
                max(
                    4_096,
                    _as_nonnegative_int(
                        section.get("max_advisory_file_bytes"), 65_536
                    ),
                ),
            ),
            advisory_transition_log_enabled=_as_bool(
                section.get("advisory_transition_log_enabled"), False
            ),
            max_advisory_transition_record_bytes=min(
                1_000_000,
                max(
                    4_096,
                    _as_nonnegative_int(
                        section.get("max_advisory_transition_record_bytes"),
                        65_536,
                    ),
                ),
            ),
            max_advisory_transition_total_bytes=min(
                100_000_000,
                max(
                    64_000,
                    _as_nonnegative_int(
                        section.get("max_advisory_transition_total_bytes"),
                        8_000_000,
                    ),
                ),
            ),
            max_advisory_transition_records=min(
                4_096,
                max(
                    1,
                    _as_nonnegative_int(
                        section.get("max_advisory_transition_records"), 1_024
                    ),
                ),
            ),
        )
    except Exception:
        logger.debug("execution-shadow config read failed", exc_info=True)
        return Settings()

    thresholds_valid = (
        0.0 <= settings.scope_freeze_ratio
        <= settings.closure_ratio
        <= settings.finalization_ratio
        <= 1.0
    )
    if settings.mode != "shadow" or not thresholds_valid:
        logger.warning(
            "execution-shadow disabled: mode must be 'shadow' and budget ratios "
            "must be ordered within [0, 1]"
        )
        settings = Settings()

    _CONFIG_CACHE[cache_key] = (signature, settings)
    return settings


def _bounded_id(value: Any) -> str:
    return str(value or "")[:256]


def _state_key(home: Path, session_id: Any, turn_id: Any, task_id: Any = "") -> tuple[str, str, str]:
    session = _bounded_id(session_id) or "unknown-session"
    turn = _bounded_id(turn_id) or ("task:" + _bounded_id(task_id)) or "unknown-turn"
    return (str(home), session, turn)


def _get_state(
    home: Path,
    *,
    session_id: Any,
    turn_id: Any,
    task_id: Any = "",
    platform: Any = "",
) -> TurnState:
    key = _state_key(home, session_id, turn_id, task_id)
    state = _STATES.get(key)
    if state is None:
        state = TurnState(
            home=str(home),
            session_id=key[1],
            task_id=_bounded_id(task_id),
            turn_id=key[2],
            platform=_bounded_id(platform),
        )
        _STATES[key] = state
    else:
        if task_id and not state.task_id:
            state.task_id = _bounded_id(task_id)
        if platform and not state.platform:
            state.platform = _bounded_id(platform)
    return state


def _advance_phase(state: TurnState, candidate: str) -> None:
    if _PHASE_ORDER.get(candidate, -1) > _PHASE_ORDER.get(state.phase, -1):
        state.phase = candidate
        state.phase_history.append(candidate)


def _apply_budget(state: TurnState, settings: Settings, *, used: Any, maximum: Any) -> None:
    budget_max = _as_nonnegative_int(maximum, state.budget_max)
    budget_used = _as_nonnegative_int(used, state.budget_used)
    state.budget_used = max(state.budget_used, budget_used)
    state.budget_max = max(state.budget_max, budget_max)
    if state.budget_max > 0:
        state.budget_ratio = round(min(1.0, state.budget_used / state.budget_max), 4)
    state.scope_frozen = state.budget_ratio >= settings.scope_freeze_ratio
    state.closure_reserved = state.budget_ratio >= settings.closure_ratio
    state.finalization_only = state.budget_ratio >= settings.finalization_ratio
    if state.finalization_only:
        _advance_phase(state, "finalization")
    elif state.closure_reserved:
        _advance_phase(state, "closure")


def _state_payload(state: TurnState, settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "mode": "shadow",
        "session_id": state.session_id,
        "task_id": state.task_id,
        "turn_id": state.turn_id,
        "platform": state.platform,
        "phase": state.phase,
        "phase_history": list(state.phase_history),
        "policy": {
            "scope_freeze_ratio": settings.scope_freeze_ratio,
            "closure_ratio": settings.closure_ratio,
            "finalization_ratio": settings.finalization_ratio,
            "max_reviewers": settings.max_reviewers,
            "max_remediations": settings.max_remediations,
        },
        "budget": {
            "used": state.budget_used,
            "max": state.budget_max,
            "ratio": state.budget_ratio,
            "scope_frozen": state.scope_frozen,
            "closure_reserved": state.closure_reserved,
            "finalization_only": state.finalization_only,
        },
        "counts": {
            "api_calls": state.api_calls,
            "api_errors": state.api_errors,
            "tool_calls_attempted": state.tool_calls_attempted,
            "tool_errors": state.tool_errors,
            "assistant_tool_calls": state.assistant_tool_calls,
            "helpers_started": state.helpers_started,
            "helpers_completed": state.helpers_completed,
            "reviewers_started": state.reviewers_started,
            "remediations_started": state.remediations_started,
        },
        "evidence": {
            "artifact_created": state.artifact_created,
            "tests_passed": state.tests_passed,
            "commit_created": state.commit_created,
            "delivery_verified": state.delivery_verified,
            "rollback_ready": state.rollback_ready,
            "final_response_present": state.final_response_present,
            "terminal_verdict": state.terminal_verdict,
            "durable_execution_proven": state.durable_execution_proven,
        },
        "terminal": {
            "completed": state.completed,
            "failed": state.failed,
            "interrupted": state.interrupted,
            "exit_reason": state.exit_reason,
        },
        "warnings": sorted(state.warnings),
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }


def _telemetry_root(state: TurnState) -> Path:
    return Path(state.home) / "telemetry" / "execution-shadow"


def _snapshot_path(state: TurnState) -> Path:
    digest = hashlib.sha256(
        f"{state.session_id}\0{state.turn_id}".encode("utf-8", errors="replace")
    ).hexdigest()[:24]
    return _telemetry_root(state) / "turns" / f"{digest}.json"


def _ensure_private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        path.chmod(0o700)


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    max_bytes: int | None = None,
) -> bool:
    _ensure_private_dir(path.parent)
    encoded = (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if max_bytes is not None and len(encoded) > max_bytes:
        return False

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded)
        if os.name != "nt":
            tmp.chmod(0o600)
        os.replace(tmp, path)
        if os.name != "nt":
            path.chmod(0o600)
        return True
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _decision_signature(
    advisory: dict[str, Any],
) -> tuple[str, tuple[str, ...], tuple[str, ...], str]:
    primary_action = advisory.get("primary_action")
    actions = advisory.get("actions")
    reason_codes = advisory.get("reason_codes")
    urgency = advisory.get("urgency")
    if not isinstance(primary_action, str) or not primary_action:
        raise ValueError("invalid transition primary action")
    if not isinstance(actions, list) or not all(
        isinstance(item, str) and 0 < len(item) <= 64 for item in actions
    ):
        raise ValueError("invalid transition actions")
    if not isinstance(reason_codes, list) or not all(
        isinstance(item, str) and 0 < len(item) <= 128 for item in reason_codes
    ):
        raise ValueError("invalid transition reason codes")
    if len(actions) > 16 or len(reason_codes) > 32:
        raise ValueError("transition decision vector exceeds item cap")
    if not isinstance(urgency, str) or urgency not in {"none", "low", "medium", "high"}:
        raise ValueError("invalid transition urgency")
    return primary_action, tuple(actions), tuple(reason_codes), urgency


def _decision_digest(
    signature: tuple[str, tuple[str, ...], tuple[str, ...], str],
) -> str:
    primary_action, actions, reason_codes, urgency = signature
    canonical = json.dumps(
        {
            "actions": list(actions),
            "primary_action": primary_action,
            "reason_codes": list(reason_codes),
            "urgency": urgency,
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()[:24]


def _advisory_transition_record(
    state: TurnState,
    source_payload: dict[str, Any],
    advisory: dict[str, Any],
    signature: tuple[str, tuple[str, ...], tuple[str, ...], str],
) -> dict[str, Any]:
    observed = advisory.get("observed")
    if not isinstance(observed, dict):
        raise ValueError("invalid transition observations")
    primary_action, actions, reason_codes, urgency = signature
    return {
        "schema_version": "hermes.execution-advisory-transition.v1",
        "mode": "advisory-transition",
        "source_schema_version": advisory.get("source_schema_version"),
        "timestamp": source_payload.get("updated_at"),
        "turn_digest": advisory.get("turn_digest"),
        "decision_digest": _decision_digest(signature),
        "previous_primary_action": state.last_advisory_primary_action,
        "primary_action": primary_action,
        "actions": list(actions),
        "reason_codes": list(reason_codes),
        "urgency": urgency,
        "observed": dict(observed),
        "control_effects": dict(advisory["control_effects"]),
    }


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _transition_record_name(record: dict[str, Any]) -> str:
    return (
        f"transition-{_utc_day()}-{time.time_ns()}-"
        f"{record['decision_digest']}-{secrets.token_hex(8)}.json"
    )


def _open_advisory_transition_directory(root: Path) -> int:
    flags = os.O_RDONLY
    for flag_name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
        flags |= getattr(os, flag_name, 0)
    root_fd = os.open(root, flags)
    try:
        try:
            os.mkdir("advisory-transitions", 0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        directory_fd = os.open(
            "advisory-transitions",
            flags,
            dir_fd=root_fd,
        )
    finally:
        os.close(root_fd)
    opened = os.fstat(directory_fd)
    if not stat_module.S_ISDIR(opened.st_mode):
        os.close(directory_fd)
        raise OSError("unsafe advisory transition directory type")
    if os.name != "nt":
        os.fchmod(directory_fd, 0o700)
    return directory_fd


@contextmanager
def _advisory_transition_lock(directory_fd: int):
    flags = os.O_RDWR | os.O_CREAT
    for flag_name in ("O_CLOEXEC", "O_NOFOLLOW"):
        flags |= getattr(os, flag_name, 0)
    lock_fd = os.open(".transition.lock", flags, 0o600, dir_fd=directory_fd)
    try:
        opened = os.fstat(lock_fd)
        path_info = os.stat(
            ".transition.lock",
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if not (
            stat_module.S_ISREG(opened.st_mode)
            and opened.st_nlink == 1
            and (opened.st_dev, opened.st_ino) == (path_info.st_dev, path_info.st_ino)
        ):
            raise OSError("unsafe advisory transition lock file")
        if os.name != "nt":
            os.fchmod(lock_fd, 0o600)
        if fcntl is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
        try:
            os.close(lock_fd)
        except OSError:
            pass


def _write_transition_bytes(fd: int, encoded: bytes) -> None:
    remaining = memoryview(encoded)
    while remaining:
        written = os.write(fd, remaining)
        if written <= 0:
            raise OSError("advisory transition write made no progress")
        remaining = remaining[written:]


def _sync_transition_file(fd: int) -> None:
    os.fsync(fd)


def _sync_transition_directory(fd: int) -> None:
    os.fsync(fd)


def _close_transition_file(fd: int) -> None:
    os.close(fd)


def _open_transition_temp(directory_fd: int) -> tuple[str, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for flag_name in ("O_CLOEXEC", "O_NOFOLLOW", "O_NONBLOCK"):
        flags |= getattr(os, flag_name, 0)
    for _attempt in range(32):
        name = f".transition-tmp-{secrets.token_hex(16)}"
        try:
            fd = os.open(name, flags, 0o600, dir_fd=directory_fd)
        except FileExistsError:
            continue
        opened = os.fstat(fd)
        if not stat_module.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
            os.close(fd)
            try:
                os.unlink(name, dir_fd=directory_fd)
            except OSError:
                pass
            raise OSError("unsafe advisory transition temporary file")
        if os.name != "nt":
            os.fchmod(fd, 0o600)
        return name, fd
    raise OSError("unable to reserve advisory transition temporary file")


def _transition_record_entries(
    directory_fd: int,
) -> list[tuple[int, str, int]]:
    entries: list[tuple[int, str, int]] = []
    pattern = re.compile(r"^transition-\d{8}-\d+-[0-9a-f]{24}-[0-9a-f]+\.json$")
    scanned = 0
    with os.scandir(directory_fd) as iterator:
        for entry in iterator:
            scanned += 1
            if scanned > 16_384:
                raise OSError("advisory transition directory scan cap exceeded")
            if pattern.fullmatch(entry.name) is None:
                continue
            try:
                info = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            if stat_module.S_ISREG(info.st_mode) and info.st_nlink == 1:
                entries.append((info.st_mtime_ns, entry.name, info.st_size))
    return entries


def _prune_advisory_transition_records(
    directory_fd: int,
    settings: Settings,
    *,
    protected_name: str,
) -> None:
    entries = _transition_record_entries(directory_fd)
    entries.sort(key=lambda item: (item[0], item[1]))
    count = len(entries)
    total_bytes = sum(item[2] for item in entries)
    removed = False
    for _mtime, name, size in entries:
        if (
            count <= settings.max_advisory_transition_records
            and total_bytes <= settings.max_advisory_transition_total_bytes
        ):
            break
        if name == protected_name:
            continue
        os.unlink(name, dir_fd=directory_fd)
        count -= 1
        total_bytes -= size
        removed = True
    if (
        count > settings.max_advisory_transition_records
        or total_bytes > settings.max_advisory_transition_total_bytes
    ):
        raise OSError("advisory transition retention bounds cannot be satisfied")
    if removed:
        _sync_transition_directory(directory_fd)


def _publish_transition_record(
    directory_fd: int,
    settings: Settings,
    record: dict[str, Any],
    encoded: bytes,
) -> bool:
    temp_name: str | None = None
    temp_fd: int | None = None
    final_name: str | None = None
    directory_synced = False
    try:
        temp_name, temp_fd = _open_transition_temp(directory_fd)
        _write_transition_bytes(temp_fd, encoded)
        _sync_transition_file(temp_fd)
        try:
            _close_transition_file(temp_fd)
        except Exception:
            try:
                os.close(temp_fd)
            except OSError:
                pass
            temp_fd = None
            raise
        temp_fd = None

        for _attempt in range(32):
            candidate = _transition_record_name(record)
            try:
                os.link(
                    temp_name,
                    candidate,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                continue
            final_name = candidate
            break
        if final_name is None:
            raise OSError("unable to reserve advisory transition record name")

        os.unlink(temp_name, dir_fd=directory_fd)
        temp_name = None
        published = os.stat(
            final_name,
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if not stat_module.S_ISREG(published.st_mode) or published.st_nlink != 1:
            raise OSError("unsafe published advisory transition record")
        try:
            _sync_transition_directory(directory_fd)
        except Exception:
            os.unlink(final_name, dir_fd=directory_fd)
            final_name = None
            try:
                _sync_transition_directory(directory_fd)
            except OSError:
                pass
            raise
        directory_synced = True

        try:
            _prune_advisory_transition_records(
                directory_fd,
                settings,
                protected_name=final_name,
            )
        except OSError:
            logger.debug(
                "execution-shadow transition retention pruning failed open",
                exc_info=True,
            )
        return True
    finally:
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=directory_fd)
            except OSError:
                pass
        if final_name is not None and not directory_synced:
            try:
                os.unlink(final_name, dir_fd=directory_fd)
            except OSError:
                pass


def _store_advisory_transition(
    state: TurnState,
    settings: Settings,
    record: dict[str, Any],
) -> bool:
    encoded = (
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    record_cap = min(
        settings.max_advisory_file_bytes,
        settings.max_advisory_transition_record_bytes,
        settings.max_advisory_transition_total_bytes,
    )
    if len(encoded) > record_cap:
        return False

    directory_fd = _open_advisory_transition_directory(_telemetry_root(state))
    try:
        with _advisory_transition_lock(directory_fd):
            return _publish_transition_record(
                directory_fd,
                settings,
                record,
                encoded,
            )
    finally:
        try:
            os.close(directory_fd)
        except OSError:
            pass


def _persist_advisory(
    state: TurnState,
    settings: Settings,
    source_payload: dict[str, Any],
) -> None:
    if not settings.advisory_enabled:
        return None
    try:
        advisory = evaluate_snapshot(source_payload)
        if advisory.get("schema_version") != "hermes.execution-advisory.v1":
            raise ValueError("unexpected advisory schema")
        if advisory.get("mode") != "advisory":
            raise ValueError("unexpected advisory mode")
        if advisory.get("control_effects") != {
            "block": False,
            "rewrite": False,
            "route": False,
            "schedule": False,
            "spawn": False,
            "stop": False,
        }:
            raise ValueError("advisory control boundary violated")
        digest = advisory.get("turn_digest")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{24}", digest) is None:
            raise ValueError("invalid advisory digest")
        path = _telemetry_root(state) / "advisories" / f"{digest}.json"
        written = _atomic_write_json(
            path,
            advisory,
            max_bytes=settings.max_advisory_file_bytes,
        )
        if not written:
            logger.warning(
                "execution-shadow advisory skipped: serialized result exceeds byte cap"
            )
            return None
        if settings.advisory_transition_log_enabled:
            signature = _decision_signature(advisory)
            if signature != state.last_advisory_signature:
                record = _advisory_transition_record(
                    state,
                    source_payload,
                    advisory,
                    signature,
                )
                if _store_advisory_transition(state, settings, record):
                    state.last_advisory_signature = signature
                    state.last_advisory_primary_action = advisory["primary_action"]
                else:
                    logger.warning(
                        "execution-shadow advisory transition skipped: "
                        "serialized record exceeds byte cap"
                    )
    except Exception:
        logger.debug("execution-shadow advisory failed open", exc_info=True)
    return None


def _event_file(root: Path) -> Path:
    key = str(root)
    path = _EVENT_FILES.get(key)
    if path is None:
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = root / "events" / f"events-{day}-{os.getpid()}.jsonl"
        _EVENT_FILES[key] = path
    return path


def _append_event(state: TurnState, settings: Settings, event: str) -> None:
    root = _telemetry_root(state)
    events_dir = root / "events"
    _ensure_private_dir(events_dir)
    path = _event_file(root)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "event": event,
        "timestamp": state.updated_at,
        "session_id": state.session_id,
        "task_id": state.task_id,
        "turn_id": state.turn_id,
        "phase": state.phase,
        "budget_used": state.budget_used,
        "budget_max": state.budget_max,
        "warning_count": len(state.warnings),
    }
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    encoded = line.encode("utf-8")

    try:
        if path.exists() and path.stat().st_size + len(encoded) > settings.max_event_file_bytes:
            rotated = path.with_name(f"{path.stem}-{int(datetime.now(timezone.utc).timestamp())}{path.suffix}")
            os.replace(path, rotated)
            if os.name != "nt":
                rotated.chmod(0o600)
    except OSError:
        pass

    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, encoded)
    finally:
        os.close(fd)
    if os.name != "nt":
        path.chmod(0o600)


def _persist_state(state: TurnState, settings: Settings, event: str) -> None:
    # ``mkdir(parents=True)`` applies the explicit mode only to the leaf; make
    # the telemetry root private as well so filenames and turn identifiers
    # are not directory-listable by other local users.
    _ensure_private_dir(_telemetry_root(state))
    payload = _state_payload(state, settings)
    _atomic_write_json(_snapshot_path(state), payload)
    _append_event(state, settings, event)
    _persist_advisory(state, settings, payload)


def _safe_update(
    event: str,
    *,
    session_id: Any,
    turn_id: Any,
    task_id: Any = "",
    platform: Any = "",
    mutate: Callable[[TurnState, Settings], None] | None = None,
) -> None:
    """Run one observational update. Exceptions never escape the hook."""
    try:
        home = get_hermes_home().expanduser().resolve()
        settings = _load_settings(home)
        if not settings.enabled:
            return None
        with _STATE_LOCK:
            state = _get_state(
                home,
                session_id=session_id,
                turn_id=turn_id,
                task_id=task_id,
                platform=platform,
            )
            if mutate is not None:
                mutate(state, settings)
            state.updated_at = _utc_now()
            _persist_state(state, settings, event)
    except Exception:
        logger.debug("execution-shadow hook failed open: %s", event, exc_info=True)
    return None


def _command(args: Any) -> str:
    if not isinstance(args, dict):
        return ""
    return str(args.get("command") or "").strip().lower()


def _tool_phase(tool_name: str, args: Any) -> str | None:
    name = str(tool_name or "").strip().lower()
    if name in {"patch", "write_file", "skill_manage", "memory"}:
        return "implementation"
    if name in {
        "read_file", "search_files", "session_search", "browser_navigate",
        "browser_snapshot", "browser_console", "browser_get_images",
        "vision_analyze", "web_search", "web_extract", "skill_view",
    }:
        return "discovery"
    if name in {"send_message", "feishu_drive_reply_comment", "feishu_drive_add_comment"}:
        return "closure"
    if name == "cronjob":
        action = str(args.get("action") or "") if isinstance(args, dict) else ""
        return "discovery" if action in {"list"} else "implementation"
    if name.startswith("kanban_"):
        return "discovery" if name in {"kanban_list", "kanban_show"} else "implementation"
    if name != "terminal":
        return None

    cmd = _command(args)
    if re.search(r"\bgit\s+commit\b|\bhermes\s+send\b|\bdeploy\b|\bpublish\b", cmd):
        return "closure"
    if re.search(
        r"\b(pytest|unittest|tox|nox|ruff|mypy|eslint|tsc|cargo\s+test|go\s+test)\b"
        r"|\bhermes\s+(config\s+check|gateway\s+status|cron\s+status)\b"
        r"|\bgit\s+(diff|status)\b|\bsha256sum\b|\bcheck\b|\bverify\b",
        cmd,
    ):
        return "verification"
    if re.search(
        r"\b(pip|npm|pnpm|yarn|uv)\s+install\b|\bgit\s+(checkout|switch|merge|cherry-pick|apply)\b"
        r"|\b(cp|mv|install|systemctl)\b|\bhermes\s+config\s+set\b",
        cmd,
    ):
        return "implementation"
    return "discovery"


def _result_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _result_succeeded(result: Any, status: Any, error_type: Any) -> bool:
    if str(status or "").lower() in {"error", "blocked", "cancelled", "failed"}:
        return False
    if error_type:
        return False
    parsed = _result_dict(result)
    if "exit_code" in parsed and parsed.get("exit_code") != 0:
        return False
    if parsed.get("success") is False:
        return False
    if parsed.get("error") and parsed.get("success") is not True:
        return False
    return True


def _is_test_command(command: str) -> bool:
    return bool(re.search(r"\b(pytest|unittest|tox|nox|cargo\s+test|go\s+test|npm\s+test)\b", command))


def _classify_helper(goal: Any) -> str:
    text = str(goal or "").lower()
    if re.search(r"\bremediat|\bfix only\b|blocking findings|reported issues|fix agent", text):
        return "remediation"
    if re.search(r"\breviewer\b|\bcode review\b|\bread-only review\b|\bindependent review\b|\bsecurity review\b|\bquality review\b|\bspec review\b", text):
        return "review"
    if re.search(r"\breview\b", text) and re.search(
        r"\bstaged\b|\bpre-commit\b|\bread-only\b|\bfindings\b|"
        r"\bverdict\b|go\s*(?:/|or)\s*no-go",
        text,
    ):
        return "review"
    return "helper"


def _verdict(text: Any) -> str | None:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines:
        return None
    candidate = lines[-1].strip("`*_#> ")
    match = re.fullmatch(
        r"(?:(?:final\s+)?verdict\s*:\s*)?"
        r"(NO[- ]GO|GO|DONE_WITH_BLOCKER|DONE)[.!]?",
        candidate,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group(1).upper().replace(" ", "-")
    return value


_EXIT_REASON_CODES = {
    "all_retries_exhausted_no_response",
    "budget_exhausted",
    "completed",
    "content_policy_blocked",
    "empty_response_exhausted",
    "error_near_max_iterations",
    "failed",
    "fallback_prior_turn_content",
    "guardrail_halt",
    "incomplete",
    "interrupted",
    "interrupted_by_user",
    "interrupted_during_api_call",
    "max_iterations_reached",
    "ollama_runtime_context_too_small",
    "partial_stream_recovery",
    "text_response",
    "unhandled_exception",
    "unknown",
}


def _canonical_exit_reason(value: Any) -> str:
    """Return a finite reason code and discard all dynamic exception text."""
    raw = str(value or "").strip().lower()
    base = raw.split("(", 1)[0].split(":", 1)[0].strip()
    base = re.sub(r"[^a-z0-9_]+", "_", base).strip("_")
    return base if base in _EXIT_REASON_CODES else "unknown"


def _claims_durable_running(text: Any) -> bool:
    value = str(text or "").lower()
    durable = bool(re.search(r"durable|background|worker|kanban|task|handoff|批次|背景|工作者", value))
    active = bool(re.search(r"\bqueued\b|\brunning\b|handed off|排隊|運行中|執行中|已交接", value))
    return durable and active


def on_pre_llm_call(**kwargs: Any) -> None:
    def mutate(state: TurnState, _settings: Settings) -> None:
        state.platform = _bounded_id(kwargs.get("platform")) or state.platform

    return _safe_update(
        "turn_started",
        session_id=kwargs.get("session_id"),
        task_id=kwargs.get("task_id"),
        turn_id=kwargs.get("turn_id"),
        platform=kwargs.get("platform"),
        mutate=mutate,
    )


def on_pre_api_request(**kwargs: Any) -> None:
    def mutate(state: TurnState, settings: Settings) -> None:
        state.api_calls = max(state.api_calls, _as_nonnegative_int(kwargs.get("api_call_count"), 0))
        _apply_budget(
            state,
            settings,
            used=kwargs.get("budget_used", kwargs.get("api_call_count")),
            maximum=kwargs.get("budget_max", 0),
        )

    return _safe_update(
        "api_started",
        session_id=kwargs.get("session_id"),
        task_id=kwargs.get("task_id"),
        turn_id=kwargs.get("turn_id"),
        platform=kwargs.get("platform"),
        mutate=mutate,
    )


def on_post_api_request(**kwargs: Any) -> None:
    def mutate(state: TurnState, settings: Settings) -> None:
        state.api_calls = max(state.api_calls, _as_nonnegative_int(kwargs.get("api_call_count"), 0))
        state.assistant_tool_calls += _as_nonnegative_int(
            kwargs.get("assistant_tool_call_count"), 0
        )
        _apply_budget(
            state,
            settings,
            used=kwargs.get("budget_used", kwargs.get("api_call_count")),
            maximum=kwargs.get("budget_max", 0),
        )

    return _safe_update(
        "api_finished",
        session_id=kwargs.get("session_id"),
        task_id=kwargs.get("task_id"),
        turn_id=kwargs.get("turn_id"),
        platform=kwargs.get("platform"),
        mutate=mutate,
    )


def on_api_request_error(**kwargs: Any) -> None:
    def mutate(state: TurnState, settings: Settings) -> None:
        state.api_calls = max(
            state.api_calls,
            _as_nonnegative_int(kwargs.get("api_call_count"), 0),
        )
        state.api_errors += 1
        _apply_budget(
            state,
            settings,
            used=kwargs.get("budget_used", kwargs.get("api_call_count")),
            maximum=kwargs.get("budget_max", 0),
        )

    return _safe_update(
        "api_failed",
        session_id=kwargs.get("session_id"),
        task_id=kwargs.get("task_id"),
        turn_id=kwargs.get("turn_id"),
        platform=kwargs.get("platform"),
        mutate=mutate,
    )


def on_pre_tool_call(**kwargs: Any) -> None:
    def mutate(state: TurnState, _settings: Settings) -> None:
        state.tool_calls_attempted += 1
        candidate = _tool_phase(str(kwargs.get("tool_name") or ""), kwargs.get("args"))
        if state.scope_frozen and candidate == "discovery":
            state.warnings.add("discovery_started_after_scope_freeze")
        if state.finalization_only and candidate in {"discovery", "implementation"}:
            state.warnings.add("new_work_started_during_finalization")
        if candidate == "implementation" and _PHASE_ORDER[state.phase] >= _PHASE_ORDER["verification"]:
            state.warnings.add("implementation_after_verification")
        if candidate:
            _advance_phase(state, candidate)

    return _safe_update(
        "tool_started",
        session_id=kwargs.get("session_id"),
        task_id=kwargs.get("task_id"),
        turn_id=kwargs.get("turn_id"),
        mutate=mutate,
    )


def on_post_tool_call(**kwargs: Any) -> None:
    def mutate(state: TurnState, _settings: Settings) -> None:
        name = str(kwargs.get("tool_name") or "").strip().lower()
        args = kwargs.get("args") if isinstance(kwargs.get("args"), dict) else {}
        result = kwargs.get("result")
        success = _result_succeeded(result, kwargs.get("status"), kwargs.get("error_type"))
        if not success:
            state.tool_errors += 1
            return

        command = _command(args)
        if name in {"write_file", "patch", "skill_manage"}:
            state.artifact_created = True
        if name == "terminal":
            if _is_test_command(command):
                state.tests_passed = True
            if re.search(r"\bgit\s+commit\b", command):
                state.commit_created = True
            if re.search(r"\bhermes\s+send\b", command):
                state.delivery_verified = True
            if re.search(r"\b(snapshot|backup)\b", command) or (
                re.search(r"\b(cp|install)\b", command) and "backup" in command
            ):
                state.rollback_ready = True
            parsed = _result_dict(result)
            if args.get("background") is True and parsed.get("session_id"):
                state.durable_execution_proven = True
        elif name == "cronjob":
            parsed = _result_dict(result)
            if str(args.get("action") or "") == "create" and (
                parsed.get("job_id") or parsed.get("id") or parsed.get("success") is True
            ):
                state.durable_execution_proven = True
        elif name == "kanban_create":
            parsed = _result_dict(result)
            if parsed.get("task_id") or parsed.get("id") or parsed.get("success") is True:
                state.durable_execution_proven = True
        elif name in {"send_message", "feishu_drive_reply_comment", "feishu_drive_add_comment"}:
            state.delivery_verified = True

    return _safe_update(
        "tool_finished",
        session_id=kwargs.get("session_id"),
        task_id=kwargs.get("task_id"),
        turn_id=kwargs.get("turn_id"),
        mutate=mutate,
    )


def on_subagent_start(**kwargs: Any) -> None:
    kind = _classify_helper(kwargs.get("child_goal"))

    def mutate(state: TurnState, settings: Settings) -> None:
        state.helpers_started += 1
        if state.scope_frozen:
            state.warnings.add("helper_started_after_scope_freeze")
        if state.finalization_only:
            state.warnings.add("helper_started_during_finalization")
        if kind == "review":
            state.reviewers_started += 1
            if state.reviewers_started > settings.max_reviewers:
                state.warnings.add("reviewer_limit_exceeded")
        elif kind == "remediation":
            state.remediations_started += 1
            if state.remediations_started > settings.max_remediations:
                state.warnings.add("remediation_limit_exceeded")

    return _safe_update(
        "helper_started",
        session_id=kwargs.get("parent_session_id"),
        task_id="",
        turn_id=kwargs.get("parent_turn_id"),
        mutate=mutate,
    )


def on_subagent_stop(**kwargs: Any) -> None:
    def mutate(state: TurnState, _settings: Settings) -> None:
        state.helpers_completed += 1

    return _safe_update(
        "helper_finished",
        session_id=kwargs.get("parent_session_id"),
        task_id="",
        turn_id=kwargs.get("parent_turn_id"),
        mutate=mutate,
    )


def on_post_llm_call(**kwargs: Any) -> None:
    response = kwargs.get("assistant_response")

    def mutate(state: TurnState, settings: Settings) -> None:
        state.api_calls = max(state.api_calls, _as_nonnegative_int(kwargs.get("api_call_count"), 0))
        _apply_budget(
            state,
            settings,
            used=kwargs.get("budget_used", kwargs.get("api_call_count")),
            maximum=kwargs.get("budget_max", 0),
        )
        state.final_response_present = bool(str(response or "").strip())
        state.terminal_verdict = _verdict(response)
        if _claims_durable_running(response) and not state.durable_execution_proven:
            state.warnings.add("passive_durable_claim_without_execution_proof")
        state.completed = kwargs.get("completed") if isinstance(kwargs.get("completed"), bool) else None
        state.failed = kwargs.get("failed") if isinstance(kwargs.get("failed"), bool) else None
        state.interrupted = kwargs.get("interrupted") if isinstance(kwargs.get("interrupted"), bool) else None
        state.exit_reason = _canonical_exit_reason(kwargs.get("turn_exit_reason"))
        _advance_phase(state, "terminal")

    return _safe_update(
        "turn_finished",
        session_id=kwargs.get("session_id"),
        task_id=kwargs.get("task_id"),
        turn_id=kwargs.get("turn_id"),
        platform=kwargs.get("platform"),
        mutate=mutate,
    )


def on_turn_end(**kwargs: Any) -> None:
    def mutate(state: TurnState, settings: Settings) -> None:
        state.api_calls = max(
            state.api_calls,
            _as_nonnegative_int(kwargs.get("api_call_count"), 0),
        )
        _apply_budget(
            state,
            settings,
            used=kwargs.get("budget_used", kwargs.get("api_call_count")),
            maximum=kwargs.get("budget_max", 0),
        )
        state.final_response_present = (
            state.final_response_present
            or kwargs.get("final_response_present") is True
        )
        state.completed = (
            kwargs.get("completed")
            if isinstance(kwargs.get("completed"), bool)
            else state.completed
        )
        state.failed = (
            kwargs.get("failed")
            if isinstance(kwargs.get("failed"), bool)
            else state.failed
        )
        state.interrupted = (
            kwargs.get("interrupted")
            if isinstance(kwargs.get("interrupted"), bool)
            else state.interrupted
        )
        state.exit_reason = _canonical_exit_reason(kwargs.get("turn_exit_reason"))
        _advance_phase(state, "terminal")

    return _safe_update(
        "turn_ended",
        session_id=kwargs.get("session_id"),
        task_id=kwargs.get("task_id"),
        turn_id=kwargs.get("turn_id"),
        platform=kwargs.get("platform"),
        mutate=mutate,
    )


def _reset_for_tests() -> None:
    with _STATE_LOCK:
        _STATES.clear()
        _CONFIG_CACHE.clear()
        _EVENT_FILES.clear()


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("pre_api_request", on_pre_api_request)
    ctx.register_hook("post_api_request", on_post_api_request)
    ctx.register_hook("api_request_error", on_api_request_error)
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("on_turn_end", on_turn_end)
    ctx.register_hook("subagent_start", on_subagent_start)
    ctx.register_hook("subagent_stop", on_subagent_stop)
