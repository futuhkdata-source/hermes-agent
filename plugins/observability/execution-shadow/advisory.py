"""Offline-only P1-B advisory evaluation for execution-shadow snapshots.

This module is intentionally not imported by the live execution-shadow plugin.
It accepts metadata-only P1-A snapshots and returns deterministic advice.  It
cannot block, stop, rewrite, route, schedule, or spawn anything.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


_SOURCE_SCHEMA = "hermes.execution-shadow.v1"
_ADVISORY_SCHEMA = "hermes.execution-advisory.v1"
_REPLAY_SCHEMA = "hermes.execution-advisory-replay.v1"
_PHASE_ORDER = {
    "discovery": 0,
    "implementation": 1,
    "verification": 2,
    "closure": 3,
    "finalization": 4,
    "terminal": 5,
}
_PHASES = set(_PHASE_ORDER)
_SOURCE_KEYS = {
    "schema_version",
    "mode",
    "session_id",
    "task_id",
    "turn_id",
    "platform",
    "phase",
    "phase_history",
    "policy",
    "budget",
    "counts",
    "evidence",
    "terminal",
    "warnings",
    "created_at",
    "updated_at",
}
_POLICY_KEYS = {
    "scope_freeze_ratio",
    "closure_ratio",
    "finalization_ratio",
    "max_reviewers",
    "max_remediations",
}
_BUDGET_KEYS = {
    "used",
    "max",
    "ratio",
    "scope_frozen",
    "closure_reserved",
    "finalization_only",
}
_COUNT_KEYS = {
    "api_calls",
    "api_errors",
    "tool_calls_attempted",
    "tool_errors",
    "assistant_tool_calls",
    "helpers_started",
    "helpers_completed",
    "reviewers_started",
    "remediations_started",
}
_EVIDENCE_KEYS = {
    "artifact_created",
    "tests_passed",
    "commit_created",
    "delivery_verified",
    "rollback_ready",
    "final_response_present",
    "terminal_verdict",
    "durable_execution_proven",
}
_TERMINAL_KEYS = {"completed", "failed", "interrupted", "exit_reason"}
_EVIDENCE_BOOLEAN_KEYS = _EVIDENCE_KEYS - {"terminal_verdict"}
_TERMINAL_VERDICTS = {None, "GO", "NO-GO", "DONE", "DONE_WITH_BLOCKER"}
_WARNING_CODES = {
    "discovery_started_after_scope_freeze",
    "helper_started_after_scope_freeze",
    "helper_started_during_finalization",
    "implementation_after_verification",
    "new_work_started_during_finalization",
    "passive_durable_claim_without_execution_proof",
    "remediation_limit_exceeded",
    "reviewer_limit_exceeded",
}
_ACTION_PRECEDENCE = {
    "REVIEW_TERMINAL_OUTCOME": 700,
    "REVIEW_STALE_TURN": 600,
    "VERIFY_EXECUTION_PROOF": 500,
    "FINALIZE_ONLY": 400,
    "MOVE_TO_CLOSURE": 300,
    "FREEZE_SCOPE": 200,
    "NO_ACTION": 0,
}
_ACTION_URGENCY = {
    "REVIEW_TERMINAL_OUTCOME": "high",
    "REVIEW_STALE_TURN": "high",
    "VERIFY_EXECUTION_PROOF": "high",
    "FINALIZE_ONLY": "high",
    "MOVE_TO_CLOSURE": "medium",
    "FREEZE_SCOPE": "low",
    "NO_ACTION": "none",
}
_TERMINAL_SUCCESS_REASONS = {"completed", "text_response"}
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
_FINALIZATION_WARNINGS = {
    "helper_started_during_finalization",
    "new_work_started_during_finalization",
    "remediation_limit_exceeded",
    "reviewer_limit_exceeded",
}
_FREEZE_WARNINGS = {
    "discovery_started_after_scope_freeze",
    "helper_started_after_scope_freeze",
    "implementation_after_verification",
}
_CONTROL_EFFECTS = {
    "block": False,
    "rewrite": False,
    "route": False,
    "schedule": False,
    "spawn": False,
    "stop": False,
}


class AdvisoryInputError(ValueError):
    """The source snapshot is not a valid P1-A metadata contract."""


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise AdvisoryInputError(f"{field} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if any(not isinstance(key, str) for key in value) or set(value) != expected:
        raise AdvisoryInputError(f"{field} does not match the source schema")


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AdvisoryInputError(f"{field} must be a non-negative integer")
    return value


def _finite_ratio(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdvisoryInputError(f"{field} must be numeric")
    try:
        ratio = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise AdvisoryInputError(f"{field} must be finite") from exc
    if not 0.0 <= ratio <= 1.0:
        raise AdvisoryInputError(f"{field} must be within [0, 1]")
    return ratio


def _aware_datetime(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise AdvisoryInputError(f"{field} is not ISO-8601") from exc
    else:
        raise AdvisoryInputError(f"{field} must be a datetime or ISO-8601 string")
    if parsed.tzinfo is None:
        raise AdvisoryInputError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _validated(snapshot: Any) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[str, Any],
    list[str],
    str,
    float,
]:
    source = _mapping(snapshot, "snapshot")
    _exact_keys(source, _SOURCE_KEYS, "snapshot")
    if source.get("schema_version") != _SOURCE_SCHEMA:
        raise AdvisoryInputError("unsupported source schema")
    if source.get("mode") != "shadow":
        raise AdvisoryInputError("source mode must be shadow")

    for field in ("session_id", "task_id", "turn_id", "platform"):
        value = source.get(field)
        if not isinstance(value, str) or len(value) > 256 or "\n" in value or "\r" in value:
            raise AdvisoryInputError(f"{field} is not a bounded identifier")
    if not source["session_id"] or not source["turn_id"]:
        raise AdvisoryInputError("session_id and turn_id are required")

    phase = source.get("phase")
    if not isinstance(phase, str) or phase not in _PHASES:
        raise AdvisoryInputError("unknown execution phase")
    history = source.get("phase_history")
    if (
        not isinstance(history, list)
        or not history
        or any(not isinstance(item, str) or item not in _PHASES for item in history)
        or history[0] != "discovery"
        or history[-1] != phase
        or any(
            _PHASE_ORDER[current] >= _PHASE_ORDER[nxt]
            for current, nxt in zip(history, history[1:])
        )
    ):
        raise AdvisoryInputError("phase_history is not monotonic")

    policy = _mapping(source.get("policy"), "policy")
    budget = _mapping(source.get("budget"), "budget")
    counts = _mapping(source.get("counts"), "counts")
    evidence = _mapping(source.get("evidence"), "evidence")
    terminal = _mapping(source.get("terminal"), "terminal")
    _exact_keys(policy, _POLICY_KEYS, "policy")
    _exact_keys(budget, _BUDGET_KEYS, "budget")
    _exact_keys(counts, _COUNT_KEYS, "counts")
    _exact_keys(evidence, _EVIDENCE_KEYS, "evidence")
    _exact_keys(terminal, _TERMINAL_KEYS, "terminal")

    ratio = _finite_ratio(budget.get("ratio"), "budget.ratio")
    freeze = _finite_ratio(policy.get("scope_freeze_ratio"), "policy.scope_freeze_ratio")
    closure = _finite_ratio(policy.get("closure_ratio"), "policy.closure_ratio")
    finalization = _finite_ratio(
        policy.get("finalization_ratio"), "policy.finalization_ratio"
    )
    if not freeze <= closure <= finalization:
        raise AdvisoryInputError("budget policy thresholds must be ordered")
    for field in ("max_reviewers", "max_remediations"):
        _nonnegative_int(policy.get(field), f"policy.{field}")

    used = _nonnegative_int(budget.get("used"), "budget.used")
    maximum = _nonnegative_int(budget.get("max"), "budget.max")
    expected_ratio = round(min(1.0, used / maximum), 4) if maximum else 0.0
    if not math.isclose(ratio, expected_ratio, abs_tol=1e-9):
        raise AdvisoryInputError("budget ratio does not match used/max")
    expected_flags = {
        "scope_frozen": ratio >= freeze,
        "closure_reserved": ratio >= closure,
        "finalization_only": ratio >= finalization,
    }
    for field, expected in expected_flags.items():
        if budget.get(field) is not expected:
            raise AdvisoryInputError(f"budget.{field} is inconsistent")

    for field in _COUNT_KEYS:
        _nonnegative_int(counts.get(field), f"counts.{field}")
    if counts["helpers_completed"] > counts["helpers_started"]:
        raise AdvisoryInputError("completed helpers exceed started helpers")
    if counts["reviewers_started"] > counts["helpers_started"]:
        raise AdvisoryInputError("reviewers exceed started helpers")
    if counts["remediations_started"] > counts["helpers_started"]:
        raise AdvisoryInputError("remediations exceed started helpers")

    for field in _EVIDENCE_BOOLEAN_KEYS:
        if not isinstance(evidence.get(field), bool):
            raise AdvisoryInputError(f"evidence.{field} must be boolean")
    if evidence.get("terminal_verdict") not in _TERMINAL_VERDICTS:
        raise AdvisoryInputError("evidence.terminal_verdict is not canonical")

    exit_reason = terminal.get("exit_reason")
    if not isinstance(exit_reason, str) or (
        exit_reason and exit_reason not in _EXIT_REASON_CODES
    ):
        raise AdvisoryInputError("terminal.exit_reason is not canonical")
    terminal_flags = [terminal.get(field) for field in ("completed", "failed", "interrupted")]
    if phase == "terminal":
        if any(not isinstance(value, bool) for value in terminal_flags):
            raise AdvisoryInputError("terminal flags must be boolean at terminal phase")
        if sum(value is True for value in terminal_flags) > 1:
            raise AdvisoryInputError("terminal flags are contradictory")
        if terminal.get("completed") is True and exit_reason not in _TERMINAL_SUCCESS_REASONS:
            raise AdvisoryInputError("completed terminal exit reason is inconsistent")
    elif any(value is not None for value in terminal_flags) or exit_reason:
        raise AdvisoryInputError("non-terminal snapshot contains terminal state")

    raw_warnings = source.get("warnings")
    if (
        not isinstance(raw_warnings, list)
        or any(not isinstance(item, str) or item not in _WARNING_CODES for item in raw_warnings)
        or len(raw_warnings) != len(set(raw_warnings))
    ):
        raise AdvisoryInputError("warnings are not canonical reason codes")

    created_at = _aware_datetime(source.get("created_at"), "created_at")
    updated_at = _aware_datetime(source.get("updated_at"), "updated_at")
    if created_at > updated_at:
        raise AdvisoryInputError("updated_at precedes created_at")

    return source, policy, budget, counts, terminal, list(raw_warnings), phase, ratio


def _turn_digest(snapshot: Mapping[str, Any]) -> str:
    bounded = [str(snapshot.get(field) or "")[:256] for field in ("session_id", "task_id", "turn_id")]
    return hashlib.sha256("\0".join(bounded).encode("utf-8", errors="replace")).hexdigest()[:24]


def _terminal_reasons(terminal: Mapping[str, Any]) -> list[str]:
    failed = terminal.get("failed") is True
    interrupted = terminal.get("interrupted") is True
    completed = terminal.get("completed") is True
    exit_reason = str(terminal.get("exit_reason") or "unknown")

    if failed:
        return ["terminal_failed"]
    if interrupted:
        return ["terminal_interrupted"]
    if completed and exit_reason in _TERMINAL_SUCCESS_REASONS:
        return []
    if exit_reason not in {"", "unknown"}:
        return [f"terminal_{exit_reason}"]
    return ["terminal_incomplete"]


def _validate_replay_options(
    as_of: datetime | str | None,
    stale_after_seconds: int,
) -> datetime | None:
    if isinstance(stale_after_seconds, bool) or not isinstance(stale_after_seconds, int):
        raise AdvisoryInputError("stale_after_seconds must be an integer")
    if stale_after_seconds <= 0:
        raise AdvisoryInputError("stale_after_seconds must be positive")
    return _aware_datetime(as_of, "as_of") if as_of is not None else None


def _evaluate_snapshot(
    snapshot: Mapping[str, Any],
    *,
    as_of: datetime | str | None = None,
    stale_after_seconds: int = 900,
) -> dict[str, Any]:
    """Return deterministic, non-enforcing advice for one P1-A snapshot."""
    observed_at = _validate_replay_options(as_of, stale_after_seconds)
    (
        source,
        policy,
        _budget,
        counts,
        terminal,
        warnings,
        phase,
        ratio,
    ) = _validated(snapshot)

    actions: set[str] = set()
    reasons: list[str] = []

    if phase == "terminal":
        terminal_reasons = _terminal_reasons(terminal)
        if terminal_reasons:
            actions.add("REVIEW_TERMINAL_OUTCOME")
            reasons.extend(terminal_reasons)
    elif observed_at is not None:
        updated_at = _aware_datetime(source.get("updated_at"), "updated_at")
        if (observed_at - updated_at).total_seconds() > stale_after_seconds:
            actions.add("REVIEW_STALE_TURN")
            reasons.append("nonterminal_stale")

    warning_set = set(warnings)
    if "passive_durable_claim_without_execution_proof" in warning_set:
        actions.add("VERIFY_EXECUTION_PROOF")
        reasons.append("passive_durable_claim_without_execution_proof")

    if phase != "terminal":
        for warning in sorted(warning_set & _FINALIZATION_WARNINGS):
            actions.add("FINALIZE_ONLY")
            reasons.append(warning)
        for warning in sorted(warning_set & _FREEZE_WARNINGS):
            actions.add("FREEZE_SCOPE")
            reasons.append(warning)

        freeze = float(policy["scope_freeze_ratio"])
        closure = float(policy["closure_ratio"])
        finalization = float(policy["finalization_ratio"])
        if ratio >= finalization:
            actions.add("FINALIZE_ONLY")
            reasons.append("budget_finalization_only")
        elif ratio >= closure:
            actions.add("MOVE_TO_CLOSURE")
            reasons.append("budget_closure_reserved")
        elif ratio >= freeze:
            actions.add("FREEZE_SCOPE")
            reasons.append("budget_scope_frozen")

    if not actions:
        actions.add("NO_ACTION")

    ordered_actions = sorted(
        actions,
        key=lambda action: (-_ACTION_PRECEDENCE[action], action),
    )
    deduped_reasons = list(dict.fromkeys(reasons))
    primary = ordered_actions[0]
    open_helpers = max(0, int(counts["helpers_started"]) - int(counts["helpers_completed"]))

    return {
        "schema_version": _ADVISORY_SCHEMA,
        "mode": "advisory",
        "source_schema_version": _SOURCE_SCHEMA,
        "turn_digest": _turn_digest(source),
        "primary_action": primary,
        "actions": ordered_actions,
        "reason_codes": deduped_reasons,
        "urgency": _ACTION_URGENCY[primary],
        "observed": {
            "phase": phase,
            "budget_ratio": ratio,
            "warning_count": len(warnings),
            "open_helpers": open_helpers,
            "terminal": phase == "terminal",
            "completed": terminal.get("completed")
            if isinstance(terminal.get("completed"), bool)
            else None,
            "failed": terminal.get("failed")
            if isinstance(terminal.get("failed"), bool)
            else None,
            "interrupted": terminal.get("interrupted")
            if isinstance(terminal.get("interrupted"), bool)
            else None,
            "exit_reason": str(terminal.get("exit_reason") or ""),
        },
        "control_effects": dict(_CONTROL_EFFECTS),
    }


def evaluate_snapshot(
    snapshot: Mapping[str, Any],
    *,
    as_of: datetime | str | None = None,
    stale_after_seconds: int = 900,
) -> dict[str, Any]:
    """Public input boundary: normalize all data-shaped failures."""
    try:
        return _evaluate_snapshot(
            snapshot,
            as_of=as_of,
            stale_after_seconds=stale_after_seconds,
        )
    except AdvisoryInputError:
        raise
    except (KeyError, OverflowError, TypeError, ValueError) as exc:
        raise AdvisoryInputError("snapshot contains invalid data") from exc


def summarize_snapshots(
    snapshots: Iterable[Mapping[str, Any]],
    *,
    as_of: datetime | str | None = None,
    stale_after_seconds: int = 900,
) -> dict[str, Any]:
    """Aggregate an offline replay without retaining identifiers or raw rows."""
    replay_clock = _validate_replay_options(as_of, stale_after_seconds)
    primary_actions: Counter[str] = Counter()
    urgency: Counter[str] = Counter()
    reason_codes: Counter[str] = Counter()
    seen = 0
    evaluated = 0
    invalid = 0

    for snapshot in snapshots:
        seen += 1
        try:
            advisory = evaluate_snapshot(
                snapshot,
                as_of=replay_clock,
                stale_after_seconds=stale_after_seconds,
            )
        except AdvisoryInputError:
            invalid += 1
            continue
        evaluated += 1
        primary_actions[advisory["primary_action"]] += 1
        urgency[advisory["urgency"]] += 1
        reason_codes.update(advisory["reason_codes"])

    return {
        "schema_version": _REPLAY_SCHEMA,
        "mode": "offline-replay",
        "snapshots_seen": seen,
        "snapshots_evaluated": evaluated,
        "snapshots_invalid": invalid,
        "primary_actions": dict(sorted(primary_actions.items())),
        "urgency": dict(sorted(urgency.items())),
        "reason_codes": dict(sorted(reason_codes.items())),
        "control_effect_free": True,
    }
