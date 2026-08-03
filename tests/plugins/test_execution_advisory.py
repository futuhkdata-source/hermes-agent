"""Contract tests for the offline-only P1-B execution advisory engine."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ADVISORY_FILE = (
    REPO_ROOT / "plugins" / "observability" / "execution-shadow" / "advisory.py"
)
REPLAY_FILE = (
    REPO_ROOT / "plugins" / "observability" / "execution-shadow" / "replay.py"
)


def _load_advisory():
    assert ADVISORY_FILE.is_file(), "P1-B advisory module is not implemented"
    name = "execution_advisory_under_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, ADVISORY_FILE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_replay():
    advisory_spec = importlib.util.spec_from_file_location("advisory", ADVISORY_FILE)
    assert advisory_spec and advisory_spec.loader
    advisory = importlib.util.module_from_spec(advisory_spec)
    sys.modules["advisory"] = advisory
    advisory_spec.loader.exec_module(advisory)

    name = "execution_advisory_replay_under_test"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, REPLAY_FILE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(
    *,
    ratio: float = 0.2,
    phase: str = "implementation",
    warnings: list[str] | None = None,
    completed: bool | None = None,
    failed: bool | None = None,
    interrupted: bool | None = None,
    exit_reason: str = "",
    updated_at: datetime | None = None,
) -> dict:
    used = int(ratio * 100)
    terminal = phase == "terminal"
    return {
        "schema_version": "hermes.execution-shadow.v1",
        "mode": "shadow",
        "session_id": "SECRET_SESSION_ID",
        "task_id": "SECRET_TASK_ID",
        "turn_id": "SECRET_TURN_ID",
        "platform": "feishu",
        "phase": phase,
        "phase_history": ["discovery", phase] if phase != "discovery" else ["discovery"],
        "policy": {
            "scope_freeze_ratio": 0.60,
            "closure_ratio": 0.75,
            "finalization_ratio": 0.85,
            "max_reviewers": 1,
            "max_remediations": 1,
        },
        "budget": {
            "used": used,
            "max": 100,
            "ratio": ratio,
            "scope_frozen": ratio >= 0.60,
            "closure_reserved": ratio >= 0.75,
            "finalization_only": ratio >= 0.85,
        },
        "counts": {
            "api_calls": used,
            "api_errors": 0,
            "tool_calls_attempted": 3,
            "tool_errors": 0,
            "assistant_tool_calls": 3,
            "helpers_started": 1,
            "helpers_completed": 1,
            "reviewers_started": 0,
            "remediations_started": 0,
        },
        "evidence": {
            "artifact_created": False,
            "tests_passed": False,
            "commit_created": False,
            "delivery_verified": False,
            "rollback_ready": False,
            "final_response_present": terminal,
            "terminal_verdict": None,
            "durable_execution_proven": False,
        },
        "terminal": {
            "completed": completed,
            "failed": failed,
            "interrupted": interrupted,
            "exit_reason": exit_reason,
        },
        "warnings": list(warnings or []),
        "created_at": "2026-08-01T00:00:00+00:00",
        "updated_at": (updated_at or datetime(2026, 8, 1, tzinfo=timezone.utc)).isoformat(),
    }


@pytest.mark.parametrize(
    ("ratio", "expected_action", "expected_urgency"),
    [
        (0.20, "NO_ACTION", "none"),
        (0.60, "FREEZE_SCOPE", "low"),
        (0.75, "MOVE_TO_CLOSURE", "medium"),
        (0.85, "FINALIZE_ONLY", "high"),
    ],
)
def test_budget_boundaries_emit_advice_without_control_effects(
    ratio, expected_action, expected_urgency
):
    advisory = _load_advisory()

    result = advisory.evaluate_snapshot(_snapshot(ratio=ratio))

    assert result["primary_action"] == expected_action
    assert result["actions"][0] == expected_action
    assert result["urgency"] == expected_urgency
    assert result["control_effects"] == {
        "block": False,
        "rewrite": False,
        "route": False,
        "schedule": False,
        "spawn": False,
        "stop": False,
    }


def test_action_precedence_is_finite_and_deterministic():
    advisory = _load_advisory()
    snapshot = _snapshot(
        ratio=0.90,
        warnings=[
            "passive_durable_claim_without_execution_proof",
            "reviewer_limit_exceeded",
            "discovery_started_after_scope_freeze",
        ],
    )

    first = advisory.evaluate_snapshot(snapshot)
    second = advisory.evaluate_snapshot(snapshot)

    assert first == second
    assert first["primary_action"] == "VERIFY_EXECUTION_PROOF"
    assert first["actions"] == [
        "VERIFY_EXECUTION_PROOF",
        "FINALIZE_ONLY",
        "FREEZE_SCOPE",
    ]
    assert set(first["reason_codes"]) == {
        "budget_finalization_only",
        "passive_durable_claim_without_execution_proof",
        "reviewer_limit_exceeded",
        "discovery_started_after_scope_freeze",
    }


@pytest.mark.parametrize(
    ("terminal", "reason"),
    [
        (
            {"completed": False, "failed": True, "interrupted": False, "exit_reason": "failed"},
            "terminal_failed",
        ),
        (
            {"completed": False, "failed": False, "interrupted": True, "exit_reason": "interrupted"},
            "terminal_interrupted",
        ),
        (
            {"completed": False, "failed": False, "interrupted": False, "exit_reason": "max_iterations_reached"},
            "terminal_max_iterations_reached",
        ),
    ],
)
def test_abnormal_terminal_outcomes_require_review(terminal, reason):
    advisory = _load_advisory()
    snapshot = _snapshot(phase="terminal", **terminal)

    result = advisory.evaluate_snapshot(snapshot)

    assert result["primary_action"] == "REVIEW_TERMINAL_OUTCOME"
    assert reason in result["reason_codes"]
    assert result["observed"]["terminal"] is True


def test_successful_terminal_turn_needs_no_action():
    advisory = _load_advisory()
    snapshot = _snapshot(
        ratio=0.90,
        phase="terminal",
        warnings=["implementation_after_verification"],
        completed=True,
        failed=False,
        interrupted=False,
        exit_reason="text_response",
    )

    result = advisory.evaluate_snapshot(snapshot)

    assert result["primary_action"] == "NO_ACTION"
    assert result["reason_codes"] == []


def test_stale_detection_uses_explicit_as_of_and_never_guesses():
    advisory = _load_advisory()
    updated = datetime(2026, 8, 1, tzinfo=timezone.utc)
    snapshot = _snapshot(updated_at=updated)

    without_clock = advisory.evaluate_snapshot(snapshot)
    with_clock = advisory.evaluate_snapshot(
        snapshot,
        as_of=updated + timedelta(minutes=16),
        stale_after_seconds=900,
    )

    assert without_clock["primary_action"] == "NO_ACTION"
    assert with_clock["primary_action"] == "REVIEW_STALE_TURN"
    assert with_clock["reason_codes"] == ["nonterminal_stale"]


def test_output_is_metadata_only_and_uses_digest_not_identifiers():
    advisory = _load_advisory()

    result = advisory.evaluate_snapshot(_snapshot())
    serialized = json.dumps(result, sort_keys=True)

    assert result["schema_version"] == "hermes.execution-advisory.v1"
    assert result["mode"] == "advisory"
    assert result["source_schema_version"] == "hermes.execution-shadow.v1"
    assert len(result["turn_digest"]) == 24
    for secret in (
        "SECRET_SESSION_ID",
        "SECRET_TASK_ID",
        "SECRET_TURN_ID",
    ):
        assert secret not in serialized
    for forbidden_key in ("session_id", "task_id", "turn_id", "content", "tool_result"):
        assert forbidden_key not in result


@pytest.mark.parametrize(
    "mutation",
    [
        lambda snap: snap.update(schema_version="hermes.execution-shadow.v2"),
        lambda snap: snap.update(phase="invented"),
        lambda snap: snap["budget"].update(ratio=1.5),
        lambda snap: snap["budget"].update(used=90, ratio=0.2),
        lambda snap: snap["budget"].update(scope_frozen=True),
        lambda snap: snap["policy"].update(scope_freeze_ratio=0.9, closure_ratio=0.7),
        lambda snap: snap["terminal"].update(
            exit_reason="failed: SECRET_DYNAMIC_EXCEPTION"
        ),
        lambda snap: snap.update(content="SECRET_RAW_CONTENT"),
    ],
)
def test_invalid_or_future_snapshot_is_rejected(mutation):
    advisory = _load_advisory()
    snapshot = _snapshot()
    mutation(snapshot)

    with pytest.raises(advisory.AdvisoryInputError):
        advisory.evaluate_snapshot(snapshot)


def test_offline_replay_aggregates_without_identifiers_or_raw_text():
    advisory = _load_advisory()
    snapshots = [
        _snapshot(ratio=0.2),
        _snapshot(ratio=0.75),
        _snapshot(
            phase="terminal",
            completed=False,
            failed=False,
            interrupted=False,
            exit_reason="max_iterations_reached",
        ),
    ]

    result = advisory.summarize_snapshots(snapshots)
    serialized = json.dumps(result, sort_keys=True)

    assert result == {
        "schema_version": "hermes.execution-advisory-replay.v1",
        "mode": "offline-replay",
        "snapshots_seen": 3,
        "snapshots_evaluated": 3,
        "snapshots_invalid": 0,
        "primary_actions": {
            "MOVE_TO_CLOSURE": 1,
            "NO_ACTION": 1,
            "REVIEW_TERMINAL_OUTCOME": 1,
        },
        "urgency": {"high": 1, "medium": 1, "none": 1},
        "reason_codes": {
            "budget_closure_reserved": 1,
            "terminal_max_iterations_reached": 1,
        },
        "control_effect_free": True,
    }
    assert "SECRET_" not in serialized


def test_offline_replay_counts_invalid_records_without_leaking_them():
    advisory = _load_advisory()
    invalid = _snapshot(phase="terminal")
    invalid["terminal"]["exit_reason"] = "failed: SECRET_RAW_CONTENT"

    result = advisory.summarize_snapshots([invalid])

    assert result["snapshots_seen"] == 1
    assert result["snapshots_evaluated"] == 0
    assert result["snapshots_invalid"] == 1
    assert "SECRET" not in json.dumps(result)


def test_replay_cli_emits_deterministic_aggregate_only(tmp_path):
    assert REPLAY_FILE.is_file(), "P1-B replay CLI is not implemented"
    root = tmp_path / "telemetry" / "execution-shadow"
    turns = root / "turns"
    turns.mkdir(parents=True)
    for index, snapshot in enumerate((_snapshot(ratio=0.2), _snapshot(ratio=0.75))):
        (turns / f"turn-{index}.json").write_text(
            json.dumps(snapshot), encoding="utf-8"
        )

    command = [
        sys.executable,
        str(REPLAY_FILE),
        "--telemetry-root",
        str(root),
        "--as-of",
        "2026-08-01T00:05:00+00:00",
    ]
    first = subprocess.run(command, check=False, capture_output=True, text=True)
    second = subprocess.run(command, check=False, capture_output=True, text=True)

    assert first.returncode == 0
    assert second.returncode == 0
    assert first.stdout == second.stdout
    assert first.stderr == second.stderr == ""
    payload = json.loads(first.stdout)
    assert payload["snapshots_seen"] == 2
    assert payload["snapshots_evaluated"] == 2
    assert payload["snapshots_invalid"] == 0
    assert payload["primary_actions"] == {"MOVE_TO_CLOSURE": 1, "NO_ACTION": 1}
    assert "SECRET_" not in first.stdout
    assert str(root) not in first.stdout


def test_replay_cli_reports_invalid_count_without_row_details(tmp_path):
    assert REPLAY_FILE.is_file(), "P1-B replay CLI is not implemented"
    root = tmp_path / "telemetry" / "execution-shadow"
    turns = root / "turns"
    turns.mkdir(parents=True)
    (turns / "bad-secret-name.json").write_text(
        '{"content":"SECRET_INVALID_JSON"', encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(REPLAY_FILE),
            "--telemetry-root",
            str(root),
            "--as-of",
            "2026-08-01T00:05:00+00:00",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["snapshots_seen"] == 1
    assert payload["snapshots_evaluated"] == 0
    assert payload["snapshots_invalid"] == 1
    assert "SECRET" not in completed.stdout
    assert "bad-secret-name" not in completed.stdout


def test_replay_cli_missing_turns_directory_fails_closed(tmp_path):
    assert REPLAY_FILE.is_file(), "P1-B replay CLI is not implemented"
    root = tmp_path / "SECRET_MISSING_ROOT"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPLAY_FILE),
            "--telemetry-root",
            str(root),
            "--as-of",
            "2026-08-01T00:05:00+00:00",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["snapshots_seen"] == 0
    assert payload["snapshots_evaluated"] == 0
    assert "SECRET_MISSING_ROOT" not in completed.stdout


def test_terminal_review_and_execution_proof_coexist_without_stale_phase_actions():
    advisory = _load_advisory()
    snapshot = _snapshot(
        ratio=0.90,
        phase="terminal",
        warnings=[
            "implementation_after_verification",
            "passive_durable_claim_without_execution_proof",
        ],
        completed=False,
        failed=False,
        interrupted=False,
        exit_reason="max_iterations_reached",
    )

    result = advisory.evaluate_snapshot(snapshot)

    assert result["actions"] == [
        "REVIEW_TERMINAL_OUTCOME",
        "VERIFY_EXECUTION_PROOF",
    ]
    assert result["reason_codes"] == [
        "terminal_max_iterations_reached",
        "passive_durable_claim_without_execution_proof",
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda snap: snap.pop("session_id"),
        lambda snap: snap.update(session_id=["unhashable"]),
        lambda snap: snap.update(phase=["unhashable"]),
        lambda snap: snap.update(
            phase_history=["discovery", "verification", "implementation"]
        ),
        lambda snap: snap["counts"].update(api_errors=-1),
        lambda snap: snap["counts"].update(
            helpers_started=1, helpers_completed=2
        ),
        lambda snap: snap["evidence"].pop("tests_passed"),
        lambda snap: snap["policy"].update(max_reviewers=-1),
        lambda snap: snap["warnings"].append("SECRET_DYNAMIC_WARNING"),
        lambda snap: snap.update(updated_at="2025-01-01T00:00:00+00:00"),
        lambda snap: (
            snap.update(phase="terminal", phase_history=["discovery", "terminal"]),
            snap["terminal"].update(
                completed=True,
                failed=True,
                interrupted=False,
                exit_reason="text_response",
            ),
        ),
        lambda snap: snap["terminal"].update(
            completed=True, failed=False, interrupted=False, exit_reason="text_response"
        ),
        lambda snap: snap["budget"].update(ratio=10**10000),
    ],
)
def test_complete_p1a_schema_and_semantics_are_required(mutation):
    advisory = _load_advisory()
    snapshot = _snapshot()
    mutation(snapshot)

    with pytest.raises(advisory.AdvisoryInputError):
        advisory.evaluate_snapshot(snapshot)


def test_empty_turns_directory_fails_closed(tmp_path):
    root = tmp_path / "telemetry" / "execution-shadow"
    (root / "turns").mkdir(parents=True)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPLAY_FILE),
            "--telemetry-root",
            str(root),
            "--as-of",
            "2026-08-01T00:05:00+00:00",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["snapshots_seen"] == 0


@pytest.mark.parametrize(
    "options",
    [
        ["--as-of", "SECRET_BAD_CLOCK"],
        [
            "--as-of",
            "2026-08-01T00:05:00+00:00",
            "--stale-after-seconds",
            "0",
        ],
    ],
)
def test_invalid_replay_options_fail_without_traceback_or_path_leak(
    tmp_path, options
):
    root = tmp_path / "SECRET_OPTION_ROOT"
    turns = root / "turns"
    turns.mkdir(parents=True)
    terminal = _snapshot(
        phase="terminal",
        completed=True,
        failed=False,
        interrupted=False,
        exit_reason="text_response",
    )
    (turns / "terminal.json").write_text(json.dumps(terminal), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPLAY_FILE),
            "--telemetry-root",
            str(root),
            *options,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    payload = json.loads(completed.stdout)
    assert payload["snapshots_evaluated"] == 0
    assert "SECRET" not in completed.stdout


@pytest.mark.skipif(os.name == "nt", reason="FIFO and O_NOFOLLOW are POSIX-specific")
def test_replay_rejects_symlink_without_following_external_target(tmp_path):
    root = tmp_path / "telemetry" / "execution-shadow"
    turns = root / "turns"
    turns.mkdir(parents=True)
    target = tmp_path / "SECRET_EXTERNAL_TARGET.json"
    target.write_text(json.dumps(_snapshot()), encoding="utf-8")
    (turns / "linked.json").symlink_to(target)

    completed = subprocess.run(
        [
            sys.executable,
            str(REPLAY_FILE),
            "--telemetry-root",
            str(root),
            "--as-of",
            "2026-08-01T00:05:00+00:00",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["snapshots_invalid"] == 1
    assert "SECRET" not in completed.stdout


@pytest.mark.skipif(os.name == "nt", reason="FIFO is POSIX-specific")
def test_replay_rejects_fifo_without_blocking(tmp_path):
    root = tmp_path / "telemetry" / "execution-shadow"
    turns = root / "turns"
    turns.mkdir(parents=True)
    os.mkfifo(turns / "SECRET_PIPE.json")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPLAY_FILE),
            "--telemetry-root",
            str(root),
            "--as-of",
            "2026-08-01T00:05:00+00:00",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )

    assert completed.returncode == 2
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["snapshots_invalid"] == 1
    assert "SECRET" not in completed.stdout


def test_replay_snapshot_iteration_is_lazy_and_does_not_preload_rows(tmp_path):
    replay = _load_replay()
    turns = tmp_path / "turns"
    turns.mkdir()
    for index in range(3):
        (turns / f"{index}.json").write_text(
            json.dumps(_snapshot()), encoding="utf-8"
        )

    stats = replay.LoadStats()
    snapshots = replay._iter_snapshots(turns, stats)

    assert iter(snapshots) is snapshots
    assert stats.seen == 0
    first = next(snapshots)
    assert first["schema_version"] == "hermes.execution-shadow.v1"
    assert stats.seen == 1
    assert stats.invalid == 0
