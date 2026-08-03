"""P1-C.1 contracts for secure bounded advisory transition records."""
from __future__ import annotations

import importlib.util
import json
import multiprocessing
import os
import re
import stat
import sys
from pathlib import Path

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "execution-shadow"
PLUGIN_INIT = PLUGIN_DIR / "__init__.py"
TRANSITION_KEYS = {
    "schema_version",
    "mode",
    "source_schema_version",
    "timestamp",
    "turn_digest",
    "decision_digest",
    "previous_primary_action",
    "primary_action",
    "actions",
    "reason_codes",
    "urgency",
    "observed",
    "control_effects",
}
CONTROL_EFFECTS = {
    "block": False,
    "rewrite": False,
    "route": False,
    "schedule": False,
    "spawn": False,
    "stop": False,
}


def _load_plugin(name: str = "execution_shadow_transition_plugin_under_test"):
    for key in [key for key in sys.modules if key == name or key.startswith(f"{name}.")]:
        sys.modules.pop(key, None)
    spec = importlib.util.spec_from_file_location(
        name,
        PLUGIN_INIT,
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module._reset_for_tests()
    return module


def _write_config(
    home: Path,
    *,
    transition_enabled: object | None = None,
    max_record_bytes: object | None = None,
    max_total_bytes: object | None = None,
    max_records: object | None = None,
    advisory_enabled: object = True,
) -> None:
    home.mkdir(parents=True, exist_ok=True)
    section: dict[str, object] = {
        "enabled": True,
        "mode": "shadow",
        "scope_freeze_ratio": 0.60,
        "closure_ratio": 0.75,
        "finalization_ratio": 0.85,
        "max_reviewers": 1,
        "max_remediations": 1,
        "max_event_file_bytes": 5_000_000,
        "advisory_enabled": advisory_enabled,
        "max_advisory_file_bytes": 65_536,
    }
    if transition_enabled is not None:
        section["advisory_transition_log_enabled"] = transition_enabled
    if max_record_bytes is not None:
        section["max_advisory_transition_record_bytes"] = max_record_bytes
    if max_total_bytes is not None:
        section["max_advisory_transition_total_bytes"] = max_total_bytes
    if max_records is not None:
        section["max_advisory_transition_records"] = max_records
    (home / "config.yaml").write_text(
        yaml.safe_dump({"execution_shadow": section}, sort_keys=False),
        encoding="utf-8",
    )


@pytest.fixture
def profile_home(tmp_path):
    home = tmp_path / "profile-home"
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def _base(turn: str = "turn-transition") -> dict[str, str]:
    return {
        "session_id": "SECRET_SESSION_ID",
        "task_id": "SECRET_TASK_ID",
        "turn_id": turn,
        "platform": "feishu",
    }


def _advisory_paths(home: Path) -> list[Path]:
    return sorted((home / "telemetry" / "execution-shadow" / "advisories").glob("*.json"))


def _transition_dir(home: Path) -> Path:
    return home / "telemetry" / "execution-shadow" / "advisory-transitions"


def _transition_paths(home: Path) -> list[Path]:
    return sorted(_transition_dir(home).glob("transition-*.json"))


def _transition_rows(home: Path) -> list[dict]:
    paths = sorted(_transition_paths(home), key=lambda item: (item.stat().st_mtime_ns, item.name))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_transition_store_is_independently_default_off(profile_home):
    _write_config(profile_home)
    plugin = _load_plugin()
    plugin.on_pre_llm_call(**_base(), user_message="SECRET_MESSAGE")
    plugin.on_pre_api_request(
        **_base(), api_call_count=8, budget_used=8, budget_max=10
    )

    assert len(_advisory_paths(profile_home)) == 1
    assert _transition_paths(profile_home) == []
    assert plugin._load_settings(profile_home).advisory_transition_log_enabled is False


def test_only_decision_vector_changes_are_stored_and_terminal_resets(profile_home):
    _write_config(profile_home, transition_enabled=True)
    plugin = _load_plugin()
    results = [
        plugin.on_pre_llm_call(**_base(), user_message="SECRET_MESSAGE"),
        plugin.on_pre_api_request(
            **_base(), api_call_count=1, budget_used=1, budget_max=10
        ),
        plugin.on_pre_api_request(
            **_base(), api_call_count=6, budget_used=6, budget_max=10
        ),
        plugin.on_pre_api_request(
            **_base(), api_call_count=6, budget_used=6, budget_max=10
        ),
        plugin.on_pre_api_request(
            **_base(), api_call_count=8, budget_used=8, budget_max=10
        ),
        plugin.on_pre_api_request(
            **_base(), api_call_count=9, budget_used=9, budget_max=10
        ),
        plugin.on_turn_end(
            **_base(),
            api_call_count=9,
            budget_used=9,
            budget_max=10,
            final_response_present=True,
            completed=True,
            failed=False,
            interrupted=False,
            turn_exit_reason="completed",
        ),
    ]

    assert all(result is None for result in results)
    rows = _transition_rows(profile_home)
    assert [row["primary_action"] for row in rows] == [
        "NO_ACTION",
        "FREEZE_SCOPE",
        "MOVE_TO_CLOSURE",
        "FINALIZE_ONLY",
        "NO_ACTION",
    ]
    assert [row["previous_primary_action"] for row in rows] == [
        None,
        "NO_ACTION",
        "FREEZE_SCOPE",
        "MOVE_TO_CLOSURE",
        "FINALIZE_ONLY",
    ]
    assert all(set(row) == TRANSITION_KEYS for row in rows)
    assert all(row["control_effects"] == CONTROL_EFFECTS for row in rows)
    assert all(re.fullmatch(r"[0-9a-f]{24}", row["decision_digest"]) for row in rows)
    serialized = json.dumps(rows, sort_keys=True)
    for secret in (
        "SECRET_SESSION_ID",
        "SECRET_TASK_ID",
        "turn-transition",
        "SECRET_MESSAGE",
    ):
        assert secret not in serialized
    sidecar = json.loads(_advisory_paths(profile_home)[0].read_text())
    assert sidecar["schema_version"] == "hermes.execution-advisory.v1"
    assert sidecar["primary_action"] == "NO_ACTION"


def test_reason_change_with_same_primary_action_is_a_transition(profile_home, monkeypatch):
    _write_config(profile_home, transition_enabled=True)
    plugin = _load_plugin()
    original = plugin.evaluate_snapshot
    calls = 0

    def varying_advice(snapshot):
        nonlocal calls
        calls += 1
        advice = original(snapshot)
        advice["primary_action"] = "VERIFY_EXECUTION_PROOF"
        advice["actions"] = ["VERIFY_EXECUTION_PROOF"]
        advice["urgency"] = "high"
        advice["reason_codes"] = ["passive_durable_claim_without_execution_proof"]
        if calls > 1:
            advice["reason_codes"].append("implementation_after_verification")
        return advice

    monkeypatch.setattr(plugin, "evaluate_snapshot", varying_advice)
    plugin.on_pre_llm_call(**_base(), user_message="first")
    plugin.on_pre_api_request(
        **_base(), api_call_count=1, budget_used=1, budget_max=10
    )

    rows = _transition_rows(profile_home)
    assert len(rows) == 2
    assert rows[0]["primary_action"] == rows[1]["primary_action"]
    assert rows[0]["decision_digest"] != rows[1]["decision_digest"]
    assert rows[1]["previous_primary_action"] == "VERIFY_EXECUTION_PROOF"


def test_config_cache_hot_enables_transition_store(profile_home):
    _write_config(profile_home, transition_enabled=False)
    plugin = _load_plugin()
    plugin.on_pre_llm_call(**_base("turn-off"), user_message="off")
    assert _transition_paths(profile_home) == []
    _write_config(profile_home, transition_enabled=True)
    plugin.on_pre_llm_call(**_base("turn-on"), user_message="on")
    assert len(_transition_rows(profile_home)) == 1
    assert plugin._load_settings(profile_home).advisory_transition_log_enabled is True


def test_transition_store_gate_is_profile_scoped(profile_home, tmp_path):
    _write_config(profile_home, transition_enabled=True)
    plugin = _load_plugin()
    plugin.on_pre_llm_call(**_base("default-turn"), user_message="default")
    assert len(_transition_rows(profile_home)) == 1

    other_home = tmp_path / "other-profile"
    _write_config(other_home, transition_enabled=False)
    token = set_hermes_home_override(other_home)
    try:
        plugin.on_pre_llm_call(**_base("other-turn"), user_message="other")
    finally:
        reset_hermes_home_override(token)
    assert len(_advisory_paths(other_home)) == 1
    assert _transition_paths(other_home) == []


def test_store_failure_preserves_sidecar_and_retries(profile_home, monkeypatch):
    _write_config(profile_home, transition_enabled=True)
    plugin = _load_plugin()
    original = plugin._store_advisory_transition
    attempts = 0

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise OSError("SECRET_TRANSITION_FAILURE")

    monkeypatch.setattr(plugin, "_store_advisory_transition", fail_once)
    plugin.on_pre_llm_call(**_base(), user_message="first")
    assert len(_advisory_paths(profile_home)) == 1
    assert _transition_paths(profile_home) == []
    monkeypatch.setattr(plugin, "_store_advisory_transition", original)
    plugin.on_pre_api_request(
        **_base(), api_call_count=1, budget_used=1, budget_max=10
    )
    assert attempts == 1
    assert [row["primary_action"] for row in _transition_rows(profile_home)] == [
        "NO_ACTION"
    ]


def test_oversized_record_is_rejected_and_retried(profile_home, monkeypatch):
    _write_config(
        profile_home,
        transition_enabled=True,
        max_record_bytes=4_096,
    )
    plugin = _load_plugin()
    original_dumps = plugin.json.dumps

    def inflate_transition(payload, *args, **kwargs):
        serialized = original_dumps(payload, *args, **kwargs)
        if payload.get("schema_version") == "hermes.execution-advisory-transition.v1":
            return serialized + (" " * 5_000)
        return serialized

    monkeypatch.setattr(plugin.json, "dumps", inflate_transition)
    plugin.on_pre_llm_call(**_base(), user_message="first")
    assert _transition_paths(profile_home) == []
    assert len(_advisory_paths(profile_home)) == 1

    monkeypatch.setattr(plugin.json, "dumps", original_dumps)
    plugin.on_pre_api_request(
        **_base(), api_call_count=1, budget_used=1, budget_max=10
    )
    assert [row["primary_action"] for row in _transition_rows(profile_home)] == [
        "NO_ACTION"
    ]


def test_count_and_total_byte_retention_are_bounded(profile_home, monkeypatch):
    _write_config(
        profile_home,
        transition_enabled=True,
        max_record_bytes=64_000,
        max_total_bytes=100_000,
        max_records=2,
    )
    plugin = _load_plugin()
    original_dumps = plugin.json.dumps

    def inflate_transition(payload, *args, **kwargs):
        serialized = original_dumps(payload, *args, **kwargs)
        if payload.get("schema_version") == "hermes.execution-advisory-transition.v1":
            return serialized + (" " * 40_000)
        return serialized

    monkeypatch.setattr(plugin.json, "dumps", inflate_transition)
    plugin.on_pre_llm_call(**_base(), user_message="first")
    for used in (6, 8, 9):
        plugin.on_pre_api_request(
            **_base(), api_call_count=used, budget_used=used, budget_max=10
        )
    plugin.on_turn_end(
        **_base(),
        completed=True,
        failed=False,
        interrupted=False,
        final_response_present=True,
        turn_exit_reason="completed",
        budget_used=9,
        budget_max=10,
    )

    paths = _transition_paths(profile_home)
    assert len(paths) == 2
    assert sum(path.stat().st_size for path in paths) <= 100_000
    assert all(path.stat().st_size <= 64_000 for path in paths)


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink semantics required")
def test_symlinked_transition_directory_is_rejected(profile_home):
    _write_config(profile_home, transition_enabled=True)
    plugin = _load_plugin()
    root = profile_home / "telemetry" / "execution-shadow"
    root.mkdir(parents=True, mode=0o700)
    external = profile_home / "external"
    external.mkdir(mode=0o700)
    transition_dir = root / "advisory-transitions"
    transition_dir.symlink_to(external, target_is_directory=True)

    plugin.on_pre_llm_call(**_base(), user_message="first")

    assert list(external.iterdir()) == []
    assert transition_dir.is_symlink()
    assert len(_advisory_paths(profile_home)) == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory-fd semantics required")
def test_directory_path_swap_cannot_redirect_write(profile_home, monkeypatch):
    _write_config(profile_home, transition_enabled=True)
    plugin = _load_plugin()
    original_open = plugin._open_advisory_transition_directory
    root = profile_home / "telemetry" / "execution-shadow"
    parked = root / "advisory-transitions-parked"
    external = profile_home / "external"
    external.mkdir(mode=0o700)

    def open_then_swap(path):
        fd = original_open(path)
        current = path / "advisory-transitions"
        current.rename(parked)
        current.symlink_to(external, target_is_directory=True)
        return fd

    monkeypatch.setattr(plugin, "_open_advisory_transition_directory", open_then_swap)
    plugin.on_pre_llm_call(**_base(), user_message="first")

    assert list(external.iterdir()) == []
    assert len(list(parked.glob("transition-*.json"))) == 1
    assert len(_advisory_paths(profile_home)) == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX hardlink semantics required")
def test_hardlinked_lock_file_is_rejected(profile_home):
    _write_config(profile_home, transition_enabled=True)
    plugin = _load_plugin()
    root = profile_home / "telemetry" / "execution-shadow"
    transition_dir = root / "advisory-transitions"
    transition_dir.mkdir(parents=True, mode=0o700)
    external = profile_home / "external-lock"
    external.write_text("SENTINEL", encoding="utf-8")
    os.link(external, transition_dir / ".transition.lock")

    plugin.on_pre_llm_call(**_base(), user_message="first")

    assert external.read_text(encoding="utf-8") == "SENTINEL"
    assert external.stat().st_nlink == 2
    assert _transition_paths(profile_home) == []
    assert len(_advisory_paths(profile_home)) == 1


def test_partial_write_failure_publishes_no_record(profile_home, monkeypatch):
    _write_config(profile_home, transition_enabled=True)
    plugin = _load_plugin()

    def partial_then_fail(fd, data):
        plugin.os.write(fd, data[: max(1, len(data) // 2)])
        raise OSError("partial write")

    monkeypatch.setattr(plugin, "_write_transition_bytes", partial_then_fail)
    plugin.on_pre_llm_call(**_base(), user_message="first")

    assert _transition_paths(profile_home) == []
    assert not list(_transition_dir(profile_home).glob(".transition-tmp-*"))
    assert len(_advisory_paths(profile_home)) == 1


def test_file_sync_failure_publishes_no_record(profile_home, monkeypatch):
    _write_config(profile_home, transition_enabled=True)
    plugin = _load_plugin()
    monkeypatch.setattr(
        plugin,
        "_sync_transition_file",
        lambda _fd: (_ for _ in ()).throw(OSError("sync failed")),
    )

    plugin.on_pre_llm_call(**_base(), user_message="first")

    assert _transition_paths(profile_home) == []
    assert not list(_transition_dir(profile_home).glob(".transition-tmp-*"))
    assert len(_advisory_paths(profile_home)) == 1


def test_directory_sync_failure_rolls_back_published_record(profile_home, monkeypatch):
    _write_config(profile_home, transition_enabled=True)
    plugin = _load_plugin()
    monkeypatch.setattr(
        plugin,
        "_sync_transition_directory",
        lambda _fd: (_ for _ in ()).throw(OSError("directory sync failed")),
    )

    plugin.on_pre_llm_call(**_base(), user_message="first")

    assert _transition_paths(profile_home) == []
    assert not list(_transition_dir(profile_home).glob(".transition-tmp-*"))
    assert len(_advisory_paths(profile_home)) == 1


def test_close_failure_publishes_no_record(profile_home, monkeypatch):
    _write_config(profile_home, transition_enabled=True)
    plugin = _load_plugin()
    original_close = plugin._close_transition_file

    def close_then_fail(fd):
        original_close(fd)
        raise OSError("close failed")

    monkeypatch.setattr(plugin, "_close_transition_file", close_then_fail)
    plugin.on_pre_llm_call(**_base(), user_message="first")

    assert _transition_paths(profile_home) == []
    assert not list(_transition_dir(profile_home).glob(".transition-tmp-*"))
    assert len(_advisory_paths(profile_home)) == 1


def test_publish_collision_preserves_existing_record_and_retries(profile_home, monkeypatch):
    _write_config(profile_home, transition_enabled=True, max_records=10)
    plugin = _load_plugin()
    plugin.on_pre_llm_call(**_base("first-turn"), user_message="first")
    existing = _transition_paths(profile_home)[0]
    existing_bytes = existing.read_bytes()
    original_name = plugin._transition_record_name
    calls = 0

    def collide_once(record):
        nonlocal calls
        calls += 1
        if calls == 1:
            return existing.name
        return original_name(record)

    monkeypatch.setattr(plugin, "_transition_record_name", collide_once)
    plugin.on_pre_llm_call(**_base("second-turn"), user_message="second")

    assert existing.read_bytes() == existing_bytes
    assert len(_transition_paths(profile_home)) == 2
    assert calls >= 2


def test_day_rollover_changes_record_prefix(profile_home, monkeypatch):
    _write_config(profile_home, transition_enabled=True)
    plugin = _load_plugin()
    days = iter(("20260804", "20260805"))
    monkeypatch.setattr(plugin, "_utc_day", lambda: next(days))
    plugin.on_pre_llm_call(**_base("day-one"), user_message="one")
    plugin.on_pre_llm_call(**_base("day-two"), user_message="two")
    names = [path.name for path in _transition_paths(profile_home)]
    assert any(name.startswith("transition-20260804-") for name in names)
    assert any(name.startswith("transition-20260805-") for name in names)


def _multiprocess_writer(home: str, start, queue, index: int) -> None:
    try:
        path = Path(home)
        token = set_hermes_home_override(path)
        try:
            plugin = _load_plugin(f"transition_worker_{os.getpid()}_{index}")
            start.wait(10)
            for item in range(5):
                plugin.on_pre_llm_call(
                    session_id=f"secret-session-{index}",
                    task_id=f"secret-task-{index}",
                    turn_id=f"turn-{index}-{item}",
                    platform="feishu",
                    user_message="secret-message",
                )
        finally:
            reset_hermes_home_override(token)
        queue.put(None)
    except BaseException as exc:  # pragma: no cover - child diagnostic only
        queue.put(type(exc).__name__)


@pytest.mark.skipif(os.name == "nt", reason="POSIX flock/fork semantics required")
def test_multiprocess_writers_share_lock_and_retention(profile_home):
    _write_config(
        profile_home,
        transition_enabled=True,
        max_records=10,
        max_total_bytes=1_000_000,
    )
    ctx = multiprocessing.get_context("fork")
    start = ctx.Event()
    queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=_multiprocess_writer,
            args=(str(profile_home), start, queue, index),
        )
        for index in range(4)
    ]
    for worker in workers:
        worker.start()
    start.set()
    for worker in workers:
        worker.join(15)
        assert worker.exitcode == 0
    assert [queue.get(timeout=2) for _ in workers] == [None] * len(workers)

    paths = _transition_paths(profile_home)
    assert len(paths) == 10
    assert len(_transition_rows(profile_home)) == 10
    assert not list(_transition_dir(profile_home).glob(".transition-tmp-*"))
    if os.name != "nt":
        lock_path = _transition_dir(profile_home) / ".transition.lock"
        assert stat.S_IMODE(_transition_dir(profile_home).stat().st_mode) == 0o700
        assert stat.S_IMODE(lock_path.stat().st_mode) == 0o600
        assert lock_path.stat().st_nlink == 1
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in paths)


def test_malformed_transition_config_fails_closed_and_caps_are_clamped(profile_home):
    _write_config(
        profile_home,
        transition_enabled="true",
        max_record_bytes=99_999_999,
        max_total_bytes=999_999_999,
        max_records=999_999,
    )
    plugin = _load_plugin()
    settings = plugin._load_settings(profile_home)
    assert settings.advisory_transition_log_enabled is False
    assert settings.max_advisory_transition_record_bytes == 1_000_000
    assert settings.max_advisory_transition_total_bytes == 100_000_000
    assert settings.max_advisory_transition_records == 4_096
    plugin.on_pre_llm_call(**_base(), user_message="disabled")
    assert _transition_paths(profile_home) == []


def test_transition_gate_cannot_bypass_advisory_gate(profile_home):
    _write_config(
        profile_home,
        transition_enabled=True,
        advisory_enabled=False,
    )
    plugin = _load_plugin()
    plugin.on_pre_llm_call(**_base(), user_message="disabled")
    assert plugin._load_settings(profile_home).advisory_transition_log_enabled is True
    assert _advisory_paths(profile_home) == []
    assert _transition_paths(profile_home) == []
