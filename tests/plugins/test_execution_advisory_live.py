"""Stage C contract tests for default-off live execution advice sidecars."""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "execution-shadow"
PLUGIN_INIT = PLUGIN_DIR / "__init__.py"


def _load_plugin():
    name = "execution_shadow_live_plugin_under_test"
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
    advisory_enabled: object | None = None,
    max_advisory_file_bytes: object | None = None,
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
    }
    if advisory_enabled is not None:
        section["advisory_enabled"] = advisory_enabled
    if max_advisory_file_bytes is not None:
        section["max_advisory_file_bytes"] = max_advisory_file_bytes
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


def _base(turn: str = "turn-live") -> dict[str, str]:
    return {
        "session_id": "SECRET_SESSION_ID",
        "task_id": "SECRET_TASK_ID",
        "turn_id": turn,
        "platform": "feishu",
    }


def _snapshot_paths(home: Path) -> list[Path]:
    return sorted((home / "telemetry" / "execution-shadow" / "turns").glob("*.json"))


def _advisory_paths(home: Path) -> list[Path]:
    return sorted((home / "telemetry" / "execution-shadow" / "advisories").glob("*.json"))


def _event_paths(home: Path) -> list[Path]:
    return sorted((home / "telemetry" / "execution-shadow" / "events").glob("*.jsonl"))


def test_advisory_is_default_off_and_existing_hooks_return_none(profile_home):
    _write_config(profile_home)
    plugin = _load_plugin()

    first = plugin.on_pre_llm_call(
        **_base(), user_message="SECRET_USER_MESSAGE"
    )
    second = plugin.on_pre_api_request(
        **_base(), api_call_count=8, budget_used=8, budget_max=10
    )

    assert first is second is None
    assert len(_snapshot_paths(profile_home)) == 1
    assert len(_event_paths(profile_home)) == 1
    assert _advisory_paths(profile_home) == []
    assert plugin._load_settings(profile_home).advisory_enabled is False


def test_enabled_advisory_writes_bounded_atomic_metadata_only_sidecar(profile_home):
    _write_config(
        profile_home,
        advisory_enabled=True,
        max_advisory_file_bytes=65_536,
    )
    plugin = _load_plugin()

    plugin.on_pre_llm_call(
        **_base(), user_message="SECRET_USER_MESSAGE"
    )
    plugin.on_pre_api_request(
        **_base(), api_call_count=8, budget_used=8, budget_max=10
    )

    paths = _advisory_paths(profile_home)
    assert len(paths) == 1
    assert paths[0].stat().st_size <= 65_536
    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["schema_version"] == "hermes.execution-advisory.v1"
    assert payload["primary_action"] == "MOVE_TO_CLOSURE"
    assert paths[0].stem == payload["turn_digest"]
    assert payload["control_effects"] == {
        "block": False,
        "rewrite": False,
        "route": False,
        "schedule": False,
        "spawn": False,
        "stop": False,
    }

    serialized = json.dumps(payload, sort_keys=True)
    for secret in (
        "SECRET_SESSION_ID",
        "SECRET_TASK_ID",
        "turn-live",
        "SECRET_USER_MESSAGE",
    ):
        assert secret not in serialized
    assert not list(paths[0].parent.glob("*.tmp"))
    assert not list(paths[0].parent.glob(".*.tmp"))

    if os.name != "nt":
        assert stat.S_IMODE(paths[0].parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(paths[0].stat().st_mode) == 0o600


def test_config_cache_observes_enable_gate_without_module_reload(profile_home):
    _write_config(profile_home, advisory_enabled=False)
    plugin = _load_plugin()
    plugin.on_pre_llm_call(**_base("turn-off"), user_message="off")
    assert _advisory_paths(profile_home) == []

    _write_config(profile_home, advisory_enabled=True)
    plugin.on_pre_llm_call(**_base("turn-on"), user_message="on")

    assert len(_advisory_paths(profile_home)) == 1
    assert plugin._load_settings(profile_home).advisory_enabled is True


def test_advisory_gate_is_profile_scoped(profile_home, tmp_path):
    _write_config(profile_home, advisory_enabled=True)
    plugin = _load_plugin()
    plugin.on_pre_llm_call(**_base("default-turn"), user_message="default")
    assert len(_advisory_paths(profile_home)) == 1

    other_home = tmp_path / "other-profile"
    _write_config(other_home, advisory_enabled=False)
    token = set_hermes_home_override(other_home)
    try:
        plugin.on_pre_llm_call(**_base("other-turn"), user_message="other")
    finally:
        reset_hermes_home_override(token)

    assert len(_snapshot_paths(other_home)) == 1
    assert _advisory_paths(other_home) == []


def test_evaluator_failure_preserves_p1a_snapshot_and_event(profile_home, monkeypatch):
    _write_config(profile_home, advisory_enabled=True)
    plugin = _load_plugin()
    monkeypatch.setattr(
        plugin,
        "evaluate_snapshot",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("SECRET_FAILURE")),
    )

    result = plugin.on_pre_llm_call(**_base(), user_message="SECRET_INPUT")

    assert result is None
    assert len(_snapshot_paths(profile_home)) == 1
    assert len(_event_paths(profile_home)) == 1
    assert _advisory_paths(profile_home) == []


def test_advisory_write_failure_preserves_p1a_and_future_updates(profile_home, monkeypatch):
    _write_config(profile_home, advisory_enabled=True)
    plugin = _load_plugin()
    original = plugin._atomic_write_json
    advisory_attempts = 0

    def fail_advisory(path, payload, *, max_bytes=None):
        nonlocal advisory_attempts
        if path.parent.name == "advisories":
            advisory_attempts += 1
            raise OSError("SECRET_DISK_FAILURE")
        if max_bytes is None:
            return original(path, payload)
        return original(path, payload, max_bytes=max_bytes)

    monkeypatch.setattr(plugin, "_atomic_write_json", fail_advisory)
    plugin.on_pre_llm_call(**_base(), user_message="first")
    plugin.on_pre_api_request(
        **_base(), api_call_count=8, budget_used=8, budget_max=10
    )

    assert len(_snapshot_paths(profile_home)) == 1
    snapshot = json.loads(_snapshot_paths(profile_home)[0].read_text())
    assert snapshot["budget"]["ratio"] == 0.8
    assert len(_event_paths(profile_home)) == 1
    assert advisory_attempts == 2
    assert _advisory_paths(profile_home) == []


def test_oversized_advice_does_not_replace_last_good_sidecar(profile_home, monkeypatch):
    _write_config(
        profile_home,
        advisory_enabled=True,
        max_advisory_file_bytes=4_096,
    )
    plugin = _load_plugin()
    plugin.on_pre_llm_call(**_base(), user_message="first")
    path = _advisory_paths(profile_home)[0]
    original = path.read_bytes()
    original_write = plugin._atomic_write_json
    original_dumps = plugin.json.dumps
    advisory_write_attempts = 0
    inflated_serializations = 0

    def track_advisory_write(path, payload, *, max_bytes=None):
        nonlocal advisory_write_attempts
        if path.parent.name == "advisories":
            advisory_write_attempts += 1
        return original_write(path, payload, max_bytes=max_bytes)

    def inflate_advisory_json(payload, *args, **kwargs):
        nonlocal inflated_serializations
        serialized = original_dumps(payload, *args, **kwargs)
        if payload.get("schema_version") == "hermes.execution-advisory.v1":
            inflated_serializations += 1
            return serialized + (" " * 10_000)
        return serialized

    monkeypatch.setattr(plugin, "_atomic_write_json", track_advisory_write)
    monkeypatch.setattr(plugin.json, "dumps", inflate_advisory_json)
    plugin.on_pre_api_request(
        **_base(), api_call_count=2, budget_used=2, budget_max=10
    )

    assert advisory_write_attempts == 1
    assert inflated_serializations == 1
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".*.tmp"))


def test_atomic_replace_failure_cleans_temp_and_preserves_existing(profile_home, monkeypatch):
    plugin = _load_plugin()
    path = profile_home / "atomic" / "result.json"
    plugin._atomic_write_json(path, {"version": 1})
    original = path.read_bytes()

    monkeypatch.setattr(
        plugin.os,
        "replace",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("replace failed")),
    )
    with pytest.raises(OSError):
        plugin._atomic_write_json(path, {"version": 2})

    assert path.read_bytes() == original
    assert not list(path.parent.glob(".*.tmp"))


def test_malformed_advisory_config_fails_closed_and_cap_is_bounded(profile_home):
    _write_config(
        profile_home,
        advisory_enabled="true",
        max_advisory_file_bytes=99_999_999,
    )
    plugin = _load_plugin()

    settings = plugin._load_settings(profile_home)

    assert settings.advisory_enabled is False
    assert settings.max_advisory_file_bytes == 1_000_000
    plugin.on_pre_llm_call(**_base(), user_message="disabled")
    assert _advisory_paths(profile_home) == []
