"""Contract tests for the opt-in execution-shadow observability plugin."""
from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "execution-shadow"
PLUGIN_INIT = PLUGIN_DIR / "__init__.py"
PLUGIN_KEY = "observability/execution-shadow"


def _load_plugin():
    name = "execution_shadow_plugin_under_test"
    sys.modules.pop(name, None)
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


def _write_config(home: Path, *, enabled: bool = True) -> None:
    home.mkdir(parents=True, exist_ok=True)
    data = {
        "plugins": {"enabled": [PLUGIN_KEY]},
        "execution_shadow": {
            "enabled": enabled,
            "mode": "shadow",
            "scope_freeze_ratio": 0.60,
            "closure_ratio": 0.75,
            "finalization_ratio": 0.85,
            "max_reviewers": 1,
            "max_remediations": 1,
        },
    }
    (home / "config.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )


@pytest.fixture
def enabled_home(tmp_path):
    home = tmp_path / "default-home"
    _write_config(home, enabled=True)
    token = set_hermes_home_override(home)
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def _snapshot(home: Path) -> dict:
    paths = sorted((home / "telemetry" / "execution-shadow" / "turns").glob("*.json"))
    assert paths, "execution-shadow did not write a turn snapshot"
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def _all_telemetry(home: Path) -> str:
    root = home / "telemetry" / "execution-shadow"
    return "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


class TestManifestAndDiscovery:
    def test_plugin_layout_and_manifest_contract(self):
        assert PLUGIN_INIT.is_file()
        manifest = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text(encoding="utf-8"))
        assert manifest["name"] == "execution-shadow"
        assert manifest["version"] == "0.3.0"
        assert manifest["kind"] == "standalone"
        assert set(manifest["hooks"]) == {
            "pre_llm_call",
            "pre_api_request",
            "post_api_request",
            "api_request_error",
            "pre_tool_call",
            "post_tool_call",
            "post_llm_call",
            "on_turn_end",
            "subagent_start",
            "subagent_stop",
        }

    def test_plugin_is_opt_in_and_loads_only_when_enabled(self, tmp_path, monkeypatch):
        from hermes_cli import plugins as plugins_mod

        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = plugins_mod.PluginManager()
        manager.discover_and_load()
        discovered = manager._plugins[PLUGIN_KEY]
        assert discovered.enabled is False
        assert "not enabled" in (discovered.error or "").lower()

        _write_config(home, enabled=True)
        manager = plugins_mod.PluginManager()
        manager.discover_and_load()
        loaded = manager._plugins[PLUGIN_KEY]
        assert loaded.enabled is True
        assert set(loaded.hooks_registered) == {
            "pre_llm_call",
            "pre_api_request",
            "post_api_request",
            "api_request_error",
            "pre_tool_call",
            "post_tool_call",
            "post_llm_call",
            "on_turn_end",
            "subagent_start",
            "subagent_stop",
        }


def test_default_config_is_disabled_shadow_mode():
    from hermes_cli.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG["execution_shadow"]
    assert cfg == {
        "enabled": False,
        "mode": "shadow",
        "scope_freeze_ratio": 0.60,
        "closure_ratio": 0.75,
        "finalization_ratio": 0.85,
        "max_reviewers": 1,
        "max_remediations": 1,
        "max_event_file_bytes": 5_000_000,
        "advisory_enabled": False,
        "max_advisory_file_bytes": 65_536,
        "advisory_transition_log_enabled": False,
        "max_advisory_transition_record_bytes": 65_536,
        "max_advisory_transition_total_bytes": 8_000_000,
        "max_advisory_transition_records": 1_024,
    }


def test_runtime_gate_is_profile_scoped(enabled_home, tmp_path):
    plugin = _load_plugin()
    plugin.on_pre_llm_call(
        session_id="default-session",
        task_id="task-1",
        turn_id="turn-1",
        user_message="default request",
        platform="feishu",
    )
    assert (enabled_home / "telemetry" / "execution-shadow").is_dir()

    disabled_home = tmp_path / "department-home"
    _write_config(disabled_home, enabled=False)
    token = set_hermes_home_override(disabled_home)
    try:
        plugin.on_pre_llm_call(
            session_id="department-session",
            task_id="task-2",
            turn_id="turn-2",
            user_message="department request",
            platform="feishu",
        )
    finally:
        reset_hermes_home_override(token)

    assert not (disabled_home / "telemetry" / "execution-shadow").exists()


def test_phase_and_budget_gates_are_monotonic(enabled_home):
    plugin = _load_plugin()
    base = {"session_id": "s1", "task_id": "t1", "turn_id": "turn-1"}

    plugin.on_pre_llm_call(**base, user_message="repair it", platform="feishu")
    assert _snapshot(enabled_home)["phase"] == "discovery"

    plugin.on_pre_tool_call(**base, tool_name="patch", args={"path": "a.py"})
    assert _snapshot(enabled_home)["phase"] == "implementation"

    plugin.on_pre_tool_call(
        **base,
        tool_name="terminal",
        args={"command": "python -m pytest tests/unit -q"},
    )
    assert _snapshot(enabled_home)["phase"] == "verification"

    plugin.on_pre_api_request(**base, api_call_count=6, budget_used=6, budget_max=10)
    snap = _snapshot(enabled_home)
    assert snap["phase"] == "verification"
    assert snap["budget"] == {
        "used": 6,
        "max": 10,
        "ratio": 0.6,
        "scope_frozen": True,
        "closure_reserved": False,
        "finalization_only": False,
    }

    plugin.on_pre_api_request(**base, api_call_count=8, budget_used=8, budget_max=10)
    snap = _snapshot(enabled_home)
    assert snap["phase"] == "closure"
    assert snap["budget"]["closure_reserved"] is True

    plugin.on_pre_api_request(**base, api_call_count=9, budget_used=9, budget_max=10)
    snap = _snapshot(enabled_home)
    assert snap["phase"] == "finalization"
    assert snap["budget"]["finalization_only"] is True

    plugin.on_post_llm_call(
        **base,
        assistant_response="GO",
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
        api_call_count=9,
        budget_used=9,
        budget_max=10,
    )
    snap = _snapshot(enabled_home)
    assert snap["phase"] == "terminal"
    assert snap["phase_history"] == [
        "discovery",
        "implementation",
        "verification",
        "closure",
        "finalization",
        "terminal",
    ]


def test_helper_review_and_remediation_limits_are_observed(enabled_home):
    plugin = _load_plugin()
    base = {"session_id": "s2", "task_id": "t2", "turn_id": "turn-2"}
    plugin.on_pre_llm_call(**base, user_message="implement", platform="feishu")
    plugin.on_pre_api_request(**base, api_call_count=6, budget_used=6, budget_max=10)

    for idx in range(2):
        plugin.on_subagent_start(
            parent_session_id="s2",
            parent_turn_id="turn-2",
            child_session_id=f"review-{idx}",
            child_subagent_id=f"r-{idx}",
            child_role="leaf",
            child_goal="Independent read-only code reviewer; return GO or NO-GO",
        )
    for idx in range(2):
        plugin.on_subagent_start(
            parent_session_id="s2",
            parent_turn_id="turn-2",
            child_session_id=f"fix-{idx}",
            child_subagent_id=f"f-{idx}",
            child_role="leaf",
            child_goal="Fix only the reported blocking findings in one remediation pass",
        )

    plugin.on_pre_api_request(**base, api_call_count=9, budget_used=9, budget_max=10)
    plugin.on_subagent_start(
        parent_session_id="s2",
        parent_turn_id="turn-2",
        child_session_id="late-helper",
        child_subagent_id="late",
        child_role="leaf",
        child_goal="Compare another optional architecture",
    )

    snap = _snapshot(enabled_home)
    assert snap["counts"]["helpers_started"] == 5
    assert snap["counts"]["reviewers_started"] == 2
    assert snap["counts"]["remediations_started"] == 2
    assert set(snap["warnings"]) >= {
        "helper_started_after_scope_freeze",
        "helper_started_during_finalization",
        "reviewer_limit_exceeded",
        "remediation_limit_exceeded",
    }


def test_actual_precommit_review_wording_is_classified_as_review():
    plugin = _load_plugin()
    goal = (
        "Perform an independent, bounded, read-only pre-commit review of the "
        "staged P1-A implementation and return GO or NO-GO with findings."
    )
    assert plugin._classify_helper(goal) == "review"
    assert plugin._classify_helper("Review the staged diff and return GO or NO-GO") == "review"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("GO", "GO"),
        ("Verdict: NO-GO", "NO-GO"),
        ("Here you go.", None),
        ("The work is done and tests pass.", None),
        ("Earlier verdict: NO-GO\nIssues fixed.\nGO", "GO"),
        ("GO\nAdditional explanation after the verdict.", None),
    ],
)
def test_terminal_verdict_requires_explicit_final_line(text, expected):
    plugin = _load_plugin()
    assert plugin._verdict(text) == expected


def test_completion_evidence_is_derived_without_storing_payloads(enabled_home):
    plugin = _load_plugin()
    base = {"session_id": "s3", "task_id": "t3", "turn_id": "turn-3"}
    plugin.on_pre_llm_call(**base, user_message="SECRET_USER_TEXT", platform="feishu")

    calls = [
        ("write_file", {"path": "artifact.md", "content": "SECRET_TOOL_ARG"}, '{"success": true}'),
        ("terminal", {"command": "python -m pytest -q SECRET_TOOL_ARG"}, '{"exit_code": 0, "output": "SECRET_RESULT"}'),
        ("terminal", {"command": "git commit -m verified"}, '{"exit_code": 0, "output": "committed"}'),
        ("terminal", {"command": "hermes send --to feishu status"}, '{"exit_code": 0, "output": "sent"}'),
        ("terminal", {"command": "cp config.yaml backups/config.yaml"}, '{"exit_code": 0, "output": "copied"}'),
    ]
    for idx, (tool, args, result) in enumerate(calls):
        plugin.on_pre_tool_call(
            **base, tool_name=tool, args=args, tool_call_id=f"call-{idx}"
        )
        plugin.on_post_tool_call(
            **base,
            tool_name=tool,
            args=args,
            result=result,
            tool_call_id=f"call-{idx}",
            status="ok",
        )

    plugin.on_post_api_request(
        **base,
        api_call_count=5,
        budget_used=5,
        budget_max=10,
        assistant_tool_call_count=0,
        assistant_content_chars=20,
        finish_reason="stop",
    )
    plugin.on_post_llm_call(
        **base,
        assistant_response="SECRET_ASSISTANT_TEXT\nGO",
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
        api_call_count=5,
        budget_used=5,
        budget_max=10,
    )

    snap = _snapshot(enabled_home)
    assert snap["evidence"] == {
        "artifact_created": True,
        "tests_passed": True,
        "commit_created": True,
        "delivery_verified": True,
        "rollback_ready": True,
        "final_response_present": True,
        "terminal_verdict": "GO",
        "durable_execution_proven": False,
    }
    assert snap["terminal"] == {
        "completed": True,
        "failed": False,
        "interrupted": False,
        "exit_reason": "text_response",
    }

    telemetry = _all_telemetry(enabled_home)
    for secret in (
        "SECRET_USER_TEXT",
        "SECRET_TOOL_ARG",
        "SECRET_RESULT",
        "SECRET_ASSISTANT_TEXT",
    ):
        assert secret not in telemetry
    for raw_field in ("user_message", "assistant_response", "child_goal", '"args"', '"result"'):
        assert raw_field not in telemetry

    if os.name != "nt":
        root = enabled_home / "telemetry" / "execution-shadow"
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
        assert stat.S_IMODE((root / "turns").stat().st_mode) == 0o700
        assert all(
            stat.S_IMODE(path.stat().st_mode) == 0o600
            for path in root.rglob("*")
            if path.is_file()
        )


def test_passive_durable_claim_is_flagged(enabled_home):
    plugin = _load_plugin()
    base = {"session_id": "s4", "task_id": "t4", "turn_id": "turn-4"}
    plugin.on_pre_llm_call(**base, user_message="batch", platform="feishu")
    plugin.on_post_llm_call(
        **base,
        assistant_response="The durable task is queued and running.",
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
        api_call_count=1,
        budget_used=1,
        budget_max=10,
    )
    assert "passive_durable_claim_without_execution_proof" in _snapshot(enabled_home)["warnings"]


def test_api_error_and_abnormal_turn_end_are_observed(enabled_home):
    plugin = _load_plugin()
    base = {"session_id": "s-error", "task_id": "t-error", "turn_id": "turn-error"}
    plugin.on_pre_llm_call(**base, user_message="fail safely", platform="feishu")
    plugin.on_api_request_error(
        **base,
        api_call_count=4,
        budget_used=4,
        budget_max=10,
        error={"type": "TimeoutError", "message": "SECRET_PROVIDER_ERROR"},
        retryable=False,
        reason="timeout",
    )
    plugin.on_turn_end(
        **base,
        completed=False,
        failed=True,
        interrupted=False,
        turn_exit_reason="error_near_max_iterations(SECRET_PROVIDER_ERROR)",
        api_call_count=4,
        budget_used=4,
        budget_max=10,
        final_response_present=False,
    )

    snap = _snapshot(enabled_home)
    assert snap["phase"] == "terminal"
    assert snap["counts"]["api_errors"] == 1
    assert snap["terminal"] == {
        "completed": False,
        "failed": True,
        "interrupted": False,
        "exit_reason": "error_near_max_iterations",
    }
    assert snap["evidence"]["final_response_present"] is False
    assert "SECRET_PROVIDER_ERROR" not in _all_telemetry(enabled_home)


def test_hook_failures_are_fail_open(enabled_home, monkeypatch):
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_persist_state", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")))

    plugin.on_pre_llm_call(
        session_id="s5",
        task_id="t5",
        turn_id="turn-5",
        user_message="hello",
        platform="feishu",
    )
    plugin.on_pre_api_request(
        session_id="s5",
        task_id="t5",
        turn_id="turn-5",
        api_call_count=1,
        budget_used=1,
        budget_max=10,
    )
    plugin.on_post_llm_call(
        session_id="s5",
        task_id="t5",
        turn_id="turn-5",
        assistant_response="done",
        completed=True,
        failed=False,
        interrupted=False,
        turn_exit_reason="text_response(stop)",
        api_call_count=1,
        budget_used=1,
        budget_max=10,
    )


def test_concurrent_turns_keep_separate_snapshots(enabled_home):
    plugin = _load_plugin()
    for turn in ("turn-a", "turn-b"):
        plugin.on_pre_llm_call(
            session_id="shared-session",
            task_id=f"task-{turn}",
            turn_id=turn,
            user_message="same session",
            platform="feishu",
        )
    paths = sorted((enabled_home / "telemetry" / "execution-shadow" / "turns").glob("*.json"))
    assert len(paths) == 2
    assert {json.loads(p.read_text())["turn_id"] for p in paths} == {"turn-a", "turn-b"}
