"""Exactly-once, metadata-only contract for the public turn wrapper."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import conversation_loop


def _agent():
    return SimpleNamespace(
        session_id="session-1",
        _current_task_id="task-1",
        _current_turn_id="turn-1",
        _api_call_count=0,
        model="test-model",
        platform="cli",
        iteration_budget=SimpleNamespace(used=2, max_total=10),
    )


@pytest.mark.parametrize(
    ("result", "expected_reason", "expected_failed", "expected_interrupted"),
    [
        (
            {
                "completed": False,
                "failed": False,
                "interrupted": False,
                "error": "SECRET_NON_RETRYABLE_PROVIDER_ERROR",
                "api_calls": 2,
                "final_response": None,
            },
            "failed",
            True,
            False,
        ),
        (
            {
                "completed": False,
                "failed": False,
                "interrupted": True,
                "api_calls": 2,
                "final_response": "SECRET_INTERRUPT_TEXT",
            },
            "interrupted",
            False,
            True,
        ),
        (
            {
                "completed": False,
                "failed": False,
                "interrupted": False,
                "partial": True,
                "api_calls": 2,
                "final_response": "SECRET_PARTIAL_TEXT",
            },
            "incomplete",
            False,
            False,
        ),
    ],
)
def test_every_result_shape_emits_one_sanitized_turn_end(
    result, expected_reason, expected_failed, expected_interrupted
):
    calls = []

    def record(name, **kwargs):
        calls.append((name, kwargs))
        return []

    with (
        patch.object(conversation_loop, "_run_conversation_impl", return_value=result),
        patch("hermes_cli.plugins.invoke_hook", side_effect=record),
    ):
        returned = conversation_loop.run_conversation(_agent(), "SECRET_USER_TEXT")

    assert returned is result
    turn_end = [kwargs for name, kwargs in calls if name == "on_turn_end"]
    assert len(turn_end) == 1
    assert turn_end[0]["turn_exit_reason"] == expected_reason
    assert turn_end[0]["failed"] is expected_failed
    assert turn_end[0]["interrupted"] is expected_interrupted
    serialized = json.dumps(turn_end[0])
    assert "SECRET" not in serialized
    assert "final_response" not in turn_end[0]


def test_unhandled_exception_emits_one_sanitized_turn_end_then_reraises():
    calls = []

    def record(name, **kwargs):
        calls.append((name, kwargs))
        return []

    with (
        patch.object(
            conversation_loop,
            "_run_conversation_impl",
            side_effect=RuntimeError("SECRET_UNHANDLED_ERROR"),
        ),
        patch("hermes_cli.plugins.invoke_hook", side_effect=record),
        pytest.raises(RuntimeError, match="SECRET_UNHANDLED_ERROR"),
    ):
        conversation_loop.run_conversation(_agent(), "SECRET_USER_TEXT")

    turn_end = [kwargs for name, kwargs in calls if name == "on_turn_end"]
    assert len(turn_end) == 1
    assert turn_end[0]["turn_exit_reason"] == "unhandled_exception"
    assert turn_end[0]["failed"] is True
    assert "SECRET" not in json.dumps(turn_end[0])
