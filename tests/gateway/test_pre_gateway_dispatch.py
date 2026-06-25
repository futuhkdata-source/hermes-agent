"""Tests for the pre_gateway_dispatch plugin hook.

The hook allows plugins to intercept incoming messages before auth and
agent dispatch. It runs in _handle_message and acts on returned action
dicts: {"action": "skip"|"rewrite"|"allow"|"route"}.
"""

import dataclasses
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, SendResult
from gateway.session import SessionSource


def _clear_auth_env(monkeypatch) -> None:
    for key in (
        "TELEGRAM_ALLOWED_USERS",
        "WHATSAPP_ALLOWED_USERS",
        "GATEWAY_ALLOWED_USERS",
        "TELEGRAM_ALLOW_ALL_USERS",
        "WHATSAPP_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "FEISHU_ALLOWED_USERS",
        "FEISHU_ALLOW_ALL_USERS",
    ):
        monkeypatch.delenv(key, raising=False)


def _make_event(text: str = "hello", platform: Platform = Platform.WHATSAPP) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_id="m1",
        source=SessionSource(
            platform=platform,
            user_id="15551234567@s.whatsapp.net",
            chat_id="15551234567@s.whatsapp.net",
            user_name="tester",
            chat_type="dm",
        ),
    )


def _make_department_group_event() -> MessageEvent:
    return MessageEvent(
        text="route this",
        message_id="msg-root",
        source=SessionSource(
            platform=Platform.FEISHU,
            chat_id="dept-chat",
            chat_name="Dept Chat",
            chat_type="group",
            user_id="user-a",
            user_name="tester",
            thread_id="platform-thread",
            message_id="msg-root",
        ),
    )


def _make_runner(platform: Platform):
    from gateway.run import GatewayRunner

    config = GatewayConfig(
        platforms={platform: PlatformConfig(enabled=True)},
    )
    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False
    runner.pairing_store._is_rate_limited.return_value = False
    runner.session_store = MagicMock()
    runner._running_agents = {}
    runner._update_prompt_pending = {}
    return runner, adapter


class _RoutingTestAdapter(BasePlatformAdapter):
    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="sent-1")

    async def get_chat_info(self, chat_id):
        return {}


@pytest.mark.asyncio
async def test_hook_skip_short_circuits_dispatch(monkeypatch):
    """A plugin returning {'action': 'skip'} drops the message before auth."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "skip", "reason": "plugin-handled"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)

    result = await runner._handle_message(_make_event("hi"))

    assert result is None
    adapter.send.assert_not_awaited()
    runner.pairing_store.generate_code.assert_not_called()


@pytest.mark.asyncio
async def test_hook_rewrite_replaces_event_text(monkeypatch):
    """A plugin returning {'action': 'rewrite', 'text': ...} mutates event.text."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    seen_text = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "rewrite", "text": "REWRITTEN"}]
        return []

    async def _capture(event, source, _quick_key, _run_generation):
        seen_text["value"] = event.text
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    await runner._handle_message(_make_event("original"))

    assert seen_text.get("value") == "REWRITTEN"


@pytest.mark.asyncio
async def test_hook_allow_falls_through_to_auth(monkeypatch):
    """A plugin returning {'action': 'allow'} continues to normal dispatch."""
    _clear_auth_env(monkeypatch)
    # No allowed users set → auth fails → pairing flow triggers.
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "allow"}]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store.generate_code.return_value = "12345"

    result = await runner._handle_message(_make_event("hi"))

    # auth chain ran → pairing code was generated
    assert result is None
    runner.pairing_store.generate_code.assert_called_once()


@pytest.mark.asyncio
async def test_hook_exception_does_not_break_dispatch(monkeypatch):
    """A raising plugin hook does not break the gateway."""
    _clear_auth_env(monkeypatch)
    monkeypatch.delenv("WHATSAPP_ALLOWED_USERS", raising=False)

    def _fake_hook(name, **kwargs):
        raise RuntimeError("plugin blew up")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner.pairing_store.generate_code.return_value = None

    # Should not raise; falls through to auth chain.
    result = await runner._handle_message(_make_event("hi"))
    assert result is None


@pytest.mark.asyncio
async def test_internal_events_bypass_hook(monkeypatch):
    """Internal events (event.internal=True) skip the plugin hook entirely."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")

    called = {"count": 0}

    def _fake_hook(name, **kwargs):
        called["count"] += 1
        return [{"action": "skip"}]

    async def _capture(event, source, _quick_key, _run_generation):
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.WHATSAPP)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    event = _make_event("hi")
    event.internal = True

    # Even though the hook would say skip, internal events bypass it.
    await runner._handle_message(event)
    assert called["count"] == 0


@pytest.mark.asyncio
async def test_hook_allow_after_source_rewrite_routes_normal_dispatch_to_profile(monkeypatch):
    """A hook may replace event.source, return allow, and continue native dispatch."""
    _clear_auth_env(monkeypatch)
    monkeypatch.setenv("FEISHU_ALLOWED_USERS", "*")

    seen = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            event = kwargs["event"]
            event.source = dataclasses.replace(
                event.source,
                profile="purchase-agent",
                thread_id=None,
                session_anchor_id="msg-root",
            )
            return [{"action": "allow"}]
        return []

    async def _capture(event, source, _quick_key, run_generation):
        seen.update(
            event_profile=event.source.profile,
            source_profile=source.profile,
            event_anchor=event.source.session_anchor_id,
            source_anchor=source.session_anchor_id,
            source_thread=source.thread_id,
            quick_key=_quick_key,
        )
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.FEISHU)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    result = await runner._handle_message(_make_department_group_event())

    assert result == "ok"
    assert seen == {
        "event_profile": "purchase-agent",
        "source_profile": "purchase-agent",
        "event_anchor": "msg-root",
        "source_anchor": "msg-root",
        "source_thread": None,
        "quick_key": "agent:purchase-agent:feishu:group:dept-chat:msg-root",
    }


@pytest.mark.asyncio
async def test_hook_route_action_uses_native_profile_dispatch_without_subprocess(monkeypatch):
    """A route action stamps source.profile/session lane and continues normal dispatch."""
    _clear_auth_env(monkeypatch)

    subprocess_calls = []

    def _forbid_subprocess_run(*args, **kwargs):
        subprocess_calls.append(("subprocess.run", args, kwargs))
        raise AssertionError("native profile route must not use legacy subprocess.run")

    async def _forbid_create_subprocess_exec(*args, **kwargs):
        subprocess_calls.append(("asyncio.create_subprocess_exec", args, kwargs))
        raise AssertionError("native profile route must not spawn a subprocess")

    monkeypatch.setattr("subprocess.run", _forbid_subprocess_run)
    monkeypatch.setattr("asyncio.create_subprocess_exec", _forbid_create_subprocess_exec)

    seen = {}

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [
                {
                    "action": "route",
                    "profile": "purchase-agent",
                    "session_anchor_id": "msg-root",
                    "thread_id": None,
                    "skip_auth": True,
                }
            ]
        return []

    async def _capture(event, source, _quick_key, run_generation):
        seen.update(
            event_profile=event.source.profile,
            source_profile=source.profile,
            event_anchor=event.source.session_anchor_id,
            source_anchor=source.session_anchor_id,
            source_thread=source.thread_id,
            quick_key=_quick_key,
        )
        return "ok"

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.FEISHU)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    result = await runner._handle_message(_make_department_group_event())

    assert result == "ok"
    assert subprocess_calls == []
    assert seen == {
        "event_profile": "purchase-agent",
        "source_profile": "purchase-agent",
        "event_anchor": "msg-root",
        "source_anchor": "msg-root",
        "source_thread": None,
        "quick_key": "agent:purchase-agent:feishu:group:dept-chat:msg-root",
    }


@pytest.mark.asyncio
async def test_adapter_applies_route_before_session_guard(monkeypatch):
    """Platform adapters must session-key routed groups under the routed profile."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [
                {
                    "action": "route",
                    "profile": "purchase-agent",
                    "session_anchor_id": "msg-root",
                    "thread_id": None,
                    "skip_auth": True,
                }
            ]
        return []

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.FEISHU)
    adapter = _RoutingTestAdapter(PlatformConfig(enabled=True), Platform.FEISHU)
    adapter.set_message_handler(runner._handle_message)
    captured = {}

    def _capture_start(event, session_key, **_kwargs):
        captured["profile"] = event.source.profile
        captured["session_key"] = session_key
        return True

    adapter._start_session_processing = _capture_start  # noqa: SLF001

    await adapter.handle_message(_make_department_group_event())

    assert captured == {
        "profile": "purchase-agent",
        "session_key": "agent:purchase-agent:feishu:group:dept-chat:msg-root",
    }


@pytest.mark.asyncio
async def test_route_action_without_valid_profile_does_not_skip_auth(monkeypatch):
    """skip_auth is honored only after a valid routed profile is accepted."""
    _clear_auth_env(monkeypatch)

    def _fake_hook(name, **kwargs):
        if name == "pre_gateway_dispatch":
            return [{"action": "route", "skip_auth": True}]
        return []

    async def _capture(*_args, **_kwargs):
        raise AssertionError("invalid route must not reach agent dispatch")

    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", _fake_hook)

    runner, _adapter = _make_runner(Platform.FEISHU)
    runner._is_user_authorized = MagicMock(return_value=False)
    runner._handle_message_with_agent = _capture  # noqa: SLF001

    result = await runner._handle_message(_make_department_group_event())

    assert result is None
    runner._is_user_authorized.assert_called_once()


def test_load_gateway_config_honors_profile_runtime_scope(tmp_path, monkeypatch):
    """Explicit profile routing must read the routed profile's config.yaml."""
    import yaml
    import gateway.run as gateway_run
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    default_home = tmp_path / "default"
    profile_home = tmp_path / "profiles" / "purchase-agent"
    default_home.mkdir(parents=True)
    profile_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text(yaml.safe_dump({"marker": "default"}), encoding="utf-8")
    (profile_home / "config.yaml").write_text(yaml.safe_dump({"marker": "profile"}), encoding="utf-8")
    monkeypatch.setattr(gateway_run, "_hermes_home", default_home)

    token = set_hermes_home_override(profile_home)
    try:
        assert gateway_run._load_gateway_config()["marker"] == "profile"
    finally:
        reset_hermes_home_override(token)
