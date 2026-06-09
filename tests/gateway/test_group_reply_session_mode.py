import pytest

from gateway.config import Platform, PlatformConfig
from gateway.group_reply_session_mode import (
    record_department_route_session,
    record_group_message_anchor,
    resolve_department_route_session,
    resolve_group_reply_anchor,
)
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key


class DummyFeishuAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), Platform.FEISHU)

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="bot-msg-1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


@pytest.fixture
def isolated_hermes_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    (hermes_home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _make_event(*, message_id: str, reply_to_message_id: str | None = None, user_id: str = "user-a") -> MessageEvent:
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        message_id=message_id,
        reply_to_message_id=reply_to_message_id,
        source=SessionSource(
            platform=Platform.FEISHU,
            chat_id="oc_test_group",
            chat_name="Test Group",
            chat_type="group",
            user_id=user_id,
            user_name=user_id,
            thread_id="omt_platform_thread_should_be_ignored",
            message_id=message_id,
        ),
    )


def test_build_session_key_ignores_user_when_session_anchor_is_present():
    source_a = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_test_group",
        chat_type="group",
        user_id="user-a",
        session_anchor_id="msg-root-1",
    )
    source_b = SessionSource(
        platform=Platform.FEISHU,
        chat_id="oc_test_group",
        chat_type="group",
        user_id="user-b",
        session_anchor_id="msg-root-1",
    )

    assert build_session_key(source_a) == "agent:main:feishu:group:oc_test_group:msg-root-1"
    assert build_session_key(source_a) == build_session_key(source_b)


def test_non_reply_group_message_starts_new_anchor_session(isolated_hermes_home):
    adapter = DummyFeishuAdapter()

    event = _make_event(message_id="msg-1")
    normalized = adapter._apply_group_reply_session_mode(event)

    assert normalized.source.thread_id is None
    assert normalized.source.session_anchor_id == "msg-1"
    assert build_session_key(normalized.source) == "agent:main:feishu:group:oc_test_group:msg-1"
    assert resolve_group_reply_anchor(
        platform=Platform.FEISHU,
        chat_type="group",
        chat_id="oc_test_group",
        reply_to_message_id="msg-1",
    ) == "msg-1"


def test_reply_message_reuses_original_anchor_session(isolated_hermes_home):
    adapter = DummyFeishuAdapter()

    root = adapter._apply_group_reply_session_mode(_make_event(message_id="msg-root", user_id="user-a"))
    reply = adapter._apply_group_reply_session_mode(
        _make_event(message_id="msg-reply", reply_to_message_id="msg-root", user_id="user-b")
    )

    assert root.source.session_anchor_id == "msg-root"
    assert reply.source.thread_id is None
    assert reply.source.session_anchor_id == "msg-root"
    assert build_session_key(reply.source) == build_session_key(root.source)


def test_department_route_session_map_round_trip(isolated_hermes_home):
    record_group_message_anchor(
        platform=Platform.FEISHU,
        chat_type="group",
        chat_id="oc_test_group",
        message_id="msg-root",
        anchor_id="msg-root",
    )
    record_department_route_session(
        profile="purchase-agent",
        chat_id="oc_test_group",
        anchor_id="msg-root",
        session_id="session-123",
    )

    assert resolve_group_reply_anchor(
        platform=Platform.FEISHU,
        chat_type="group",
        chat_id="oc_test_group",
        reply_to_message_id="msg-root",
    ) == "msg-root"
    assert resolve_department_route_session(
        profile="purchase-agent",
        chat_id="oc_test_group",
        anchor_id="msg-root",
    ) == "session-123"
