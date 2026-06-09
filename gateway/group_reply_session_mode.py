from __future__ import annotations

import json
import logging
import os
import threading
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from utils import atomic_replace

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_DEFAULT_TTL_DAYS = 7


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _hermes_home() -> Path:
    raw = (os.getenv("HERMES_HOME") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (Path.home() / ".hermes").resolve()


def _state_path() -> Path:
    return _hermes_home() / "sessions" / "group_reply_session_anchors.json"


def _route_state_path() -> Path:
    return _hermes_home() / "sessions" / "department_route_session_map.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.debug("group reply session mode: failed to read %s", path, exc_info=True)
        return {}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".anchors_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _message_key(platform: Any, chat_type: str, chat_id: str, message_id: str) -> str:
    plat = getattr(platform, "value", platform)
    return f"{plat}:{chat_type}:{chat_id}:{message_id}"


def _route_key(profile: str, chat_id: str, anchor_id: str) -> str:
    return f"{profile}:{chat_id}:{anchor_id}"


def _purge_expired(data: Dict[str, Any], *, now: Optional[datetime] = None) -> Dict[str, Any]:
    now = now or _utc_now()
    changed = False
    for key in list(data.keys()):
        row = data.get(key)
        if not isinstance(row, dict):
            data.pop(key, None)
            changed = True
            continue
        expires_at = str(row.get("expires_at") or "").strip()
        if not expires_at:
            data.pop(key, None)
            changed = True
            continue
        try:
            expiry = datetime.fromisoformat(expires_at)
        except Exception:
            data.pop(key, None)
            changed = True
            continue
        if expiry <= now:
            data.pop(key, None)
            changed = True
    return data if changed else data


def resolve_group_reply_anchor(
    *,
    platform: Any,
    chat_type: str,
    chat_id: str,
    reply_to_message_id: Optional[str],
    ttl_days: int = _DEFAULT_TTL_DAYS,
) -> Optional[str]:
    del ttl_days  # retention is enforced on write + purge; reader just checks expiry
    if not reply_to_message_id:
        return None
    key = _message_key(platform, chat_type, str(chat_id), str(reply_to_message_id))
    path = _state_path()
    now = _utc_now()
    with _LOCK:
        data = _load_json(path)
        _purge_expired(data, now=now)
        row = data.get(key)
        if not isinstance(row, dict):
            return None
        anchor_id = str(row.get("anchor_id") or "").strip()
        return anchor_id or None


def record_group_message_anchor(
    *,
    platform: Any,
    chat_type: str,
    chat_id: str,
    message_id: Optional[str],
    anchor_id: Optional[str],
    ttl_days: int = _DEFAULT_TTL_DAYS,
) -> None:
    if not message_id or not anchor_id:
        return
    path = _state_path()
    now = _utc_now()
    expires_at = (now + timedelta(days=max(1, int(ttl_days)))).isoformat()
    key = _message_key(platform, chat_type, str(chat_id), str(message_id))
    with _LOCK:
        data = _load_json(path)
        _purge_expired(data, now=now)
        data[key] = {
            "anchor_id": str(anchor_id),
            "updated_at": now.isoformat(),
            "expires_at": expires_at,
        }
        _save_json(path, data)


def resolve_department_route_session(
    *,
    profile: str,
    chat_id: str,
    anchor_id: Optional[str],
) -> Optional[str]:
    if not profile or not chat_id or not anchor_id:
        return None
    path = _route_state_path()
    now = _utc_now()
    key = _route_key(str(profile), str(chat_id), str(anchor_id))
    with _LOCK:
        data = _load_json(path)
        _purge_expired(data, now=now)
        row = data.get(key)
        if not isinstance(row, dict):
            return None
        session_id = str(row.get("session_id") or "").strip()
        return session_id or None


def record_department_route_session(
    *,
    profile: str,
    chat_id: str,
    anchor_id: Optional[str],
    session_id: Optional[str],
    ttl_days: int = _DEFAULT_TTL_DAYS,
) -> None:
    if not profile or not chat_id or not anchor_id or not session_id:
        return
    path = _route_state_path()
    now = _utc_now()
    expires_at = (now + timedelta(days=max(1, int(ttl_days)))).isoformat()
    key = _route_key(str(profile), str(chat_id), str(anchor_id))
    with _LOCK:
        data = _load_json(path)
        _purge_expired(data, now=now)
        data[key] = {
            "session_id": str(session_id),
            "updated_at": now.isoformat(),
            "expires_at": expires_at,
        }
        _save_json(path, data)
