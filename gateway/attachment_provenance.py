from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from hermes_constants import get_default_hermes_root


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_PROFILE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_INBOUND_PDF_RE = re.compile(r"^doc_[0-9a-fA-F]{12}_.+\.pdf$")
_MANIFEST_RE = re.compile(r"^message-[0-9a-f]{24}\.json$")
_MAX_MANIFEST_BYTES = 1 * 1024 * 1024
_MAX_ATTACHMENTS = 16


class AttachmentProvenanceError(RuntimeError):
    pass


def _root(value: str | Path | None) -> Path:
    return Path(value) if value is not None else get_default_hermes_root()


def _validate_session_id(session_id: str) -> str:
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        raise AttachmentProvenanceError("runtime session_id failed validation")
    return session_id


def _validate_profile_name(profile_name: str) -> str:
    if not isinstance(profile_name, str) or not _PROFILE_NAME_RE.fullmatch(profile_name):
        raise AttachmentProvenanceError("runtime profile failed validation")
    return profile_name


def _platform_name(event: Any) -> str:
    platform = getattr(getattr(event, "source", None), "platform", None)
    return str(getattr(platform, "value", platform) or "").lower()


def _profile_name(event: Any) -> str:
    profile = str(getattr(getattr(event, "source", None), "profile", None) or "default")
    return _validate_profile_name(profile)


def _hash_pdf(path: Path) -> tuple[str, int]:
    try:
        before = path.stat()
        if not path.is_file() or path.is_symlink() or before.st_nlink != 1:
            raise AttachmentProvenanceError("attachment link aliases are not allowed")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            header = handle.read(8)
            if not header.startswith(b"%PDF-"):
                raise AttachmentProvenanceError("attachment is not a real PDF")
            digest.update(header)
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except AttachmentProvenanceError:
        raise
    except OSError:
        raise AttachmentProvenanceError("attachment could not be read safely") from None
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise AttachmentProvenanceError("attachment changed during validation")
    return digest.hexdigest(), int(after.st_size)


def _validated_inbound_pdf(raw_path: str, hermes_root: Path) -> tuple[Path, str, int]:
    documents_raw = hermes_root / "cache" / "documents"
    try:
        if documents_raw.is_symlink():
            raise AttachmentProvenanceError("inbound document cache must not be a link")
        documents = documents_raw.resolve(strict=True)
        candidate_raw = Path(raw_path)
        if candidate_raw.is_symlink():
            raise AttachmentProvenanceError("attachment link aliases are not allowed")
        candidate = candidate_raw.resolve(strict=True)
    except AttachmentProvenanceError:
        raise
    except (OSError, RuntimeError):
        raise AttachmentProvenanceError("inbound document cache path is unavailable") from None
    if (
        candidate.parent != documents
        or candidate.suffix.lower() != ".pdf"
        or not _INBOUND_PDF_RE.fullmatch(candidate.name)
    ):
        raise AttachmentProvenanceError("attachment is outside the inbound document cache")
    digest, size = _hash_pdf(candidate)
    return candidate, digest, size


def _manifest_root(hermes_root: Path) -> Path:
    root = hermes_root / "cache" / "attachment_manifests"
    try:
        if root.is_symlink():
            raise AttachmentProvenanceError("attachment manifest root must not be a link")
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        root.chmod(0o700)
        resolved = root.resolve(strict=True)
    except AttachmentProvenanceError:
        raise
    except (OSError, RuntimeError):
        raise AttachmentProvenanceError("attachment manifest root could not be prepared") from None
    if resolved != root:
        raise AttachmentProvenanceError("attachment manifest root failed validation")
    return resolved


def _session_manifest_dir(hermes_root: Path, session_id: str, *, create: bool) -> Path:
    root = _manifest_root(hermes_root)
    directory = root / _validate_session_id(session_id)
    try:
        if directory.is_symlink():
            raise AttachmentProvenanceError("attachment manifest session directory must not be a link")
        if create:
            directory.mkdir(parents=False, exist_ok=True, mode=0o700)
            directory.chmod(0o700)
        resolved = directory.resolve(strict=True)
    except AttachmentProvenanceError:
        raise
    except (OSError, RuntimeError):
        raise AttachmentProvenanceError("trusted attachment manifest is unavailable") from None
    if resolved.parent != root or resolved.name != session_id:
        raise AttachmentProvenanceError("attachment manifest session directory failed validation")
    return resolved


def _atomic_manifest_write(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_MANIFEST_BYTES:
        raise AttachmentProvenanceError("attachment manifest exceeds the size limit")
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = None
    try:
        descriptor = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        path.chmod(0o600)
    except OSError:
        raise AttachmentProvenanceError("attachment manifest could not be written atomically") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temp.unlink(missing_ok=True)


def persist_trusted_attachment_manifest(
    event: Any,
    session_id: str,
    *,
    hermes_root: str | Path | None = None,
) -> Path | None:
    """Persist trusted structured media provenance after gateway session resolution.

    Only the gateway calls this function. User-visible transcript text is never
    consulted, so a typed lookalike attachment annotation cannot authorize data.
    """
    session_id = _validate_session_id(session_id)
    if _platform_name(event) != "feishu":
        return None
    urls = list(getattr(event, "media_urls", None) or [])[:_MAX_ATTACHMENTS]
    types = list(getattr(event, "media_types", None) or [])[:_MAX_ATTACHMENTS]
    root = _root(hermes_root)
    attachments: list[dict[str, Any]] = []
    for index, raw_path in enumerate(urls):
        media_type = str(types[index] if index < len(types) else "").lower()
        if media_type and media_type not in {"application/pdf", "application/octet-stream"}:
            continue
        if not media_type and Path(str(raw_path)).suffix.lower() != ".pdf":
            continue
        path, digest, size = _validated_inbound_pdf(str(raw_path), root)
        attachments.append({
            "path": str(path),
            "sha256": digest,
            "size": size,
            "media_type": media_type or "application/pdf",
        })
    if not attachments:
        return None

    message_id = str(getattr(event, "message_id", None) or "")
    if not message_id:
        raise AttachmentProvenanceError("trusted attachment event has no message_id")
    manifest_name = f"message-{hashlib.sha256(message_id.encode()).hexdigest()[:24]}.json"
    directory = _session_manifest_dir(root, session_id, create=True)
    path = directory / manifest_name
    timestamp = getattr(event, "timestamp", None)
    try:
        created_at = float(timestamp.timestamp())
    except (AttributeError, TypeError, ValueError):
        created_at = time.time()
    payload = {
        "version": 2,
        "session_id": session_id,
        "profile": _profile_name(event),
        "platform": "feishu",
        "message_id_sha256": hashlib.sha256(message_id.encode()).hexdigest(),
        "created_at": created_at,
        "attachments": attachments,
    }
    _atomic_manifest_write(path, payload)
    return path


def _read_manifest(path: Path, session_id: str, profile_name: str) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise AttachmentProvenanceError("trusted attachment manifest link is not allowed")
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise AttachmentProvenanceError("trusted attachment manifest permissions are too broad")
        if path.stat().st_size > _MAX_MANIFEST_BYTES:
            raise AttachmentProvenanceError("trusted attachment manifest exceeds the size limit")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except AttachmentProvenanceError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise AttachmentProvenanceError("trusted attachment manifest is malformed") from None
    if (
        not isinstance(payload, dict)
        or payload.get("version") != 2
        or payload.get("session_id") != session_id
        or payload.get("profile") != profile_name
        or payload.get("platform") != "feishu"
        or not isinstance(payload.get("attachments"), list)
    ):
        raise AttachmentProvenanceError("trusted attachment manifest identity or profile failed validation")
    return payload


def resolve_trusted_pdf_attachment(
    session_id: str,
    *,
    profile_name: str,
    hermes_root: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve the latest verified PDF for one runtime session and profile."""
    session_id = _validate_session_id(session_id)
    profile_name = _validate_profile_name(profile_name)
    root = _root(hermes_root)
    directory = _session_manifest_dir(root, session_id, create=False)
    manifests = sorted(
        (
            path for path in directory.iterdir()
            if _MANIFEST_RE.fullmatch(path.name)
        ),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )[:64]
    if not manifests:
        raise AttachmentProvenanceError("no trusted attachment manifest exists for the runtime session")
    for manifest in manifests:
        payload = _read_manifest(manifest, session_id, profile_name)
        for record in reversed(payload["attachments"]):
            if not isinstance(record, dict):
                continue
            path, digest, size = _validated_inbound_pdf(str(record.get("path") or ""), root)
            if digest != record.get("sha256"):
                raise AttachmentProvenanceError("trusted attachment digest no longer matches")
            if size != record.get("size"):
                raise AttachmentProvenanceError("trusted attachment size no longer matches")
            return {
                "path": str(path),
                "sha256": digest,
                "size": size,
                "media_type": str(record.get("media_type") or "application/pdf"),
                "manifest": str(manifest),
            }
    raise AttachmentProvenanceError("trusted attachment manifest contains no valid PDF")
