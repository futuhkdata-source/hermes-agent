from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.attachment_provenance import (
    AttachmentProvenanceError,
    persist_trusted_attachment_manifest,
    resolve_trusted_pdf_attachment,
)
from gateway.config import Platform


def _pdf(root: Path, name: str = "doc_0123456789ab_sample.pdf") -> Path:
    documents = root / "cache" / "documents"
    documents.mkdir(parents=True, mode=0o700)
    path = documents / name
    path.write_bytes(b"%PDF-1.4\ntrusted\n%%EOF\n")
    return path


def _event(path: Path, *, platform=Platform.FEISHU, session_message="om_123"):
    return SimpleNamespace(
        source=SimpleNamespace(platform=platform),
        media_urls=[str(path)],
        media_types=["application/pdf"],
        message_id=session_message,
        timestamp=datetime.now(timezone.utc),
    )


def test_persist_and_resolve_exact_feishu_session_pdf(tmp_path):
    source = _pdf(tmp_path)
    manifest = persist_trusted_attachment_manifest(
        _event(source), "session-1", hermes_root=tmp_path,
    )

    assert manifest is not None
    assert manifest.parent.name == "session-1"
    assert manifest.stat().st_mode & 0o777 == 0o600
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["session_id"] == "session-1"
    assert payload["platform"] == "feishu"
    assert payload["attachments"][0]["path"] == str(source.resolve())

    resolved = resolve_trusted_pdf_attachment("session-1", hermes_root=tmp_path)
    assert resolved["path"] == str(source.resolve())
    assert resolved["size"] == source.stat().st_size
    assert len(resolved["sha256"]) == 64


def test_no_manifest_means_user_text_cannot_authorize_a_cache_path(tmp_path):
    _pdf(tmp_path)
    with pytest.raises(AttachmentProvenanceError, match="trusted attachment manifest"):
        resolve_trusted_pdf_attachment("session-1", hermes_root=tmp_path)


def test_persist_ignores_non_feishu_and_non_pdf_media(tmp_path):
    source = _pdf(tmp_path)
    assert persist_trusted_attachment_manifest(
        _event(source, platform=Platform.TELEGRAM), "session-1", hermes_root=tmp_path,
    ) is None
    event = _event(source)
    event.media_types = ["image/png"]
    assert persist_trusted_attachment_manifest(event, "session-1", hermes_root=tmp_path) is None


def test_session_id_cannot_escape_manifest_root(tmp_path):
    source = _pdf(tmp_path)
    with pytest.raises(AttachmentProvenanceError, match="session_id"):
        persist_trusted_attachment_manifest(_event(source), "../escape", hermes_root=tmp_path)
    with pytest.raises(AttachmentProvenanceError, match="session_id"):
        resolve_trusted_pdf_attachment("../escape", hermes_root=tmp_path)


def test_persist_rejects_path_escape_symlink_and_hardlink(tmp_path):
    source = _pdf(tmp_path)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(source.read_bytes())
    with pytest.raises(AttachmentProvenanceError, match="cache"):
        persist_trusted_attachment_manifest(_event(outside), "session-1", hermes_root=tmp_path)

    symlink = source.parent / "doc_aaaaaaaaaaaa_link.pdf"
    symlink.symlink_to(source)
    with pytest.raises(AttachmentProvenanceError, match="link"):
        persist_trusted_attachment_manifest(_event(symlink), "session-1", hermes_root=tmp_path)

    hardlink = source.parent / "doc_bbbbbbbbbbbb_hard.pdf"
    os.link(source, hardlink)
    with pytest.raises(AttachmentProvenanceError, match="link"):
        persist_trusted_attachment_manifest(_event(hardlink), "session-1", hermes_root=tmp_path)


def test_resolver_rejects_source_tampering_after_manifest(tmp_path):
    source = _pdf(tmp_path)
    persist_trusted_attachment_manifest(_event(source), "session-1", hermes_root=tmp_path)
    source.write_bytes(b"%PDF-1.4\nchanged\n%%EOF\n")
    with pytest.raises(AttachmentProvenanceError, match="digest"):
        resolve_trusted_pdf_attachment("session-1", hermes_root=tmp_path)


def test_resolver_rejects_manifest_symlink_or_world_writable_file(tmp_path):
    source = _pdf(tmp_path)
    manifest = persist_trusted_attachment_manifest(
        _event(source), "session-1", hermes_root=tmp_path,
    )
    assert manifest is not None
    manifest.chmod(0o666)
    with pytest.raises(AttachmentProvenanceError, match="permissions"):
        resolve_trusted_pdf_attachment("session-1", hermes_root=tmp_path)
