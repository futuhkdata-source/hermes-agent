from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def profile_scope():
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    token = None

    def activate(path: Path) -> None:
        nonlocal token
        if token is not None:
            reset_hermes_home_override(token)
        path.mkdir(parents=True, exist_ok=True)
        token = set_hermes_home_override(str(path))

    yield activate

    if token is not None:
        reset_hermes_home_override(token)


def _artifact(engine, source: Path):
    return engine.SourceArtifact(
        path=source.resolve(),
        digest=hashlib.sha256(source.read_bytes()).hexdigest(),
        size=source.stat().st_size,
        filename="sample.pdf",
    )


def test_schema_is_bounded_and_has_no_model_supplied_source_path():
    from document_processing.schema import DOCUMENT_EXTRACT_SCHEMA

    parameters = DOCUMENT_EXTRACT_SCHEMA["parameters"]
    properties = parameters["properties"]
    assert "source_path" not in properties
    assert "output_path" not in properties
    assert parameters["additionalProperties"] is False
    assert set(properties["action"]["enum"]) == {
        "metadata",
        "text_layer",
        "render_pages",
        "detect_regions",
        "tile_image",
        "crop_image",
        "ocr_image",
        "cleanup",
    }


def test_work_root_follows_each_active_profile(profile_scope, tmp_path):
    from document_processing import engine

    profile_a = tmp_path / "profile-a"
    profile_scope(profile_a)
    root_a = engine._prepare_work_root()

    profile_b = tmp_path / "profile-b"
    profile_scope(profile_b)
    root_b = engine._prepare_work_root()

    assert root_a == (profile_a / "cache" / "document-extract").resolve()
    assert root_b == (profile_b / "cache" / "document-extract").resolve()
    assert root_a != root_b
    assert root_a.stat().st_mode & 0o777 == 0o700
    assert root_b.stat().st_mode & 0o777 == 0o700


def test_work_root_rejects_linked_profile_cache(profile_scope, tmp_path):
    from document_processing import engine

    profile = tmp_path / "profile"
    profile.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (profile / "cache").symlink_to(outside, target_is_directory=True)
    profile_scope(profile)

    with pytest.raises(engine.DocumentExtractError, match="Linked document work directories"):
        engine._prepare_work_root()


def test_requirements_do_not_depend_on_chatbot_profile(monkeypatch, profile_scope, tmp_path):
    from document_processing import engine

    existing = Path(engine.__file__).resolve()
    for name in (
        "PDFINFO_BIN",
        "PDFTOTEXT_BIN",
        "PDFTOPPM_BIN",
        "TESSERACT_BIN",
        "PYTHON_BIN",
        "PRLIMIT_BIN",
    ):
        monkeypatch.setattr(engine, name, str(existing))
    monkeypatch.setattr(engine, "PADDLE_RUNNER", existing)
    monkeypatch.setattr(engine, "_paddle_models_present", lambda: True)
    monkeypatch.setattr(engine.importlib.util, "find_spec", lambda _name: object())

    profile_scope(tmp_path / "chatbot-agent")
    assert engine.check_requirements() is True

    profile_scope(tmp_path / "other-agent")
    assert engine.check_requirements() is True


def test_runtime_profile_name_is_derived_from_scoped_home(
    monkeypatch, profile_scope, tmp_path
):
    from document_processing import engine

    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setattr(engine, "get_default_hermes_root", lambda: root)

    profile_scope(root)
    assert engine._runtime_profile_name() == "default"

    profile_scope(root / "profiles" / "hr-agent")
    assert engine._runtime_profile_name() == "hr-agent"


def test_authorization_uses_runtime_session_manifest_not_user_path(monkeypatch, tmp_path):
    from document_processing import engine

    source = tmp_path / "doc_0123456789ab_sample.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    seen: dict[str, object] = {}

    def resolve(session_id: str, *, profile_name: str, hermes_root: Path):
        seen.update(
            session_id=session_id,
            profile_name=profile_name,
            hermes_root=Path(hermes_root),
        )
        return {
            "path": str(source),
            "sha256": digest,
            "size": source.stat().st_size,
            "media_type": "application/pdf",
        }

    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(engine, "resolve_trusted_pdf_attachment", resolve)
    monkeypatch.setattr(engine, "get_default_hermes_root", lambda: root)
    monkeypatch.setattr(engine, "_runtime_profile_name", lambda: "hr-agent")

    artifact = engine._authorize_session_pdf("session-1")

    assert artifact.path == source.resolve()
    assert artifact.digest == digest
    assert artifact.filename == "sample.pdf"
    assert seen == {
        "session_id": "session-1",
        "profile_name": "hr-agent",
        "hermes_root": root,
    }
    with pytest.raises(engine.DocumentExtractError, match="session_id"):
        engine._authorize_session_pdf("")


def test_handler_succeeds_in_non_chatbot_profile_with_trusted_artifact(
    monkeypatch, profile_scope, tmp_path
):
    from document_processing import engine

    profile_scope(tmp_path / "hr-agent")
    source = tmp_path / "doc_0123456789ab_sample.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    artifact = _artifact(engine, source)
    monkeypatch.setattr(engine, "_authorize_session_pdf", lambda session_id: artifact)
    monkeypatch.setattr(
        engine,
        "_read_pdf_metadata",
        lambda _artifact: {
            "pages": 2,
            "encrypted": False,
            "page_size": "595 x 842 pts (A4)",
            "raw": "Pages: 2\nEncrypted: no\nPage size: 595 x 842 pts (A4)\n",
        },
    )

    result = json.loads(
        engine.handle_document_extract({"action": "metadata"}, session_id="session-1")
    )

    assert result["success"] is True
    assert result["pages"] == 2
    assert result["source_id"] == artifact.digest[:16]


def test_paddle_child_environment_drops_gateway_and_provider_secrets(
    monkeypatch, tmp_path
):
    from document_processing import engine

    monkeypatch.setenv("FEISHU_APP_SECRET", "must-not-leak")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    runtime = tmp_path / "runtime"
    runtime.mkdir()

    env = engine._build_subprocess_env(
        [engine.PYTHON_BIN, str(engine.PADDLE_RUNNER), "/tmp/input.png"],
        {"HOME": str(runtime), "TMPDIR": str(runtime)},
    )

    assert "FEISHU_APP_SECRET" not in env
    assert "OPENAI_API_KEY" not in env
    assert env["HOME"] == str(runtime)
    assert env["TMPDIR"] == str(runtime)
    assert env["NO_PROXY"] == ""


def test_real_scanned_pdf_runs_through_manifest_render_ocr_and_cleanup(
    monkeypatch, profile_scope, tmp_path
):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from PIL import Image, ImageDraw

    from document_processing import engine
    from gateway.attachment_provenance import persist_trusted_attachment_manifest
    from gateway.config import Platform

    root = tmp_path / "hermes-root"
    documents = root / "cache" / "documents"
    documents.mkdir(parents=True)
    source = documents / "doc_0123456789ab_scan.pdf"

    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.text((100, 100), "HERMES DOCUMENT 12345", fill="black")
    image.save(source, "PDF", resolution=150.0)

    event = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.FEISHU, profile="hr-agent"),
        media_urls=[str(source)],
        media_types=["application/pdf"],
        message_id="om_neutral_engine_integration",
        timestamp=datetime.now(timezone.utc),
    )
    persist_trusted_attachment_manifest(event, "session-1", hermes_root=root)
    monkeypatch.setattr(engine, "get_default_hermes_root", lambda: root)
    profile = root / "profiles" / "hr-agent"
    profile_scope(profile)

    metadata = json.loads(
        engine.handle_document_extract({"action": "metadata"}, session_id="session-1")
    )
    assert metadata["success"] is True
    assert metadata["pages"] == 1

    rendered = json.loads(
        engine.handle_document_extract(
            {"action": "render_pages", "first_page": 1, "last_page": 1, "dpi": 150},
            session_id="session-1",
        )
    )
    assert rendered["success"] is True
    assert len(rendered["pages"]) == 1
    ocr_image = Path(rendered["pages"][0]["ocr_image"])
    assert ocr_image.is_file()
    assert ocr_image.is_relative_to(profile / "cache" / "document-extract")

    ocr = json.loads(
        engine.handle_document_extract(
            {
                "action": "ocr_image",
                "image_path": str(ocr_image),
                "engine": "tesseract",
                "psm": 6,
            },
            session_id="session-1",
        )
    )
    assert ocr["success"] is True
    assert "HERMES" in ocr["text"].upper()
    assert ocr["item_count"] >= 1

    cleaned = json.loads(
        engine.handle_document_extract({"action": "cleanup"}, session_id="session-1")
    )
    assert cleaned["success"] is True
    assert cleaned["removed_files"] >= 2
    assert not (profile / "cache" / "document-extract" / metadata["source_id"]).exists()


def test_text_layer_preserves_fixed_binary_and_bounded_page_contract(
    monkeypatch, profile_scope, tmp_path
):
    from document_processing import engine

    profile_scope(tmp_path / "purchase-agent")
    source = tmp_path / "doc_0123456789ab_sample.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    artifact = _artifact(engine, source)
    calls: list[tuple[list[str], int, object]] = []

    monkeypatch.setattr(engine, "_authorize_session_pdf", lambda _session_id: artifact)
    monkeypatch.setattr(
        engine,
        "_read_pdf_metadata",
        lambda _artifact: {
            "pages": 12,
            "encrypted": False,
            "page_size": "595 x 842 pts (A4)",
            "raw": "Pages: 12\nEncrypted: no\n",
        },
    )

    def run(command, timeout=60, env_overrides=None, **_kwargs):
        calls.append((command, timeout, env_overrides))
        return subprocess.CompletedProcess(command, 0, stdout="Hello 表格", stderr="")

    monkeypatch.setattr(engine, "_run", run)

    result = json.loads(
        engine.handle_document_extract(
            {"action": "text_layer", "first_page": 2, "last_page": 3},
            session_id="session-1",
        )
    )

    assert result["success"] is True
    assert result["text"] == "Hello 表格"
    assert calls == [
        (
            [
                engine.PDFTOTEXT_BIN,
                "-f",
                "2",
                "-l",
                "3",
                str(source.resolve()),
                "-",
            ],
            120,
            None,
        )
    ]
