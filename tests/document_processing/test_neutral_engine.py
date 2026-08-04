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
        "extract",
        "metadata",
        "text_layer",
        "render_pages",
        "detect_regions",
        "tile_image",
        "crop_image",
        "ocr_image",
        "cleanup",
    }
    assert properties["max_paddle_pages"]["maximum"] == 2


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

    adaptive = json.loads(
        engine.handle_document_extract(
            {
                "action": "extract",
                "first_page": 1,
                "last_page": 1,
                "dpi": 150,
                "max_paddle_pages": 0,
            },
            session_id="session-1",
        )
    )
    assert adaptive["success"] is True
    assert adaptive["pages"][0]["method"] == "tesseract"
    assert "HERMES" in adaptive["text"].upper()
    assert adaptive["paddle_pages_used"] == 0
    assert adaptive["has_more"] is False

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


def test_real_text_pdf_extract_uses_text_layer_fast_path(
    monkeypatch, profile_scope, tmp_path
):
    from datetime import datetime, timezone
    from types import SimpleNamespace

    from reportlab.pdfgen import canvas

    from document_processing import engine
    from gateway.attachment_provenance import persist_trusted_attachment_manifest
    from gateway.config import Platform

    root = tmp_path / "hermes-root"
    documents = root / "cache" / "documents"
    documents.mkdir(parents=True)
    source = documents / "doc_abcdef123456_text.pdf"
    writer = canvas.Canvas(str(source))
    writer.drawString(72, 720, "HERMES TEXT LAYER INVOICE 67890")
    writer.save()

    event = SimpleNamespace(
        source=SimpleNamespace(platform=Platform.FEISHU, profile="purchase-agent"),
        media_urls=[str(source)],
        media_types=["application/pdf"],
        message_id="om_neutral_text_integration",
        timestamp=datetime.now(timezone.utc),
    )
    persist_trusted_attachment_manifest(event, "session-text", hermes_root=root)
    monkeypatch.setattr(engine, "get_default_hermes_root", lambda: root)
    profile = root / "profiles" / "purchase-agent"
    profile_scope(profile)

    result = json.loads(
        engine.handle_document_extract(
            {"action": "extract", "max_paddle_pages": 0},
            session_id="session-text",
        )
    )

    assert result["success"] is True
    assert result["pages"][0]["method"] == "text_layer"
    assert "HERMES TEXT LAYER INVOICE 67890" in result["text"]
    assert result["paddle_pages_used"] == 0
    assert result["review_pages"] == []
    assert not (profile / "cache" / "document-extract").exists()


def test_paddle_runner_uses_host_slot(monkeypatch, tmp_path):
    from contextlib import contextmanager

    from document_processing import engine

    image = tmp_path / "page-0001-300dpi-ocr.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    work = tmp_path / "work"
    work.mkdir()
    seen: dict[str, object] = {}

    @contextmanager
    def slot(profile_name: str, *, timeout: float):
        seen["profile"] = profile_name
        seen["timeout"] = timeout
        yield 1.25

    monkeypatch.setattr(engine, "paddle_slot", slot)
    monkeypatch.setattr(engine, "_runtime_profile_name", lambda: "hr-agent")
    monkeypatch.setattr(engine, "_prepare_paddle_runtime", lambda _work: {"HOME": "/tmp"})
    monkeypatch.setattr(
        engine,
        "_run",
        lambda command, timeout, env: subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"success": True, "items": []}),
            stderr="",
        ),
    )

    result, waited = engine._run_paddle_ocr(image, work)

    assert json.loads(result.stdout)["success"] is True
    assert waited == 1.25
    assert seen == {"profile": "hr-agent", "timeout": 180.0}


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


def test_extract_page_uses_text_layer_without_render_or_ocr(monkeypatch):
    from document_processing import engine

    calls: list[dict] = []

    def internal(_session_id: str, args: dict):
        calls.append(args)
        assert args["action"] == "text_layer"
        return {"success": True, "text": "Invoice number 12345", "truncated": False}

    monkeypatch.setattr(engine, "_internal_document_call", internal)

    result = engine._extract_page_auto(
        "session-1", page=3, dpi=300, paddle_allowed=True,
    )

    assert result["page"] == 3
    assert result["method"] == "text_layer"
    assert result["text"] == "Invoice number 12345"
    assert result["paddle_used"] is False
    assert result["needs_vision_review"] is False
    assert calls == [{"action": "text_layer", "first_page": 3, "last_page": 3}]


def test_extract_page_uses_bounded_paddle_fallback_for_weak_scan(monkeypatch):
    from document_processing import engine

    calls: list[dict] = []

    def internal(_session_id: str, args: dict):
        calls.append(args)
        action = args["action"]
        if action == "text_layer":
            return {"success": True, "text": "", "truncated": False}
        if action == "render_pages":
            return {
                "success": True,
                "pages": [{
                    "page": 2,
                    "vision_image": "/tmp/page-0002-300dpi-vision.png",
                    "ocr_image": "/tmp/page-0002-300dpi-ocr.png",
                    "rotation_degrees": 0,
                    "deskew_degrees": 0.0,
                }],
            }
        if action == "ocr_image" and args["engine"] == "tesseract":
            return {
                "success": True,
                "text": "BAD",
                "average_confidence": 40.0,
                "low_confidence_count": 3,
                "item_count": 3,
            }
        if action == "ocr_image" and args["engine"] == "paddle":
            return {
                "success": True,
                "text": "GOOD DOCUMENT TEXT",
                "average_confidence": 0.95,
                "low_confidence_count": 0,
                "item_count": 3,
            }
        raise AssertionError(args)

    monkeypatch.setattr(engine, "_internal_document_call", internal)

    result = engine._extract_page_auto(
        "session-1", page=2, dpi=300, paddle_allowed=True,
    )

    assert result["method"] == "paddle"
    assert result["text"] == "GOOD DOCUMENT TEXT"
    assert result["paddle_used"] is True
    assert result["needs_vision_review"] is False
    assert result["vision_image"].endswith("vision.png")
    assert [call["action"] for call in calls] == [
        "text_layer", "render_pages", "ocr_image", "ocr_image",
    ]


def test_extract_action_caps_range_and_paddle_budget(
    monkeypatch, profile_scope, tmp_path
):
    from document_processing import engine

    profile_scope(tmp_path / "profile")
    source = tmp_path / "doc_0123456789ab_sample.pdf"
    source.write_bytes(b"%PDF-1.4\n%%EOF\n")
    artifact = _artifact(engine, source)
    seen: list[tuple[int, bool]] = []

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

    def extract(_session_id: str, *, page: int, dpi: int, paddle_allowed: bool):
        seen.append((page, paddle_allowed))
        return {
            "page": page,
            "method": "paddle" if paddle_allowed else "tesseract",
            "text": f"page {page}",
            "text_chars": 6,
            "average_confidence": 95.0,
            "low_confidence_count": 0,
            "needs_vision_review": False,
            "paddle_used": paddle_allowed,
        }

    monkeypatch.setattr(engine, "_extract_page_auto", extract)

    result = json.loads(
        engine.handle_document_extract(
            {"action": "extract", "max_paddle_pages": 1},
            session_id="session-1",
        )
    )

    assert result["success"] is True
    assert result["first_page"] == 1
    assert result["last_page"] == 10
    assert result["has_more"] is True
    assert result["next_page"] == 11
    assert len(result["pages"]) == 10
    assert seen == [(1, True)] + [(page, False) for page in range(2, 11)]
