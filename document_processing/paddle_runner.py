#!/usr/bin/env python3
"""Offline, fixed-model PaddleOCR runner for document_extract."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any


os.environ.setdefault("OMP_THREAD_LIMIT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_MODEL_SPECS = {
    "orientation": {
        "name": "PP-LCNet_x1_0_textline_ori",
        "hashes": {
            "inference.pdiparams": "0de2bcf996cf553e2b848dd7b1769dafffc6917b1ccdf55c1d8efe7909fbf743",
            "inference.json": "b81929eeaff8e52db0fafb49c9fbfc5bc0572f81641fe0179cc52096323fb4d4",
            "inference.yml": "8d5120d0e1a30a9df7ed46aa9119da3796ed066777089d1c1d705f132d5e90f9",
        },
    },
    "detection": {
        "name": "PP-OCRv5_mobile_det",
        "hashes": {
            "inference.pdiparams": "afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b",
            "inference.json": "05feef1acb00aa4cd7362b15f7f501fc4f99d7b1fa73c1c871e0c7b1504b0f5c",
            "inference.yml": "98069072e1b6b37d727fd9d9f11725faa46d6ea0de012f2ed26caea011c37699",
        },
    },
    "recognition": {
        "name": "PP-OCRv5_mobile_rec",
        "hashes": {
            "inference.pdiparams": "2460da90875937c94db97eba74ae3d9e5d4c4c57c42f1f41531c09a26bcc771a",
            "inference.json": "24587345250c7332d0fc6f9a44e794d078cdaeb64c302fef906f325619de2569",
            "inference.yml": "5dfeb2777f6d0db8177d8128a8acfcf6e6276dc4ac73ea3bf0dc06d6a5e85d8e",
        },
    },
}


def _failure(message: str) -> int:
    print(json.dumps({"success": False, "error": message}, ensure_ascii=False))
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_model_dirs() -> dict[str, Path]:
    raw_root = os.environ.get("DOCUMENT_EXTRACT_PADDLE_MODEL_ROOT", "")
    if not raw_root:
        raise RuntimeError("local model root is missing")
    root = Path(raw_root)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("local model root is unavailable")
    root = root.resolve(strict=True)
    resolved: dict[str, Path] = {}
    for role, spec in _MODEL_SPECS.items():
        directory = root / str(spec["name"])
        if directory.is_symlink() or not directory.is_dir() or directory.resolve().parent != root:
            raise RuntimeError(f"local {role} model directory is unavailable")
        for filename, expected in dict(spec["hashes"]).items():
            model_file = directory / filename
            if model_file.is_symlink() or not model_file.is_file() or model_file.stat().st_nlink != 1:
                raise RuntimeError(f"local {role} model file is unavailable")
            if _sha256(model_file) != expected:
                raise RuntimeError(f"local {role} model hash mismatch")
        resolved[role] = directory.resolve(strict=True)
    return resolved


def _install_network_guard() -> None:
    original_socket = socket.socket

    class OfflineSocket(original_socket):
        def connect(self, address):  # type: ignore[override]
            if self.family in {socket.AF_INET, socket.AF_INET6}:
                raise OSError("network disabled for document OCR")
            return super().connect(address)

        def connect_ex(self, address):  # type: ignore[override]
            if self.family in {socket.AF_INET, socket.AF_INET6}:
                return 101
            return super().connect_ex(address)

    def blocked(*_args, **_kwargs):
        raise OSError("network disabled for document OCR")

    socket.socket = OfflineSocket
    socket.create_connection = blocked
    socket.getaddrinfo = blocked


def _bbox_from_polygon(polygon: list[list[int]]) -> dict[str, int]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    left, top = min(xs), min(ys)
    right, bottom = max(xs), max(ys)
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


def _as_dict(result: Any) -> dict[str, Any]:
    value = getattr(result, "json", result)
    return value if isinstance(value, dict) else {}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        return _failure("Exactly one generated PNG path is required")
    image = Path(argv[1])
    try:
        if image.is_symlink() or not image.is_file() or image.stat().st_nlink != 1:
            return _failure("OCR image is unavailable or linked")
        with image.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                return _failure("OCR input is not a real PNG")
        model_dirs = _validated_model_dirs()
    except (OSError, RuntimeError) as exc:
        return _failure(str(exc))

    try:
        _install_network_guard()
        quiet = io.StringIO()
        with contextlib.redirect_stdout(quiet):
            from paddleocr import PaddleOCR

            ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
                textline_orientation_model_dir=str(model_dirs["orientation"]),
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_detection_model_dir=str(model_dirs["detection"]),
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                text_recognition_model_dir=str(model_dirs["recognition"]),
                enable_mkldnn=False,
            )
            results = ocr.predict(str(image))
    except Exception as exc:
        return _failure(f"PaddleOCR failed safely: {type(exc).__name__}")

    items: list[dict[str, Any]] = []
    for result in results:
        payload = _as_dict(result).get("res")
        if not isinstance(payload, dict):
            continue
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        polygons = payload.get("rec_polys") or payload.get("dt_polys") or []
        for text, score, raw_polygon in zip(texts, scores, polygons):
            text = str(text or "").strip()
            try:
                confidence = float(score)
                polygon = [[int(point[0]), int(point[1])] for point in raw_polygon]
            except (TypeError, ValueError, IndexError):
                continue
            if not text or not polygon or not 0.0 <= confidence <= 1.0:
                continue
            bbox = _bbox_from_polygon(polygon)
            if bbox["width"] <= 0 or bbox["height"] <= 0:
                continue
            items.append({
                "text": text[:1000],
                "confidence": round(confidence, 6),
                "bbox": bbox,
                "polygon": polygon,
            })
            if len(items) >= 2000:
                break
        if len(items) >= 2000:
            break

    print(json.dumps({"success": True, "items": items}, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
