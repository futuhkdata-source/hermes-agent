from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, NamedTuple

from gateway.attachment_provenance import (
    AttachmentProvenanceError,
    resolve_trusted_pdf_attachment,
)
from hermes_constants import get_default_hermes_root, get_hermes_home


PADDLE_RUNNER = Path(__file__).resolve().with_name("paddle_runner.py")
PYTHON_BIN = sys.executable
PDFINFO_BIN = shutil.which("pdfinfo") or "/usr/bin/pdfinfo"
PDFTOTEXT_BIN = shutil.which("pdftotext") or "/usr/bin/pdftotext"
PDFTOPPM_BIN = shutil.which("pdftoppm") or "/usr/bin/pdftoppm"
TESSERACT_BIN = shutil.which("tesseract") or "/usr/bin/tesseract"
PRLIMIT_BIN = shutil.which("prlimit") or "/usr/bin/prlimit"
PADDLE_MODEL_ROOT = Path.home() / ".paddlex" / "official_models"

_MAX_SOURCE_BYTES = 200 * 1024 * 1024
_MAX_PAGES = 500
_MAX_PAGES_PER_CALL = 10
_MAX_TEXT_CHARS = 200_000
_MAX_OCR_ITEMS = 2_000
_MAX_REGIONS = 50
_MAX_IMAGE_PIXELS = 40_000_000
_MAX_IMAGE_SIDE = 12_000
_MAX_IMAGE_FILE_BYTES = 128 * 1024 * 1024
_MAX_SOURCE_WORK_BYTES = 512 * 1024 * 1024
_DEFAULT_OUTPUT_LIMIT = 8 * 1024 * 1024

_GENERATED_IMAGE_RE = re.compile(
    r"^(?:"
    r"page-[0-9]{4}-[0-9]{3}dpi-(?:vision|ocr)|"
    r"tile-page-[0-9]{4}-r[0-9]{2}-c[0-9]{2}-[0-9a-f]{10}|"
    r"crop-page-[0-9]{4}-[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9a-f]{10}|"
    r"region-page-[0-9]{4}-[0-9]{3}-[0-9a-f]{10}"
    r")\.png$"
)
_PAGE_RE = re.compile(r"page-([0-9]{4})")


class DocumentExtractError(RuntimeError):
    pass


class SourceArtifact(NamedTuple):
    path: Path
    digest: str
    size: int
    filename: str


def _success(**payload: Any) -> str:
    return json.dumps({"success": True, **payload}, ensure_ascii=False, allow_nan=False)


def _failure(message: str) -> str:
    return json.dumps({"success": False, "error": message}, ensure_ascii=False, allow_nan=False)


def _paddle_models_present() -> bool:
    required = (
        PADDLE_MODEL_ROOT / "PP-LCNet_x1_0_textline_ori",
        PADDLE_MODEL_ROOT / "PP-OCRv5_mobile_det",
        PADDLE_MODEL_ROOT / "PP-OCRv5_mobile_rec",
    )
    return all(
        directory.is_dir()
        and not directory.is_symlink()
        and (directory / "inference.pdiparams").is_file()
        and (directory / "inference.json").is_file()
        and (directory / "inference.yml").is_file()
        for directory in required
    )


def check_requirements() -> bool:
    return (
        all(Path(binary).is_file() for binary in (
            PDFINFO_BIN, PDFTOTEXT_BIN, PDFTOPPM_BIN, TESSERACT_BIN,
            PYTHON_BIN, PRLIMIT_BIN,
        ))
        and PADDLE_RUNNER.is_file()
        and _paddle_models_present()
        and importlib.util.find_spec("cv2") is not None
    )


def _runtime_profile_name() -> str:
    try:
        profile_home = Path(get_hermes_home()).resolve(strict=True)
        root = Path(get_default_hermes_root()).resolve(strict=True)
    except (OSError, RuntimeError):
        raise DocumentExtractError("Runtime profile home failed identity validation") from None
    if profile_home == root:
        return "default"
    if profile_home.parent.name == "profiles" and profile_home.parent.parent == root:
        return profile_home.name
    raise DocumentExtractError("Runtime profile home failed identity validation")


def _authorize_session_pdf(session_id: str) -> SourceArtifact:
    """Resolve only gateway-persisted structured provenance for this session."""
    if not session_id or not isinstance(session_id, str):
        raise DocumentExtractError("A runtime session_id is required")
    try:
        record = resolve_trusted_pdf_attachment(
            session_id,
            profile_name=_runtime_profile_name(),
            hermes_root=get_default_hermes_root(),
        )
    except AttachmentProvenanceError as exc:
        raise DocumentExtractError(str(exc)) from None
    candidate = Path(str(record["path"])).resolve(strict=True)
    size = int(record["size"])
    if size <= 8 or size > _MAX_SOURCE_BYTES:
        raise DocumentExtractError("The PDF attachment size is outside the allowed range")
    display_name = candidate.name.split("_", 2)[-1]
    return SourceArtifact(candidate, str(record["sha256"]), size, display_name)


def _build_subprocess_env(
    command: list[str],
    env_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    if command and command[0] == PYTHON_BIN:
        # Paddle receives an explicit minimal environment. Gateway/API secrets
        # are intentionally not inherited by the OCR child.
        python_dir = str(Path(PYTHON_BIN).parent)
        env = {
            "PATH": os.pathsep.join((python_dir, "/usr/bin", "/bin")),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "NO_PROXY": "",
            "no_proxy": "",
            "HTTP_PROXY": "http://127.0.0.1:9",
            "HTTPS_PROXY": "http://127.0.0.1:9",
            "ALL_PROXY": "http://127.0.0.1:9",
            "OMP_THREAD_LIMIT": "1",
            "OMP_NUM_THREADS": "1",
            "FLAGS_use_mkldnn": "0",
            "FLAGS_enable_pir_api": "0",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DOCUMENT_EXTRACT_PADDLE_MODEL_ROOT": str(PADDLE_MODEL_ROOT),
        }
    else:
        env = os.environ.copy()
    if env_overrides:
        env.update({str(key): str(value) for key, value in env_overrides.items()})
    return env


def _read_capped_stream(handle: Any, limit: int) -> tuple[str, bool]:
    handle.flush()
    size = handle.tell()
    handle.seek(0)
    payload = handle.read(limit + 1)
    return payload.decode("utf-8", errors="replace"), size > limit or len(payload) > limit


def _run(
    command: list[str],
    timeout: int = 60,
    env_overrides: dict[str, str] | None = None,
    *,
    output_limit: int = _DEFAULT_OUTPUT_LIMIT,
) -> subprocess.CompletedProcess[str]:
    if not command or not isinstance(command, list):
        raise DocumentExtractError("Document command is invalid")
    allowed = {PDFINFO_BIN, PDFTOTEXT_BIN, PDFTOPPM_BIN, TESSERACT_BIN, PYTHON_BIN}
    if command[0] not in allowed:
        raise DocumentExtractError("Document command is not allowlisted")
    if command[0] == PYTHON_BIN and (
        len(command) < 2 or Path(command[1]).resolve() != PADDLE_RUNNER.resolve()
    ):
        raise DocumentExtractError("Python document runner is not allowlisted")
    if not 1 <= int(output_limit) <= _MAX_IMAGE_FILE_BYTES:
        raise DocumentExtractError("Subprocess output limit is invalid")
    if not Path(PRLIMIT_BIN).is_file():
        raise DocumentExtractError("prlimit is required for document subprocess isolation")

    env = _build_subprocess_env(command, env_overrides)
    constrained = [
        PRLIMIT_BIN,
        "--as=6442450944",
        f"--fsize={_MAX_IMAGE_FILE_BYTES}",
        f"--cpu={max(5, int(timeout) + 5)}",
        "--nofile=64",
        "--",
        *command,
    ]
    with tempfile.TemporaryFile(mode="w+b") as stdout_handle, tempfile.TemporaryFile(mode="w+b") as stderr_handle:
        try:
            process = subprocess.Popen(
                constrained,
                stdout=stdout_handle,
                stderr=stderr_handle,
                shell=False,
                env=env,
                close_fds=True,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
                raise DocumentExtractError("Document command timed out") from None
        except DocumentExtractError:
            raise
        except OSError as exc:
            raise DocumentExtractError(
                f"Document command could not start: {type(exc).__name__}"
            ) from None

        stdout, stdout_overflow = _read_capped_stream(stdout_handle, int(output_limit))
        stderr, stderr_overflow = _read_capped_stream(stderr_handle, min(int(output_limit), 1024 * 1024))
        if stdout_overflow or stderr_overflow:
            raise DocumentExtractError("Document command exceeded its output limit")
        result = subprocess.CompletedProcess(command, process.returncode, stdout=stdout, stderr=stderr)
    if result.returncode != 0:
        error = (result.stderr or result.stdout or "").strip().splitlines()
        detail = error[-1][:240] if error else f"exit code {result.returncode}"
        raise DocumentExtractError(f"Document command failed safely: {detail}")
    return result


def _bounded_text(value: str) -> tuple[str, bool]:
    if len(value) <= _MAX_TEXT_CHARS:
        return value, False
    return value[:_MAX_TEXT_CHARS], True


def _read_pdf_metadata(artifact: SourceArtifact) -> dict[str, Any]:
    result = _run([PDFINFO_BIN, str(artifact.path)], 60)
    pages_match = re.search(r"(?mi)^Pages:\s*([0-9]+)\s*$", result.stdout)
    if not pages_match:
        raise DocumentExtractError("pdfinfo did not report a page count")
    pages = int(pages_match.group(1))
    if pages < 1 or pages > _MAX_PAGES:
        raise DocumentExtractError(f"PDF page count must be within 1-{_MAX_PAGES}")
    encrypted_match = re.search(r"(?mi)^Encrypted:\s*(yes|no)\b", result.stdout)
    page_size_match = re.search(r"(?mi)^Page size:\s*(.+)$", result.stdout)
    return {
        "pages": pages,
        "encrypted": bool(encrypted_match and encrypted_match.group(1).lower() == "yes"),
        "page_size": page_size_match.group(1).strip() if page_size_match else "",
        "raw": result.stdout,
    }


def _validate_pages(first_raw: Any, last_raw: Any, total_pages: int) -> tuple[int, int]:
    try:
        first = int(first_raw)
        last = int(last_raw)
    except (TypeError, ValueError):
        raise DocumentExtractError("first_page and last_page must be integers") from None
    if first < 1 or last < first or last > total_pages:
        raise DocumentExtractError(f"Page range must be within 1-{total_pages}")
    if last - first + 1 > _MAX_PAGES_PER_CALL:
        raise DocumentExtractError(f"At most {_MAX_PAGES_PER_CALL} pages may be processed per call")
    return first, last


def _validate_page_dimensions(
    artifact: SourceArtifact,
    *,
    page: int,
    dpi: int,
) -> tuple[int, int]:
    result = _run(
        [
            PDFINFO_BIN, "-f", str(page), "-l", str(page), "-box",
            str(artifact.path),
        ],
        60,
        output_limit=1024 * 1024,
    )
    match = re.search(
        r"(?mi)^Page(?:\s+[0-9]+)?\s+size:\s*"
        r"([0-9]+(?:\.[0-9]+)?)\s+x\s+([0-9]+(?:\.[0-9]+)?)\s+pts",
        result.stdout,
    )
    if not match:
        raise DocumentExtractError("pdfinfo did not report bounded page dimensions")
    width_points = float(match.group(1))
    height_points = float(match.group(2))
    width = int(math.ceil(width_points * dpi / 72.0))
    height = int(math.ceil(height_points * dpi / 72.0))
    if (
        width < 1 or height < 1
        or width > _MAX_IMAGE_SIDE or height > _MAX_IMAGE_SIDE
        or width * height > _MAX_IMAGE_PIXELS
    ):
        raise DocumentExtractError("Rendered page would exceed the pixel limit")
    return width, height


def _validate_png_bounds(path: Path) -> tuple[int, int]:
    from PIL import Image

    try:
        stat_result = path.stat()
        if (
            path.is_symlink() or not path.is_file() or stat_result.st_nlink != 1
            or stat_result.st_size <= 8 or stat_result.st_size > _MAX_IMAGE_FILE_BYTES
        ):
            raise DocumentExtractError("Generated PNG exceeds the file-size limit")
        with path.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                raise DocumentExtractError("Generated image is not a real PNG")
        with Image.open(path) as image:
            width, height = (int(image.size[0]), int(image.size[1]))
    except DocumentExtractError:
        raise
    except Exception:
        raise DocumentExtractError("Generated PNG header could not be validated") from None
    if (
        width < 1 or height < 1
        or width > _MAX_IMAGE_SIDE or height > _MAX_IMAGE_SIDE
        or width * height > _MAX_IMAGE_PIXELS
    ):
        raise DocumentExtractError("Generated PNG exceeds the pixel limit")
    return width, height


def _ensure_work_quota(work: Path) -> int:
    total = 0
    try:
        for candidate in work.rglob("*"):
            if candidate.is_symlink():
                raise DocumentExtractError("Work artifact link aliases are not allowed")
            if candidate.is_file():
                stat_result = candidate.stat()
                if stat_result.st_nlink != 1:
                    raise DocumentExtractError("Work artifact link aliases are not allowed")
                total += int(stat_result.st_size)
                if total > _MAX_SOURCE_WORK_BYTES:
                    raise DocumentExtractError("Source document work quota exceeded")
    except DocumentExtractError:
        raise
    except OSError:
        raise DocumentExtractError("Source document work quota could not be verified") from None
    return total


def _prepare_work_root() -> Path:
    profile_home_raw = Path(get_hermes_home())
    try:
        if profile_home_raw.is_symlink():
            raise DocumentExtractError("Linked document work directories are not allowed")
        profile_home_raw.mkdir(parents=True, exist_ok=True, mode=0o700)
        profile_home = profile_home_raw.resolve(strict=True)
        cache = profile_home / "cache"
        work_root = cache / "document-extract"
        if cache.is_symlink() or work_root.is_symlink():
            raise DocumentExtractError("Linked document work directories are not allowed")
        work_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        work_root.chmod(0o700)
        resolved = work_root.resolve(strict=True)
    except DocumentExtractError:
        raise
    except (OSError, RuntimeError):
        raise DocumentExtractError("Document work root could not be prepared") from None
    if resolved != work_root or resolved.parent != cache:
        raise DocumentExtractError("Document work root failed path validation")
    return resolved


def _source_workdir(artifact: SourceArtifact) -> Path:
    root = _prepare_work_root()
    work = root / artifact.digest[:16]
    try:
        if work.is_symlink():
            raise DocumentExtractError("Linked source work directories are not allowed")
        work.mkdir(parents=False, exist_ok=True, mode=0o700)
        work.chmod(0o700)
        resolved = work.resolve(strict=True)
    except DocumentExtractError:
        raise
    except (OSError, RuntimeError):
        raise DocumentExtractError("Source work directory could not be prepared") from None
    if resolved.parent != root or resolved.name != artifact.digest[:16]:
        raise DocumentExtractError("Source work directory failed path validation")
    _ensure_work_quota(resolved)
    return resolved


def _prepare_paddle_runtime(work: Path) -> dict[str, str]:
    runtime = work / "paddle-runtime"
    if runtime.is_symlink():
        raise DocumentExtractError("Paddle runtime directory must not be a link")
    try:
        runtime.mkdir(parents=False, exist_ok=True, mode=0o700)
        runtime.chmod(0o700)
        resolved = runtime.resolve(strict=True)
        if resolved.parent != work:
            raise DocumentExtractError("Paddle runtime escaped the source work directory")
        locations = {}
        for name in ("home", "tmp", "cache"):
            location = resolved / name
            if location.is_symlink():
                raise DocumentExtractError("Paddle runtime location must not be a link")
            location.mkdir(parents=False, exist_ok=True, mode=0o700)
            location.chmod(0o700)
            locations[name] = str(location.resolve(strict=True))
    except DocumentExtractError:
        raise
    except (OSError, RuntimeError):
        raise DocumentExtractError("Paddle runtime could not be prepared") from None
    _ensure_work_quota(work)
    return {
        "HOME": locations["home"],
        "TMPDIR": locations["tmp"],
        "XDG_CACHE_HOME": locations["cache"],
        "DOCUMENT_EXTRACT_RUNTIME_ROOT": str(resolved),
    }


def _validate_generated_image(value: Any, artifact: SourceArtifact) -> Path:
    work = _source_workdir(artifact)
    raw = Path(str(value or "")).expanduser()
    try:
        if raw.is_symlink():
            raise DocumentExtractError("Generated image link aliases are not allowed")
        image = raw.resolve(strict=True)
        stat_result = image.stat()
    except DocumentExtractError:
        raise
    except (OSError, RuntimeError):
        raise DocumentExtractError("Generated image is unavailable") from None
    if (
        image.parent != work
        or not image.is_file()
        or stat_result.st_nlink != 1
        or not _GENERATED_IMAGE_RE.fullmatch(image.name)
    ):
        raise DocumentExtractError("Generated image must be inside the current source work directory")
    _validate_png_bounds(image)
    return image


def _write_png(path: Path, image: Any) -> None:
    import cv2

    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise DocumentExtractError("Generated image target failed validation")
    temp = path.with_name(f".{path.name}.tmp.png")
    try:
        if temp.exists() or temp.is_symlink():
            temp.unlink(missing_ok=True)
        if not cv2.imwrite(str(temp), image):
            raise DocumentExtractError("OpenCV could not write a generated PNG")
        _validate_png_bounds(temp)
        os.replace(temp, path)
        path.chmod(0o600)
    finally:
        temp.unlink(missing_ok=True)


def _rotate_image(image: Any, degrees: int) -> Any:
    import cv2

    normalized = degrees % 360
    if normalized == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if normalized == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if normalized == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _count_upright_faces(image: Any) -> int:
    """Count upright frontal faces only as a local 0-vs-180 orientation signal."""
    import cv2

    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    if not cascade_path.is_file():
        return 0
    detector = cv2.CascadeClassifier(str(cascade_path))
    if detector.empty():
        return 0
    height, width = image.shape[:2]
    if width > 1600:
        scale = 1600.0 / width
        image = cv2.resize(
            image,
            (1600, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24),
    )
    return int(len(faces))


def _face_orientation_adjustment(image: Any) -> tuple[int, dict[str, int]]:
    current = _count_upright_faces(image)
    rotated = _count_upright_faces(_rotate_image(image, 180))
    adjustment = 0
    if rotated >= max(5, current * 2) and rotated - current >= 4:
        adjustment = 180
    return adjustment, {
        "face_count_current": current,
        "face_count_rotated_180": rotated,
    }


def _estimate_skew_degrees(image: Any) -> float:
    import cv2
    import numpy as np

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_length = max(80, min(gray.shape[:2]) // 6)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=80,
                            minLineLength=min_length, maxLineGap=20)
    if lines is None:
        return 0.0
    angles: list[float] = []
    for line in lines[:500]:
        x1, y1, x2, y2 = [int(v) for v in line[0]]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180
        if abs(angle) <= 5:
            angles.append(angle)
    if len(angles) < 3:
        return 0.0
    median = float(statistics.median(angles))
    return median if 0.25 <= abs(median) <= 5.0 else 0.0


def _deskew_image(image: Any, skew_degrees: float) -> Any:
    import cv2

    if not skew_degrees:
        return image
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), skew_degrees, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _detect_osd_rotation(raw_path: Path) -> tuple[int, float]:
    try:
        result = _run(
            [TESSERACT_BIN, str(raw_path), "stdout", "-l", "osd", "--psm", "0"],
            120,
            {"OMP_THREAD_LIMIT": "1", "OMP_NUM_THREADS": "1"},
        )
    except DocumentExtractError:
        return 0, 0.0
    combined = f"{result.stdout}\n{result.stderr}"
    rotation_match = re.search(r"(?mi)^Rotate:\s*(0|90|180|270)\s*$", combined)
    confidence_match = re.search(r"(?mi)^Orientation confidence:\s*([0-9.]+)\s*$", combined)
    rotation = int(rotation_match.group(1)) if rotation_match else 0
    confidence = float(confidence_match.group(1)) if confidence_match else 0.0
    return rotation, confidence


def _preprocess_page(raw_path: Path, vision_path: Path, ocr_path: Path) -> dict[str, Any]:
    import cv2

    _validate_png_bounds(raw_path)
    image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if image is None:
        raise DocumentExtractError("Rendered PDF page is not a readable image")
    osd_rotation, orientation_confidence = _detect_osd_rotation(raw_path)
    oriented = _rotate_image(image, osd_rotation)
    face_adjustment, face_evidence = _face_orientation_adjustment(oriented)
    oriented = _rotate_image(oriented, face_adjustment)
    rotation = (osd_rotation + face_adjustment) % 360
    skew = _estimate_skew_degrees(oriented)
    corrected = _deskew_image(oriented, skew)
    _write_png(vision_path, corrected)

    gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _write_png(ocr_path, enhanced)
    height, width = corrected.shape[:2]
    return {
        "rotation_degrees": rotation,
        "osd_rotation_degrees": osd_rotation,
        "face_orientation_adjustment_degrees": face_adjustment,
        "orientation_confidence": round(orientation_confidence, 3),
        "deskew_degrees": round(skew, 3),
        "width": int(width),
        "height": int(height),
        **face_evidence,
    }


def _page_from_name(path: Path) -> int:
    match = _PAGE_RE.search(path.name)
    if not match:
        raise DocumentExtractError("Generated image filename does not preserve page evidence")
    return int(match.group(1))


def _bbox(x: int, y: int, width: int, height: int) -> dict[str, int]:
    return {"x": int(x), "y": int(y), "width": int(width), "height": int(height)}


def _load_cv_image(path: Path) -> Any:
    import cv2

    _validate_png_bounds(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise DocumentExtractError("Generated PNG could not be decoded")
    return image


def _crop_to_path(image: Any, box: dict[str, int], output: Path) -> None:
    x, y, width, height = box["x"], box["y"], box["width"], box["height"]
    crop = image[y:y + height, x:x + width]
    if crop.size == 0:
        raise DocumentExtractError("Requested crop is empty")
    try:
        _write_png(output, crop)
        _ensure_work_quota(output.parent)
    except Exception:
        output.unlink(missing_ok=True)
        raise


def _nms_boxes(boxes: list[tuple[int, int, int, int]], threshold: float = 0.55):
    def area(box):
        return box[2] * box[3]

    def iou(left, right):
        lx, ly, lw, lh = left
        rx, ry, rw, rh = right
        x1, y1 = max(lx, rx), max(ly, ry)
        x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = area(left) + area(right) - intersection
        return intersection / union if union else 0.0

    kept: list[tuple[int, int, int, int]] = []
    for candidate in sorted(boxes, key=area, reverse=True):
        if all(iou(candidate, existing) < threshold for existing in kept):
            kept.append(candidate)
    return kept


def _detect_rectangle_boxes(image: Any, max_regions: int) -> list[tuple[int, int, int, int]]:
    import cv2

    height, width = image.shape[:2]
    page_area = width * height
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 140)
    kernel_size = max(3, int(round(min(width, height) / 150)))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        ratio = (bw * bh) / page_area
        aspect = bw / bh if bh else 0.0
        if 0.005 <= ratio <= 0.5 and 0.25 <= aspect <= 4.0 and bw >= 40 and bh >= 30:
            boxes.append((int(x), int(y), int(bw), int(bh)))
    boxes = _nms_boxes(boxes)
    boxes.sort(key=lambda box: (box[1], box[0]))
    return boxes[:max_regions]


def _parse_tesseract_tsv(tsv: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row.get("conf") or -1)
            x = int(row.get("left") or 0)
            y = int(row.get("top") or 0)
            width = int(row.get("width") or 0)
            height = int(row.get("height") or 0)
        except (TypeError, ValueError):
            continue
        if confidence < 0 or width <= 0 or height <= 0:
            continue
        items.append({
            "text": text[:1000],
            "confidence": round(confidence, 3),
            "bbox": _bbox(x, y, width, height),
        })
        if len(items) >= _MAX_OCR_ITEMS:
            break
    return items


def _summarize_ocr_items(items: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    confidences = [float(item["confidence"]) for item in items]
    text, truncated = _bounded_text("\n".join(str(item["text"]) for item in items))
    return {
        "text": text,
        "text_truncated": truncated,
        "item_count": len(items),
        "average_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
        "low_confidence_count": sum(1 for value in confidences if value < threshold),
    }


def _sanitize_paddle_items(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise DocumentExtractError("PaddleOCR runner returned malformed items")
    items: list[dict[str, Any]] = []
    for raw in value[:_MAX_OCR_ITEMS]:
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text") or "").strip()
        bbox = raw.get("bbox")
        polygon = raw.get("polygon")
        try:
            confidence = float(raw.get("confidence"))
            clean_bbox = _bbox(
                int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"]),
            )
            clean_polygon = [[int(point[0]), int(point[1])] for point in polygon]
        except (TypeError, ValueError, KeyError, IndexError):
            continue
        if not text or not 0.0 <= confidence <= 1.0:
            continue
        items.append({
            "text": text[:1000],
            "confidence": round(confidence, 6),
            "bbox": clean_bbox,
            "polygon": clean_polygon,
        })
    return items


def handle_document_extract(args: dict[str, Any], **kwargs: Any) -> str:
    try:
        session_id = str(kwargs.get("session_id") or "").strip()
        if not session_id:
            raise DocumentExtractError("A runtime session_id is required")
        artifact = _authorize_session_pdf(session_id)
        action = str(args.get("action") or "").strip().lower()
        metadata = _read_pdf_metadata(artifact)
        source_evidence = {
            "source_id": artifact.digest[:16],
            "filename": artifact.filename,
            "source_bytes": artifact.size,
        }

        if action == "metadata":
            raw, truncated = _bounded_text(str(metadata.get("raw") or ""))
            return _success(
                **source_evidence,
                pages=metadata["pages"],
                encrypted=metadata["encrypted"],
                page_size=metadata.get("page_size") or "",
                metadata=raw,
                metadata_truncated=truncated,
            )

        if metadata["encrypted"]:
            raise DocumentExtractError("Encrypted PDFs are not supported")

        if action == "text_layer":
            first, last = _validate_pages(args.get("first_page"), args.get("last_page"), metadata["pages"])
            result = _run([
                PDFTOTEXT_BIN, "-f", str(first), "-l", str(last), str(artifact.path), "-",
            ], 120)
            text, truncated = _bounded_text(result.stdout)
            return _success(
                **source_evidence, first_page=first, last_page=last,
                text=text, text_chars=len(result.stdout), truncated=truncated,
            )

        if action == "render_pages":
            first, last = _validate_pages(args.get("first_page"), args.get("last_page"), metadata["pages"])
            try:
                dpi = int(args.get("dpi", 300))
            except (TypeError, ValueError):
                raise DocumentExtractError("dpi must be an integer") from None
            if dpi < 150 or dpi > 300:
                raise DocumentExtractError("dpi must be within 150-300")
            work = _source_workdir(artifact)
            pages: list[dict[str, Any]] = []
            for page in range(first, last + 1):
                stem = f"page-{page:04d}-{dpi:03d}dpi"
                raw_prefix = work / f".{stem}-raw"
                raw_path = Path(str(raw_prefix) + ".png")
                vision_path = work / f"{stem}-vision.png"
                ocr_path = work / f"{stem}-ocr.png"
                sidecar = work / f".{stem}-evidence.json"
                cached = False
                evidence: dict[str, Any]
                if vision_path.is_file() and ocr_path.is_file() and sidecar.is_file():
                    try:
                        evidence = json.loads(sidecar.read_text(encoding="utf-8"))
                        _validate_generated_image(str(vision_path), artifact)
                        _validate_generated_image(str(ocr_path), artifact)
                        cached = True
                    except Exception:
                        cached = False
                if not cached:
                    for candidate in (raw_path, vision_path, ocr_path, sidecar):
                        if candidate.exists() or candidate.is_symlink():
                            if candidate.is_dir():
                                raise DocumentExtractError("Generated page target is unexpectedly a directory")
                            candidate.unlink()
                    _validate_page_dimensions(artifact, page=page, dpi=dpi)
                    _ensure_work_quota(work)
                    try:
                        _run([
                            PDFTOPPM_BIN, "-f", str(page), "-l", str(page),
                            "-r", str(dpi), "-png", "-singlefile",
                            str(artifact.path), str(raw_prefix),
                        ], 180)
                        if (
                            raw_path.is_symlink() or not raw_path.is_file()
                            or raw_path.resolve().parent != work
                        ):
                            raise DocumentExtractError("Rendered page output failed path validation")
                        _validate_png_bounds(raw_path)
                        evidence = _preprocess_page(raw_path, vision_path, ocr_path)
                        sidecar.write_text(
                            json.dumps(evidence, ensure_ascii=False), encoding="utf-8",
                        )
                        sidecar.chmod(0o600)
                        _validate_generated_image(str(vision_path), artifact)
                        _validate_generated_image(str(ocr_path), artifact)
                        _ensure_work_quota(work)
                    except Exception:
                        for candidate in (raw_path, vision_path, ocr_path, sidecar):
                            if candidate.exists() and candidate.is_file():
                                candidate.unlink(missing_ok=True)
                        raise
                    finally:
                        raw_path.unlink(missing_ok=True)
                pages.append({
                    "page": page,
                    "dpi": dpi,
                    "vision_image": str(vision_path.resolve()),
                    "ocr_image": str(ocr_path.resolve()),
                    "cached": cached,
                    **evidence,
                })
            return _success(**source_evidence, pages=pages)

        if action in {"detect_regions", "tile_image", "crop_image", "ocr_image"}:
            image_path = _validate_generated_image(args.get("image_path"), artifact)
            image = _load_cv_image(image_path)
            page = _page_from_name(image_path)
            height, width = image.shape[:2]
            work = _source_workdir(artifact)

            if action == "detect_regions":
                try:
                    max_regions = int(args.get("max_regions", 30))
                except (TypeError, ValueError):
                    raise DocumentExtractError("max_regions must be an integer") from None
                if not 1 <= max_regions <= _MAX_REGIONS:
                    raise DocumentExtractError(f"max_regions must be within 1-{_MAX_REGIONS}")
                create_crops = bool(args.get("create_crops", True))
                boxes = _detect_rectangle_boxes(image, max_regions)
                regions: list[dict[str, Any]] = []
                for index, (x, y, bw, bh) in enumerate(boxes, start=1):
                    box = _bbox(x, y, bw, bh)
                    digest = hashlib.sha256(f"{image_path.name}:{x}:{y}:{bw}:{bh}".encode()).hexdigest()[:10]
                    output = work / f"region-page-{page:04d}-{index:03d}-{digest}.png"
                    if create_crops:
                        _crop_to_path(image, box, output)
                    regions.append({
                        "region": index,
                        "page": page,
                        "bbox": box,
                        "area_ratio": round((bw * bh) / (width * height), 6),
                        "image_path": str(output.resolve()) if create_crops else None,
                    })
                return _success(
                    **source_evidence, page=page, source_image=str(image_path),
                    image_width=width, image_height=height, regions=regions,
                    fallback=("Use tile_image when rectangle detection misses dense or irregular regions"
                              if not regions else ""),
                )

            if action == "tile_image":
                try:
                    rows = int(args.get("rows", 2))
                    columns = int(args.get("columns", 2))
                    overlap = float(args.get("overlap", 0.08))
                except (TypeError, ValueError):
                    raise DocumentExtractError("rows, columns and overlap must be numeric") from None
                if not 1 <= rows <= 6 or not 1 <= columns <= 6:
                    raise DocumentExtractError("rows and columns must be within 1-6")
                if not 0.0 <= overlap <= 0.2:
                    raise DocumentExtractError("overlap must be within 0.0-0.2")
                base_w = math.ceil(width / columns)
                base_h = math.ceil(height / rows)
                pad_x = int(round(base_w * overlap))
                pad_y = int(round(base_h * overlap))
                regions = []
                for row in range(rows):
                    for column in range(columns):
                        x1 = max(0, column * base_w - pad_x)
                        y1 = max(0, row * base_h - pad_y)
                        x2 = min(width, (column + 1) * base_w + pad_x)
                        y2 = min(height, (row + 1) * base_h + pad_y)
                        box = _bbox(x1, y1, x2 - x1, y2 - y1)
                        digest = hashlib.sha256(
                            f"{image_path.name}:{rows}:{columns}:{row}:{column}:{overlap}".encode()
                        ).hexdigest()[:10]
                        output = work / f"tile-page-{page:04d}-r{row + 1:02d}-c{column + 1:02d}-{digest}.png"
                        _crop_to_path(image, box, output)
                        regions.append({
                            "row": row + 1, "column": column + 1, "page": page,
                            "bbox": box, "image_path": str(output.resolve()),
                        })
                return _success(
                    **source_evidence, page=page, source_image=str(image_path),
                    image_width=width, image_height=height, regions=regions,
                )

            if action == "crop_image":
                try:
                    x = int(args.get("x"))
                    y = int(args.get("y"))
                    crop_width = int(args.get("width"))
                    crop_height = int(args.get("height"))
                except (TypeError, ValueError):
                    raise DocumentExtractError("Crop coordinates must be integers") from None
                if (
                    x < 0 or y < 0 or crop_width < 16 or crop_height < 16
                    or x + crop_width > width or y + crop_height > height
                ):
                    raise DocumentExtractError("Crop bounding box is outside the generated image")
                box = _bbox(x, y, crop_width, crop_height)
                digest = hashlib.sha256(
                    f"{image_path.name}:{x}:{y}:{crop_width}:{crop_height}".encode()
                ).hexdigest()[:10]
                output = work / (
                    f"crop-page-{page:04d}-{x}-{y}-{crop_width}-{crop_height}-{digest}.png"
                )
                _crop_to_path(image, box, output)
                return _success(
                    **source_evidence, page=page, bbox=box,
                    source_image=str(image_path), image_path=str(output.resolve()),
                )

            engine = str(args.get("engine") or "tesseract").strip().lower()
            if engine == "tesseract":
                try:
                    psm = int(args.get("psm", 11))
                except (TypeError, ValueError):
                    raise DocumentExtractError("psm must be 6 or 11") from None
                if psm not in {6, 11}:
                    raise DocumentExtractError("psm must be 6 or 11")
                result = _run([
                    TESSERACT_BIN, str(image_path), "stdout", "-l", "chi_tra+eng",
                    "--psm", str(psm), "tsv",
                ], 180, {"OMP_THREAD_LIMIT": "1", "OMP_NUM_THREADS": "1"})
                items = _parse_tesseract_tsv(result.stdout)
                summary = _summarize_ocr_items(items, 70.0)
            elif engine == "paddle":
                paddle_env = _prepare_paddle_runtime(work)
                result = _run(
                    [PYTHON_BIN, str(PADDLE_RUNNER), str(image_path)],
                    240,
                    paddle_env,
                )
                try:
                    payload = json.loads(result.stdout)
                except json.JSONDecodeError:
                    raise DocumentExtractError("PaddleOCR runner returned invalid JSON") from None
                if not isinstance(payload, dict) or payload.get("success") is not True:
                    raise DocumentExtractError("PaddleOCR runner failed safely")
                items = _sanitize_paddle_items(payload.get("items"))
                summary = _summarize_ocr_items(items, 0.7)
            else:
                raise DocumentExtractError("engine must be tesseract or paddle")
            return _success(
                **source_evidence, page=page, engine=engine,
                source_image=str(image_path), items=items, **summary,
            )

        if action == "cleanup":
            root = _prepare_work_root()
            work = root / artifact.digest[:16]
            removed_files = 0
            if work.exists() or work.is_symlink():
                if work.is_symlink() or not work.is_dir() or work.resolve().parent != root:
                    raise DocumentExtractError("Source cleanup target failed path validation")
                removed_files = sum(1 for candidate in work.rglob("*") if candidate.is_file())
                shutil.rmtree(work)
            return _success(**source_evidence, removed_files=removed_files)

        raise DocumentExtractError(
            "Unknown action; use metadata, text_layer, render_pages, detect_regions, "
            "tile_image, crop_image, ocr_image, or cleanup"
        )
    except DocumentExtractError as exc:
        return _failure(str(exc))
    except Exception as exc:
        return _failure(f"document_extract failed safely: {type(exc).__name__}")
