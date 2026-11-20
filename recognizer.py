"""
Sehath Saathi - standalone printed-text OCR module.

This module performs generic OCR only. It does not identify medicines,
interpret ingredients, infer strengths/dosages, or make medical decisions.

Backend usage:
    from recognizer import recognize_text
    result = recognize_text(image)

Accepted input:
    - NumPy ndarray in OpenCV BGR format (preferred)
    - NumPy grayscale ndarray
    - PIL.Image.Image
    - bytes / bytearray / memoryview containing an image
    - filesystem path (str or pathlib.Path)

Install:
    python -m pip install -r requirements.txt
"""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

__version__ = "1.0.0"
SCHEMA_VERSION = "1.0"

MAX_IMAGE_SIDE = 2000
MIN_LINE_CONFIDENCE = 0.25

_engine = None
_engine_lock = threading.Lock()


class ImageLoadError(ValueError):
    """Raised internally when the supplied image cannot be decoded."""


def _normalise_image(image: Any) -> np.ndarray:
    """Return a uint8 OpenCV BGR image."""
    if isinstance(image, np.ndarray):
        arr = image
        if arr.size == 0:
            raise ImageLoadError("The supplied NumPy image is empty.")
        if arr.dtype != np.uint8:
            arr = np.nan_to_num(arr)
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim == 2:
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            bgr = arr.copy()  # ndarray input is assumed to be OpenCV BGR
        elif arr.ndim == 3 and arr.shape[2] == 4:
            bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        else:
            raise ImageLoadError(
                "NumPy image must be HxW, HxWx3 (BGR), or HxWx4 (BGRA)."
            )
    else:
        try:
            if isinstance(image, (str, Path)):
                path = Path(image)
                if not path.is_file():
                    raise ImageLoadError(f"Image file does not exist: {path}")
                pil_image = Image.open(path)
            elif isinstance(image, (bytes, bytearray, memoryview)):
                pil_image = Image.open(io.BytesIO(bytes(image)))
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                raise ImageLoadError(
                    f"Unsupported image input type: {type(image).__name__}"
                )

            pil_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            rgb = np.asarray(pil_image)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except ImageLoadError:
            raise
        except Exception as exc:
            raise ImageLoadError(f"Could not decode the supplied image: {exc}") from exc

    height, width = bgr.shape[:2]
    if height < 1 or width < 1:
        raise ImageLoadError("The supplied image has invalid dimensions.")

    largest_side = max(height, width)
    if largest_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / largest_side
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        bgr = cv2.resize(bgr, new_size, interpolation=cv2.INTER_AREA)

    return bgr


def assess_image_quality(bgr: np.ndarray) -> dict:
    """
    Return basic image-quality measurements.

    These are image heuristics, not a guarantee that OCR will be correct.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    glare_percent = float((gray >= 250).mean() * 100.0)

    if (
        blur_score < 35
        or brightness < 30
        or brightness > 248
        or contrast < 15
        or min(height, width) < 100
    ):
        label = "poor"
    elif (
        blur_score < 100
        or brightness < 55
        or brightness > 240
        or contrast < 25
    ):
        label = "fair"
    else:
        label = "good"

    return {
        "label": label,
        "width": int(width),
        "height": int(height),
        "blur_score": round(blur_score, 2),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "glare_percent": round(glare_percent, 2),
    }


def _get_engine():
    """Create the RapidOCR engine once; model loading is lazy in RapidOCR."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from rapidocr import RapidOCR
                _engine = RapidOCR()
    return _engine


def warmup() -> None:
    """
    Optional backend-startup hook.

    Call once during application startup if you prefer to load the OCR engine
    before the first user request. The first OCR call may otherwise be slower.
    """
    engine = _get_engine()
    # A tiny blank image initializes the engine/model path without requiring
    # a real user image. Any output is intentionally ignored.
    engine(np.full((64, 64, 3), 255, dtype=np.uint8))


def _run_rapidocr(engine, bgr: np.ndarray) -> list[dict]:
    """Adapt RapidOCR 3.x output to a stable list-of-lines structure."""
    result = engine(bgr)
    if result is None:
        return []

    boxes = getattr(result, "boxes", None)
    texts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)

    if boxes is None or texts is None or scores is None:
        return []

    lines = []
    for box, text, score in zip(boxes, texts, scores):
        clean_text = str(text).strip()
        try:
            confidence = float(score)
        except (TypeError, ValueError):
            continue

        if not clean_text or confidence < MIN_LINE_CONFIDENCE:
            continue

        try:
            polygon = [
                [int(round(float(point[0]))), int(round(float(point[1])))]
                for point in box
            ]
        except (TypeError, ValueError, IndexError):
            polygon = []

        lines.append(
            {
                "text": clean_text,
                "confidence": round(max(0.0, min(1.0, confidence)), 4),
                "box": polygon,
            }
        )

    return lines


def recognize_text(image: Any) -> dict:
    """
    Extract printed text from an image.

    Args:
        image: OpenCV BGR ndarray (preferred), grayscale ndarray, PIL image,
               encoded image bytes, or a caller-supplied file path.

    Returns:
        A dictionary with:
          status: "success", "no_text", or "error"
          extracted_text: list[str]
          lines: list[dict] containing text, confidence, and polygon box
          ocr_confidence: mean confidence, from 0.0 to 1.0
          image_quality: "good", "fair", or "poor"
          quality_metrics: image measurements
          processing_ms: elapsed processing time
          error: present only when status is "error"
    """
    started = time.perf_counter()

    try:
        bgr = _normalise_image(image)
        quality = assess_image_quality(bgr)
        engine = _get_engine()

        # Serialize access to the shared engine. This avoids concurrent calls
        # into one model instance from multiple backend request threads.
        with _engine_lock:
            lines = _run_rapidocr(engine, bgr)

        confidences = [line["confidence"] for line in lines]
        mean_confidence = (
            round(sum(confidences) / len(confidences), 4)
            if confidences
            else 0.0
        )

        status = "success" if lines else "no_text"
        warnings = []
        if quality["label"] == "poor":
            warnings.append(
                "Image quality is poor; OCR output may be incomplete or incorrect."
            )
        if quality["glare_percent"] >= 10:
            warnings.append(
                "Bright glare is present and may obscure some printed text."
            )
        if not lines:
            warnings.append(
                "No text was detected. Try a sharper, well-lit image with the "
                "printed area clearly visible."
            )

        return {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "extracted_text": [line["text"] for line in lines],
            "lines": lines,
            "ocr_confidence": mean_confidence,
            "image_quality": quality["label"],
            "quality_metrics": quality,
            "warnings": warnings,
            "processing_ms": int((time.perf_counter() - started) * 1000),
        }

    except ImageLoadError as exc:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "extracted_text": [],
            "lines": [],
            "ocr_confidence": 0.0,
            "image_quality": "poor",
            "quality_metrics": None,
            "warnings": [],
            "error": str(exc),
            "processing_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        # Keep the backend contract stable. Log the returned error on the
        # backend side; do not expose stack traces to end users.
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "extracted_text": [],
            "lines": [],
            "ocr_confidence": 0.0,
            "image_quality": "unknown",
            "quality_metrics": None,
            "warnings": ["OCR processing failed."],
            "error": f"{type(exc).__name__}: {exc}",
            "processing_ms": int((time.perf_counter() - started) * 1000),
        }


# Backwards-friendly generic alias. Prefer recognize_text in new integrations.
read_text = recognize_text


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Run generic printed-text OCR on an image."
    )
    parser.add_argument("image", help="Path to an image file")
    parser.add_argument("--out", help="Optional path to write the JSON result")
    args = parser.parse_args()

    result = recognize_text(args.image)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)

    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    sys.exit(0 if result["status"] in {"success", "no_text"} else 1)
