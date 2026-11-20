import logging
import os

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool
from PIL import Image
import io

from recognizer import recognize_text

import database
import preprocessing
import matching

app = FastAPI(title="Sehath Saathi Backend", version="1.0.0")

# Allow Gokul's GitHub Pages frontend to call the API.
# Replace the origin with the actual domain when deployed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("sehath-saathi")

logger = logging.getLogger("sehath-saathi")

# Database path: allow override via environment variable for testing.
# Defaults to medicines.db in the project directory.
DB_PATH = os.environ.get("SEHATH_DB_PATH", database.DB_PATH)

# Ensure the database schema exists.
database.init_db(DB_PATH)

# Supported image extensions and MIME types
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
}

# Maximum file size: 8 MB
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8,388,608 bytes


@app.get("/api/health")
async def health():
    """Health check endpoint to verify the API is running."""
    return {"status": "ok", "service": "sehath-saathi-backend"}


@app.post("/api/scan")
async def scan_medicine_image(file: UploadFile = File(...)):
    """
    Accept an uploaded medicine image, validate its format and size,
    run OCR on the image, and return upload metadata alongside OCR results.
    """
    filename = file.filename or ""

    # --- Validate file extension ---
    if "." not in filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename: no file extension found.",
        )

    ext = filename.rsplit(".", 1)[-1].lower()
    if f".{ext}" not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '.{ext}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            ),
        )

    # --- Validate content type (secondary check) ---
    content_type = file.content_type or ""
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type '{content_type}'. "
                f"Supported formats: JPG, JPEG, PNG, WEBP."
            ),
        )

    # --- Read file contents ---
    contents = await file.read()

    # --- Validate file size ---
    if len(contents) > MAX_FILE_SIZE:
        size_mb = len(contents) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {size_mb:.2f} MB. "
                f"Maximum allowed size is 8 MB."
            ),
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --- Validate image and extract dimensions ---
    try:
        image = Image.open(io.BytesIO(contents))
        image.load()  # Force full decode to catch corrupt files
        width, height = image.size
        image_format = image.format  # e.g. "JPEG", "PNG", "WEBP"
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file: {exc}",
        )

    # --- Run OCR in a threadpool to avoid blocking the event loop ---
    try:
        ocr_result = await run_in_threadpool(recognize_text, contents)
    except Exception as exc:
        logger.exception("OCR processing failed for file %s", filename)
        raise HTTPException(
            status_code=500,
            detail="OCR processing failed.",
        )

    return {
        "filename": filename,
        "format": image_format,
        "width": width,
        "height": height,
        "size_bytes": len(contents),
        "upload_status": "success",
        "ocr": ocr_result,
    }


@app.post("/api/v1/scan")
async def scan_and_identify(image: UploadFile = File(...)):
    """
    Accept an uploaded medicine image, validate it, run OCR, and attempt
    medicine identification against the database.

    Returns OCR results alongside medicine matching results.
    """
    filename = image.filename or ""

    # --- Validate file extension ---
    if "." not in filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename: no file extension found.",
        )

    ext = filename.rsplit(".", 1)[-1].lower()
    if f".{ext}" not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '.{ext}'. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
            ),
        )

    # --- Validate content type (secondary check) ---
    content_type = image.content_type or ""
    if content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported content type '{content_type}'. "
                f"Supported formats: JPG, JPEG, PNG, WEBP."
            ),
        )

    # --- Read file contents ---
    contents = await image.read()

    # --- Validate file size ---
    if len(contents) > MAX_FILE_SIZE:
        size_mb = len(contents) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {size_mb:.2f} MB. "
                f"Maximum allowed size is 8 MB."
            ),
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    # --- Validate image and extract dimensions ---
    try:
        pil_image = Image.open(io.BytesIO(contents))
        pil_image.load()
        width, height = pil_image.size
        image_format = pil_image.format
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or corrupted image file: {exc}",
        )

    # --- Run OCR in a threadpool ---
    try:
        ocr_result = await run_in_threadpool(recognize_text, contents)
    except Exception as exc:
        logger.exception("OCR processing failed for file %s", filename)
        raise HTTPException(
            status_code=500,
            detail="OCR processing failed.",
        )

    ocr_status = ocr_result.get("status", "error")

    # --- Build OCR section of response ---
    ocr_response = {
        "status": ocr_status,
        "extracted_text": ocr_result.get("extracted_text", []),
        "ocr_confidence": ocr_result.get("ocr_confidence", 0.0),
        "image_quality": ocr_result.get("image_quality", "unknown"),
        "warnings": ocr_result.get("warnings", []),
    }

    # --- If OCR failed or returned no text, skip matching ---
    if ocr_status != "success":
        return {
            "scan_status": "success",
            "ocr": ocr_response,
            "identification": {
                "status": "no_match",
                "candidates": [],
                "message": "OCR did not return usable text for identification.",
            },
        }

    # --- Preprocess OCR text ---
    extracted_text = ocr_result.get("extracted_text", [])
    ocr_text = " ".join(extracted_text) if extracted_text else ""
    entities = preprocessing.preprocess_ocr_text(ocr_text)

    # --- Load medicines and match ---
    conn = database.get_connection(DB_PATH)
    try:
        medicines = database.list_medicines(conn)
        match_result = matching.match_medicine(entities, medicines)
    finally:
        conn.close()

    # --- Build candidate list ---
    candidates = []
    for cand in match_result["candidates"]:
        med_record = None
        for m in medicines:
            if m["id"] == cand["medicine_id"]:
                med_record = m
                break

        candidates.append(
            {
                "medicine_id": cand["medicine_id"],
                "brand_name": cand["brand_name"],
                "active_ingredients": (
                    med_record.get("active_ingredients", "")
                    if med_record
                    else cand["score_breakdown"]["active_ingredients"]["db_value"]
                ),
                "strength": (
                    med_record.get("strength", "")
                    if med_record
                    else cand["score_breakdown"]["strength"]["db_value"]
                ),
                "dosage_form": (
                    med_record.get("dosage_form", "")
                    if med_record
                    else cand["score_breakdown"]["dosage_form"]["db_value"]
                ),
                "manufacturer": (
                    med_record.get("manufacturer", "")
                    if med_record
                    else cand["score_breakdown"]["manufacturer"]["db_value"]
                ),
                "matching_score": cand["score"],
                "matched_attributes": cand["matched_attributes"],
                "missing_attributes": cand["missing_attributes"],
                "contradictory_attributes": cand["contradictory_attributes"],
                "verification_status": (
                    med_record.get("verification_status", "unverified")
                    if med_record
                    else "unverified"
                ),
            }
        )

    # --- Determine identification status ---
    overall_status = match_result["overall_status"]

    # Safety: a strength conflict must not produce a confident identification.
    if overall_status == "match" and candidates:
        top = candidates[0]
        if "strength" in top["contradictory_attributes"]:
            overall_status = "ambiguous"

    # Build user-facing message.
    if overall_status == "no_match":
        message = "No matching medicine found."
        candidates = []
    elif overall_status == "ambiguous":
        message = "Multiple similarly matching medicines found. Result is ambiguous."
    else:
        message = "Medicine identified."

    return {
        "scan_status": "success",
        "ocr": ocr_response,
        "identification": {
            "status": overall_status,
            "candidates": candidates,
            "message": message,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)