# Sehath Saathi OCR module (generic OCR only)

## Files
- `recognizer.py` — standalone OCR module.
- `requirements.txt` — Python dependencies.

## Install
Use the same Python environment as the backend:

```powershell
python -m pip install -r requirements.txt
```

## Backend integration

```python
from recognizer import recognize_text

def process_uploaded_image(decoded_image):
    result = recognize_text(decoded_image)

    if result["status"] == "error":
        # Log result.get("error") on the server.
        return {"ok": False, "error": "OCR processing failed"}

    return {
        "ok": True,
        "text": result["extracted_text"],
        "ocr_confidence": result["ocr_confidence"],
        "image_quality": result["image_quality"],
        "status": result["status"],
    }
```

`decoded_image` may be an OpenCV BGR NumPy array, a PIL image, encoded image
bytes, or a path. For API uploads, decode the upload in the backend and pass
the resulting image object to `recognize_text`.

## Optional startup warmup

```python
from recognizer import warmup
warmup()
```

This can make the first request less likely to pay the model initialization
cost. It may take time and memory, so call it during startup only if suitable.

## Test

```powershell
python -m py_compile recognizer.py
python recognizer.py path\to\sample.jpg --out result.json
```

The CLI command performs real OCR and requires the dependencies and model
runtime to be installed. A successful Python syntax check alone does not
prove OCR runtime compatibility.

## Output shape

- `status`: `success`, `no_text`, or `error`
- `extracted_text`: list of detected text strings
- `lines`: each detected line's text, confidence, and polygon coordinates
- `ocr_confidence`: mean line confidence (not a guarantee of correctness)
- `image_quality`: `good`, `fair`, or `poor`
- `quality_metrics`: dimensions, blur, brightness, contrast, and glare
- `warnings`, `processing_ms`, and optional `error`

## Scope

This module extracts visible text only. It does not identify products or
medicines, interpret ingredients, infer dosage/strength, or recommend actions.
Any downstream interpretation must be handled separately and reviewed
appropriately.
