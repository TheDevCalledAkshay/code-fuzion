# Sehath Saathi Backend (Milestone 1)

Backend API for the Sehath Saathi hackathon project. This milestone provides a health-check endpoint and an image-upload endpoint for medicine images (OCR and database integration come in later milestones).

## Tech Stack

- Python 3.11+
- FastAPI
- Uvicorn
- python-multipart
- Pillow (for image dimension extraction)
- SQLite (planned for a later milestone)

## Project Structure

```
sehath-saathi-backend/
├── main.py          # FastAPI application
├── requirements.txt # Python dependencies
├── README.md        # This file
└── venv/            # Python virtual environment (created locally)
```

## Installation

### 1. Create and activate the virtual environment

From inside the `sehath-saathi-backend` folder:

```powershell
# Create the virtual environment (run once)
python -m venv venv

# Activate it (PowerShell)
.\venv\Scripts\Activate.ps1
```

> If you get an execution-policy error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## Running the Server

```powershell
# Make sure the venv is activated, then:
python main.py
```

The server will start at `http://localhost:8000`.

Alternatively, run with uvicorn directly:

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### GET /api/health

Health check to confirm the API is running.

**Response:**
```json
{
  "status": "ok",
  "service": "sehath-saathi-backend"
}
```

### POST /api/scan

Upload a medicine image for validation. Returns filename, dimensions, and upload status.

**Supported formats:** JPG, JPEG, PNG, WEBP  
**Maximum file size:** 8 MB

**Request:** `multipart/form-data` with a field named `file`.

**Response (success):**
```json
{
  "filename": "medicine.jpg",
  "format": "JPEG",
  "width": 1024,
  "height": 768,
  "size_bytes": 245760,
  "upload_status": "success",
  "message": "Image uploaded and validated successfully."
}
```

**Error responses:**
- `400` — Unsupported format, missing extension, empty file, or invalid/corrupted image.
- `413` — File exceeds 8 MB.

## Testing with the Interactive Docs

FastAPI automatically generates Swagger UI documentation. Once the server is running, open:

```
http://localhost:8000/docs
```

You can:
1. Click **GET /api/health** → "Try it out" → "Execute" to test the health endpoint.
2. Click **POST /api/scan** → "Try it out" → choose an image file → "Execute" to test image upload.

## Notes

- OCR (medicine text recognition) is **not** implemented in this milestone.
- No medicine database is created yet (SQLite is planned for a later milestone).
- Only the project's `venv` is used for installations; no existing Python environments are modified.