"""
Medicine database module for Sehath Saathi.

SQLite backend with schema, CRUD operations, and indexes.
All test records are clearly labelled via data_source and verification_status
so they can be distinguished from any future production medicine catalogue.
"""
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "medicines.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS medicines (
    id TEXT PRIMARY KEY,
    brand_name TEXT NOT NULL,
    active_ingredients TEXT NOT NULL,
    strength TEXT NOT NULL,
    dosage_form TEXT NOT NULL,
    manufacturer TEXT NOT NULL,
    aliases TEXT,
    data_source TEXT NOT NULL DEFAULT 'test_dataset',
    verification_status TEXT NOT NULL DEFAULT 'test_unverified'
        CHECK (verification_status IN ('unverified', 'verified', 'test_unverified')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_medicines_brand_name ON medicines(brand_name);
CREATE INDEX IF NOT EXISTS idx_medicines_manufacturer ON medicines(manufacturer);
CREATE INDEX IF NOT EXISTS idx_medicines_active_ingredients ON medicines(active_ingredients);
CREATE INDEX IF NOT EXISTS idx_medicines_data_source ON medicines(data_source);
"""


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path = DB_PATH) -> None:
    """Create the medicines table and indexes if they do not exist."""
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_medicine(
    conn: sqlite3.Connection,
    brand_name: str,
    active_ingredients: str,
    strength: str,
    dosage_form: str,
    manufacturer: str,
    aliases: str | None = None,
    data_source: str = "test_dataset",
    verification_status: str = "test_unverified",
) -> str:
    """Insert a medicine record and return its generated UUID."""
    medicine_id = str(uuid.uuid4())
    now = _now()
    conn.execute(
        """
        INSERT INTO medicines (
            id, brand_name, active_ingredients, strength, dosage_form,
            manufacturer, aliases, data_source, verification_status,
            created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            medicine_id,
            brand_name,
            active_ingredients,
            strength,
            dosage_form,
            manufacturer,
            aliases,
            data_source,
            verification_status,
            now,
            now,
        ),
    )
    conn.commit()
    return medicine_id


def get_medicine(conn: sqlite3.Connection, medicine_id: str) -> dict | None:
    """Retrieve a single medicine by ID."""
    cur = conn.execute("SELECT * FROM medicines WHERE id = ?", (medicine_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def list_medicines(conn: sqlite3.Connection) -> list[dict]:
    """Return all medicines ordered by brand name."""
    cur = conn.execute("SELECT * FROM medicines ORDER BY brand_name")
    return [dict(row) for row in cur.fetchall()]


def search_by_brand(conn: sqlite3.Connection, brand_name: str) -> list[dict]:
    """Case-insensitive partial match on brand name."""
    cur = conn.execute(
        "SELECT * FROM medicines WHERE brand_name LIKE ? ORDER BY brand_name",
        (f"%{brand_name}%",),
    )
    return [dict(row) for row in cur.fetchall()]


def delete_all_medicines(conn: sqlite3.Connection) -> int:
    """Delete all medicine rows. Mostly useful for test isolation."""
    cur = conn.execute("DELETE FROM medicines")
    conn.commit()
    return cur.rowcount


def count_medicines(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COUNT(*) AS c FROM medicines")
    return cur.fetchone()["c"]