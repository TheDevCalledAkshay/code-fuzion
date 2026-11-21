"""
Initialize the Sehath Saathi medicine database and insert a demo Dolo-650 record.

This script is idempotent: it checks for an existing Dolo-650 record before inserting,
never creates duplicate records, and never overwrites existing records.

Uses the configured database path via SEHATH_DB_PATH environment variable,
or defaults to medicines.db in the project directory.

Ensures proper connection handling: always commits and closes.

Designed for safe deployment on Render (ephemeral filesystem) - run once
before or during startup, then the database persists for the session.
"""
import os
import sys

# Add project root to path so we can import database module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database


def init_demo_record(db_path=None):
    """
    Initialize the database and insert a Dolo-650 demo record if one does not
    already exist.

    Parameters:
        db_path (str | None): Path to SQLite database file.
            If None, uses SEHATH_DB_PATH env var, or defaults to medicines.db
            in the project directory.

    Returns:
        tuple: (db_path, record_inserted_bool)
    """
    # Determine database path
    if db_path is None:
        db_path = os.environ.get("SEHATH_DB_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "medicines.db"))

    conn = database.get_connection(db_path)
    try:
        # Ensure schema exists
        database.init_db(db_path)

        # Check for existing Dolo-650 record (idempotent)
        existing = database.search_by_brand(conn, "Dolo-650")
        if existing:
            print("Dolo-650 demo record already exists - skipping insertion")
            return db_path, False

        # Insert Dolo-650 demo record
        # Manufacturer: "Not specified" - honest notation that no manufacturer info is available
        # This does not disqualify a match per matching engine design
        database.insert_medicine(
            conn,
            brand_name="Dolo-650",
            active_ingredients="Paracetamol",
            strength="650 mg",
            dosage_form="Tablet",
            manufacturer="Not specified",
            data_source="demo_dataset",
            verification_status="test_unverified",
        )
        print("Dolo-650 demo record inserted successfully")
        return db_path, True
    finally:
        conn.commit()
        conn.close()


def main():
    """Entry point: initialize database and insert demo record."""
    db_path, inserted = init_demo_record()
    if inserted:
        print(f"Database initialized at: {db_path}")
    else:
        print(f"Database already initialized at: {db_path}")


if __name__ == "__main__":
    main()