#!/usr/bin/env python3
"""Reset known_person IDs so the next stranger starts at 1."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = PROJECT_DIR / "visitor_tracking.db"
BEST_FACES = PROJECT_DIR / "captures" / "best_faces"


def main() -> int:
    if not DB_PATH.is_file():
        print(f"No database at {DB_PATH}")
        return 1

    con = sqlite3.connect(str(DB_PATH), timeout=60)
    try:
        before = con.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM known_persons").fetchone()
        print(f"[RESET-IDS] known_persons before: count={before[0]} max_id={before[1]}")

        con.execute("BEGIN")
        # Clear identity gallery + embeddings (all current rows are unlabeled strangers).
        for table in (
            "person_encodings",
            "person_body_encodings",
            "person_best_faces",
            "gray_zone_audit",
            "known_persons",
        ):
            try:
                con.execute(f"DELETE FROM {table}")
                print(f"[RESET-IDS] cleared {table}")
            except sqlite3.Error as exc:
                print(f"[RESET-IDS] skip {table}: {exc}")

        # Detach old IDs from visit history (keep capture rows/images).
        for table, cols in (
            ("tracking_logs", ("known_person_id",)),
            ("tracking_sessions", ("known_person_id",)),
        ):
            try:
                con.execute(f"UPDATE {table} SET known_person_id = NULL")
                print(f"[RESET-IDS] nulled known_person_id on {table}")
            except sqlite3.Error as exc:
                print(f"[RESET-IDS] skip {table}: {exc}")

        # Next INSERT gets id = 1.
        try:
            con.execute(
                "UPDATE sqlite_sequence SET seq = 0 WHERE name = 'known_persons'"
            )
            if con.total_changes == 0:
                con.execute(
                    "INSERT OR REPLACE INTO sqlite_sequence(name, seq) VALUES ('known_persons', 0)"
                )
            print("[RESET-IDS] sqlite_sequence known_persons -> 0")
        except sqlite3.Error as exc:
            print(f"[RESET-IDS] sequence reset note: {exc}")

        con.commit()

        after = con.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM known_persons").fetchone()
        seq = con.execute(
            "SELECT seq FROM sqlite_sequence WHERE name = 'known_persons'"
        ).fetchone()
        print(f"[RESET-IDS] known_persons after: count={after[0]} max_id={after[1]} seq={seq}")
    finally:
        con.close()

    removed = 0
    if BEST_FACES.is_dir():
        for path in BEST_FACES.glob("person_*_best.jpg"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
    print(f"[RESET-IDS] removed {removed} best-face thumbnail(s)")

    # Rebuild FAISS / watchlist indexes if available.
    try:
        from identity import init_identity_db, refresh_gpu_face_index

        conn = sqlite3.connect(str(DB_PATH), timeout=60)
        try:
            init_identity_db(conn)
            n = refresh_gpu_face_index(conn)
            print(f"[RESET-IDS] rebuilt face index entries={n}")
        finally:
            conn.close()
    except Exception as exc:
        print(f"[RESET-IDS] face index rebuild skipped: {exc}")

    print("[RESET-IDS] Done. Next new person will be ID 1.")
    print("[RESET-IDS] Restart run_system.py so live tracker drops old in-memory ID maps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
