"""Backfill missing age/gender on tracking_logs from saved face captures."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import cv2

PROJECT_DIR = Path(__file__).parent
DEFAULT_DB = PROJECT_DIR / "visitor_tracking.db"
CAPTURES_DIR = PROJECT_DIR / "captures"


def backfill(db_path: Path, limit: int = 0) -> tuple[int, int]:
    from face_attributes import (
        active_backend_name,
        attribute_models_ready,
        preload_attribute_models,
        predict_face_attributes,
        UNKNOWN_GENDER,
    )

    if not attribute_models_ready():
        raise RuntimeError(
            "No age/gender backend available. Install InsightFace buffalo_l or DeepFace."
        )
    preload_attribute_models()
    backend = active_backend_name()
    print(f"Backfill backend: {backend}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, capture_path, gender, age
        FROM tracking_logs
        WHERE capture_path IS NOT NULL AND TRIM(capture_path) != ''
        ORDER BY id DESC
        """
    ).fetchall()

    updated = 0
    scanned = 0
    for row in rows:
        if limit and scanned >= limit:
            break
        scanned += 1
        gender = str(row["gender"] or "").strip()
        age = row["age"]
        needs_gender = not gender or gender.lower() in ("unknown", "n/a")
        needs_age = age is None
        if not needs_gender and not needs_age:
            continue

        capture_name = Path(str(row["capture_path"])).name
        capture_path = CAPTURES_DIR / capture_name
        if not capture_path.is_file():
            continue

        img = cv2.imread(str(capture_path))
        if img is None or img.size == 0:
            continue

        pred_gender, pred_age = predict_face_attributes(img)
        new_gender = pred_gender if needs_gender and pred_gender != UNKNOWN_GENDER else gender
        new_age = pred_age if needs_age and pred_age is not None else age
        if new_gender == gender and new_age == age:
            continue

        conn.execute(
            "UPDATE tracking_logs SET gender = ?, age = ? WHERE id = ?",
            (new_gender or UNKNOWN_GENDER, new_age, row["id"]),
        )
        updated += 1

    conn.commit()
    conn.close()
    return scanned, updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill tracking_logs demographics")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--limit", type=int, default=0, help="Max rows to scan (0 = all)")
    args = parser.parse_args()

    scanned, updated = backfill(Path(args.db), limit=args.limit)
    print(f"Scanned {scanned} rows, updated {updated} demographics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
