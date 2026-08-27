"""
Named people photo gallery — persistent enroll photos per person.

Files live under ``captures/named_gallery/person_{id}/`` with a ``manifest.json``
that tracks which image is the roster thumbnail (primary).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
NAMED_GALLERY_ROOT = PROJECT_DIR / "captures" / "named_gallery"
MANIFEST_NAME = "manifest.json"


def _person_dir(person_id: int) -> Path:
    return NAMED_GALLERY_ROOT / f"person_{int(person_id)}"


def _manifest_path(person_id: int) -> Path:
    return _person_dir(person_id) / MANIFEST_NAME


def _load_manifest(person_id: int) -> Dict[str, Any]:
    path = _manifest_path(person_id)
    if not path.is_file():
        return {"primary": None, "photos": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"primary": None, "photos": []}
        photos = [str(p) for p in (data.get("photos") or []) if p]
        primary = data.get("primary")
        if primary and str(primary) not in photos and photos:
            primary = photos[0]
        return {"primary": str(primary) if primary else None, "photos": photos}
    except Exception:
        return {"primary": None, "photos": []}


def _save_manifest(person_id: int, manifest: Dict[str, Any]) -> None:
    folder = _person_dir(person_id)
    folder.mkdir(parents=True, exist_ok=True)
    photos = [str(p) for p in (manifest.get("photos") or []) if p]
    primary = manifest.get("primary")
    if primary and str(primary) not in photos:
        primary = photos[0] if photos else None
    elif not primary and photos:
        primary = photos[0]
    out = {"primary": primary, "photos": photos}
    _manifest_path(person_id).write_text(
        json.dumps(out, indent=2),
        encoding="utf-8",
    )


def gallery_photo_url(person_id: int, filename: str) -> str:
    safe = Path(str(filename)).name
    return f"/captures/named_gallery/person_{int(person_id)}/{safe}"


def list_gallery_photos(person_id: int) -> List[Dict[str, Any]]:
    manifest = _load_manifest(person_id)
    primary = manifest.get("primary")
    items: List[Dict[str, Any]] = []
    folder = _person_dir(person_id)
    for name in manifest.get("photos") or []:
        safe = Path(str(name)).name
        path = folder / safe
        if not path.is_file():
            continue
        items.append(
            {
                "filename": safe,
                "url": gallery_photo_url(person_id, safe),
                "is_primary": safe == primary,
            }
        )
    return items


def primary_photo_url(person_id: int) -> Optional[str]:
    photos = list_gallery_photos(person_id)
    if not photos:
        return None
    for item in photos:
        if item.get("is_primary"):
            return str(item["url"])
    return str(photos[0]["url"])


def _write_gallery_image(dest: Path, img: np.ndarray) -> bool:
    from captures import CAPTURE_JPEG_QUALITY, prepare_face_thumbnail

    dest.parent.mkdir(parents=True, exist_ok=True)
    thumb = prepare_face_thumbnail(img)
    return bool(
        cv2.imwrite(
            str(dest),
            thumb,
            [int(cv2.IMWRITE_JPEG_QUALITY), CAPTURE_JPEG_QUALITY],
        )
    )


def add_gallery_photo_from_array(person_id: int, img: np.ndarray) -> Optional[str]:
    if img is None or img.size == 0:
        return None
    filename = f"{uuid.uuid4().hex[:12]}.jpg"
    dest = _person_dir(person_id) / filename
    if not _write_gallery_image(dest, img):
        return None
    manifest = _load_manifest(person_id)
    photos = list(manifest.get("photos") or [])
    if filename not in photos:
        photos.append(filename)
    if not manifest.get("primary"):
        manifest["primary"] = filename
    manifest["photos"] = photos
    _save_manifest(person_id, manifest)
    return filename


def add_gallery_photo_from_path(person_id: int, source: Path) -> Optional[str]:
    if not source.is_file():
        return None
    img = cv2.imread(str(source))
    if img is None or img.size == 0:
        return None
    return add_gallery_photo_from_array(person_id, img)


def _sync_primary_to_best_face(
    conn: sqlite3.Connection,
    person_id: int,
    filename: str,
) -> None:
    """Point person_best_faces.image_path at the gallery primary."""
    from identity import BEST_FACES_DIR

    src = _person_dir(person_id) / Path(str(filename)).name
    if not src.is_file():
        return
    rel = f"named_gallery/person_{int(person_id)}/{Path(str(filename)).name}"
    try:
        BEST_FACES_DIR.mkdir(parents=True, exist_ok=True)
        mirror = BEST_FACES_DIR / f"person_{int(person_id)}_best.jpg"
        shutil.copy2(src, mirror)
    except OSError:
        pass
    now = datetime.now().isoformat(timespec="seconds")
    try:
        conn.execute(
            """
            UPDATE person_best_faces
            SET image_path = ?, updated_at = ?
            WHERE known_person_id = ?
            """,
            (rel, now, int(person_id)),
        )
    except sqlite3.Error:
        pass


def set_primary_photo(
    conn: sqlite3.Connection, person_id: int, filename: str
) -> bool:
    safe = Path(str(filename)).name
    manifest = _load_manifest(person_id)
    if safe not in (manifest.get("photos") or []):
        return False
    manifest["primary"] = safe
    _save_manifest(person_id, manifest)
    _sync_primary_to_best_face(conn, person_id, safe)
    return True


def delete_gallery_photo(
    conn: sqlite3.Connection, person_id: int, filename: str
) -> bool:
    safe = Path(str(filename)).name
    manifest = _load_manifest(person_id)
    photos = [p for p in (manifest.get("photos") or []) if p != safe]
    path = _person_dir(person_id) / safe
    try:
        if path.is_file():
            path.unlink()
    except OSError:
        pass
    if manifest.get("primary") == safe:
        manifest["primary"] = photos[0] if photos else None
    manifest["photos"] = photos
    _save_manifest(person_id, manifest)
    if photos and manifest.get("primary"):
        _sync_primary_to_best_face(conn, person_id, str(manifest["primary"]))
    return True


def clear_gallery_photos(conn: sqlite3.Connection, person_id: int) -> int:
    manifest = _load_manifest(person_id)
    photos = [Path(str(name)).name for name in (manifest.get("photos") or []) if name]
    removed = 0
    folder = _person_dir(person_id)
    for safe in photos:
        path = folder / safe
        try:
            if path.is_file():
                path.unlink()
                removed += 1
        except OSError:
            pass
    try:
        manifest_path = _manifest_path(person_id)
        if manifest_path.is_file():
            manifest_path.unlink()
    except OSError:
        pass
    try:
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
    except OSError:
        pass
    try:
        conn.execute(
            """
            UPDATE person_best_faces
            SET image_path = NULL, updated_at = ?
            WHERE known_person_id = ?
            """,
            (datetime.now().isoformat(timespec="seconds"), int(person_id)),
        )
    except sqlite3.Error:
        pass
    try:
        from identity import BEST_FACES_DIR

        mirror = BEST_FACES_DIR / f"person_{int(person_id)}_best.jpg"
        if mirror.is_file():
            mirror.unlink()
    except Exception:
        pass
    return removed


def archive_enroll_batch(
    conn: sqlite3.Connection, person_id: int, rows: list
) -> int:
    """Save enroll source images into the person's gallery. Returns count added."""
    added = 0
    for row in rows:
        _feature, _quality, img, path, _name = row
        if path is not None and Path(path).is_file():
            if add_gallery_photo_from_path(person_id, Path(path)):
                added += 1
                continue
        if img is not None and getattr(img, "size", 0) > 0:
            if add_gallery_photo_from_array(person_id, img):
                added += 1
    primary = _load_manifest(person_id).get("primary")
    if primary:
        _sync_primary_to_best_face(conn, person_id, str(primary))
    return added


def remove_person_gallery(person_id: int) -> int:
    folder = _person_dir(person_id)
    if not folder.is_dir():
        return 0
    count = sum(1 for p in folder.glob("*.jpg") if p.is_file())
    try:
        shutil.rmtree(folder)
    except OSError:
        pass
    return count


def person_gallery_summary(person_id: int) -> Dict[str, Any]:
    photos = list_gallery_photos(person_id)
    primary = next((p for p in photos if p.get("is_primary")), photos[0] if photos else None)
    return {
        "photo_count": len(photos),
        "photos": photos,
        "photo_url": primary["url"] if primary else None,
        "primary_filename": primary["filename"] if primary else None,
    }


def score_query_against_named_galleries(
    query: np.ndarray,
    conn: sqlite3.Connection,
    *,
    skip_person_ids: Optional[Set[int]] = None,
) -> List[tuple[int, float]]:
    """
    Compare a live face embed to Named People enrollment JPEGs on disk.

    ``skip_person_ids``: people who already have SQLite encodings (preferred).
    """
    from identity import _arcface_embedding_distance, _normalize_face_feature

    q = _normalize_face_feature(query)
    if q is None:
        return []
    skip = {int(x) for x in (skip_person_ids or set())}
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id FROM known_persons
            WHERE COALESCE(is_named, 0) = 1
              AND COALESCE(is_watchlist, 0) = 0
            """
        ).fetchall()
    except sqlite3.Error:
        return []

    hits: List[tuple[int, float]] = []
    for row in rows:
        person_id = int(row["id"] if isinstance(row, sqlite3.Row) else row[0])
        if person_id in skip:
            continue
        manifest = _load_manifest(person_id)
        photo_names = [str(p) for p in (manifest.get("photos") or []) if p]
        if not photo_names:
            continue
        best = float("inf")
        folder = _person_dir(person_id)
        for name in photo_names:
            path = folder / Path(name).name
            if not path.is_file():
                continue
            img = cv2.imread(str(path))
            if img is None or img.size == 0:
                continue
            try:
                from identity import load_bgr_image_oriented

                oriented = load_bgr_image_oriented(path)
                if oriented is not None:
                    img = oriented
            except Exception:
                pass
            try:
                from identity import _embed_scan_reference_face

                embedded = _embed_scan_reference_face(img)
            except Exception:
                embedded = None
            if embedded is None:
                # Sideways phone photo: try cardinal rotations once.
                try:
                    from identity import embed_named_enrollment_variants

                    variants = embed_named_enrollment_variants(img)
                except Exception:
                    variants = []
                if not variants:
                    continue
                feat = variants[0][0]
                dist = _arcface_embedding_distance(q, feat)
                for feat_i, _q, _label in variants[1:]:
                    d = _arcface_embedding_distance(q, feat_i)
                    if d < dist:
                        dist = d
                        feat = feat_i
                if dist < best:
                    best = float(dist)
                continue
            feat, _meta = embedded
            dist = _arcface_embedding_distance(q, feat)
            if dist < best:
                best = float(dist)
        if np.isfinite(best):
            hits.append((person_id, best))
    hits.sort(key=lambda item: item[1])
    return hits


def sync_named_encodings_from_gallery(
    conn: sqlite3.Connection,
    person_id: int,
) -> int:
    """
    Replace SQLite face embeddings for one Named person with gallery photo embeds only.

    Stores upright + rotated variants so CCTV matches work for sideways / tilted
    enrollment photos. Removes live CCTV pollution.
    """
    from identity import (
        load_bgr_image_oriented,
        refresh_gpu_face_index,
        store_named_enrollment_embeddings,
    )

    pid = int(person_id)
    manifest = _load_manifest(pid)
    photo_names = [str(p) for p in (manifest.get("photos") or []) if p]
    if not photo_names:
        return 0
    folder = _person_dir(pid)
    conn.execute("DELETE FROM person_encodings WHERE known_person_id = ?", (pid,))
    conn.execute("DELETE FROM person_best_faces WHERE known_person_id = ?", (pid,))
    added = 0
    for name in photo_names:
        path = folder / Path(name).name
        if not path.is_file():
            continue
        img = load_bgr_image_oriented(path)
        if img is None or img.size == 0:
            continue
        added += int(store_named_enrollment_embeddings(conn, pid, img, replace_existing=False))
    conn.commit()
    try:
        refresh_gpu_face_index(conn)
    except Exception:
        pass
    return added
