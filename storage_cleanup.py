"""
Automated retention cleanup for tracking logs and saved capture images.

Default: keep captured JPEGs for 14 days, export them to CSV, then delete.
The daemon in run_system.py checks on an interval (default daily).
"""

from __future__ import annotations

import csv
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from captures import CAPTURES_DIR

log = logging.getLogger("storage_cleanup")

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = PROJECT_DIR / "visitor_tracking.db"
EXPORTS_DIR = PROJECT_DIR / "exports"
ENV_PATH = PROJECT_DIR / ".env"


def _load_dotenv(path: Path = ENV_PATH) -> None:
    """Load .env so retention settings apply even if the launcher imported us first."""
    if not path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=False)
        return
    except ImportError:
        pass
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv()

CAPTURE_RETENTION_DAYS = float(os.environ.get("CAPTURE_RETENTION_DAYS", "14"))
LOG_RETENTION_DAYS = float(
    os.environ.get("LOG_RETENTION_DAYS", str(CAPTURE_RETENTION_DAYS))
)
RETENTION_HOURS = CAPTURE_RETENTION_DAYS * 24.0
CLEANUP_INTERVAL_SEC = float(os.environ.get("CAPTURE_CLEANUP_INTERVAL_SEC", str(24 * 3600)))

_CAPTURE_DIRS: Tuple[Path, ...] = (
    CAPTURES_DIR,
    PROJECT_DIR / "static" / "captured_faces",
    PROJECT_DIR / "static" / "crops",
)
_CAPTURE_SKIP_DIRS = frozenset({"best_faces"})
_JPEG_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
_CSV_FIELDS = (
    "filename",
    "relative_path",
    "absolute_path",
    "size_bytes",
    "modified_at",
    "log_id",
    "person_id",
    "camera_location",
    "timestamp",
    "known_person_id",
    "gender",
    "age",
    "capture_path",
    "scene_path",
    "exported_at",
)


def _iso_cutoff(days: float) -> str:
    return (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")


def _mtime_cutoff(days: float) -> float:
    return time.time() - (days * 86400.0)


def _iter_capture_files(folder: Path) -> Iterable[Path]:
    if not folder.is_dir():
        return
    for path in folder.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _CAPTURE_SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in _JPEG_SUFFIXES:
            yield path


def collect_stale_capture_images(
    *,
    retention_days: float = CAPTURE_RETENTION_DAYS,
    capture_dirs: Tuple[Path, ...] = _CAPTURE_DIRS,
) -> List[Path]:
    """JPEG/PNG files older than the retention window (skips best_faces)."""
    cutoff = _mtime_cutoff(retention_days)
    stale: List[Path] = []
    for base in capture_dirs:
        if not base.is_dir():
            continue
        for path in _iter_capture_files(base):
            try:
                if os.path.getmtime(path) < cutoff:
                    stale.append(path)
            except OSError:
                continue
    return stale


def _load_log_index(db_path: Path) -> Dict[str, Dict[str, object]]:
    """Map capture/scene filename → tracking_logs row."""
    index: Dict[str, Dict[str, object]] = {}
    if not db_path.is_file():
        return index
    try:
        with sqlite3.connect(str(db_path), timeout=30.0) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=30000")
            cols = {row[1] for row in conn.execute("PRAGMA table_info(tracking_logs)")}
            if "capture_path" not in cols:
                return index
            extra = ", scene_path" if "scene_path" in cols else ""
            rows = conn.execute(
                f"""
                SELECT id, person_id, camera_location, timestamp, known_person_id,
                       capture_path, gender, age{extra}
                FROM tracking_logs
                """
            ).fetchall()
    except Exception as exc:
        log.warning("Could not read tracking_logs for CSV export: %s", exc)
        return index

    for row in rows:
        record = dict(row)
        for key in ("capture_path", "scene_path"):
            raw = record.get(key)
            if not raw:
                continue
            name = Path(str(raw)).name.lower()
            if name and name not in index:
                index[name] = record
    return index


def export_capture_images_csv(
    files: List[Path],
    *,
    db_path: Path = DEFAULT_DB_PATH,
    dest: Optional[Path] = None,
) -> Optional[Path]:
    """Write a CSV catalog of capture files (joined to visit logs when possible)."""
    if not files:
        return None
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = dest or (EXPORTS_DIR / f"captures_archive_{stamp}.csv")
    log_index = _load_log_index(db_path)
    exported_at = datetime.now().isoformat(timespec="seconds")

    with dest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for path in files:
            try:
                rel = str(path.resolve().relative_to(PROJECT_DIR.resolve()))
            except ValueError:
                rel = str(path)
            try:
                size = path.stat().st_size
                modified = datetime.fromtimestamp(path.stat().st_mtime).isoformat(
                    timespec="seconds"
                )
            except OSError:
                size = ""
                modified = ""
            row = log_index.get(path.name.lower(), {})
            writer.writerow(
                {
                    "filename": path.name,
                    "relative_path": rel.replace("\\", "/"),
                    "absolute_path": str(path),
                    "size_bytes": size,
                    "modified_at": modified,
                    "log_id": row.get("id", ""),
                    "person_id": row.get("person_id", ""),
                    "camera_location": row.get("camera_location", ""),
                    "timestamp": row.get("timestamp", ""),
                    "known_person_id": row.get("known_person_id", ""),
                    "gender": row.get("gender", ""),
                    "age": row.get("age", ""),
                    "capture_path": row.get("capture_path", ""),
                    "scene_path": row.get("scene_path", ""),
                    "exported_at": exported_at,
                }
            )
    return dest


def purge_old_tracking_logs(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    retention_days: float = LOG_RETENTION_DAYS,
) -> int:
    """Delete tracking_logs rows older than the retention window. Returns rows removed."""
    if not db_path.is_file():
        return 0

    cutoff = _iso_cutoff(retention_days)
    try:
        with sqlite3.connect(str(db_path), timeout=30.0) as conn:
            conn.execute("PRAGMA busy_timeout=30000")
            cur = conn.execute(
                "DELETE FROM tracking_logs WHERE timestamp < ?",
                (cutoff,),
            )
            conn.commit()
            return int(cur.rowcount)
    except Exception as exc:
        log.warning("Database purge failed (%s): %s", db_path.name, exc)
        return 0


def purge_stale_capture_images(
    files: Optional[List[Path]] = None,
    *,
    retention_days: float = CAPTURE_RETENTION_DAYS,
    capture_dirs: Tuple[Path, ...] = _CAPTURE_DIRS,
) -> int:
    """Delete stale capture files. Returns files removed."""
    stale = files if files is not None else collect_stale_capture_images(
        retention_days=retention_days,
        capture_dirs=capture_dirs,
    )
    removed = 0
    for path in stale:
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except OSError as exc:
            log.debug("Could not delete %s: %s", path, exc)
    return removed


def run_cleanup_cycle(
    db_path: Path = DEFAULT_DB_PATH,
    *,
    retention_hours: Optional[float] = None,
    retention_days: Optional[float] = None,
) -> Tuple[int, int]:
    """
    Export stale captures to CSV, delete those images, then purge old visit logs.

    ``retention_hours`` is accepted for launcher compatibility (hours → days).
    """
    if retention_days is None:
        if retention_hours is not None:
            retention_days = float(retention_hours) / 24.0
        else:
            retention_days = CAPTURE_RETENTION_DAYS

    stale = collect_stale_capture_images(retention_days=retention_days)
    csv_path: Optional[Path] = None
    if stale:
        try:
            csv_path = export_capture_images_csv(stale, db_path=db_path)
            if csv_path is not None:
                log.info(
                    "Exported %d capture image(s) to %s",
                    len(stale),
                    csv_path,
                )
        except Exception as exc:
            log.warning(
                "CSV export failed — skipping image/log delete this cycle: %s",
                exc,
            )
            return 0, 0

    files = purge_stale_capture_images(stale, retention_days=retention_days)
    rows = purge_old_tracking_logs(db_path, retention_days=retention_days)
    if rows or files:
        log.info(
            "Retention cleanup: removed %d tracking log(s), %d capture image(s) "
            "(older than %.0f days)%s",
            rows,
            files,
            retention_days,
            f"; CSV {csv_path.name}" if csv_path else "",
        )
        try:
            from purge_notify import emit_database_purged

            emit_database_purged("clear_all")
        except Exception as exc:
            log.warning("Failed to notify dashboard after purge: %s", exc)
    else:
        log.debug(
            "Retention cleanup: nothing to remove (retention %.0f days)",
            retention_days,
        )
    return rows, files


def _cleanup_worker(
    stop_event: threading.Event,
    *,
    db_path: Path,
    interval_sec: float,
    retention_hours: float,
) -> None:
    log.info(
        "Storage cleanup daemon started (every %.1f h, capture retention %.0f days)",
        interval_sec / 3600.0,
        retention_hours / 24.0,
    )
    while not stop_event.is_set():
        try:
            run_cleanup_cycle(db_path, retention_hours=retention_hours)
        except Exception as exc:
            log.warning("Storage cleanup cycle failed: %s", exc)

        if stop_event.wait(interval_sec):
            break

    log.info("Storage cleanup daemon stopped")


def start_storage_cleanup_daemon(
    *,
    db_path: Path = DEFAULT_DB_PATH,
    interval_sec: float = CLEANUP_INTERVAL_SEC,
    retention_hours: float = RETENTION_HOURS,
    run_initial: bool = False,
) -> Tuple[threading.Thread, threading.Event]:
    """
    Start the background cleanup thread (daemon=True).
    Returns (thread, stop_event) so the launcher can signal shutdown.
    """
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_cleanup_worker,
        kwargs={
            "stop_event": stop_event,
            "db_path": db_path,
            "interval_sec": interval_sec,
            "retention_hours": retention_hours,
        },
        name="storage-cleanup",
        daemon=True,
    )
    thread.start()
    if run_initial:
        try:
            run_cleanup_cycle(db_path, retention_hours=retention_hours)
        except Exception as exc:
            log.warning("Initial storage cleanup failed: %s", exc)
    return thread, stop_event


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export capture images older than N days to CSV, then delete them."
    )
    parser.add_argument(
        "--days",
        type=float,
        default=CAPTURE_RETENTION_DAYS,
        help="Retention window in days (default: 14)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Write CSV of stale captures without deleting",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    stale = collect_stale_capture_images(retention_days=args.days)
    print(f"Stale capture files: {len(stale)} (older than {args.days:.0f} days)")
    if args.export_only:
        if stale:
            csv_path = export_capture_images_csv(stale)
            print(f"CSV: {csv_path}")
        return 0
    rows, files = run_cleanup_cycle(retention_days=args.days)
    print(f"Deleted {files} image(s) and {rows} tracking log(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
