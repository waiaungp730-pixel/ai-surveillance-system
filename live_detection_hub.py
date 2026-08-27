"""
Cross-process live face-detection event bus.

main.py (tracker) publishes events; dashboard.py streams them over SSE.
Persisted as a JSONL file + face crops under static/crops/ so separate
subprocesses can share detections without shared memory.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
CROPS_DIR = PROJECT_DIR / "static" / "crops"
EVENTS_PATH = PROJECT_DIR / "live_preview" / "detection_events.jsonl"
EVENTS_META_PATH = PROJECT_DIR / "live_preview" / "detection_events.meta.json"
ALERTS_PATH = PROJECT_DIR / "live_preview" / "person_alerts.jsonl"
ALERTS_META_PATH = PROJECT_DIR / "live_preview" / "person_alerts.meta.json"
FACE_BOXES_PATH = PROJECT_DIR / "live_preview" / "face_boxes.json"
MAX_EVENTS_ON_DISK = 300
MAX_ALERTS_ON_DISK = 120
# Gallery / side-panel only shows captures at or above this confidence.
HIGH_CONFIDENCE_THRESHOLD = 50.0
# Instant WebSocket POI / watchlist alert threshold (match %).
POI_ALERT_MIN_CONFIDENCE = 75.0


def _env_alert_min_confidence(default: float = 50.0) -> float:
    raw = (
        os.environ.get("TELEGRAM_ALERT_MIN_PERCENT")
        or os.environ.get("WATCHLIST_ALERT_MIN_PERCENT")
        or ""
    ).strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value > 1.0:
        value = value / 100.0
    return max(0.0, min(100.0, value))


BLACKLIST_ALERT_MIN_CONFIDENCE = _env_alert_min_confidence(50.0)
_WRITE_LOCK = threading.Lock()
_ALERT_LOCK = threading.Lock()
_BOX_LOCK = threading.Lock()
_last_alert_emit: Dict[str, float] = {}
ALERT_DEDUPE_SEC = 2.0
# camera_key -> {boxes: [[x1,y1,x2,y2],...], width, height, updated}
_FACE_BOXES: Dict[str, Dict[str, Any]] = {}


def _safe_token(value: str, limit: int = 32) -> str:
    safe = re.sub(r"[^\w\-]+", "_", str(value or "").strip())
    return (safe[:limit] if safe else "cam")


def ensure_live_detection_dirs() -> None:
    CROPS_DIR.mkdir(parents=True, exist_ok=True)
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)


def _trim_alerts_file() -> None:
    if not ALERTS_PATH.is_file():
        return
    try:
        lines = ALERTS_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) <= MAX_ALERTS_ON_DISK:
        return
    keep = lines[-MAX_ALERTS_ON_DISK:]
    tmp = ALERTS_PATH.with_suffix(".jsonl.tmp")
    try:
        tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
        tmp.replace(ALERTS_PATH)
    except OSError:
        pass


def _alert_dedupe_key(camera: str, known_person_id: Optional[int], name: str) -> str:
    kid = known_person_id if known_person_id is not None else name
    return f"{camera}|{kid}"


def _should_emit_alert(camera: str, known_person_id: Optional[int], name: str) -> bool:
    key = _alert_dedupe_key(camera, known_person_id, name)
    now = time.time()
    with _ALERT_LOCK:
        last = _last_alert_emit.get(key, 0.0)
        if now - last < ALERT_DEDUPE_SEC:
            return False
        _last_alert_emit[key] = now
        # Bound memory
        if len(_last_alert_emit) > 400:
            cutoff = now - 60.0
            stale = [k for k, t in _last_alert_emit.items() if t < cutoff]
            for k in stale:
                _last_alert_emit.pop(k, None)
        return True


def _is_auto_capture_name(name: str) -> bool:
    """True for temporary gallery labels that must not open a red watchlist card."""
    n = str(name or "").strip()
    return (
        n.startswith("Passerby_")
        or n.startswith("Stranger_")
        or n.startswith("Unknown")
        or n in ("", "Person of Interest", "Unknown Visitor")
    )


def classify_person_alert(
    *,
    known_person_id: Optional[int],
    confidence: float,
    force: bool = False,
    name: Optional[str] = None,
) -> Optional[str]:
    """
    Return alert_type ('blacklist' | 'poi') for instant dashboard WebSocket alerts.

    Red WATCHLIST MATCH cards require a real gallery person id — never fire on
    confidence alone (that was creating Passerby / back-of-head red boxes).
    """
    try:
        conf = float(confidence)
    except (TypeError, ValueError):
        conf = 0.0
    if not force:
        if known_person_id is None:
            return None
        if name is not None and _is_auto_capture_name(name):
            return None
    if force:
        return "blacklist" if conf >= BLACKLIST_ALERT_MIN_CONFIDENCE else "poi"
    if conf >= BLACKLIST_ALERT_MIN_CONFIDENCE:
        return "blacklist"
    if conf >= POI_ALERT_MIN_CONFIDENCE:
        return "poi"
    return None


def _http_push_person_alert(event: Dict[str, Any]) -> None:
    """Best-effort POST into the dashboard process for zero-latency Socket.IO emit."""
    import os
    import urllib.error
    import urllib.request

    port = str(os.environ.get("DASHBOARD_PORT", "5000")).strip() or "5000"
    token = str(os.environ.get("INTERNAL_ALERT_TOKEN") or "").strip()
    if not token:
        try:
            from security_bootstrap import resolve_dashboard_secrets

            token = resolve_dashboard_secrets()[3]
        except Exception:
            token = ""
    url = f"http://127.0.0.1:{port}/api/internal/person_alert"
    try:
        body = json.dumps(event, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Internal-Alert-Token": token,
            },
        )
        with urllib.request.urlopen(req, timeout=0.6) as resp:
            _ = resp.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        pass


def publish_person_alert(
    *,
    camera: str,
    name: str,
    confidence: float,
    face_bgr: Optional[np.ndarray] = None,
    face_crop_url: Optional[str] = None,
    person_id: Optional[int] = None,
    known_person_id: Optional[int] = None,
    timestamp: Optional[str] = None,
    bbox: Optional[Sequence[float]] = None,
    alert_type: Optional[str] = None,
    force: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Persist + push a Person-of-Interest / watchlist alert for instant WebSocket delivery.
    Returns the alert payload, or None when deduped / not an alert.
    """
    kind = alert_type or classify_person_alert(
        known_person_id=known_person_id,
        confidence=confidence,
        force=force,
        name=name,
    )
    if not kind and not force:
        return None
    if force and not kind:
        kind = "blacklist" if float(confidence or 0) >= BLACKLIST_ALERT_MIN_CONFIDENCE else "poi"

    cam = str(camera or "CAM")
    display = str(name or "Person of Interest")
    # Never show the red watchlist card for auto passerby / stranger labels.
    if _is_auto_capture_name(display):
        return None
    if not force and known_person_id is None:
        return None
    if not _should_emit_alert(cam, known_person_id, display):
        return None

    ensure_live_detection_dirs()
    now = datetime.now()
    crop_url = face_crop_url
    if not crop_url and face_bgr is not None:
        crop_url = save_detection_crop(
            face_bgr, camera=cam, person_id=person_id or known_person_id, person_name=display
        )

    bbox_out: Optional[List[int]] = None
    if bbox is not None and len(bbox) >= 4:
        try:
            bbox_out = [int(round(float(v))) for v in bbox[:4]]
        except (TypeError, ValueError):
            bbox_out = None

    try:
        conf = float(round(float(confidence), 1))
    except (TypeError, ValueError):
        conf = 0.0

    event: Dict[str, Any] = {
        "event": "new_person_alert",
        "camera": cam,
        "name": display,
        "confidence": conf,
        "timestamp": timestamp or now.strftime("%H:%M:%S"),
        "face_crop_url": crop_url,
        "person_id": person_id,
        "known_person_id": known_person_id,
        "iso_timestamp": now.isoformat(timespec="seconds"),
        "id": uuid.uuid4().hex[:12],
        "bbox": bbox_out,
        "is_poi": True,
        "alert_type": kind,
        "blacklist": kind == "blacklist",
    }

    line = json.dumps(event, ensure_ascii=False)
    with _WRITE_LOCK:
        try:
            with ALERTS_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            if int(time.time()) % 11 == 0:
                _trim_alerts_file()
        except OSError:
            pass
        try:
            ALERTS_META_PATH.write_text(
                json.dumps({"updated": time.time(), "last_id": event["id"]}),
                encoding="utf-8",
            )
        except OSError:
            pass

    # Non-blocking HTTP push so tracker inference is not delayed.
    threading.Thread(
        target=_http_push_person_alert,
        args=(event,),
        name="poi-alert-push",
        daemon=True,
    ).start()
    return event


def read_alerts_since(
    byte_offset: int = 0,
    *,
    limit: int = 40,
) -> Tuple[List[Dict[str, Any]], int]:
    """Read POI alert JSONL rows after ``byte_offset``."""
    ensure_live_detection_dirs()
    if not ALERTS_PATH.is_file():
        return [], 0
    events: List[Dict[str, Any]] = []
    try:
        size = ALERTS_PATH.stat().st_size
        if byte_offset < 0 or byte_offset > size:
            byte_offset = 0
        with ALERTS_PATH.open("r", encoding="utf-8") as fh:
            fh.seek(byte_offset)
            if byte_offset > 0:
                _ = fh.readline()
            while len(events) < limit:
                line = fh.readline()
                if not line:
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    events.append(payload)
            new_offset = fh.tell()
    except OSError:
        return [], byte_offset
    return events, new_offset


def save_detection_crop(
    face_bgr: np.ndarray,
    *,
    camera: str,
    person_id: Optional[int] = None,
    person_name: Optional[str] = None,
) -> Optional[str]:
    """Persist a face crop and return its public URL path."""
    if face_bgr is None or getattr(face_bgr, "size", 0) == 0:
        return None
    frame = np.asarray(face_bgr)
    if frame.ndim != 3 or frame.shape[2] < 3:
        return None
    # Guard against accidental single-channel / empty tensors masquerading as crops.
    if int(frame.shape[0]) < 8 or int(frame.shape[1]) < 8:
        return None
    ensure_live_detection_dirs()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = person_id if person_id is not None else "unknown"
    slug = ""
    try:
        from captures import capture_filename_slug

        token = capture_filename_slug(person_name, pid)
        if token and str(token) != str(pid):
            slug = f"{token}_"
    except Exception:
        slug = ""
    filename = f"{ts}_{_safe_token(camera)}_{slug}{pid}_{uuid.uuid4().hex[:8]}.jpg"
    out = CROPS_DIR / filename
    bgr = np.ascontiguousarray(frame[:, :, :3])
    try:
        from captures import prepare_face_thumbnail

        bgr = prepare_face_thumbnail(bgr, enhance=False)
    except Exception:
        pass
    ok = cv2.imwrite(
        str(out),
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )
    if not ok or not out.is_file() or out.stat().st_size < 64:
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
        return None
    return f"/static/crops/{filename}"


def publish_face_boxes(
    camera: str,
    boxes: Sequence[Sequence[float]],
    *,
    frame_width: int,
    frame_height: int,
) -> None:
    """
    Publish latest face bounding boxes for a camera (pixel xyxy).
    Used by /api/face_detection_stream to draw green overlays.
    """
    cam = str(camera or "CAM").strip() or "CAM"
    cleaned: List[List[int]] = []
    for box in boxes or []:
        try:
            x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
        except (TypeError, ValueError, IndexError):
            continue
        if x2 > x1 and y2 > y1:
            cleaned.append([x1, y1, x2, y2])
    payload = {
        "camera": cam,
        "boxes": cleaned,
        "width": int(max(1, frame_width)),
        "height": int(max(1, frame_height)),
        "updated": time.time(),
    }
    with _BOX_LOCK:
        _FACE_BOXES[cam] = payload
        # Also index by common id aliases (cameras.json uses "1"/"2").
        _FACE_BOXES[_safe_token(cam)] = payload
        try:
            ensure_live_detection_dirs()
            FACE_BOXES_PATH.write_text(
                json.dumps(_FACE_BOXES, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:
            pass


def get_face_boxes(camera: str) -> Dict[str, Any]:
    """Return latest face-box payload for a camera key/location."""
    key = str(camera or "").strip()
    with _BOX_LOCK:
        if key in _FACE_BOXES:
            return dict(_FACE_BOXES[key])
        alt = _safe_token(key)
        if alt in _FACE_BOXES:
            return dict(_FACE_BOXES[alt])
        for cam_key, payload in _FACE_BOXES.items():
            if str(cam_key).lower() == key.lower():
                return dict(payload)
    # Disk fallback (cross-process)
    try:
        if FACE_BOXES_PATH.is_file():
            data = json.loads(FACE_BOXES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                if key in data and isinstance(data[key], dict):
                    return dict(data[key])
                tok = _safe_token(key)
                if tok in data and isinstance(data[tok], dict):
                    return dict(data[tok])
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return {"camera": key, "boxes": [], "width": 0, "height": 0, "updated": 0}


def _trim_events_file() -> None:
    if not EVENTS_PATH.is_file():
        return
    try:
        lines = EVENTS_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) <= MAX_EVENTS_ON_DISK:
        return
    keep = lines[-MAX_EVENTS_ON_DISK:]
    tmp = EVENTS_PATH.with_suffix(".jsonl.tmp")
    try:
        tmp.write_text("\n".join(keep) + "\n", encoding="utf-8")
        tmp.replace(EVENTS_PATH)
    except OSError:
        pass


def publish_stranger_watchlist_alert(
    *,
    camera: str,
    confidence: float,
    face_bgr: Optional[np.ndarray] = None,
    face_crop_url: Optional[str] = None,
    person_id: Optional[int] = None,
    known_person_id: Optional[int] = None,
    timestamp: Optional[str] = None,
    bbox: Optional[Sequence[float]] = None,
) -> Optional[Dict[str, Any]]:
    """Push an unknown visitor to the dashboard stranger watchlist (not Named People)."""
    cam = str(camera or "CAM")
    if not _should_emit_alert(cam, known_person_id or person_id, "Stranger"):
        return None

    ensure_live_detection_dirs()
    now = datetime.now()
    crop_url = face_crop_url
    if not crop_url and face_bgr is not None:
        crop_url = save_detection_crop(
            face_bgr,
            camera=cam,
            person_id=person_id or known_person_id,
            person_name="Stranger",
        )
    if not crop_url:
        return None

    bbox_out: Optional[List[int]] = None
    if bbox is not None and len(bbox) >= 4:
        try:
            bbox_out = [int(round(float(v))) for v in bbox[:4]]
        except (TypeError, ValueError):
            bbox_out = None

    try:
        conf = float(round(float(confidence), 1))
    except (TypeError, ValueError):
        conf = 0.0

    event: Dict[str, Any] = {
        "event": "new_person_alert",
        "camera": cam,
        "name": "Stranger",
        "confidence": conf,
        "timestamp": timestamp or now.strftime("%H:%M:%S"),
        "face_crop_url": crop_url,
        "person_id": person_id,
        "known_person_id": known_person_id,
        "iso_timestamp": now.isoformat(timespec="seconds"),
        "id": uuid.uuid4().hex[:12],
        "bbox": bbox_out,
        "is_stranger": True,
        "is_named": False,
        "is_watchlist": True,
        "alert_type": "stranger",
        "blacklist": False,
    }

    line = json.dumps(event, ensure_ascii=False)
    with _WRITE_LOCK:
        try:
            with ALERTS_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass

    threading.Thread(
        target=_http_push_person_alert,
        args=(event,),
        name="stranger-watchlist-push",
        daemon=True,
    ).start()
    return event


def publish_detection_event(
    *,
    camera: str,
    name: str,
    confidence: float,
    face_bgr: Optional[np.ndarray] = None,
    face_crop_url: Optional[str] = None,
    person_id: Optional[int] = None,
    known_person_id: Optional[int] = None,
    timestamp: Optional[str] = None,
    bbox: Optional[Sequence[float]] = None,
    is_stranger: Optional[bool] = None,
    is_named: Optional[bool] = None,
    matched_named_id: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Publish one live detection for SSE consumers.
    Returns the event dict (always includes camera/name/confidence/timestamp).
    """
    if face_bgr is not None:
        try:
            from face_print_filter import (
                is_junk_person_plate_reason,
                is_likely_invalid_face_crop,
                is_likely_non_face_object,
                is_likely_rear_view,
                is_likely_wood_post_or_rail,
            )

            catch_every = os.environ.get("CATCH_EVERY_PERSON", "1").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            catch_backs = os.environ.get("CATCH_BACKS", "0").strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            rear, rear_reason = is_likely_rear_view(face_bgr)
            if rear and not catch_backs:
                return {}
            if catch_every:
                wood, _ = is_likely_wood_post_or_rail(face_bgr)
                if wood:
                    return {}
                bad, reason = is_likely_non_face_object(face_bgr)
                if bad and is_junk_person_plate_reason(reason):
                    return {}
            else:
                bad, _reason = is_likely_invalid_face_crop(face_bgr)
                if bad:
                    return {}
        except Exception:
            pass
    ensure_live_detection_dirs()
    now = datetime.now()
    crop_url = face_crop_url
    if not crop_url and face_bgr is not None:
        crop_url = save_detection_crop(
            face_bgr, camera=camera, person_id=person_id, person_name=name
        )

    bbox_out: Optional[List[int]] = None
    if bbox is not None and len(bbox) >= 4:
        try:
            bbox_out = [int(round(float(v))) for v in bbox[:4]]
        except (TypeError, ValueError):
            bbox_out = None

    event: Dict[str, Any] = {
        "camera": str(camera or "CAM"),
        "name": str(name or "Unknown Visitor"),
        "confidence": float(round(float(confidence), 1)),
        "timestamp": timestamp or now.strftime("%H:%M:%S"),
        "face_crop_url": crop_url,
        "person_id": person_id,
        "known_person_id": known_person_id,
        "matched_named_id": matched_named_id,
        "iso_timestamp": now.isoformat(timespec="seconds"),
        "id": uuid.uuid4().hex[:12],
        "bbox": bbox_out,
    }
    if is_stranger is not None:
        event["is_stranger"] = bool(is_stranger)
    if is_named is not None:
        event["is_named"] = bool(is_named)

    stranger_hit = bool(is_stranger) and not bool(is_named)
    named_hit = bool(is_named) and not stranger_hit

    alert_kind = None
    if named_hit:
        event["is_watchlist"] = False
    elif stranger_hit:
        event["is_watchlist"] = True
        event["alert_type"] = "stranger"
    else:
        alert_kind = classify_person_alert(
            known_person_id=known_person_id,
            confidence=event["confidence"],
            name=event["name"],
        )
        if alert_kind:
            event["is_poi"] = True
            event["alert_type"] = alert_kind
            event["blacklist"] = alert_kind == "blacklist"

    line = json.dumps(event, ensure_ascii=False)
    with _WRITE_LOCK:
        try:
            with EVENTS_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            if int(time.time()) % 17 == 0:
                _trim_events_file()
        except OSError:
            pass
        try:
            EVENTS_META_PATH.write_text(
                json.dumps({"updated": time.time(), "last_id": event["id"]}),
                encoding="utf-8",
            )
        except OSError:
            pass

    if stranger_hit:
        publish_stranger_watchlist_alert(
            camera=event["camera"],
            confidence=event["confidence"],
            face_crop_url=crop_url,
            person_id=person_id,
            known_person_id=known_person_id,
            timestamp=event["timestamp"],
            bbox=bbox_out,
        )
    elif alert_kind:
        publish_person_alert(
            camera=event["camera"],
            name=event["name"],
            confidence=event["confidence"],
            face_crop_url=crop_url,
            person_id=person_id,
            known_person_id=known_person_id,
            timestamp=event["timestamp"],
            bbox=bbox_out,
            alert_type=alert_kind,
        )
    return event


def read_events_since(
    byte_offset: int = 0,
    *,
    limit: int = 50,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Read JSONL detection events after ``byte_offset``.
    Returns (events, new_offset).
    """
    ensure_live_detection_dirs()
    if not EVENTS_PATH.is_file():
        return [], 0

    events: List[Dict[str, Any]] = []
    try:
        size = EVENTS_PATH.stat().st_size
        if byte_offset < 0 or byte_offset > size:
            byte_offset = 0
        with EVENTS_PATH.open("r", encoding="utf-8") as fh:
            fh.seek(byte_offset)
            # If we landed mid-line (after truncation), skip to next newline.
            if byte_offset > 0:
                _ = fh.readline()
            while len(events) < limit:
                line = fh.readline()
                if not line:
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    events.append(payload)
            new_offset = fh.tell()
    except OSError:
        return [], byte_offset
    return events, new_offset


def recent_events(limit: int = 24) -> List[Dict[str, Any]]:
    """Return the most recent detection events (newest first)."""
    events, _ = read_events_since(0, limit=10_000)
    if not events:
        return []
    return list(reversed(events[-limit:]))


def _crop_path_from_url(crop_url: str) -> Optional[Path]:
    """Map /static/crops/foo.jpg to on-disk path."""
    if not crop_url:
        return None
    name = str(crop_url).replace("\\", "/").split("/")[-1].strip()
    if not name or not name.lower().endswith((".jpg", ".jpeg", ".png")):
        return None
    path = CROPS_DIR / name
    return path if path.is_file() else None


def _event_face_crop_is_valid(event: Dict[str, Any]) -> bool:
    """Drop gallery tiles for hand/desk/back-of-head crops saved before gating."""
    crop_url = event.get("face_crop_url") or event.get("crop_url")
    path = _crop_path_from_url(str(crop_url or ""))
    if path is None:
        return True
    try:
        from captures import (
            capture_display_is_body_crop,
            capture_display_is_full_frame,
            display_image_is_portrait_body,
        )
        from face_print_filter import is_likely_invalid_face_crop

        if capture_display_is_full_frame():
            return True
        img = cv2.imread(str(path))
        if img is None:
            return True
        if capture_display_is_body_crop():
            return display_image_is_portrait_body(img)
        bad, _reason = is_likely_invalid_face_crop(img)
        return not bad
    except Exception:
        return True


def recent_high_confidence_faces(
    limit: int = 24,
    *,
    min_confidence: float = HIGH_CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Recent detections with a face crop and confidence >= threshold."""
    out: List[Dict[str, Any]] = []
    for event in recent_events(limit=max(limit * 4, 80)):
        conf = event.get("confidence")
        try:
            score = float(conf)
        except (TypeError, ValueError):
            continue
        if score < float(min_confidence):
            continue
        crop = event.get("face_crop_url") or event.get("crop_url")
        if not crop:
            continue
        if not _event_face_crop_is_valid(event):
            continue
        out.append(event)
        if len(out) >= limit:
            break
    return out


def list_captured_faces(
    limit: int = 24,
    *,
    min_confidence: float = HIGH_CONFIDENCE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Captured-faces gallery payload for the Face Detection side panel.
    Prefers SSE event metadata; falls back to raw crop files on disk.
    """
    entries = recent_high_confidence_faces(limit=limit, min_confidence=min_confidence)
    if entries:
        return [
            {
                "id": e.get("id"),
                "name": e.get("name") or "Unknown Visitor",
                "camera": e.get("camera") or "CAM",
                "timestamp": e.get("timestamp") or "",
                "iso_timestamp": e.get("iso_timestamp"),
                "confidence": e.get("confidence"),
                "face_crop_url": e.get("face_crop_url"),
                "person_id": e.get("person_id"),
                "known_person_id": e.get("known_person_id"),
                "bbox": e.get("bbox"),
            }
            for e in entries
        ]

    # Fallback: newest crop files (no confidence metadata).
    ensure_live_detection_dirs()
    files = sorted(
        CROPS_DIR.glob("*.jpg"),
        key=lambda p: p.stat().st_mtime if p.is_file() else 0,
        reverse=True,
    )
    out: List[Dict[str, Any]] = []
    for path in files:
        crop_url = f"/static/crops/{path.name}"
        if not _event_face_crop_is_valid({"face_crop_url": crop_url}):
            continue
        parts = path.stem.split("_")
        camera = parts[2] if len(parts) >= 3 else "CAM"
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        out.append(
            {
                "id": path.stem[-12:],
                "name": "Captured Face",
                "camera": camera,
                "timestamp": mtime.strftime("%H:%M:%S"),
                "iso_timestamp": mtime.isoformat(timespec="seconds"),
                "confidence": None,
                "face_crop_url": crop_url,
                "person_id": None,
                "known_person_id": None,
                "bbox": None,
            }
        )
        if len(out) >= limit:
            break
    return out
