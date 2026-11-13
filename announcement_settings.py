"""
Announcement (Telegram alert) schedule — configurable from the dashboard Settings page.

Settings are stored in ``announcement_settings.json`` at the project root so both
the dashboard and the tracker process can read them without a restart.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

log = logging.getLogger("announcement_settings")

PROJECT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = PROJECT_DIR / "announcement_settings.json"

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_lock = threading.Lock()
_cache: Optional[Tuple[float, Dict[str, Any]]] = None
_CACHE_TTL_SEC = 2.0


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None and name == "TELEGRAM_ALERT_NON_WATCHLIST":
        raw = os.environ.get("TELEGRAM_ALERT_STRANGERS")
    if raw is None:
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _default_telegram_alert_flags() -> Dict[str, bool]:
    return {
        "telegram_alert_strangers": _env_bool("TELEGRAM_ALERT_NON_WATCHLIST", True),
        "telegram_alert_watchlist": _env_bool("TELEGRAM_ALERT_WATCHLIST", False),
        "telegram_alert_named": _env_bool("TELEGRAM_ALERT_NAMED", False),
    }


def _default_settings() -> Dict[str, Any]:
    base = {
        "schedule_enabled": True,
        "start_time": os.environ.get("ANNOUNCEMENT_START_TIME", "08:00").strip() or "08:00",
        "end_time": os.environ.get("ANNOUNCEMENT_END_TIME", "18:00").strip() or "18:00",
    }
    base.update(_default_telegram_alert_flags())
    return base


def _parse_hhmm(raw: Any) -> Optional[time]:
    text = str(raw or "").strip()
    m = _TIME_RE.match(text)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return None
    return time(hour, minute)


def _normalize_settings(data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base = _default_settings()
    src = data if isinstance(data, dict) else {}
    out = dict(base)
    out["schedule_enabled"] = bool(src.get("schedule_enabled", base["schedule_enabled"]))
    start = _parse_hhmm(src.get("start_time", base["start_time"]))
    end = _parse_hhmm(src.get("end_time", base["end_time"]))
    out["start_time"] = start.strftime("%H:%M") if start else base["start_time"]
    out["end_time"] = end.strftime("%H:%M") if end else base["end_time"]
    flags = _default_telegram_alert_flags()
    out["telegram_alert_strangers"] = bool(
        src.get("telegram_alert_strangers", flags["telegram_alert_strangers"])
    )
    out["telegram_alert_watchlist"] = bool(
        src.get("telegram_alert_watchlist", flags["telegram_alert_watchlist"])
    )
    out["telegram_alert_named"] = bool(
        src.get("telegram_alert_named", flags["telegram_alert_named"])
    )
    return out


def load_announcement_settings(*, use_cache: bool = True) -> Dict[str, Any]:
    global _cache
    now_ts = datetime.now().timestamp()
    if use_cache and _cache is not None:
        cached_at, cached = _cache
        if now_ts - cached_at <= _CACHE_TTL_SEC:
            return dict(cached)

    with _lock:
        if use_cache and _cache is not None:
            cached_at, cached = _cache
            if now_ts - cached_at <= _CACHE_TTL_SEC:
                return dict(cached)

        settings = _default_settings()
        try:
            if SETTINGS_PATH.is_file():
                # utf-8-sig tolerates a BOM from editors / PowerShell Set-Content.
                raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))
                settings = _normalize_settings(raw)
        except Exception as exc:
            log.warning("Failed to read %s: %s", SETTINGS_PATH, exc)

        if use_cache:
            _cache = (now_ts, dict(settings))
        return settings


def save_announcement_settings(data: Dict[str, Any]) -> Dict[str, Any]:
    global _cache
    normalized = _normalize_settings(data)
    payload = {
        "schedule_enabled": normalized["schedule_enabled"],
        "start_time": normalized["start_time"],
        "end_time": normalized["end_time"],
        "telegram_alert_strangers": normalized["telegram_alert_strangers"],
        "telegram_alert_watchlist": normalized["telegram_alert_watchlist"],
        "telegram_alert_named": normalized["telegram_alert_named"],
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    with _lock:
        SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _cache = (datetime.now().timestamp(), dict(normalized))
    return normalized


def _time_in_window(now_t: time, start_t: time, end_t: time) -> bool:
    """True when ``now_t`` falls inside [start, end), including overnight windows."""
    if start_t == end_t:
        return True
    if start_t < end_t:
        return start_t <= now_t < end_t
    return now_t >= start_t or now_t < end_t


def announcements_allowed_now(now: Optional[datetime] = None) -> bool:
    """Return False when the current time is outside the configured window."""
    return captures_allowed_now(now)


def captures_allowed_now(now: Optional[datetime] = None) -> bool:
    """Same window gates live face/scene/plate saves and Telegram announcements."""
    settings = load_announcement_settings()
    if not settings.get("schedule_enabled", True):
        return True

    start_t = _parse_hhmm(settings.get("start_time"))
    end_t = _parse_hhmm(settings.get("end_time"))
    if start_t is None or end_t is None:
        return True

    current = now or datetime.now()
    return _time_in_window(current.time().replace(second=0, microsecond=0), start_t, end_t)


def _schedule_window_label(settings: Dict[str, Any]) -> str:
    return f"{_format_time_12h(settings.get('start_time'))} – {_format_time_12h(settings.get('end_time'))}"


def _format_time_12h(raw: Any) -> str:
    """Display ``HH:MM`` as ``8:00 AM`` / ``6:30 PM``."""
    parsed = _parse_hhmm(raw)
    if parsed is None:
        return str(raw or "")
    hour24 = parsed.hour
    minute = parsed.minute
    ampm = "AM" if hour24 < 12 else "PM"
    hour12 = hour24 % 12
    if hour12 == 0:
        hour12 = 12
    if minute:
        return f"{hour12}:{minute:02d} {ampm}"
    return f"{hour12} {ampm}"


def telegram_alert_strangers_enabled() -> bool:
    """Unknown visitors / non-watchlist incidents may trigger Telegram."""
    return bool(load_announcement_settings().get("telegram_alert_strangers", True))


def telegram_alert_watchlist_enabled() -> bool:
    """Enrolled watchlist / POI matches may trigger Telegram."""
    return bool(load_announcement_settings().get("telegram_alert_watchlist", False))


def telegram_alert_named_enabled() -> bool:
    """Named gallery matches may trigger Telegram."""
    return bool(load_announcement_settings().get("telegram_alert_named", False))


def announcement_schedule_status(now: Optional[datetime] = None) -> Dict[str, Any]:
    """Human-readable schedule state for the dashboard."""
    settings = load_announcement_settings()
    current = now or datetime.now()
    enabled = bool(settings.get("schedule_enabled", True))
    start_text = str(settings.get("start_time") or "08:00")
    end_text = str(settings.get("end_time") or "18:00")
    start_label = _format_time_12h(start_text)
    end_label = _format_time_12h(end_text)
    allowed = announcements_allowed_now(current)
    if not enabled:
        return {
            "schedule_enabled": False,
            "start_time": start_text,
            "end_time": end_text,
            "start_label": start_label,
            "end_label": end_label,
            "active_now": True,
            "label": "24/7 faces & plates — schedule off",
            "captures_active": True,
        }
    window = f"{start_label} – {end_label}"
    if allowed:
        label = f"Capturing faces & plates · {window}"
    else:
        label = f"Faces & plates paused · resumes {start_label}"
    return {
        "schedule_enabled": True,
        "start_time": start_text,
        "end_time": end_text,
        "start_label": start_label,
        "end_label": end_label,
        "active_now": allowed,
        "captures_active": allowed,
        "label": label,
    }
