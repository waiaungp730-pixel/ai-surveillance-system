"""
Notify the dashboard process to broadcast a Socket.IO purge event.

The storage cleanup daemon runs inside run_system.py while Socket.IO lives in
dashboard.py (separate subprocess). This module POSTs to a localhost-only
internal route that performs socketio.emit('database_purged', ...).
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request

log = logging.getLogger("purge_notify")

DEFAULT_NOTIFY_URL = os.environ.get(
    "DASHBOARD_PURGE_NOTIFY_URL",
    "http://127.0.0.1:5000/internal/database-purged",
)


def emit_database_purged(message: str = "clear_all") -> bool:
    """Ask dashboard.py to broadcast database_purged to all connected clients."""
    payload = json.dumps({"message": message}).encode("utf-8")
    req = urllib.request.Request(
        DEFAULT_NOTIFY_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            ok = 200 <= resp.status < 300
            if ok:
                log.info("Dashboard notified: database_purged (%s)", message)
            return ok
    except urllib.error.HTTPError as exc:
        log.warning(
            "Dashboard purge notify HTTP %s (%s)",
            exc.code,
            DEFAULT_NOTIFY_URL,
        )
    except Exception as exc:
        log.debug("Dashboard purge notify failed (%s): %s", DEFAULT_NOTIFY_URL, exc)
    return False
