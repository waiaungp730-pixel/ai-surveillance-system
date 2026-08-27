"""
Flask application entry point.

Detection routes (including ``/api/detect_smoking``) and all dashboard APIs are
implemented in ``dashboard.py``. This module re-exports the shared ``app`` instance
for tooling that expects ``app.py``.

Live face detections stream over SSE at ``/api/detection-stream``
(``text/event-stream``). Face crops are served from ``static/crops/``.
"""
import os

from dashboard import app, socketio

__all__ = ["app", "socketio"]

if __name__ == "__main__":
    host = (os.environ.get("DASHBOARD_HOST") or "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.environ.get("DASHBOARD_PORT") or "5000")
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
