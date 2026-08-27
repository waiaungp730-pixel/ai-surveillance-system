"""
Local security bootstrap — secrets from env or a gitignored file (never hardcoded).

Creates ``.local_secrets.json`` on first run when env vars are missing.
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit

PROJECT_DIR = Path(__file__).resolve().parent
SECRETS_PATH = PROJECT_DIR / ".local_secrets.json"

_BURNED_PASSWORDS = frozenset(
    {
        "74821433@Pwa",
        "tactical_command_center_secure_key_2026",
        "orion-local-alert",
    }
)


def redact_media_url(url: str) -> str:
    """Strip userinfo from rtsp/http URLs for safe logging."""
    text = str(url or "").strip()
    if not text or "@" not in text:
        return text
    try:
        parts = urlsplit(text)
        if not parts.scheme or not parts.netloc or "@" not in parts.netloc:
            return text
        host = parts.netloc.rsplit("@", 1)[-1]
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    except Exception:
        return "***"


def _load_secrets_file() -> Dict[str, Any]:
    if not SECRETS_PATH.is_file():
        return {}
    try:
        data = json.loads(SECRETS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_secrets_file(data: Dict[str, Any]) -> None:
    try:
        SECRETS_PATH.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        try:
            os.chmod(SECRETS_PATH, 0o600)
        except OSError:
            pass
    except OSError as exc:
        print(f"[security] WARNING: could not write {SECRETS_PATH}: {exc}", file=sys.stderr)


def resolve_dashboard_secrets() -> Tuple[str, str, str, str]:
    """
    Returns (flask_secret, admin_user, admin_password, internal_alert_token).

    Never falls back to burned-in source defaults. Persists generated values
    to ``.local_secrets.json`` (gitignored).
    """
    stored = _load_secrets_file()
    dirty = False

    def _pick(env_name: str, store_key: str, *, generator) -> str:
        nonlocal dirty
        env_val = (os.environ.get(env_name) or "").strip()
        if env_val and env_val not in _BURNED_PASSWORDS:
            return env_val
        file_val = str(stored.get(store_key) or "").strip()
        if file_val and file_val not in _BURNED_PASSWORDS:
            return file_val
        value = str(generator())
        stored[store_key] = value
        dirty = True
        return value

    flask_secret = _pick(
        "FLASK_SECRET_KEY",
        "flask_secret_key",
        generator=lambda: secrets.token_urlsafe(48),
    )
    admin_user = (os.environ.get("DASHBOARD_ADMIN_USER") or "").strip() or str(
        stored.get("admin_user") or "admin"
    )
    if "admin_user" not in stored:
        stored["admin_user"] = admin_user
        dirty = True

    admin_password = _pick(
        "DASHBOARD_ADMIN_PASSWORD",
        "admin_password",
        generator=lambda: secrets.token_urlsafe(18),
    )
    internal_token = _pick(
        "INTERNAL_ALERT_TOKEN",
        "internal_alert_token",
        generator=lambda: secrets.token_urlsafe(32),
    )

    if dirty:
        _save_secrets_file(stored)
        print(
            f"[security] Wrote local secrets to {SECRETS_PATH.name} "
            "(gitignored — set env vars for production).",
            file=sys.stderr,
        )
        if not (os.environ.get("DASHBOARD_ADMIN_PASSWORD") or "").strip():
            print(
                f"[security] Dashboard login: user={admin_user!r} "
                f"password={admin_password!r}",
                file=sys.stderr,
            )

    # Export so tracker subprocesses / live_detection_hub see the same token.
    os.environ.setdefault("FLASK_SECRET_KEY", flask_secret)
    os.environ.setdefault("DASHBOARD_ADMIN_USER", admin_user)
    os.environ.setdefault("DASHBOARD_ADMIN_PASSWORD", admin_password)
    os.environ.setdefault("INTERNAL_ALERT_TOKEN", internal_token)
    return flask_secret, admin_user, admin_password, internal_token


def scan_root_allowlist() -> List[Path]:
    """Directories allowed for folder/video scan APIs."""
    roots: List[Path] = [PROJECT_DIR]
    # Common local footage roots (present on this machine).
    for candidate in (
        Path(r"D:\CCTV_Footage"),
        Path(r"E:\CCTV_Footage"),
        Path(r"D:\msa"),
        Path(r"E:\msa"),
    ):
        try:
            if candidate.is_dir():
                roots.append(candidate.resolve())
        except OSError:
            pass
    raw = (os.environ.get("SCAN_ROOT_ALLOWLIST") or "").strip()
    for part in raw.split(os.pathsep if os.name == "nt" and ";" in raw else ","):
        part = part.strip().strip('"').strip("'")
        if not part:
            continue
        try:
            roots.append(Path(part).expanduser().resolve())
        except OSError:
            roots.append(Path(part).expanduser())
    # Deduplicate
    out: List[Path] = []
    seen = set()
    for r in roots:
        key = str(r).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def assert_scan_path_allowed(
    path: Path,
    *,
    storage_mode: str = "local",
) -> Path:
    """
    Resolve ``path`` and ensure it stays under an allowlisted root.

    Server/UNC mode still requires SCAN_ROOT_ALLOWLIST entries for remote roots.
    """
    mode = (storage_mode or "local").strip().lower()
    try:
        resolved = path if mode == "server" else path.resolve()
    except OSError:
        resolved = path

    allow = scan_root_allowlist()
    if mode == "server":
        # UNC / mapped shares must be explicitly listed.
        if not (os.environ.get("SCAN_ROOT_ALLOWLIST") or "").strip():
            raise PermissionError(
                "SCAN_ROOT_ALLOWLIST is required for server/UNC folder scans"
            )

    resolved_s = str(resolved)
    for root in allow:
        try:
            root_s = str(root.resolve() if mode != "server" else root)
        except OSError:
            root_s = str(root)
        if resolved_s == root_s or resolved_s.startswith(root_s + os.sep) or resolved_s.startswith(
            root_s + "/"
        ):
            return resolved
        # Windows case-insensitive
        if os.name == "nt" and (
            resolved_s.lower() == root_s.lower()
            or resolved_s.lower().startswith(root_s.lower() + "\\")
        ):
            return resolved

    raise PermissionError(
        f"Path not in SCAN_ROOT_ALLOWLIST: {resolved}. "
        f"Allowed roots: {', '.join(str(r) for r in allow)}"
    )


def passwords_match(provided: str, expected: str) -> bool:
    """Constant-time password compare."""
    try:
        return secrets.compare_digest(str(provided), str(expected))
    except (TypeError, ValueError):
        return False
