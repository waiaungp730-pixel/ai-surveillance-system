"""Keycloak sign-in for the Cyber Eye dashboard.

Password login stays in place. A CyberHub user with the cyber-eye access role
can enter through /auth/sso/login and land on the existing dashboard session.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlencode

import requests
from flask import Flask, jsonify, redirect, request, session
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

PROJECT_DIR = Path(__file__).resolve().parent
CLIENT_ID = "cyber-eye"
ATTEMPT_COOKIE = "cybereye.attempt"
ATTEMPT_TTL_S = 10 * 60
SESSION_TTL_S = 8 * 60 * 60
IDLE_TTL_S = 30 * 60
LOGOUT_EVENT = "http://schemas.openid.net/event/backchannel-logout"

_store_lock = threading.Lock()
_discovery_cache: dict[str, tuple[float, dict]] = {}


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def sso_settings() -> dict:
    public = (os.environ.get("CYBER_EYE_PUBLIC_URL") or "https://192.168.11.65:5000").rstrip("/")
    return {
        "enabled": _flag("CYBER_EYE_SSO_ENABLED"),
        "public_url": public,
        "issuer": (os.environ.get("KEYCLOAK_ISSUER_URL") or "").rstrip("/"),
        "client_id": (os.environ.get("KEYCLOAK_CLIENT_ID") or CLIENT_ID).strip() or CLIENT_ID,
        "client_secret": os.environ.get("KEYCLOAK_CLIENT_SECRET") or "",
        "allow_insecure_http": _flag("CYBER_EYE_ALLOW_INSECURE_HTTP"),
        "store_path": Path(os.environ.get("CYBER_EYE_SSO_STORE") or (PROJECT_DIR / ".sso_sessions.json")),
    }


def sso_ready(settings: dict | None = None) -> bool:
    settings = settings or sso_settings()
    if not settings["enabled"]:
        return False
    if not settings["issuer"] or not settings["client_secret"]:
        return False
    urls = (settings["issuer"], settings["public_url"])
    if any(url.startswith("http://") for url in urls) and not settings["allow_insecure_http"]:
        return False
    return settings["public_url"].startswith(("https://", "http://"))


def _secure_cookie(settings: dict) -> bool:
    return settings["public_url"].startswith("https://")


def _serializer(secret: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret, salt="cyber-eye-sso-attempt")


def _discovery(issuer: str) -> dict:
    now = time.time()
    cached = _discovery_cache.get(issuer)
    if cached and cached[0] > now:
        return cached[1]
    response = requests.get(
        issuer.rstrip("/") + "/.well-known/openid-configuration",
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json()
    _discovery_cache[issuer] = (now + 300, payload)
    return payload


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def begin_login(settings: dict) -> dict:
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(32)
    nonce = secrets.token_urlsafe(32)
    metadata = _discovery(settings["issuer"])
    query = urlencode(
        {
            "response_type": "code",
            "client_id": settings["client_id"],
            "redirect_uri": settings["public_url"] + "/auth/sso/callback",
            "scope": "openid",
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    return {
        "url": f"{metadata['authorization_endpoint']}?{query}",
        "verifier": verifier,
        "state": state,
        "nonce": nonce,
    }


def _verify_token(token: str, jwks_uri: str, issuer: str, audience: str) -> dict:
    import jwt
    from jwt import PyJWKClient

    key = PyJWKClient(jwks_uri).get_signing_key_from_jwt(token)
    claims = jwt.decode(
        token,
        key.key,
        algorithms=["RS256"],
        issuer=issuer,
        options={"verify_aud": False},
    )
    if not isinstance(claims, dict):
        raise ValueError("claims")
    audiences = claims.get("aud")
    if not isinstance(audiences, list):
        audiences = [audiences]
    if audience not in audiences and claims.get("azp") != audience:
        raise ValueError("audience")
    return claims


def complete_login(settings: dict, code: str, verifier: str, nonce: str) -> dict:
    metadata = _discovery(settings["issuer"])
    response = requests.post(
        metadata["token_endpoint"],
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": settings["public_url"] + "/auth/sso/callback",
            "client_id": settings["client_id"],
            "client_secret": settings["client_secret"],
            "code_verifier": verifier,
        },
        timeout=10,
    )
    response.raise_for_status()
    token = response.json().get("id_token")
    if not isinstance(token, str):
        raise ValueError("missing id token")
    claims = _verify_token(token, metadata["jwks_uri"], settings["issuer"], settings["client_id"])
    if claims.get("nonce") != nonce:
        raise ValueError("nonce")
    username = claims.get("preferred_username") or claims.get("name") or "cyberhub"
    return {
        "issuer": claims["iss"],
        "subject": claims["sub"],
        "username": str(username)[:80],
        "provider_sid": claims.get("sid"),
        "resource_access": claims.get("resource_access") or {},
    }


def has_workspace_access(resource_access: dict, client_id: str) -> bool:
    entry = (resource_access or {}).get(client_id) or {}
    roles = entry.get("roles") if isinstance(entry, dict) else []
    return isinstance(roles, list) and "access" in roles


def end_session_url(settings: dict) -> str:
    metadata = _discovery(settings["issuer"])
    endpoint = metadata.get("end_session_endpoint")
    if not endpoint:
        return settings["public_url"] + "/login"
    query = urlencode(
        {
            "client_id": settings["client_id"],
            "post_logout_redirect_uri": settings["public_url"] + "/login",
        }
    )
    return f"{endpoint}?{query}"


def verify_logout(settings: dict, token: str) -> dict:
    metadata = _discovery(settings["issuer"])
    claims = _verify_token(token, metadata["jwks_uri"], settings["issuer"], settings["client_id"])
    events = claims.get("events") or {}
    if claims.get("nonce") is not None or LOGOUT_EVENT not in events:
        raise ValueError("logout token")
    jti = claims.get("jti")
    if not isinstance(jti, str) or len(jti) < 20:
        raise ValueError("jti")
    if not claims.get("sid") and not claims.get("sub"):
        raise ValueError("logout subject")
    exp = int(claims.get("exp") or 0)
    if exp <= int(time.time()):
        raise ValueError("expired")
    return {"jti": jti, "sid": claims.get("sid"), "sub": claims.get("sub"), "exp": exp}


def _empty_store() -> dict:
    return {"sessions": {}, "logout_jti": {}}


def _load_store(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _empty_store()
    if not isinstance(data, dict):
        return _empty_store()
    data.setdefault("sessions", {})
    data.setdefault("logout_jti", {})
    return data


def _save_store(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError:
        pass
    temporary.replace(path)


def _prune(data: dict, now: float) -> None:
    sessions = data["sessions"]
    for sid, row in list(sessions.items()):
        if row.get("exp", 0) <= now or row.get("idle_exp", 0) <= now:
            sessions.pop(sid, None)
    for jti, exp in list(data["logout_jti"].items()):
        if exp <= now:
            data["logout_jti"].pop(jti, None)


def remember_session(settings: dict, local_sid: str, provider_sid: str | None, subject: str) -> None:
    now = time.time()
    with _store_lock:
        data = _load_store(settings["store_path"])
        _prune(data, now)
        data["sessions"][local_sid] = {
            "provider_sid": provider_sid,
            "subject": subject,
            "exp": now + SESSION_TTL_S,
            "idle_exp": now + IDLE_TTL_S,
        }
        _save_store(settings["store_path"], data)


def sso_session_active(local_sid: str | None, settings: dict | None = None) -> bool:
    if not local_sid:
        return False
    settings = settings or sso_settings()
    now = time.time()
    with _store_lock:
        data = _load_store(settings["store_path"])
        row = data["sessions"].get(local_sid)
        if not row or row.get("exp", 0) <= now or row.get("idle_exp", 0) <= now:
            return False
        if row.get("idle_exp", 0) - now < IDLE_TTL_S - 60:
            row["idle_exp"] = now + IDLE_TTL_S
            _save_store(settings["store_path"], data)
        return True


def drop_session(settings: dict, local_sid: str | None) -> None:
    if not local_sid:
        return
    with _store_lock:
        data = _load_store(settings["store_path"])
        data["sessions"].pop(local_sid, None)
        _save_store(settings["store_path"], data)


def revoke_provider_sessions(settings: dict, provider_sid: str | None, subject: str | None, jti: str, exp: int) -> bool:
    with _store_lock:
        data = _load_store(settings["store_path"])
        _prune(data, time.time())
        if jti in data["logout_jti"]:
            return False
        data["logout_jti"][jti] = exp
        for sid, row in list(data["sessions"].items()):
            if (provider_sid and row.get("provider_sid") == provider_sid) or (
                subject and row.get("subject") == subject
            ):
                data["sessions"].pop(sid, None)
        _save_store(settings["store_path"], data)
    return True


def sso_logout_target() -> str | None:
    """Keycloak end-session URL for the current SSO browser session, if any."""
    local_sid = session.get("sso_sid")
    if not local_sid:
        return None
    settings = sso_settings()
    drop_session(settings, local_sid)
    if not sso_ready(settings):
        return None
    try:
        return end_session_url(settings)
    except Exception:
        return None


def _deny(status: int, detail: str):
    messages = {
        "workspace_access_required": "This CyberHub account is not allowed to open Cyber Eye.",
        "invalid_auth_attempt": "The sign-in attempt is invalid or has expired.",
    }
    text = messages.get(detail, "Sign-in was rejected.")
    return f"<!DOCTYPE html><html><body><p>{text}</p></body></html>", status


def _read_attempt(secret: str, state: str) -> dict | None:
    raw = request.cookies.get(ATTEMPT_COOKIE)
    if not raw or not state:
        return None
    try:
        payload = _serializer(secret).loads(raw, max_age=ATTEMPT_TTL_S)
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(payload, dict) or payload.get("state") != state:
        return None
    if not payload.get("verifier") or not payload.get("nonce"):
        return None
    return payload


def register_sso(app: Flask) -> None:
    def _cookie(response, settings: dict, value: str | None):
        if value is None:
            response.delete_cookie(ATTEMPT_COOKIE, path="/")
            return response
        response.set_cookie(
            ATTEMPT_COOKIE,
            value,
            httponly=True,
            secure=_secure_cookie(settings),
            samesite="Lax",
            path="/",
            max_age=ATTEMPT_TTL_S,
        )
        return response

    @app.route("/auth/sso/login", methods=["GET"], endpoint="sso_login")
    def sso_login():
        settings = sso_settings()
        if not settings["enabled"]:
            return jsonify({"detail": "sso_disabled"}), 404
        if not sso_ready(settings):
            return jsonify({"detail": "sso_misconfigured"}), 503
        if session.get("logged_in") is True and sso_session_active(session.get("sso_sid"), settings):
            return redirect("/dashboard")
        started = begin_login(settings)
        token = _serializer(app.secret_key).dumps(
            {"state": started["state"], "nonce": started["nonce"], "verifier": started["verifier"]}
        )
        response = redirect(started["url"], code=303)
        return _cookie(response, settings, token)

    @app.route("/auth/sso/callback", methods=["GET"], endpoint="sso_callback")
    def sso_callback():
        settings = sso_settings()
        if not settings["enabled"]:
            return jsonify({"detail": "sso_disabled"}), 404
        attempt = _read_attempt(app.secret_key, request.args.get("state") or "")
        code = request.args.get("code") or ""
        if not attempt or not code:
            response = app.make_response(_deny(400, "invalid_auth_attempt"))
            return _cookie(response, settings, None)
        try:
            identity = complete_login(settings, code, attempt["verifier"], attempt["nonce"])
        except Exception as error:
            print(f"Cyber Eye SSO token exchange failed: {type(error).__name__}")
            response = app.make_response(_deny(400, "invalid_auth_attempt"))
            return _cookie(response, settings, None)
        if not has_workspace_access(identity["resource_access"], settings["client_id"]):
            response = app.make_response(_deny(403, "workspace_access_required"))
            return _cookie(response, settings, None)
        local_sid = secrets.token_urlsafe(32)
        remember_session(settings, local_sid, identity.get("provider_sid"), identity["subject"])
        session.clear()
        session.permanent = True
        session["logged_in"] = True
        session["username"] = identity["username"]
        session["sso"] = True
        session["sso_sid"] = local_sid
        response = redirect("/dashboard", code=303)
        return _cookie(response, settings, None)

    @app.route("/auth/sso/logout", methods=["GET", "POST"], endpoint="sso_logout")
    def sso_logout():
        target = sso_logout_target() or "/login"
        session.clear()
        return redirect(target, code=303)

    @app.route("/auth/sso/backchannel-logout", methods=["POST"], endpoint="sso_backchannel_logout")
    def sso_backchannel_logout():
        settings = sso_settings()
        if not settings["enabled"]:
            return jsonify({"detail": "sso_disabled"}), 404
        token = request.form.get("logout_token") or ""
        if len(token) < 20:
            return jsonify({"detail": "invalid_logout_token"}), 400
        try:
            notice = verify_logout(settings, token)
        except Exception:
            return jsonify({"detail": "invalid_logout_token"}), 400
        if not revoke_provider_sessions(
            settings, notice.get("sid"), notice.get("sub"), notice["jti"], int(notice["exp"])
        ):
            return jsonify({"detail": "logout_replay"}), 400
        return "", 204
