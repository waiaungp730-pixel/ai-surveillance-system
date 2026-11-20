import os
import tempfile
import unittest
from unittest.mock import patch

from flask import Flask

import sso_auth


class SsoRoutesTest(unittest.TestCase):
    def setUp(self):
        self.store = tempfile.NamedTemporaryFile(delete=False)
        self.store.close()
        self.env = patch.dict(
            os.environ,
            {
                "CYBER_EYE_SSO_ENABLED": "true",
                "CYBER_EYE_PUBLIC_URL": "https://192.168.11.65:5000",
                "KEYCLOAK_ISSUER_URL": "https://id.example.test/realms/cyberhub",
                "KEYCLOAK_CLIENT_ID": "cyber-eye",
                "KEYCLOAK_CLIENT_SECRET": "test-secret",
                "CYBER_EYE_ALLOW_INSECURE_HTTP": "false",
                "CYBER_EYE_SSO_STORE": self.store.name,
            },
            clear=False,
        )
        self.env.start()
        self.app = Flask(__name__)
        self.app.secret_key = "test-secret-that-is-at-least-thirty-two-characters"
        sso_auth.register_sso(self.app)
        self.client = self.app.test_client()

    def tearDown(self):
        self.env.stop()
        try:
            os.remove(self.store.name)
        except OSError:
            pass

    def test_disabled_sso_is_not_found(self):
        with patch.dict(os.environ, {"CYBER_EYE_SSO_ENABLED": "false"}):
            response = self.client.get("/auth/sso/login")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["detail"], "sso_disabled")

    def test_missing_secret_is_misconfigured(self):
        with patch.dict(os.environ, {"KEYCLOAK_CLIENT_SECRET": ""}):
            response = self.client.get("/auth/sso/login")
        self.assertEqual(response.status_code, 503)

    def test_login_redirects_to_keycloak_with_pkce(self):
        with patch.object(
            sso_auth,
            "begin_login",
            return_value={
                "url": "https://id.example.test/realms/cyberhub/protocol/openid-connect/auth?code_challenge=abc",
                "state": "state-1",
                "nonce": "nonce-1",
                "verifier": "verifier-1",
            },
        ):
            response = self.client.get("/auth/sso/login")
        self.assertEqual(response.status_code, 303)
        self.assertIn("code_challenge=abc", response.headers["Location"])
        self.assertIn("cybereye.attempt", response.headers.get("Set-Cookie", ""))

    def test_callback_opens_the_dashboard_for_the_access_role(self):
        with patch.object(
            sso_auth,
            "begin_login",
            return_value={
                "url": "https://id.example.test/auth",
                "state": "state-1",
                "nonce": "nonce-1",
                "verifier": "verifier-1",
            },
        ):
            self.client.get("/auth/sso/login")
        with patch.object(
            sso_auth,
            "complete_login",
            return_value={
                "issuer": "https://id.example.test/realms/cyberhub",
                "subject": "user-1",
                "username": "ada",
                "provider_sid": "provider-1",
                "resource_access": {"cyber-eye": {"roles": ["access"]}},
            },
        ):
            response = self.client.get("/auth/sso/callback?code=abc&state=state-1")
        self.assertEqual(response.status_code, 303)
        self.assertEqual(response.headers["Location"], "/dashboard")
        with self.client.session_transaction() as current:
            self.assertTrue(current["logged_in"])
            self.assertEqual(current["username"], "ada")
            self.assertTrue(sso_auth.sso_session_active(current["sso_sid"]))

    def test_callback_rejects_an_account_without_the_access_role(self):
        with patch.object(
            sso_auth,
            "begin_login",
            return_value={
                "url": "https://id.example.test/auth",
                "state": "state-1",
                "nonce": "nonce-1",
                "verifier": "verifier-1",
            },
        ):
            self.client.get("/auth/sso/login")
        with patch.object(
            sso_auth,
            "complete_login",
            return_value={
                "issuer": "https://id.example.test/realms/cyberhub",
                "subject": "user-1",
                "username": "ada",
                "provider_sid": "provider-1",
                "resource_access": {"cyberhub-portal": {"roles": ["portal-user"]}},
            },
        ):
            response = self.client.get("/auth/sso/callback?code=abc&state=state-1")
        self.assertEqual(response.status_code, 403)
        self.assertIn(b"not allowed", response.data)

    def test_backchannel_logout_removes_the_session(self):
        sso_auth.remember_session(sso_auth.sso_settings(), "local-1", "provider-1", "user-1")
        notice = {"jti": "logout-jti-value-123456", "sid": "provider-1", "sub": "user-1", "exp": 4_000_000_000}
        with patch.object(sso_auth, "verify_logout", return_value=notice):
            response = self.client.post(
                "/auth/sso/backchannel-logout",
                data={"logout_token": "signed-logout-token-value"},
            )
        self.assertEqual(response.status_code, 204)
        self.assertFalse(sso_auth.sso_session_active("local-1"))
        with patch.object(sso_auth, "verify_logout", return_value=notice):
            replay = self.client.post(
                "/auth/sso/backchannel-logout",
                data={"logout_token": "signed-logout-token-value"},
            )
        self.assertEqual(replay.status_code, 400)


if __name__ == "__main__":
    unittest.main()
