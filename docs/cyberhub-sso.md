# Cyber Eye SSO

Cyber Eye keeps username and password login. A Keycloak user with the `cyber-eye` / `access` role can also enter through `/auth/sso/login` and land on `/dashboard`.

The dashboard origin is `http://192.168.11.65:5000`. The browser callback is `http://192.168.11.65:5000/auth/sso/callback`.

`KEYCLOAK_ISSUER_URL` must be the same realm address the user already used to sign in to CyberHub. The Cyber Eye process has to be able to open that address. Set `CYBER_EYE_ALLOW_INSECURE_HTTP=true` because the dashboard origin is HTTP. Also set it when the Keycloak issuer is HTTP. Restart the dashboard after changing `.env`.

## Routes

```text
GET  /auth/sso/login
GET  /auth/sso/callback
GET  /auth/sso/logout
POST /auth/sso/backchannel-logout
```

## Environment

```text
CYBER_EYE_PUBLIC_URL=http://192.168.11.65:5000
CYBER_EYE_SSO_ENABLED=true
KEYCLOAK_ISSUER_URL
KEYCLOAK_CLIENT_ID=cyber-eye
KEYCLOAK_CLIENT_SECRET
CYBER_EYE_ALLOW_INSECURE_HTTP=true
```

`CYBER_EYE_SSO_ENABLED=false` leaves password login in place and answers the SSO routes with `sso_disabled`.

The SSO session uses the existing dashboard cookie plus a server-side record in `.sso_sessions.json`. It lasts eight hours and becomes idle after thirty minutes. Back-channel logout removes that record. The next request then returns to the login page. Password-login sessions are not revoked by central logout.

There is one dashboard gate. The Keycloak role allows entry. It does not create a separate local user.
