"""The sign-in and consent page of the MCP authorization server.

/authorize sends the user here with a pending request id. The page shows which
app asks for what, signs the user in with Firebase in the browser (the password
goes to Google, never to us) and posts the Firebase ID token to /oauth/callback.
The callback verifies it, ends the pending request and returns the client's
redirect URL with a one-time code, which the page then navigates to.
"""

from html import escape
from importlib import resources
from typing import Annotated, Any
from urllib.parse import urlsplit

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field

from chef_backend.auth import InvalidTokenError, TokenVerifier, get_token_verifier
from chef_backend.config import Settings, get_settings
from chef_backend.mcp_server.auth import RECIPES_READ_SCOPE
from chef_backend.mcp_server.oauth import ChefOAuthProvider, get_oauth_provider

SCOPE_DESCRIPTIONS = {RECIPES_READ_SCOPE: "Read your recipes and the community's shared recipes"}

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; "
        "script-src 'self' https://www.gstatic.com; "
        "style-src 'self'; "
        "connect-src 'self' https://identitytoolkit.googleapis.com "
        "https://securetoken.googleapis.com; "
        "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
    ),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    # The URL carries the pending request id, so only the origin may leave the page.
    # Not no-referrer: the browser API key is restricted by referrer, and Firebase Auth
    # rejects requests that carry none.
    "Referrer-Policy": "strict-origin",
    "Cache-Control": "no-store",
}

RequestId = Annotated[str, Field(min_length=1, max_length=200)]


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    request: RequestId
    id_token: str = Field(alias="idToken", min_length=1, max_length=8192)


class DenyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: RequestId


class RedirectResponse(BaseModel):
    redirect_url: str = Field(serialization_alias="redirectUrl")


router = APIRouter(prefix="/oauth", tags=["mcp-oauth"], include_in_schema=False)

Provider = Annotated[ChefOAuthProvider, Depends(get_oauth_provider)]


def _page(body: str, status_code: int = status.HTTP_200_OK) -> HTMLResponse:
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chef: connect an app</title>
<link rel="stylesheet" href="/oauth/static/login.css">
</head>
<body>
<main>
{body}
</main>
</body>
</html>"""
    return HTMLResponse(html, status_code=status_code, headers=SECURITY_HEADERS)


def _message(title: str, text: str, status_code: int) -> HTMLResponse:
    return _page(f"<h1>{escape(title)}</h1>\n<p>{escape(text)}</p>", status_code)


@router.get("/login")
def login_page(
    provider: Provider,
    settings: Annotated[Settings, Depends(get_settings)],
    request: Annotated[str, Query(min_length=1, max_length=200)],
) -> HTMLResponse:
    if not (settings.firebase_web_api_key and settings.firebase_web_app_id):
        return _message(
            "Sign-in is not available",
            "Please try again later.",
            status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    found = provider.pending_login(request)
    if found is None:
        return _message(
            "This sign-in link has expired",
            "Go back to the app you came from and connect Chef again.",
            status.HTTP_400_BAD_REQUEST,
        )
    pending, client = found
    app_name = client.client_name or "An unnamed app"
    destination = urlsplit(str(pending.redirect_uri)).netloc
    scopes = "".join(
        f"<li>{escape(SCOPE_DESCRIPTIONS.get(scope, scope))}</li>" for scope in pending.scopes
    )
    project_id = settings.gcp_project_id or ""
    data: dict[str, Any] = {
        "request": request,
        "api-key": settings.firebase_web_api_key,
        "app-id": settings.firebase_web_app_id,
        "project-id": project_id,
        "auth-domain": f"{project_id}.firebaseapp.com",
    }
    attributes = " ".join(f'data-{key}="{escape(str(value))}"' for key, value in data.items())
    return _page(f"""<h1>Connect {escape(app_name)} to Chef</h1>
<p><strong>{escape(app_name)}</strong> wants to:</p>
<ul>{scopes}</ul>
<p class="destination">After you sign in you will be sent to <strong>{escape(destination)}</strong>.
Only continue if you started this from that app.</p>
<form id="login" {attributes}>
<label>Email <input name="email" type="email" autocomplete="username" required></label>
<label>Password
<input name="password" type="password" autocomplete="current-password" required></label>
<p id="error" role="alert" hidden></p>
<button type="submit">Sign in and allow</button>
<button type="button" id="deny" class="secondary">Cancel</button>
</form>
<script type="module" src="/oauth/static/login.js"></script>""")


@router.post("/callback")
def login_callback(
    body: LoginRequest,
    provider: Provider,
    verifier: Annotated[TokenVerifier, Depends(get_token_verifier)],
) -> RedirectResponse:
    """Trade a Firebase ID token for the client's redirect URL with a one-time code."""
    try:
        claims = verifier.verify_id_token(body.id_token)
    except InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid_id_token") from None
    firebase_claims: dict[str, Any] = claims.get("firebase") or {}
    if firebase_claims.get("sign_in_provider") == "anonymous":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "anonymous_not_allowed")
    if claims.get("email_verified") is not True:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "email_not_verified")

    redirect_url = provider.complete_login(body.request, uid=claims["uid"])
    if redirect_url is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "login_expired")
    return RedirectResponse(redirect_url=redirect_url)


@router.post("/deny")
def login_deny(body: DenyRequest, provider: Provider) -> RedirectResponse:
    redirect_url = provider.deny_login(body.request)
    if redirect_url is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "login_expired")
    return RedirectResponse(redirect_url=redirect_url)


def _static(name: str, media_type: str) -> Response:
    content = resources.files("chef_backend.mcp_server").joinpath("static", name).read_bytes()
    headers = {"X-Content-Type-Options": "nosniff", "Cache-Control": "public, max-age=300"}
    return Response(content, media_type=media_type, headers=headers)


@router.get("/static/login.js")
def login_js() -> Response:
    return _static("login.js", "text/javascript")


@router.get("/static/login.css")
def login_css() -> Response:
    return _static("login.css", "text/css")
