from secrets import compare_digest, token_urlsafe

from fastapi import Form, HTTPException, Request
from itsdangerous import BadSignature


CSRF_SESSION_KEY = "csrf_nonce"


def csrf_token(request: Request) -> str:
    nonce = request.session.get(CSRF_SESSION_KEY)
    if not isinstance(nonce, str) or not nonce:
        nonce = token_urlsafe(32)
        request.session[CSRF_SESSION_KEY] = nonce
    return request.app.state.csrf_signer.dumps(nonce)


def require_csrf(request: Request, csrf_token: str = Form("")) -> None:
    expected = request.session.get(CSRF_SESSION_KEY)
    if not isinstance(expected, str) or not expected or not csrf_token:
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")
    try:
        submitted = request.app.state.csrf_signer.loads(csrf_token)
    except BadSignature as error:
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid") from error
    if not isinstance(submitted, str) or not compare_digest(submitted, expected):
        raise HTTPException(status_code=403, detail="CSRF token missing or invalid")
