"""Bounded, secret-safe forms: multipart and file uploads are unsupported."""
from urllib.parse import parse_qsl

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from ainovel.db import get_session
from ainovel.services.model_profiles import ProfileInput, validate_profile_input
from ainovel.web.routes import templates
from ainovel.web.security import csrf_token, require_csrf

router = APIRouter(prefix="/model-profiles")
MAX_BODY = 65536
FIELDS = {"csrf_token", "name", "base_url", "connection_kind", "model_name", "api_key",
          "context_limit", "output_limit", "keep_existing_key", "confirm", "enabled"}


def _error(code=422):
    return HTMLResponse("配置操作未完成，请检查输入、确认选项或配置状态；密钥不会回显。", status_code=code,
                        headers={"Cache-Control": "no-store"})


async def _form(request):
    # Never use request.form/Pydantic or log secret-bearing parsing exceptions.
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > MAX_BODY:
            raise HTTPException(413)
        body.extend(chunk)
    if request.headers.get("content-type", "").split(";", 1)[0].lower() != "application/x-www-form-urlencoded":
        raise HTTPException(422)
    pairs = parse_qsl(body.decode("utf-8", errors="strict"), keep_blank_values=True,
                      strict_parsing=True, max_num_fields=20, errors="strict")
    values = {}
    for name, value in pairs:
        if name not in FIELDS or name in values:
            raise HTTPException(422)
        values[name] = value
    require_csrf(request, values.get("csrf_token", ""))
    return values


def _input(form):
    key = form.get("api_key") or None
    if key is not None and (len(key) > 8192 or any(ord(c) < 33 or ord(c) > 126 for c in key)):
        raise ValueError()
    values = ProfileInput(form.get("name", ""), form.get("base_url", ""),
                          form.get("connection_kind", ""), form.get("model_name", ""),
                          int(form.get("context_limit", "32000")), int(form.get("output_limit", "12000")))
    validate_profile_input(values, api_key=key)
    return values, key


def _view(service, profile_id):
    view = next((item for item in service.list_public() if item.profile_id == profile_id), None)
    if view is None:
        raise HTTPException(404)
    return view


def _page(request, service, view=None, *, models=(), message=None, status_code=200):
    return templates.TemplateResponse(request, "model_profile.html" if view else "model_profiles.html",
        {"request": request, "csrf_token": csrf_token(request), "profile": view,
         "profiles": service.list_public() if view is None else (), "models": models,
         "versions": service.list_versions_public(view.profile_id) if view else (),
         "counts": service.impacted_task_counts(view.profile_id) if view else {}, "message": message},
        status_code=status_code, headers={"Cache-Control": "no-store"})


@router.get("")
def profiles(request: Request, session: Session = Depends(get_session)):
    return _page(request, request.app.state.model_profile_service_factory(session))


@router.get("/{profile_id}")
def profile(profile_id: str, request: Request, session: Session = Depends(get_session)):
    service = request.app.state.model_profile_service_factory(session)
    try:
        return _page(request, service, _view(service, profile_id))
    except Exception:
        return _error(404)


@router.post("")
@router.post("/check")
@router.post("/{profile_id}/revise")
@router.post("/{profile_id}/revoke")
@router.post("/versions/{version_id}/enabled")
@router.post("/versions/{version_id}/models")
@router.post("/versions/{version_id}/test")
async def mutate(request: Request, session: Session = Depends(get_session)):
    service = request.app.state.model_profile_service_factory(session)
    view = None
    try:
        form = await _form(request)
        action = request.url.path.rsplit("/", 1)[-1]
        if action in {"model-profiles", "check", "revise"}:
            values, key = _input(form)
            if action == "check":
                if values.connection_kind == "remote" and key is None and form.get("keep_existing_key") != "yes":
                    raise ValueError()
                return _page(request, service, message="本地语法检查通过；未解析 DNS、未连接服务商、未保存。")
            if action == "revise":
                view = service.revise(request.path_params["profile_id"], values, api_key=key,
                                      keep_existing_key=form.get("keep_existing_key") == "yes")
            else:
                view = service.create(values, api_key=key)
            return RedirectResponse(f"/model-profiles/{view.profile_id}", status_code=303,
                                    headers={"Cache-Control": "no-store"})
        if form.get("confirm") != "yes":
            raise ValueError()
        if action == "revoke":
            service.revoke(request.path_params["profile_id"])
            return RedirectResponse("/model-profiles", status_code=303, headers={"Cache-Control": "no-store"})
        view = service.get_public(request.path_params["version_id"])
        if action == "enabled":
            if form.get("enabled") not in {"yes", "no"}:
                raise ValueError()
            service.set_enabled(view.version_id, form["enabled"] == "yes")
            return RedirectResponse(f"/model-profiles/{view.profile_id}", status_code=303,
                                    headers={"Cache-Control": "no-store"})
        provider = request.app.state.provider_resolver.resolve("compatible", view.model_name,
                                                               model_profile_version_id=view.version_id)
        if action == "models":
            return _page(request, service, view, models=provider.list_models(), message="列表不代表模型能力；仍可手填模型名。")
        provider.test_connection()
        return _page(request, service, view, message="合成连接测试成功；未发送小说内容。")
    except HTTPException as error:
        return _error(error.status_code)
    except Exception:
        if view is not None:
            return _page(request, service, view, message="操作失败；仍可手填模型名，请检查配置或稍后重试。", status_code=422)
        return _error()
