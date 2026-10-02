import re

from sqlalchemy import func, select
from ainovel.models.workflow import ModelAttempt

from fastapi.testclient import TestClient

from ainovel.services.batches import BatchService
from ainovel.services.outlines import OutlineNodeInput, OutlineService
from ainovel.services.projects import ProjectService


def _csrf_token(client: TestClient, path: str = "/") -> str:
    response = client.get(path)
    assert response.status_code == 200
    match = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
    assert match is not None
    return match.group(1)


def test_empty_batch_has_readable_empty_state_and_disabled_review(client, session, project, official_outline):
    batch = BatchService(session).create(project.id, official_outline.id, 1)
    page = client.get(f"/projects/{project.id}")
    assert f'href="/batches/{batch.id}"' in page.text
    form = re.search(rf'<form action="/batches/{batch.id}/ready".*?</form>', page.text, re.S)
    assert form and re.search(r'<button[^>]*disabled', form.group())
    preview = client.get(f"/batches/{batch.id}")
    assert preview.status_code == 200
    assert '尚未生成正文' in preview.text
    assert session.scalar(select(func.count()).select_from(ModelAttempt)) == 0
    rejected = client.post(f"/batches/{batch.id}/ready", data={"csrf_token": _csrf_token(client)}, follow_redirects=False)
    assert rejected.status_code == 422
    session.expire_all()
    assert BatchService(session).get(batch.id).status == 'draft'


def test_batch_preview_renders_escaped_body_and_enables_complete_review(client, session, project, official_outline):
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 1)
    body = '<script>alert(1)</script>' + '甲' * 4500
    service.save_candidate_chapter(batch.id, 1, '<b>章节</b>', body, {})
    page = client.get(f"/batches/{batch.id}")
    assert page.status_code == 200
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in page.text
    assert '<script>alert(1)</script>' not in page.text
    assert '甲' * 4500 in page.text
    form = re.search(rf'<form action="/batches/{batch.id}/ready".*?</form>', page.text, re.S)
    assert form and 'disabled' not in form.group()
    assert client.get('/batches/missing').status_code == 404


def test_incomplete_batch_keeps_review_disabled(client, session, project, official_outline):
    service = BatchService(session)
    batch = service.create(project.id, official_outline.id, 2)
    service.save_candidate_chapter(batch.id, 1, '第一章', '甲' * 4500, {})
    page = client.get(f"/batches/{batch.id}")
    assert page.status_code == 200
    form = re.search(rf'<form action="/batches/{batch.id}/ready".*?</form>', page.text, re.S)
    assert form and re.search(r'<button[^>]*disabled', form.group())


def test_every_mutation_rejects_a_missing_csrf_token(client: TestClient) -> None:
    mutation_requests = [
        ("/projects", {"title": "未授权", "target_chars_min": "100", "target_chars_max": "200"}),
        ("/projects/missing/batches", {"planned_chapters": "1"}),
        ("/batches/missing/ready", {}),
        ("/batches/missing/approve", {}),
        ("/batches/missing/reject", {"reason": "no"}),
    ]

    for path, data in mutation_requests:
        response = client.post(path, data=data, follow_redirects=False)
        assert response.status_code == 403, path

    assert "未授权" not in client.get("/").text


def test_mutation_rejects_an_invalid_csrf_token(client: TestClient) -> None:
    response = client.post(
        "/projects",
        data={
            "csrf_token": "invalid",
            "title": "未授权",
            "target_chars_min": "100",
            "target_chars_max": "200",
        },
        follow_redirects=False,
    )

    assert response.status_code == 403
    assert "未授权" not in client.get("/").text


def test_valid_csrf_token_allows_a_mutation(client: TestClient) -> None:
    response = client.post(
        "/projects",
        data={
            "csrf_token": _csrf_token(client),
            "title": "已授权",
            "target_chars_min": "100",
            "target_chars_max": "200",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert "已授权" in client.get(response.headers["location"]).text


def test_every_post_form_contains_the_session_csrf_token(
    client: TestClient, session
) -> None:
    home = client.get("/")
    home_token = _csrf_token(client)
    assert home.text.count('name="csrf_token"') == home.text.count('<form ')
    assert home_token in home.text

    project = ProjectService(session).create("表单保护", 100, 200)
    outline = OutlineService(session).create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="总纲", order=0)],
        reason="initial",
    )
    outline = OutlineService(session).approve(outline.id)
    batch = BatchService(session).create(project.id, outline.id, 1)
    page = client.get(f"/projects/{project.id}")

    assert page.text.count('name="csrf_token"') == page.text.count('<form ')
    assert home_token in page.text


def test_untrusted_host_is_rejected(client: TestClient) -> None:
    response = client.get("/health", headers={"host": "attacker.example"})

    assert response.status_code == 400


def test_loopback_hosts_and_testserver_are_allowed(client: TestClient) -> None:
    for host in ("127.0.0.1", "localhost", "testserver"):
        assert client.get("/health", headers={"host": host}).status_code == 200


def test_only_approve_and_reject_forms_require_confirmation(
    client: TestClient, session
) -> None:
    project = ProjectService(session).create("确认范围", 100, 200)
    outline = OutlineService(session).create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="总纲", order=0)],
        reason="initial",
    )
    outline = OutlineService(session).approve(outline.id)
    service = BatchService(session)
    batch = service.create(project.id, outline.id, 1)

    draft_page = client.get(f"/projects/{project.id}").text
    ready_form = re.search(
        rf'<form action="/batches/{batch.id}/ready"[^>]*>', draft_page
    )
    reject_form = re.search(
        rf'<form action="/batches/{batch.id}/reject"[^>]*>', draft_page
    )
    assert ready_form is not None
    assert "data-confirm" not in ready_form.group(0)
    assert reject_form is not None
    assert "data-confirm" in reject_form.group(0)

    service.save_candidate_chapter(batch.id, 1, "第一章", "甲" * 4500, {})
    service.mark_ready(batch.id)
    review_page = client.get(f"/projects/{project.id}").text
    approve_form = re.search(
        rf'<form action="/batches/{batch.id}/approve"[^>]*>', review_page
    )
    assert approve_form is not None
    assert "data-confirm" in approve_form.group(0)


def test_home_page_creates_and_opens_project(client: TestClient) -> None:
    response = client.post(
        "/projects",
        data={
            "csrf_token": _csrf_token(client),
            "title": "万界行舟",
            "target_chars_min": "2000000",
            "target_chars_max": "5000000",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    project_page = client.get(response.headers["location"])
    assert project_page.status_code == 200
    assert "万界行舟" in project_page.text
    assert "候选批次" in project_page.text


def test_invalid_project_form_returns_422_without_creating_project(client: TestClient) -> None:
    response = client.post(
        "/projects",
        data={
            "csrf_token": _csrf_token(client),
            "title": "",
            "target_chars_min": "5000000",
            "target_chars_max": "2000000",
        },
    )

    assert response.status_code == 422
    assert "项目名称不能为空" in response.text
    assert "暂无项目" in client.get("/").text


def test_dashboard_uses_chinese_accessibility_markers_and_local_script(
    client: TestClient,
) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert '<html lang="zh-CN">' in response.text
    assert '<meta charset="utf-8">' in response.text.lower()
    assert 'role="status"' in response.text
    assert '<label for="title">项目名称</label>' in response.text
    assert '<label for="target_chars_min">目标字数下限</label>' in response.text
    assert client.get("/static/app.js").status_code == 200


def test_batch_form_requires_current_official_outline_without_creating_batch(
    client: TestClient, session
) -> None:
    project = ProjectService(session).create("未设大纲", 100, 200)

    response = client.post(
        f"/projects/{project.id}/batches",
        data={"csrf_token": _csrf_token(client), "planned_chapters": "1"},
    )

    assert response.status_code == 422
    assert "请先创建并确认官方大纲后再创建批次" in response.text
    assert BatchService(session).list_for_project(project.id) == []


def test_ready_batch_can_be_approved_using_its_bound_official_outline(
    client: TestClient, session
) -> None:
    project = ProjectService(session).create("审批测试", 100, 200)
    outline = OutlineService(session).create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="总纲", order=0)],
        reason="initial",
    )
    outline = OutlineService(session).approve(outline.id)
    batch_service = BatchService(session)
    batch = batch_service.create(project.id, outline.id, 1)
    batch_service.save_candidate_chapter(batch.id, 1, "第一章", "甲" * 4500, {})
    batch_service.mark_ready(batch.id)

    response = client.post(
        f"/batches/{batch.id}/approve",
        data={"csrf_token": _csrf_token(client, f"/projects/{project.id}")},
        follow_redirects=False,
    )

    assert response.status_code == 303
    session.expire_all()
    assert BatchService(session).get(batch.id).status == "approved"
    page = client.get(f"/projects/{project.id}")
    assert "已批准" in page.text
    assert "官方章节：1 章" in page.text


def test_rejecting_a_batch_redirects_and_records_its_status(client: TestClient, session) -> None:
    project = ProjectService(session).create("驳回测试", 100, 200)
    outline = OutlineService(session).create_candidate(
        project.id,
        [OutlineNodeInput(key="book", parent_key=None, kind="book", title="总纲", order=0)],
        reason="initial",
    )
    outline = OutlineService(session).approve(outline.id)
    batch = BatchService(session).create(project.id, outline.id, 1)

    response = client.post(
        f"/batches/{batch.id}/reject",
        data={
            "csrf_token": _csrf_token(client, f"/projects/{project.id}"),
            "reason": "请重写",
        },
        follow_redirects=False,
    )

    assert response.status_code == 303
    session.expire_all()
    assert BatchService(session).get(batch.id).status == "rejected"
