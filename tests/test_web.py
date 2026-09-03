from fastapi.testclient import TestClient

from ainovel.services.batches import BatchService
from ainovel.services.outlines import OutlineNodeInput, OutlineService
from ainovel.services.projects import ProjectService


def test_home_page_creates_and_opens_project(client: TestClient) -> None:
    response = client.post(
        "/projects",
        data={
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
        data={"title": "", "target_chars_min": "5000000", "target_chars_max": "2000000"},
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
        f"/projects/{project.id}/batches", data={"planned_chapters": "1"}
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

    response = client.post(f"/batches/{batch.id}/approve", follow_redirects=False)

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
        f"/batches/{batch.id}/reject", data={"reason": "请重写"}, follow_redirects=False
    )

    assert response.status_code == 303
    session.expire_all()
    assert BatchService(session).get(batch.id).status == "rejected"
