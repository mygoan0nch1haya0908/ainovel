from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ainovel.models.project import ConstitutionVersion, NovelProject

VERSION_ALLOCATION_ATTEMPTS = 3


class ProjectService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(
        self, title: str, target_chars_min: int, target_chars_max: int
    ) -> NovelProject:
        normalized = title.strip()
        if not normalized:
            raise ValueError("title is required")
        if target_chars_min > target_chars_max:
            raise ValueError("minimum target must not exceed maximum")
        project = NovelProject(
            id=str(uuid4()),
            title=normalized,
            target_chars_min=target_chars_min,
            target_chars_max=target_chars_max,
        )
        self.session.add(project)
        try:
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return project

    def get(self, project_id: str) -> NovelProject:
        project = self.session.scalar(
            select(NovelProject).where(NovelProject.id == project_id)
        )
        if project is None:
            raise ValueError("project not found")
        return project

    def add_constitution(
        self, project_id: str, content: dict[str, object], author_approved: bool
    ) -> ConstitutionVersion:
        for attempt in range(VERSION_ALLOCATION_ATTEMPTS):
            project = self.session.get(NovelProject, project_id)
            if project is None:
                self.session.rollback()
                raise ValueError("project not found")
            version_number = self.session.scalar(
                select(func.coalesce(func.max(ConstitutionVersion.version_number), 0) + 1).where(
                    ConstitutionVersion.project_id == project_id
                )
            )
            constitution = ConstitutionVersion(
                id=str(uuid4()),
                project_id=project_id,
                version_number=version_number,
                content=content,
                author_approved=author_approved,
            )
            self.session.add(constitution)
            try:
                self.session.flush()
            except IntegrityError:
                self.session.rollback()
                if attempt == VERSION_ALLOCATION_ATTEMPTS - 1:
                    raise
                continue
            try:
                if author_approved:
                    project.current_constitution_version_id = constitution.id
                self.session.commit()
            except Exception:
                self.session.rollback()
                raise
            return constitution
        raise RuntimeError("constitution version allocation exhausted")
