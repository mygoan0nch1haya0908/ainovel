"""Versioned model profiles. Only public DTOs may cross the presentation boundary.

Mutations own their session transaction, as other application services do. Each
revision owns a distinct credential reference, even for explicitly reused keys.
Revocation is durable before best-effort secret deletion. Retrying revoke retries
cleanup; it cannot recall an already-sent request or erase previous backups.
"""

from dataclasses import dataclass
from uuid import uuid4

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from ainovel.models.model_profile import ModelProfile, ModelProfileVersion
from ainovel.providers.endpoint_policy import Endpoint, normalize_endpoint
from ainovel.security.secret_vault import DpapiSecretVault, SecretVault


class ModelProfileError(ValueError):
    pass


@dataclass(frozen=True)
class ProfileInput:
    name: str
    base_url: str
    connection_kind: str
    model_name: str
    context_limit: int = 32000
    output_limit: int = 12000


@dataclass(frozen=True)
class ProfileView:
    profile_id: str
    version_id: str
    name: str
    base_url: str
    connection_kind: str
    model_name: str
    context_limit: int
    output_limit: int
    enabled: bool
    revoked: bool
    has_key: bool

    @property
    def kind(self) -> str:
        return self.connection_kind


class ResolvedProfile:
    """Internal short-lived dispatch input; deliberately not a serializable DTO."""
    __slots__ = ("view", "endpoint", "api_key")

    def __init__(self, view: ProfileView, endpoint: Endpoint, api_key: str | None):
        self.view = view
        self.endpoint = endpoint
        self.api_key = api_key

    def __repr__(self):
        return "<ResolvedProfile redacted>"

    @property
    def profile_id(self):
        return self.view.profile_id

    @property
    def version_id(self):
        return self.view.version_id

    @property
    def model_name(self):
        return self.view.model_name

    @property
    def context_limit(self):
        return self.view.context_limit

    @property
    def output_limit(self):
        return self.view.output_limit


def validate_profile_input(values: ProfileInput, *, api_key: str | None = None) -> Endpoint:
    """Local-only checks, usable by an explicit UI check action without saving."""
    try:
        for value, maximum in ((values.name, 120), (values.model_name, 255)):
            if (not isinstance(value, str) or not 1 <= len(value) <= maximum or not value.strip()
                    or any(ord(c) < 32 or ord(c) == 127 for c in value)):
                raise ValueError()
        if (type(values.context_limit) is not int or type(values.output_limit) is not int
                or not 1 <= values.context_limit <= 2147483647
                or not 1 <= values.output_limit <= min(64000, values.context_limit)):
            raise ValueError()
        endpoint = normalize_endpoint(values.base_url, values.connection_kind)
        if api_key is not None:
            if (not isinstance(api_key, str) or not 1 <= len(api_key) <= 8192 or not api_key.strip()
                    or any(ord(c) < 32 or ord(c) == 127 for c in api_key)
                    or values.name.strip() == api_key or values.model_name.strip() == api_key
                    or api_key in values.base_url):
                raise ValueError()
        return endpoint
    except Exception:
        raise ModelProfileError("invalid model profile") from None


class ModelProfileService:
    def __init__(self, session: Session, *, vault: SecretVault | None = None):
        self.session = session
        self.vault = vault if vault is not None else DpapiSecretVault()

    def _pair(self, version_id: str):
        pair = self.session.execute(
            select(ModelProfileVersion, ModelProfile).join(ModelProfile, ModelProfile.id == ModelProfileVersion.profile_id)
            .where(ModelProfileVersion.id == version_id).execution_options(populate_existing=True)
        ).one_or_none()
        if pair is None:
            raise ModelProfileError("model profile unavailable")
        return pair

    @staticmethod
    def _view(version: ModelProfileVersion, profile: ModelProfile) -> ProfileView:
        revoked = profile.revoked or version.revoked
        return ProfileView(profile.id, version.id, version.name, version.base_url, version.connection_kind,
                           version.model_name, version.context_limit, version.output_limit,
                           version.enabled and not revoked, revoked, bool(version.credential_ref) and not revoked)

    def get_public(self, version_id: str) -> ProfileView:
        try:
            return self._view(*self._pair(version_id))
        except Exception:
            raise ModelProfileError("model profile unavailable") from None

    def list_public(self) -> list[ProfileView]:
        try:
            pairs = self.session.execute(
                select(ModelProfileVersion, ModelProfile).join(ModelProfile,
                    (ModelProfile.id == ModelProfileVersion.profile_id) &
                    (ModelProfile.revision == ModelProfileVersion.version_number))
                .order_by(ModelProfile.created_at, ModelProfile.id).execution_options(populate_existing=True)
            ).all()
            return [self._view(*pair) for pair in pairs]
        except Exception:
            raise ModelProfileError("model profile unavailable") from None

    def impacted_task_counts(self, profile_id: str) -> dict[str, int]:
        """Count all bound history and workflows that may still dispatch."""
        from ainovel.models.workflow import GenerationWorkflow
        from ainovel.models.stage import StageRoadmapVersion
        from ainovel.services.workflows import TERMINAL_WORKFLOW_STATUSES

        versions = select(ModelProfileVersion.id).where(ModelProfileVersion.profile_id == profile_id)
        workflows = self.session.scalar(select(func.count()).select_from(GenerationWorkflow)
            .where(GenerationWorkflow.model_profile_version_id.in_(versions))) or 0
        active = self.session.scalar(select(func.count()).select_from(GenerationWorkflow)
            .where(GenerationWorkflow.model_profile_version_id.in_(versions),
                   GenerationWorkflow.status.not_in(TERMINAL_WORKFLOW_STATUSES))) or 0
        roadmaps = self.session.scalar(select(func.count()).select_from(StageRoadmapVersion)
            .where(StageRoadmapVersion.model_profile_version_id.in_(versions))) or 0
        return {"workflows": workflows, "active_workflows": active, "roadmaps": roadmaps}

    def list_versions_public(self, profile_id: str) -> list[ProfileView]:
        """Historical controls receive the same credential-free DTO as current versions."""
        try:
            pairs = self.session.execute(
                select(ModelProfileVersion, ModelProfile).join(ModelProfile,
                    ModelProfile.id == ModelProfileVersion.profile_id)
                .where(ModelProfile.id == profile_id)
                .order_by(ModelProfileVersion.version_number.desc())
                .execution_options(populate_existing=True)
            ).all()
            return [self._view(*pair) for pair in pairs]
        except Exception:
            raise ModelProfileError("model profile unavailable") from None

    def _credential(self, version: ModelProfileVersion) -> str | None:
        if version.credential_ref:
            return self.vault.get(version.credential_ref)
        if version.connection_kind == "remote":
            raise ModelProfileError("model profile unavailable")
        return None

    def _write_version(self, profile_id: str, number: int, values: ProfileInput, endpoint: Endpoint,
                       api_key: str | None) -> ModelProfileVersion:
        if endpoint.kind == "remote" and api_key is None:
            raise ModelProfileError("remote model credential required")
        reference = None
        try:
            if api_key is not None:
                reference = self.vault.put(api_key)
            version = ModelProfileVersion(id=str(uuid4()), profile_id=profile_id, version_number=number,
                name=values.name.strip(), base_url=endpoint.base_url, connection_kind=endpoint.kind,
                model_name=values.model_name, context_limit=values.context_limit,
                output_limit=values.output_limit, credential_ref=reference, enabled=True, revoked=False)
            self.session.add(version)
            self.session.commit()
            return version
        except Exception:
            self.session.rollback()
            if reference:
                try:
                    self.vault.delete(reference)
                except Exception:
                    raise ModelProfileError("model profile save failed; credential cleanup required") from None
            raise ModelProfileError("model profile save failed") from None

    def create(self, values: ProfileInput, *, api_key: str | None) -> ProfileView:
        endpoint = validate_profile_input(values, api_key=api_key)
        if endpoint.kind == "remote" and api_key is None:
            raise ModelProfileError("remote model credential required")
        profile = ModelProfile(id=str(uuid4()), revision=1, revoked=False)
        try:
            self.session.add(profile)
            # Order the parent insert explicitly; no ORM relationship is required.
            self.session.flush()
            version = self._write_version(profile.id, 1, values, endpoint, api_key)
            return self._view(version, profile)
        except Exception as error:
            self.session.rollback()
            if isinstance(error, ModelProfileError):
                raise error from None
            raise ModelProfileError("model profile save failed") from None

    def revise(self, profile_id: str, values: ProfileInput, *, api_key: str | None = None,
               keep_existing_key: bool = False) -> ProfileView:
        endpoint = validate_profile_input(values, api_key=api_key)
        try:
            pair = self.session.execute(select(ModelProfileVersion, ModelProfile).join(ModelProfile,
                (ModelProfile.id == ModelProfileVersion.profile_id) &
                (ModelProfile.revision == ModelProfileVersion.version_number))
                .where(ModelProfile.id == profile_id).execution_options(populate_existing=True)).one_or_none()
            if pair is None or pair[1].revoked or pair[0].revoked:
                raise ModelProfileError("model profile unavailable")
            previous, profile = pair
            changed_target = (endpoint.base_url, endpoint.kind) != (previous.base_url, previous.connection_kind)
            if changed_target and api_key is None and (keep_existing_key or previous.credential_ref):
                raise ModelProfileError("new credential required for changed target")
            if api_key is None and keep_existing_key:
                api_key = self._credential(previous)
                validate_profile_input(values, api_key=api_key)
            if endpoint.kind == "remote" and api_key is None:
                raise ModelProfileError("remote model credential required")
            number = profile.revision + 1
            changed = self.session.execute(update(ModelProfile).where(ModelProfile.id == profile_id,
                ModelProfile.revision == profile.revision, ModelProfile.revoked.is_(False))
                .values(revision=number).execution_options(synchronize_session=False))
            if changed.rowcount != 1:
                raise ModelProfileError("model profile changed; retry required")
            version = self._write_version(profile_id, number, values, endpoint, api_key)
            return self._view(version, profile)
        except Exception as error:
            self.session.rollback()
            if isinstance(error, ModelProfileError):
                raise error from None
            raise ModelProfileError("model profile save failed") from None

    def resolve_for_call(self, version_id: str) -> ResolvedProfile:
        try:
            version, profile = self._pair(version_id)
            view = self._view(version, profile)
            if not view.enabled:
                raise ModelProfileError("model profile unavailable")
            key = self._credential(version)
            return ResolvedProfile(view, normalize_endpoint(view.base_url, view.connection_kind), key)
        except Exception:
            raise ModelProfileError("model profile unavailable") from None

    def set_enabled(self, version_id: str, enabled: bool) -> None:
        try:
            if type(enabled) is not bool:
                raise ModelProfileError("invalid model profile state")
            version, profile = self._pair(version_id)
            if profile.revoked or version.revoked:
                raise ModelProfileError("model profile unavailable")
            if enabled:
                self._credential(version)
            changed = self.session.execute(update(ModelProfileVersion).where(
                ModelProfileVersion.id == version_id, ModelProfileVersion.revoked.is_(False),
                ModelProfileVersion.profile_id.in_(select(ModelProfile.id).where(ModelProfile.revoked.is_(False))))
                .values(enabled=enabled).execution_options(synchronize_session=False))
            if changed.rowcount != 1:
                raise ModelProfileError("model profile unavailable")
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise ModelProfileError("model profile state change failed") from None

    def revoke(self, profile_id: str) -> None:
        try:
            profile = self.session.get(ModelProfile, profile_id, populate_existing=True)
            if profile is None:
                raise ModelProfileError("model profile unavailable")
            # Acquire the write lock before collecting refs: a concurrent revision
            # must be either included here or rejected by the revoked predicate.
            self.session.execute(update(ModelProfile).where(ModelProfile.id == profile_id)
                                 .values(revoked=True).execution_options(synchronize_session=False))
            references = self.session.scalars(select(ModelProfileVersion.credential_ref)
                .where(ModelProfileVersion.profile_id == profile_id)).all()
            self.session.execute(update(ModelProfileVersion).where(ModelProfileVersion.profile_id == profile_id)
                .values(revoked=True, enabled=False).execution_options(synchronize_session=False))
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise ModelProfileError("model profile revocation failed") from None
        failed = False
        for reference in references:
            if reference:
                try:
                    self.vault.delete(reference)
                except Exception:
                    failed = True
        if failed:
            raise ModelProfileError("model profile revoked; credential cleanup required") from None
