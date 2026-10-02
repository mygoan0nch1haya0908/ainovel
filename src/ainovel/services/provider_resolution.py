"""Resolve an immutable profile version anew for each compatible dispatch."""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.orm import Session

from ainovel.providers.compatible import CompatibleProvider
from ainovel.providers.contracts import (
    ModelProvider, ProviderAuthenticationError, ProviderProtocolError,
)
from ainovel.providers.registry import ProviderRegistry
from ainovel.security.secret_vault import SecretVault
from ainovel.services.model_profiles import ModelProfileError, ModelProfileService


def validate_profile_binding(
    session: Session, provider_name: str, model_name: str,
    model_profile_version_id: str | None,
) -> object | None:
    """Check the public binding inside the caller's ownership transaction."""
    if provider_name != "compatible":
        if model_profile_version_id is not None:
            raise ValueError("legacy provider cannot use a model profile")
        return None
    if not model_profile_version_id:
        raise ValueError("compatible provider requires a model profile")
    try:
        view = ModelProfileService(session).get_public(model_profile_version_id)
    except ModelProfileError:
        raise ValueError("model profile unavailable") from None
    if not view.enabled or view.model_name != model_name:
        raise ValueError("model profile unavailable or model mismatch")
    return view


class _BoundProvider:
    __slots__ = ("_resolver", "_version_id", "_model_name")

    def __init__(self, resolver: ProviderResolver, version_id: str, model_name: str):
        self._resolver = resolver
        self._version_id = version_id
        self._model_name = model_name

    def __repr__(self) -> str:
        return "<BoundCompatibleProvider redacted>"

    def _fresh(self) -> CompatibleProvider:
        with self._resolver.session_factory() as session:
            try:
                resolved = ModelProfileService(session, vault=self._resolver.vault).resolve_for_call(self._version_id)
            except ModelProfileError:
                raise ProviderAuthenticationError("model profile unavailable") from None
            if resolved.model_name != self._model_name:
                raise ProviderProtocolError("model profile mismatch")
            return CompatibleProvider(
                resolved.endpoint, resolved.model_name, api_key=resolved.api_key,
                transport=self._resolver.transport, allow_real_calls=True,
                context_window_limit=resolved.context_limit,
                max_output_tokens_limit=resolved.output_limit,
            )

    def capabilities(self, model):
        return self._fresh().capabilities(model)

    def generate(self, request):
        return self._fresh().generate(request)

    def diagnose(self, model=None):
        return self._fresh().diagnose(model)

    def list_models(self):
        return self._fresh().list_models()

    def test_connection(self):
        return self._fresh().test_connection()


class ProviderResolver:
    """Legacy providers retain their registry behavior; compatible uses pinned versions."""

    def __init__(self, session_factory: Callable[[], Session], registry: ProviderRegistry,
                 *, vault: SecretVault | None = None, transport=None):
        self.session_factory = session_factory
        self.registry = registry
        self.vault = vault
        self.transport = transport

    def resolve(self, provider_name: str, model_name: str, *,
                model_profile_version_id: str | None = None) -> ModelProvider:
        if provider_name != "compatible":
            if model_profile_version_id is not None:
                raise ProviderProtocolError("legacy provider cannot use a model profile")
            return self.registry.get(provider_name)
        if not model_profile_version_id:
            raise ProviderProtocolError("compatible provider requires a model profile")
        with self.session_factory() as session:
            try:
                view = ModelProfileService(session, vault=self.vault).get_public(model_profile_version_id)
            except ModelProfileError:
                raise ProviderAuthenticationError("model profile unavailable") from None
            if view.model_name != model_name:
                raise ProviderProtocolError("model profile mismatch")
            if not view.enabled:
                raise ProviderAuthenticationError("model profile unavailable")
        return _BoundProvider(self, model_profile_version_id, model_name)
