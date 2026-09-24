"""Public-only helpers shared by author model selectors and frozen destinations."""


def available_profiles(request, session):
    return [view for view in request.app.state.model_profile_service_factory(session).list_public() if view.enabled]


def selected_profile(request, session, version_id, consent):
    if consent != "yes":
        raise ValueError("provider consent required")
    view = request.app.state.model_profile_service_factory(session).get_public(version_id)
    if not view.enabled:
        raise ValueError("model profile unavailable")
    return view


def bound_profile(request, session, version_id):
    if not version_id:
        return None
    return request.app.state.model_profile_service_factory(session).get_public(version_id)
