"""Public-only helpers shared by author model selectors and frozen destinations."""


def available_profiles(request, session):
    return [view for view in request.app.state.model_profile_service_factory(session).list_public() if view.enabled]


def retry_selection_context(request, session, values):
    profiles = available_profiles(request, session)
    selected_version = values.get("model_profile_version_id", "")
    if selected_version and not any(item.version_id == selected_version for item in profiles):
        try:
            submitted = request.app.state.model_profile_service_factory(session).get_public(selected_version)
            if submitted.enabled:
                profiles.append(submitted)
        except ValueError:
            pass
    return {
        "model_profiles": profiles,
        "selected_profile_version_id": selected_version,
        "unavailable_profile_selection": bool(selected_version and not any(
            item.version_id == selected_version for item in profiles)),
        "selected_profile_model": values.get("model_name", ""),
    }


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
