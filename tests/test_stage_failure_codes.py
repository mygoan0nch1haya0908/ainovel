import pytest
from ainovel.providers.contracts import ProviderTimeout, ProviderAuthenticationError, ProviderUnavailable, ProviderProtocolError
from ainovel.providers.diagnostics import ResponseFailure, FailureReason, safe_failure_code, failure_code_detail, safe_failure_detail


@pytest.mark.parametrize("error,code", [
    (ProviderTimeout("SECRET"), "provider_timeout"),
    (ProviderAuthenticationError("SECRET"), "provider_auth"),
    (ProviderUnavailable("SECRET"), "provider_unavailable"),
    (ProviderProtocolError("SECRET"), "provider_protocol"),
    (RuntimeError("SECRET"), "unknown"),
    (ResponseFailure(FailureReason.HTTP, http_status=503), "response_http_503"),
    (ResponseFailure(FailureReason.EMPTY), "response_empty"),
    (ResponseFailure(FailureReason.SCHEMA), "schema_mismatch"),
])
def test_safe_stage_failure_classification(error, code):
    assert safe_failure_code(error) == code
    assert "SECRET" not in failure_code_detail(safe_failure_code(error))


def test_mutated_exception_attributes_and_unknown_history_are_not_echoed():
    error = ResponseFailure(FailureReason.HTTP, http_status=429)
    error.http_status = "SECRET"
    error.reason = "SECRET"
    assert safe_failure_code(error) == "unknown"
    assert "SECRET" not in safe_failure_detail(error)
    assert "SECRET" not in failure_code_detail("SECRET")
    assert "旧记录" in failure_code_detail("paused_invalid")
