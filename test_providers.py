import pytest

from litlm_providers import _fallback_models, _litellm_model


def test_direct_route_bypasses_resolution():
    route = "direct/gemini/gemini-3.7-flash"
    assert _fallback_models(route) == [route]
    assert _litellm_model(route) == "gemini/gemini-3.7-flash"


def test_direct_route_requires_provider_and_model():
    with pytest.raises(ValueError, match="direct/<litellm-provider>/<model>"):
        _fallback_models("direct/gemini")
