import pytest
from types import SimpleNamespace
from unittest.mock import patch

import litlm
from litlm_providers import _fallback_models, _litellm_model


def test_direct_route_bypasses_resolution():
    route = "direct/gemini/gemini-3.7-flash"
    assert _fallback_models(route) == [route]
    assert _litellm_model(route) == "gemini/gemini-3.7-flash"


def test_direct_route_requires_provider_and_model():
    with pytest.raises(ValueError, match="direct/<litellm-provider>/<model>"):
        _fallback_models("direct/gemini")


def test_bare_gemini_prefers_direct_byok_before_openrouter():
    with (
        patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=True),
        patch("litlm_providers._fetch_albert_models", return_value=[]),
        patch("litlm_providers._fetch_nvidia_models", return_value=[]),
        patch(
            "litlm_providers._fetch_or_models",
            return_value=["google/gemini-3.7-flash"],
        ),
    ):
        assert _fallback_models("gemini-3.7-flash") == [
            "direct/gemini/gemini-3.7-flash",
            "openrouter/gemini-3.7-flash:free",
            "openrouter/google/gemini-3.7-flash",
        ]


def test_bare_gemini_skips_direct_route_without_key():
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("litlm_providers._fetch_albert_models", return_value=[]),
        patch("litlm_providers._fetch_nvidia_models", return_value=[]),
        patch(
            "litlm_providers._fetch_or_models",
            return_value=["google/gemini-3.7-flash"],
        ),
    ):
        assert _fallback_models("gemini-3.7-flash") == [
            "openrouter/gemini-3.7-flash:free",
            "openrouter/google/gemini-3.7-flash",
        ]


def test_exhausted_route_is_disabled_for_rest_of_batch():
    calls = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"].startswith("gemini/"):
            raise RuntimeError("RESOURCE_EXHAUSTED: quota exceeded")
        prompt = kwargs["messages"][0]["content"]
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=prompt, provider_specific_fields={}),
                finish_reason="stop",
            )],
            usage={},
            _hidden_params={},
        )

    with patch.object(litlm, "acompletion", fake_acompletion):
        result = litlm.complete(
            ["one", "two", "three"],
            model="gemini-3.7-flash",
            fallbacks=[
                "direct/gemini/gemini-3.7-flash",
                "openrouter/google/gemini-3.7-flash",
            ],
            max_concurrency=1,
            num_retries=0,
            show_progress=False,
        )

    assert [str(item) for item in result] == ["one", "two", "three"]
    assert calls.count("gemini/gemini-3.7-flash") == 1
    assert calls.count("openrouter/google/gemini-3.7-flash") == 3


def test_openrouter_key_limit_is_quota_exhaustion():
    assert litlm._quota_exhausted(RuntimeError("Key limit exceeded (total limit)"))
