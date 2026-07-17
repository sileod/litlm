"""Model-name resolution and provider fallback ordering for litlm.

A bare model name (e.g. "gpt-oss-120b") is resolved into an ordered fallback list, cheapest first:
    albert (free)  ->  nvidia_nim (free)  ->  openrouter :free  ->  openrouter (paid)
A provider-prefixed name ("openrouter/…", "nvidia_nim/…", "albert/…", "vendor/model") is passed through
as-is (no fuzzy fallback). Albert (albert.api.etalab.gouv.fr) is an OpenAI-compatible endpoint, so it is
reached through litellm's openai/ handler + api_base (see _litellm_model).
"""
import os
import requests
from datetime import datetime, timedelta
from functools import lru_cache

# --- provider config / caches ---
_NVIDIA_PROVIDER_ALIASES = {"deepseek": "deepseek-ai"}
_NVIDIA_MODELS_TTL = timedelta(days=1)
_NVIDIA_MODELS_CACHE = {}

ALBERT_API_BASE = "https://albert.api.etalab.gouv.fr/v1"
_ALBERT_MODELS_TTL = timedelta(days=1)
_ALBERT_MODELS_CACHE = {}
_ALBERT_CHAT_TYPES = {"text-generation", "image-text-to-text"}


# --- model catalogues (per provider) ---
@lru_cache(maxsize=1)
def _fetch_or_models():
    try:
        return [m["id"] for m in requests.get("https://openrouter.ai/api/v1/models", timeout=2).json()["data"]]
    except Exception:
        return []


def _fetch_nvidia_models(api_base, api_key):
    if not api_key:
        return []
    now = datetime.now()
    cache_key = (api_base, api_key)
    cached = _NVIDIA_MODELS_CACHE.get(cache_key)
    if cached and now - cached[0] < _NVIDIA_MODELS_TTL:
        return cached[1]
    try:
        r = requests.get(f"{api_base.rstrip('/')}/models",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=2)
        models = [m["id"] for m in r.json().get("data", [])]
        _NVIDIA_MODELS_CACHE[cache_key] = (now, models)
        return models
    except Exception:
        return cached[1] if cached else []


def _fetch_albert_models(api_base, api_key):
    """Chat-capable model ids exposed by an Albert (etalab) OpenAI-compatible endpoint."""
    if not api_key:
        return []
    now = datetime.now()
    cache_key = (api_base, api_key)
    cached = _ALBERT_MODELS_CACHE.get(cache_key)
    if cached and now - cached[0] < _ALBERT_MODELS_TTL:
        return cached[1]
    try:
        r = requests.get(f"{api_base.rstrip('/')}/models",
                         headers={"Authorization": f"Bearer {api_key}"}, timeout=2)
        models = [m["id"] for m in r.json().get("data", [])
                  if m.get("type") in _ALBERT_CHAT_TYPES or "type" not in m]
        _ALBERT_MODELS_CACHE[cache_key] = (now, models)
        return models
    except Exception:
        return cached[1] if cached else []


# --- name helpers ---
def _strip_prefix(model):
    for prefix in ("openrouter/", "nvidia_nim/", "albert/"):
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


def _strip_free(model):
    return model[:-5] if model.endswith(":free") else model


def _uniq(items):
    out = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _nvidia_slug(model):
    if "/" not in model:
        return model
    provider, rest = model.split("/", 1)
    return f"{_NVIDIA_PROVIDER_ALIASES.get(provider, provider)}/{rest}"


def _albert_slug(target, models):
    """Best Albert model id for a bare `target`, or None. Exact (full id or basename) then substring."""
    if not target or not models:
        return None
    t = _strip_free(_strip_prefix(target)).lower()
    for m in models:
        if m.lower() == t or m.split("/")[-1].lower() == t:
            return m
    cands = [m for m in models if t in m.lower()]
    return sorted(cands, key=len)[0] if cands else None


def _resolve_or_model(model, free=False):
    target = _strip_free(_strip_prefix(model))
    models = _fetch_or_models()
    pool = [m for m in models if m.endswith(":free") == free]
    exact = f"{target}:free" if free else target
    if exact in pool:
        return exact
    cands = [m for m in pool if target in _strip_free(m)]
    if cands:
        return sorted(cands, key=lambda m: ("latest" not in m, _strip_free(m) != target, len(m)))[0]
    return exact


def _known_or_model(model):
    target = _strip_free(_strip_prefix(model))
    return any(target in _strip_free(m) for m in _fetch_or_models())


# --- routing ---
def _litellm_model(m):
    # albert/<id> is an OpenAI-compatible endpoint; litellm reaches it via its openai/ handler + api_base.
    return f"openai/{m[len('albert/'):]}" if m.startswith("albert/") else m


def _fallback_models(model):
    """Ordered fallback list for `model`, cheapest first: albert -> nvidia_nim -> or:free -> or(paid)."""
    if model.startswith("openrouter/"):
        return [model]
    if model.startswith("nvidia_nim/"):
        return [f"nvidia_nim/{_nvidia_slug(_strip_prefix(model))}"]
    if model.startswith("albert/"):
        return [model]
    if "/" in model:
        return [f"openrouter/{model}"]

    base = _strip_free(_strip_prefix(model))

    # 1) albert (free)
    albert_slug = _albert_slug(base, _fetch_albert_models(
        os.environ.get("ALBERT_API_BASE", ALBERT_API_BASE), os.environ.get("ALBERT_API_KEY")))
    albert = [f"albert/{albert_slug}"] if albert_slug else []

    # bare names must fuzzy-match *some* provider; only error if nothing matched anywhere
    if not albert and _fetch_or_models() and not _known_or_model(model):
        raise ValueError(f"Unknown model '{model}'. Use an exact provider/model name to bypass fuzzy matching.")

    # 3/4) openrouter free then paid
    free_or = f"openrouter/{_resolve_or_model(base, free=True)}"
    paid_slug = _resolve_or_model(base)
    paid_or = f"openrouter/{paid_slug}"

    # 2) nvidia_nim (free)
    nvidia_slug = _nvidia_slug(paid_slug)
    nvidia_models = _fetch_nvidia_models(
        os.environ.get("NVIDIA_NIM_API_BASE", "https://integrate.api.nvidia.com/v1"),
        os.environ.get("NVIDIA_NIM_API_KEY"))
    nvidia = [f"nvidia_nim/{nvidia_slug}"] if nvidia_slug in nvidia_models else []

    return _uniq(albert + nvidia + [free_or, paid_or])
