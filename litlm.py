# Save as litlm.py (or llm.py)
import asyncio
import json as jsonlib
import nest_asyncio; nest_asyncio.apply()
import litellm
from litellm import acompletion, Cache
from tqdm.asyncio import tqdm as tq
import requests
import warnings
import inspect
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
import appdirs
import os, logging

# suppression of annoying messages
os.environ["PYDANTIC_ERRORS_OMIT_URL"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.*")
warnings.filterwarnings("ignore", message=".*Expected.*serialized value may not be as expected.*")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
litellm.suppress_debug_info = True

def extract_answer(s, tag="answer"):
    first_tag, last_tag = f"<{tag}>", f"</{tag}>"
    if first_tag not in s or last_tag not in s:
        return s
    return s.split(first_tag)[1].split(last_tag)[0].strip()


# --- Configuration & Cache ---
CACHE_DIR = Path(appdirs.user_cache_dir("litlm"))
_CACHE_READY = False

_HISTORY = []
_COSTS = []
_MODEL_USED = {}
_NVIDIA_PROVIDER_ALIASES = {
    "deepseek": "deepseek-ai",
}
_NVIDIA_MODELS_TTL = timedelta(days=1)
_NVIDIA_MODELS_CACHE = {}

# --- Helpers ---
def _ensure_cache():
    global _CACHE_READY
    if _CACHE_READY or not hasattr(litellm, "Cache"): return
    if not CACHE_DIR.exists(): CACHE_DIR.mkdir(parents=True, exist_ok=True)
    litellm.cache = Cache(type="disk", disk_cache_dir=str(CACHE_DIR))
    _CACHE_READY = True

def validate_args(kwargs):
    """Warns on likely typos in arguments."""
    sig = inspect.signature(acompletion)
    valid = set(sig.parameters.keys()) | {
        "caching", "num_retries", "max_tokens", "response_format", "extra_headers", 
        "base_url", "api_key", "api_base", "deployment_id", "timeout",
        "cache_control", "extra_body", "reasoning"
    }
    for k in kwargs:
        if k not in valid:
            warnings.warn(f"Argument '{k}' is not a standard parameter. Typo?", UserWarning)

@lru_cache(maxsize=1)
def _fetch_or_models():
    try: return [m['id'] for m in requests.get("https://openrouter.ai/api/v1/models", timeout=2).json()['data']]
    except: return []

def _fetch_nvidia_models(api_base, api_key):
    if not api_key: return []
    now = datetime.now()
    cache_key = (api_base, api_key)
    cached = _NVIDIA_MODELS_CACHE.get(cache_key)
    if cached and now - cached[0] < _NVIDIA_MODELS_TTL:
        return cached[1]
    try:
        url = f"{api_base.rstrip('/')}/models"
        r = requests.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=2)
        models = [m["id"] for m in r.json().get("data", [])]
        _NVIDIA_MODELS_CACHE[cache_key] = (now, models)
        return models
    except Exception:
        return cached[1] if cached else []

def _strip_prefix(model):
    for prefix in ("openrouter/", "nvidia_nim/"):
        if model.startswith(prefix): return model[len(prefix):]
    return model

def _strip_free(model):
    return model[:-5] if model.endswith(":free") else model

def _resolve_or_model(model, free=False):
    target = _strip_free(_strip_prefix(model))
    models = _fetch_or_models()
    pool = [m for m in models if m.endswith(":free") == free]
    exact = f"{target}:free" if free else target
    if exact in pool: return exact
    cands = [m for m in pool if target in _strip_free(m)]
    if cands: return sorted(cands, key=lambda m: ("latest" not in m, _strip_free(m) != target, len(m)))[0]
    return exact

def _known_or_model(model):
    target = _strip_free(_strip_prefix(model))
    return any(target in _strip_free(m) for m in _fetch_or_models())

def _uniq(items):
    out = []
    for item in items:
        if item not in out: out.append(item)
    return out

def _nvidia_slug(model):
    if "/" not in model: return model
    provider, rest = model.split("/", 1)
    return f"{_NVIDIA_PROVIDER_ALIASES.get(provider, provider)}/{rest}"

def _fallback_models(model):
    if model.startswith("openrouter/"):
        return [model]
    if model.startswith("nvidia_nim/"):
        return [f"nvidia_nim/{_nvidia_slug(_strip_prefix(model))}"]
    if "/" in model:
        return [f"openrouter/{model}"]
    base = _strip_free(_strip_prefix(model))
    if "/" not in model and _fetch_or_models() and not _known_or_model(model):
        raise ValueError(f"Unknown model '{model}'. Use an exact provider/model name to bypass fuzzy matching.")
    free_slug = _resolve_or_model(base, free=True)
    paid_slug = _resolve_or_model(base)
    free_or = f"openrouter/{free_slug}"
    paid_or = f"openrouter/{paid_slug}"
    nvidia_slug = _nvidia_slug(base if model.startswith("nvidia_nim/") else paid_slug)
    nvidia_models = _fetch_nvidia_models(
        os.environ.get("NVIDIA_NIM_API_BASE", "https://integrate.api.nvidia.com/v1"),
        os.environ.get("NVIDIA_NIM_API_KEY"),
    )
    nvidia = [f"nvidia_nim/{nvidia_slug}"] if nvidia_slug in nvidia_models else []
    return _uniq(nvidia + [free_or, paid_or])

def _with_provider_env(model, kwargs):
    call_kwargs = dict(kwargs)
    if model.startswith("nvidia_nim/"):
        call_kwargs.setdefault("api_key", os.environ.get("NVIDIA_NIM_API_KEY"))
        if "api_base" not in call_kwargs and "base_url" not in call_kwargs:
            call_kwargs["api_base"] = os.environ.get("NVIDIA_NIM_API_BASE")
    elif model.startswith("openrouter/"):
        call_kwargs.setdefault("api_key", os.environ.get("OPENROUTER_API_KEY"))
        if "api_base" not in call_kwargs and "base_url" not in call_kwargs:
            call_kwargs["api_base"] = os.environ.get("OPENROUTER_API_BASE")
    return {k: v for k, v in call_kwargs.items() if v is not None}

def _cache_control(prompt_cache, cache_control):
    if cache_control is not None: return cache_control
    if isinstance(prompt_cache, dict): return prompt_cache
    if prompt_cache == "1h": return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"} if prompt_cache else None

def _messages(req, system=None):
    msgs = req if isinstance(req, list) else [req]
    if system and not (msgs and msgs[0].get("role") == "system"):
        msgs = [{"role": "system", "content": system}] + msgs
    return msgs

def _get(obj, name, default=None):
    if isinstance(obj, dict): return obj.get(name, default)
    return getattr(obj, name, default)

def _response_cost(response):
    hp = _get(response, "_hidden_params", {}) or {}
    usage = _get(response, "usage", {}) or {}
    return _get(response, "cost", _get(hp, "response_cost", _get(usage, "cost")))

def _record_cost(response, model):
    _MODEL_USED[id(response)] = model
    cost = _response_cost(response)
    row = {
        "time": datetime.now(),
        "model": model,
        "cost": float(cost or 0),
        "usage": _get(response, "usage", None),
    }
    _COSTS.append(row)
    return row

def cost_breakdown(period="day", by="model"):
    now = datetime.now()
    start = now - (timedelta(days=7) if period == "week" else timedelta(days=1))
    rows = [r for r in _COSTS if r["time"] >= start]
    out = {}
    for r in rows:
        key = r["model"] if by == "model" else r["time"].strftime("%Y-%m-%d")
        out[key] = out.get(key, 0) + r["cost"]
    return out

# --- Core Objects ---
class Text(str):
    def __new__(cls, content, response, call_id, prompt):
        obj = super().__new__(cls, content)
        obj._r, obj.call_id, obj.prompt = response, call_id, prompt
        obj.model_used = _MODEL_USED.get(id(response), _get(response, "model"))
        obj.cost = _response_cost(response)
        msg = response.choices[0].message
        psf = getattr(msg, "provider_specific_fields", None) or {}
        rc = getattr(msg, "reasoning_content", None) or psf.get("reasoning_content")
        obj.reasoning = obj.reasoning_content = rc
        return obj
    def __getattr__(self, name): return getattr(self._r, name)
    def __repr__(self): return super().__repr__()

def get_history(idx=-1): return _HISTORY[idx] if _HISTORY else None

def complete(inputs, model="openrouter/openai/gpt-4.1-nano", system=None, json=False, show_progress=True, caching=False, prompt_cache=False, cache_control=None, num_retries=3, max_tokens=1024, **kwargs):
    validate_args(kwargs)
    if caching: _ensure_cache()
    if json and "response_format" not in kwargs: kwargs["response_format"] = {"type": "json_object"}

    # Normalize Inputs
    if hasattr(inputs, "tolist"): inputs = inputs.tolist()
    is_batch = True
    if isinstance(inputs, str): reqs, is_batch = [[{"role": "user", "content": inputs}]], False
    elif isinstance(inputs, dict): reqs, is_batch = [[inputs]], False
    elif isinstance(inputs, list):
        if not inputs: return []
        if isinstance(inputs[0], str):
            reqs = [[{"role": "user", "content": i}] for i in inputs]
        elif isinstance(inputs[0], dict):
            reqs, is_batch = [inputs], False
        else:
            reqs = [_messages(i) for i in inputs]
    else: raise ValueError("Input must be string, list, dict, or pandas object.")
    reqs = [_messages(r, system) for r in reqs]

    # Async Runner
    async def _runner():
        models = _fallback_models(model)
        cc = _cache_control(prompt_cache, cache_control)
        async def _one(r):
            last = None
            for m in models:
                call_kwargs = _with_provider_env(m, kwargs)
                if cc and m.startswith("openrouter/"): call_kwargs["cache_control"] = cc
                try:
                    res = await acompletion(model=m, messages=r, caching=caching, num_retries=num_retries, max_tokens=max_tokens, **call_kwargs)
                    try: res._litlm_model_used = m
                    except Exception: pass
                    _record_cost(res, m)
                    return res
                except Exception as e:
                    last = e
            raise last
        tasks = [_one(r) for r in reqs]
        if show_progress and len(reqs) > 1: return await tq.gather(*tasks, desc="Completing")
        return await asyncio.gather(*tasks)

    # Execute
    raw = asyncio.get_event_loop().run_until_complete(_runner())
    call_id = len(_HISTORY)
    out = [Text(r.choices[0].message.content or "", r, call_id, q) for r, q in zip(raw, reqs)]
    res = [jsonlib.loads(str(x)) for x in out] if json else out
    res = res if is_batch else res[0]
    _HISTORY.append(res)
    return res

# Alias for explicit OpenRouter intent (optional, since complete handles it now)
or_complete = complete
