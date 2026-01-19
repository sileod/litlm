# Save as litlm.py (or llm.py)
import asyncio
import nest_asyncio; nest_asyncio.apply()
import litellm
from litellm import acompletion, Cache
from tqdm.asyncio import tqdm as tq
import requests
import warnings
import inspect
from functools import lru_cache
from pathlib import Path
import appdirs
import uuid

# --- Configuration & Cache ---
CACHE_DIR = Path(appdirs.user_cache_dir("litlm"))
if not CACHE_DIR.exists(): CACHE_DIR.mkdir(parents=True, exist_ok=True)
if hasattr(litellm, "Cache"):
    litellm.cache = Cache(type="disk", disk_cache_dir=str(CACHE_DIR))

_HISTORY = []

# --- Helpers ---
def validate_args(kwargs):
    """Warns on likely typos in arguments."""
    sig = inspect.signature(acompletion)
    valid = set(sig.parameters.keys()) | {
        "caching", "num_retries", "max_tokens", "response_format", "extra_headers", 
        "base_url", "api_key", "api_base", "deployment_id", "timeout"
    }
    for k in kwargs:
        if k not in valid:
            warnings.warn(f"Argument '{k}' is not a standard parameter. Typo?", UserWarning)

@lru_cache(maxsize=1)
def _fetch_or_models():
    try: return [m['id'] for m in requests.get("https://openrouter.ai/api/v1/models", timeout=2).json()['data']]
    except: return []

# --- Core Objects ---
class Text(str):
    def __new__(cls, content, response, call_id, prompt):
        obj = super().__new__(cls, content)
        obj._r, obj.call_id, obj.prompt = response, call_id, prompt
        msg = response.choices[0].message
        psf = getattr(msg, "provider_specific_fields", None) or {}
        rc = getattr(msg, "reasoning_content", None) or psf.get("reasoning_content")
        obj.reasoning = obj.reasoning_content = rc
        return obj
    def __getattr__(self, name): return getattr(self._r, name)
    def __repr__(self): return super().__repr__()

def get_history(idx=-1): return _HISTORY[idx] if _HISTORY else None

def complete(inputs, model="openrouter/openai/gpt-4.1-nano", show_progress=True, caching=False, num_retries=3, max_tokens=1024, **kwargs):
    validate_args(kwargs)
    
    # Resolver Logic (Embedded for conciseness)
    if "openrouter" not in model and "/" not in model:
        cands = [m for m in _fetch_or_models() if model in m]
        if cands:
            best = sorted(cands, key=len)[0]
            model = f"openrouter/{best}" if not best.startswith("openrouter/") else best

    # Normalize Inputs
    if hasattr(inputs, "tolist"): inputs = inputs.tolist()
    is_batch = True
    if isinstance(inputs, str): reqs, is_batch = [[{"role": "user", "content": inputs}]], False
    elif isinstance(inputs, dict): reqs, is_batch = [inputs], False
    elif isinstance(inputs, list):
        if not inputs: return []
        reqs = [[{"role": "user", "content": i}] for i in inputs] if isinstance(inputs[0], str) else inputs
    else: raise ValueError("Input must be string, list, dict, or pandas object.")

    # Async Runner
    async def _runner():
        tasks = [acompletion(model=model, messages=r, caching=caching, num_retries=num_retries, max_tokens=max_tokens, **kwargs) for r in reqs]
        if show_progress and len(reqs) > 1: return await tq.gather(*tasks, desc="Completing")
        return await asyncio.gather(*tasks)

    # Execute
    raw = asyncio.get_event_loop().run_until_complete(_runner())
    call_id = len(_HISTORY)
    out = [Text(r.choices[0].message.content or "", r, call_id, q) for r, q in zip(raw, reqs)]
    res = out if is_batch else out[0]
    _HISTORY.append(res)
    return res

# Alias for explicit OpenRouter intent (optional, since complete handles it now)
or_complete = complete