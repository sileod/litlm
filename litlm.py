# Save as litlm.py (or llm.py)
import asyncio
import json as jsonlib
import nest_asyncio; nest_asyncio.apply()
import litellm
from litellm import acompletion, Cache
from tqdm.auto import tqdm as tq
import warnings
import inspect
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
import appdirs
import os, logging, sys
import time
from litlm_providers import _fallback_models, _litellm_model

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


def _first_json_span(s):
    """Span of the first balanced {...} or [...] in s, respecting strings/escapes (so braces inside
    string values don't fool the scan). Returns (start, end) or None."""
    start = next((i for i, c in enumerate(s) if c in "{["), None)
    if start is None:
        return None
    open_c = s[start]; close_c = "}" if open_c == "{" else "]"
    depth = 0; in_str = False; esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            esc = (c == "\\" and not esc)
            if c == '"' and not esc: in_str = False
        elif c == '"': in_str = True
        elif c == open_c: depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0: return (start, i + 1)
    return None


def extract_json(s, default=None):
    """Best-effort parse of JSON emitted by an LLM. Tolerates ```json fences and prose around the object
    by falling back to the first balanced {...}/[...] span. Returns `default` if nothing parses (when
    `default` is left None and parsing fails, raises ValueError)."""
    s = str(s).strip()
    if s.startswith("```"):                                   # strip a leading ```json / ``` fence
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("`")
        if s.lstrip().lower().startswith("json"): s = s.lstrip()[4:]
    for candidate in (s, (lambda sp: s[sp[0]:sp[1]] if sp else None)(_first_json_span(s))):
        if candidate is None: continue
        try:
            return jsonlib.loads(candidate)
        except Exception:
            continue
    if default is not None:
        return default
    raise ValueError(f"could not extract JSON from model reply: {s[:200]!r}")


# --- Configuration & Cache ---
CACHE_DIR = Path(appdirs.user_cache_dir("litlm"))
_CACHE_READY = False

_HISTORY = []
_COSTS = []
_FAILURES = []
_MODEL_USED = {}

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
        "cache_control", "extra_body", "reasoning", "reasoning_effort"
    }
    for k in kwargs:
        if k not in valid:
            warnings.warn(f"Argument '{k}' is not a standard parameter. Typo?", UserWarning)

# Model-name resolution + provider fallback ordering live in litlm_providers.py
# (albert -> nvidia_nim -> openrouter:free -> openrouter paid). Imported at top.

def _normalize_reasoning(call_kwargs):
    reasoning = call_kwargs.pop("reasoning", None)
    effort = call_kwargs.pop("reasoning_effort", None)
    if effort is None and isinstance(reasoning, dict):
        effort = reasoning.get("effort")
    if not effort:
        return call_kwargs

    extra_body = dict(call_kwargs.get("extra_body") or {})
    extra_body.setdefault("reasoning_effort", effort)
    call_kwargs["extra_body"] = extra_body
    return call_kwargs

def _drop_reasoning_effort(call_kwargs):
    changed = False
    if "reasoning_effort" in call_kwargs:
        call_kwargs.pop("reasoning_effort", None)
        changed = True
    extra_body = call_kwargs.get("extra_body")
    if isinstance(extra_body, dict) and "reasoning_effort" in extra_body:
        extra_body = dict(extra_body)
        extra_body.pop("reasoning_effort", None)
        if extra_body:
            call_kwargs["extra_body"] = extra_body
        else:
            call_kwargs.pop("extra_body", None)
        changed = True
    return changed

def _mentions_unsupported_reasoning_effort(exc):
    msg = str(exc).lower()
    if "reasoning_effort" not in msg:
        return False
    return any(
        token in msg
        for token in ("unsupported", "not supported", "unrecognized", "unknown", "invalid", "extra inputs")
    ) or ("literal_error" in msg and "input should be" in msg)

def _drop_response_format(call_kwargs):
    """Drop response_format (json mode) so a provider that rejects it can still answer; extract_json then
    recovers the JSON from the free-form reply."""
    if "response_format" in call_kwargs:
        call_kwargs.pop("response_format", None)
        return True
    return False


def _mentions_unsupported_response_format(exc):
    msg = str(exc).lower()
    if "response_format" not in msg and "json_object" not in msg:
        return False
    return any(t in msg for t in ("unsupported", "not supported", "unrecognized", "unknown", "invalid", "extra inputs", "does not support"))


def _quota_exhausted(exc):
    """Whether a provider failure makes retrying the same route wasteful for this batch."""
    msg = str(exc).lower().replace("_", " ")
    return any(
        token in msg
        for token in (
            "quota exceeded",
            "resource exhausted",
            "key limit exceeded",
            "insufficient quota",
            "out of credits",
            "credit balance",
            "spending limit",
            "payment required",
        )
    )


def _with_provider_env(model, kwargs):
    call_kwargs = dict(kwargs)
    call_kwargs.pop("debug", None)
    if model.startswith("nvidia_nim/"):
        call_kwargs.setdefault("api_key", os.environ.get("NVIDIA_NIM_API_KEY"))
        if "api_base" not in call_kwargs and "base_url" not in call_kwargs:
            call_kwargs["api_base"] = os.environ.get("NVIDIA_NIM_API_BASE")
        call_kwargs = _normalize_reasoning(call_kwargs)
    elif model.startswith("openrouter/"):
        call_kwargs.setdefault("api_key", os.environ.get("OPENROUTER_API_KEY"))
        if "api_base" not in call_kwargs and "base_url" not in call_kwargs:
            call_kwargs["api_base"] = os.environ.get("OPENROUTER_API_BASE")
        call_kwargs = _normalize_reasoning(call_kwargs)
    elif model.startswith("albert/"):
        call_kwargs.setdefault("api_key", os.environ.get("ALBERT_API_KEY"))
        if "api_base" not in call_kwargs and "base_url" not in call_kwargs:
            call_kwargs["api_base"] = os.environ.get(
                "ALBERT_API_BASE", "https://albert.api.etalab.gouv.fr/v1")
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
    if period == "session":
        rows = list(_COSTS)
    else:
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
        obj.failed = False
        obj.latency_s = getattr(response, "_litlm_latency_s", None)
        return obj
    def __getattr__(self, name): return getattr(self._r, name)
    def __repr__(self): return super().__repr__()

class Failure(str):
    """Empty string-compatible result for one failed item in a batch."""
    def __new__(cls, error, call_id, prompt, model):
        obj = super().__new__(cls, "")
        obj.error, obj.call_id, obj.prompt = error, call_id, prompt
        obj.model_used, obj.cost, obj.failed = model, 0.0, True
        obj.latency_s = getattr(error, "_litlm_latency_s", None)
        return obj

class BatchResult(list):
    """List-compatible batch output that can retry only its currently failed items."""
    def __init__(self, values=(), resume_options=None):
        super().__init__(values)
        self._resume_options = dict(resume_options or {})

    @property
    def failures(self):
        """Failures still present in this batch (resolved failures are excluded)."""
        return [item for item in self if getattr(item, "failed", False)]

    def resume(
        self,
        timeout: Optional[float] = None,
        num_retries: Optional[int] = None,
        max_concurrency: Optional[int] = None,
        **overrides: Any,
    ) -> "BatchResult":
        """Retry failed positions in place, optionally overriding completion settings."""
        if "inputs" in overrides:
            raise TypeError("resume() determines inputs from failed batch items")

        failed_indexes = [
            index for index, item in enumerate(self)
            if getattr(item, "failed", False)
        ]
        if not failed_indexes:
            return self

        options = dict(self._resume_options)
        if timeout is not None:
            options["timeout"] = timeout
        if num_retries is not None:
            options["num_retries"] = num_retries
        if max_concurrency is not None:
            options["max_concurrency"] = max_concurrency
        options.update(overrides)

        prompts = [self[index].prompt for index in failed_indexes]
        call_options = dict(options)
        on_result = call_options.get("on_result")
        if on_result is not None:
            call_options["on_result"] = (
                lambda retry_index, result: on_result(
                    failed_indexes[retry_index], result
                )
            )
        retried = complete(inputs=prompts, **call_options)
        for index, result in zip(failed_indexes, retried):
            self[index] = result
        self._resume_options = options
        return self

def get_history(idx: int = -1): return _HISTORY[idx] if _HISTORY else None

def get_failures(call_id: Optional[int] = None) -> List[Failure]:
    """Failed batch items for this session, optionally restricted to one complete() call."""
    return [f for f in _FAILURES if call_id is None or f.call_id == call_id]

def get_failure(call_id: Optional[int] = None) -> Optional[Exception]:
    """Most recent underlying exception, optionally restricted to one complete() call."""
    failures = get_failures(call_id)
    return failures[-1].error if failures else None

def _error_groups(errors):
    """Exception counts and first messages, ordered by count then first occurrence."""
    by_type = {}
    for error in errors:
        name = type(error).__name__
        if name in by_type:
            by_type[name][0] += 1
        else:
            message = " ".join(str(error).split()) or "(no error message)"
            by_type[name] = [1, message, len(by_type)]
    return sorted(by_type.items(), key=lambda item: (-item[1][0], item[1][2]))

def _crop(text, max_chars):
    return text if len(text) <= max_chars else text[:max_chars - 1].rstrip() + "…"

def _compact_error_breakdown(errors, max_types=2, max_chars=80):
    """Bounded, single-line error breakdown suitable for a tqdm postfix."""
    groups = _error_groups(errors)
    shown = groups[:max_types]
    counts = ", ".join(f"{name}×{count}" for name, (count, _, _) in shown)
    if len(groups) > max_types:
        counts += f", +{len(groups) - max_types} types"
    message = shown[0][1][1] if shown else ""
    detail = f"{counts} | {message}" if message else counts
    return _crop(detail, max_chars)

def _representative_failure_lines(failures, max_types=3, max_chars=240):
    """One cropped example per exception type, including how often that type occurred."""
    groups = _error_groups(failure.error for failure in failures)
    lines = []
    for name, (count, message, _) in groups[:max_types]:
        prefix = f"{count}× " if count > 1 else ""
        lines.append("  " + _crop(f"{prefix}{name}: {message}", max_chars))
    remaining = len(groups) - len(lines)
    if remaining:
        lines.append(f"  … and {remaining} other error type{'s' if remaining != 1 else ''}")
    return lines

def complete(
    inputs: Any,
    model: str = "openrouter/openai/gpt-4.1-nano",
    system: Optional[str] = None,
    json: bool = False,
    show_progress: bool = True,
    caching: bool = False,
    prompt_cache: Union[bool, str, Dict[str, Any]] = False,
    cache_control: Optional[Dict[str, Any]] = None,
    num_retries: int = 3,
    max_tokens: int = 1024,
    timeout: Optional[float] = 60,
    attempt_timeout: Optional[float] = None,
    debug: bool = False,
    max_concurrency: Optional[int] = None,
    rpm: Optional[float] = None,
    reasoning_effort: Optional[str] = None,
    temperature: Optional[float] = None,
    on_result: Optional[Callable[[int, Union[Text, Failure]], None]] = None,
    fallbacks: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Union[Text, Failure, BatchResult, Dict[str, Any], List[Any]]:
    """Complete one prompt or an ordered batch with a synchronous, notebook-friendly API.

    Args:
        inputs: A string, message dict, conversation, or batch of those values.
        model: Provider/model name or a bare model name resolved through configured fallbacks.
        system: Optional system message prepended to each prompt.
        json: Request JSON output and parse it before returning.
        show_progress: Display a compact tqdm progress bar for batches.
        caching: Cache complete responses locally so identical calls can be reused.
        prompt_cache: Enable provider prompt caching; ``"1h"`` requests a one-hour TTL.
        cache_control: Explicit provider cache-control object.
        num_retries: Retries performed by LiteLLM for each provider request.
        max_tokens: Maximum generated tokens per request.
        timeout: Timeout in seconds for each provider attempt.
        attempt_timeout: Optional hard wall-clock bound around an entire
            provider attempt. Unlike `timeout`, this also covers providers
            that fail to honor LiteLLM's request timeout.
        debug: Print provider routing and fallback details.
        max_concurrency: Maximum simultaneous batch requests; ``None`` is unbounded.
        rpm: Maximum request starts per minute; ``None`` disables throttling.
        reasoning_effort: Reasoning level such as ``"none"``, ``"low"``, or ``"high"``.
        temperature: Sampling temperature forwarded to the provider.
        on_result: Optional callback invoked as each batch item settles. It
            receives the item's original input index and its ``Text`` or
            ``Failure`` result. Callback exceptions abort ``complete``.
        fallbacks: Optional exact, ordered route list. When omitted, a bare
            model uses litlm's free/BYOK-first provider hierarchy.
        **kwargs: Additional LiteLLM/provider parameters.

    Returns:
        A Text-like scalar for one prompt, or a list-compatible BatchResult. Call
        ``batch.resume(timeout=...)`` to retry only failed batch positions in place.
    """
    debug = bool(debug or kwargs.pop("debug", False))
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if temperature is not None:
        kwargs["temperature"] = temperature
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

    call_id = len(_HISTORY)

    def _wrap_result(index, response):
        if isinstance(response, (Text, Failure)):
            result = response
        elif isinstance(response, Exception):
            result = Failure(response, call_id, reqs[index], model)
            _FAILURES.append(result)
        else:
            result = Text(
                response.choices[0].message.content or "",
                response,
                call_id,
                reqs[index],
            )
        if on_result is not None:
            on_result(index, result)
        return result

    # Async Runner
    async def _runner():
        if fallbacks is None:
            models = _fallback_models(model)
        else:
            if isinstance(fallbacks, str) or not fallbacks:
                raise ValueError("fallbacks must be a non-empty sequence of exact routes")
            models = []
            for route in fallbacks:
                models.extend(_fallback_models(route))
            models = list(dict.fromkeys(models))
        if debug:
            print(f"litlm fallback models: {models}")
        disabled_routes = set()
        cc = _cache_control(prompt_cache, cache_control)
        # rpm throttle: pace request STARTS to stay under `rpm` (parallel underneath, up to max_concurrency).
        _interval = 60.0 / rpm if rpm else 0.0
        _rl_lock = asyncio.Lock(); _next = [0.0]
        async def _throttle():
            if not _interval: return
            async with _rl_lock:
                loop = asyncio.get_event_loop(); now = loop.time()
                wait = _next[0] - now
                if wait > 0: await asyncio.sleep(wait)
                _next[0] = max(now, _next[0]) + _interval
        async def _one(r):
            started = time.perf_counter()
            await _throttle()
            last = None
            for m in models:
                if m in disabled_routes:
                    if debug:
                        print(f"litlm skipping batch-disabled route: {m}")
                    continue
                call_kwargs = _with_provider_env(m, kwargs)
                if cc and m.startswith("openrouter/"): call_kwargs["cache_control"] = cc

                # Retry loop for local parameter dropping (0 network requests wasted)
                while True:
                    try:
                        if debug:
                            print(f"litlm trying model: {m}")
                        request = acompletion(
                            model=_litellm_model(m), messages=r, caching=caching,
                            num_retries=num_retries, max_tokens=max_tokens,
                            timeout=timeout, **call_kwargs,
                        )
                        res = (
                            await asyncio.wait_for(request, timeout=attempt_timeout)
                            if attempt_timeout else await request
                        )
                        try: res._litlm_model_used = m
                        except Exception: pass
                        try: res._litlm_latency_s = time.perf_counter() - started
                        except Exception: pass
                        _record_cost(res, m)
                        if debug:
                            print(f"litlm succeeded with model: {m}")
                        return res
                    except litellm.UnsupportedParamsError as e:
                        # This is a local pre-flight error in litellm, no network request was made yet.
                        if "reasoning_effort" in str(e) and _drop_reasoning_effort(call_kwargs):
                            # Automatically ignore reasoning_effort if litellm says it's unsupported
                            if debug:
                                print(f"litlm Dropped reasoning_effort locally for {m} due to UnsupportedParamsError")
                            continue
                        if "response_format" in str(e).lower() and _drop_response_format(call_kwargs):
                            if debug:
                                print(f"litlm Dropped response_format locally for {m}; will parse JSON from free text")
                            continue

                        if debug:
                            print(f"litlm failed with model {m}: {type(e).__name__}: {e}")
                        last = e
                        break
                    except TimeoutError:
                        last = TimeoutError(f"litlm timed out after {timeout}s while calling {m}")
                        if debug:
                            print(f"litlm failed with model {m}: TimeoutError: {last}")
                        break
                    except Exception as e:
                        if _mentions_unsupported_reasoning_effort(e) and _drop_reasoning_effort(call_kwargs):
                            # Some providers reject extra_body.reasoning_effort at request time.
                            if debug:
                                print(f"litlm Dropped reasoning_effort for {m} after provider rejection")
                            continue
                        if _mentions_unsupported_response_format(e) and _drop_response_format(call_kwargs):
                            # Some providers reject response_format=json_object at request time.
                            if debug:
                                print(f"litlm Dropped response_format for {m} after provider rejection")
                            continue
                        if debug:
                            print(f"litlm failed with model {m}: {type(e).__name__}: {e}")
                        if _quota_exhausted(e):
                            disabled_routes.add(m)
                            if debug:
                                print(f"litlm disabling exhausted route for this batch: {m}")
                        last = e
                        break
            if last is None:
                last = RuntimeError("all configured fallback routes are disabled")
            try: last._litlm_latency_s = time.perf_counter() - started
            except Exception: pass
            raise last from None
        if max_concurrency and max_concurrency > 0:
            _sem = asyncio.Semaphore(int(max_concurrency))
            async def _bounded(r):
                async with _sem: return await _one(r)
            tasks = [_bounded(r) for r in reqs]
        else:
            tasks = [_one(r) for r in reqs]
        if (show_progress or on_result is not None) and len(reqs) > 1:
            results = [None] * len(tasks)
            batch_cost = 0.0
            async def _settled(i, task):
                try:
                    response = await task
                except Exception as error:
                    response = error
                # Wrap and notify in the settling task so callbacks observe true
                # completion order rather than as_completed() iteration order.
                return i, _wrap_result(i, response)

            pending = [_settled(i, task) for i, task in enumerate(tasks)]
            settled = asyncio.as_completed(pending)
            bar = tq(settled, total=len(tasks), desc="Completing") if show_progress else settled
            failed = 0
            completed = 0
            errors = []
            try:
                for future in bar:
                    i, result = await future
                    results[i] = result
                    completed += 1
                    if result.failed:
                        failed += 1
                        errors.append(result.error)
                    else:
                        batch_cost += float(result.cost or 0)
                    status = f"cost=${batch_cost:.6f}"
                    if failed:
                        status += (
                            f", ⚠ {failed}/{completed} ({failed / completed:.1%}), "
                            f"{_compact_error_breakdown(errors)}"
                        )
                    if show_progress:
                        bar.set_postfix_str(status)
            finally:
                if show_progress:
                    bar.close()
            return results
        if len(reqs) > 1:
            return await asyncio.gather(*tasks, return_exceptions=True)
        return await asyncio.gather(*tasks)

    # Execute
    raw = asyncio.get_event_loop().run_until_complete(_runner())
    out = [
        r if isinstance(r, (Text, Failure)) else _wrap_result(index, r)
        for index, r in enumerate(raw)
    ]
    failed = sum(x.failed for x in out)
    if failed:
        call_failures = [x for x in out if x.failed]
        summary = [
            f"litlm: ⚠ {failed}/{len(out)} failed ({failed / len(out):.1%})",
            "litlm: representative error(s):",
            *_representative_failure_lines(call_failures),
            f"litlm: inspect the full error with litlm.get_failure({call_id})",
        ]
        print("\n".join(summary), file=sys.stderr)
    res = [x if x.failed else extract_json(str(x)) for x in out] if json else out
    if is_batch:
        resume_options = {
            "model": model,
            "json": json,
            "show_progress": show_progress,
            "caching": caching,
            "prompt_cache": prompt_cache,
            "cache_control": cache_control,
            "num_retries": num_retries,
            "max_tokens": max_tokens,
            "timeout": timeout,
            "attempt_timeout": attempt_timeout,
            "debug": debug,
            "max_concurrency": max_concurrency,
            "rpm": rpm,
            "reasoning_effort": reasoning_effort,
            "temperature": temperature,
            "on_result": on_result,
            "fallbacks": fallbacks,
            **kwargs,
        }
        res = BatchResult(res, resume_options=resume_options)
    else:
        res = res[0]
    _HISTORY.append(res)
    return res

# Alias for explicit OpenRouter intent (optional, since complete handles it now)
or_complete = complete


# --- Benchmarks ---
def _percentile(values, percentile):
    """Linearly interpolated percentile without a NumPy dependency."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _output_tokens(result):
    usage = getattr(result, "usage", None)
    for name in ("completion_tokens", "output_tokens"):
        value = _get(usage, name)
        if value is not None:
            return int(value)
    return 0


class BenchmarkResult(list):
    """List-compatible benchmark rows with Markdown and notebook table renderers."""
    columns = (
        "provider", "model", "mode", "concurrency", "success",
        "wall_s", "mean_s", "p50_s", "p95_s", "req_s", "tok_s", "error",
    )

    def to_markdown(self):
        def display(value):
            if value is None:
                return "-"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)

        header = "| " + " | ".join(self.columns) + " |"
        divider = "| " + " | ".join("---" for _ in self.columns) + " |"
        lines = [header, divider]
        lines.extend(
            "| " + " | ".join(display(row.get(column)) for column in self.columns) + " |"
            for row in self
        )
        return "\n".join(lines)

    def _repr_html_(self):
        import html
        head = "".join(f"<th>{html.escape(column)}</th>" for column in self.columns)
        body = "".join(
            "<tr>" + "".join(
                f"<td>{html.escape(str(row.get(column, '')))}</td>"
                for column in self.columns
            ) + "</tr>"
            for row in self
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    def __str__(self):
        return self.to_markdown()

    def save_markdown(self, path, title="litlm benchmark results", notes=None):
        """Write a standalone report and return its path."""
        path = Path(path)
        lines = [
            f"# {title}",
            "",
            f"Generated {datetime.now().astimezone().isoformat(timespec='seconds')}.",
            "",
        ]
        if notes:
            lines.extend([str(notes).strip(), ""])
        lines.extend([
            self.to_markdown(),
            "",
            "`mean_s`, `p50_s`, and `p95_s` are end-to-end request latency; "
            "`req_s` and `tok_s` use whole-scenario wall time.",
            "",
        ])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path


def benchmark(
    models: Union[str, Sequence[str]],
    prompt: Any = "Reply with exactly: OK",
    requests: int = 8,
    concurrency: Sequence[int] = (1, 4, 8),
    warmup: int = 0,
    show_table: bool = True,
    **completion_kwargs: Any,
) -> BenchmarkResult:
    """Benchmark provider/model routes under sequential and parallel workloads.

    Each scenario sends ``requests`` identical prompts. Concurrency 1 measures a
    sequential workload; larger values measure bounded parallel workloads.
    Retries, caching, and progress are disabled by default. Pass an explicit
    provider-prefixed model to prevent normal bare-model fallback routing from
    mixing providers in a comparison.

    Rows contain aggregate metrics plus ``latencies_s``, ``output_tokens``,
    ``cost``, ``failures``, and ``routes`` for programmatic analysis.
    """
    if isinstance(models, str):
        models = [models]
    else:
        models = list(models)
    levels = [int(value) for value in concurrency]
    if not models:
        raise ValueError("models must contain at least one provider/model route")
    if requests < 1:
        raise ValueError("requests must be at least 1")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")
    if not levels or any(value < 1 for value in levels):
        raise ValueError("concurrency values must be positive integers")

    options = dict(completion_kwargs)
    options.setdefault("show_progress", False)
    options.setdefault("caching", False)
    options.setdefault("num_retries", 0)
    if "max_concurrency" in options:
        raise TypeError("benchmark() controls max_concurrency via concurrency")

    rows = BenchmarkResult()
    for requested_model in models:
        for _ in range(warmup):
            complete(prompt, model=requested_model, **options)

        for level in levels:
            started = time.perf_counter()
            results = complete(
                [prompt for _ in range(requests)],
                model=requested_model,
                max_concurrency=level,
                **options,
            )
            wall_s = time.perf_counter() - started
            successes = [result for result in results if not result.failed]
            latencies = [
                result.latency_s for result in successes
                if result.latency_s is not None
            ]
            tokens = sum(_output_tokens(result) for result in successes)
            routes = sorted(set(result.model_used for result in successes))
            route = routes[0] if len(routes) == 1 else ", ".join(routes)
            provider, _, model_name = (route or requested_model).partition("/")
            if not model_name:
                model_name = provider
                provider = "resolved"
            success_count = len(successes)
            failures = [result for result in results if result.failed]
            error_counts = {}
            for failure in failures:
                name = type(failure.error).__name__
                error_counts[name] = error_counts.get(name, 0) + 1
            rows.append({
                "provider": provider,
                "model": model_name,
                "requested_model": requested_model,
                "mode": "sequential" if level == 1 else "parallel",
                "concurrency": level,
                "requests": requests,
                "successes": success_count,
                "failures": len(failures),
                "success": f"{success_count}/{requests}",
                "wall_s": wall_s,
                "mean_s": sum(latencies) / len(latencies) if latencies else None,
                "p50_s": _percentile(latencies, 50),
                "p95_s": _percentile(latencies, 95),
                "req_s": success_count / wall_s if wall_s else 0.0,
                "tok_s": tokens / wall_s if wall_s else 0.0,
                "output_tokens": tokens,
                "cost": sum(float(result.cost or 0) for result in successes),
                "latencies_s": latencies,
                "routes": routes,
                "errors": error_counts,
                "error": ", ".join(
                    f"{name}×{count}" for name, count in error_counts.items()
                ) or "-",
            })

    if show_table:
        print(rows.to_markdown())
    return rows
