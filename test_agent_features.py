import asyncio
import io
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import litlm


def _response(content):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)], usage=None,
        model="test-model", _hidden_params={"response_cost": 0.5},
    )


@pytest.fixture(autouse=True)
def clean_state():
    for store in (litlm._HISTORY, litlm._COSTS, litlm._FAILURES, litlm._MODEL_USED):
        store.clear()


def fake(replies, calls=None):
    """acompletion stub answering from `replies`, keyed by the last user message."""
    async def acompletion(**kwargs):
        content = kwargs["messages"][-1]["content"]
        if calls is not None:
            calls.append(content)
        await asyncio.sleep(0)
        reply = replies(content) if callable(replies) else replies
        return _response(reply)
    return acompletion


def run(inputs, replies, calls=None, **kwargs):
    kwargs.setdefault("model", "openrouter/test")
    kwargs.setdefault("show_progress", False)
    with patch.object(litlm, "acompletion", fake(replies, calls)), redirect_stderr(io.StringIO()):
        return litlm.complete(inputs, **kwargs)


def test_template_fills_dict_rows_as_a_batch():
    calls = []
    out = run([{"name": "Ada"}, {"name": "Alan"}], lambda c: c.upper(), calls, template="Hi {name}")
    assert calls == ["Hi Ada", "Hi Alan"]
    assert out == ["HI ADA", "HI ALAN"]


def test_template_fills_scalars_and_single_rows():
    assert run("x", lambda c: c, template="<{input}>") == "<x>"
    assert run({"a": 1}, lambda c: c, template="a={a}") == "a=1"


def test_template_accepts_dataframe_rows():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"q": ["one", "two"]})
    assert run(df, lambda c: c, template="Q: {q}") == ["Q: one", "Q: two"]


def test_choices_normalize_labels_and_instruct_once():
    calls = []
    out = run(["a", "b"], lambda c: "**Positive**." if c.startswith("a") else "I'd say negative here",
              calls, choices=["positive", "negative"])
    assert out == ["positive", "negative"]
    assert out[1].raw_text == "I'd say negative here"
    assert calls[0].endswith("Answer with exactly one of: positive, negative.")


def test_unparseable_choice_becomes_resumable_failure():
    replies = iter(["maybe", "positive"])
    calls = []
    out = run(["a"] * 1 + ["b"], lambda c: "negative" if c.startswith("b") else next(replies),
              calls, choices=["positive", "negative"])
    assert out[0].failed and isinstance(out[0].error, ValueError)
    assert out[1] == "negative"
    with patch.object(litlm, "acompletion", fake(lambda c: next(replies), calls)), redirect_stderr(io.StringIO()):
        out.resume()
    assert out == ["positive", "negative"]
    # the instruction is not appended a second time on resume
    assert calls[-1].count("Answer with exactly one of") == 1


def test_json_parse_failure_does_not_discard_batch():
    out = run(["a", "b"], lambda c: '{"k": 1}' if c == "a" else "no json", json=True)
    assert out[0] == {"k": 1}
    assert out[1].failed


def test_scalar_parse_failure_still_raises():
    with pytest.raises(ValueError):
        run("a", "not json", json=True)


def test_json_and_choices_are_exclusive():
    with pytest.raises(ValueError):
        litlm.complete("a", json=True, choices=["x"])


def test_summary_is_one_line():
    out = run(["a", "b"], lambda c: "ok")
    summary = out.summary()
    assert "\n" not in summary
    assert summary.startswith("2/2 ok | cost=$1.000000 | routes: openrouter/test×2")


def test_routes_and_doctor(monkeypatch):
    assert litlm.routes("openrouter/x/y") == ["openrouter/x/y"]
    assert litlm.routes("x", fallbacks=["direct/a/b", "direct/a/b", "openrouter/c"]) == [
        "direct/a/b", "openrouter/c"]
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status = litlm.doctor()
    assert status["gemini"] is True and status["anthropic"] is False


# --- async API, tool calls, concurrency, import hygiene ---
import inspect
import subprocess
import sys


def test_sync_and_async_signatures_match():
    assert inspect.signature(litlm.complete) == inspect.signature(litlm.acomplete)
    assert "synchronously" in litlm.complete.__doc__


def test_acomplete_runs_inside_a_running_loop_without_nest_asyncio():
    async def main():
        with patch.object(litlm, "acompletion", fake(lambda c: c * 2)):
            batch = await litlm.acomplete(["a", "b"], model="openrouter/t", show_progress=False)
            return batch, getattr(asyncio.get_running_loop(), "_nest_patched", False)

    batch, patched = asyncio.run(main())
    assert batch == ["aa", "bb"] and not patched


def test_aresume_retries_failures_in_a_running_loop():
    replies = iter(["no json", '{"ok": 1}'])

    async def main():
        with patch.object(litlm, "acompletion", fake(lambda c: '{"ok": 0}' if c == "a" else next(replies))), \
                redirect_stderr(io.StringIO()):
            batch = await litlm.acomplete(["a", "b"], model="openrouter/t", json=True, show_progress=False)
            assert batch[1].failed
            await batch.aresume()
            return batch

    assert asyncio.run(main()) == [{"ok": 0}, {"ok": 1}]


def test_import_does_not_patch_asyncio_or_environment():
    code = (
        "import os, asyncio; before = dict(os.environ); import litlm; "
        "assert dict(os.environ) == before; "
        "assert not hasattr(asyncio, '_nest_patched'); print('ok')"
    )
    env = {k: v for k, v in __import__("os").environ.items() if k != "NVIDIA_NIM_API_KEY"}
    env["NVIDIA_API_KEY"] = "x"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.stdout.strip() == "ok", out.stderr


def test_tool_calls_are_first_class():
    call = SimpleNamespace(id="c1", type="function",
                           function=SimpleNamespace(name="f", arguments='{"x": 1}'))

    async def acompletion(**kwargs):
        assert kwargs["tools"] == [{"type": "function"}]
        message = SimpleNamespace(content=None, tool_calls=[call])
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
                               usage=None, model="m", _hidden_params={})

    with patch.object(litlm, "acompletion", acompletion):
        answer = litlm.complete("hi", model="openrouter/t", tools=[{"type": "function"}])
    assert answer == "" and answer.tool_calls == [call]
    assert answer.finish_reason == "tool_calls"
    assert answer.message.tool_calls == [call] and answer.raw.model == "m"


def test_default_concurrency_is_bounded():
    active = peak = 0

    async def acompletion(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        active -= 1
        return _response("ok")

    with patch.object(litlm, "acompletion", acompletion):
        litlm.complete(["x"] * 200, model="openrouter/t", show_progress=False)
        assert peak == 64
        peak = 0
        litlm.complete(["x"] * 200, model="openrouter/t", show_progress=False, max_concurrency=None)
        assert peak == 200
