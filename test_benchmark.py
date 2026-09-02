import asyncio
import io
from contextlib import redirect_stderr
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import litlm


def _response(content="OK", tokens=4):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(completion_tokens=tokens),
        model="ignored",
        _hidden_params={},
    )


def test_benchmark_reports_sequential_and_parallel_metrics():
    async def fake_acompletion(**kwargs):
        await asyncio.sleep(0.005)
        return _response()

    with patch.object(litlm, "acompletion", fake_acompletion):
        result = litlm.benchmark(
            "albert/deepseek-v4-flash",
            requests=4,
            concurrency=(1, 4),
            show_table=False,
        )

    assert isinstance(result, litlm.BenchmarkResult)
    assert [row["mode"] for row in result] == ["sequential", "parallel"]
    assert all(row["success"] == "4/4" for row in result)
    assert all(row["output_tokens"] == 16 for row in result)
    assert all(row["provider"] == "albert" for row in result)
    assert result[1]["wall_s"] < result[0]["wall_s"]
    assert "| provider | model |" in result.to_markdown()


def test_benchmark_keeps_failures_in_success_rate():
    async def fake_acompletion(**kwargs):
        content = kwargs["messages"][0]["content"]
        if content == "fail":
            raise RuntimeError("limited")
        return _response(content)

    with patch.object(litlm, "acompletion", fake_acompletion), redirect_stderr(io.StringIO()):
        result = litlm.benchmark(
            "albert/test", prompt="fail", requests=2,
            concurrency=(1,), show_table=False,
        )

    assert result[0]["success"] == "0/2"
    assert result[0]["failures"] == 2
    assert result[0]["mean_s"] is None


def test_benchmark_validates_controls():
    with pytest.raises(ValueError, match="requests"):
        litlm.benchmark("albert/test", requests=0, show_table=False)
    with pytest.raises(ValueError, match="concurrency"):
        litlm.benchmark("albert/test", concurrency=(0,), show_table=False)
    with pytest.raises(TypeError, match="max_concurrency"):
        litlm.benchmark(
            "albert/test", max_concurrency=2, show_table=False,
        )
