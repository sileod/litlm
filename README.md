# litlm

`litlm` is a small, notebook-first interface to [LiteLLM](https://github.com/BerriAI/litellm). One function handles a prompt or a parallel batch, while keeping costs, provider metadata, failures, and retries close at hand.

```python
from litlm import complete

answer = complete("What is 2 + 2?")

answers = complete(
    ["Summarize Ada Lovelace", "Summarize Alan Turing"],
    model="gpt-4.1-mini",
    max_concurrency=16,
)
```

It is designed for exploratory work where the full SDK response is useful, but SDK ceremony is not.

## Install

```bash
pip install litlm
```

Set the keys for the providers you use:

```python
import os

os.environ["OPENROUTER_API_KEY"] = "sk-or-..."
os.environ["NVIDIA_NIM_API_KEY"] = "nvapi-..."      # optional
os.environ["ALBERT_API_KEY"] = "..."                # optional
```

## Why litlm

- A string in, a string-like result out.
- Lists, NumPy arrays, and Pandas Series run as ordered async batches.
- Compact progress shows cost and a bounded error breakdown.
- Partial batches stay usable and can retry only failed positions.
- Results expose usage, reasoning, cost, model, and the raw LiteLLM response.
- Bare model names can resolve through free and paid provider fallbacks.
- The typed signature and docstring work well with Jupyter completion and Shift-Tab help.

## Results that remain simple

A scalar result behaves like `str`:

```python
answer = complete("Write a haiku")

print(answer)
print(answer.model_used)
print(answer.cost)
print(answer.usage)
print(answer.reasoning)
print(answer.call_id)
```

Any other response field remains accessible through the same object.

Batch results behave like an ordinary `list`, so existing Python and Pandas code continues to work:

```python
answers = complete(["Capital of France?", "Capital of Japan?"])

answers[0]
len(answers)
df["answer"] = answers
isinstance(answers, list)  # True
```

## Resilient batches

One failed request does not discard the rest of a batch. Failed positions are empty-string-compatible objects with the original exception and prompt attached, so output order and length remain stable.

During a batch, the progress line stays bounded while showing cost, failure rate, error types, and the beginning of a representative message:

```text
Completing: 95%|...| cost=$0.126242, ⚠ 375/755 (49.7%), Timeout×375 | Timeout Error: OpenRouter…
```

Retry only the positions that failed, optionally with safer settings:

```python
answers.resume(
    timeout=180,
    num_retries=5,
    max_concurrency=8,
)

answers.failures  # failures still present after the retry
```

`resume()` updates the same list-compatible result in place. Successful answers are neither requested again nor reordered.

For the latest full provider exception:

```python
import litlm

print(litlm.get_failure())
```

Or inspect every failed item and its metadata:

```python
failures = litlm.get_failures()
print(failures[-1].error)
print(failures[-1].prompt)
```

## Model routing

Use a bare model name when you want `litlm` to find a suitable route:

```python
complete("Hello", model="gpt-4.1-mini")
complete("Hello", model="deepseek-v4-flash")
complete("Hello", model="haiku")
```

Depending on availability and configured keys, bare names are tried through Albert, NVIDIA NIM, OpenRouter free models, then paid OpenRouter models.

Use an exact slug when routing should be explicit:

```python
complete("Hello", model="openrouter/anthropic/claude-sonnet-4")
complete("Hello", model="nvidia_nim/deepseek-ai/deepseek-r1")
```

To bypass litlm routing and call a LiteLLM provider directly, prefix the exact
LiteLLM route with `direct/`:

```python
complete("Hello", model="direct/gemini/gemini-3.7-flash")
complete("Hello", model="direct/openai/gpt-5.6-luna")
```

The returned `Text.model_used` records the route that answered.

Bare model names use a free/BYOK-first fallback hierarchy: Albert, NVIDIA NIM,
direct Gemini when `GEMINI_API_KEY` is available, OpenRouter free, then paid
OpenRouter. Provider/model names remain exact and do not fall back. For a fully
explicit hierarchy, pass exact routes in order:

```python
complete(
    "Hello",
    model="gemini-3.7-flash",
    fallbacks=[
        "direct/gemini/gemini-3.7-flash",
        "openrouter/google/gemini-3.7-flash",
    ],
)
```

Fallback is per item. If a route reports exhausted quota or credits, litlm
disables it for the rest of that batch so later items proceed directly to the
next route. Already in-flight requests may still settle. Set `attempt_timeout`
for a hard wall-clock bound around providers that fail to honor their own
request timeout.

## Useful controls

Common options are explicit and typed; additional LiteLLM parameters pass through unchanged:

```python
answer = complete(
    "Explain the result briefly",
    system="You are a careful mathematician.",
    model="openrouter/deepseek/deepseek-v4-flash",
    reasoning_effort="none",
    temperature=0.2,
    max_tokens=512,
    timeout=60,
)
```

Request and parse JSON directly:

```python
data = complete(
    "Return a JSON object with a string field named topic",
    json=True,
)
```

Throttle large batches by concurrency or request starts per minute:

```python
answers = complete(inputs, max_concurrency=12, rpm=120)
```

Persist or stream results as soon as each item settles without coupling
`litlm` to an application's storage format:

```python
def save_result(index, result):
    if not result.failed:
        checkpoint(index, str(result), result.usage)

answers = complete(inputs, max_concurrency=12, on_result=save_result)
```

The callback receives the original input index and a `Text` or `Failure`.
`BatchResult.resume()` preserves original indexes when it retries failed items.

## Caching

Local response caching avoids paying twice for identical calls and survives notebook restarts:

```python
answer = complete("Expensive stable query", caching=True)
```

Provider-side prompt caching is separate:

```python
complete("Question over stable context", prompt_cache=True)
complete("Question over stable context", prompt_cache="1h")
complete(
    "Question over stable context",
    cache_control={"type": "ephemeral", "ttl": "1h"},
)
```

## History and cost

```python
from litlm import cost_breakdown, get_history

last_result = get_history()
first_result = get_history(0)

cost_breakdown("session")
cost_breakdown("day")
cost_breakdown("week", by="day")
```

Cost history is lightweight and in memory for the current Python process.

## Provider benchmarks

`benchmark` compares exact provider/model routes with the same workload. A
concurrency of 1 is sequential; larger values are bounded parallel batches.
It prints a Markdown table and returns every aggregate and per-request metric.

```python
from litlm import benchmark

results = benchmark(
    [
        "nvidia_nim/moonshotai/kimi-k3",
        "nvidia_nim/deepseek-ai/deepseek-v4-pro-0813",
        "albert/deepseek-v4-flash",
    ],
    requests=8,
    concurrency=(1, 4, 8),
    max_tokens=32,
    timeout=120,
)

# Useful in notebooks, scripts, and agent handoffs.
results.to_markdown()
results.save_markdown("benchmarks/BENCHMARK_RESULTS.md", notes="Short response workload.")
results[0]["latencies_s"]
```

The table reports success count, whole-workload time, mean/p50/p95 end-to-end
latency, successful requests per second, and output tokens per second. Exact
routes are important: a bare model name may use litlm's normal provider fallback
chain. API keys are read from the existing provider environment variables and
are never included in the result rows or report.
