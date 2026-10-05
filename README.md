# litlm

`litlm` is a small interface to [LiteLLM](https://github.com/BerriAI/litellm). It works well both interactively and for coding agents. One function handles a single prompt or a parallel batch, and keeps costs, provider metadata, failures, and retries close at hand.

```python
from litlm import complete

answer = complete("What is 2 + 2?")

answers = complete(
    ["Summarize Ada Lovelace", "Summarize Alan Turing"],
    model="gpt-4.1-mini",
    max_concurrency=16,
)
```

It is designed for work where the full SDK response is useful, but SDK
ceremony is not. The same property makes it cheap for coding agents. An agent that needs LLM calls
can run a single `litlm` command or `complete()` call instead of writing and
debugging a LiteLLM script with async batching, retries, key lookups, JSON
parsing, and checkpoints. Output stays compact: answers go to stdout, and a
one-line summary replaces pages of tracebacks. See [AGENTS.md](AGENTS.md) for
the agent-facing guide.

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

## Command line

Installation also provides a thin `litlm` command that delegates to `complete()`:

```bash
litlm "Explain transformers briefly"
printf 'Explain transformers briefly' | litlm
litlm "Hello" --model gpt-4.1-mini
litlm "Hello" --output json
```

Text output contains only the answer on stdout. Progress, debug output, and errors go to stderr, so shell pipelines and agent subprocesses can consume stdout directly. `--output json` returns a stable envelope with `text`, `model`, `cost`, `usage`, `reasoning`, and `failed` fields.

JSONL stdin runs a batch and defaults to JSONL output:

```bash
printf '"Capital of France?"\n"Capital of Japan?"\n' | litlm --input-jsonl
```

Each JSONL line may be a JSON string, a message object, or a conversation represented as an array of message objects. A batch must use one input shape consistently. Any failed batch item makes the process exit nonzero while preserving successful output rows.

`--json` is separate from `--output json`: it asks the model for JSON and parses the response, while `--output` controls the CLI serialization. Common `complete()` controls are exposed as matching flags; additional LiteLLM arguments can be passed with repeatable `--param KEY=VALUE` options.

### For agents and long batches

```bash
# One prompt per line from a file; records are checkpointed as they settle.
litlm --lines -i prompts.txt -o answers.jsonl
# -> 998/1000 ok (0 reused), 2 failed, cost=$0.041210 -> answers.jsonl

# Rerun the same command to retry only failed or missing items.
litlm --lines -i prompts.txt -o answers.jsonl
# -> 1000/1000 ok (998 reused), 0 failed, cost=$0.041290 -> answers.jsonl

# Fill a template from JSONL rows and normalize each answer to one label.
litlm -i reviews.jsonl -t 'Review: {text}' --choices positive,negative --output text

# Keep only the fields you need on stdout.
litlm --lines -i prompts.txt --fields text,cost

litlm --routes -m deepseek-v4-flash   # provider routes a bare name resolves to
litlm --doctor                        # which provider keys are set (never their values)
```

With `--out`, stdout carries only the summary line. Each record includes its
`index` and a key derived from the input and prompt options, so a changed input
line is recomputed instead of silently reused. The exit status is nonzero while
any item is still failed or missing.

### Adaptive concurrency

For long jobs with variable latency or congestion, opt in with
`--adaptive-concurrency --max-concurrency 128` (Python:
`adaptive_concurrency=True, max_concurrency=128`). It starts at up to eight
concurrent requests, grows after low-error completion windows, and halves
concurrency on a 429 or repeated timeout/server errors. Already running requests
finish normally; failures from that previous wave do not trigger repeated backoff.
Progress lines show the current concurrency and ceiling. Python batch results
also expose `batch.tuning` and include the final concurrency in `batch.summary()`.

The positive `max_concurrency` value is a hard ceiling; configured `rpm` and
`per_key_rpm` limits remain unchanged. Prompts, models, output limits and examples
per request remain unchanged. Use `num_retries=0` to expose each provider attempt
to the scheduler; hidden provider retries can mask congestion. Authentication,
exhausted quota and malformed-input errors do not drive concurrency tuning.
This is a per-call heuristic, not a throughput-optimality guarantee or a token
quota controller. It is most useful for long batches; small jobs may finish before
it learns. Separate calls/processes still do not coordinate quotas. Adaptive mode
is off by default, and changing it does not invalidate CLI checkpoints.

### Multiple keys

For large jobs on one provider, name the environment variables holding its keys:

```bash
litlm -i rows.jsonl -t 'Classify: {text}' --choices yes,no \
  -m albert/deepseek-v4-flash-0731 \
  --api-key-envs KEY,KEY_2,KEY_3,KEY_4 \
  --per-key-rpm 40 --max-concurrency 16 --num-retries 0 -o labels.jsonl
```

The pool balances request starts across keys and paces each key independently.
`--rpm` remains an optional global limit. Quota-exhausted or invalid keys are
disabled for the batch, and another key on the same provider is tried; ordinary
request failures remain retryable through the checkpoint. The pool requires an
exact provider route and every named variable to be set. Duplicate key values
share one slot. Limits apply to each `complete()` call; separate processes do
not coordinate their quotas. Use `--num-retries 0` for pacing every attempt;
LiteLLM's internal retries otherwise happen inside a reserved request slot.

In Python, use `complete(..., api_key_envs=[...], per_key_rpm=40)`.
Keys are read from the environment at runtime; checkpoint keys and resume
options contain the environment variable names, not their values. Changing the
model or generation settings invalidates completed CLI records.
CLI batches also report progress and estimated remaining time on stderr every
30 seconds (`--progress-interval`), including background jobs. The ETA uses
observed completions and includes pacing; early estimates can fluctuate.
Pooled records include `key_env` and `latency_s` so throughput and failures can
be compared by key without exposing credentials.

The reusable coding-agent skill is in [skills/litlm/SKILL.md](skills/litlm/SKILL.md).

## Why litlm

- A string in, a string-like result out.
- Lists, NumPy arrays, and Pandas Series run as ordered async batches.
- Compact progress shows cost and a bounded error breakdown.
- Partial batches stay usable and can retry only failed positions.
- `template=` and `choices=` cover the common "fill rows, classify" batch without extra code.
- `summary()`, `routes()`, `doctor()`, and the CLI give compact output that suits agent contexts.
- Results expose usage, reasoning, cost, model, and the raw LiteLLM response.
- Bare model names can resolve through free and paid provider fallbacks.
- `acomplete()` is the native async API; `complete()` is its synchronous wrapper.
- The typed signature and docstring work well with editor completion and inline help.

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

For tool use, the assistant message is first-class. litlm does not run an
agent loop; it passes `tools=` through and exposes what came back:

```python
answer = complete(messages, tools=tools)

answer.tool_calls      # [] when the model answered in text
answer.message         # assistant message, ready to append to `messages`
answer.finish_reason
answer.raw             # the full LiteLLM response
```

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

Fill a template from rows (dicts, a DataFrame, or plain values via `{input}`),
and constrain answers to a label set:

```python
labels = complete(
    df,                                   # or a list of dicts
    template="Review: {text}\nSentiment?",
    choices=["positive", "negative", "neutral"],
    max_tokens=8,
)
labels.summary()   # '1000/1000 ok | cost=$0.012 | routes: openrouter/...×1000'
```

Every successful item is exactly one of the labels, and `.raw_text` keeps the
model's original reply. A reply that names no label, or several, becomes a
`Failure`, so `labels.resume()` retries only those items. With a template, a
list of dicts is a batch of rows, not a conversation. Literal braces in a
template must be doubled (`{{` and `}}`).

Request and parse JSON directly:

```python
data = complete(
    "Return a JSON object with a string field named topic",
    json=True,
)
```

In a batch, a reply that cannot be parsed becomes a resumable `Failure`
instead of aborting the whole batch. A scalar call still raises `ValueError`.

Throttle large batches by concurrency or request starts per minute.
Batches run at most 64 requests at a time by default. Pass
`max_concurrency=None` (or `0`) for unbounded concurrency:

```python
answers = complete(inputs, max_concurrency=12, rpm=120)
```

## Async

In async code (agent runtimes, web servers), await `acomplete()`. It takes the
same arguments and returns the same results:

```python
from litlm import acomplete

answer = await acomplete("Hello")
batch = await acomplete(prompts, choices=["yes", "no"])
await batch.aresume()
```

The synchronous `complete()` runs on a private event loop. It applies
`nest_asyncio` only when it is called from inside an already running loop, as
in Jupyter; importing litlm patches nothing. LiteLLM's debug logging and a few
pydantic serialization warnings are silenced by default. Set `LITLM_QUIET=0`
before import to leave them untouched.

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

Local response caching avoids paying twice for identical calls and survives process restarts:

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

# Useful in reports, scripts, and agent handoffs.
results.to_markdown()
results.save_markdown("benchmarks/BENCHMARK_RESULTS.md", notes="Short response workload.")
results[0]["latencies_s"]
```

The table reports success count, whole-workload time, mean/p50/p95 end-to-end
latency, successful requests per second, and output tokens per second. Exact
routes are important: a bare model name may use litlm's normal provider fallback
chain. API keys are read from the existing provider environment variables and
are never included in the result rows or report.
