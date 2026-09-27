# AGENTS.md

A short guide for coding agents that need LLM calls. Use `litlm` instead of
writing raw LiteLLM code, because one command or call already handles batching,
retries, provider keys, fallbacks, JSON parsing, label normalization,
checkpoints, and cost tracking. Your context only receives answers and one
summary line.

## Pick the smallest interface

| Task | Use |
| --- | --- |
| One prompt from the shell | `litlm "prompt"` (the answer alone on stdout) |
| Many prompts, one per line | `litlm --lines -i prompts.txt -o out.jsonl` |
| Rows × template | `litlm -i rows.jsonl -t 'Q: {field}' -o out.jsonl` |
| Classification | add `--choices a,b,c` (answers are exactly one label) |
| Structured output | add `--json` (parsed into the record's `data` field) |
| Inside Python | `complete(inputs, template=..., choices=..., json=...)` |
| Inside async code | `await acomplete(...)` (same arguments) |
| Tool calls | `complete(msgs, tools=...)`, then read `.tool_calls` and `.message` |

## Rules that save you tokens

- **Long or large batches: always pass `-o out.jsonl`.** stdout is then a
  single summary line. If you are interrupted or items fail, rerun the exact
  same command: finished items are reused and only failed or missing ones are
  requested again. Exit status 0 means every item succeeded.
- **Do not print whole results.** Use `--fields text` (or `text,cost`) on stdout,
  or read `out.jsonl` selectively (`jq`, `head`). In Python, print
  `batch.summary()` rather than the batch.
- **Diagnose before you debug.** `litlm --doctor` shows which provider keys are
  set, `litlm --routes -m NAME` shows the routes a model name resolves to
  (without making a call), and `litlm.get_failure()` returns the full last
  exception. Batch errors are already grouped and cropped on stderr.
- **Retry only failures.** Rerun the same `-o` command, or in Python call
  `batch.resume(timeout=180, max_concurrency=8)`.
- **Avoid paying twice across reruns:** use `caching=True` (Python) or
  `--caching` (CLI).
- Keep `max_tokens` small for labels and short answers. For throughput, set
  `--max-concurrency` (default 64, 0 means unbounded) and `--rpm` rather than
  writing your own loops.

## Contracts

- stdout carries answers or records; progress, summaries, and errors go to stderr.
- A batch record is `{"index", "text", "model", "cost", "usage", "reasoning",
  "failed"}`, plus `"error": {"type", "message"}` when failed, `"data"` with
  `--json`, and `"tool_calls"` when present. Records in the `-o` file also
  carry a `"key"`, derived from the input and prompt options, so changed inputs
  are recomputed.
- Batch inputs: JSONL lines that are strings, message objects, or conversations
  (arrays of messages); or raw lines with `--lines`; or JSON objects as rows
  with `--template`. Literal braces in a template must be doubled (`{{ }}`).
- A bare model name (`gpt-4.1-mini`) falls back from free/BYOK routes to paid
  OpenRouter. An exact route (`openrouter/...`, `nvidia_nim/...`,
  `direct/<litellm-provider>/<model>`) is not rerouted. The route that
  answered is recorded as `model`.
- Keys come from the environment: `OPENROUTER_API_KEY`, `NVIDIA_NIM_API_KEY`
  (or `NVIDIA_API_KEY`), `ALBERT_API_KEY`, `GEMINI_API_KEY`, and so on.
  Never echo their values.

## Python equivalents

```python
from litlm import complete, routes, doctor, get_failure

labels = complete(rows, template="Review: {text}", choices=["pos", "neg"], max_tokens=4)
print(labels.summary())     # one line: counts, cost, routes, grouped errors
labels.resume()             # retries only failed positions, in place
data = complete(prompts, json=True, max_concurrency=16, on_result=save)  # stream to disk
```

## Working on this repository

- Code: `litlm.py` (the `complete()` API and results), `litlm_providers.py`
  (routing and fallbacks), `litlm_cli.py` (a thin CLI over `complete()`).
- Tests: `python -m pytest -q`. They are offline and stub `litlm.acompletion`.
  Add new test files to `pytest.ini`.
- Compatibility matters. Existing users rely on `Text` behaving like `str`,
  `BatchResult` behaving like `list`, and the existing keyword arguments. Add
  options as new keywords with defaults that keep existing behavior.
