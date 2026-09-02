# Provider benchmark results

Generated 2026-09-02T12:24:16+02:00.

Workload: 8 identical short deterministic-answer requests per concurrency level; max_tokens=32. NVIDIA scenarios are pending because NVIDIA_NIM_API_KEY is not configured in this environment.

| provider | model | mode | concurrency | success | wall_s | mean_s | p50_s | p95_s | req_s | tok_s | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| albert | deepseek-v4-flash | sequential | 1 | 8/8 | 2.324 | 0.290 | 0.172 | 0.781 | 3.442 | 3.442 | - |
| albert | deepseek-v4-flash | parallel | 4 | 8/8 | 0.736 | 0.352 | 0.348 | 0.506 | 10.874 | 10.874 | - |
| albert | deepseek-v4-flash | parallel | 8 | 8/8 | 0.750 | 0.325 | 0.195 | 0.738 | 10.662 | 10.662 | - |

`mean_s`, `p50_s`, and `p95_s` are end-to-end request latency; `req_s` and `tok_s` use whole-scenario wall time.

## Pending NVIDIA NIM targets

No NVIDIA result is recorded yet because `NVIDIA_NIM_API_KEY` was absent. The
current NVIDIA catalog lists both targets as free endpoints:

- `nvidia_nim/moonshotai/kimi-k3`
- `nvidia_nim/deepseek-ai/deepseek-v4-pro-0813`

After setting the key, append a fresh comparison by running:

```python
from litlm import benchmark

nim = benchmark(
    [
        "nvidia_nim/moonshotai/kimi-k3",
        "nvidia_nim/deepseek-ai/deepseek-v4-pro-0813",
    ],
    requests=8,
    concurrency=(1, 4, 8),
    max_tokens=32,
    timeout=120,
)
```
