# Albert / DeepSeek endpoint notes

Read when using `albert/deepseek-v4-flash-0731`, particularly for answer audits.
These are observations from probes on 2026-10-05; verify against the live endpoint
before assuming they apply to another model or deployment.

The `/v1/models` endpoint exposed `deepseek-v4-flash-0731` to all four configured
keys. `litlm` reaches the exact route through its OpenAI-compatible handler.
The key environment names are arbitrary (`KEY`, `KEY_2`, etc.). Choose per-key
limits for the current provider quotas and other jobs using the same accounts.

Non-thinking answer audits returned plausible but incorrect explanations on
known examples. Repeating the same model often repeated the error. Blind solving
plus a conservative confirmation prompt reduced false alarms, but prompt changes
alone did not establish reliable filtering.

In probes, `extra_body={"thinking": {"type": "enabled"}}` did not yield reasoning
metadata or fix those examples. Enabling the chat template's thinking mode while
constraining JSON produced duplicate prefixes, parse failures, and timeouts.
This combination succeeded and corrected the two test errors:

```python
complete(
    prompts,
    model="albert/deepseek-v4-flash-0731",
    json=True,
    response_format={"type": "text"},
    extra_body={"chat_template_kwargs": {"thinking": True}},
    api_key_envs=["KEY", "KEY_2", "KEY_3", "KEY_4"],
    per_key_rpm=35,
    num_retries=0,
    max_tokens=8192,
    timeout=300,
    attempt_timeout=330,
)
```

The prompt still requests JSON. `response_format=text` disables constrained
decoding; `json=True` parses the final content locally. Validate IDs, types,
verdicts, and option bounds after parsing. Test smaller batches before scaling:
reasoning takes longer and can exhaust the token or time budget.

The successful probes exposed reasoning metadata, but reported completion-token
usage did not include all of that text. Estimate throughput from measured
latency and settled requests rather than trusting those token counts alone.
Previously measured non-thinking throughput does not predict thinking throughput.

Useful primary references: [vLLM reasoning outputs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)
and [DeepSeek thinking controls](https://api-docs.deepseek.com/guides/thinking_mode/).
Native DeepSeek API controls and an OpenAI-compatible hosted deployment can
behave differently; an HTTP success alone does not verify the requested mode.
