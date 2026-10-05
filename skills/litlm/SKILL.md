---
name: litlm
description: Run LLM requests, structured extraction, and resumable batch labeling with litlm, including provider routing, multiple API keys, checkpoints, and ETA reporting.
---

Use `litlm` for the request scheduling and checkpointing; keep dataset-specific
selection, prompts, validation, and merge rules in the user's project.
Find the installed module with `python -c 'import litlm; print(litlm.__file__)'`.
For a source checkout, read its `AGENTS.md` before editing it.

## Run a batch

Choose the inputs and output contract. For datasets, keep stable source IDs,
original labels, and split assignments. Distinguish extracting an existing answer from checking whether the
answer is correct: an extraction prompt must not silently re-solve the question.
Use `--choices` for a fixed label set or `--json` for structured data, then validate
the returned schema and indices before merging. A model saying it cannot extract
an answer is a domain rejection, distinct from a failed API request.

Start with a small representative pilot, including unusual formats. Inspect
selective examples and rejection counts, then scale the same prompt and validation.
Save raw outputs separately from accepted annotations for reproducibility.

For answer checking and bad-example filtering, read
[references/answer-audits.md](references/answer-audits.md). For self-containedness,
presentation and source-grounded repairs, read
[references/dataset-presentation.md](references/dataset-presentation.md). For Albert's DeepSeek
endpoint, read [references/albert-deepseek.md](references/albert-deepseek.md) when
reasoning, structured output, or audit quality matters.

Set `MODEL_ROUTE` to the exact provider route chosen for the task.

```bash
litlm -i rows.jsonl -t 'Classify this text: {text}' --choices yes,no \
  -m "$MODEL_ROUTE" \
  --api-key-envs KEY,KEY_2,KEY_3,KEY_4 \
  --per-key-rpm 40 --num-retries 0 --max-concurrency 16 \
  --max-tokens 16 -o labels.jsonl
```

Choose rate and token limits for the provider and output length. With a key pool,
use an exact route so credentials stay on the intended provider. All named key
variables must be set; duplicate values share one rate slot. `--rpm` is a global
limit, `--per-key-rpm` is per key. Scheduling is local to a call: simultaneous jobs
with the same keys share provider quotas but do not coordinate their limits.
Request pacing does not enforce input-token quotas. Budget the full rendered
prompt, including options and references; large label menus can dominate its size.
Use `--num-retries 0` when each network attempt must be scheduled by the pool.
Quota-exhausted and invalid keys are disabled for the batch; request failures
remain in the checkpoint. Keys belong in the environment, outside tracked files.

Always use `-o` for large batches. Rerun the same command to reuse successful
records and retry failed or missing ones. The CLI prints ETA updates to stderr
every 30 seconds (`--progress-interval`); report them with a wider range during
warmup. Inputs and generation settings identify checkpoint entries; changing key selection does not invalidate completed work.
Inspect the summary and grouped errors instead of printing the whole result file.
Stop repeated retries when the failures require a prompt, model, credential, or
quota change. Keep the completed results and report what remains.
Separate request count from example count when estimating a batched job. Small
pilots may not saturate the configured concurrency, and one slow final request
can distort the extrapolation. Refine the ETA from steady full-job throughput
and budget additional time for confirmations and failed-request retries.

Use `litlm --doctor` for key presence and `litlm --routes -m MODEL` for routing.
In Python, use `complete` or `acomplete` with `api_key_envs`, `per_key_rpm`,
`on_result`, and `batch.resume()`. `on_result` can stream results; durable automatic
resume is provided by the CLI's `-o` checkpoint, not by merely setting `caching=True`.
