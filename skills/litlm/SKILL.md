---
name: litlm
description: Run resumable LLM labeling, structured extraction, relabeling, and answer audits over large datasets with litlm, including multiple provider keys and per-key rate limits.
---

Use `litlm` for the request scheduling and checkpointing; keep dataset-specific
selection, prompts, validation, and merge rules in the user's project.
Find the installed module with `python -c 'import litlm; print(litlm.__file__)'`.
For a source checkout, read its `AGENTS.md` before editing it. The local checkout
in this workspace is `/mnt/nfs_share_magnet2/dsileo/libs/litlm`.

## Run a labeling job

Choose the target rows and stable source IDs. Preserve original labels and split
assignments. Distinguish extracting an existing answer from checking whether the
answer is correct: an extraction prompt must not silently re-solve the question.
Use `--choices` for a fixed label set or `--json` for structured data, then validate
the returned schema and indices before merging. A model saying it cannot extract
an answer is a domain rejection, distinct from a failed API request.

Start with a small representative pilot, including unusual formats. Inspect
selective examples and rejection counts, then scale the same prompt and validation.
For MC extraction, check the gold index against the actual option order, reject
multiple-answer annotations, and verify that extracted text comes from the source.
Save raw outputs separately from accepted annotations for reproducibility.

For answer-correctness audits, hide the source gold in the screening prompt and
compare the model's final answer with gold in code. Showing gold first can make
the model defend it despite contradictory evidence. In confirmation, show gold
and the proposed flag and ask the model to challenge the flag. Repeating the same
model is not independent evidence: it can repeat a systematic mistake. Keep
defensible answers and unresolved disputes; sample confirmed flags before applying
a removal manifest.

Number MC options explicitly. Validate that every input ID occurs exactly once,
the answer index is in range, and the verdict is consistent with the answer.
Valid JSON and a successful API call do not establish these invariants. Reject
partial batches and mark their checkpoint records failed so reruns request them
again. Preserve the original response for diagnosing semantic validation failures.

Reasoning and structured-output controls depend on the endpoint. Verify the
current provider documentation, then probe known examples before scaling. A
parameter being accepted does not prove it took effect; compare quality, token
usage, finish reasons, and available reasoning metadata. Missing reasoning metadata
alone does not prove reasoning was disabled. If combining reasoning and constrained
JSON produces repeated prefixes, malformed output, or timeouts, test the modes
separately rather than paying for a large batch of the same failure.

```bash
litlm -i rows.jsonl -t 'Classify this text: {text}' --choices yes,no \
  -m albert/deepseek-v4-flash-0731 \
  --api-key-envs KEY,KEY_2,KEY_3,KEY_4 \
  --per-key-rpm 40 --num-retries 0 --max-concurrency 16 \
  --max-tokens 16 -o labels.jsonl
```

Choose rate and token limits for the provider and output length. With a key pool,
use an exact route so credentials stay on the intended provider. All named key
variables must be set; duplicate values share one rate slot. `--rpm` is a global
limit, `--per-key-rpm` is per key. Scheduling is local to a call: simultaneous jobs
with the same keys share provider quotas but do not coordinate their limits.
Use `--num-retries 0` when each network attempt must be scheduled by the pool.
Quota-exhausted and invalid keys are disabled for the batch; request failures
remain in the checkpoint. Keys belong in the environment, outside tracked files.

Always use `-o` for large batches. Rerun the same command to reuse successful
records and retry failed or missing ones. The CLI prints observed-throughput ETA updates to stderr
every 30 seconds (`--progress-interval`); report these estimates to the user,
with a wider range while the pilot or job is still warming up. Inputs and generation settings identify
checkpoint entries; changing key selection does not invalidate completed work.
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

Before publishing a derived dataset, document source revision, prompt/model,
retained and rejected counts, validation rules, changed labels, and split handling.
Publishing uses the user's existing authorization and destination.
