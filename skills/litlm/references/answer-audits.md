# Answer correctness audits

Use this workflow when checking existing gold answers or filtering incorrectly
labeled or badly presented examples.

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
Prefer structured JSON with an explicit list of required example indices. A
successful response may cover only the first example or omit later ones; API
success and coverage are separate checks. Save each response immediately rather
than waiting for a large chunk. Replay valid saved responses before requesting
missing annotations again.

Keep extraction and re-solving separate. A parsing task should recover the source
answer; an answer audit evaluates whether it is correct. Retain the original
question, answer, split, and stable ID in derived data.

Check missing passages, images, tables, or definitions separately from answer
correctness. A hard question is not malformed merely because the model cannot
solve it. Use an uncertain verdict and keep unresolved cases. For binary data,
ensure the prompt actually supports the offered yes/no or true/false answers.

For MC extraction, preserve source option order when mapping gold, remove the
options from the model's question field, and verify extracted text against the
source. Do not guess labels for multiple-answer annotations.

Produce a removal manifest containing IDs, reasons, and model/prompt provenance.
Validate the filter locally before updating a published dataset. Review a sample
of the strongest proposed removals as well as disagreements; agreement between
two prompts to the same model can still be systematically wrong.

For a derived release, document source revision, prompt/model, retained and
rejected counts, validation rules, changed labels, and split handling. Publishing
uses the user's existing authorization and destination.
