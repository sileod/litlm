# Dataset presentation audits

Keep presentation checks separate from solving or relabeling. Compare the full
source with the prepared input; missing context may have been stranded after
the options rather than absent from the source. Accept ordinary domain knowledge,
readable tables, harmless markup and valid true/false statements.

Before accepting an example, check that referenced targets are identifiable:
underlined words, highlighted spans, a given example or a figure must either be
present or unnecessary for the actual claim. A whole sentence does not identify
its missing underlined target. Supplied numbers may suffice to check arithmetic
while still being insufficient to validate how those numbers were obtained.

Use distinct outcomes for usable, source-repairable, missing-context, broken and
uncertain. Missing information cannot be repaired by guessing. Challenge flags
individually and inspect a bounded random sample of passed examples too: precision
on flags does not measure misses. Fix compact criteria before adding a long list
of dataset-specific exceptions. Rechecking examples that informed a prompt change
is a regression check, not an independent estimate of quality.

Preserve source IDs, splits, answers, option order and original text. Prefer exact
source excerpts for moving context or trimming options. Validate option counts,
source grounding, mathematical expressions, numbers and polarity before accepting
an edit; check that the resulting example preserves the original meaning. A clean
rewrite can still change the task. Keep rejected edits separate from API failures.

Maintain separate unfiltered and filtered configurations, an explicit exclusion
manifest, a repair manifest and provenance with model, prompt/settings hashes,
counts and source revision. Mark uncertain or unrepaired retained cases in the
documentation. A successful API request is not a successful semantic validation;
resume partial or invalid batches before publishing complete-coverage claims.
