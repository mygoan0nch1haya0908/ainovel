# Stage context capacity

Stage roadmap proposals now reserve input from the selected model's configured
context window, minus up to 8,000 output tokens and 1,024 safety tokens. Each
proposal still has at most two attempts and persisted finite aggregate budgets.
Explicit service-level StageBudgets remain supported. Web entry points supply
the provider's local capabilities, or use the immutable selected profile.

Model profiles no longer impose the old 32,000-token ceiling. Enter the actual
capacity supported by the endpoint/model; this is a declaration, not automatic
verification or an expansion of a provider's physical context. Positive signed
32-bit integers are accepted for storage. Existing profile versions are not
modified. The 12,000 output ceiling and author approval gates remain unchanged.
Legacy environment-configured providers keep their configured capacities; use
a saved model profile to configure a larger context through the UI.

The roadmap request replaces identical strings of at least 256 characters with
JSON Pointer references to their first occurrence. The instruction explains the
references. Original frozen snapshots retain all text; no semantic summarization
or truncation occurs. Token estimates are conservative local estimates, not the
provider's tokenizer or billing measurement. Chapter-writing retrieval budgets
are unchanged by this stage-planning fix.

On the stage page, inspect estimated input, effective capacity, output reservation
and cumulative budgets before generating. For PAUSED_CONTEXT_OVERFLOW, follow
the rebuild link to prefill the original architecture/model selection, review
the settings and reconfirm consent, then create a new proposal. This does not
call a model. Old records remain intact. The new proposal freezes current
approved outline/constitution versions; author review is still required.

## Upgrade

Stop the local service only when no generation attempt is running. Back up the
SQLite database using its backup API, run `alembic upgrade head` with the correct
database URL, check integrity/foreign keys, then restart the service. Migration
0007 replaces only the profile context CHECK and preserves other constraints.
Downgrade refuses when any profile exceeds 32,000; it never truncates metadata.
Restore a pre-upgrade backup if a rollback is required in that situation.

## Offline verification

The regression suite covers large profile save/dispatch, stage requests above
the old 16K cap, lossless request deduplication with unchanged snapshots, paused
task prefilling without network calls, and migration upgrade/downgrade guards.
No real-model generation is part of these tests.
