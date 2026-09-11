# Phase 2 operational boundaries and final rulings

The final integration repair preserves both author approval gates, existing
approved chapter text, persisted model response metadata, and existing prompt
snapshots. It does not add a Phase 3 fact system or change database metadata.

## Claims and project ownership

Every model attempt start and pre-provider pause requires the revision returned
by the original step claim. Capture that integer before any further work; do not
read the current revision to renew a claim. The step UPDATE checks that revision
and the live lease, and attempt start also atomically rejects any existing
RUNNING attempt. A worker name alone is not an execution identity.

Manual candidate creation requires a project with no active workflow. A workflow
source requires the same project owner and a source workflow currently in
CREATING_CANDIDATE_BATCH, checked in the project UPDATE. Legacy manual creation
on an idle project retains the Phase 1 path.

Cost: service callers must pass their original claim revision and cannot attach
arbitrary provenance strings to manual batches. No new lease column is needed.

## Structured output and model windows

New ChapterSummaryDelta request schemas encode `state_delta` as a JSON string
containing an object. For example, the wire value `"{\"door\":\"north\"}"`
becomes the domain dictionary `{"door": "north"}` after validation. Nested keys,
arrays, nulls, numbers, booleans, and Unicode remain intact. Domain persistence
and existing Fake results still use dictionaries. Ollama follows the same new
wire schema, and its output is decoded at the same typed validation boundary.

All objects in new Agent request schemas are closed and their fields required,
as required by [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
The schema stored in each new run snapshot is the actual schema sent to the
provider. Existing snapshots are immutable: older runs keep their original
contract; a run created with an obsolete OpenAI schema must be replaced after
its pause, rather than rewriting the historical snapshot.

Ollama `num_ctx` uses the complete configured capability window; `num_predict`
remains the reserved output maximum. With a 16,000 window, 4,000 output and the
shared 1,024 safety reserve, input capacity is 10,976. Direct adapter calls that
exceed the available input or output capability fail before HTTP dispatch.

Cost: the JSON string requires escaping and decoding. Model capability ceilings
remain operator configuration, not automatic model discovery. Real OpenAI calls
and optional local Ollama inference are not part of offline acceptance.

## Approved memory between batches

The official index derives summary and event entries from Phase 1 approved
batches and official/published chapters. Each chapter must map to the same
project's source workflow and its accepted active draft and summary artifacts.
The final approved body must match the draft text and SHA-256 hash; the existing
chapter delta must match the validated summary delta. Edited candidates that no
longer match their generated draft, rejected batches, foreign projects, and
inactive or unvalidated artifacts are excluded.

The index selects the most recent 30 formal chapters, retains their existing
state deltas as event entries, and adds summaries for only the most recent 5.
Retrieval orders these entries by formal chapter number, independent of batch
ordinal, revision or insertion time. Each entry and packet retains its artifact
identity, version fingerprint, content hash and canonical official chapter ID.
The existing hard context budget may trim optional entries further. Historical
full text remains a search source and is not automatically put into model input.

Cost: manual Phase 1 chapters without validated workflow summaries and chapters
edited before approval receive no derived summary/delta memory in this phase.
There is no automatic resummarization, fact merging, character knowledge graph,
or resource ledger. Rebuilding the existing raw-text index still scans stored
official chapters; the new derived memory work is bounded to 30 chapters.

## Cancelling pauses

Authors can explicitly cancel a paused workflow through a confirmed,
CSRF-protected action. Cancellation records an author audit event, preserves
attempts, artifacts and snapshots, and releases only matching project ownership.
The project can then start a new workflow.

If a source candidate batch exists, the author must decide that batch through
Phase 1 and synchronize the outcome. Cancellation cannot bypass that decision.
A candidate-ready crash followed by an outline change can leave the workflow
PAUSED_STALE_VERSION; a narrow reconciliation path closes this pause only after
the matching Phase 1 batch is approved or rejected. It does not enable resume or
any direct promotion of content from a pause.

Cost: cancelled workflows are retained as history and cannot resume. Existing
candidate batches need an explicit author decision before ownership is freed.

## Deliberately deferred findings

P3 findings remain deferred: FTS virtual/shadow-table prefix filtering, exact
low-level runtime type validation, provider client lifetime management, and
import grouping. Literal whitespace-insensitive goal/ending-hook coverage remains
the accepted deterministic rule; semantic paraphrases may require regeneration.
These limitations do not claim Phase 3 functionality.
