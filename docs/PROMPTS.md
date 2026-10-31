# Prompt log

How the LLM was used: I owned the requirements, the design decisions, the scope and the review. The
model proposed options and wrote the code from a design spec and a task plan I approved before any code
existed, and an independent reviewer checked every task.

## Models

Claude Code (desktop app). Main session: Claude Sonnet 5.5 (`claude-sonnet-5-5`) for design and build,
Claude Opus 5.5 (`claude-opus-5-5`) for the review and scaling rounds, and Claude Sonnet 5 / Opus 5 for
the final reviewer-perspective rounds. Implementer, test-author and per-task reviewer subagents: Claude
Sonnet 5.5. The final whole-branch review ran on Claude Opus 5.5 — a different model from the one that
wrote the code.

## Key prompts

A selection of the prompts that shaped the project, lightly edited for spelling and clarity; the meaning
is unchanged, and consecutive prompts on one topic are combined. Short go-ahead prompts and the subagent
briefs (extracts of the task plan) are left out.

### Scope and design

1. **Decide for the role.**
   "Let's choose what is best from the perspective of an Experienced Software Engineer – Data
   Infrastructure, the role I'm applying for. Focus every decision on that. Give me your view first,
   then I'll decide."

2. **Simplicity check against the requirements.**
   "Are we overthinking this? Should the plan be simpler? Does the simpler version still support every
   requirement in the assignment? We must meet all of them while keeping the solution simple but
   professional."
   *Result: a smaller design the whole team (one person) can explain end to end.*

3. **Engineering standards.**
   "Document everything, keep functions small, use type hints. Don't assume anything: ask me about every
   ambiguity."
   *Result: enforced by tooling (`ruff`, `mypy --strict` in `make check`) and Google-style docstrings.*

4. **Test-first on realistic data.**
   "Create mock data that simulates the real flow as realistically as we can, with a validation step on
   it. Write unit tests from that data, implement until they all pass, then add an end-to-end sanity test
   that must pass too before we move to the next phase."
   *Result: a seeded, fault-injecting drive simulator, golden fixtures, and a validator that checks the
   pipeline's output against the simulator's ground-truth labels.*

5. **Tests that earn their place.**
   "About 630 tests is far more than we need. Reduce the suite significantly, to only the tests that
   matter."
   *Result: 638 to 198 tests with line coverage unchanged at 94%; every test that guards a defect found
   in review was kept.*

### Challenging the model's answers

6. **Check claims against the source.**
   "Can we be sure that's what the assignment says? If we shouldn't build anything for clips, why are
   they in the product diagram?"
   *Result: the PDF doesn't say clips are out of scope; leaving them out is my own decision, now stated
   as one in the README.*

7. **A targeted correctness review.**
   "Do a quick, strict review of all our work so far, in under 10 bullets. Look for ms-vs-s mix-ups,
   off-by-one frames, wrong denominators, double-counted events, silent row drops and similar. Give
   file:line and severity. Don't fix anything."

8. **Impact before change.**
   "Rethink these findings. Can we fix the four quickly without affecting other parts of the system?
   What are the side effects?"
   *Result: the model withdrew part of its claims.*

9. **Precise scope.**
   "Fix only items 1 and 3 in the validator report (duplicates collapsed, frames lost) and only document
   items 2 and 4 in `error-codes.md` and `release-notes.md`. Run `make check` and the validator tests,
   commit, don't push."

10. **Gap analysis.**
    "Re-read the assignment and give me your honest opinion: what are we missing, and what could we do
    better? And we don't use `z` at all — what do you think about that?"
    *Result: a scripted demo drive that shows every event type, and the 3D and clips assumptions written
    down rather than implied.*

### Scale

11. **Performance of a hot path.**
    "Does the validation step process all the JSON records in the fastest way possible? This is exactly
    the kind of knowledge the reviewer wants to see."
    *Result: a measurement instead of an opinion. On the demo data the JSONL route took about 5.2 s,
    almost all of it in the per-line parse, against 1.2 s for Parquet — which led to the scaling work
    below (the cloud check itself later became 2.4× faster).*

12. **Investigate before optimizing.**
    "The system must handle large volumes of VL and raw data, where every optimization matters. Go
    through each component of the pipeline and work out how to make it efficient. Investigate first and
    write down your conclusions; then we'll decide what to change."

13. **Challenge your own plan.**
    "Rethink the problems and suggestions you just gave. Can you find better solutions?"
    *Result: profiling showed the proposed cross-drive transform rewrite wouldn't pay off, so it was
    dropped. Built instead, in this order: a benchmark (`make bench`), message v2 (centroid plus a
    reference to the cloud) with a lossless cloud codec 28× smaller than JSON, an incremental transform,
    then producer batching and compression, bounded Kafka micro-batches and zstd Parquet.*

14. **Proof, not a report.**
    "Did you test the whole pipeline with the new implementation, including screenshots of the dashboard
    results?"
    *Result: running `make demo-v2` exposed a disagreement between the Spark ingest and the validator
    (3 of 4 labelled faults found); fixed test-first.*

### The reviewer's view

15. **Assess as the hiring manager.**
    "Re-read the assignment and assess our implementation as the reviewer hiring an Experienced Software
    Engineer – Data Infrastructure would: the right approach at each step, the right technologies, and
    the most efficient handling of this data volume and format. Then tell me honestly whether we
    delivered what they expect and, if not, what to change."
    *Result: two issues no earlier review had found. The incremental transform briefly left the events
    and drives tables missing during a swap (fixed); message v2 still batches clouds per drive
    (documented as a known gap). Also a "Scope and judgment calls" section in the README.*

### Production readiness

16. **Find the blind spots.**
    "Look at what we've done so far and tell me honestly what we're missing. Where would a Data
    Infrastructure reviewer still find gaps, and what would you check next to close them?"
    *Result: the gaps were reliability, data contracts, storage layout, operations and privacy, each
    with a concrete check to run. I ran the cheapest ones (17 to 21), fixed what they found (22), then
    ran three more (23 to 25).*

17. **Failure-mode review.**
    "List every way this pipeline can lose, duplicate or corrupt data (broker restart, Spark crash
    mid-batch, replayed offsets, partial Parquet write). For each, say what happens today and what would
    detect it."
    *Result: a failure-mode table in the architecture doc and three new findings, two confirmed by
    running them. A replayed micro-batch flagged every replayed frame VL-004 and dropped it from the
    kinematics; replaying a whole drive hid its hard brake — a case the release notes had called
    harmless. The DuckDB views read Spark's uncommitted `_temporary` files after a crashed write. The
    incremental transform didn't notice a code change. All three were fixed (prompt 22).*

18. **Prove idempotency.**
    "Prove that re-running ingest and transform on the same input gives the same tables. Write the test
    that would fail if it didn't."
    *Result: already proven by three existing tests (`test_idempotent`, `test_rerun_is_idempotent`,
    `test_replayed_duplicates_not_duplicated`), so no new test was added. The weak point was the replay
    in prompt 17, not the rerun.*

19. **Schema evolution.**
    "Design how the message schema evolves. A v1 producer and a v2 consumer must coexist, and adding a
    field must not break old data. State the compatibility rules and where they're enforced."
    *Result: written compatibility rules for messages and tables, the deploy order (consumers first),
    and pointers to the code and tests that enforce them.*

20. **Size the fleet.**
    "Estimate throughput, storage per day and the Kafka partition count for 1,000 vehicles at 16 fps with
    1M-point clouds. Show the arithmetic and the bottleneck."
    *Result: Kafka needs 4.8 MB/s and the warehouse 4.6 GB per driving hour, both small. The clouds need
    8.6 GB/s even after the codec. So the real decision is to compute the centroid on the vehicle and
    upload clouds selectively, not to make the pipeline faster.*

21. **Privacy by design.**
    "Location traces identify people. What access control, retention, and anonymization or aggregation
    should this data product have, and what is the minimum to implement now?"
    *Result: a prioritized control list. The sharpest point: a drive's first and last STOP events are
    usually a home and a workplace, so trip-end trimming matters more than encryption alone.*

22. **Fix what the review found, test-first.**
    "Fix the three findings first, before running more prompts."
    *Result: one failing test per finding, then the fix.*
    - *Idempotent ingest:* each micro-batch is written to a staging directory and moved into the tables
      under names that carry the batch. A replay replaces its earlier attempt's files, at whatever point
      the crash happened.
    - *Committed files only:* the DuckDB views read only the files Spark committed, not its `_temporary`
      leftovers.
    - *Code fingerprint:* the incremental state now records a fingerprint of the transform code, so a
      code change forces a full recompute.

23. **Use it as its consumer would.**
    "Act as a data scientist new to this warehouse. Try to answer real questions with SQL only — for
    example the hard brakes of trucks — and report what's missing or awkward in the schema."
    *Result: ten questions run on the demo warehouse, seven direct. Three were awkward: an event's
    absolute time (the key linking it to its clips) has to be rebuilt from a relative offset and a
    string, its location is a join to `frames`, and a stop split by a flagged frame shows up as two. The
    next step is written down — put `start_utc`, `end_utc` and the position on `events`.*

24. **Operate it.**
    "What metrics and alerts would tell an on-call engineer that the pipeline is unhealthy? Which would
    page, and at what thresholds?"
    *Result: an alert table with one rule — page when data is late or lost, open a ticket when its quality
    drifts. It also surfaced two gaps: nothing is exported today, and `raw_frames` has no ingest time, so
    freshness can't be queried.*

25. **Onboard from the README alone.**
    "Read the repository as a new engineer following only the README. Report every step that would fail,
    is unclear, or depends on knowledge that isn't written down."
    *Result: every make target, file path and link in the README resolves. What was missing: the
    supported operating systems, and what to do when setup fails. Both are in the README now — an exact
    Python 3.11, Java 17, the Kafka port, the first-run package download, and a stale checkpoint.*

## Where I corrected the model

- **Clips** (prompt 6): the model cited the PDF for a decision the PDF never makes.
- **Over-stated bugs** (prompt 8): part of a reported "medium" finding was already handled.
- **An unsupported rewrite** (prompt 13): dropped after profiling — about 54 ms of fixed cost per drive
  disappears at a realistic drive length.
- **A predicted crash that was really silent data loss:** a review predicted a stale Kafka checkpoint
  would crash the ingest. Running it showed exit code 0 with 0 rows added. The fix (`kafka-reset` also
  drops the checkpoint) targets what was observed, not what was predicted.
- **Two parsers that disagreed** (prompt 14): Spark accepted a malformed `centroid_m` the validator
  rejected. A failing test came first, then the fix.
- **A known issue that was wrong** (prompt 17): the release notes claimed a replayed batch left `frames`
  correct. Running the replay showed the hard brake disappearing; the note now says so.

## How the model's output was checked

- Tests first for every feature and fix — 282 tests at the last full run.
- `make check` (ruff, `mypy --strict`) clean on every commit.
- Every performance claim measured with `make bench`, not estimated.
- `make demo` end to end through a real Kafka broker, checked by the validator against the simulator's
  ground-truth labels; the incremental transform compared against a full recompute.
- The final review ran on a different model from the one that wrote the code.
