# Architecture and design

What this product is, what was considered, what was built, and why. Numbers marked
"measured during review" were taken on a development laptop at the time each step was reviewed and
were not re-run for this document.

## Use cases

Four data products on top of the VL signal (a dense point cloud of the vehicle's position at 16 fps)
were on the table:

1. **Scenario mining.** Derive ego motion from the VL signal, detect events (hard brake, sharp turn,
   stop) and let a data scientist search drives and clips by scenario. This needs the whole platform
   — stream ingestion, per-drive processing, a generic event table, a query layer and a UI — and its
   output, an event index over drives, is exactly what a clips catalog consumes. **Chosen.**
2. **Signal-quality monitoring.** Detect dropped frames, position jumps and physically impossible
   motion. Data quality belongs to the platform, not to any one model. **Built together with 1**,
   because mining events from bad frames produces bad events, and because it gives us the error-code
   catalog.
3. **Coverage analysis.** Map where drives have been to find gaps and guide collection. Its
   infrastructure value is spatial indexing and partitioning. Not built: it needs real-world map
   coordinates, and the simulated drives live in a local frame.
4. **Storage efficiency.** Consecutive frames differ little, so storing deltas rather than full
   clouds could save a great deal (see [Scale](#scale-notes)) — the largest cost line at fleet scale.
   Not built: it needs real clouds to be meaningful, and the arithmetic below shows the saving hinges
   on ego-motion compensation. Listed under "coming soon".

1 and 2 win because together they exercise every layer the role asks about — ingestion, processing,
warehouse, BI, observability — on a small enough codebase to explain end to end, and the validator
makes the data trustworthy and testable. 3 and 4 each need real data or a research question first.

## Architecture

```
                 docker-compose                      local Python / PySpark (venv)
            ┌───────────────────────┐   ┌──────────────────────────────────────────────────────┐
 simulator  │  Kafka (KRaft, 1 node)│   │ ingest job (Structured Streaming, trigger availableNow)│
 + faults ──┼─> topic vl.frames     ├──>│  strict parse, centroid per frame                     │
 (producer) │  3 partitions, key =  │   │      ├── valid  ──> raw_frames   (Parquet, by drive_id)│
            │  drive_id             │   │      └── invalid ──> rejects     (Parquet, VL-005)    │
            └───────────────────────┘   └───────────────────────────┬──────────────────────────┘
                                                                    │ checkpoint = Kafka offsets
                                        ┌───────────────────────────v──────────────────────────┐
                                        │ transform job (batch, groupBy drive_id, applyInPandas)│
                                        │  error codes, EM features, events, drive summary      │
                                        │      └──> frames, events, drives (Parquet)            │
                                        └───────────────────────────┬──────────────────────────┘
                                                                    │
                          DuckDB views (SQL) <──────────────────────┘
                              ├── Dash dashboard (scenario explorer, data quality)
                              └── ad-hoc SQL, EXAMPLE_QUERIES

 vl.validation (the single rule module) is used by the transform and by python -m vl.validate,
 which also reads the simulator's ground-truth labels.
```

Kafka runs in `docker-compose`; the Python components run from a virtual environment behind `make`
targets. Containerizing them is listed under "coming soon".

### Components

| Component | Module | Responsibility |
|---|---|---|
| Simulator | `vl.mock` (`drive_sim`, `profiles`, `vehicles`, `faults`) | Seeded, deterministic drives with sensor noise, timestamp jitter, five driving profiles, three vehicle kinds and injected faults; ground-truth labels for every fault |
| Producer | `vl.mock.producer`, `vl.kafka_io` | Publishes frame messages to `vl.frames`, key = `drive_id`; checks the broker and delivery, and fails with one clear line |
| Ingest | `vl.etl.ingest` | Spark Structured Streaming from Kafka (or files) under the strict message contract; centroid per frame; valid frames to `raw_frames`, the rest to `rejects` with VL-005 |
| Transform | `vl.etl.transform`, `vl.etl.features` | Per drive: error codes, velocity, speed, acceleration, yaw rate, events, drive summary; full recompute, overwrites `frames`, `events`, `drives` |
| Validator | `vl.validation`, `vl.validate` | Every error-code rule in one place; a report CLI for any source (JSONL or Parquet), with label and realism checks |
| Warehouse | `vl.warehouse` | In-process DuckDB, one view per table, `EXAMPLE_QUERIES`; safe to call from several threads |
| Dashboard | `vl.dashboard` | Dash app: scenario explorer and data-quality tabs; reads only through `Warehouse.query` |
| Config | `vl.config` | Frozen `Config` with every threshold, `VL_<FIELD>` overrides, validation |

### Why two Spark jobs

Velocity and acceleration need the previous frame, which may sit in a different micro-batch. So the
streaming job does only what is local to one message — parse and centroid — and a batch job groups a
drive's rows and runs the sequential logic. The cost is a second job and a full recompute per run;
the payoff is a transform that is one pure function per drive: easy to test against hand-computed
fixtures, and identical in the Spark and non-Spark paths.

### Position definition

A frame's vehicle position is the centroid of its point cloud, in meters — the mean of the
integer-centimeter coordinates divided by 100. This is the project's choice, not a property of the VL
signal. The pipeline stores the centroid and the point count, not the cloud.

### Assumptions

The assignment leaves two points open. These are the decisions I made.

- **Clips are out of scope.** The product diagram shows Data Collection producing clips alongside the
  VL signal, with the clips going straight to the catalog. The assignment never defines clips, and
  its exercise asks for the VL path only (data source, ETL, warehouse, BI). I assume clips are
  recordings of the same drive and don't build them. The join keys a clips catalog would need are
  already here: `drive_id` plus a time range (`events.start_ts`, `events.end_ts`), so "clips of every
  hard brake" is one query once clips are indexed by drive and time.
- **Motion is planar (x, y).** The VL signal is a 3D location, and the centroid's `z` is kept in
  `raw_frames`, but speed, acceleration, heading and events use x and y only — every event in the use
  case (hard brake, hard accel, sharp turn, stop) is a horizontal-plane event. On a 10% slope, 3D
  speed differs from planar speed by about 0.5%. The simulated roads are flat (`z` never changes), so
  vertical kinematics here would only measure noise. Vertical signals (road grade, speed bumps) are
  under "coming soon".

## Message contract

One Kafka message is one frame: UTF-8 JSON with compact separators, key = `drive_id`.

| Field | Type | Notes |
|---|---|---|
| `drive_id` | string | required |
| `frame_idx` | integer | required; 64-bit |
| `ts` | number | required; seconds, finite |
| `points` | array of `[x, y, z]` | required; integer centimeters; may be empty |
| `vehicle_id`, `vehicle_kind`, `drive_start_utc` | string | optional metadata, carried in every message |

Extra fields are ignored. Anything else — wrong type, a float where an integer is required, a blank
line, trailing text, single quotes, `NaN` — is rejected as VL-005. The strictness is enforced with
Spark SQL expressions only, no Python UDF, because Spark's `from_json` on its own is lenient; the
`vl.etl.ingest` docstring has the details.

### Message v2: the claim check

At the assignment's resolution — about 1M points per frame — a v1 message is roughly 15 MB of JSON:
over Kafka's default 1 MB limit, and about 1.5 s to decode (`make bench`), against 16 frames per
second per vehicle. Message v2 keeps Kafka small and moves the clouds into storage.

| Field | Type | Notes |
|---|---|---|
| `schema_version` | integer `2` | required for v2; absent, null or `1` means v1; anything else is VL-005 |
| `centroid_m` | `[x, y, z]` numbers | required (finite) when `n_points > 0`; absent or null for an empty cloud |
| `n_points` | integer | required; 0 to 2,147,483,647 |
| `cloud_ref` | string | required; `<cloud file>#<position in the drive>` — an object-store key in production |

The other fields are as in v1, and a v2 message's `points`, if present, are ignored. The producer
(`--cloud-dir`) writes each drive's clouds to one file with the rigid-body codec below, then sends
the v2 messages. Both strict parsers — the Spark ingest and the validator — accept v1 and v2 and
reject the same malformed v2 lines (`tests/test_message_v2.py`). `make demo-v2` produces exactly the
same warehouse and validator report as `make demo`, with 1.2 MB of messages in total instead of
27.3 MB.

**Known limitation: v2 batches per drive, not per frame.** `to_v2` writes a whole drive's cloud file
and builds every v2 message before the producer sends the first one (`src/vl/mock/producer.py`,
`_as_v2`), because the demo simulates a whole drive at once. A vehicle streaming in real time can't
wait for the drive to finish before publishing frame 1. The size and decode wins above are real, but
the streaming story attached to them is not: a production version needs per-frame or small-batch
cloud writes — append to a rolling object, or flush every N frames — rather than one file per drive.
`load_cloud` also decodes the entire drive's codec file to return a single frame's cloud, with no
random-access index: fine for replaying a whole drive, wasteful for fetching one frame.

### Rigid-body cloud codec

The cloud is the vehicle's own body, so every frame is the same shape moved somewhere else.
`vl.cloud_codec` stores that shape once per drive and, per frame, only the integer shift and the
residual `cloud - shape - shift`, which sensor noise keeps within a few centimeters (`int8`). A frame
that doesn't fit — a different point count, or a residual beyond `int8`, meaning a deformed or corrupt
cloud — is stored raw. The codec is lossless (exact `int64` round trip, tested) and 28× smaller than
the JSON clouds of a simulated 1,000-frame drive (2.98 MB → 108 KB). `max_residuals` reports how far
each frame departs from the shape — a ready signal for a future "deformed cloud" quality rule.

## Data model

The schema is generic by design: nothing in it names a scenario. A new scenario type means new rows
in `events`, not a schema change. Definitions are in `src/vl/schemas.py`.

| Table | Columns | Layout |
|---|---|---|
| `raw_frames` | `drive_id`, `vehicle_id`, `vehicle_kind`, `drive_start_utc`, `frame_idx`, `ts`, `n_points`, `x`, `y`, `z` | Parquet, partitioned by `drive_id`; appended by the ingest |
| `rejects` | `payload`, `error_code`, `ingested_at` | Parquet, flat; malformed messages have no usable `drive_id` |
| `frames` | `drive_id`, `vehicle_id`, `vehicle_kind`, `drive_start_utc`, `frame_idx`, `ts`, `x`, `y`, `z`, `vx`, `vy`, `speed`, `accel`, `yaw_rate`, `error_code` | Parquet, partitioned by `drive_id`; `error_code` null when clean |
| `events` | `drive_id`, `event_type`, `start_frame`, `end_frame`, `start_ts`, `end_ts`, `params` | Parquet, flat; `params` is a JSON string with the measured value |
| `drives` | `drive_id`, `vehicle_id`, `vehicle_kind`, `drive_start_utc`, `start_ts`, `duration_s`, `n_frames`, `distance_m`, `n_events`, `n_errors` | Parquet, flat |

`drive_id` is a string everywhere (partition type inference is off), so an id like `007` keeps its
zeros. The named queries in `EXAMPLE_QUERIES` cover hard brakes per drive, events by type, error
counts, a drives overview and the fastest drives.

### Querying it as a data scientist

I ran ten questions a data scientist new to the warehouse would ask, in plain SQL on the demo data.
Seven are direct. "Hard brakes on trucks" joins `events` to `drives`; "STOPs longer than 5 s" reads
`params` with `json_extract`; "events per 100 km" works best from `drives`
(`sum(n_events)` over `sum(distance_m)`), since counting through a join double-counts distances. Three
are awkward:

- **The absolute time of an event.** `start_ts` and `end_ts` are seconds from the drive start, and
  `drive_start_utc` is a string, so joining an event to a clip catalog needs a cast and interval
  arithmetic — and DuckDB shows the result in the session's time zone unless asked for UTC. Every
  consumer redoes this, and that time is the key linking an event to its clips.
- **Where it happened.** `events` carries no position; the location is a join to `frames` on
  `start_frame`.
- **One stop, reported twice.** On drive-002, two STOP events at the same spot sit 0.2 s apart,
  because a flagged frame splits the run (a known issue). A consumer counting stops sees two.

The fix is written down: add `start_utc`, `end_utc` and the start position to `events`, computed once
in the transform.

## Schema evolution

**Messages.** `schema_version` names the contract: absent or `1` is v1, `2` is v2.

| Change | Compatible? | How |
|---|---|---|
| Add an optional field | Yes, both ways | Consumers ignore unknown fields (both strict parsers do) |
| Add a required field, remove or rename a field, change a type or unit | No | New `schema_version`; consumers accept both versions while old messages can still arrive |
| An unknown `schema_version` | Rejected, not guessed | VL-005; `rejects` keeps the whole payload, so those messages can be replayed once a consumer understands them |

Deploy order for a new version: consumers first (the Spark ingest and the validator), then the
producers. Enforced in `_version_ok` and `_is_v2` (`vl.etl.ingest`), `_version` (`vl.validate`), and
`tests/test_message_v2.py`, which runs v1, v2 and the malformed v2 cases through both parsers. With
many producer teams, a schema registry would hold the contract in place of two parsers.

**Tables.** The schemas are the DDLs in `vl.schemas`, and every reader passes them explicitly:

- Adding a column: a Spark reader sees it as null in older files. The DuckDB views read each file's
  own schema, so mixed files need `union_by_name` (not set today) or a table rewrite.
- `raw_frames` is append-only history, so its schema should only gain nullable columns.
- `frames`, `events` and `drives` are derived: change the code and the DDL together, then transform.
  The incremental run notices the code change — a fingerprint of the transform code is part of its
  state — and recomputes every drive.
- The production answer is a table format (Delta or Iceberg) with schema evolution and a catalog,
  already the upgrade path in the cloud mapping.

## Technology choices and alternatives

| Choice | Why | Alternatives considered |
|---|---|---|
| Kafka | VL is a stream; a replayable log decouples the producer from the ETL and gives the ingest durable offsets | Kinesis, Pulsar, RabbitMQ (no replay), plain files (not a stream; used only as a test source) |
| PySpark (Structured Streaming + batch) | One API for the stream and the batch job; scales out per drive; the team knows it | pandas alone (doesn't scale out), Flink (more to operate), Beam |
| pandas inside Spark (`applyInPandas`) | The per-drive logic is sequential (previous frame, smoothing windows); a pure function is simple to test against fixtures and is the same code the validator runs | Spark SQL window functions (harder to read and test for this logic) |
| Parquet | Columnar, compressed, self-describing, readable by every engine | CSV (no schema, larger), Delta or Iceberg (ACID and schema evolution, more setup — the upgrade path) |
| DuckDB | Plain SQL over the Parquet files, in process, no server | Spark SQL (heavy to start), Trino, Athena, BigQuery |
| Dash and Plotly | Pure Python, one language across the project | Streamlit, Grafana, Superset |
| docker-compose | One-command, reproducible broker | Kubernetes (overkill here) |
| A separate validator module | The same rule functions serve the ETL, the report CLI and the tests, so the rules can't drift apart | Rules inside the Spark job only |
| Message v2: claim check (centroid + `cloud_ref`, cloud in a file) | At the PDF's resolution the cloud is the bottleneck (see Scale notes); keeping Kafka small and the cloud in storage is the standard pattern for an oversized payload | Keep v1 and compress the JSON in flight (saves bytes, not the per-message parse); a schema registry with a binary wire format (more setup, same idea) |
| Rigid-body cloud codec (shape once per drive, shift and residual per frame) | Exploits what a vehicle's point cloud actually is instead of compressing opaque bytes; lossless and simple | Generic compression (gzip/zstd on the raw cloud: much smaller gain, see Scale notes); a lossy codec (rejected — no accuracy budget was given) |

## Reliability

- **Ingest checkpoints.** The streaming query runs with `trigger(availableNow)` and a checkpoint
  directory (`data/warehouse/_checkpoints/ingest`) holding the consumed Kafka offsets. A second run
  reads only new messages (checked during review: 100 added messages were picked up once).
- **A replayed micro-batch lands once.** Kafka delivers at least once, and Spark replays a micro-batch
  (same id, same rows) that failed before its checkpoint commit. `write_batch` writes each batch to
  `_staging/<tag>/` first, then moves the files into `raw_frames` and `rejects` under names that carry
  the batch. A replay removes its earlier attempt's files before moving the new ones in, so the tables
  hold one copy of every batch whatever point the crash happened at (tested). The tag includes a run
  id kept in the checkpoint directory, because batch ids restart at 0 with a new checkpoint. Sends the
  producer itself retries are a different case — two messages, flagged VL-004 (below).
- **Deterministic duplicate resolution.** The transform sorts each drive's rows by a fixed key
  (`frame_idx`, `ts`, `n_points`, `x`, `y`, `z`, metadata) before deriving features, so conflicting
  copies of one `frame_idx` resolve the same way on every run; every copy is flagged VL-004 and one
  row is kept.
- **Idempotent overwrite.** A full run recomputes everything from `raw_frames` and overwrites
  `frames`, `events` and `drives` with static partition overwrite, so reruns give the same result and
  drives that left the input lose their partition. The three writes aren't atomic (see known issues);
  a rerun repairs a partial failure.
- **The incremental swap shrinks, but doesn't remove, a missing-table window.** For `events`/`drives`,
  `_merge_flat` renames the live directory out of the way and the new one into place — two back-to-back
  atomic renames — instead of deleting the live files first. That cuts the window where the table
  doesn't exist from "however long the delete takes" to the gap between two renames. A reader landing
  in that gap still sees no table; closing it entirely needs a stable pointer (a symlink) or a real
  table format (Delta or Iceberg, on the roadmap). A crash in the gap is self-healing: the next run's
  `plan()` treats a missing output table as a reason to fall back to a full recompute.
- **Fail fast at the edges.** The producer and the ingest confirm the broker answers and the topic
  exists before starting Spark, and exit with one `error:` line otherwise.

### Failure modes

Every way I found for the pipeline to lose, duplicate or corrupt data, what it does, and what would
show it. Rows marked *checked* were run on purpose; the rest are read from the code.

| Failure | What happens | What detects or prevents it |
|---|---|---|
| Broker down at start | Producer and ingest exit with one `error:` line before any work | The exit code |
| Broker restarts while producing | `acks="all"` and 5 retries; the client isn't idempotent, so a retried send can be written twice | VL-004 on the copy |
| Ingest crashes after `raw_frames` is written, before the checkpoint commits | Spark replays the micro-batch. Before the fix its rows were appended again — every copy flagged VL-004 and left out of the kinematics. *Checked:* replaying a whole golden drive hid its HARD_BRAKE. Now the replay replaces the batch's own files | Prevented: idempotent batch writes (`test_a_replayed_batch_replaces_its_own_rows_instead_of_adding_them`) |
| A Spark write crashes midway (any table) | Spark leaves files under `<table>/_temporary/`. Spark and the validator (PyArrow) skip them. Before the fix the DuckDB views read them too (*checked:* an extra drive appeared) | Prevented: the views read only committed files (`test_files_left_by_a_crashed_spark_write_are_not_read`) |
| Transform crashes between its three writes | `frames` can be new while `events` and `drives` are old | A rerun repairs it (full recompute from durable `raw_frames`) |
| Crash between the two renames of the incremental swap | `events` or `drives` is missing on disk | The next run sees the missing table and does a full recompute |
| Broker volume reset, checkpoint kept | The ingest resumes from offsets the new topic lacks and adds 0 rows, exit code 0 | Prevented: `make kafka-reset` deletes the checkpoint too |
| Pipeline code changes, thresholds and raw data don't | Before the fix the incremental transform skipped every drive and kept the old logic's output. Now its state holds a fingerprint of the transform code, and a change forces a full run | Prevented: `test_a_change_to_the_pipeline_code_forces_a_full_run` |

The three "before the fix" rows were found by this review and fixed test-first.

## Data quality

- **One validator.** Every error-code rule is a small function in `vl.validation`; the transform
  composes them and `python -m vl.validate` reports on them. Precedence and details are in
  [error-codes.md](error-codes.md).
- **Ground truth.** The simulator writes `labels.json` — every injected fault, with the frame and the
  code the pipeline should assign. The validator checks that every labeled fault was detected with the
  right code (recall) and that the share of flagged unlabeled frames stays under
  `max_false_positive_rate` (1%). On the demo fleet: 4 of 4 faults detected, 0.00% false positives.
- **Suspect frames** (VL-002, VL-004, VL-006) are excluded from speed and acceleration and never serve
  as the baseline of a later frame, so one bad frame can't manufacture false events around it.
- **Realism.** The mock data aims to be as close to a real signal as practical: Gaussian position
  noise of 0.2 cm per point, timestamp jitter of 1 ms (positions sampled at the jittered time), five
  driving profiles (scripted, highway, city, curvy, parking) with physical limits, three vehicle kinds
  of different sizes (car, suv, truck), and random faults at per-frame rates (drop bursts of 1–5 frames
  0.2%, duplicates 0.1%, sparse clouds 0.1%, position jumps 0.02%, malformed messages 0.05%, late
  arrival 0.1%). The point cloud is generated at a reduced resolution (a 50 cm lattice, 200 points for
  a car), configurable, and the pipeline doesn't depend on it. `--realism` checks each derived drive
  against its profile limits: maximum speed, and the 99th percentile of acceleration and yaw rate.

## Scale notes

- **Raw rate.** One frame of 1M points × 3 coordinates × 2 bytes is 6 MB; at 16 fps that is about
  96 MB/s per vehicle, roughly 346 GB per hour (arithmetic from the task description).
- **Fleet sizing.** 1,000 vehicles driving at once at 16 fps, from the measured per-unit numbers
  (`make bench` and the demo):

  | Flow | Per unit | Fleet rate | Per driving hour |
  |---|---|---|---|
  | Kafka messages (v2) | about 300 bytes | 16,000 msg/s, 4.8 MB/s | 17 GB |
  | `frames` rows | about 79 bytes | 1.3 MB/s | 4.6 GB |
  | Raw clouds, 1M points, 2 bytes per coordinate | 6 MB per frame | 96 GB/s | 346 TB |
  | Clouds after the codec (0.54 bytes per point, measured) | about 0.54 MB per frame | 8.6 GB/s | 31 TB |

  Kafka and the warehouse are small at this scale: 4.8 MB/s is one broker's load. The ingest ran at
  about 2,850 msg/s per local task, so about 6 parallel tasks would keep up, and 24 partitions leave
  room to grow; one transform task derives about 395,000 rows/s against 16,000 arriving. **The clouds
  are the bottleneck** — even compressed, 8.6 GB/s of uplink and 31 TB per hour of storage. The codec's
  ratio was measured on simulated 0.2 cm noise, and real sensor noise would lower it. So the design
  decision is not a faster pipeline: compute the centroid on the vehicle, send message v2 always, and
  upload clouds only where they're needed — around events, or for a sample of drives.
- **Encoding, measured (`make bench`).** Plain frame-to-frame deltas don't pay off (a vehicle at
  30 m/s moves about 1.9 m per frame). Subtracting the shape and the frame's shift — the rigid-body
  codec — does: 28× on a simulated drive, lossless. As binary rather than JSON, a 1M-point frame is
  6 MB instead of 14.8 MB and decodes with its centroid in 21 ms instead of 1,465 ms (71×).
- **Message size.** v1 puts the cloud in the Kafka message; v2 (the claim check) puts it in a file and
  sends about 300 bytes. This is the single change that makes the PDF's resolution feasible at all; the
  rest is tuning.
- **What the warehouse stores.** The pipeline keeps the centroid and the point count, not the cloud: on
  the demo (4000 frames) `frames.jsonl` is 27.3 MB (about 6.8 KB per message) while the `raw_frames`
  table is about 0.14 MB on disk. With message v2 the clouds are kept, losslessly, in one codec file
  per drive (684 KB for the 4 demo drives).
- **Incremental transform.** `make transform` recomputes only drives whose `raw_frames` files changed
  since the last run (`vl.etl.incremental`), and drops drives that left `raw_frames`. The result equals
  a full run (tested). A recorded drive is long — an hour is 57,600 frames — so the fixed per-drive cost
  of `derive_frames` (about 54 ms of many small pandas operations) is paid once per drive: 395,000
  rows/s on an hour-long drive against 33,000 on a 1,000-frame drive.
- **Tuning in place.** The producer compresses (gzip) and batches (`linger_ms` 20, 256 KB batches);
  the ingest reads Kafka in bounded micro-batches (`maxOffsetsPerTrigger` 100,000); Parquet files are
  zstd; the dashboard reads only the columns it draws; the validator checks a cloud's types in one
  C-speed pass (2.4× faster at 1M points).
- **Parallelism.** The transform groups by `drive_id`, so drives run in parallel and one drive's work
  is one pandas function. A drive is the unit of parallelism and must fit in one task's memory (a
  16,000-frame drive took 2.8 s, measured during review).
- **Partitioning.** `raw_frames` and `frames` are partitioned by `drive_id`, which suits per-drive
  queries here. At fleet scale that would create very many small files; partition by date (and bucket
  by vehicle) instead, keeping `drive_id` as a clustering column.
- **Measured numbers (review, local mode, one laptop).** Ingest about 2.7–3k msg/s; transform overhead
  about 0.1 s per drive (200 tiny drives took about 30 s); `make demo` about 75 s end to end; Parquet
  `frames` about 79 bytes per row against about 202 for CSV.
- **Local limits.** One broker, one machine, `local[2]` Spark. The design scales out by adding
  partitions and executors; that was not exercised.

## Operations: metrics and alerts

What would tell an on-call engineer the pipeline is unhealthy. The rule: **page when data is late or
lost; open a ticket when its quality drifts.**

| Signal | Alert | Where it comes from |
|---|---|---|
| Consumer lag (messages behind the topic end) | Page above 5 minutes of data and growing (about 4.8M messages at the fleet rate) | Spark's Kafka metric `maxOffsetsBehindLatest`; not exported today |
| Freshness (newest frame in `raw_frames`) | Page when older than 15 minutes while vehicles drive | Not measurable today: `raw_frames` has no ingest time (`rejects` has `ingested_at`) |
| Transform failures | Page after 2 failed runs in a row | The exit code; the `_state/transform.json` time is the last success |
| Reject rate (VL-005 per message) | Ticket above 0.5%, page above 5% (a broken producer release) | Printed by the ingest after each run; the simulator injects 0.05% |
| Flag rate (frames with any code) | Ticket above `max_flag_rate` (5%) | The validator report |
| VL-004 rate | Ticket on a spike: replays are idempotent now, so every VL-004 is a real duplicate from the producer or the sensor | `error_code_counts` |
| Frames lost (VL-001) | Ticket on a spike, per vehicle | The validator's `frames_lost` |
| Events per 1,000 km, per type | Ticket on a large shift: catches a threshold or code change that silently moves the outputs | A query on `drives` and `events` |
| Transform duration | Ticket above twice its usual time | Not recorded today |

Today these live as the validator report and the one-line summaries the ingest and transform print;
nothing is exported and nothing alerts. The cheapest next steps: log Spark's `query.lastProgress`
(rows, rows/s, offsets behind) after each ingest run, and add an `ingested_at` column to `raw_frames`
so freshness becomes queryable.

## Cloud mapping

Documented only — nothing is deployed.

| Local | Cloud |
|---|---|
| Kafka in docker-compose | Amazon MSK or Kinesis (Confluent Cloud, Pub/Sub) |
| PySpark jobs | EMR, Databricks or Glue; the ingest as a long-running streaming job, the transform as a scheduled job |
| Parquet on local disk | S3 (or GCS) with a Delta or Iceberg table format and a catalog |
| DuckDB | Athena, Trino or BigQuery over the same files |
| Dash app | A container on ECS, Cloud Run or Kubernetes behind SSO |
| `make` and `.venv` | Container images built in CI, orchestrated by Airflow or Step Functions |
| Config environment variables | A parameter store or a config service |

## Security and privacy

The data is simulated, so this repository holds no personal data and implements no protection: no
authentication, no encryption, a plaintext single-broker Kafka. The one control that exists is the
dashboard listening on `127.0.0.1` by default.

With real VL data, location traces identify people. A drive's first and last stops are usually a home
and a workplace — and this pipeline marks exactly those as STOP events. What a deployment needs, by
priority:

| Control | Why | How it fits this design |
|---|---|---|
| TLS and authentication on Kafka, encryption at rest | Traces in transit and on disk | Broker and bucket settings; no code change |
| Pseudonymous `vehicle_id` | Analysts rarely need the real vehicle | A keyed hash at the producer; the key and mapping in a separate, restricted store |
| Access tiers | Most questions need events, not trajectories | Broad access to `events` and `drives`; restricted access to `raw_frames`, `frames` and the clouds |
| Trip-end trimming | Removes the home and work locations | Drop or coarsen the first and last minutes of each drive in the tables analysts read |
| Retention by table | Raw clouds are the most sensitive and the largest | Short retention on clouds and `raw_frames`, longer on derived tables |
| Erasure on request | A right-to-delete request names a vehicle | `drives` maps a vehicle to its drives, and every table is keyed or partitioned by `drive_id`, so deletion is per drive |
| Audit log and SSO on the dashboard | Who looked at which trace | The managed service's audit log; SSO in front of the app |

The minimum before any real data arrives: TLS and authentication, pseudonymous vehicle ids, and the
access tiers. The rest can follow.

## Testing strategy

- **Test-first, by phase.** For each layer: mock data first, then failing unit tests from that data
  (with interface stubs), then the implementation until green, then a sanity test, then an independent
  review of the tests and the code.
- **Golden fixtures.** Hand-computed expected outputs — inputs and the exact frames, events and drive
  summary — for twelve cases live under `tests/fixtures/golden/`, and message fixtures (10 valid, 22
  malformed) pin the message contract. They are an oracle independent of the code under test.
- **Sanity tests** (`make sanity`, marker `sanity`): a fast end-to-end run on the mock data through
  every layer, from simulator to warehouse and validator, compared against a pure-pandas route; with a
  real broker, `make sanity-kafka`.
- **Hardening tests** pin every defect the reviews found — the strict-parse bypass with a stray `]`,
  concurrent dashboard callbacks on one DuckDB connection, quadratic event shading in the speed chart,
  and the rest.
- **Static checks.** `make check` runs `ruff` and `mypy --strict`; the house rules are full type hints,
  Google-style docstrings and small functions.
- **A lean suite that earns its size.** The suite started at 638 tests and was pruned to 198, keeping
  every test that guards a reviewed defect, with line coverage unchanged at 94% (measured during review
  with a throwaway coverage install, not a project dependency). Later phases — scaling, hardening and
  the reviewer-round fixes — brought it to the current 282, each new test guarding a specific defect or
  behaviour. A mutation spot-check of 48 hand-made mutants against the pruned suite killed 45; the three
  survivors — one equivalent mutant, one caught only by the full sanity test, one error-code literal
  outside the literal-scan test's scope — also survived the unpruned suite.
