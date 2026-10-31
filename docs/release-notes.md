# Release notes — VL Data Product 0.1.0

The first version. Everything runs locally, and the data is simulated.

## What it does

The VL data product turns a stream of Vehicle Location frames (a 3D point cloud per frame, 16 fps)
into three things:

- **Validated frames.** Every frame is checked by one set of rules; failures carry an error code
  (VL-001 to VL-006, see [error-codes.md](error-codes.md)) and are kept out of the motion estimates.
- **Ego motion and events.** Per drive: position (the cloud centroid), velocity, speed, acceleration
  and yaw rate, plus the detected driving events (HARD_BRAKE, HARD_ACCEL, SHARP_TURN, STOP) with the
  measured value of each.
- **A queryable warehouse and a dashboard.** Parquet tables (`frames`, `events`, `drives`,
  `raw_frames`, `rejects`) queried through DuckDB, and a Dash dashboard with a scenario explorer and a
  data-quality view.

The implemented use case is scenario mining with built-in data-quality checks; the other use cases
considered are in [architecture.md](architecture.md#use-cases). A mock data source — a seeded drive
simulator with sensor noise, timing jitter and injected faults, publishing to Kafka — stands in for
real vehicles, and a standalone validator checks any source, including the simulator's own output
against its ground-truth labels.

Included: a Kafka producer, a Spark streaming ingest, a Spark batch transform, the DuckDB layer, the
Dash dashboard, the validator CLI, docker-compose for the broker, and a `make` workflow.

Built for large raw data: message v2 (a small Kafka message carrying the centroid and a reference to
the drive's cloud file, instead of the cloud itself), a lossless rigid-body cloud codec (28× smaller
than JSON on a simulated drive), an incremental transform (only changed drives are recomputed), and a
benchmark (`make bench`) that measures each of these. Requirements: Docker Desktop, Java 17, Python
3.11 (see the [README](../README.md)).

## Usage

### Run the whole pipeline

```
make setup
make demo        # clean, fresh broker, produce, ingest, transform, validate; about 75 s
make dashboard   # then open http://127.0.0.1:8050
make demo-v2     # the same demo with message v2 (clouds in data/clouds); same warehouse and report
make bench       # wire formats, cloud codec and per-drive transform speed, as a Markdown table
```

`make demo` ends with the validator report on the `raw_frames` table (`make validate-warehouse`;
`make validate` checks the JSONL copy instead and also reports VL-005). Expected: 4000 messages sent,
3999 `raw_frames`, 1 reject, 3998 `frames`, 6 `events`, 4 `drives`, every check `PASS`:

```
PASS error_code_counts: VL-001=2, VL-002=1, VL-004=1
PASS flag_rate: 0.10% flagged (limit 5.00%)
PASS labeled_faults_detected: 4/4 detected
PASS false_positive_rate: 0.00% of 3994 unlabeled frames flagged (limit 1.00%)
PASS realism[drive-001].max_speed: max speed 34.03 m/s (limit 38.0 m/s)
```

### Query the warehouse with SQL

After `make demo` (paths relative to the repo root):

```
PYTHONPATH=src .venv/bin/python -c "
from vl.warehouse import Warehouse
wh = Warehouse('data/warehouse')
print(wh.query('SELECT drive_id, frame_idx, error_code FROM frames WHERE error_code IS NOT NULL ORDER BY drive_id, frame_idx').to_string())
print(wh.query('SELECT event_type, start_frame, end_frame FROM events WHERE drive_id = ? ORDER BY start_frame', ['drive-002']).to_string())
"
```

Output on the demo data:

```
    drive_id  frame_idx error_code
0  drive-002        201     VL-001
1  drive-002        400     VL-002
2  drive-002        600     VL-004
3  drive-002        801     VL-001
   event_type  start_frame  end_frame
0  SHARP_TURN          204        253
1        STOP          703        799
2        STOP          802        911
```

The same engine serves the named queries in `EXAMPLE_QUERIES` (`hard_brakes_per_drive`,
`events_by_type`, `error_counts`, `drives_overview`, `fastest_drives`):

```
PYTHONPATH=src .venv/bin/python -c "
from vl.warehouse import Warehouse, EXAMPLE_QUERIES
print(Warehouse('data/warehouse').query(EXAMPLE_QUERIES['events_by_type']).to_string(index=False))
"
```

```
event_type  n_events
HARD_BRAKE         1
SHARP_TURN         2
      STOP         3
```

There's a view for all five tables; a missing table reads as an empty view with the right columns.

### Validate data

```
make generate    # 5 simulated drives to data/mock (frames.jsonl, labels.json)
.venv/bin/python -m vl.validate --source data/mock/frames.jsonl   # structural checks only
.venv/bin/python -m vl.validate --source data/mock/frames.jsonl --labels data/mock/labels.json --realism
.venv/bin/python -m vl.validate --source data/warehouse/raw_frames   # a Parquet table
```

Run these with `PYTHONPATH=src`, or use `make validate` (the second command) and
`make validate-warehouse` (the third, with labels and `--realism`). The exit code is 0 when every check
passes, 1 when one fails, 2 for unusable input. `--realism` needs `--labels` and is statistical (see
the known issues).

### Tune a threshold

```
VL_MAX_JUMP_M=5 make transform
VL_HARD_BRAKE_MS2=-3.0 make transform
```

The thresholds are the fields of `Config` (see the README); a rerun of `make transform` recomputes
everything from `raw_frames`.

### Dashboard

`make dashboard` serves two tabs. **Scenario Explorer:** pick a drive; read its summary (frames,
duration, distance, events, flagged frames) from the `drives` row; see the trajectory with each event
type in its own colour and symbol, the speed-and-acceleration chart with events as shaded spans, and
the event table; filter events by type across drives. **Data Quality:** error counts by code and by
drive, the number of rejected messages, and what each code means. On an empty warehouse it reads "No
drives in this warehouse. Run make demo first".

Known gap: `dcc.Tabs` renders each tab as a plain `div` with no `role` and no `tabindex`, so the two
tabs can't be reached by keyboard. Every other control can. The fix belongs in Dash.

## Known issues

**Message handling and ingest**

- **Duplicate JSON keys.** The contract is silent about them: Spark keeps the last value of a typed
  field and ignores a later invalid duplicate, so the outcome depends on key order and type.
- **`1e999` in a string field.** A JSON number beyond the double range in `drive_id` or a metadata
  field is read as text instead of being rejected.
- **Each micro-batch scans its tables.** To let a replayed batch replace its earlier files, the ingest
  looks for them by name in `raw_frames` and `rejects` — one directory walk per batch. At fleet scale a
  manifest of published batches (or Delta's transactional appends) would replace the walk.
- **VL-005 rows have no `drive_id`.** They appear in `rejects` only, never in `frames`.
- **Kafka source timeouts.** The Spark Kafka source has no fail-fast timeouts of its own; the broker
  and topic are checked up front (a few seconds), and a broker lost mid-run falls to Spark's defaults.

**Transform and warehouse**

- **Non-atomic writes.** The transform writes `frames`, then `events`, then `drives`. A failure in
  between leaves the tables inconsistent until the next run repairs them (full recompute).
- **The code fingerprint covers six modules.** The incremental transform recomputes everything when
  `features.py`, `transform.py`, `validation.py`, `schemas.py`, `errors.py` or `config.py` changes
  (`CODE_FILES` in `vl.etl.incremental`). A new module that shapes the outputs must be added there, and
  a pandas or NumPy upgrade goes unnoticed.
- **Incremental state can go stale.** The incremental transform trusts `_state/transform.json`: editing
  an output table by hand, or a crash between the writes and the state save, isn't detected. Run
  `python -m vl.etl.transform --warehouse data/warehouse` (without `--incremental`), or delete the state
  file, to force a full run.
- **Cloud files are local paths.** `cloud_ref` points to a file under `data/clouds`; moving the
  directory breaks the references. In production it would be an object-store key.
- **v2 malformed frames.** A frame whose cloud is malformed is still sent (as a v2 message that breaks
  the contract on purpose), so it stays visible as VL-005; its cloud file entry is empty.
- **A missing warehouse path yields empty tables.** `Warehouse('wrong/path')` doesn't fail; every view
  is empty, and the dashboard shows "No data".
- **Spark overhead per drive.** About 0.1 s per drive (measured during review), so very many tiny drives
  are slow while one long drive is cheap (16,000 frames took 2.8 s).
- **First frame time.** With timestamp jitter, the first frame's `ts` can be slightly negative.

**Rules and thresholds**

- **A forward timestamp spike contaminates its neighbours.** A `ts` that jumps forward isn't flagged on
  its own row (the row after it is flagged VL-004), so the spike and its neighbours get wrong velocities.
- **No hysteresis.** A signal hovering around a threshold can split one maneuver into several short
  events, or drop it below `min_event_frames`.
- **A sparse cloud can cause a spurious VL-002.** A VL-006 frame whose centroid is shifted by more than
  `max_jump_m` can make the next frame look like a jump, because the comparison is with the previous row,
  not the last trusted one.
- **A dropped frame and a jump on one row keep only VL-001.** By the precedence, the jump is then read
  as real motion and the neighbouring speed is too high.
- **`min_cloud_points=0` lets an empty cloud escape VL-006.** The rule is `n_points < min`.
- **`Config(...)` and `replace()` skip validation.** Only `load_config()` (used by the CLIs) validates;
  values set in code are trusted.
- **Thresholds aren't tuned on real data.** They come from the simulator's physical limits; the point
  cloud is generated at a reduced resolution; Kafka is a single local broker.

**Validator and simulator**

- **Labels need `expected_code`.** A `labels.json` whose labels lack the key is rejected (exit 2); use
  `null` for faults with no expected code.
- **Raw drive ids in `labeled_faults_detected`.** A missed fault's detail prints the drive id unescaped
  (the realism check names are escaped).
- **Duplicate `drive_id` entries in the labels' `drives` use the last one.**
- **`--realism` is statistical.** It uses tight limits (p99 of acceleration and yaw rate, maximum speed)
  and can fail a healthy drive: on short drives (about 300 frames, 13 of 45 drives failed in a review
  run) and, rarely, on a default fleet (1 of 40 seeds, a parking drive at 2.003 against 2.0). Use at
  least 1000 frames per drive.
- **Fault magnitudes aren't tied to `Config`.** The jump size (5 m) and the sparse size (10 points) are
  simulator constants; changing `max_jump_m` or `min_cloud_points` doesn't move them.
- **Late arrival isn't strictly 3 frames.** The window is "within about 3 frames".
- **Explicit and random faults can collide.** Random faults avoid each other but not explicit ones; a
  collision raises `ValueError` when the drive is simulated. The Makefile targets use no combination that
  collides.

**Validation and packaging**

- **The Parquet route has no VL-005.** `make validate-warehouse` reads `raw_frames` only, not `rejects`;
  use `make validate` (JSONL) to count malformed lines.
- **`flag_rate` leaves out frames that never arrived.** Its denominator is the received rows, and on a
  Parquet source it also leaves out the rejects, so a stream with many gaps or rejects can pass a low
  `flag_rate`. Read it with `frames_lost` and `duplicates_collapsed`. A strict "received-message" rate
  isn't implemented.
- **Frames lost at the end of a drive are invisible.** `frames_lost` counts gaps inside the `frame_idx`
  sequence; nothing tells the pipeline how many frames a drive should have had.
- **Events stop at flagged frames.** By design, so a maneuver touching a VL-001 or VL-003 frame is split
  into shorter events, or dropped when each part is shorter than `min_event_frames`. Bridging short
  flagged gaps isn't implemented.
- **Validation imports `vl.mock`.** `vl.validation` takes `PROFILE_LIMITS` and `VEHICLES` from the
  simulator package.
- **The prompt log is a selection.** `PROMPTS.md` lists the key prompts, edited for clarity; short
  go-ahead prompts and the subagent briefs (extracts of the task plan) are left out.

**Dashboard**

- **Large drives are thinned.** Above 20,000 points per trace the charts draw an evenly sampled subset,
  with a note in the figure.
- **Reads during `make transform` can fail.** A dashboard left running while the transform overwrites
  the tables may hit missing files.
- **The drive list is read at startup.** New drives appear after a restart.

## Coming soon

- A "deformed cloud" quality rule from the codec's per-frame residual (`max_residuals`)
- Cloud files in an object store (`s3://` references) and a lazy, streaming producer for load tests
- A cross-drive vectorized `derive_frames` for fleets of many short drives (about 54 ms fixed cost per
  drive today, negligible for hour-long drives)
- Delta Lake or Iceberg tables (ACID writes, schema evolution), which also removes the non-atomic
  transform writes
- Containerized jobs (ingest, transform, dashboard in docker-compose)
- A schema registry for the message contract
- Hysteresis in the event detection
- Cloud deployment following the mapping in the architecture document
- A coverage-analysis view (where drives have been)
- Vertical signals from the centroid's `z` (road grade, speed-bump events), with hilly roads in the
  simulator
- Hot reload of the dashboard's drive list
