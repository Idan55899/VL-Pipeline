PY := .venv/bin/python

# Spark workers and driver must use the venv interpreter, not the system python3.
export PYSPARK_PYTHON := $(abspath $(PY))
export PYSPARK_DRIVER_PYTHON := $(abspath $(PY))

# `python -m vl.*` must run from the repo root without installing the package.
export PYTHONPATH := $(CURDIR)/src

.PHONY: setup check bench generate validate validate-warehouse sanity sanity-kafka transform kafka-up kafka-down kafka-reset clean-data produce produce-v2 ingest demo demo-v2

setup:
	python3.11 -m venv .venv
	$(PY) -m pip install -r requirements.txt

check:
	.venv/bin/ruff check src tests
	.venv/bin/mypy --strict src

generate:
	$(PY) -m vl.mock.generate --drives 5 --frames 1000 --seed 0 --out data/mock

# Validate the generator's local JSONL copy (works after `make generate`; counts VL-005 lines).
validate:
	$(PY) -m vl.validate --source data/mock/frames.jsonl --labels data/mock/labels.json --realism

# Validate what Kafka and Spark wrote: the warehouse raw_frames table (rejects are not read, so no VL-005).
validate-warehouse:
	$(PY) -m vl.validate --source data/warehouse/raw_frames --labels data/mock/labels.json --realism

# Incremental: recomputes only drives whose raw_frames changed (a full run when there is no state yet).
transform:
	$(PY) -m vl.etl.transform --warehouse data/warehouse --incremental

# In-memory benchmark of wire formats, cloud codec and per-drive transform; prints a Markdown table.
bench:
	$(PY) -m vl.bench

sanity:
	.venv/bin/pytest -m "sanity and not integration" -p no:cacheprovider

sanity-kafka:
	.venv/bin/pytest -m "sanity and integration" -p no:cacheprovider

KAFKA_BOOTSTRAP ?= localhost:9092
# Total seconds kafka-up may wait for the broker. Each probe takes at most KAFKA_PROBE_S seconds
# (kafka-python retries a dead broker for about 30 s, so a probe is killed by an alarm) and is
# followed by a 1 s sleep, so the worst case is KAFKA_TRIES * (KAFKA_PROBE_S + 1) <= KAFKA_WAIT_S.
KAFKA_WAIT_S ?= 90
KAFKA_PROBE_S ?= 5
KAFKA_TRIES = $(shell echo $$(( $(KAFKA_WAIT_S) / ($(KAFKA_PROBE_S) + 1) )))

# Start the broker and wait until it answers a metadata request (an open TCP port is not
# enough: Docker publishes it before the broker listens).
kafka-up:
	docker compose up -d
	@for i in $$(seq 1 $(KAFKA_TRIES)); do \
	  ( $(PY) -c "import os, signal; signal.signal(signal.SIGALRM, lambda *a: os._exit(1)); signal.alarm($(KAFKA_PROBE_S)); from kafka import KafkaAdminClient as A; A(bootstrap_servers='$(KAFKA_BOOTSTRAP)', request_timeout_ms=2000).list_topics()" ) >/dev/null 2>&1 && exit 0; \
	  sleep 1; \
	done; \
	echo "Kafka did not answer on $(KAFKA_BOOTSTRAP) within $(KAFKA_WAIT_S) s" >&2; exit 1

# Fresh broker: drop the container and its volumes (so the topic and the offsets are gone), start again.
# The ingest checkpoint goes too: its offsets would point past the new, empty topic.
kafka-reset:
	docker compose down -v
	rm -rf data/warehouse/_checkpoints/ingest
	$(MAKE) kafka-up

kafka-down:
	docker compose down

produce:
	$(PY) -m vl.mock.producer --drives 3 --frames 1000 --seed 0 --faults demo --scripted --bootstrap $(KAFKA_BOOTSTRAP) --out-dir data/mock

# The same fleet as message v2 (claim check): clouds to data/clouds, small messages to Kafka.
produce-v2:
	$(PY) -m vl.mock.producer --drives 3 --frames 1000 --seed 0 --faults demo --scripted --bootstrap $(KAFKA_BOOTSTRAP) --out-dir data/mock --cloud-dir data/clouds

ingest:
	$(PY) -m vl.etl.ingest --kafka --bootstrap $(KAFKA_BOOTSTRAP) --topic vl.frames --warehouse data/warehouse

# Remove everything a demo run writes (the warehouse with its streaming checkpoint, and the mock data).
clean-data:
	rm -rf data/warehouse data/mock data/clouds

# One-command, re-runnable demo: clean data, fresh broker, produce, ingest, transform, validate-warehouse
# (the warehouse raw_frames). The dashboard is a separate target (make dashboard).
demo: clean-data kafka-reset produce ingest transform validate-warehouse

# The same demo with message v2: the warehouse and the validator report must match `make demo`.
demo-v2: clean-data kafka-reset produce-v2 ingest transform validate-warehouse

.PHONY: dashboard
dashboard:
	$(PY) -m vl.dashboard.app --warehouse data/warehouse
