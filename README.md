# eugene-plexus-eval

Checkpoint evaluation engine for [Eugene Plexus](https://github.com/eugene-plexus).

## What this is

The eval component of Eugene Plexus. It evaluates checkpoints produced by the
`trainer`: it runs eval prompt suites, computes perplexity / validation loss,
logs generated samples for review, detects mode collapse (sampled-token
entropy), and compares checkpoints for regression. It references checkpoints
(via `common.yaml#/components/schemas/Checkpoint`) and validation datasets
(via `DatasetRef`). It does NOT own the `TrainingProject` aggregate — that
lives in the `coordinator`.

```
GET    /v1/eval/suites                  list eval suites
POST   /v1/eval/suites                  create an eval suite
GET    /v1/eval/suites/{evalSuiteId}    read one eval suite
DELETE /v1/eval/suites/{evalSuiteId}    delete an eval suite
POST   /v1/eval/runs                    run a suite against a checkpoint
GET    /v1/eval/runs/{evalRunId}        read an eval run's result
POST   /v1/eval/compare                 compare two checkpoints (regression check)
```

Plus the standard Eugene Plexus config trio (`GET /v1/config`,
`GET /v1/config/schema`, `PATCH /v1/config`), `POST /v1/config/test`,
`POST /v1/admin/restart`, and `GET /healthz`.

## v0.3 skeleton status

This repo currently ships the **control-plane skeleton**: the HTTP wire
shape (routes + generated models + config + auth + health + safe mode) is
complete, but the actual eval-execution engine is **not implemented
yet**. The run-control endpoints (`POST /v1/eval/runs`,
`GET /v1/eval/runs/{evalRunId}`, `POST /v1/eval/compare`) return `501 Not
Implemented` with a standard `Problem` body explaining that the eval
engine is future work. The suite-listing endpoints return empty / real
responses: `GET /v1/eval/suites` returns an empty list, and the suite
CRUD shapes are the long-term contract.

## Quick start

```bash
pip install -e ".[dev]"
python -m eugene_plexus_eval
# default port 8089; override via PATCH /v1/config or the config file
```

The first run creates a `config.yaml` in the working directory with the
component's defaults. Edit through the UI, through `PATCH /v1/config`, or
by hand.

## Degraded-mode startup

Per the project-wide rule (`feedback_degraded_mode_required.md`), a bad
config never prevents the component from starting. Config endpoints stay
reachable so operators can fix the broken setting through the UI;
domain endpoints behave according to the skeleton (501) until the
eval engine lands.

## Codegen

Pydantic models for the eval component and shared schemas are generated
from the pinned `eugene-plexus/specs` commit:

```bash
python scripts/codegen.py
```

`SPECS_REF` records the commit SHA. Bump it to track a newer specs
release; CI re-runs codegen and fails if the working tree drifts.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and
[`CONTRIBUTING.md`](CONTRIBUTING.md) (DCO sign-off required).
