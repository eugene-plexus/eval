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

## Eval engine

The eval engine is implemented. It persists eval suites, then runs a suite
against a checkpoint and reports metrics:

- **`val_loss`** — mean next-token cross-entropy over a validation dataset
  (resolved under `dataRoot` from the suite's `validationDatasetId`).
- **`perplexity`** — `exp(val_loss)`.
- **`token_entropy`** — Shannon entropy (nats) of the tokens generated across
  the suite's prompts; a mode-collapse signal (low = the model keeps emitting
  the same few tokens).
- **`sample_review`** — a generated continuation per prompt, for qualitative
  review.

A run loads the same **self-describing checkpoint** the trainer writes
(rebuilding the model standalone from `meta.architecture` + `meta.tokenizer`),
resolved under `checkpointsDir` as `<checkpointId>.pt` (or
`<checkpointId>/latest.pt`). `POST /v1/eval/compare` runs a suite against two
checkpoints and returns both results — the seeded validation loader feeds both
the same batches, so the comparison is fair. Runs are synchronous (the small
local models this targets evaluate in seconds).

**v0.3 first-cut limits** (each a clean follow-up): CPU only; runs synchronously
(no async run queue); the validation batch count is a fixed internal default.

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
reachable so operators can fix the broken setting through the UI. The
engine builds without torch present (it's imported lazily on first run), so
the control plane always comes up; in **safe mode**
(`EUGENE_PLEXUS_EVAL_SAFE_MODE=1`) eval execution is disabled and the
run/compare routes return `503` while config stays editable.

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
