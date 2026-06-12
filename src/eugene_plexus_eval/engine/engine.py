"""The eval engine: suite persistence + run a suite against a checkpoint.

Suites and results persist as JSON under ``evalRoot`` so they survive restarts
(suites in ``evalRoot/suites/<id>.json``, results in ``evalRoot/runs/<id>.json``).
A run loads a self-describing checkpoint (resolved under ``checkpointsDir``),
optionally streams a validation dataset (resolved under ``dataRoot``) for
loss/perplexity, and generates sample completions over the suite's prompts for
sample-review + token-entropy. Runs are synchronous — the small local models
this targets evaluate in seconds.

Construction is torch-free (it only reads config and the persisted suites);
torch / tokenizers / pyarrow are imported lazily by the modules a run calls.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from .._generated.models import EvalResult, EvalSuite, Sample, Status
from .checkpoint import CheckpointError, load_checkpoint
from .metrics import (
    compute_val_loss,
    generate_samples,
    perplexity_from_loss,
    token_entropy_nats,
)
from .sampling import SamplingParams
from .tokenizer import InferenceTokenizer, TokenizerError

if TYPE_CHECKING:
    from torch import nn

    from ..config import ConfigStore
    from .checkpoint import LoadedCheckpoint

_VAL_BATCHES = 20
_VAL_BATCH_SIZE = 8
_VAL_SEED = 1337


class EvalError(Exception):
    """Base class for eval errors the routes map to HTTP status codes."""


class NotFoundError(EvalError):
    """A referenced suite / run / checkpoint / dataset does not exist (-> 404)."""


class BadRequestError(EvalError):
    """The request is well-formed but cannot be evaluated (-> 400)."""


class ConflictError(EvalError):
    """The action conflicts with current state, e.g. a duplicate suite id (-> 409)."""


class EvalEngine:
    def __init__(self, config: ConfigStore, *, device: str = "cpu") -> None:
        self._config = config
        self._device = device
        self._lock = threading.RLock()
        eval_root = Path(config.get("evalRoot") or "eval-output")
        self._suites_dir = eval_root / "suites"
        self._runs_dir = eval_root / "runs"
        self._suites: dict[UUID, EvalSuite] = {}
        self._load_suites()

    def _load_suites(self) -> None:
        if not self._suites_dir.exists():
            return
        for path in self._suites_dir.glob("*.json"):
            try:
                suite = EvalSuite.model_validate_json(path.read_text(encoding="utf-8"))
            except Exception:  # skip a corrupt suite file rather than fail startup
                continue
            self._suites[suite.evalSuiteId] = suite

    # ------------------------------------------------------------------ #
    # suite CRUD
    # ------------------------------------------------------------------ #
    def create_suite(self, suite: EvalSuite) -> EvalSuite:
        with self._lock:
            if suite.evalSuiteId in self._suites:
                raise ConflictError(f"eval suite {suite.evalSuiteId} already exists")
            self._suites_dir.mkdir(parents=True, exist_ok=True)
            (self._suites_dir / f"{suite.evalSuiteId}.json").write_text(
                suite.model_dump_json(exclude_none=True), encoding="utf-8"
            )
            self._suites[suite.evalSuiteId] = suite
            return suite.model_copy(deep=True)

    def list_suites(self) -> list[EvalSuite]:
        with self._lock:
            return [s.model_copy(deep=True) for s in self._suites.values()]

    def get_suite(self, eval_suite_id: UUID) -> EvalSuite:
        with self._lock:
            suite = self._suites.get(eval_suite_id)
            if suite is None:
                raise NotFoundError(f"eval suite {eval_suite_id} not found")
            return suite.model_copy(deep=True)

    def delete_suite(self, eval_suite_id: UUID) -> None:
        with self._lock:
            if eval_suite_id not in self._suites:
                raise NotFoundError(f"eval suite {eval_suite_id} not found")
            del self._suites[eval_suite_id]
            (self._suites_dir / f"{eval_suite_id}.json").unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # runs
    # ------------------------------------------------------------------ #
    def run_eval(self, *, eval_suite_id: UUID, checkpoint_id: UUID) -> EvalResult:
        suite = self.get_suite(eval_suite_id)
        result = self._execute(suite, checkpoint_id)
        self._persist_result(result)
        return result

    def get_result(self, eval_run_id: UUID) -> EvalResult:
        path = self._runs_dir / f"{eval_run_id}.json"
        if not path.exists():
            raise NotFoundError(f"eval run {eval_run_id} not found")
        return EvalResult.model_validate_json(path.read_text(encoding="utf-8"))

    def compare(
        self, *, eval_suite_id: UUID, baseline_checkpoint_id: UUID, candidate_checkpoint_id: UUID
    ) -> tuple[EvalResult, EvalResult]:
        suite = self.get_suite(eval_suite_id)
        baseline = self._execute(suite, baseline_checkpoint_id)
        candidate = self._execute(suite, candidate_checkpoint_id)
        self._persist_result(baseline)
        self._persist_result(candidate)
        return baseline, candidate

    # ------------------------------------------------------------------ #
    # execution
    # ------------------------------------------------------------------ #
    def _execute(self, suite: EvalSuite, checkpoint_id: UUID) -> EvalResult:
        path = self._resolve_checkpoint(checkpoint_id)  # NotFoundError -> 404
        try:
            loaded = load_checkpoint(path, map_location="cpu")
            loaded.model.to(self._device)
        except CheckpointError as e:
            raise BadRequestError(str(e)) from e

        metrics = {m.value for m in (suite.metrics or [])}
        val_loss: float | None = None
        perplexity: float | None = None
        token_entropy: float | None = None
        samples: list[Sample] | None = None

        if metrics & {"val_loss", "perplexity"}:
            vl = self._validation_loss(suite, loaded.model)
            if "val_loss" in metrics:
                val_loss = round(vl, 6)
            if "perplexity" in metrics:
                perplexity = round(perplexity_from_loss(vl), 6)

        if metrics & {"sample_review", "token_entropy"}:
            pairs, gen_ids = self._sample_review(suite, loaded)
            if "sample_review" in metrics:
                samples = [Sample(prompt=p, response=r) for p, r in pairs]
            if "token_entropy" in metrics:
                token_entropy = round(token_entropy_nats(gen_ids), 6)

        return EvalResult(
            evalRunId=uuid4(),
            evalSuiteId=suite.evalSuiteId,
            checkpointId=checkpoint_id,
            status=Status.completed,
            valLoss=val_loss,
            perplexity=perplexity,
            tokenEntropy=token_entropy,
            samples=samples,
        )

    def _validation_loss(self, suite: EvalSuite, model: nn.Module) -> float:
        if suite.validationDatasetId is None:
            raise BadRequestError(
                "suite requests val_loss/perplexity but has no validationDatasetId"
            )
        from .dataloader import ArrowShardDataLoader

        try:
            loader = ArrowShardDataLoader(
                data_root=self._data_root(),
                datasets=[(str(suite.validationDatasetId), 1.0)],
                batch_size=_VAL_BATCH_SIZE,
                seed=_VAL_SEED,
            )
        except ValueError as e:  # no pretokenized blocks for the dataset
            raise BadRequestError(
                f"validation dataset {suite.validationDatasetId} has no pretokenized blocks "
                f"under {self._data_root()}: {e}"
            ) from e
        return compute_val_loss(model, loader, n_batches=_VAL_BATCHES, device=self._device)

    def _sample_review(
        self, suite: EvalSuite, loaded: LoadedCheckpoint
    ) -> tuple[list[tuple[str, str]], list[int]]:
        if not suite.prompts:
            raise BadRequestError("suite requests sample_review/token_entropy but has no prompts")
        if loaded.tokenizer_json is None:
            raise BadRequestError(
                "checkpoint has no embedded tokenizer; cannot run sample_review/token_entropy"
            )
        tokenizer = self._build_tokenizer(loaded.tokenizer_json)
        # `is not None` (not `or`) so a configured 0 — valid, and meaning greedy
        # deterministic decoding — isn't silently replaced by the default.
        configured_temp = self._config.get("defaultSamplingTemperature")
        params = SamplingParams(
            temperature=float(configured_temp if configured_temp is not None else 0.8),
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            max_tokens=min(128, loaded.block_size),
        )
        return generate_samples(
            loaded.model,
            tokenizer,
            suite.prompts,
            params=params,
            block_size=loaded.block_size,
            device=self._device,
        )

    @staticmethod
    def _build_tokenizer(tokenizer_json: str) -> InferenceTokenizer:
        try:
            return InferenceTokenizer.from_json(tokenizer_json)
        except TokenizerError as e:
            raise BadRequestError(str(e)) from e

    def _persist_result(self, result: EvalResult) -> None:
        self._runs_dir.mkdir(parents=True, exist_ok=True)
        (self._runs_dir / f"{result.evalRunId}.json").write_text(
            result.model_dump_json(exclude_none=True), encoding="utf-8"
        )

    def _resolve_checkpoint(self, checkpoint_id: UUID) -> Path:
        checkpoints_dir = Path(self._config.get("checkpointsDir") or "eval-checkpoints")
        flat = checkpoints_dir / f"{checkpoint_id}.pt"
        if flat.exists():
            return flat
        folder = checkpoints_dir / str(checkpoint_id)
        if folder.is_dir():
            latest = folder / "latest.pt"
            if latest.exists():
                return latest
            pts = sorted(folder.glob("*.pt"))
            if pts:
                return pts[-1]
        raise NotFoundError(
            f"no checkpoint file for {checkpoint_id} under {checkpoints_dir} "
            f"(looked for {checkpoint_id}.pt and {checkpoint_id}/latest.pt)"
        )

    def _data_root(self) -> Path:
        return Path(self._config.get("dataRoot") or "data-root")
