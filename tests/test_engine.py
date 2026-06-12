"""Unit tests for the pure eval-metric functions.

The integration paths (val-loss over a real dataset, sample-review generation)
are exercised end to end by test_eval_routes.py; here we pin the metric math.
"""

from __future__ import annotations

import math
from pathlib import Path
from uuid import uuid4

from eugene_plexus_eval.engine.metrics import perplexity_from_loss, token_entropy_nats


class _Cfg:
    """Minimal ConfigStore stand-in for engine unit tests."""

    def __init__(self, **values: object) -> None:
        self._values = values

    def get(self, key: str) -> object:
        return self._values.get(key)


def test_token_entropy_zero_for_empty() -> None:
    assert token_entropy_nats([]) == 0.0


def test_token_entropy_zero_for_single_token() -> None:
    # A fully mode-collapsed output (one token repeated) has zero entropy.
    assert token_entropy_nats([5, 5, 5, 5]) == 0.0


def test_token_entropy_max_for_uniform() -> None:
    # n distinct tokens, each once -> ln(n) nats.
    assert token_entropy_nats([0, 1, 2, 3]) == math.log(4)


def test_perplexity_from_loss() -> None:
    assert perplexity_from_loss(0.0) == 1.0
    assert math.isclose(perplexity_from_loss(1.0), math.e)


def test_perplexity_clamps_huge_loss() -> None:
    # An untrained model can produce an absurd loss; the exponent is clamped so
    # perplexity stays finite rather than overflowing to inf.
    assert math.isfinite(perplexity_from_loss(1e6))
    assert perplexity_from_loss(1e6) == math.exp(80.0)


def test_compute_val_loss_excludes_moe_aux_loss() -> None:
    """For an MoE checkpoint the model's training loss is ce + aux; the eval
    metric must report pure cross-entropy, not the inflated training loss."""
    import torch
    from torch.nn import functional as F

    from eugene_plexus_eval._generated.common_models import ArchitectureConfig
    from eugene_plexus_eval.engine.metrics import compute_val_loss
    from eugene_plexus_eval.engine.model import GPTModel

    torch.manual_seed(0)
    arch = ArchitectureConfig.model_validate(
        {
            "modelType": "decoder_only",
            "nLayer": 2,
            "nHead": 2,
            "nKvHead": 1,
            "nEmbd": 32,
            "blockSize": 64,
            "vocabSize": 32,
            "ffn": {"type": "moe", "nExperts": 4, "topK": 2},
        }
    )
    model = GPTModel(arch)
    batch = torch.randint(0, 32, (4, 17))

    class _FixedLoader:
        def next_batch(self) -> torch.Tensor:
            return batch

    reported = compute_val_loss(model, _FixedLoader(), n_batches=1)  # type: ignore[arg-type]

    model.eval()
    with torch.no_grad():
        logits, combined = model(batch[:, :-1], batch[:, 1:])
        pure_ce = float(F.cross_entropy(logits.reshape(-1, 32), batch[:, 1:].reshape(-1)))

    assert abs(reported - pure_ce) < 1e-5  # eval reports pure CE ...
    assert float(combined) > pure_ce  # ... while the training loss carries a positive MoE aux


def test_suites_persist_across_engine_restart(tmp_path: Path) -> None:
    from eugene_plexus_eval._generated.models import EvalSuite
    from eugene_plexus_eval.engine.engine import EvalEngine

    cfg = _Cfg(evalRoot=str(tmp_path / "er"))
    suite_id = uuid4()
    EvalEngine(cfg, device="cpu").create_suite(
        EvalSuite(evalSuiteId=suite_id, name="persisted", metrics=["perplexity"])
    )
    # A fresh engine over the same evalRoot reloads suites from disk.
    reloaded = EvalEngine(cfg, device="cpu").get_suite(suite_id)
    assert reloaded.name == "persisted"


def test_corrupt_suite_file_skipped_at_construction(tmp_path: Path) -> None:
    from eugene_plexus_eval.engine.engine import EvalEngine

    suites_dir = tmp_path / "er" / "suites"
    suites_dir.mkdir(parents=True)
    (suites_dir / "garbage.json").write_text("not valid json {{", encoding="utf-8")
    # Construction must tolerate a corrupt suite file rather than raise.
    engine = EvalEngine(_Cfg(evalRoot=str(tmp_path / "er")), device="cpu")
    assert engine.list_suites() == []
