"""Pytest fixtures + factories for the eval test suite.

Factories produce the on-disk artifacts a real eval consumes: self-describing
checkpoints under ``checkpointsDir`` (the exact shape the trainer writes) and
pretokenized Arrow validation datasets under ``dataRoot`` (the shape the data
component writes). ``make_arithmetic_*`` use a learnable next=prev+1 pattern so
a trained checkpoint provably scores a lower val-loss than an untrained one;
``make_text_checkpoint`` embeds a real byte-level BPE tokenizer so sample-review
generation runs end to end.

torch / tokenizers / pyarrow are imported inside the factories (test deps).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from eugene_plexus_eval.app import create_app
from eugene_plexus_eval.settings import Settings

_TINY_ARCH = {
    "modelType": "decoder_only",
    "nLayer": 2,
    "nHead": 2,
    "nKvHead": 1,
    "nEmbd": 32,
    "blockSize": 64,
}
_BLOCK_LEN = 17  # arithmetic block length (rows of (start+i) % vocab)


@pytest.fixture
def checkpoints_dir(tmp_path: Path) -> Path:
    d = tmp_path / "ckpts"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    d = tmp_path / "data-root"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def settings(tmp_path: Path, checkpoints_dir: Path, data_root: Path) -> Settings:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "evalRoot": str(tmp_path / "eval-output"),
                "checkpointsDir": str(checkpoints_dir),
                "dataRoot": str(data_root),
            }
        )
    )
    return Settings(config_file=config_path)


@pytest.fixture
def app(settings: Settings) -> FastAPI:
    return create_app(settings=settings)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


# --------------------------------------------------------------------------- #
# factories
# --------------------------------------------------------------------------- #


def _build_arch(vocab: int, **over: object) -> object:
    from eugene_plexus_eval._generated.common_models import ArchitectureConfig

    return ArchitectureConfig(**{**_TINY_ARCH, "vocabSize": vocab, **over})  # type: ignore[arg-type]


def _save_ckpt(path: Path, model: object, arch: object, *, tokenizer_json: str | None) -> None:
    import torch

    payload = {
        "model": model.state_dict(),  # type: ignore[attr-defined]
        "meta": {
            "architecture": arch.model_dump(mode="json"),  # type: ignore[attr-defined]
            "tokenizer": {
                "tokenizerId": None,
                "vocabFingerprint": None,
                "tokenizerJson": tokenizer_json,
            },
            "step": 0,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _train_arithmetic(model: object, *, vocab: int, steps: int, seed: int = 0) -> None:
    import torch

    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)  # type: ignore[attr-defined]
    model.train()  # type: ignore[attr-defined]
    for _ in range(steps):
        starts = [rng.randrange(vocab) for _ in range(16)]
        rows = [[(s + i) % vocab for i in range(_BLOCK_LEN)] for s in starts]
        batch = torch.tensor(rows, dtype=torch.long)
        _, loss = model(batch[:, :-1], batch[:, 1:])  # type: ignore[misc]
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()  # type: ignore[attr-defined]


def _make_tokenizer_json() -> tuple[str, int]:
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer

    tk = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    tk.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tk.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=300,
        min_frequency=1,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        initial_alphabet=ByteLevel.alphabet(),
        show_progress=False,
    )
    tk.train_from_iterator(
        ["the quick brown fox", "hello world from eugene plexus", "evaluate the model now"],
        trainer=trainer,
    )
    return tk.to_str(), tk.get_vocab_size()


def _write_arrow_dataset(data_root: Path, dataset_id: str, *, vocab: int, n_blocks: int) -> None:
    import pyarrow as pa

    arrow_dir = data_root / "datasets" / dataset_id / "arrow"
    arrow_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    # One random start per row, then next=prev+1 (mod vocab) — the SAME learnable
    # pattern _train_arithmetic teaches, so a trained model scores low val-loss.
    rows: list[list[int]] = []
    for _ in range(n_blocks):
        start = rng.randrange(vocab)
        rows.append([(start + i) % vocab for i in range(_BLOCK_LEN)])
    schema = pa.schema([("input_ids", pa.list_(pa.int32()))])
    with (
        pa.OSFile(str(arrow_dir / "shard-00000.arrow"), "wb") as sink,
        pa.ipc.new_file(sink, schema) as writer,
    ):
        writer.write_table(pa.table({"input_ids": rows}, schema=schema))


@pytest.fixture
def make_arithmetic_checkpoint(checkpoints_dir: Path) -> Callable[..., str]:
    """Factory -> checkpoint_id. Trained (or untrained) arithmetic model, no tokenizer."""

    def _factory(*, vocab: int = 32, steps: int = 250) -> str:
        import torch

        from eugene_plexus_eval.engine.model import GPTModel

        torch.manual_seed(0)
        checkpoint_id = str(uuid4())
        arch = _build_arch(vocab)
        model = GPTModel(arch)  # type: ignore[arg-type]
        if steps:
            _train_arithmetic(model, vocab=vocab, steps=steps)
        _save_ckpt(checkpoints_dir / f"{checkpoint_id}.pt", model, arch, tokenizer_json=None)
        return checkpoint_id

    return _factory


@pytest.fixture
def make_arithmetic_dataset(data_root: Path) -> Callable[..., str]:
    """Factory -> dataset_id. Pretokenized arithmetic Arrow blocks under dataRoot."""

    def _factory(*, vocab: int = 32, n_blocks: int = 256) -> str:
        dataset_id = str(uuid4())
        _write_arrow_dataset(data_root, dataset_id, vocab=vocab, n_blocks=n_blocks)
        return dataset_id

    return _factory


@pytest.fixture
def make_text_checkpoint(checkpoints_dir: Path) -> Callable[..., str]:
    """Factory -> checkpoint_id. (Untrained) model + real tokenizer for sample-review."""

    def _factory() -> str:
        from eugene_plexus_eval.engine.model import GPTModel

        checkpoint_id = str(uuid4())
        tokenizer_json, vocab = _make_tokenizer_json()
        arch = _build_arch(vocab)
        model = GPTModel(arch)  # type: ignore[arg-type]
        _save_ckpt(
            checkpoints_dir / f"{checkpoint_id}.pt", model, arch, tokenizer_json=tokenizer_json
        )
        return checkpoint_id

    return _factory
