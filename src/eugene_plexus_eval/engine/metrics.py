"""The eval metrics themselves.

* ``compute_val_loss`` — mean next-token cross-entropy over N batches of a
  validation dataset (a plain ``model(x, y)`` forward).
* ``perplexity_from_loss`` — ``exp(val_loss)``.
* ``token_entropy_nats`` — Shannon entropy (nats) of the empirical distribution
  of generated token ids; a mode-collapse signal (low = the model keeps
  emitting the same few tokens). In nats to match the ``minLogitEntropy``
  config field's unit.
* ``generate_samples`` — per-prompt continuation, for qualitative sample review
  and as the source of tokens for ``token_entropy``.

torch is imported lazily inside the functions so the module imports without it.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import TYPE_CHECKING

from .generate import generate
from .sampling import SamplingParams

if TYPE_CHECKING:
    from torch import nn

    from .dataloader import ArrowShardDataLoader
    from .tokenizer import InferenceTokenizer


def compute_val_loss(
    model: nn.Module, loader: ArrowShardDataLoader, *, n_batches: int, device: str = "cpu"
) -> float:
    """Mean next-token cross-entropy over ``n_batches`` batches. Each batch row
    is one pretokenized block; (x, y) = (row[:-1], row[1:]).

    Cross-entropy is computed directly from the logits — NOT by reusing the
    model's training loss, which is ``ce + aux`` (the MoE/MoD load-balancing
    auxiliary term). The eval metric must measure prediction quality, not the
    training regularizer, or perplexity would be inflated for sparse models and
    a compare() across architectures would not be apples-to-apples.
    """
    import torch
    from torch.nn import functional as F

    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            batch = loader.next_batch().to(device)
            x, y = batch[:, :-1], batch[:, 1:]
            logits, _ = model(x)  # targets=None -> no aux folded in
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total += float(ce)
    return total / max(1, n_batches)


def perplexity_from_loss(val_loss: float) -> float:
    # Clamp the exponent so an untrained model's large loss can't overflow.
    return math.exp(min(val_loss, 80.0))


def token_entropy_nats(token_ids: list[int]) -> float:
    counter = Counter(token_ids)
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counter.values())


def generate_samples(
    model: nn.Module,
    tokenizer: InferenceTokenizer,
    prompts: list[str],
    *,
    params: SamplingParams,
    block_size: int,
    device: str = "cpu",
) -> tuple[list[tuple[str, str]], list[int]]:
    """For each prompt: encode (raw, no special framing — eval tests the base
    model's continuation) -> generate -> decode. Returns ``(prompt, response)``
    pairs and the flat list of all generated token ids (for entropy)."""
    import torch

    samples: list[tuple[str, str]] = []
    all_ids: list[int] = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not ids:
            ids = [tokenizer.eos_id]  # never feed an empty sequence
        input_ids = torch.tensor([ids[-block_size:]], dtype=torch.long, device=device)
        result = generate(
            model, input_ids, params=params, eos_id=tokenizer.eos_id, block_size=block_size
        )
        all_ids.extend(result.token_ids)
        samples.append((prompt, tokenizer.decode(result.token_ids)))
    return samples, all_ids
