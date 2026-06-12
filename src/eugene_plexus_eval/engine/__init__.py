"""The eval engine.

Runs an eval suite against a checkpoint and reports metrics. It loads the same
self-describing checkpoints the trainer writes (rebuilding the model standalone
from ``meta.architecture`` + ``meta.tokenizer``), computes validation loss /
perplexity over a held-out dataset, and generates sample completions for the
suite's prompts (from which a sampled-token entropy / mode-collapse signal is
derived).

The model code (``model``/``attention``/``layers``/``block``), the checkpoint
loader, tokenizer, sampler, and decode loop are copied verbatim from the
inference engine, and the Arrow-shard reader from the trainer, so an eval loads
exactly what those components produce/serve. torch / tokenizers / pyarrow are
imported lazily, so the control plane boots without them; an eval run then
fails with a clear error if they're missing.
"""

from __future__ import annotations
