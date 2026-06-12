"""Arrow-shard dataloader.

Reads the pretokenized fixed-size token blocks the `data` component wrote
(``<data_root>/datasets/<datasetId>/arrow/shard-*.arrow``, one block of int ids per
row) and yields random training batches. Shards are memory-mapped, so RAM stays
bounded regardless of corpus size. Multiple datasets are sampled by their relative
weight. Each batch row is one block; the loop forms (x, y) as (row[:-1], row[1:]).
"""

from __future__ import annotations

import bisect
from pathlib import Path

import pyarrow as pa
import torch


class ArrowShardDataLoader:
    def __init__(
        self,
        *,
        data_root: Path,
        datasets: list[tuple[str, float]],
        batch_size: int,
        seed: int = 1337,
    ) -> None:
        self.batch_size = batch_size
        self._gen = torch.Generator().manual_seed(seed)
        self._tables: list[list[pa.Table]] = []
        self._offsets: list[list[int]] = []
        self._totals: list[int] = []
        # The Arrow tables are zero-copy views over the memory maps, so the maps
        # must stay open for the loader's lifetime — keep references here.
        self._mmaps: list[pa.MemoryMappedFile] = []
        weights: list[float] = []

        for dataset_id, weight in datasets:
            arrow_dir = data_root / "datasets" / dataset_id / "arrow"
            tables: list[pa.Table] = []
            offsets: list[int] = []
            running = 0
            for shard in sorted(arrow_dir.glob("shard-*.arrow")):
                mm = pa.memory_map(str(shard), "r")
                self._mmaps.append(mm)
                table = pa.ipc.open_file(mm).read_all()
                offsets.append(running)
                running += table.num_rows
                tables.append(table)
            if running == 0:
                continue
            self._tables.append(tables)
            self._offsets.append(offsets)
            self._totals.append(running)
            weights.append(max(1e-6, weight))

        if not self._totals:
            raise ValueError("no pretokenized blocks found for the requested datasets")
        self._weights = torch.tensor(weights, dtype=torch.float)

    @property
    def total_blocks(self) -> int:
        return sum(self._totals)

    def _row(self, dataset_idx: int, row: int) -> list[int]:
        tables = self._tables[dataset_idx]
        offsets = self._offsets[dataset_idx]
        # offsets are ascending cumulative shard starts; row is in [0, total).
        shard = bisect.bisect_right(offsets, row) - 1
        local = row - offsets[shard]
        return tables[shard].column("input_ids")[local].as_py()

    def next_batch(self) -> torch.Tensor:
        ds_choices = torch.multinomial(
            self._weights, self.batch_size, replacement=True, generator=self._gen
        )
        rows: list[list[int]] = []
        for ds_idx_t in ds_choices.tolist():
            total = self._totals[ds_idx_t]
            row = int(torch.randint(0, total, (1,), generator=self._gen).item())
            rows.append(self._row(ds_idx_t, row))
        return torch.tensor(rows, dtype=torch.long)
