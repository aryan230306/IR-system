from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default, cast=str):
    v = os.environ.get(f"STREAMIR_{name}")
    return cast(v) if v is not None else default


@dataclass
class Settings:
    window_seconds: float = field(default_factory=lambda: _env("WINDOW", 1800.0, float))
    memtable_max_docs: int = field(default_factory=lambda: _env("MEMTABLE", 1000, int))
    wal_path: str | None = field(default_factory=lambda: _env("WAL", "data/wal.jsonl"))
    queue_size: int = field(default_factory=lambda: _env("QUEUE", 10_000, int))
    dedup_threshold: float = field(default_factory=lambda: _env("DEDUP", 0.8, float))
    cluster_threshold: float = field(default_factory=lambda: _env("CLUSTER", 0.30, float))
    burst_bucket: float = field(default_factory=lambda: _env("BURST_BUCKET", 10.0, float))
    burst_z: float = field(default_factory=lambda: _env("BURST_Z", 4.0, float))
    semantic_dim: int = field(default_factory=lambda: _env("DIM", 256, int))
    source: str = field(default_factory=lambda: _env("SOURCE", "synthetic"))
    rate: float = field(default_factory=lambda: _env("RATE", 40.0, float))
