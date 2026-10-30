"""Online burst / trend detection on term frequencies.

Time is cut into fixed buckets. For each term we keep an exponentially-weighted moving mean and
variance of its per-bucket count. A term is *bursting* when its count in the current (partial)
bucket is significantly above that baseline (Poisson-aware z-score). Memory is bounded by pruning
terms whose baseline decays to ~0.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass


@dataclass
class _Stat:
    mean: float = 0.0
    var: float = 0.0
    n: int = 0


class BurstDetector:
    def __init__(self, bucket_seconds: float = 30.0, alpha: float = 0.15, min_count: int = 5,
                 z_threshold: float = 4.0, warmup_buckets: int = 4, max_terms: int = 50_000) -> None:
        self.bucket = bucket_seconds
        self.alpha = alpha
        self.min_count = min_count
        self.z = z_threshold
        self.warmup = warmup_buckets
        self.max_terms = max_terms
        self.stats: dict[str, _Stat] = {}
        self.current: Counter = Counter()
        self.cur_id: int | None = None
        self.closed_buckets = 0

    def observe(self, terms, ts: float) -> None:
        """Feed the *distinct* terms of one document."""
        bid = int(ts // self.bucket)
        if self.cur_id is None:
            self.cur_id = bid
        if bid > self.cur_id:
            self._close(bid - self.cur_id)
            self.cur_id = bid
        if bid < self.cur_id:          # late event: attribute to the current bucket
            pass
        self.current.update(set(terms))

    def tick(self, now: float) -> None:
        """Advance time even if no documents arrive (so quiet periods decay baselines)."""
        bid = int(now // self.bucket)
        if self.cur_id is not None and bid > self.cur_id:
            self._close(bid - self.cur_id)
            self.cur_id = bid

    def _close(self, gap: int) -> None:
        a = self.alpha
        seen = self.current
        for term in set(self.stats) | set(seen):
            st = self.stats.setdefault(term, _Stat())
            c = seen.get(term, 0)
            if st.n == 0:
                st.mean = float(c)
            else:
                diff = c - st.mean
                incr = a * diff
                st.mean += incr
                st.var = (1 - a) * (st.var + diff * incr)
            st.n += 1
        # idle buckets (gap>1): decay toward zero
        for _ in range(max(0, gap - 1)):
            for st in self.stats.values():
                diff = -st.mean
                incr = a * diff
                st.mean += incr
                st.var = (1 - a) * (st.var + diff * incr)
        self.closed_buckets += gap
        self.current = Counter()
        if len(self.stats) > self.max_terms:
            self.stats = {t: s for t, s in self.stats.items() if s.mean > 0.05 or s.n < 3}

    def score(self, term: str) -> float:
        c = self.current.get(term, 0)
        st = self.stats.get(term)
        mean, var = (st.mean, st.var) if st else (0.0, 0.0)
        return (c - mean) / math.sqrt(max(var, mean, 0.25) + 1.0)

    def top(self, k: int = 10) -> list[dict]:
        if self.closed_buckets < self.warmup:
            return []
        out = []
        for term, c in self.current.items():
            if c < self.min_count:
                continue
            z = self.score(term)
            if z >= self.z:
                st = self.stats.get(term)
                out.append({"term": term, "count": c, "baseline": round(st.mean if st else 0.0, 2),
                            "z": round(z, 2)})
        out.sort(key=lambda r: -r["z"])
        return out[:k]
