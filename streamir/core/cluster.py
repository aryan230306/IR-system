"""Single-pass online story clustering (TDT-style) over sparse TF-IDF vectors."""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Cluster:
    cid: int
    centroid: dict[str, float] = field(default_factory=dict)   # running (unnormalised) sum
    norm: float = 0.0
    size: int = 0
    first_ts: float = 0.0
    last_ts: float = 0.0
    doc_ids: list[str] = field(default_factory=list)
    title: str = ""
    sources: set[str] = field(default_factory=set)

    def top_terms(self, n: int = 5) -> list[str]:
        return [t for t, _ in sorted(self.centroid.items(), key=lambda kv: -kv[1])[:n]]


class OnlineClusterer:
    def __init__(self, threshold: float = 0.30, max_active: int = 400, ttl: float = 3600.0,
                 max_terms_per_centroid: int = 60) -> None:
        self.threshold = threshold
        self.max_active = max_active
        self.ttl = ttl
        self.max_terms = max_terms_per_centroid
        self.clusters: dict[int, Cluster] = {}
        self._next = 0

    @staticmethod
    def vectorize(tf: Counter, idf) -> tuple[dict[str, float], float]:
        v = {t: (1 + math.log(c)) * idf(t) for t, c in tf.items()}
        n = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / n for t, x in v.items()}, n

    def assign(self, doc_id: str, title: str, source: str, tf: Counter, idf, ts: float) -> tuple[int, float]:
        vec, _ = self.vectorize(tf, idf)
        best, best_sim = None, 0.0
        for c in self.clusters.values():
            if c.norm == 0:
                continue
            small, big = (vec, c.centroid) if len(vec) < len(c.centroid) else (c.centroid, vec)
            dot = sum(w * big.get(t, 0.0) for t, w in small.items())
            sim = dot / c.norm
            if sim > best_sim:
                best, best_sim = c, sim
        if best is None or best_sim < self.threshold:
            best = Cluster(self._next, first_ts=ts, title=title)
            self._next += 1
            self.clusters[best.cid] = best
            best_sim = 1.0
        for t, w in vec.items():
            best.centroid[t] = best.centroid.get(t, 0.0) + w
        if len(best.centroid) > self.max_terms:
            keep = sorted(best.centroid.items(), key=lambda kv: -kv[1])[: self.max_terms]
            best.centroid = dict(keep)
        best.norm = math.sqrt(sum(x * x for x in best.centroid.values()))
        best.size += 1
        best.last_ts = max(best.last_ts, ts)
        best.sources.add(source)
        if len(best.doc_ids) < 50:
            best.doc_ids.append(doc_id)
        self._evict(ts)
        return best.cid, best_sim

    def _evict(self, now: float) -> None:
        stale = [cid for cid, c in self.clusters.items() if now - c.last_ts > self.ttl]
        for cid in stale:
            del self.clusters[cid]
        if len(self.clusters) > self.max_active:
            for cid, _ in sorted(self.clusters.items(), key=lambda kv: kv[1].last_ts)[: len(self.clusters) - self.max_active]:
                del self.clusters[cid]

    def top(self, k: int = 10, min_size: int = 2) -> list[dict]:
        cs = [c for c in self.clusters.values() if c.size >= min_size]
        cs.sort(key=lambda c: (-c.size, -c.last_ts))
        return [{"id": c.cid, "title": c.title, "size": c.size, "terms": c.top_terms(),
                 "sources": len(c.sources), "first_ts": c.first_ts, "last_ts": c.last_ts,
                 "doc_ids": c.doc_ids[:5]} for c in cs[:k]]
