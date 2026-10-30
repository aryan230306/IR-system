"""Unified retrieval facade: BM25, dense (random-indexing) and hybrid (RRF), with time decay."""
from __future__ import annotations

import math
import time

from ..models import Hit
from ..text import make_snippet, tokenize
from .index import StreamIndex, parse_query
from .semantic import SemanticIndex

MODES = ("bm25", "dense", "hybrid")


def rrf(rankings: list[list[int]], k: int = 60, weights: list[float] | None = None) -> dict[int, float]:
    """Reciprocal Rank Fusion (Cormack et al., 2009)."""
    weights = weights or [1.0] * len(rankings)
    out: dict[int, float] = {}
    for w, ranking in zip(weights, rankings):
        for r, did in enumerate(ranking, start=1):
            out[did] = out.get(did, 0.0) + w / (k + r)
    return out


class Retriever:
    def __init__(self, index: StreamIndex, semantic: SemanticIndex, min_sim: float = 0.2) -> None:
        self.index, self.semantic = index, semantic
        self.min_sim = min_sim

    def _hit(self, did: int, score: float, qset: set[str], explain: dict | None = None) -> Hit:
        d = self.index.docs[did]
        return Hit(d.id, score, d.title, make_snippet(d.text, qset), d.ts, d.source, explain or {})

    def search(self, query: str, k: int = 10, mode: str = "hybrid", *, halflife: float | None = None,
               since: float | None = None, now: float | None = None, explain: bool = False) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "bm25":
            return self.index.search(query, k, halflife=halflife, since=since, now=now, explain=explain)

        now = self.index.clock() if now is None else now
        cutoff = max(now - self.index.window, since if since is not None else -math.inf)
        pq = parse_query(query)
        if pq.empty:
            return []
        qtf = _count(pq.scoring_terms)
        live = lambda did: (d := self.index.docs.get(did)) is not None and d.ts >= cutoff
        dense = self.semantic.search(self.semantic.encode(qtf, self.index.idf), max(50, 5 * k), live,
                                     min_sim=self.min_sim)
        # honour +must / -not constraints on dense candidates too
        allowed = self._constraint_filter(pq, [d for d, _ in dense])
        dense = [(d, s) for d, s in dense if d in allowed]

        if mode == "dense":
            fused = dict(dense)
            ranks = {"dense": {d: i + 1 for i, (d, _) in enumerate(dense)}}
        else:
            lex = self.index.score(pq, halflife=None, since=since, now=now)
            lex_rank = [d for d, _ in sorted(lex.items(), key=lambda kv: -kv[1])[:100]]
            dense_rank = [d for d, _ in dense]
            fused = rrf([lex_rank, dense_rank], weights=[1.0, 0.7])
            ranks = {"bm25": {d: i + 1 for i, d in enumerate(lex_rank)},
                     "dense": {d: i + 1 for i, d in enumerate(dense_rank)}}
        if halflife:
            ln2 = math.log(2)
            for did in fused:
                fused[did] *= math.exp(-ln2 * max(0.0, now - self.index.docs[did].ts) / halflife)
        top = sorted(fused.items(), key=lambda kv: -kv[1])[:k]
        qset = set(pq.scoring_terms)
        return [self._hit(d, s, qset, {n: r[d] for n, r in ranks.items() if d in r} if explain else None)
                for d, s in top]

    def _constraint_filter(self, pq, dids: list[int]) -> set[int]:
        if not pq.must and not pq.must_not:
            return set(dids)
        ok = set()
        for did in dids:
            tf = self._doc_terms(did)
            if all(t in tf for t in pq.must) and not any(t in tf for t in pq.must_not):
                ok.add(did)
        return ok

    def _doc_terms(self, did: int) -> set[str]:
        d = self.index.docs[did]
        return set(self.index.analyze(d))


def _count(terms):
    from collections import Counter
    return Counter(terms)
