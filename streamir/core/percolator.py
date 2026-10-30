"""Standing ("continuous") queries - the inverse of search.

Instead of matching one query against many documents, every *incoming* document is matched
against many stored queries.

Posting pruning (prefix-filter idea): a query with n terms and `need` = ceil(min_match * n) required
matches can only fire for a document that contains at least one of ANY (n - need + 1) of its terms
(otherwise at most need-1 terms could match). We therefore index each query under only its
(n - need + 1) *rarest* terms - the rare terms generate far fewer candidates than common ones, and
correctness is preserved for any choice of terms. A `+must` term is required, so one must term
suffices. Candidate verification is then exact.
"""
from __future__ import annotations

import itertools
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .index import ParsedQuery, parse_query


@dataclass
class StandingQuery:
    qid: int
    text: str
    parsed: ParsedQuery
    min_match: float
    name: str
    need: int = 1
    indexed_terms: tuple[str, ...] = ()
    matches: int = 0
    recent: deque = field(default_factory=lambda: deque(maxlen=25))

    def to_dict(self) -> dict:
        return {"id": self.qid, "name": self.name, "query": self.text, "min_match": self.min_match,
                "matches": self.matches, "recent": list(self.recent)}


class Percolator:
    def __init__(self) -> None:
        self._queries: dict[int, StandingQuery] = {}
        self._rev: dict[str, set[int]] = {}
        self._ids = itertools.count(1)

    def register(self, text: str, min_match: float = 0.6, name: str = "",
                 idf: Callable[[str], float] | None = None) -> StandingQuery:
        pq = parse_query(text)
        if pq.empty:
            raise ValueError("query has no searchable terms")
        terms = pq.scoring_terms
        need = max(1, math.ceil(min_match * len(terms) - 1e-9))
        rarity = (lambda t: -idf(t)) if idf else (lambda t: 0.0)       # rarest (highest idf) first
        if pq.must:
            chosen = (min(pq.must, key=rarity),)
        else:
            chosen = tuple(sorted(terms, key=rarity)[: len(terms) - need + 1])
        q = StandingQuery(next(self._ids), text, pq, min_match, name or text, need, chosen)
        self._queries[q.qid] = q
        for t in chosen:
            self._rev.setdefault(t, set()).add(q.qid)
        return q

    def remove(self, qid: int) -> bool:
        q = self._queries.pop(qid, None)
        if not q:
            return False
        for t in q.indexed_terms:
            s = self._rev.get(t)
            if s:
                s.discard(qid)
                if not s:
                    del self._rev[t]
        return True

    def list(self) -> list[StandingQuery]:
        return list(self._queries.values())

    def match(self, doc_terms: set[str], idf: Callable[[str], float]) -> list[tuple[StandingQuery, float]]:
        cands: set[int] = set()
        for t in doc_terms:
            s = self._rev.get(t)
            if s:
                cands |= s
        out = []
        for qid in cands:
            q = self._queries[qid]
            p = q.parsed
            if any(t in doc_terms for t in p.must_not):
                continue
            if any(t not in doc_terms for t in p.must):
                continue
            terms = p.scoring_terms
            matched = [t for t in terms if t in doc_terms]
            if len(matched) < q.need:
                continue
            total = sum(idf(t) for t in terms) or 1.0
            out.append((q, sum(idf(t) for t in matched) / total))
        return out
