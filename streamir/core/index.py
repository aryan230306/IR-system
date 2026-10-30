"""Near-real-time inverted index (LSM-style).

Architecture
------------
* A mutable **memtable** receives every new document and is searchable immediately
  (visibility latency ~ 0, no refresh interval as in batch engines).
* When the memtable reaches `memtable_max_docs` it is frozen into an immutable
  **segment** and a fresh memtable is started.
* A **tiered merge policy** compacts the smallest segments, physically dropping
  deleted / expired documents (tombstones) during the rewrite.
* A **sliding retention window** expires documents by *event time*; whole segments whose
  newest document is outside the window are dropped in O(1).
* Ranking is **Okapi BM25** with global statistics and optional exponential **time decay**
  (half-life), so fresh content outranks stale content of equal relevance.
"""
from __future__ import annotations

import heapq
import math
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable

from ..models import Document, Hit
from ..text import make_snippet, tokenize


@dataclass
class Segment:
    seg_id: int
    postings: dict[str, dict[int, int]] = field(default_factory=dict)  # term -> {did: tf}
    doc_len: dict[int, int] = field(default_factory=dict)
    min_ts: float = math.inf
    max_ts: float = -math.inf
    frozen: bool = False

    def add(self, did: int, tf: Counter, ts: float) -> None:
        for term, c in tf.items():
            self.postings.setdefault(term, {})[did] = c
        self.doc_len[did] = sum(tf.values())
        self.min_ts = min(self.min_ts, ts)
        self.max_ts = max(self.max_ts, ts)

    def __len__(self) -> int:
        return len(self.doc_len)


@dataclass
class ParsedQuery:
    should: list[str]
    must: list[str]
    must_not: list[str]

    @property
    def scoring_terms(self) -> list[str]:
        return list(dict.fromkeys(self.should + self.must))

    @property
    def empty(self) -> bool:
        return not self.scoring_terms


def parse_query(q: str) -> ParsedQuery:
    """`foo +bar -baz` -> should=[foo], must=[bar], must_not=[baz]. Quoted text is AND-ed."""
    should, must, must_not = [], [], []
    i, n = 0, len(q)
    while i < n:
        ch = q[i]
        if ch.isspace():
            i += 1
            continue
        prefix = ""
        if ch in "+-":
            prefix, i = ch, i + 1
            if i >= n:
                break
        if i < n and q[i] == '"':
            j = q.find('"', i + 1)
            j = n if j == -1 else j
            toks = tokenize(q[i + 1:j])
            i = j + 1
            bucket = must_not if prefix == "-" else must  # quoted phrase => every word required
        else:
            j = i
            while j < n and not q[j].isspace():
                j += 1
            toks = tokenize(q[i:j])
            i = j
            bucket = {"+": must, "-": must_not, "": should}[prefix]
        bucket.extend(toks)
    return ParsedQuery(should, must, must_not)


class StreamIndex:
    def __init__(self, window_seconds: float = 3600.0, memtable_max_docs: int = 2000,
                 merge_factor: int = 4, max_segments: int = 8, k1: float = 1.2, b: float = 0.75,
                 title_boost: int = 2, clock: Callable[[], float] = time.time) -> None:
        self.window = window_seconds
        self.memtable_max_docs = memtable_max_docs
        self.merge_factor = merge_factor
        self.max_segments = max_segments
        self.k1, self.b, self.title_boost = k1, b, title_boost
        self.clock = clock
        self._lock = threading.RLock()
        self._next_did = 0
        self._next_seg = 0
        self.segments: list[Segment] = []
        self.memtable = self._new_segment()
        self.docs: dict[int, Document] = {}          # live docs only
        self.ext2int: dict[str, int] = {}
        self._expiry_heap: list[tuple[float, int]] = []  # (ts, did)
        self._total_len = 0
        self._doc_len: dict[int, int] = {}
        self.stats = Counter()

    # ------------------------------------------------------------------ writes
    def _new_segment(self) -> Segment:
        s = Segment(self._next_seg)
        self._next_seg += 1
        return s

    def analyze(self, doc: Document) -> Counter:
        toks = tokenize(doc.text)
        if doc.title:
            toks = tokenize(doc.title) * self.title_boost + toks
        return Counter(toks)

    def add(self, doc: Document, tf: Counter | None = None) -> int:
        """Index a document. Searchable the moment this returns."""
        with self._lock:
            if doc.id in self.ext2int:               # upsert = delete + insert
                self.delete(doc.id)
            tf = tf if tf is not None else self.analyze(doc)
            did = self._next_did
            self._next_did += 1
            self.memtable.add(did, tf, doc.ts)
            self.docs[did] = doc
            self.ext2int[doc.id] = did
            dl = sum(tf.values())
            self._doc_len[did] = dl
            self._total_len += dl
            heapq.heappush(self._expiry_heap, (doc.ts, did))
            self.stats["added"] += 1
            if len(self.memtable) >= self.memtable_max_docs:
                self.flush()
            return did

    def delete(self, ext_id: str) -> bool:
        with self._lock:
            did = self.ext2int.pop(ext_id, None)
            if did is None:
                return False
            self._remove(did)
            self.stats["deleted"] += 1
            return True

    def _remove(self, did: int) -> None:
        self.docs.pop(did, None)
        self._total_len -= self._doc_len.pop(did, 0)

    def flush(self) -> None:
        with self._lock:
            if len(self.memtable) == 0:
                return
            self.memtable.frozen = True
            self.segments.append(self.memtable)
            self.memtable = self._new_segment()
            self.stats["flushes"] += 1
            self.maybe_merge()

    # ------------------------------------------------------- maintenance (expiry/merge)
    def expire(self, now: float | None = None) -> int:
        """Drop everything older than the retention window. Returns #docs expired."""
        now = self.clock() if now is None else now
        cutoff = now - self.window
        n = 0
        with self._lock:
            while self._expiry_heap and self._expiry_heap[0][0] < cutoff:
                _, did = heapq.heappop(self._expiry_heap)
                d = self.docs.get(did)
                if d is None:                          # already deleted/replaced
                    continue
                self.ext2int.pop(d.id, None)
                self._remove(did)
                n += 1
            before = len(self.segments)
            self.segments = [s for s in self.segments if s.max_ts >= cutoff]
            self.stats["segments_dropped"] += before - len(self.segments)
            self.stats["expired"] += n
            if n:
                self.maybe_merge()
        return n

    def maybe_merge(self) -> None:
        with self._lock:
            while len(self.segments) > self.max_segments:
                self._merge(sorted(self.segments, key=len)[: self.merge_factor])
            # garbage-collect segments dominated by dead docs
            for s in list(self.segments):
                live = sum(1 for d in s.doc_len if d in self.docs)
                if live == 0:
                    self.segments.remove(s)
                elif live < 0.5 * len(s) and len(self.segments) > 1:
                    self._merge([s])

    def _merge(self, victims: Iterable[Segment]) -> None:
        victims = list(victims)
        out = Segment(self._next_seg, frozen=True)
        self._next_seg += 1
        for seg in victims:
            for term, plist in seg.postings.items():
                dst = out.postings.setdefault(term, {})
                for did, tf in plist.items():
                    if did in self.docs:
                        dst[did] = tf
            for did, ln in seg.doc_len.items():
                if did in self.docs:
                    out.doc_len[did] = ln
                    ts = self.docs[did].ts
                    out.min_ts, out.max_ts = min(out.min_ts, ts), max(out.max_ts, ts)
        out.postings = {t: p for t, p in out.postings.items() if p}
        ids = {id(v) for v in victims}
        self.segments = [s for s in self.segments if id(s) not in ids]
        if len(out):
            self.segments.append(out)
        self.stats["merges"] += 1

    # ------------------------------------------------------------------ stats
    @property
    def num_docs(self) -> int:
        return len(self.docs)

    @property
    def avg_dl(self) -> float:
        return self._total_len / max(1, len(self.docs))

    def df(self, term: str) -> int:
        with self._lock:
            return sum(len(s.postings.get(term, ())) for s in (*self.segments, self.memtable))

    def idf(self, term: str) -> float:
        n, df = max(1, self.num_docs), min(self.df(term), max(1, self.num_docs))
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 10, *, halflife: float | None = None,
               since: float | None = None, now: float | None = None,
               explain: bool = False) -> list[Hit]:
        pq = parse_query(query)
        if pq.empty:
            return []
        scored = self.score(pq, halflife=halflife, since=since, now=now)
        top = heapq.nlargest(k, scored.items(), key=lambda kv: kv[1])
        qset = set(pq.scoring_terms)
        hits = []
        for did, sc in top:
            d = self.docs[did]
            ex = self._explain(did, pq) if explain else {}
            hits.append(Hit(d.id, sc, d.title, make_snippet(d.text, qset), d.ts, d.source, ex))
        return hits

    def score(self, pq: ParsedQuery, *, halflife: float | None = None,
              since: float | None = None, now: float | None = None) -> dict[int, float]:
        """Return {internal_doc_id: score} for all matching live docs."""
        now = self.clock() if now is None else now
        cutoff = max(now - self.window, since if since is not None else -math.inf)
        with self._lock:
            segs = (*self.segments, self.memtable)
            avgdl = max(self.avg_dl, 1e-9)
            acc: dict[int, float] = {}
            must_hits: Counter = Counter()
            for term in pq.scoring_terms:
                idf = self.idf(term)
                is_must = term in pq.must
                for seg in segs:
                    plist = seg.postings.get(term)
                    if not plist:
                        continue
                    for did, tf in plist.items():
                        doc = self.docs.get(did)
                        if doc is None or doc.ts < cutoff:
                            continue
                        dl = seg.doc_len[did]
                        s = idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / avgdl))
                        acc[did] = acc.get(did, 0.0) + s
                        if is_must:
                            must_hits[did] += 1
            if pq.must:
                need = len(set(pq.must))
                acc = {d: s for d, s in acc.items() if must_hits[d] >= need}
            for term in pq.must_not:
                for seg in segs:
                    for did in seg.postings.get(term, ()):
                        acc.pop(did, None)
            if halflife:
                ln2 = math.log(2)
                for did in acc:
                    age = max(0.0, now - self.docs[did].ts)
                    acc[did] *= math.exp(-ln2 * age / halflife)
            return acc

    def _explain(self, did: int, pq: ParsedQuery) -> dict:
        out = {}
        for term in pq.scoring_terms:
            for seg in (*self.segments, self.memtable):
                tf = seg.postings.get(term, {}).get(did)
                if tf:
                    out[term] = {"tf": tf, "idf": round(self.idf(term), 3)}
                    break
        return out

    def get(self, ext_id: str) -> Document | None:
        did = self.ext2int.get(ext_id)
        return self.docs.get(did) if did is not None else None

    def info(self) -> dict:
        return {"live_docs": self.num_docs, "segments": len(self.segments),
                "memtable_docs": len(self.memtable), "avg_dl": round(self.avg_dl, 2),
                "window_seconds": self.window, **dict(self.stats)}
