"""The real-time ingestion pipeline that wires every component together.

    source -> [bounded queue: back-pressure] -> worker:
        analyze -> near-dup filter -> NRT index -> semantic learn/encode -> burst stats
                -> story clustering -> standing-query matching -> WAL -> pub/sub events
    maintenance loop (1 Hz): expire window, merge segments, burst tick, WAL compaction
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter, deque
from typing import Any, AsyncIterator, Callable

from .config import Settings
from .core.burst import BurstDetector
from .core.cluster import OnlineClusterer
from .core.dedup import NearDupDetector
from .core.index import StreamIndex
from .core.percolator import Percolator
from .core.retriever import Retriever
from .core.semantic import SemanticIndex
from .models import Document
from .text import tokenize


class WriteAheadLog:
    """Append-only JSONL log: crash recovery without an external database."""

    def __init__(self, path: str, flush_every: int = 100) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self._n = 0
        self._flush_every = flush_every
        self.appended = 0

    def append(self, doc: Document) -> None:
        self._fh.write(json.dumps(doc.to_dict(), separators=(",", ":")) + "\n")
        self.appended += 1
        self._n += 1
        if self._n >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        self._fh.flush()
        self._n = 0

    @staticmethod
    def read(path: str, min_ts: float):
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    d = Document.from_dict(json.loads(line))
                except Exception:           # torn last write after a crash -> skip
                    continue
                if d.ts >= min_ts:
                    yield d

    def compact(self, live_docs) -> None:
        """Atomically rewrite the log with live documents only."""
        self._fh.close()
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for d in sorted(live_docs, key=lambda d: d.ts):
                fh.write(json.dumps(d.to_dict(), separators=(",", ":")) + "\n")
        os.replace(tmp, self.path)
        self._fh = open(self.path, "a", encoding="utf-8")

    def close(self) -> None:
        self._fh.flush()
        self._fh.close()


class Metrics:
    def __init__(self) -> None:
        self.counters: Counter = Counter()
        self.lat = deque(maxlen=5000)            # per-doc processing latency (ms)
        self._arrivals = deque(maxlen=2000)      # wall-clock arrival times
        self.started = time.time()

    def record(self, latency_ms: float) -> None:
        self.lat.append(latency_ms)
        self._arrivals.append(time.time())

    def throughput(self, horizon: float = 5.0) -> float:
        now = time.time()
        n = sum(1 for t in self._arrivals if now - t <= horizon)
        return n / horizon

    def percentiles(self) -> dict[str, float]:
        if not self.lat:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        s = sorted(self.lat)
        pick = lambda q: round(s[min(len(s) - 1, int(q * len(s)))], 3)
        return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


class Pipeline:
    def __init__(self, settings: Settings | None = None, clock: Callable[[], float] = time.time) -> None:
        self.s = settings or Settings()
        self.clock = clock
        self.index = StreamIndex(window_seconds=self.s.window_seconds, memtable_max_docs=self.s.memtable_max_docs,
                                 clock=clock)
        self.semantic = SemanticIndex(dim=self.s.semantic_dim)
        self.retriever = Retriever(self.index, self.semantic)
        self.dedup = NearDupDetector(threshold=self.s.dedup_threshold)
        self.burst = BurstDetector(bucket_seconds=self.s.burst_bucket, z_threshold=self.s.burst_z)
        self.clusterer = OnlineClusterer(threshold=self.s.cluster_threshold, ttl=self.s.window_seconds)
        self.percolator = Percolator()
        self.metrics = Metrics()
        self.wal = WriteAheadLog(self.s.wal_path) if self.s.wal_path else None
        self.subscribers: set[asyncio.Queue] = set()
        self.queue: asyncio.Queue | None = None
        self._tasks: list[asyncio.Task] = []
        self._announced_bursts: dict[str, float] = {}
        self.recent_bursts: dict[str, dict] = {}          # term -> last detection (kept for burst_ttl)
        self.burst_ttl = 90.0
        self.latest: deque[dict] = deque(maxlen=100)

    # --------------------------------------------------------------- processing
    def process(self, doc: Document, *, replay: bool = False) -> dict[str, Any]:
        """Synchronous hot path. Returns a small status dict."""
        t0 = time.perf_counter()
        tokens = tokenize(doc.title + " . " + doc.text) if doc.title else tokenize(doc.text)
        if not tokens:
            self.metrics.counters["empty"] += 1
            return {"status": "empty"}
        _, dup_of = self.dedup.check_and_add(tokens, doc.id)
        if dup_of is not None:
            self.metrics.counters["duplicates"] += 1
            return {"status": "duplicate", "of": dup_of}

        tf = self.index.analyze(doc)
        did = self.index.add(doc, tf)
        self.semantic.learn(tokens)
        self.semantic.add(did, self.semantic.encode(tf, self.index.idf))
        self.burst.observe(tf.keys(), doc.ts)
        cid, sim = self.clusterer.assign(doc.id, doc.title or doc.text[:80], doc.source, tf, self.index.idf, doc.ts)

        alerts = []
        if not replay:
            for q, score in self.percolator.match(set(tf), self.index.idf):
                q.matches += 1
                a = {"query_id": q.qid, "query": q.text, "name": q.name, "score": round(score, 3),
                     "doc": {"id": doc.id, "title": doc.title, "source": doc.source, "ts": doc.ts}}
                q.recent.appendleft(a["doc"])
                alerts.append(a)
            if self.wal:
                self.wal.append(doc)
            self.metrics.record((time.perf_counter() - t0) * 1000)
        self.metrics.counters["indexed"] += 1

        if not replay:
            ev = {"type": "doc", "id": doc.id, "title": doc.title, "source": doc.source, "ts": doc.ts,
                  "cluster": cid, "topic": doc.meta.get("topic")}
            self.latest.appendleft(ev)
            self._publish(ev)
            for a in alerts:
                self._publish({"type": "alert", **a})
        return {"status": "indexed", "internal_id": did, "cluster": cid, "alerts": len(alerts)}

    # ----------------------------------------------------------------- pub/sub
    def subscribe(self, maxsize: int = 1000) -> asyncio.Queue:
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self.subscribers.discard(q)

    def _publish(self, event: dict) -> None:
        for q in list(self.subscribers):
            if q.full():                      # slow consumer: drop oldest, never block the pipeline
                try:
                    q.get_nowait()
                    self.metrics.counters["events_dropped"] += 1
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(event)

    # --------------------------------------------------------------- async glue
    async def start(self) -> None:
        self.queue = asyncio.Queue(maxsize=self.s.queue_size)
        self.recover()
        self._tasks = [asyncio.create_task(self._worker(), name="worker"),
                       asyncio.create_task(self._maintenance(), name="maintenance")]

    async def stop(self) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.wal:
            self.wal.close()

    async def submit(self, doc: Document) -> None:
        """Enqueue a document. Awaits (back-pressure) when the queue is full."""
        assert self.queue is not None, "call start() first"
        await self.queue.put(doc)

    async def feed(self, source) -> None:
        async for doc in source.stream():
            await self.submit(doc)

    async def drain(self) -> None:
        assert self.queue is not None
        await self.queue.join()

    async def _worker(self) -> None:
        assert self.queue is not None
        while True:
            doc = await self.queue.get()
            try:
                self.process(doc)
            except Exception as e:             # one bad document must never kill the pipeline
                self.metrics.counters["errors"] += 1
                self.metrics.counters["last_error"] = 0
                self.last_error = repr(e)
            finally:
                self.queue.task_done()
            if self.queue.qsize() == 0 or self.metrics.counters["indexed"] % 50 == 0:
                await asyncio.sleep(0)         # yield so API requests stay responsive

    async def _maintenance(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            self.maintain()

    def maintain(self, now: float | None = None) -> None:
        now = self.clock() if now is None else now
        self.index.expire(now)
        self.burst.tick(now)
        if len(self.semantic._ids) > 1.5 * self.index.num_docs + 200:
            self.semantic.compact(lambda d: d in self.index.docs)
        if self.wal:
            self.wal.flush()
            if self.wal.appended > 5 * max(1000, self.index.num_docs):   # log grew far beyond live set
                self.wal.compact(self.index.docs.values())
                self.wal.appended = 0
        for b in self.burst.top(10):
            prev = self.recent_bursts.get(b["term"])
            self.recent_bursts[b["term"]] = {**b, "seen": now, "z": max(b["z"], prev["z"]) if prev and now - prev["seen"] < self.burst_ttl else b["z"]}
            last = self._announced_bursts.get(b["term"], 0)
            if now - last > 120:
                self._announced_bursts[b["term"]] = now
                self._publish({"type": "burst", **b})
        self._announced_bursts = {t: x for t, x in self._announced_bursts.items() if now - x < 600}
        self.recent_bursts = {t: b for t, b in self.recent_bursts.items() if now - b["seen"] < self.burst_ttl}

    def trending(self, k: int = 10) -> list[dict]:
        """Terms that burst within the last `burst_ttl` seconds (survives bucket roll-over)."""
        out = sorted(self.recent_bursts.values(), key=lambda b: -b["z"])[:k]
        return [{"term": b["term"], "count": b["count"], "baseline": b["baseline"], "z": b["z"],
                 "age_s": round(self.clock() - b["seen"], 1)} for b in out]

    def recover(self) -> int:
        """Rebuild all in-memory state from the WAL (docs inside the retention window only)."""
        if not self.wal:
            return 0
        n = 0
        now = self.clock()
        docs = list(WriteAheadLog.read(self.wal.path, now - self.s.window_seconds))
        for d in sorted(docs, key=lambda d: d.ts):
            if self.process(d, replay=True)["status"] == "indexed":
                n += 1
        self.wal.compact(self.index.docs.values())
        self.metrics.counters["recovered"] = n
        return n

    # ------------------------------------------------------------------ reporting
    def stats(self) -> dict:
        c = self.metrics.counters
        return {"uptime_s": round(time.time() - self.metrics.started, 1),
                "indexed": c["indexed"], "duplicates_dropped": c["duplicates"], "errors": c["errors"],
                "recovered_from_wal": c["recovered"], "events_dropped": c["events_dropped"],
                "throughput_docs_per_s": round(self.metrics.throughput(), 1),
                "queue_depth": self.queue.qsize() if self.queue else 0,
                "latency_ms": self.metrics.percentiles(),
                "index": self.index.info(), "semantic_vectors": len(self.semantic),
                "semantic_vocab": len(self.semantic._ctx), "active_clusters": len(self.clusterer.clusters),
                "standing_queries": len(self.percolator.list()), "subscribers": len(self.subscribers)}
