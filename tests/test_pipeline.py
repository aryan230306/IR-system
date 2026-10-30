import asyncio

import pytest

from streamir.config import Settings
from streamir.models import Document
from streamir.pipeline import Pipeline, WriteAheadLog
from streamir.sources import SyntheticNewsSource


def settings(tmp_path=None, **kw):
    s = Settings()
    s.wal_path = str(tmp_path / "wal.jsonl") if tmp_path else None
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_end_to_end_process_alerts_and_dedup():
    t = [1000.0]
    p = Pipeline(settings(window_seconds=600), clock=lambda: t[0])
    p.percolator.register("earthquake tsunami", 0.5, idf=p.index.idf)
    r1 = p.process(Document("1", "Earthquake and tsunami hit the coast of Chile this morning", 1000.0, "Quake"))
    assert r1["status"] == "indexed" and r1["alerts"] == 1
    r2 = p.process(Document("2", "Earthquake and tsunami hit the coast of Chile this morning", 1001.0, "Quake"))
    assert r2 == {"status": "duplicate", "of": "1"}
    assert p.retriever.search("tsunami chile", 5, "bm25")[0].doc_id == "1"
    assert p.retriever.search("tsunami chile", 5, "hybrid")[0].doc_id == "1"
    assert p.stats()["duplicates_dropped"] == 1


def test_bad_docs_are_skipped_gracefully():
    p = Pipeline(settings())
    assert p.process(Document("x", "the a an", 1.0))["status"] == "empty"


def test_wal_recovery_with_torn_tail(tmp_path):
    t = [2000.0]
    s = settings(tmp_path, window_seconds=1e9)
    p1 = Pipeline(s, clock=lambda: t[0])
    for d in SyntheticNewsSource(rate=50, n_docs=300, realtime=False, start_ts=1000.0).generate():
        p1.process(d)
    p1.wal.flush(); n = p1.index.num_docs; p1.wal.close()
    with open(s.wal_path, "a") as fh:
        fh.write('{"id":"torn","te')
    p2 = Pipeline(s, clock=lambda: t[0])
    assert p2.recover() == n and p2.index.num_docs == n


def test_wal_recovery_respects_window(tmp_path):
    s = settings(tmp_path, window_seconds=100)
    w = WriteAheadLog(s.wal_path)
    w.append(Document("old", "ancient history lesson", 100.0))
    w.append(Document("new", "fresh breaking headline", 950.0))
    w.close()
    p = Pipeline(s, clock=lambda: 1000.0)
    assert p.recover() == 1 and p.index.get("old") is None and p.index.get("new")


def test_async_backpressure_and_drain():
    async def go():
        s = settings(queue_size=10)
        p = Pipeline(s)
        await p.start()
        src = SyntheticNewsSource(rate=1000, n_docs=300, realtime=False, start_ts=__import__("time").time())
        await p.feed(src)                   # feed() awaits when queue is full => no unbounded growth
        await p.drain()
        n = p.stats()["indexed"] + p.stats()["duplicates_dropped"]
        await p.stop()
        return n
    assert asyncio.run(go()) == 300


def test_pubsub_slow_consumer_does_not_block():
    p = Pipeline(settings())
    q = p.subscribe(maxsize=5)
    for i in range(50):
        p._publish({"type": "doc", "i": i})
    assert q.qsize() == 5 and p.metrics.counters["events_dropped"] == 45
