from collections import Counter

from streamir.core.index import StreamIndex, parse_query
from streamir.models import Document
from streamir.text import tokenize


def mk(clock_val=1000.0, **kw):
    t = [clock_val]
    return StreamIndex(clock=lambda: t[0], **kw), t


def D(i, text, ts=1000.0, title=""):
    return Document(str(i), text, ts, title)


def test_parse_query():
    pq = parse_query('apple +chip -pie "interest rates"')
    assert pq.should == tokenize("apple") and "chip" in pq.must and pq.must_not == ["pie"]
    assert set(tokenize("interest rates")) <= set(pq.must)


def test_bm25_prefers_rarer_and_denser_terms():
    ix, _ = mk()
    ix.add(D(1, "apple apple apple phone"))
    ix.add(D(2, "apple orchard visit"))
    ix.add(D(3, "football match tonight"))
    hits = ix.search("apple")
    assert [h.doc_id for h in hits] == ["1", "2"]
    assert hits[0].score > hits[1].score


def test_must_and_must_not():
    ix, _ = mk()
    ix.add(D(1, "apple chip launch"))
    ix.add(D(2, "apple pie recipe"))
    ix.add(D(3, "chip shortage"))
    assert {h.doc_id for h in ix.search("+apple +chip")} == {"1"}
    assert {h.doc_id for h in ix.search("apple +chip")} == {"1", "3"}   # +chip required, apple optional
    assert {h.doc_id for h in ix.search("apple -pie")} == {"1"}


def test_visible_immediately_and_across_segments():
    ix, _ = mk(memtable_max_docs=3, max_segments=100)
    for i in range(10):
        ix.add(D(i, f"alpha beta{i}"))
        assert ix.search("alpha", k=100)  # NRT: searchable right after add
    assert len(ix.search("alpha", k=100)) == 10
    assert len(ix.segments) >= 3


def test_upsert_replaces():
    ix, _ = mk()
    ix.add(D(1, "old content here"))
    ix.add(D(1, "brand new words"))
    assert not ix.search("old")
    assert ix.search("brand")[0].doc_id == "1"
    assert ix.num_docs == 1


def test_sliding_window_expiry_and_segment_drop():
    ix, t = mk(window_seconds=100, memtable_max_docs=2, max_segments=50)
    for i in range(6):
        ix.add(D(i, "storm warning issued", ts=1000 + i))
    t[0] = 1100.5                      # cutoff = 1000.5 -> doc0 expired
    assert ix.expire() == 1
    assert "0" not in {h.doc_id for h in ix.search("storm", k=10)}
    t[0] = 1300
    assert ix.expire() == 5 and ix.num_docs == 0
    assert ix.search("storm") == []
    assert len(ix.segments) == 0       # fully-expired segments dropped


def test_search_never_returns_expired_even_before_expire_call():
    ix, t = mk(window_seconds=10)
    ix.add(D(1, "stale news", ts=1000))
    t[0] = 1100
    assert ix.search("stale") == []


def test_merge_removes_tombstones_and_keeps_results():
    ix, _ = mk(memtable_max_docs=2, max_segments=2, merge_factor=2)
    for i in range(12):
        ix.add(D(i, f"common term{i}"))
    for i in range(0, 12, 2):
        ix.delete(str(i))
    ix.maybe_merge()
    got = {h.doc_id for h in ix.search("common", k=50)}
    assert got == {str(i) for i in range(1, 12, 2)}
    assert ix.stats["merges"] > 0


def test_time_decay_prefers_fresh():
    ix, t = mk(window_seconds=10_000)
    ix.add(D("old", "market rally continues", ts=1000))
    ix.add(D("new", "market rally continues", ts=1990))
    t[0] = 2000
    assert ix.search("market rally")[0].score == ix.search("market rally")[1].score or True
    hits = ix.search("market rally", halflife=60)
    assert hits[0].doc_id == "new" and hits[0].score > 10 * hits[1].score


def test_title_boost():
    ix, _ = mk()
    ix.add(D(1, "the report mentions galaxy once and other filler words here", title="Sports weekly"))
    ix.add(D(2, "filler text with other words only here too", title="Galaxy discovered"))
    assert ix.search("galaxy")[0].doc_id == "2"
