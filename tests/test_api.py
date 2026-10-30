import time

from fastapi.testclient import TestClient

from streamir.api import create_app
from streamir.config import Settings


def make():
    s = Settings()
    s.source, s.wal_path = "none", None
    return create_app(s)


def wait_indexed(c, n, timeout=5):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = c.get("/stats").json()
        if st["indexed"] + st["duplicates_dropped"] >= n:
            return st
        time.sleep(0.02)
    raise AssertionError("not indexed in time")


def test_ingest_search_and_validation():
    with TestClient(make()) as c:
        r = c.post("/ingest", json=[
            {"id": "a", "title": "Chip maker unveils processor", "text": "A new AI processor was unveiled today"},
            {"id": "b", "title": "Floods in Brazil", "text": "Heavy rain caused floods across the region"}])
        assert r.json()["accepted"] == 2
        wait_indexed(c, 2)
        j = c.get("/search", params={"q": "processor", "mode": "bm25", "explain": True}).json()
        assert j["hits"][0]["id"] == "a" and "processor" in j["hits"][0]["explain"]
        assert c.get("/search", params={"q": "flood", "mode": "hybrid"}).json()["hits"][0]["id"] == "b"
        assert c.get("/search", params={"q": "x", "mode": "nope"}).status_code == 400
        assert c.post("/ingest", json={"nothing": 1}).status_code == 422
        assert c.get("/health").json() == {"status": "ok"}


def test_standing_query_alerts_over_websocket():
    with TestClient(make()) as c:
        qid = c.post("/queries", json={"q": "earthquake tsunami", "min_match": 0.5, "name": "quake"}).json()["id"]
        assert c.post("/queries", json={"q": "the a an"}).status_code == 422
        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
            c.post("/ingest", json={"id": "q1", "title": "Earthquake strikes", "text": "A tsunami warning follows the earthquake"})
            c.post("/ingest", json={"id": "q2", "title": "Cooking tips", "text": "How to bake bread at home"})
            got = []
            for _ in range(10):
                m = ws.receive_json()
                got.append(m["type"])
                if m["type"] == "alert":
                    assert m["query_id"] == qid and m["doc"]["id"] == "q1"
                    break
            assert "alert" in got
        assert c.get("/queries").json()[0]["matches"] == 1
        assert c.delete(f"/queries/{qid}").status_code == 200
        assert c.delete(f"/queries/{qid}").status_code == 404


def test_recency_halflife_and_stats():
    with TestClient(make()) as c:
        now = time.time()
        c.post("/ingest", json=[
            {"id": "old", "title": "Market rally", "text": "markets rally on earnings news", "ts": now - 900},
            {"id": "new", "title": "Market rally continues", "text": "markets rally again on earnings news", "ts": now}])
        wait_indexed(c, 2)
        hits = c.get("/search", params={"q": "market rally", "mode": "bm25", "halflife": 60}).json()["hits"]
        assert hits[0]["id"] == "new"
        only_recent = c.get("/search", params={"q": "market rally", "mode": "bm25", "last": 60}).json()["hits"]
        assert [h["id"] for h in only_recent] == ["new"]
        assert c.get("/trends").status_code == 200 and c.get("/clusters").status_code == 200
