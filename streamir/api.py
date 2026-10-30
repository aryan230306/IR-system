"""FastAPI service: REST + WebSocket + dashboard."""
from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Union

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .core.retriever import MODES
from .models import Document
from .pipeline import Pipeline
from .sources import JsonlReplaySource, RssSource, SyntheticNewsSource

STATIC = Path(__file__).parent / "static"


def make_source(s: Settings):
    if s.source == "synthetic":
        tag = int(time.time())
        return SyntheticNewsSource(rate=s.rate, realtime=True, seed=tag, id_prefix=f"{tag}-")
    if s.source.startswith("jsonl:"):
        return JsonlReplaySource(s.source[6:], realtime=True)
    if s.source.startswith("rss:"):
        return RssSource(s.source[4:].split(","))
    if s.source in ("none", ""):
        return None
    raise ValueError(f"unknown source {s.source!r}")


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        pipe = Pipeline(settings)
        app.state.pipe = pipe
        await pipe.start()
        src = make_source(settings)
        feeder = asyncio.create_task(pipe.feed(src)) if src else None
        if src and settings.source == "synthetic":                     # demo standing queries
            pipe.percolator.register("earthquake tsunami", 0.5, "Seismic watch")
            pipe.percolator.register("outbreak quarantine", 0.5, "Health alert")
            pipe.percolator.register("ransomware breach +servers", 0.6, "Cyber incident")
        yield
        if feeder:
            feeder.cancel()
        await pipe.stop()

    app = FastAPI(title="StreamIR", version="1.0.0", lifespan=lifespan)

    def P() -> Pipeline:
        return app.state.pipe

    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(STATIC / "index.html")

    # ---------------------------------------------------------------- search
    @app.get("/search")
    async def search(q: str = Query(..., min_length=1), k: int = Query(10, ge=1, le=100),
                     mode: str = Query("hybrid"), halflife: Optional[float] = Query(None, gt=0, description="recency half-life, seconds"),
                     last: Optional[float] = Query(None, gt=0, description="only docs from the last N seconds"),
                     explain: bool = False):
        if mode not in MODES:
            raise HTTPException(400, f"mode must be one of {MODES}")
        t0 = time.perf_counter()
        since = (P().clock() - last) if last else None
        hits = P().retriever.search(q, k, mode, halflife=halflife, since=since, explain=explain)
        return {"query": q, "mode": mode, "took_ms": round((time.perf_counter() - t0) * 1000, 2),
                "total_live_docs": P().index.num_docs, "hits": [h.to_dict() for h in hits]}

    @app.get("/similar")
    async def similar(term: str, n: int = 8):
        from .text import tokenize
        toks = tokenize(term)
        if not toks:
            raise HTTPException(400, "no searchable term")
        return {"term": toks[0], "similar": P().semantic.similar_terms(toks[0], n)}

    # -------------------------------------------------------------- ingestion
    @app.post("/ingest")
    async def ingest(payload: Union[dict, list] = Body(...)):
        items = payload if isinstance(payload, list) else [payload]
        n = 0
        for it in items:
            if not isinstance(it, dict) or not (it.get("text") or it.get("title")):
                raise HTTPException(422, "bad document: need 'text' or 'title'")
            it.setdefault("id", f"api-{time.time_ns()}")
            it.setdefault("ts", time.time())
            try:
                await P().submit(Document.from_dict(it))
            except (KeyError, ValueError, TypeError) as e:
                raise HTTPException(422, f"bad document: {e}")
            n += 1
        return {"accepted": n, "queue_depth": P().queue.qsize()}

    # ---------------------------------------------------------- standing queries
    @app.post("/queries")
    async def add_query(q: str = Body(..., embed=True), min_match: float = Body(0.6, embed=True),
                        name: str = Body("", embed=True)):
        try:
            return P().percolator.register(q, min_match, name, idf=P().index.idf).to_dict()
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/queries")
    async def list_queries():
        return [q.to_dict() for q in P().percolator.list()]

    @app.delete("/queries/{qid}")
    async def del_query(qid: int):
        if not P().percolator.remove(qid):
            raise HTTPException(404, "no such query")
        return {"deleted": qid}

    # ------------------------------------------------------------- analytics
    @app.get("/trends")
    async def trends(k: int = 10):
        return {"bursting_terms": P().trending(k)}

    @app.get("/clusters")
    async def clusters(k: int = 10, min_size: int = 2):
        return {"clusters": P().clusterer.top(k, min_size)}

    @app.get("/stats")
    async def stats():
        return P().stats()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ------------------------------------------------------------- streaming out
    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        q = P().subscribe()
        try:
            await sock.send_json({"type": "hello", "latest": list(P().latest)[:20]})
            next_stats = 0.0
            while True:
                try:
                    ev = await asyncio.wait_for(q.get(), timeout=1.0)
                    await sock.send_json(ev)
                except asyncio.TimeoutError:
                    pass
                if time.monotonic() >= next_stats:       # periodic stats even under constant event load
                    next_stats = time.monotonic() + 1.0
                    await sock.send_json({"type": "stats", **P().stats(),
                                          "trends": P().trending(8), "clusters": P().clusterer.top(6)})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            P().unsubscribe(q)

    @app.get("/events")
    async def sse(types: str = "alert,burst"):
        wanted = set(types.split(","))
        q = P().subscribe()

        async def gen():
            try:
                while True:
                    ev = await q.get()
                    if ev["type"] in wanted:
                        yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
            finally:
                P().unsubscribe(q)
        return StreamingResponse(gen(), media_type="text/event-stream")

    return app


app = create_app() if os.environ.get("STREAMIR_AUTOCREATE") else None
