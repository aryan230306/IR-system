"""End-to-end evaluation on a labelled synthetic news stream (virtual clock => deterministic).

Run:  python -m streamir bench --n 20000
"""
from __future__ import annotations

import math
import os
import random
import tempfile
import time
from collections import Counter, defaultdict

from streamir.config import Settings
from streamir.core.index import parse_query
from streamir.core.percolator import Percolator
from streamir.pipeline import Pipeline
from streamir.sources import EVENTS, TOPICS, SyntheticNewsSource
from streamir.text import tokenize


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0.0


def ndcg_at(ranked, rel, k=10):
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / ideal if ideal else 0.0


def run(n_docs: int = 20000, window: float = 200.0, seed: int = 11) -> str:
    out: list[str] = []
    P = out.append
    t = [0.0]
    s = Settings()
    s.wal_path, s.window_seconds = None, window
    pipe = Pipeline(s, clock=lambda: t[0])
    src = SyntheticNewsSource(rate=50, seed=seed, n_docs=n_docs, realtime=False, start_ts=1_000_000.0,
                              event_every=40, event_len=25)
    docs = list(src.generate())
    by_id = {d.id: d for d in docs}

    # ------------------------------------------------ ingest + online analytics
    status = Counter()
    dup_tp = dup_fp = dup_fn = 0
    burst_log: list[tuple[float, str, float]] = []   # (virtual ts, term, z)
    last_sec = -1
    t0 = time.perf_counter()
    for d in docs:
        t[0] = d.ts
        r = pipe.process(d)
        status[r["status"]] += 1
        is_dup, flagged = "dup_of" in d.meta, r["status"] == "duplicate"
        dup_tp += is_dup and flagged
        dup_fp += (not is_dup) and flagged
        dup_fn += is_dup and not flagged
        if int(d.ts) != last_sec:
            last_sec = int(d.ts)
            for b in pipe.burst.top(10):
                burst_log.append((d.ts, b["term"], b["z"]))
            pipe.maintain(d.ts)
    wall = time.perf_counter() - t0
    lat = list(pipe.metrics.lat)
    P("## StreamIR benchmark\n")
    P(f"Synthetic labelled stream: {n_docs:,} docs, {docs[-1].ts - docs[0].ts:,.0f}s of event time, "
      f"retention window {window:.0f}s, single Python process (no GPU).\n")
    P("### 1. Ingestion (full pipeline per document)\n")
    P("| metric | value |\n|---|---|")
    P(f"| sustained throughput | **{n_docs / wall:,.0f} docs/s** |")
    P(f"| per-doc latency p50 / p95 / p99 | {pct(lat, .5):.2f} / {pct(lat, .95):.2f} / {pct(lat, .99):.2f} ms |")
    P(f"| indexed / duplicates dropped | {status['indexed']:,} / {status['duplicate']:,} |")
    P(f"| live docs at end (windowed) | {pipe.index.num_docs:,} in {len(pipe.index.segments)} segments |")
    P(f"| expired by sliding window | {pipe.index.stats['expired']:,} |")
    P(f"| segment merges | {pipe.index.stats['merges']} |\n")
    P("Pipeline per doc = tokenise, MinHash-LSH dedup, NRT index, semantic learn+encode, burst stats, "
      "online clustering, standing-query match.\n")

    # ------------------------------------------------ dedup
    prec = dup_tp / max(1, dup_tp + dup_fp)
    rec = dup_tp / max(1, dup_tp + dup_fn)
    P("### 2. Near-duplicate detection\n")
    P(f"Precision **{prec:.3f}**, recall **{rec:.3f}** (TP={dup_tp}, FP={dup_fp}, FN={dup_fn}) "
      f"against injected syndicated copies.\n")

    # ------------------------------------------------ retrieval quality
    now = docs[-1].ts
    rel_event: dict = defaultdict(set)
    rel_topic: dict = defaultdict(set)
    for d in pipe.index.docs.values():
        if "event" in d.meta:
            rel_event[d.meta["event"]].add(d.id)
        rel_topic[d.meta["topic"]].add(d.id)
    rng = random.Random(3)
    queries: list[tuple[str, str, set]] = []
    for topic, ename, words, place in EVENTS:
        if len(rel_event[ename]) >= 10:
            queries.append(("event", " ".join(rng.sample(words, 2)), rel_event[ename]))
    for topic, voc in TOPICS.items():
        pool = sum((c.split() for c in voc["ctx"]), [])
        for _ in range(8):
            queries.append(("topic", " ".join(rng.sample(pool, 2)), rel_topic[topic]))
    P("### 3. Retrieval effectiveness (relevance = ground-truth topic/event label; event docs count as relevant to their topic)\n")
    P(f"{len(queries)} queries over the live window ({pipe.index.num_docs:,} docs).\n")
    P("| mode | nDCG@10 | P@10 | MRR | p50 latency | p99 latency |\n|---|---|---|---|---|---|")
    for mode in ("bm25", "dense", "hybrid"):
        nd, p10, rr, ts = [], [], [], []
        for _, q, rel in queries:
            a = time.perf_counter()
            hits = pipe.retriever.search(q, 10, mode, now=now)
            ts.append((time.perf_counter() - a) * 1000)
            ids = [h.doc_id for h in hits]
            nd.append(ndcg_at(ids, rel))
            p10.append(sum(i in rel for i in ids) / 10)
            rr.append(next((1 / (i + 1) for i, x in enumerate(ids) if x in rel), 0.0))
        P(f"| {mode} | {sum(nd)/len(nd):.3f} | {sum(p10)/len(p10):.3f} | {sum(rr)/len(rr):.3f} | "
          f"{pct(ts,.5):.2f} ms | {pct(ts,.99):.2f} ms |")
    P("")

    # vocabulary-mismatch probe: dense results sharing no term with the query
    mm_hit = mm_tot = 0
    for _, q, rel in queries:
        qt = set(parse_query(q).scoring_terms)
        for h in pipe.retriever.search(q, 200, "dense", now=now):
            doc_terms = set(pipe.index.analyze(pipe.index.get(h.doc_id)))
            if not (qt & doc_terms):
                mm_tot += 1
                mm_hit += h.doc_id in rel
    P("**Vocabulary-mismatch probe.** Dense results that share *no* term with the query "
      "(unreachable for BM25 by construction): "
      f"{mm_tot} docs returned, precision against the topic label = **{mm_hit / max(1, mm_tot):.3f}** "
      f"(random-guess baseline ~{1/len(TOPICS):.3f}).\n")

    ages = {}
    for hl in (None, 30.0):
        a = []
        for _, q, _r in queries:
            a += [now - h.ts for h in pipe.retriever.search(q, 10, "bm25", halflife=hl, now=now)]
        ages[hl] = sum(a) / max(1, len(a))
    P(f"**Freshness.** Mean age of top-10 results: {ages[None]:.1f}s without decay vs "
      f"{ages[30.0]:.1f}s with a 30s half-life.\n")

    # ------------------------------------------------ burst detection
    P("### 4. Burst (breaking-news) detection\n")
    occ: list[dict] = []
    live_occ: dict[str, dict] = {}                     # event name -> its current occurrence
    for d in (d for d in docs if "event" in d.meta):
        name = d.meta["event"]
        cur = live_occ.get(name)
        toks = set(tokenize(d.title + " " + d.text))
        if cur and d.ts - cur["end"] < 25:
            cur["end"] = d.ts
            cur["vocab"] |= toks
        else:
            cur = {"name": name, "start": d.ts, "end": d.ts, "vocab": set(toks)}
            live_occ[name] = cur
            occ.append(cur)
    delays, missed = [], 0
    for o in occ:
        det = [ts for ts, term, z in burst_log if o["start"] <= ts <= o["end"] + 20 and term in o["vocab"]]
        if det:
            delays.append(min(det) - o["start"])
        else:
            missed += 1
    first_seen: dict = {}
    for ts, term, z in burst_log:
        first_seen.setdefault((term, int(ts // 120)), ts)
    fa_terms = [term for (term, _), ts in first_seen.items()
                if not any(o["start"] - 5 <= ts <= o["end"] + 25 and term in o["vocab"] for o in occ)]
    false_alarms = len(fa_terms)
    print('false-alarm terms:', fa_terms)
    P(f"{len(occ)} injected breaking events: **{len(occ) - missed} detected**, {missed} missed; "
      f"detection delay median **{pct(delays,.5):.1f}s** / max {max(delays, default=0):.1f}s of event time "
      f"(bucket = {pipe.burst.bucket:.0f}s). Distinct false-alarm term bursts: **{false_alarms}**.\n")

    # ------------------------------------------------ clustering
    P("### 5. Online story clustering\n")
    pur_num = pur_den = 0
    big = [c for c in pipe.clusterer.clusters.values() if c.size >= 5]
    for c in big:
        labs = Counter((by_id[i].meta.get("event") or by_id[i].meta["topic"]) for i in c.doc_ids if i in by_id)
        if labs:
            pur_num += labs.most_common(1)[0][1]
            pur_den += sum(labs.values())
    P(f"{len(pipe.clusterer.clusters)} active clusters, {len(big)} with >=5 docs, "
      f"purity (label of first 50 docs) **{pur_num / max(1, pur_den):.3f}**.\n")

    # ------------------------------------------------ standing queries
    P("### 6. Standing-query (percolator) scaling\n")
    P("| registered queries | match time / doc | docs/s |\n|---|---|---|")
    vocab = list(pipe.index.segments[0].postings)[:3000] if pipe.index.segments else list(pipe.index.memtable.postings)
    sample_terms = [set(pipe.index.analyze(d)) for d in docs[-300:]]
    for nq in (100, 1000, 10000):
        perc = Percolator()
        r2 = random.Random(5)
        for _ in range(nq):
            perc.register(" ".join(r2.sample(vocab, 3)), 0.66, idf=pipe.index.idf)
        a = time.perf_counter()
        for st in sample_terms:
            perc.match(st, pipe.index.idf)
        dt = (time.perf_counter() - a) / len(sample_terms)
        P(f"| {nq:,} | {dt*1000:.3f} ms | {1/dt:,.0f} |")
    P("")

    # ------------------------------------------------ recovery
    P("### 7. Crash recovery (write-ahead log)\n")
    with tempfile.TemporaryDirectory() as td:
        s2 = Settings()
        s2.window_seconds = 1e9
        s2.wal_path = os.path.join(td, "wal.jsonl")
        t2 = [docs[-1].ts]
        p1 = Pipeline(s2, clock=lambda: t2[0])
        for d in docs[:5000]:
            p1.process(d)
        p1.wal.flush()
        n1 = p1.index.num_docs
        p1.wal.close()
        with open(s2.wal_path, "a") as fh:
            fh.write('{"id": "torn", "tex')           # simulate a torn write at crash time
        a = time.perf_counter()
        p2 = Pipeline(s2, clock=lambda: t2[0])
        n2 = p2.recover()
        rt = time.perf_counter() - a
        r1 = [h.doc_id for h in p1.retriever.search("climate", 5, "bm25", now=t2[0])]
        r2_ = [h.doc_id for h in p2.retriever.search("climate", 5, "bm25", now=t2[0])]
        P(f"Indexed {n1:,} docs, simulated crash with a torn final log record, rebuilt **{n2:,} docs in {rt:.2f}s**; "
          f"identical top-5 ranking after recovery: **{bool(r1) and r1 == r2_}**.\n")

    text = "\n".join(out)
    print(text)
    with open(os.path.join(os.path.dirname(__file__), "RESULTS.md"), "w") as fh:
        fh.write(text + "\n")
    return text


if __name__ == "__main__":
    run()
