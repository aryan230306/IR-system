# CV bullets (pick 3–4; keep the numbers you can defend)

**StreamIR — Real-time stream information retrieval engine** · Python, asyncio, FastAPI, NumPy · [github link]
- Built a near-real-time search engine from scratch (LSM-style memtable/segments, BM25, sliding event-time retention) that makes documents searchable at ingest; **~650 docs/s end-to-end** and **~1 ms p50 query latency** on ~9.5k live docs in one Python process.
- Designed a streaming analytics pipeline: MinHash-LSH near-duplicate filtering (P 0.95 / R 1.00), EWMA burst detection (7/8 injected events, ~1 s median delay), single-pass story clustering, and write-ahead-log crash recovery tolerant of torn writes.
- Implemented a standing-query percolator with provably lossless posting pruning, giving **~20× faster matching at 10k queries** (0.37 ms/doc) verified against a brute-force oracle.
- Added hybrid retrieval (BM25 + online Random-Indexing vectors fused via RRF, recency decay) that retrieves topically relevant documents sharing no query terms (96% precision vs 12.5% chance on synthetic stream); shipped REST/WebSocket/SSE API, live dashboard and 30-test suite.

(Do **not** claim it beats Elasticsearch/BM25 on real data; it has only been evaluated on a synthetic stream.)

# College report outline
1. Abstract & motivation (static vs streaming IR)  2. Related work (Lucene NRT, Twitter Earlybird, TDT, Kleinberg bursts, Random Indexing, MinHash/LSH, RRF)
3. System architecture (diagram from README)  4. Algorithms per component (with complexity)  5. Implementation  6. Evaluation methodology + results (bench/RESULTS.md)
7. Threats to validity (synthetic data, GIL, stemmer — see README limitations)  8. Future work (real dataset eval, distributed partitioning, learned reranker, transformer embeddings)

# Likely viva questions
- What happens when the memtable is full? (freeze → segment, maybe merge) · How are deletions/expiry handled? (tombstone via live-doc map, physical removal at merge, whole-segment drop)
- Why BM25 parameters k1=1.2, b=0.75? · What does the half-life do mathematically? (score × 2^(−age/h))
- How do you know the percolator pruning is correct? (pigeonhole; brute-force equivalence test)
- What are the weaknesses of your evaluation? (synthetic, saturated BM25 — be upfront)
- How would you scale to 100× throughput? (partition by doc id, scatter-gather query, Kafka as the log instead of a local WAL)
