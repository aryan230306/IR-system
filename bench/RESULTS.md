## StreamIR benchmark

Synthetic labelled stream: 1,000 docs, 20s of event time, retention window 200s, single Python process (no GPU).

### 1. Ingestion (full pipeline per document)

| metric | value |
|---|---|
| sustained throughput | **1,595 docs/s** |
| per-doc latency p50 / p95 / p99 | 0.62 / 0.93 / 1.26 ms |
| indexed / duplicates dropped | 928 / 72 |
| live docs at end (windowed) | 928 in 0 segments |
| expired by sliding window | 0 |
| segment merges | 0 |

Pipeline per doc = tokenise, MinHash-LSH dedup, NRT index, semantic learn+encode, burst stats, online clustering, standing-query match.

### 2. Near-duplicate detection

Precision **1.000**, recall **1.000** (TP=72, FP=0, FN=0) against injected syndicated copies.

### 3. Retrieval effectiveness (relevance = ground-truth topic/event label; event docs count as relevant to their topic)

64 queries over the live window (928 docs).

| mode | nDCG@10 | P@10 | MRR | p50 latency | p99 latency |
|---|---|---|---|---|---|
| bm25 | 0.928 | 0.916 | 0.961 | 0.32 ms | 0.58 ms |
| dense | 0.868 | 0.866 | 0.878 | 0.38 ms | 1.70 ms |
| hybrid | 0.873 | 0.869 | 0.914 | 0.47 ms | 1.13 ms |

**Vocabulary-mismatch probe.** Dense results that share *no* term with the query (unreachable for BM25 by construction): 8328 docs returned, precision against the topic label = **0.323** (random-guess baseline ~0.125).

**Freshness.** Mean age of top-10 results: 10.3s without decay vs 6.4s with a 30s half-life.

### 4. Burst (breaking-news) detection

0 injected breaking events: **0 detected**, 0 missed; detection delay median **0.0s** / max 0.0s of event time (bucket = 10s). Distinct false-alarm term bursts: **0**.

### 5. Online story clustering

58 active clusters, 57 with >=5 docs, purity (label of first 50 docs) **1.000**.

### 6. Standing-query (percolator) scaling

| registered queries | match time / doc | docs/s |
|---|---|---|
| 100 | 0.006 ms | 175,670 |
| 1,000 | 0.044 ms | 22,604 |
| 10,000 | 0.478 ms | 2,091 |

### 7. Crash recovery (write-ahead log)

Indexed 928 docs, simulated crash with a torn final log record, rebuilt **928 docs in 0.64s**; identical top-5 ranking after recovery: **True**.

