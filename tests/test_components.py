import random
from collections import Counter

import numpy as np

from streamir.core.burst import BurstDetector
from streamir.core.cluster import OnlineClusterer
from streamir.core.dedup import NearDupDetector, minhash, jaccard_est
from streamir.core.index import StreamIndex
from streamir.core.percolator import Percolator
from streamir.core.semantic import SemanticIndex
from streamir.models import Document
from streamir.text import stem, tokenize


# ---------------------------------------------------------------- text
def test_tokenize_stems_and_stopwords():
    assert "run" in tokenize("The runners were running quickly")
    assert "the" not in tokenize("the cat")
    assert stem("markets") == "market" and stem("glass") == "glass"
    assert len({stem(w) for w in ("rate", "rates", "rated", "rating")}) == 1
    assert stem("boxes") == "box" and stem("studies") == "study"


# ---------------------------------------------------------------- dedup
def test_dedup_detects_near_dups_not_distinct():
    det = NearDupDetector(threshold=0.8)
    a = tokenize("Apple unveils a new smartphone with an AI chip and a better camera system for photographers worldwide today")
    b = a + ["syndicated"]
    c = tokenize("Floods in Brazil displace thousands as rivers overflow and rescue teams struggle to reach villages")
    assert det.check_and_add(a, "a")[1] is None
    assert det.check_and_add(b, "b")[1] == "a"
    assert det.check_and_add(c, "c")[1] is None


def test_minhash_estimates_jaccard():
    a = [f"w{i}" for i in range(100)]
    b = [f"w{i}" for i in range(50, 150)]           # unigram jaccard 1/3; bigrams add further shape
    j = jaccard_est(minhash(a), minhash(b))
    assert 0.1 < j < 0.5


def test_dedup_bounded_memory():
    det = NearDupDetector(retention=50)
    for i in range(500):
        det.check_and_add([f"tok{i}x{j}" for j in range(12)], str(i))
    assert len(det) <= 50
    assert sum(len(v) for v in det.tables.values()) <= 50 * 16


# ---------------------------------------------------------------- burst
def test_burst_detector_flags_spike_and_ignores_steady():
    b = BurstDetector(bucket_seconds=10, min_count=5, z_threshold=4, warmup_buckets=3)
    rng = random.Random(0)
    ts = 0.0
    for _ in range(300):                               # 30 buckets of steady background
        ts += 1.0
        b.observe(["steady", "noise"] if rng.random() < 0.5 else ["noise"], ts)
    assert not [x for x in b.top() if x["term"] in ("steady", "noise")]
    for _ in range(40):                                # sudden burst of a new term
        ts += 0.1
        b.observe(["quake", "noise"], ts)
    top = b.top()
    assert top and top[0]["term"] == "quake"


def test_burst_decays_after_quiet_period():
    b = BurstDetector(bucket_seconds=10, min_count=3, warmup_buckets=1)
    for i in range(30):
        b.observe(["x"], i * 1.0)
    b.tick(10_000)
    assert b.stats["x"].mean < 0.01


# ---------------------------------------------------------------- clustering
def test_online_clustering_groups_related_docs():
    ix = StreamIndex()
    cl = OnlineClusterer(threshold=0.3)
    def feed(i, text):
        d = Document(str(i), text, float(i))
        tf = ix.analyze(d); ix.add(d, tf)
        return cl.assign(d.id, text, "s", tf, ix.idf, d.ts)[0]
    for i in range(100, 130):                          # background so idf is meaningful
        feed(i, f"filler story number{i} about topic{i % 7} and subject{i % 11} plus thing{i}")
    a1 = feed(1, "earthquake strikes chile tsunami warning issued coast")
    a2 = feed(2, "chile earthquake tsunami alert coast evacuated")
    b1 = feed(3, "nvidia launches new ai chip for data centers")
    b2 = feed(4, "ai chip nvidia data centers demand")
    assert a1 == a2 and b1 == b2 and a1 != b1


# ---------------------------------------------------------------- percolator
def test_percolator_semantics_and_pruning_is_lossless():
    idf = lambda t: 1.0 + (hash(t) % 5) / 5
    p = Percolator()
    q1 = p.register("ransomware breach +servers", 0.6, idf=idf)
    q2 = p.register("earthquake tsunami", 0.5, idf=idf)
    q3 = p.register("apple -pie", 1.0, idf=idf)
    got = {q.qid for q, _ in p.match({"ransomware", "server"}, idf)}
    assert got == {q1.qid}
    assert {q.qid for q, _ in p.match({"server"}, idf)} == set()          # below min_match
    assert {q.qid for q, _ in p.match({"tsunami"}, idf)} == {q2.qid}
    assert {q.qid for q, _ in p.match({"apple"}, idf)} == {q3.qid}
    assert {q.qid for q, _ in p.match({"apple", "pie"}, idf)} == set()     # excluded term


def test_percolator_pruned_index_equals_bruteforce():
    rng = random.Random(1)
    vocab = [f"t{i}" for i in range(40)]
    idf = lambda t: 1.0 + (int(t[1:]) % 7)
    p = Percolator()
    for _ in range(300):
        p.register(" ".join(rng.sample(vocab, rng.randint(1, 5))), rng.choice([0.3, 0.5, 0.67, 1.0]), idf=idf)
    for _ in range(200):
        doc = set(rng.sample(vocab, rng.randint(2, 15)))
        fast = {q.qid for q, _ in p.match(doc, idf)}
        slow = {q.qid for q in p.list()
                if len([t for t in q.parsed.scoring_terms if t in doc]) >= q.need}
        assert fast == slow


# ---------------------------------------------------------------- semantic
def test_random_indexing_learns_synonyms_from_stream():
    rng = random.Random(2)
    sem = SemanticIndex(dim=256, window=3)
    ctx_words = ["engine", "road", "driver", "wheel", "fuel", "traffic"]
    other = ["recipe", "oven", "flour", "butter", "sugar", "bake"]
    for _ in range(400):
        w = rng.choice(["car", "automobile"])
        sem.learn(rng.sample(ctx_words, 3) + [w] + rng.sample(ctx_words, 2))
        sem.learn(rng.sample(other, 3) + [rng.choice(["cake", "pastry"])] + rng.sample(other, 2))
    sims = dict(sem.similar_terms("car", 20))
    assert sims["automobile"] > sims.get("cake", -1) and sims["automobile"] > 0.6


def test_semantic_search_and_removal():
    sem = SemanticIndex(dim=64)
    v1, v2 = np.ones(64, np.float32) / 8, -np.ones(64, np.float32) / 8
    sem.add(1, v1); sem.add(2, v2)
    assert sem.search(v1, 5, lambda d: True, min_sim=0.1)[0][0] == 1
    sem.remove(1)
    assert sem.search(v1, 5, lambda d: True, min_sim=0.1) == []
