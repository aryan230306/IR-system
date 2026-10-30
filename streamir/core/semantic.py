"""Streaming semantic vectors via Random Indexing (Sahlgren, 2005).

Every term gets a fixed sparse ternary *index vector*. A term's *context vector* is the running
sum of the index vectors of its neighbours in the stream. Terms used in similar contexts end up
with similar context vectors - so the semantic space is learned **online, in one pass, with no
training phase, no GPU and no pretrained model**, and keeps adapting as the stream drifts.

    doc_vec = normalize( sum_t  w(t) * ( ctx(t)/|ctx(t)| + lex * idx(t)/|idx(t)| ) )
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Callable

import numpy as np


def _seed(term: str) -> int:
    return int.from_bytes(hashlib.blake2b(term.encode(), digest_size=8).digest(), "big")


class SemanticIndex:
    def __init__(self, dim: int = 256, nnz: int = 8, window: int = 4, lex_weight: float = 0.6,
                 max_terms: int = 40_000) -> None:
        self.dim, self.nnz, self.window, self.lex = dim, nnz, window, lex_weight
        self.max_terms = max_terms
        self._idx: dict[str, np.ndarray] = {}
        self._ctx: dict[str, np.ndarray] = {}
        self._mat = np.zeros((1024, dim), dtype=np.float32)
        self._ids: list[int] = []
        self._pos: dict[int, int] = {}
        self._dead = 0
        self._vocab_cache: tuple[float, list[str], np.ndarray] | None = None
        self.learned_docs = 0

    # ---------------------------------------------------------- vector helpers
    def index_vec(self, term: str) -> np.ndarray:
        v = self._idx.get(term)
        if v is None:
            rng = np.random.default_rng(_seed(term))
            v = np.zeros(self.dim, dtype=np.float32)
            pos = rng.choice(self.dim, size=self.nnz, replace=False)
            v[pos] = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=self.nnz)
            v /= np.sqrt(self.nnz)
            if len(self._idx) < 200_000:
                self._idx[term] = v
        return v

    def term_vec(self, term: str) -> np.ndarray:
        idx = self.index_vec(term)
        ctx = self._ctx.get(term)
        if ctx is None:
            return self.lex * idx
        n = np.linalg.norm(ctx)
        return (ctx / n if n > 0 else 0) + self.lex * idx

    # ---------------------------------------------------------------- learning
    def learn(self, tokens: list[str]) -> None:
        """Update context vectors from an ordered token list (one pass, O(L * dim))."""
        L = len(tokens)
        if L < 2:
            return
        I = np.stack([self.index_vec(t) for t in tokens])               # L x dim
        cs = np.vstack([np.zeros((1, self.dim), np.float32), np.cumsum(I, axis=0)])
        w = self.window
        for i, t in enumerate(tokens):
            lo, hi = max(0, i - w), min(L, i + w + 1)
            s = cs[hi] - cs[lo] - I[i]
            c = self._ctx.get(t)
            if c is None:
                if len(self._ctx) >= self.max_terms:
                    continue
                self._ctx[t] = s.copy()
            else:
                c += s
        self.learned_docs += 1

    # --------------------------------------------------------------- encoding
    def encode(self, tf: Counter, idf: Callable[[str], float]) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t, c in tf.items():
            v += (1 + np.log(c)) * idf(t) * self.term_vec(t)
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # ------------------------------------------------------------------ store
    def add(self, did: int, vec: np.ndarray) -> None:
        n = len(self._ids)
        if n >= self._mat.shape[0]:
            self._mat = np.vstack([self._mat, np.zeros_like(self._mat)])
        self._mat[n] = vec
        self._pos[did] = n
        self._ids.append(did)

    def remove(self, did: int) -> None:
        if self._pos.pop(did, None) is not None:
            self._dead += 1

    def compact(self, is_live: Callable[[int], bool]) -> int:
        keep = [i for i, d in enumerate(self._ids) if d in self._pos and is_live(d)]
        for d in [d for d in self._pos if not is_live(d)]:
            del self._pos[d]
        before = len(self._ids)
        self._mat[: len(keep)] = self._mat[keep]
        self._ids = [self._ids[i] for i in keep]
        self._pos = {d: i for i, d in enumerate(self._ids)}
        self._dead = 0
        return before - len(self._ids)

    def search(self, qvec: np.ndarray, k: int, is_live: Callable[[int], bool],
               min_sim: float = 0.0) -> list[tuple[int, float]]:
        n = len(self._ids)
        if n == 0 or not np.any(qvec):
            return []
        sims = self._mat[:n] @ qvec
        cand = np.argpartition(-sims, min(n - 1, 5 * k))[: 5 * k] if n > 5 * k else np.arange(n)
        cand = cand[np.argsort(-sims[cand])]
        out = []
        for i in cand:
            did = self._ids[i]
            if did in self._pos and is_live(did) and sims[i] > min_sim:
                out.append((did, float(sims[i])))
                if len(out) >= k:
                    break
        return out

    # --------------------------------------------------- term-level semantics
    def similar_terms(self, term: str, n: int = 8, min_ctx: float = 3.0) -> list[tuple[str, float]]:
        """Nearest terms in the learned context space (query expansion / explainability)."""
        if term not in self._ctx:
            return []
        vocab = [t for t, c in self._ctx.items() if np.abs(c).sum() >= min_ctx]
        if not vocab:
            return []
        M = np.stack([self._ctx[t] for t in vocab])
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
        q = self._ctx[term] / (np.linalg.norm(self._ctx[term]) + 1e-9)
        sims = M @ q
        order = np.argsort(-sims)
        return [(vocab[i], round(float(sims[i]), 3)) for i in order if vocab[i] != term][:n]

    def __len__(self) -> int:
        return len(self._pos)
