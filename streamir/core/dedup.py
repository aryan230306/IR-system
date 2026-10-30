"""Streaming near-duplicate detection: word-shingle MinHash + banded LSH + verification.

1. Each document -> set of unigram + bigram shingles -> 64-permutation MinHash signature.
2. The signature is cut into 16 bands of 4 rows; documents sharing a whole band land in the same
   bucket, so candidates are found with 16 dict probes regardless of how many docs are stored.
3. Candidates are *verified* by the signature-estimated Jaccard similarity; only pairs above the
   threshold are reported as duplicates. (Candidate generation gives recall, verification gives
   precision - an LSH-only filter flags too many template-like news items.)
"""
from __future__ import annotations

import hashlib
from collections import deque
from functools import lru_cache

import numpy as np

_NPERM, _BANDS = 64, 16
_ROWS = _NPERM // _BANDS
_rng = np.random.default_rng(0xC0FFEE)
_A = (_rng.integers(1, 2**63, size=_NPERM, dtype=np.uint64) | np.uint64(1))
_B = _rng.integers(0, 2**63, size=_NPERM, dtype=np.uint64)
_MASK = (1 << 64) - 1


@lru_cache(maxsize=500_000)
def _h64(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")


def shingles(tokens: list[str]) -> np.ndarray:
    hs = [_h64(t) for t in tokens]
    hs += [((a * 1000003) ^ b) & _MASK for a, b in zip(hs, hs[1:])]
    return np.unique(np.array(hs, dtype=np.uint64))


def minhash(tokens: list[str]) -> np.ndarray:
    sh = shingles(tokens)
    with np.errstate(over="ignore"):
        mixed = sh[:, None] * _A[None, :] + _B[None, :]
    return (mixed >> np.uint64(32)).min(axis=0).astype(np.uint32)


def jaccard_est(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a == b))


class NearDupDetector:
    def __init__(self, threshold: float = 0.8, retention: int = 100_000) -> None:
        self.threshold = threshold
        self.retention = retention
        self.tables: dict[tuple[int, bytes], list[str]] = {}
        self.sigs: dict[str, np.ndarray] = {}
        self._fifo: deque[str] = deque()

    def _keys(self, sig: np.ndarray):
        for b in range(_BANDS):
            yield (b, sig[b * _ROWS:(b + 1) * _ROWS].tobytes())

    def find(self, sig: np.ndarray) -> tuple[str | None, float]:
        best, best_j, seen = None, 0.0, set()
        for key in self._keys(sig):
            for doc_id in self.tables.get(key, ()):
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                j = jaccard_est(sig, self.sigs[doc_id])
                if j > best_j:
                    best, best_j = doc_id, j
        return (best, best_j) if best_j >= self.threshold else (None, best_j)

    def add(self, sig: np.ndarray, doc_id: str) -> None:
        self.sigs[doc_id] = sig
        for key in self._keys(sig):
            self.tables.setdefault(key, []).append(doc_id)
        self._fifo.append(doc_id)
        while len(self._fifo) > self.retention:                 # bounded memory
            old = self._fifo.popleft()
            osig = self.sigs.pop(old, None)
            if osig is None:
                continue
            for key in self._keys(osig):
                bucket = self.tables.get(key)
                if bucket:
                    try:
                        bucket.remove(old)
                    except ValueError:
                        pass
                    if not bucket:
                        del self.tables[key]

    def check_and_add(self, tokens: list[str], doc_id: str) -> tuple[np.ndarray, str | None]:
        sig = minhash(tokens)
        dup, _ = self.find(sig)
        if dup is None:
            self.add(sig, doc_id)
        return sig, dup

    def __len__(self) -> int:
        return len(self.sigs)
