"""Tokenisation, stop-words and a light suffix stemmer (no external NLP deps)."""
from __future__ import annotations

import re
from functools import lru_cache

STOPWORDS = frozenset("""
a about above after again all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers him his how i if in into is it its just me more most my no nor not now of
off on once only or other our out over own same she should so some such than that the their them
then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your says said say new
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_CVC = re.compile(r"[^aeiou][aeiou][^aeiouwxy]$")
_SUFFIXES = (("ational", "ate"), ("ization", "ize"), ("fulness", "ful"), ("ousness", "ous"),
             ("iveness", "ive"), ("ments", "ment"), ("ities", "ity"), ("ings", "ing"),
             ("sses", "ss"), ("ies", "y"), ("ing", ""), ("ed", ""), ("ly", ""), ("es", ""),
             ("s", ""))


@lru_cache(maxsize=200_000)
def stem(word: str) -> str:
    """Light, fast suffix stemmer. Goal: conflate rate/rates/rated/rating -> one term."""
    if len(word) <= 3 or word.isdigit():
        return word
    out = word
    for suf, rep in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            if suf == "s" and word.endswith(("ss", "us", "is")):
                continue
            if suf == "es" and not word[:-2].endswith(("s", "x", "z", "ch", "sh")):
                suf, rep = "s", ""                       # rates -> rate, boxes -> box
            out = word[: -len(suf)] + rep
            # collapse doubled final consonant: "running" -> "run"
            if suf in ("ing", "ed"):
                if len(out) > 3 and out[-1] == out[-2] and out[-1] not in "lsz":
                    out = out[:-1]
                elif (len(out) <= 4 and _CVC.search(out)) or out.endswith("v") or \
                        (out.endswith("c") and not out.endswith("ic")):
                    out += "e"                             # rated -> rate, announced -> announce
            break
    return out


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if keep_stopwords:
        return [stem(t) for t in toks]
    return [stem(t) for t in toks if t not in STOPWORDS and len(t) > 1]


def make_snippet(text: str, query_terms: set[str], width: int = 180) -> str:
    """Return the window of `text` with the highest density of query terms."""
    if len(text) <= width:
        return text
    words = text.split()
    if not words:
        return ""
    flags = [1 if stem(re.sub(r"[^a-z0-9]", "", w.lower())) in query_terms else 0 for w in words]
    best_i, best, run, size = 0, -1, 0, max(6, width // 6)
    for i in range(len(words)):
        run += flags[i]
        if i >= size:
            run -= flags[i - size]
        if run > best:
            best, best_i = run, max(0, i - size + 1)
    s = " ".join(words[best_i:best_i + size])
    return ("…" if best_i else "") + s + ("…" if best_i + size < len(words) else "")
