"""Lexical (BM25) scoring + Reciprocal Rank Fusion. Stdlib only.

RRF fuses rankings by RANK, not score, so dense (cosine) and lexical (BM25)
signals combine without needing a comparable scale.

Tokens are stemmed with a compact Porter algorithm implementation so BM25
matches morphological variants ("reading" ↔ "read", "offering" ↔ "offer",
"visited" ↔ "visit"). Without stemming, verb-specific retrieval fails on
every surface-form mismatch between the question and the stored text.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------
# Porter stemmer — compact stdlib-only implementation. Follows Martin Porter's
# 1980 paper. Handles the common English suffix families (-ing, -ed, -s,
# -ational, -ization, -ate, -er, -est, -ly, ...). Not exhaustive on irregular
# forms; those slip through unstemmed, which is safe for BM25 (they just
# don't match their inflected variants — the pre-stemmer behaviour).
# Adapted from the classic algorithm; conservative on aggressive stripping.

_VOWELS = set("aeiou")


def _is_vowel(word: str, i: int) -> bool:
    ch = word[i]
    if ch in _VOWELS:
        return True
    if ch == "y" and i > 0 and not _is_vowel(word, i - 1):
        return True
    return False


def _measure(stem: str) -> int:
    """Porter's `m`: count of VC transitions in the stem."""
    if not stem:
        return 0
    n = len(stem)
    seen_vowel = False
    m = 0
    for i in range(n):
        v = _is_vowel(stem, i)
        if v:
            seen_vowel = True
        elif seen_vowel:
            m += 1
            seen_vowel = False
    return m


def _has_vowel(stem: str) -> bool:
    return any(_is_vowel(stem, i) for i in range(len(stem)))


def _ends_double_consonant(stem: str) -> bool:
    if len(stem) < 2:
        return False
    return (stem[-1] == stem[-2]
            and not _is_vowel(stem, len(stem) - 1))


def _cvc(stem: str) -> bool:
    """Ends CVC where the last C is not w, x, or y (Porter's `*o`)."""
    if len(stem) < 3:
        return False
    if _is_vowel(stem, len(stem) - 1):
        return False
    if not _is_vowel(stem, len(stem) - 2):
        return False
    if _is_vowel(stem, len(stem) - 3):
        return False
    return stem[-1] not in "wxy"


def _replace(word: str, suffix: str, replacement: str) -> str:
    return word[: len(word) - len(suffix)] + replacement


def _step1a(w: str) -> str:
    if w.endswith("sses"):
        return w[:-2]
    if w.endswith("ies"):
        return w[:-2]
    if w.endswith("ss"):
        return w
    if w.endswith("s"):
        return w[:-1]
    return w


def _step1b(w: str) -> str:
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            return w[:-1]
        return w
    for suf in ("ed", "ing"):
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if _has_vowel(stem):
                w = stem
                if w.endswith(("at", "bl", "iz")):
                    w += "e"
                elif _ends_double_consonant(w) and w[-1] not in "lsz":
                    w = w[:-1]
                elif _measure(w) == 1 and _cvc(w):
                    w += "e"
                return w
    return w


def _step1c(w: str) -> str:
    if len(w) > 1 and w.endswith("y") and _has_vowel(w[:-1]):
        return w[:-1] + "i"
    return w


_STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("bli", "ble"), ("alli", "al"), ("entli", "ent"),
    ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
    ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
    ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
    ("logi", "log"),
]


def _step2(w: str) -> str:
    for suf, rep in _STEP2:
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if _measure(stem) > 0:
                return stem + rep
            return w
    return w


_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ful", ""), ("ness", ""),
]


def _step3(w: str) -> str:
    for suf, rep in _STEP3:
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if _measure(stem) > 0:
                return stem + rep
            return w
    return w


_STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
    "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]


def _step4(w: str) -> str:
    for suf in _STEP4:
        if w.endswith(suf):
            stem = w[: -len(suf)]
            if _measure(stem) > 1:
                return stem
            return w
    # Special: "ion" only after s or t
    if w.endswith("ion"):
        stem = w[:-3]
        if _measure(stem) > 1 and stem and stem[-1] in "st":
            return stem
    return w


def _step5a(w: str) -> str:
    if w.endswith("e"):
        stem = w[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            return stem
    return w


def _step5b(w: str) -> str:
    if len(w) > 1 and w.endswith("ll") and _measure(w) > 1:
        return w[:-1]
    return w


_STEM_CACHE: dict[str, str] = {}


def porter_stem(word: str) -> str:
    """Reduce an English word to its Porter stem. Cached per-token per-process."""
    if len(word) < 3:
        return word
    if word in _STEM_CACHE:
        return _STEM_CACHE[word]
    w = word
    w = _step1a(w)
    w = _step1b(w)
    w = _step1c(w)
    w = _step2(w)
    w = _step3(w)
    w = _step4(w)
    w = _step5a(w)
    w = _step5b(w)
    _STEM_CACHE[word] = w
    return w


def tokenize(text: str) -> list[str]:
    return [porter_stem(t) for t in _TOKEN.findall(text.lower())]


class BM25:
    """Okapi BM25 over a fixed corpus. Build once, score many queries."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus = [tokenize(d) for d in docs]
        self.n = len(self.corpus)
        self.avgdl = (sum(len(d) for d in self.corpus) / self.n) if self.n else 1.0
        self.tf = [Counter(d) for d in self.corpus]
        self.df: Counter = Counter()
        for d in self.corpus:
            self.df.update(set(d))

    def scores(self, query: str) -> list[float]:
        if not self.n:
            return []
        q = tokenize(query)
        out = []
        for tf, doc in zip(self.tf, self.corpus):
            dl = len(doc)
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                idf = math.log(1 + (self.n - self.df[t] + 0.5) / (self.df[t] + 0.5))
                s += idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


def rrf(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over several id-rankings (each best-first).

    score(id) = sum over rankings of 1 / (k + rank). Uses rank only, so scorers
    on different scales (cosine vs BM25) fuse safely.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, _id in enumerate(ranking, start=1):
            fused[_id] = fused.get(_id, 0.0) + 1.0 / (k + rank)
    return fused
