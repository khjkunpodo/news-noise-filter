"""Stage 2 — group near-duplicate headlines.

The same event gets written by a dozen outlets with slightly different wording.
We collapse those to one representative using character-bigram Jaccard
similarity — no model call, no embeddings, just set overlap. Entity aliases are
normalized first so "KB Insurance" and "KB Ins." count as the same word.
"""

from __future__ import annotations

import re


def _normalize(title: str, aliases: dict[str, str]) -> str:
    text = title.lower()
    for variant, canonical in aliases.items():
        text = text.replace(variant.lower(), canonical.lower())
    # Keep letters/digits/spaces only; collapse whitespace.
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _bigrams(text: str) -> set[str]:
    packed = text.replace(" ", "")
    return {packed[i : i + 2] for i in range(len(packed) - 1)}


def similar(a: str, b: str, threshold: float, aliases: dict[str, str]) -> bool:
    ga = _bigrams(_normalize(a, aliases))
    gb = _bigrams(_normalize(b, aliases))
    if not ga or not gb:
        return False
    overlap = len(ga & gb) / len(ga | gb)
    return overlap >= threshold


def dedup(
    titles: list[str],
    threshold: float = 0.55,
    aliases: dict[str, str] | None = None,
) -> list[str]:
    """Return one representative per near-duplicate cluster, order preserved.

    The first headline seen in a cluster wins; later matches are dropped.
    """
    aliases = aliases or {}
    kept: list[str] = []
    for title in titles:
        if any(similar(title, k, threshold, aliases) for k in kept):
            continue
        kept.append(title)
    return kept
