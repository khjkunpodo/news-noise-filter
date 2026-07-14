"""Stage 3 — LLM relevance judge (optional, fail-open).

Rules catch the obvious noise; this catches what rules can't phrase — photo
captions, promotional puff pieces, daily roundups. A small, cheap model reads
the surviving headlines and returns the ones worth a professional's attention.

Two safety properties make this safe to put in a pipeline:

  * fail-open — if the API key is missing or the call errors, every headline
    passes through. A broken judge must never silently empty the digest.
  * rule override — a headline the rules marked must_keep is re-added even if
    the model dropped it. The model advises; it does not overrule the operator.
"""

from __future__ import annotations

import os
import re

from .dedup import similar
from .rules import Rules

# A classification/filtering task — a small model is the right tool. Override
# with NOISE_FILTER_MODEL if you want to A/B a larger one.
DEFAULT_MODEL = "claude-haiku-4-5"

_SYSTEM = (
    "You screen news headlines for a busy professional who tracks a specific "
    "industry. Keep headlines with substantive, work-relevant information: "
    "company results, product and regulatory changes, market-moving events. "
    "Drop promotional filler, photo captions, human-interest fluff, and generic "
    "daily roundups. Reply with ONLY the numbers of the headlines to KEEP, "
    "comma-separated (e.g. '1, 3, 4'). If all are worth keeping, list them all."
)


def judge(
    titles: list[str],
    rules: Rules,
    model: str | None = None,
    sim_threshold: float = 0.55,
    aliases: dict[str, str] | None = None,
) -> list[str]:
    """Return the subset of ``titles`` the model judged relevant.

    must_keep headlines are always in the result. On any failure the full input
    is returned unchanged (fail-open).
    """
    if not titles:
        return titles
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return titles  # fail-open: no credentials, no filtering

    try:
        import anthropic
    except ImportError:
        return titles

    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(titles, 1))
    try:
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model or os.environ.get("NOISE_FILTER_MODEL", DEFAULT_MODEL),
            max_tokens=256,
            system=_SYSTEM,
            messages=[{"role": "user", "content": numbered}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
    except Exception:
        return titles  # fail-open on any API/parse error

    keep_idx = {int(n) for n in re.findall(r"\d+", text)}
    if not keep_idx:
        return titles  # model said nothing usable — don't trust an empty verdict

    kept = [t for i, t in enumerate(titles, 1) if i in keep_idx]

    # Rule override: re-add any must_keep the model dropped, guarding against
    # near-duplicates that are already present so we don't resurrect a dupe.
    aliases = aliases or {}
    for t in titles:
        if not rules.must_keep(t):
            continue
        if any(similar(t, k, sim_threshold, aliases) for k in kept):
            continue
        kept.append(t)
    return kept
