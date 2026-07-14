"""Stage 1 — cheap regex gate.

A single compiled-regex pass decides whether a headline is obvious noise, and
whether it is one of the few headlines a rule says we must *always* keep. Both
are driven by the config, so the same engine works for any domain — swap the
patterns, not the code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Rules:
    """Compiled rule set. Build it once from config, reuse per headline."""

    noise: re.Pattern
    market: re.Pattern
    earnings: re.Pattern
    analyst: re.Pattern
    priority_entities: re.Pattern

    @classmethod
    def from_config(cls, cfg: dict) -> "Rules":
        def compile_group(patterns: list[str]) -> re.Pattern:
            # An empty group must never match; "(?!)" is "match nothing".
            joined = "|".join(f"(?:{p})" for p in patterns) or "(?!)"
            return re.compile(joined, re.IGNORECASE)

        return cls(
            noise=compile_group(cfg.get("noise_patterns", [])),
            market=compile_group(cfg.get("market_patterns", [])),
            earnings=compile_group(cfg.get("earnings_patterns", [])),
            analyst=compile_group(cfg.get("analyst_patterns", [])),
            priority_entities=compile_group(cfg.get("priority_entities", [])),
        )

    def is_noise(self, title: str) -> bool:
        """True if the headline should be dropped by rule alone.

        Order matters: a must-keep headline is never noise, even if it also
        matches a market/earnings pattern.
        """
        if self.must_keep(title):
            return False
        if self.noise.search(title):
            return True
        # Stock/market chatter (target prices, up/downgrades) is noise for a
        # reader who tracks the business, not the ticker.
        if self.market.search(title):
            return True
        # Earnings coverage is only interesting for the entities we track; a
        # generic "sector earnings roundup" or an analyst estimate is not.
        if self.earnings.search(title):
            if not self.priority_entities.search(title):
                return True
            if self.analyst.search(title):
                return True
        return False

    def must_keep(self, title: str) -> bool:
        """A hard 'always keep' rule that later stages cannot override.

        Real earnings news about an entity we track — and not merely an
        analyst's estimate of it — is the signal the whole pipeline exists to
        deliver. The LLM stage is explicitly not allowed to drop these.
        """
        return bool(
            self.earnings.search(title)
            and self.priority_entities.search(title)
            and not self.analyst.search(title)
        )
