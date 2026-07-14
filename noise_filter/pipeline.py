"""The 3-stage pipeline, wired together with per-stage counts.

    raw headlines
        │  Stage 1: regex gate      (rules.is_noise)
        ▼
    rule-clean
        │  Stage 2: dedup           (bigram Jaccard)
        ▼
    deduped
        │  Stage 3: LLM judge       (optional, fail-open, must_keep override)
        ▼
    kept
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .dedup import dedup
from .llm_judge import judge
from .rules import Rules


@dataclass
class Result:
    kept: list[str]
    counts: dict[str, int] = field(default_factory=dict)


def run(cfg: dict, titles: list[str], use_llm: bool = True) -> Result:
    rules = Rules.from_config(cfg)
    threshold = cfg.get("sim_threshold", 0.55)
    aliases = cfg.get("aliases", {})

    counts = {"raw": len(titles)}

    stage1 = [t for t in titles if not rules.is_noise(t)]
    counts["after_rules"] = len(stage1)

    stage2 = dedup(stage1, threshold=threshold, aliases=aliases)
    counts["after_dedup"] = len(stage2)

    if use_llm:
        stage3 = judge(
            stage2, rules, sim_threshold=threshold, aliases=aliases
        )
    else:
        stage3 = stage2
    counts["kept"] = len(stage3)

    return Result(kept=stage3, counts=counts)
