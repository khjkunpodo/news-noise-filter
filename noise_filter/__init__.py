"""A 3-stage news-headline noise filter: regex → dedup → LLM judge."""

from .pipeline import Result, run
from .rules import Rules

__all__ = ["Rules", "Result", "run"]
