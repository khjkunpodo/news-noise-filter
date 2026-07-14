#!/usr/bin/env python3
"""Run the filter on the committed sample data and print a before/after report.

Works offline with no API key — the LLM stage fails open and passes headlines
through untouched, so you can see stages 1 and 2 immediately. Set
ANTHROPIC_API_KEY to enable stage 3.

    python demo.py                 # all three stages (stage 3 is a no-op w/o key)
    python demo.py --no-llm        # rules + dedup only
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib

import yaml

from noise_filter import run

ROOT = pathlib.Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.example.yaml"))
    ap.add_argument("--data", default=str(ROOT / "sample_data" / "headlines.json"))
    ap.add_argument("--no-llm", action="store_true", help="skip the LLM judge")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    titles = json.loads(pathlib.Path(args.data).read_text())

    result = run(cfg, titles, use_llm=not args.no_llm)
    c = result.counts

    print(f"raw           {c['raw']:>3}")
    print(f"after rules   {c['after_rules']:>3}  (-{c['raw'] - c['after_rules']})")
    print(f"after dedup   {c['after_dedup']:>3}  (-{c['after_rules'] - c['after_dedup']})")
    llm_note = "" if os.environ.get("ANTHROPIC_API_KEY") or args.no_llm else "  (no API key — stage skipped)"
    print(f"after LLM     {c['kept']:>3}  (-{c['after_dedup'] - c['kept']}){llm_note}")
    print()
    print("KEPT:")
    for t in result.kept:
        print(f"  • {t}")


if __name__ == "__main__":
    main()
