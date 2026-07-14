# news-noise-filter

A three-stage filter that turns a raw firehose of news headlines into the
handful a professional actually needs to read — cheaply, and without letting a
language model quietly delete the one headline that mattered.

I built this because a daily news digest I run kept burying real signal under
promotional filler: charity photo-ops, award ceremonies, "shares rose 2%"
tickers, and the same event rewritten by eight outlets. A flat keyword blocklist
was too blunt (it dropped real earnings news) and a pure LLM classifier was too
expensive and too unpredictable (it sometimes dropped the exact headline I most
wanted). The fix was to layer three cheap-to-expensive stages and give the
operator a rule the model cannot override.

## The pipeline

```
raw headlines
    │  Stage 1 — regex gate          cheap, deterministic
    ▼
rule-clean
    │  Stage 2 — near-dup grouping   cheap, deterministic
    ▼
deduped
    │  Stage 3 — LLM relevance judge optional · fail-open · rule-overridable
    ▼
kept
```

**Stage 1 — regex gate** (`noise_filter/rules.py`). One compiled pass drops the
obvious noise: donations, awards, MOUs, giveaways, photo captions, and stock
chatter (target prices, up/downgrades, intraday moves). Earnings coverage is
kept *only* for the entities you track — a generic "sector earnings roundup" or
an analyst's estimate is dropped even when it names a company you follow.

**Stage 2 — near-duplicate grouping** (`noise_filter/dedup.py`). The same event
gets written a dozen ways. Character-bigram Jaccard similarity collapses those to
one representative — no embeddings, no model call, just set overlap. Entity
aliases are normalized first so "Acme Corp" and "Acme" count as the same word.

**Stage 3 — LLM relevance judge** (`noise_filter/llm_judge.py`). Rules can't
phrase everything ("is this a substantive story or a puff piece?"), so a small,
cheap model reads the survivors and returns the ones worth attention. This stage
is where the two design decisions that make the whole thing trustworthy live.

## The two things that make stage 3 safe

An LLM in a pipeline is a liability unless you bound its failure modes. Two
properties do that here:

1. **Fail-open.** No API key, or the call errors, or the model returns garbage →
   every headline passes through untouched. A broken judge must never silently
   empty the digest. A missing filter is a nuisance; a missing digest is a
   failure.

2. **The operator outranks the model.** A headline the rules marked `must_keep`
   — real results from an entity you track, and not merely an analyst's estimate
   of it — is re-added even if the model dropped it. In practice a cheap model
   *does* occasionally misjudge a genuine earnings headline as promotional; this
   rule guarantees it can't cost you the one story the digest exists to deliver.
   The re-add is guarded against near-duplicates so it never resurrects a dupe
   already collapsed in stage 2.

Ordering cheap-and-deterministic before expensive-and-fuzzy also means the model
only ever sees the ~half of headlines the rules couldn't decide — less token
spend, and a smaller, cleaner input for it to reason about.

## Try it

Runs offline against committed sample data. With no API key, stage 3 fails open
(passes everything through), so you see stages 1 and 2 immediately:

```bash
python -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python demo.py --no-llm
```

Measured on the 32-headline sample:

```
raw           32
after rules   18   (-14)   promotional filler, market chatter, analyst estimates
after dedup   16   (-2)    two rewrites of the same two stories
after LLM     ~14  (-2)    generic roundups & captions rules can't phrase*
```

\* Stage 3 needs credentials. Export `ANTHROPIC_API_KEY` and drop `--no-llm` to
run it — it drops things like a "weekly market wrap" roundup and a bare analyst
line that carry no earnings keyword for the rules to catch. The exact count
varies with the model; the point is that stages 1–2 do the deterministic bulk of
the work and the model handles only the residue.

## Configuration

Everything domain-specific is data (`config.example.yaml`) — the engine is
domain-neutral. Point the patterns at your own beat and language and the pipeline
follows; no code changes. The sample uses an English tech-industry example with
fictional companies.

## Model

Stage 3 is a classification task, so it defaults to a small model
(`claude-haiku-4-5`) via the official Anthropic SDK. Override with the
`NOISE_FILTER_MODEL` environment variable to A/B a larger one.

## Tests

```bash
./.venv/bin/pip install pytest && ./.venv/bin/python -m pytest
```

The rule and dedup stages are covered deterministically (no network). Stage 3's
guarantees — fail-open and the `must_keep` override — are asserted through the
pipeline without a live API call.

## Layout

```
noise_filter/
  rules.py       Stage 1 — regex gate + must_keep
  dedup.py       Stage 2 — bigram-Jaccard near-dup grouping
  llm_judge.py   Stage 3 — LLM judge (optional, fail-open, rule-overridable)
  pipeline.py    wires the three stages with per-stage counts
config.example.yaml   patterns, entities, aliases, threshold
sample_data/          committed headlines so the demo runs offline
demo.py               before/after report
tests/
```
