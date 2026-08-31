# ADR-003 — Verify a result that looks too good before publishing it

**Status:** Accepted · **Date:** 2026-07

## Context

The intent classifier was validated on [BANKING77](https://huggingface.co/datasets/PolyAI/banking77) —
13,083 real customer service queries across 77 fine-grained banking intents, with an
official train/test split.

The expectation, stated in advance, was that a TF-IDF linear model would do poorly: 77
classes, many of them near neighbours, real customers with real typos.

It scored **macro-F1 0.9164**.

A number that beats its own prediction that badly has two explanations, and only one of them
is good. The other is leakage.

## Decision

Test it before publishing it, and publish the test alongside the number.

Verbatim overlap between the two official files is **6 messages, 0.195% of the test set** —
far too small to explain the result, and a property of the dataset rather than of any split
made here. **That check runs on every training run, and a test fails if overlap rises above
1%.**

## Alternatives considered

**Publish 0.9164 and move on.** It is the official split; no split decision was made here,
so arguably no verification was owed. Rejected because the portfolio's whole claim is that
its numbers were checked, and a check performed only on disappointing results is not a
check — it is a filter. **Suspicion has to be symmetric or it is bias.**

**Suppress it as implausible.** Rejected in the opposite direction. The result is real, and
discarding a real result for being inconveniently high is the same error with the sign
flipped.

**Verify once, by hand, and mention it in prose.** The first instinct. Rejected because a
one-off check goes stale the moment anything changes — a preprocessing change, a different
dataset revision — and prose cannot fail. Making it a test that runs every time is the
difference between having checked and having a check.

## Consequences

- The result stands with its verification attached, which makes it stronger than the number
  alone.
- **A mechanism is in place that did not exist before**: verification is triggered by a
  result being surprising in either direction, not by it being bad. This was applied again
  later in the meeting service, where owner extraction hit 1.0000 on a held-out split and
  the investigation found the corpus was too easy rather than the extractor being perfect.
- The explanation is stated rather than left as mystery: these queries are lexically
  distinctive — a lost card and a declined transfer share almost no vocabulary — and
  character n-grams absorb the typos. That suits a sparse linear model better than expected.
- Where it fails is published too. The hardest intents
  (`balance_not_updated_after_bank_transfer` 0.7470, `top_up_failed` 0.8101) are near
  neighbours differing by circumstance rather than vocabulary, which is a limit of
  bag-of-n-grams that a sentence encoder would mostly close.
- **Scope is bounded explicitly.** BANKING77 carries no sentiment labels, so the sentiment
  classifier remains validated on synthetic data alone, and the 0.6476 and 0.9164 figures
  are *not comparable and not compared* — one is 5-way on our phrasing with an adversarial
  split, the other 77-way on real phrasing with the official one.

## Revisit when

Nothing changes this. It is a standing rule rather than a configuration: any result that
beats its stated expectation gets investigated before it gets published.
