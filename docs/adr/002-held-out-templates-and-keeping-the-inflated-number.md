# ADR-002 — Split by template, and keep the inflated number visible

**Status:** Accepted · **Date:** 2026-04

## Context

The training corpus is generated from paraphrase templates: each intent has a set of
phrasings, and messages are produced from them with varied wording, typos, negation and
mixed sentiment.

Split that by **row** and the same phrasing appears in train and test. The classifier only
has to recognise wording it has already memorised, and it scores **1.00 intent macro-F1**.

Split by **template** — remove 20 phrasings from training entirely, so every test message is
worded in a way the classifier has never seen — and the same pipeline on the same data
scores **0.6476**.

Both numbers are real. Only one is evidence.

## Decision

Split by held-out template. Report **0.6476** as the intent figure.

**Keep the 1.00 in `models/artifacts/metrics.json`** under the key
`naive_row_split_contrast`, explicitly labelled as inflated — and add a test asserting it
stays labelled.

## Alternatives considered

**Row split, because it is standard.** It is standard, and here it measures memorisation.
The distinction is not subtle once stated: a row split asks "can it recognise this
phrasing", a template split asks "can it handle a phrasing it has not seen", and only the
second is the deployed question.

**Template split, and delete the 1.00 entirely.** The obvious tidy option, and the one
rejected most deliberately. Deleting it removes the evidence that the honest number *is*
honest. On its own, 0.6476 looks like a weak classifier; next to the 1.00 it is a
demonstration that the evaluation design is worth 0.35 macro-F1. **The contrast is the
finding, and the finding needs both numbers.**

Keeping it also inoculates against it: a figure sitting in the artifact labelled `inflated`
cannot later be rediscovered and quoted as a result.

**Report both without labelling which is which.** Rejected as worse than either. Two numbers
and no guidance means the reader picks, and readers pick the high one.

**Generate more templates so the split hurts less.** Rejected as treating the symptom. More
templates would raise 0.6476 by making the held-out phrasings less novel — which is
improving the score by making the test easier, and is the failure mode this repository was
rebuilt to remove.

## Consequences

- The headline intent figure is 0.6476 against a ~0.20 chance baseline on 5 classes.
  Visibly modest, and the one the deployed system would reproduce.
- A test enforces the label on the inflated figure, so the guard survives refactoring rather
  than depending on someone remembering.
- End-to-end accuracy (0.6468) is reported alongside, below both component scores, because
  errors compound (ADR-001). Three numbers where one would flatter more.
- The same discipline recurs across the portfolio: chronological splits in sales, held-out
  templates here, human judgments in retrieval. Each is the hard split for its data shape,
  and the pattern is deliberate rather than incidental.

## Revisit when

Real labelled customer messages replace the generated corpus. Templates stop existing as a
concept, and the honest split becomes chronological instead — the same question about
generalisation, asked the way that data shape demands.
