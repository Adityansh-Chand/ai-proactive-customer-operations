# ADR-001 — Learn the perception, keep the policy deterministic

**Status:** Accepted · **Date:** 2026-04

## Context

This service turns a customer message into a decision: which policy applies and what action
to take. There are two ways to draw the line between learned and written.

Everything learned: one classifier from message text straight to action. Or split it —
learn the things that are genuinely about language (what is this person asking, how upset
are they), and write the things that are genuinely about business rules (an enterprise
account inside its SLA with an open incident on the service it is complaining about gets an
incident response, not an individual refund).

The original code did neither honestly. Intent routing was a keyword grep, sentiment was a
twelve-word wordlist, and the sample messages contained the literal keywords the router
searched for — an evaluation that could not fail.

## Decision

**Learn perception, write policy.** Two TF-IDF (word + character n-gram) → logistic
regression classifiers for intent and sentiment. `agents/policy_agent.py` stays deterministic
business rules over their outputs.

Label it accurately: a *deterministic policy layer over learned intent and sentiment*. Not
"multi-agent reasoning".

## Alternatives considered

**End-to-end classification, message → action.** Fewer moving parts, and it would optimise
the thing actually cared about rather than a proxy. Rejected on three grounds. It needs
labelled *actions*, which are far more expensive to produce than labelled intents and change
whenever the business rules change. It cannot explain itself — "why was this escalated" gets
answered by a coefficient rather than a rule. And a policy change would require retraining
rather than an edit, which is exactly backwards: business rules change more often than
language does.

**Learn the policy too, as a second model over predicted labels.** Rejected because the
policy is not uncertain. "Enterprise tier, SLA breach imminent, known incident on the named
service" has a correct answer that someone in the business decided; approximating a known
function with a fitted model adds error and removes auditability for nothing.

**Keep the rules and skip the classifiers.** The status quo being replaced. Rejected because
the rules need intent and sentiment as *inputs*, and a keyword grep supplying them is where
the whole thing was dishonest.

**An LLM for the whole path.** Available through the provider-agnostic seam and deliberately
not the default. Same objections as end-to-end, plus non-reproducibility.

## Consequences

- The deterministic layer is auditable: any decision can be traced to a specific rule and
  the labels that triggered it, which is what a customer operations team would actually be
  asked for.
- **Errors compound across the boundary, and this is reported rather than hidden.** Intent
  macro-F1 is 0.6476 and sentiment 0.9121, but end-to-end policy accuracy is **0.6468** —
  below both, because a single error in *either* classifier changes the downstream decision.
  That compounding is a real property of the architecture and both levels are published.
- The two-level evaluation topology exists in no other service in this portfolio, which is
  the point: the problem shape dictated it rather than a template.
- Policy changes are code changes requiring review — slower than retraining, and correct
  for rules someone is accountable for.
- Calling a deterministic rule engine an "agent" would have been the easy description. It is
  not used.

## Revisit when

The rules stop being enumerable — if policy starts depending on many interacting factors
where the correct answer is genuinely uncertain rather than merely unwritten. Learning a
function nobody can state beats approximating one everybody can.
