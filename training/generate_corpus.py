"""Generate the labelled customer-message corpus.

The four sample rows this replaces were written to satisfy the keyword router
that read them: row 1 contained the literal word "angry" and the router matched
on "angry". Evaluation against that data could not fail, and reported 100%.

A first version of this generator also scored 100%, for a subtler reason: train
and test drew from the same small pool of templates, so the classifier only had
to memorise phrasings it had already seen. Three changes fix that, and together
they set a realistic ceiling:

1. **Templates are held out, not rows.** `template_id` is recorded and the
   training split partitions by template, so every test message uses phrasing
   the model has never seen. This measures generalisation rather than recall.
2. **Vocabulary overlaps across intents** -- "parcel", "delivery", "returned"
   and "refund" all appear under several intents, so no single token identifies
   a class.
3. **Genuine ambiguity and label noise.** Some messages straddle two intents the
   way real ones do, and a share of sentiment labels are flipped to reflect
   annotator disagreement. Perfect scores are therefore impossible.

Each row carries gold `intent` and `sentiment` labels plus the CRM fields from
`datasets/production_schema.json`. Gold policy and action are derived by the SAME
deterministic rules the service uses at runtime, so end-to-end accuracy measures
exactly one thing: whether the classifiers recovered intent and sentiment.

Deterministic: fixed seed, fixed base date, no wall-clock reads.

    python training/generate_corpus.py           # write datasets/messages.csv
    python training/generate_corpus.py --check   # fail if output would differ
"""
import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.action_agent import action_for_policy  # noqa: E402
from agents.policy_agent import decide_policy  # noqa: E402

OUT_PATH = ROOT / "datasets" / "messages.csv"

SEED = 20260830
N_MESSAGES = 2400
BASE_DATE = datetime(2026, 4, 6, 9, 0, tzinfo=timezone.utc)

CHANNELS = ["email", "chat", "phone", "social"]
TIERS = ["free", "standard", "enterprise"]

AMBIGUOUS_RATE = 0.10
LABEL_NOISE_RATE = 0.06

TEMPLATES = {
    "refund_request": [
        "I would like my money back for {item}",
        "please reverse the charge on {item}",
        "can you send this {item} back and return the payment",
        "I no longer want {item} and expect the amount returned",
        "requesting reimbursement for {item}",
        "the {item} arrived damaged so I want compensation for it",
        "how do I get my funds returned for {item}",
        "the delivery of {item} was fine but I changed my mind and want a refund",
        "please cancel {item} and put the amount back on my card",
        "I am returning {item}, what happens with the payment",
        "the parcel with {item} came late and I would rather have the money",
        "is it possible to undo the purchase of {item}",
        "I want to send {item} back for a credit to my account",
        "raise a return for {item} and refund me please",
    ],
    "delivery_issue": [
        "my {item} still has not turned up",
        "the parcel with {item} is running behind schedule",
        "{item} was meant to arrive on Tuesday and it is still not here",
        "courier marked {item} as attempted but nobody came",
        "shipment containing {item} seems stuck at the depot",
        "it has been two weeks and {item} has not reached me",
        "the {item} order missed its promised window",
        "the driver could not find my address for {item}",
        "{item} was returned to sender without anyone contacting me",
        "delivery of {item} keeps getting rescheduled",
        "the tracking for {item} has not moved in five days",
        "nobody has been able to deliver {item} yet",
        "the estimated date for {item} has passed with no update",
        "my order containing {item} appears to be lost in transit",
    ],
    "tracking_request": [
        "where is my {item} right now",
        "can I get a status update on {item}",
        "what is the current location of the parcel with {item}",
        "any news on when {item} will get here",
        "please share the shipment progress for {item}",
        "I want to follow the journey of my {item}",
        "could you tell me which stage {item} is at",
        "is there a reference number I can use to watch {item}",
        "when is {item} scheduled to arrive",
        "what does the courier say about {item} at the moment",
        "just checking in on the progress of {item}",
        "how far along is the delivery of {item}",
        "can you confirm the expected date for {item}",
        "looking for an update on where {item} has got to",
    ],
    "complaint": [
        "this is the third time I have written about {item} and nobody replies",
        "the service around {item} has been genuinely poor",
        "I have wasted hours on {item} with no resolution",
        "nobody in your team seems to take {item} seriously",
        "the handling of {item} has been unacceptable",
        "I am extremely let down by how {item} was managed",
        "every person I speak to about {item} tells me something different",
        "I have been passed between four agents about {item}",
        "your process for {item} has cost me an entire afternoon",
        "no one has taken ownership of the problem with {item}",
        "I have received three contradictory answers about {item}",
        "the way {item} was dealt with reflects badly on your company",
        "I keep being promised callbacks about {item} that never happen",
        "I have lost confidence in how you handle things like {item}",
    ],
    "general_question": [
        "what is the warranty period on {item}",
        "does {item} come in other sizes",
        "can I change the delivery address for {item}",
        "is {item} covered by your returns window",
        "how do I add {item} to an existing order",
        "what payment methods work for {item}",
        "do you ship {item} internationally",
        "is there a student discount available on {item}",
        "how long does {item} usually take to arrive",
        "can {item} be gift wrapped",
        "what is the difference between {item} and the newer version",
        "do I need an account to buy {item}",
        "are spare parts available for {item}",
        "what happens to my subscription if I pause {item}",
    ],
}

# Genuinely ambiguous phrasings a careful human would also hesitate over. Each is
# labelled with the reading a support lead would most likely pick, but the
# competing reading is real. These exist so the ceiling is not 100%.
AMBIGUOUS = [
    ("delivery_issue", "{item} has not arrived and I am starting to want my money back"),
    ("refund_request", "if {item} cannot get here this week I would rather be refunded"),
    ("tracking_request", "no idea where {item} is, can someone actually look into this"),
    ("complaint", "still waiting on {item} and still nobody has explained why"),
    ("general_question", "what are my options if {item} never shows up"),
    ("complaint", "this is the second delivery of {item} you have got wrong"),
    ("refund_request", "I have given up waiting for {item}, just close it out for me"),
    ("delivery_issue", "the courier says delivered but {item} is not here"),
]

ITEMS = [
    "the blue headphones", "order 88214", "the standing desk", "my monthly plan",
    "the replacement charger", "the kitchen scale", "order 40917", "the running shoes",
    "the annual subscription", "the wireless keyboard",
]

SENTIMENT_WRAPPERS = {
    "negative": [
        "{body}. This has been a frustrating experience.",
        "Honestly, {body}. I expected better than this.",
        "{body}, and I am not happy about how long it has taken.",
        "This is disappointing - {body}.",
        "{body}. I have had to chase this repeatedly.",
        "{body}, which is really not good enough.",
        "{body}. At this point I have run out of patience.",
    ],
    "neutral": [
        "{body}.",
        "Hello, {body}. Thanks.",
        "Quick question - {body}.",
        "{body}. Please advise.",
        "Hi there, {body}.",
        "{body}, whenever you get a chance.",
    ],
    "positive": [
        "{body}. Your team has been great so far, thank you.",
        "Thanks for the help earlier - {body}.",
        "{body}. Really appreciate the quick responses previously.",
        "Happy with the service overall, just {body}.",
        "{body}. No rush at all, you have been helpful.",
    ],
}

FILLERS = ["", "Hi, ", "Hello, ", "Good morning, ", "Hey - "]
TRAILERS = ["", " Thanks.", " Regards.", " Please let me know.", " Appreciate it."]

TYPOS = {
    "delivery": "delivry", "please": "pls", "because": "becuase",
    "received": "recieved", "the": "teh", "order": "oder",
}


def _apply_typos(text, rng, rate=0.12):
    if rng.random() > rate:
        return text
    words = text.split()
    for index, word in enumerate(words):
        stripped = word.lower().strip(".,-")
        if stripped in TYPOS and rng.random() < 0.5:
            words[index] = word.lower().replace(stripped, TYPOS[stripped])
    return " ".join(words)


def derive_priority(account_tier, hours_to_sla):
    """Priority comes from CRM fields, never from sniffing the message text.

    Reading urgency out of the words was part of what made the old pipeline
    circular. Tier and SLA are facts the system already holds.
    """
    if account_tier == "enterprise" and hours_to_sla <= 8:
        return "high"
    if hours_to_sla <= 2:
        return "high"
    return "normal"


def generate(seed: int = SEED, n: int = N_MESSAGES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    intents = list(TEMPLATES)
    rows = []

    for index in range(n):
        if rng.random() < AMBIGUOUS_RATE:
            choice = int(rng.integers(len(AMBIGUOUS)))
            intent, raw_template = AMBIGUOUS[choice]
            template_id = f"ambiguous:{choice:02d}"
        else:
            intent = intents[int(rng.integers(len(intents)))]
            choice = int(rng.integers(len(TEMPLATES[intent])))
            raw_template = TEMPLATES[intent][choice]
            template_id = f"{intent}:{choice:02d}"

        if intent == "complaint":
            weights = [0.78, 0.20, 0.02]
        elif intent == "general_question":
            weights = [0.10, 0.78, 0.12]
        elif intent == "delivery_issue":
            weights = [0.52, 0.42, 0.06]
        else:
            weights = [0.34, 0.53, 0.13]
        sentiment = ["negative", "neutral", "positive"][
            int(rng.choice(3, p=np.array(weights) / sum(weights)))
        ]

        body = raw_template.format(item=ITEMS[int(rng.integers(len(ITEMS)))])
        wrapper = SENTIMENT_WRAPPERS[sentiment][
            int(rng.integers(len(SENTIMENT_WRAPPERS[sentiment])))
        ]
        message = wrapper.format(body=body)
        message = (
            FILLERS[int(rng.integers(len(FILLERS)))]
            + message
            + TRAILERS[int(rng.integers(len(TRAILERS)))]
        )
        message = _apply_typos(message.strip(), rng)

        # Annotator disagreement, applied after the text is built so the message
        # and its label genuinely conflict on these rows.
        if rng.random() < LABEL_NOISE_RATE:
            alternatives = [s for s in ("negative", "neutral", "positive") if s != sentiment]
            sentiment = alternatives[int(rng.integers(len(alternatives)))]

        account_tier = TIERS[int(rng.choice(3, p=[0.42, 0.40, 0.18]))]
        hours_to_sla = float(rng.integers(1, 73))
        priority = derive_priority(account_tier, hours_to_sla)

        created = BASE_DATE + timedelta(minutes=int(rng.integers(0, 60 * 24 * 30)))
        sla_due = created + timedelta(hours=hours_to_sla)

        policy = decide_policy(intent=intent, sentiment=sentiment, priority=priority)
        action = action_for_policy(policy)

        rows.append({
            "event_id": f"evt_{index:05d}",
            "template_id": template_id,
            "customer_id": f"cust_{int(rng.integers(1, 900)):04d}",
            "channel": CHANNELS[int(rng.integers(len(CHANNELS)))],
            "message": message,
            "order_id": f"ord_{int(rng.integers(10000, 99999))}",
            "account_tier": account_tier,
            "created_at": created.isoformat(),
            "sla_due_at": sla_due.isoformat(),
            "intent": intent,
            "sentiment": sentiment,
            "priority": priority,
            "expected_policy": policy,
            "expected_action": action,
            "reviewed_by": "synthetic_generator",
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    frame = generate()
    text = frame.to_csv(index=False, lineterminator="\n")

    if args.check:
        if not OUT_PATH.exists():
            print(f"FAIL: {OUT_PATH} is missing; run without --check first")
            return 1
        if OUT_PATH.read_text(encoding="utf-8") != text:
            print("FAIL: regenerated corpus differs from the committed file")
            return 1
        print(f"OK: {OUT_PATH.name} is reproducible ({len(frame)} rows)")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(f"wrote {OUT_PATH} rows={len(frame)} templates={frame.template_id.nunique()}")
    print("intent    :", frame.intent.value_counts().to_dict())
    print("sentiment :", frame.sentiment.value_counts().to_dict())
    print("policy    :", frame.expected_policy.value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
