"""Fit the intent and sentiment classifiers used by the agent DAG.

Two classifiers replace two keyword lookups:

- **intent** replaced `route_request`, which matched on the substrings "refund",
  "delay" and "track".
- **sentiment** replaced a 12-word set (`{"angry", "bad", "complaint", ...}`).
  It returned "negative" if any of those words appeared and "neutral" otherwise,
  so "this is the third time I have written and nobody replies" scored neutral.

Both are TF-IDF (word plus character n-grams) into logistic regression. Character
n-grams are included so the typos in the corpus do not destroy the signal.

The policy layer is deliberately NOT learned -- see `agents/policy_agent.py`.
Evaluation is therefore two-level: per-component macro-F1, and end-to-end policy
and action accuracy once the deterministic rules run on predicted labels.

    python training/train.py             # fit and write models/artifacts/
    python training/train.py --verify    # refit and fail if metrics drifted
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitoring.drift import build_categorical_reference  # noqa: E402

from agents.action_agent import action_for_policy  # noqa: E402
from agents.policy_agent import decide_policy  # noqa: E402

DATA_PATH = ROOT / "datasets" / "messages.csv"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
DRIFT_REFERENCE_PATH = ARTIFACT_DIR / "drift_reference.json"
INTENT_PATH = ARTIFACT_DIR / "intent_classifier.joblib"
SENTIMENT_PATH = ARTIFACT_DIR / "sentiment_classifier.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.md"

RANDOM_STATE = 42
TEST_SIZE = 0.25
C_GRID = [0.5, 1.0, 4.0, 10.0]
CV_FOLDS = 5
VERIFY_TOLERANCE = 0.05


def build_pipeline():
    """Word n-grams for phrasing, character n-grams for typo tolerance."""
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(
                analyzer="word", ngram_range=(1, 2), sublinear_tf=True, min_df=2
            )),
            ("char", TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2
            )),
        ])),
        ("model", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])


def fit_classifier(x_train, y_train):
    search = GridSearchCV(
        build_pipeline(),
        {"model__C": C_GRID},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_, float(search.best_params_["model__C"]), float(search.best_score_)


def template_split(frame):
    """Partition by TEMPLATE, so no phrasing appears in both train and test.

    A row-level split scores ~1.00 here and means nothing: both halves draw from
    the same template pool, so the classifier only has to recognise wording it
    has already seen. Holding out whole templates asks the real question --
    does this generalise to phrasings it has never encountered?

    Templates are held out per intent so every class stays represented on
    both sides.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    held_out = set()
    for intent, group in frame.groupby("intent", sort=True):
        templates = sorted(group["template_id"].unique())
        # Ambiguous templates are shared across intents; leave them in training
        # so the held-out set stays interpretable per class.
        templates = [t for t in templates if not t.startswith("ambiguous:")]
        n_hold = max(1, int(round(len(templates) * TEST_SIZE)))
        held_out.update(rng.choice(templates, size=n_hold, replace=False).tolist())

    mask = frame["template_id"].isin(held_out).to_numpy()
    return np.flatnonzero(~mask), np.flatnonzero(mask), sorted(held_out)


def train():
    frame = pd.read_csv(DATA_PATH)
    messages = frame["message"].to_numpy()

    train_idx, test_idx, held_out_templates = template_split(frame)
    train_rows, test_rows = frame.iloc[train_idx], frame.iloc[test_idx]

    intent_model, intent_c, intent_cv = fit_classifier(
        messages[train_idx], train_rows["intent"].to_numpy()
    )
    sentiment_model, sentiment_c, sentiment_cv = fit_classifier(
        messages[train_idx], train_rows["sentiment"].to_numpy()
    )

    # The test split is scored here, once.
    predicted_intent = intent_model.predict(messages[test_idx])

    # Drift reference over the PREDICTED INTENT MIX, not classifier confidence.
    #
    # Confidence was tried first and abandoned. On a template-generated corpus it
    # is bimodal -- near 1.0 on phrasings the model effectively memorised, much
    # lower on anything else -- so PSI swung wildly on ordinary traffic and
    # reported drift that was not there. A monitor that cries wolf gets ignored.
    #
    # The mix of predicted intents is the robust signal: if the share of refund
    # requests doubles, something changed, regardless of how confident the model
    # happens to feel about each one.
    drift_reference = build_categorical_reference(
        intent_model.predict(messages)
    )
    predicted_sentiment = sentiment_model.predict(messages[test_idx])

    # Deliberate contrast: the same pipeline under a naive ROW-level split, where
    # train and test share templates. It scores near-perfectly and the number is
    # worthless. Reported so the honest figure has something to be read against,
    # and so nobody is tempted to quote the flattering one.
    row_train_idx, row_test_idx = train_test_split(
        np.arange(len(frame)), test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=frame["intent"],
    )
    row_intent_model, _, _ = fit_classifier(
        messages[row_train_idx], frame.iloc[row_train_idx]["intent"].to_numpy()
    )
    row_intent_f1 = float(f1_score(
        frame.iloc[row_test_idx]["intent"],
        row_intent_model.predict(messages[row_test_idx]),
        average="macro",
    ))

    # End-to-end: deterministic rules applied to PREDICTED labels.
    predicted_policy = [
        decide_policy(intent=i, sentiment=s, priority=p)
        for i, s, p in zip(predicted_intent, predicted_sentiment, test_rows["priority"])
    ]
    predicted_action = [action_for_policy(p) for p in predicted_policy]

    metrics = {
        "model_type": "TF-IDF (word + char n-grams) -> LogisticRegression, x2",
        "data_source": "synthetic -- training/generate_corpus.py",
        "architecture": (
            "learned intent and sentiment, deterministic policy and action layer"
        ),
        "n_total": int(len(frame)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "random_state": RANDOM_STATE,
        "split": (
            "held-out TEMPLATES, not rows -- every test message uses phrasing "
            "absent from training"
        ),
        "n_held_out_templates": len(held_out_templates),
        "held_out_templates": held_out_templates,
        "components": {
            "intent": {
                "best_C": intent_c,
                "cv_macro_f1": round(intent_cv, 4),
                "test_macro_f1": round(
                    float(f1_score(test_rows["intent"], predicted_intent, average="macro")), 4
                ),
                "test_accuracy": round(
                    float(accuracy_score(test_rows["intent"], predicted_intent)), 4
                ),
                "labels": sorted(frame["intent"].unique().tolist()),
            },
            "sentiment": {
                "best_C": sentiment_c,
                "cv_macro_f1": round(sentiment_cv, 4),
                "test_macro_f1": round(
                    float(f1_score(test_rows["sentiment"], predicted_sentiment, average="macro")), 4
                ),
                "test_accuracy": round(
                    float(accuracy_score(test_rows["sentiment"], predicted_sentiment)), 4
                ),
                "labels": sorted(frame["sentiment"].unique().tolist()),
            },
        },
        "naive_row_split_contrast": {
            "intent_macro_f1": round(row_intent_f1, 4),
            "note": (
                "Same pipeline, same data, split by ROW instead of by template so "
                "train and test share phrasings. This number is inflated and is "
                "recorded only as a warning -- it is what the honest split replaced."
            ),
        },
        "end_to_end": {
            "policy_accuracy": round(
                float(accuracy_score(test_rows["expected_policy"], predicted_policy)), 4
            ),
            "action_accuracy": round(
                float(accuracy_score(test_rows["expected_action"], predicted_action)), 4
            ),
            "note": (
                "Deterministic rules applied to PREDICTED intent and sentiment. "
                "Errors here are classification errors; the policy layer is exact "
                "by construction."
            ),
        },
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }

    reports = {
        "intent": classification_report(test_rows["intent"], predicted_intent, zero_division=0),
        "sentiment": classification_report(
            test_rows["sentiment"], predicted_sentiment, zero_division=0
        ),
    }
    return intent_model, sentiment_model, metrics, reports, drift_reference


def render_model_card(metrics):
    intent = metrics["components"]["intent"]
    sentiment = metrics["components"]["sentiment"]
    end = metrics["end_to_end"]
    return f"""# Model Card - Customer Operations Classifiers

## What this is

Two fitted text classifiers feeding a deterministic policy layer.

| Stage | Implementation |
|---|---|
| Intent | **Learned** - TF-IDF (word 1-2, char_wb 3-5) -> LogisticRegression |
| Sentiment | **Learned** - same architecture |
| Priority | Derived from CRM fields (account tier, hours to SLA) - not from the text |
| Policy | **Deterministic rules**, `agents/policy_agent.py` |
| Action | **Deterministic mapping**, `agents/action_agent.py` |

The policy layer is rules on purpose. A system that decides refunds and
escalations should be auditable and changeable without retraining; every decision
names the rule that produced it. Learning it would trade that away for nothing.

## Training data - synthetic

Generated by `training/generate_corpus.py` (seeded, reproducible). Not real
customer messages. Messages come from paraphrase templates that avoid the keyword
vocabulary the previous implementation matched on, and include typos, filler and
polite negatives that carry no angry-word.

## Measured performance (held-out, n={metrics['n_test']})

| Component | Macro-F1 | Accuracy | CV macro-F1 |
|---|---|---|---|
| Intent ({len(intent['labels'])} classes) | **{intent['test_macro_f1']}** | {intent['test_accuracy']} | {intent['cv_macro_f1']} |
| Sentiment ({len(sentiment['labels'])} classes) | **{sentiment['test_macro_f1']}** | {sentiment['test_accuracy']} | {sentiment['cv_macro_f1']} |

**End-to-end** (rules applied to predicted labels):

| Metric | Value |
|---|---|
| Policy accuracy | **{end['policy_accuracy']}** |
| Action accuracy | **{end['action_accuracy']}** |

End-to-end accuracy sits below the component scores because a single
misclassification in either component changes the downstream decision. That
compounding is the honest cost of the architecture and is why both levels are
reported rather than only the flattering one.

## Known limitations

- Trained on templated synthetic text. Real inboxes carry multi-intent messages,
  code-switching, forwarded threads and attachments -- none of which appear here.
- Single-label intent: a message asking for both a refund and tracking gets one label.
- Sentiment is three-class; no intensity, sarcasm, or mixed sentiment.
- No confidence thresholding or human-review routing on low-confidence predictions.
- Priority uses tier and SLA only; no customer history or lifetime value.
- One seeded synthetic draw; no confidence intervals across seeds.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    intent_model, sentiment_model, metrics, reports, drift_reference = train()

    if args.verify:
        if not METRICS_PATH.exists():
            print("FAIL: metrics.json missing; run without --verify first")
            return 1
        committed = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        old = committed["end_to_end"]["policy_accuracy"]
        new = metrics["end_to_end"]["policy_accuracy"]
        if abs(old - new) > VERIFY_TOLERANCE:
            print(f"FAIL: policy accuracy drifted {old} -> {new}")
            return 1
        print(f"OK: retrained policy accuracy {new} matches committed {old}")
        return 0

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(intent_model, INTENT_PATH)
    joblib.dump(sentiment_model, SENTIMENT_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    MODEL_CARD_PATH.write_text(render_model_card(metrics), encoding="utf-8")
    # Router confidence on the training corpus, so the running service can tell
    # whether it is still reading the kind of language it was fitted on.
    DRIFT_REFERENCE_PATH.write_text(
        json.dumps(drift_reference, indent=2) + "\n", encoding="utf-8"
    )

    intent = metrics["components"]["intent"]
    sentiment = metrics["components"]["sentiment"]
    print(f"split     : {metrics['n_held_out_templates']} templates held out "
          f"({metrics['n_train']} train / {metrics['n_test']} test rows)")
    print(f"intent    macro-F1 {intent['test_macro_f1']}  (C={intent['best_C']})")
    print(f"sentiment macro-F1 {sentiment['test_macro_f1']}  (C={sentiment['best_C']})")
    print(f"end-to-end policy accuracy {metrics['end_to_end']['policy_accuracy']}")
    print(f"end-to-end action accuracy {metrics['end_to_end']['action_accuracy']}")
    print()
    print("[contrast] naive row-level split intent macro-F1: "
          f"{metrics['naive_row_split_contrast']['intent_macro_f1']}"
          " -- inflated, shares templates across the split")
    print()
    for name, report in reports.items():
        print(f"-- {name} --")
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
