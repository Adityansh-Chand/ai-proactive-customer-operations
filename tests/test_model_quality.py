"""Quality gate for the classifiers and the end-to-end decision path.

Metrics are recomputed from the artifacts against the held-out template split
rather than read from metrics.json.

Observed at the time of writing: intent macro-F1 0.6476, sentiment macro-F1
0.9121, end-to-end policy accuracy 0.6468.

Bars are set against the *honest* split. They look low next to the 1.0 a naive
row-level split produces, and that gap is the point -- see
`test_naive_row_split_is_recorded_as_inflated`.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.action_agent import action_for_policy  # noqa: E402
from agents.classifiers import load_artifacts  # noqa: E402
from agents.policy_agent import decide_policy  # noqa: E402
from training.train import template_split  # noqa: E402

MIN_INTENT_MACRO_F1 = 0.55
MIN_SENTIMENT_MACRO_F1 = 0.80
MIN_POLICY_ACCURACY = 0.55
# Five intent classes, so chance is ~0.20. Anything near that is not learning.
CHANCE_INTENT_F1 = 0.25


@pytest.fixture(scope="module")
def evaluated():
    intent_model, sentiment_model, metadata = load_artifacts()
    frame = pd.read_csv(ROOT / "datasets" / "messages.csv")
    _, test_idx, held_out = template_split(frame)
    test = frame.iloc[test_idx]
    messages = test["message"].to_numpy()

    predicted_intent = intent_model.predict(messages)
    predicted_sentiment = sentiment_model.predict(messages)
    predicted_policy = [
        decide_policy(intent=i, sentiment=s, priority=p)
        for i, s, p in zip(predicted_intent, predicted_sentiment, test["priority"])
    ]
    return metadata, test, predicted_intent, predicted_sentiment, predicted_policy, held_out


def test_intent_classifier_meets_bar(evaluated):
    _, test, predicted, _, _, _ = evaluated
    assert f1_score(test["intent"], predicted, average="macro") >= MIN_INTENT_MACRO_F1


def test_intent_classifier_beats_chance_decisively(evaluated):
    _, test, predicted, _, _, _ = evaluated
    assert f1_score(test["intent"], predicted, average="macro") > CHANCE_INTENT_F1 * 2


def test_sentiment_classifier_meets_bar(evaluated):
    _, test, _, predicted, _, _ = evaluated
    assert f1_score(test["sentiment"], predicted, average="macro") >= MIN_SENTIMENT_MACRO_F1


def test_end_to_end_policy_accuracy_meets_bar(evaluated):
    _, test, _, _, predicted_policy, _ = evaluated
    assert accuracy_score(test["expected_policy"], predicted_policy) >= MIN_POLICY_ACCURACY


def test_end_to_end_action_accuracy_tracks_policy(evaluated):
    """Action is a pure function of policy, so the two must agree exactly."""
    _, test, _, _, predicted_policy, _ = evaluated
    predicted_action = [action_for_policy(p) for p in predicted_policy]
    assert accuracy_score(test["expected_action"], predicted_action) == pytest.approx(
        accuracy_score(test["expected_policy"], predicted_policy), abs=1e-9
    )


def test_split_holds_out_whole_templates(evaluated):
    """The central correctness property of this evaluation."""
    _, test, _, _, _, held_out = evaluated
    frame = pd.read_csv(ROOT / "datasets" / "messages.csv")
    train_idx, _, _ = template_split(frame)
    train_templates = set(frame.iloc[train_idx]["template_id"])
    test_templates = set(test["template_id"])
    assert train_templates.isdisjoint(test_templates)
    assert len(held_out) >= 5


def test_naive_row_split_is_recorded_as_inflated(evaluated):
    """The repo must keep the misleading number visible, and labelled.

    A row-level split scores ~1.0 because train and test share phrasings. Keeping
    it recorded next to the honest figure is what stops it being quoted.
    """
    metadata, _, _, _, _, _ = evaluated
    contrast = metadata["naive_row_split_contrast"]
    assert contrast["intent_macro_f1"] > metadata["components"]["intent"]["test_macro_f1"]
    assert "inflated" in contrast["note"]


def test_data_provenance_is_declared_as_synthetic(evaluated):
    metadata, _, _, _, _, _ = evaluated
    assert "synthetic" in metadata["data_source"].lower()


def test_architecture_declares_the_deterministic_layer(evaluated):
    metadata, _, _, _, _, _ = evaluated
    assert "deterministic" in metadata["architecture"]
